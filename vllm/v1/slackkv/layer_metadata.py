# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental per-layer tables and two batched integer preparation kernels."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.slackkv.protocol import AppliedVersion, RequestRef


@triton.jit
def apply_layer_page_deltas(
    table, packed_deltas, count: tl.constexpr, ratio: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    record = offsets // ratio
    subpage = offsets % ratio
    valid = record < count
    column = tl.load(packed_deltas + 2 * record, mask=valid, other=0)
    page = tl.load(packed_deltas + 2 * record + 1, mask=valid, other=0)
    tl.store(table + column + subpage, page * ratio + subpage, mask=valid)


@triton.jit
def compute_layer_slots(
    tables, positions, starts, slots, num_tokens: tl.constexpr,
    table_rows: tl.constexpr, width: tl.constexpr,
    slot_stride: tl.constexpr, kernel_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, layer = tl.program_id(0), tl.program_id(1)
    begin = tl.load(starts + row)
    end = tl.load(starts + row + 1)
    for chunk in range(begin, end, BLOCK):
        token = chunk + tl.arange(0, BLOCK)
        valid = (token < end) & (token < num_tokens)
        position = tl.load(positions + token, mask=valid, other=0).to(tl.int64)
        page = tl.load(
            tables + (layer * table_rows + row) * width + position // kernel_size,
            mask=valid, other=0,
        ).to(tl.int64)
        slot = page * kernel_size + position % kernel_size
        tl.store(slots + layer * slot_stride + token, slot, mask=valid)


@dataclass
class MappingMirror:
    version: int
    blocks: int
    pages: list[list]


@dataclass(frozen=True)
class PreparedLayerMetadata:
    tables: dict[str, torch.Tensor]
    slots: dict[str, torch.Tensor]
    versions: tuple[AppliedVersion, ...]


def clone_layer_metadata(base, table, slots):
    """Bind one layer to fresh FA2 metadata without the generic copy protocol."""
    # S3 requires FA2's plain dataclass: no slots or custom copy/state hooks.
    # Copy every field from this step's base, matching its native shallow copy.
    result = object.__new__(type(base))
    result.__dict__ = base.__dict__.copy()
    result.block_table = table
    result.slot_mapping = slots
    return result


def build_metadata(common, builder, prepared):
    first = next(iter(prepared.tables))
    common = common.replace(block_table_tensor=prepared.tables[first],
                            slot_mapping=prepared.slots[first])
    base = builder.build(common_prefix_len=0, common_attn_metadata=common)
    return {
        name: base if name == first else clone_layer_metadata(
            base, prepared.tables[name], prepared.slots[name],
        ) for name in prepared.tables
    }


class LayerMetadataBuffers:
    """One lane's writable tables; its prior ticket must drain before reuse."""

    def __init__(self, device, layer_names, max_rows, max_tokens, max_context,
                 block_size, kernel_size, num_pages):
        if block_size % kernel_size or min(max_rows, max_tokens, num_pages) <= 0:
            raise ValueError("Invalid LayerPage metadata geometry")
        self.layer_names = tuple(layer_names)
        self.max_rows, self.max_tokens = max_rows, max_tokens
        self.block_size, self.kernel_size = block_size, kernel_size
        self.ratio = block_size // kernel_size
        self.num_pages = num_pages
        self.max_blocks = (max_context + block_size - 1) // block_size
        self.width = self.max_blocks * self.ratio
        self.table_rows = max_rows + 1
        shape = (len(self.layer_names), self.table_rows, self.width)
        if len(self.layer_names) * self.table_rows * self.width > 2**31:
            raise ValueError("LayerPage table flat indices exceed int32")
        if num_pages * self.ratio > 2**31:
            raise ValueError("LayerPage kernel page IDs exceed int32")
        self.active = torch.zeros(shape, dtype=torch.int32, device=device)
        self.scratch = torch.zeros_like(self.active)
        self.slots = torch.full(
            (len(self.layer_names), max_tokens), -1, dtype=torch.int64, device=device,
        )
        self.max_deltas = len(self.layer_names) * max_rows * self.max_blocks
        self.header_bytes = 8 * self.table_rows
        staging_bytes = self.header_bytes + 8 * self.max_deltas
        self.host = torch.empty(staging_bytes, dtype=torch.int8, device="cpu",
                                pin_memory=True)
        self.staging = torch.empty_like(self.host, device=device)
        self.host_permutation = self.host[:self.header_bytes].view(torch.int64)
        self.permutation = self.staging[:self.header_bytes].view(torch.int64)
        self.host_deltas = self.host[self.header_bytes:].view(torch.int32).view(-1, 2)
        self.deltas = self.staging[self.header_bytes:].view(torch.int32).view(-1, 2)
        self.host_permutation_np = self.host_permutation.numpy()
        self.host_deltas_np = self.host_deltas.numpy()
        self.mirrors: dict[RequestRef, MappingMirror] = {}
        self.rows: tuple[RequestRef | None, ...] = ()
        self.last_prepared: PreparedLayerMetadata | None = None
        # Cache views only; request identities and applied versions remain per-step.
        self._table_view_key: tuple[int, int] | None = None
        self._table_views: dict[str, torch.Tensor] = {}
        self._slot_view_tokens = 0
        self._slot_views: dict[str, torch.Tensor] = {}
        self.counters = {"table_h2d_bytes": 0, "table_h2d_copies": 0,
                         "patch_launches": 0, "gathers": 0, "slot_launches": 0}

    def _apply_update(self, update):
        current = self.mirrors.get(update.ref)
        if update.kind == "RESET":
            if update.append_start != 0:
                raise ValueError("LayerPage RESET has a nonzero append boundary")
            if current is not None and update.target_version <= current.version:
                raise ValueError("LayerPage RESET does not advance allocation version")
            pages = [[] for _ in self.layer_names]
        elif update.kind == "APPEND":
            if (current is None or current.version != update.base_version
                    or current.blocks != update.append_start):
                raise ValueError("LayerPage APPEND base/version does not match worker")
            pages = current.pages
        else:
            raise ValueError(f"Unknown LayerPage update kind: {update.kind}")
        if not update.base_version < update.target_version:
            raise ValueError("LayerPage update version is not increasing")
        count = update.target_blocks - update.append_start
        if (not 0 <= update.target_blocks <= self.max_blocks
                or count <= 0 or len(update.layer_pages) != len(self.layer_names)
                or any(len(layer) != count for layer in update.layer_pages)):
            raise ValueError("LayerPage update does not cover its declared layer range")
        for layer in update.layer_pages:
            if any(not 0 < page.page_id < self.num_pages for page in layer):
                raise ValueError("LayerPage update points outside the arena/null page")
        for destination, tail in zip(pages, update.layer_pages):
            destination.extend(tail)
        self.mirrors[update.ref] = MappingMirror(
            update.target_version, update.target_blocks, pages,
        )

    def prepare(self, plan, row_refs, positions, query_start_loc, num_tokens):
        row_refs = tuple(row_refs)
        if (not 0 < num_tokens <= self.max_tokens
                or not 0 < len(row_refs) <= self.max_rows
                or len(set(row_refs)) != len(row_refs)
                or set(row_refs) != set(plan.refs)):
            raise ValueError("LayerPage metadata rows do not match the final lane plan")
        leases = {lease.ref: lease for lease in plan.leases}
        if len(leases) != len(plan.leases) or set(leases) != set(row_refs):
            raise ValueError("LayerPage metadata lacks an exact step lease per row")
        updates = {}
        for update in plan.layer_updates:
            if update.ref not in leases or update.ref in updates:
                raise ValueError("LayerPage update has duplicate or foreign owner")
            self._apply_update(update)
            updates[update.ref] = update
        for ref in row_refs:
            mirror = self.mirrors.get(ref)
            lease = leases[ref]
            if (mirror is None or mirror.version != lease.allocation_version
                    or lease.lane != plan.lane or lease.step_id != plan.step_id):
                raise ValueError("LayerPage step lease differs from installed mapping")
        inputs = {source.ref: source for source in plan.inputs}
        if (len(inputs) != len(plan.inputs) or set(inputs) != set(row_refs)
                or sum(len(source.input_ids) for source in inputs.values()) != num_tokens):
            raise ValueError("LayerPage token inputs do not match the scheduled rows")
        for ref, source in inputs.items():
            if (source.computed < 0 or not source.input_ids
                    or source.computed + len(source.input_ids)
                    > self.mirrors[ref].blocks * self.block_size):
                raise ValueError("LayerPage mapping does not cover the scheduled history")

        if row_refs != self.rows or updates:
            old_rows = {ref: row for row, ref in enumerate(self.rows)}
            reset = {ref for ref, update in updates.items() if update.kind == "RESET"}
            permutation = [self.max_rows] * self.table_rows
            for row, ref in enumerate(row_refs):
                if ref in old_rows and ref not in reset:
                    permutation[row] = old_rows[ref]
            permutation[-1] = self.max_rows
            gather = row_refs != self.rows or bool(reset)
            records = []
            for row, ref in enumerate(row_refs):
                mirror, update = self.mirrors[ref], updates.get(ref)
                install_full = ref not in old_rows or ref in reset
                if install_full:
                    start, pages = 0, mirror.pages
                elif update is not None:
                    start, pages = update.append_start, update.layer_pages
                else:
                    continue
                for layer_id, layer in enumerate(pages):
                    base = (layer_id * self.table_rows + row) * self.width
                    records.extend((base + (start + i) * self.ratio, page.page_id)
                                   for i, page in enumerate(layer))
            count = len(records)
            if count > self.max_deltas:
                raise ValueError("LayerPage packed delta capacity exceeded")
            if gather or count:
                self.host_permutation_np[:] = permutation
                if count:
                    self.host_deltas_np[:count] = records
                copied_bytes = self.header_bytes + 8 * count
                self.staging[:copied_bytes].copy_(self.host[:copied_bytes],
                                                   non_blocking=True)
                self.counters["table_h2d_bytes"] += copied_bytes
                self.counters["table_h2d_copies"] += 1
            if gather:
                torch.index_select(self.active, 1, self.permutation, out=self.scratch)
                self.active, self.scratch = self.scratch, self.active
                self.counters["gathers"] += 1
            if count:
                apply_layer_page_deltas[(triton.cdiv(count * self.ratio, 256),)](
                    self.active, self.deltas, count, self.ratio, BLOCK=256,
                )
                self.counters["patch_launches"] += 1
        compute_layer_slots[(len(row_refs), len(self.layer_names))](
            self.active, positions, query_start_loc, self.slots, num_tokens,
            self.table_rows, self.width, self.max_tokens, self.kernel_size, BLOCK=256,
        )
        self.counters["slot_launches"] += 1
        self.rows = row_refs
        table_view_key = (id(self.active), len(row_refs))
        if table_view_key != self._table_view_key:
            self._table_views = {
                name: self.active[i, :len(row_refs)]
                for i, name in enumerate(self.layer_names)
            }
            self._table_view_key = table_view_key
        if num_tokens != self._slot_view_tokens:
            self._slot_views = {
                name: self.slots[i, :num_tokens]
                for i, name in enumerate(self.layer_names)
            }
            self._slot_view_tokens = num_tokens
        result = PreparedLayerMetadata(
            self._table_views, self._slot_views,
            tuple(AppliedVersion(ref, self.mirrors[ref].version) for ref in row_refs),
        )
        self.last_prepared = result
        return result

    def cleanup(self, keys):
        for key in keys:
            current = self.mirrors.get(key.ref)
            if (current is not None
                    and current.version <= key.retired_allocation_version):
                del self.mirrors[key.ref]
                # Preserve old row indices until the next non-inplace gather.
                self.rows = tuple(None if ref == key.ref else ref for ref in self.rows)
        return tuple(keys)

    def clear(self):
        self.mirrors.clear()
        self.rows = ()
        self.last_prepared = None
        self._table_view_key = None
        self._table_views = {}
        self._slot_view_tokens = 0
        self._slot_views = {}

    def manifest(self):
        return {
            "layers": len(self.layer_names), "capacity_rows": self.max_rows,
            "table_rows_with_null": self.table_rows, "kernel_table_width": self.width,
            "table_active_bytes": self.active.numel() * self.active.element_size(),
            "table_scratch_bytes": self.scratch.numel() * self.scratch.element_size(),
            "slots_bytes": self.slots.numel() * self.slots.element_size(),
            "gpu_staging_bytes": self.staging.numel(),
            "pinned_staging_bytes": self.host.numel(),
            "installed_requests": len(self.mirrors),
            "cpu_mirror_page_references": sum(
                mirror.blocks * len(self.layer_names)
                for mirror in self.mirrors.values()
            ),
            "counters": dict(self.counters),
        }
