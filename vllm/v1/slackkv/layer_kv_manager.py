# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed all-resident LayerPage allocation at the scheduler boundary."""

from collections.abc import Callable
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import RequestStatus
from vllm.v1.slackkv.kv_arena import LayerPageArena, PageOwner
from vllm.v1.slackkv.protocol import (
    CleanupKey, LayerMappingUpdate, PageHandle, RequestRef, StepLease,
)


@dataclass(frozen=True)
class LayerAllocationResult:
    ref: RequestRef
    allocation_version: int
    target_blocks: int
    new_pages: tuple[PageHandle, ...] = ()


@dataclass
class LayerAllocation:
    ref: RequestRef
    version: int
    target_blocks: int
    layer_pages: list[list[PageHandle]]
    update: LayerMappingUpdate | None = None
    zero_pages: tuple[PageHandle, ...] = ()
    lease: StepLease | None = None
    retired: bool = False
    retired_pages: tuple[PageHandle, ...] = ()


class LayerKVManager:
    """The feature-on manager; never exposes a native block pool/coordinator."""

    def __init__(
        self, kv_cache_config, max_model_len, scheduler_block_size,
        hash_block_size, max_in_flight_tokens=None, enable_caching=False,
        use_eagle=False, num_prefill_lookahead=0, log_stats=False,
        enable_kv_cache_events=False, dcp_world_size=1, pcp_world_size=1,
        metrics_collector=None, watermark=0.0,
    ):
        if (
            enable_caching or use_eagle or num_prefill_lookahead or
            dcp_world_size != 1 or pcp_world_size != 1 or
            len(kv_cache_config.kv_cache_groups) != 1
        ):
            raise ValueError(
                "LayerPage manager requires one plain full-attention group"
            )
        group = kv_cache_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec or spec.sliding_window is not None
            or spec.attention_chunk_size is not None or spec.non_causal
            or spec.block_size != scheduler_block_size
        ):
            raise ValueError("LayerPage manager requires homogeneous causal attention")
        if watermark < 0:
            raise ValueError("watermark must be nonnegative")
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.block_size = spec.block_size
        self.layer_names = tuple(group.layer_names)
        self.num_layers = len(self.layer_names)
        self.num_blocks = kv_cache_config.num_blocks
        self.arena = LayerPageArena(
            self.num_layers * self.num_blocks, spec.page_size_bytes,
        )
        self.watermark_pages = int(watermark * self.num_blocks) * self.num_layers
        self.log_stats = log_stats
        self.enable_caching = False
        self.empty_kv_cache_blocks = KVCacheBlocks(((),))
        self.allocations: dict[RequestRef, LayerAllocation] = {}
        self.retired: dict[tuple[RequestRef, int], LayerAllocation] = {}
        self.leases: dict[StepLease, LayerAllocation] = {}
        self.worker_versions: dict[RequestRef, int] = {}
        self.request_versions: dict[RequestRef, int] = {}
        self.revision = 0
        self.ref_resolver: Callable[[str], RequestRef] | None = None
        self.last_failure: dict | None = None
        self.mapping_observers: list[Callable] = []

    def add_mapping_observer(self, observer):
        self.mapping_observers.append(observer)

    def _notify_mapping(self, action, allocation, added=()):
        for observer in self.mapping_observers:
            observer(action, allocation, added)

    def set_ref_resolver(self, resolver):
        self.ref_resolver = resolver

    def _ref(self, request):
        if self.ref_resolver is None:
            raise RuntimeError("LayerPage request identity resolver is not installed")
        return self.ref_resolver(request.request_id)

    def allocation(self, ref: RequestRef):
        return self.allocations.get(ref)

    def latest_version(self, ref: RequestRef):
        return self.request_versions.get(ref, 0)

    def allocation_snapshot(self, ref: RequestRef):
        allocation = self.allocations.get(ref)
        if allocation is None:
            return None
        return {
            "ref": ref, "version": allocation.version,
            "target_blocks": allocation.target_blocks,
            "layer_pages": tuple(tuple(row) for row in allocation.layer_pages),
            "lease": allocation.lease,
        }

    def capacity_snapshot(self):
        return self.arena.snapshot()

    @property
    def usage(self):
        return 1 - self.arena.counts["FREE"] / (self.arena.capacity - 1)

    def required_pages(self, ref: RequestRef, target_tokens: int):
        allocation = self.allocations.get(ref)
        current = allocation.target_blocks if allocation is not None else 0
        target = cdiv(min(target_tokens, self.max_model_len), self.block_size)
        return max(0, target - current) * self.num_layers

    def _check_capacity(self, request, needed, watermark, phase):
        available = self.arena.counts["FREE"]
        profile = getattr(self, "_slackkv_profile", None)
        if profile is not None:
            profile.allocation(request, needed, needed + watermark, available, phase)
        if needed + watermark > available:
            self.last_failure = {
                "unit": "LayerPage", "phase": phase,
                "cause": "physical_pages" if needed > available else "watermark",
                "needed_pages": needed, "watermark_pages": watermark,
                "free_pages": available,
            }
            return False
        return True

    def allocate_slots(
        self, request, num_new_tokens, num_new_computed_tokens=0,
        new_computed_blocks=None, num_lookahead_tokens=0,
        num_external_computed_tokens=0, delay_cache_blocks=False,
        num_encoder_tokens=0, full_sequence_must_fit=False,
        reserved_blocks=0, has_scheduled_reqs=True,
    ):
        if (
            num_new_computed_tokens or num_lookahead_tokens
            or num_external_computed_tokens or delay_cache_blocks
            or num_encoder_tokens or reserved_blocks
            or (new_computed_blocks is not None
                and any(new_computed_blocks.blocks))
        ):
            raise ValueError(
                "unsupported nonresident/prefix/speculative LayerPage work"
            )
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")
        ref = self._ref(request)
        current = self.allocations.get(ref)
        if current is not None and current.lease is not None:
            raise RuntimeError("cannot change a leased LayerPage allocation")
        self.last_failure = None
        watermark = self.watermark_pages if (
            has_scheduled_reqs and request.status in (
                RequestStatus.WAITING, RequestStatus.PREEMPTED,
            )
        ) else 0
        if full_sequence_must_fit and not self._check_capacity(
            request, self.required_pages(ref, request.num_tokens),
            watermark, "admission",
        ):
            return None
        target_tokens = min(
            request.num_computed_tokens + num_new_tokens, self.max_model_len,
        )
        old_blocks = current.target_blocks if current is not None else 0
        target_blocks = max(old_blocks, cdiv(target_tokens, self.block_size))
        delta_blocks = target_blocks - old_blocks
        needed = delta_blocks * self.num_layers
        if not self._check_capacity(request, needed, watermark, "slots"):
            return None
        if not needed:
            if current is None:
                raise RuntimeError("scheduled tokens require a LayerPage allocation")
            return LayerAllocationResult(ref, current.version, old_blocks)

        owners = tuple(
            PageOwner(ref, layer, logical_block)
            for layer in range(self.num_layers)
            for logical_block in range(old_blocks, target_blocks)
        )
        pages = self.arena.reserve(owners)
        if pages is None:
            raise RuntimeError("LayerPage free count changed during CPU reservation")
        rows = None
        new_version = self.revision + 1
        try:
            rows = current.layer_pages if current is not None else [
                [] for _ in range(self.num_layers)
            ]
            added = tuple(
                pages[layer * delta_blocks:(layer + 1) * delta_blocks]
                for layer in range(self.num_layers)
            )
            for row, extension in zip(rows, added):
                row.extend(extension)
            prior = current.update if current is not None else None
            if current is None or (prior is not None and prior.kind == "RESET"):
                update = LayerMappingUpdate(
                    ref, "RESET", 0, new_version, 0, target_blocks,
                    tuple(tuple(row) for row in rows),
                )
            elif prior is not None:
                update = LayerMappingUpdate(
                    ref, "APPEND", prior.base_version, new_version,
                    prior.append_start, target_blocks,
                    tuple(a + b for a, b in zip(prior.layer_pages, added)),
                )
            else:
                update = LayerMappingUpdate(
                    ref, "APPEND", current.version, new_version, old_blocks,
                    target_blocks, added,
                )
            zero_pages = (current.zero_pages if current is not None else ()) + pages
            if current is None:
                current = LayerAllocation(ref, new_version, target_blocks, rows)
        except BaseException:
            if rows is not None:
                for row in rows:
                    del row[old_blocks:]
            self.arena.rollback(pages)
            raise

        # From commit onward an unexpected failure is fail-stop, not rollback.
        self.arena.commit(pages)
        current.version = new_version
        current.target_blocks = target_blocks
        current.update = update
        current.zero_pages = zero_pages
        self.allocations[ref] = current
        self.request_versions[ref] = new_version
        self.revision = new_version
        self._notify_mapping("commit", current, added)
        return LayerAllocationResult(ref, new_version, target_blocks, pages)

    def acquire_step(self, ref: RequestRef, lane: str, step_id: int):
        allocation = self.allocations[ref]
        if allocation.retired or allocation.lease is not None:
            raise RuntimeError("LayerPage allocation is retired or already leased")
        lease = StepLease(ref, allocation.version, lane, step_id)
        self.leases[lease] = allocation
        allocation.lease = lease
        return lease

    def mapping_update(self, ref: RequestRef):
        return self.allocations[ref].update

    def zero_pages_for(self, lease: StepLease):
        allocation = self.leases.get(lease)
        if allocation is None or allocation.lease != lease:
            raise RuntimeError("unknown LayerPage step lease")
        # Only pending growth pages are visited, never the historical mapping.
        self.arena.validate_owners(allocation.zero_pages, lease.ref)
        return allocation.zero_pages

    def ack_mapping(self, lease: StepLease):
        allocation = self.leases.get(lease)
        if allocation is None or allocation.version != lease.allocation_version:
            raise RuntimeError("LayerPage mapping ACK identity mismatch")
        self.worker_versions[lease.ref] = lease.allocation_version
        allocation.update = None
        allocation.zero_pages = ()

    def detach_request(self, ref: RequestRef, expected_version=None):
        allocation = self.allocations.get(ref)
        if allocation is None:
            return 0
        if expected_version is not None and allocation.version != expected_version:
            raise RuntimeError("LayerPage detach version mismatch")
        pages = tuple(page for row in allocation.layer_pages for page in row)
        del self.allocations[ref]
        allocation.retired = True
        allocation.retired_pages = pages
        if allocation.lease is None:
            self.arena.reclaim(pages)
        else:
            self.retired[ref, allocation.version] = allocation
            self.arena.retire(pages)
        self._notify_mapping("detach", allocation)
        return allocation.version

    def release_step(self, lease: StepLease):
        allocation = self.leases.get(lease)
        if (
            allocation is None or allocation.lease != lease
            or allocation.ref != lease.ref
            or allocation.version != lease.allocation_version
        ):
            raise RuntimeError("LayerPage completion lease identity mismatch")
        del self.leases[lease]
        allocation.lease = None
        if allocation.retired:
            self.arena.reclaim(allocation.retired_pages)
            del self.retired[lease.ref, lease.allocation_version]

    def cleanup_ready(self, key: CleanupKey):
        allocation = self.retired.get((key.ref, key.retired_allocation_version))
        return allocation is None or allocation.lease is None

    def cleanup_ack(self, key: CleanupKey):
        version = self.worker_versions.get(key.ref)
        if version is not None and version <= key.retired_allocation_version:
            del self.worker_versions[key.ref]
        if key.scope == "request":
            version = self.request_versions.get(key.ref)
            if version is not None and version <= key.retired_allocation_version:
                del self.request_versions[key.ref]

    def free(self, request):
        return self.detach_request(self._ref(request))

    def get_computed_blocks(self, request):
        return self.empty_kv_cache_blocks, 0, 0

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def estimate_cached_tokens(self, request):
        return 0

    def record_prefix_cache_stats(self, request, num_hits):
        if num_hits:
            raise RuntimeError("prefix reuse is disabled for LayerPage allocations")

    def make_prefix_cache_stats(self):
        return PrefixCacheStats() if self.log_stats else None

    def new_step_starts(self):
        pass

    def take_events(self):
        # Native prefix events are absent; arena events have a separate schema.
        return []

    def take_kv_cache_block_copies(self):
        return [], []

    def reset_prefix_cache(self):
        return True

    def evict_blocks(self, block_hashes):
        raise NotImplementedError("external KV eviction is unsupported for LayerPages")
