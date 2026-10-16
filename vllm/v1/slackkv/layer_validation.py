# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in real-GPU S3 diagnostics; never installed in normal serving runs."""

from types import SimpleNamespace
import time

import torch

from vllm.v1.slackkv.metrics import SlackKVAuditWorker


class S3ValidationWorker(SlackKVAuditWorker):
    def slackkv_start_timing(self):
        """Opt-in identical instrumentation for native and A P-only workers."""
        runner = self.model_runner
        assert getattr(runner, "slackkv_audit", None) is None
        assert not getattr(self, "slackkv_validation_timing", None)
        assert self.slackkv_runtime is None or not self.slackkv_runtime.tickets
        state = {"prepare": runner._prepare_inputs, "forward": runner._model_forward,
                 "records": [], "previous_rows": (), "current_shape": None}
        block_size = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

        def prepare(output, *args, **kwargs):
            rows = tuple(runner.input_batch.req_ids)
            computed = [int(value) for value in
                        runner.input_batch.num_computed_tokens_cpu[:len(rows)]]
            queries = [output.num_scheduled_tokens[rid] for rid in rows]
            prompts = [int(value) for value in
                       runner.input_batch.num_prompt_tokens[:len(rows)]]
            shape = {
                "rows": list(rows), "computed": computed, "query_tokens": queries,
                "phase": "prefill-or-mixed" if any(c < p for c, p in zip(computed, prompts))
                         else "decode",
                "growth": any((c + q + block_size - 1) // block_size >
                              (c + block_size - 1) // block_size
                              for c, q in zip(computed, queries)),
                "row_order_changed": bool(state["previous_rows"] and
                                          rows != state["previous_rows"]),
            }
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream()
            begin.record(stream)
            started = time.perf_counter_ns()
            result = state["prepare"](output, *args, **kwargs)
            host_ns = time.perf_counter_ns() - started
            end.record(stream)
            state["records"].append(("prepare", shape, host_ns, begin, end))
            state["previous_rows"] = rows
            state["current_shape"] = shape
            return result

        def forward(*args, **kwargs):
            shape = state["current_shape"]
            assert shape is not None
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream()
            begin.record(stream)
            started = time.perf_counter_ns()
            result = state["forward"](*args, **kwargs)
            host_ns = time.perf_counter_ns() - started
            end.record(stream)
            state["records"].append(("forward", shape, host_ns, begin, end))
            return result

        runner._prepare_inputs, runner._model_forward = prepare, forward
        self.slackkv_validation_timing = state
        return {"status": "installed", "audit_enabled": False}

    def slackkv_stop_timing(self, collect=True):
        state = self.slackkv_validation_timing
        assert state is not None
        runner = self.model_runner
        runner._prepare_inputs, runner._model_forward = state["prepare"], state["forward"]
        self.slackkv_validation_timing = None
        result = []
        if collect:
            for kind, shape, host_ns, begin, end in state["records"]:
                assert end.query(), "timing collection requires real drained engine"
                result.append(shape | {"kind": kind, "cpu_ns": host_ns,
                                       "stream_elapsed_ms": begin.elapsed_time(end)})
        return {"status": "collected" if collect else "removed", "records": result,
                "prepare_stream_time_includes_host_submission_gaps": True,
                "forward_scope": "model forward; excludes logits/sampling/cleanup"}

    @torch.inference_mode()
    def slackkv_validate_layer_layout(self):
        """Exercise production layout/metadata helpers and native FA2 read/write.

        R=2 is deliberately a standalone kernel diagnostic. It does not change
        the production backend's choice of kernel block size.
        """
        from dataclasses import is_dataclass

        import msgspec

        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func, reshape_and_cache_flash,
        )
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        from vllm.v1.slackkv.kv_arena import PageOwner
        from vllm.v1.slackkv.kv_layout import LayerPageLayout
        from vllm.v1.slackkv.layer_metadata import LayerMetadataBuffers
        from vllm.v1.slackkv.protocol import (
            LaneStepPlan, LayerMappingUpdate, PageHandle, ProtectedStepInput,
            RequestRef, StepLease,
        )
        from vllm.v1.core.sched.output import SchedulerOutput
        from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

        runner = self.model_runner
        runtime = self.slackkv_runtime
        assert runtime is not None and not runtime.tickets
        assert runner.input_batch.num_reqs == 0
        group = runner.kv_cache_config.kv_cache_groups[0]
        names = tuple(group.layer_names)
        assert len(names) == 36
        results = []
        page_record_contract = None
        torch.cuda.synchronize()
        production = runtime.caches()[names[0]]
        production_layout = runner.slackkv_kv_layout
        ratio = production_layout.ratio
        target, neighbor = 1, 2
        saved = production[target * ratio:(neighbor + 1) * ratio].clone()
        production[target * ratio:(target + 1) * ratio].fill_(3)
        production[neighbor * ratio:(neighbor + 1) * ratio].fill_(5)
        runner._zero_block_ids([target])
        assert torch.count_nonzero(production[target * ratio:
                                              (target + 1) * ratio]).item() == 0
        assert torch.all(production[neighbor * ratio:
                                    (neighbor + 1) * ratio] == 5).item()
        production[target * ratio:(neighbor + 1) * ratio].copy_(saved)
        torch.cuda.synchronize()
        production_zeroer = {"status": "passed", "target_page": target,
                             "neighbor_page": neighbor,
                             "actual_runner_zero_path": True,
                             "manager_free_or_lease_state_changed": False}
        for block_size, kernel_size in ((16, 16), (32, 32), (32, 16)):
            spec = group.kv_cache_spec.copy_with_new_block_size(block_size)
            layout = LayerPageLayout(names, spec, FlashAttentionBackend, 3,
                                     kernel_size, "auto")
            backing, caches = layout.allocate(self.device)
            cache = caches[names[0]]
            assert all(t.data_ptr() == cache.data_ptr() and
                       t.storage_offset() == 0 and t.stride() == cache.stride()
                       for t in caches.values())
            assert backing.numel() == layout.arena_bytes
            query_length = block_size + 1
            buffers = LayerMetadataBuffers(self.device, names, 1, query_length,
                                           2 * block_size, block_size,
                                           kernel_size, layout.num_pages)
            pages = tuple(tuple(PageHandle(1 + 2 * i + j, 1) for j in range(2))
                          for i in range(len(names)))
            ref = RequestRef("layout-diagnostic", 1)
            if not results:
                # Check the actual handle used by this GPU mapping below.
                handle = pages[0][0]
                same_handle = PageHandle(handle.page_id, handle.generation)
                owner = PageOwner(ref, 0, 0)
                same_owner = PageOwner(RequestRef(ref.request_id, ref.epoch), 0, 0)
                for record, equal, values in (
                    (handle, same_handle, (handle.page_id, handle.generation)),
                    (owner, same_owner, (ref, 0, 0)),
                ):
                    assert isinstance(record, msgspec.Struct)
                    assert type(record) is type(equal) and record == equal
                    assert hash(record) == hash(equal)
                    assert {record} & {equal} == {record}
                    assert record != values
                assert handle != PageHandle(handle.page_id, handle.generation + 1)
                assert owner != PageOwner(RequestRef(ref.request_id, ref.epoch + 1),
                                          0, 0)
                for record, fields in (
                    (handle, ("page_id", "generation")),
                    (owner, ("ref", "layer", "logical_block")),
                ):
                    for field in fields:
                        try:
                            setattr(record, field, getattr(record, field))
                        except AttributeError:
                            pass
                        else:
                            raise AssertionError("LayerPage record is mutable")
                page_record_contract = {
                    "status": "passed", "representation": "msgspec.Struct",
                    "same_type_equality_hash_and_set_intersection": True,
                    "ordinary_tuple_unequal": True,
                    "generation_and_epoch_distinct": True,
                    "frozen_attribute_error": True,
                    "dataclass_metadata": {
                        "PageHandle": is_dataclass(handle),
                        "PageOwner": is_dataclass(owner),
                    },
                    "gpu_mapping_page_id": handle.page_id,
                    "allocation_block_size": block_size,
                    "kernel_block_size": kernel_size,
                }
            update = LayerMappingUpdate(ref, "RESET", 0, 1, 0, 2, pages)
            plan = LaneStepPlan("E", 1, (ref,), SchedulerOutput.make_empty(),
                                inputs=(ProtectedStepInput(
                                    ref, tuple(range(query_length)), 0,
                                    query_length, query_length),),
                                leases=(StepLease(ref, 1, "E", 1),),
                                layer_updates=(update,))
            positions = torch.arange(query_length, dtype=torch.int64,
                                     device=self.device)
            starts = torch.tensor([0, query_length], dtype=torch.int32,
                                  device=self.device)
            prepared = buffers.prepare(plan, (ref,), positions, starts,
                                       query_length)
            lengths = torch.tensor([query_length], dtype=torch.int32,
                                   device=self.device)
            scale = torch.ones((), dtype=torch.float32, device=self.device)
            heads, dimension = spec.num_kv_heads, spec.head_size
            element = torch.arange(query_length * heads * dimension,
                                   device=self.device, dtype=torch.float32)
            element = element.reshape(query_length, heads, dimension)
            query = ((element.remainder(23) - 11) / 32).to(torch.bfloat16)
            expected = []
            comparisons = 0
            for i, name in enumerate(names):
                key = ((element.remainder(17) + i) / 64).to(torch.bfloat16)
                value = ((element.remainder(31) - i) / 64).to(torch.bfloat16)
                expected.append(torch.cat((key, value), -1))
                key_cache, value_cache = cache.transpose(1, 2).split(dimension, -1)
                reshape_and_cache_flash(key, value, key_cache, value_cache,
                                        prepared.slots[name], "auto", scale, scale)

            # Verify after all layers wrote, so a common/overlapping mapping
            # cannot pass simply because each layer is checked immediately.
            for i, name in enumerate(names):
                expected_kernel = [p.page_id * layout.ratio + sub
                                   for p in pages[i] for sub in range(layout.ratio)]
                assert prepared.tables[name][0].tolist() == expected_kernel
                expected_slots = [pages[i][p // block_size].page_id * block_size
                                  + p % block_size for p in range(query_length)]
                assert prepared.slots[name].tolist() == expected_slots
                indices = torch.tensor(expected_kernel, dtype=torch.int64,
                                       device=self.device)
                actual = cache.index_select(0, indices).permute(0, 2, 1, 3)
                actual = actual.reshape(-1, heads, 2 * dimension)[:query_length]
                assert torch.equal(actual, expected[i])
                comparisons += 1
                # Independent native packed cache, native write kernel and
                # native FA2 read: equal content, different physical page IDs.
                reference = torch.zeros((2 * layout.ratio, kernel_size, heads,
                                         2 * dimension), dtype=torch.bfloat16,
                                        device=self.device).permute(0, 2, 1, 3)
                rk, rv = reference.transpose(1, 2).split(dimension, -1)
                key, value = expected[i].split(dimension, -1)
                reshape_and_cache_flash(key.contiguous(), value.contiguous(), rk,
                                        rv, positions, "auto", scale, scale)
                native_table = torch.arange(2 * layout.ratio, dtype=torch.int32,
                                            device=self.device).unsqueeze(0)
                outputs = []
                for kv, table in ((cache, prepared.tables[name]),
                                  (reference, native_table)):
                    k_cache, v_cache = kv.transpose(1, 2).split(dimension, -1)
                    out = torch.empty_like(query)
                    flash_attn_varlen_func(
                        q=query, k=k_cache, v=v_cache, out=out,
                        cu_seqlens_q=starts, max_seqlen_q=query_length,
                        seqused_k=lengths, max_seqlen_k=query_length,
                        softmax_scale=dimension ** -0.5, causal=True,
                        block_table=table, fa_version=2,
                    )
                    outputs.append(out)
                assert torch.equal(*outputs)
                comparisons += 1

            # Native zeroer must see one unique backing, and must zero every
            # K-sized subpage in the selected B-sized allocation only.
            zeroer = KVBlockZeroer(
                self.device,
                [AttentionGroup(FlashAttentionBackend, list(names), spec, 0)],
                [kernel_size], "auto",
                {name: SimpleNamespace(kv_cache=caches[name]) for name in names},
            )
            preserved_id = pages[1][0].page_id
            preserved = cache[preserved_id * layout.ratio:
                              (preserved_id + 1) * layout.ratio].clone()
            zero_id = pages[0][0].page_id
            zeroer.zero_block_ids([zero_id])
            assert torch.count_nonzero(cache[zero_id * layout.ratio:
                                            (zero_id + 1) * layout.ratio]).item() == 0
            assert torch.equal(preserved, cache[preserved_id * layout.ratio:
                                                (preserved_id + 1) * layout.ratio])
            assert zeroer._meta[-1] == layout.ratio
            torch.cuda.synchronize()
            results.append(layout.manifest(cache) | {
                "status": "passed", "comparisons": comparisons,
                "rtol": 0, "atol": 0, "all_layers_distinct": True,
                "native_zeroer_segments": zeroer._meta[-1],
                "metadata_counters": dict(buffers.counters),
                "scope": "standalone-layout-native-FA2-kernels",
                "production_kernel_selection_changed": False,
            })
            del backing, caches, cache, buffers, prepared, zeroer
        return {"status": "passed", "production_zeroer": production_zeroer,
                "page_record_contract": page_record_contract,
                "cases": results}

    @torch.inference_mode()
    def slackkv_delay_layer_reader(self, lane="E", cycles=300_000_000):
        """Extend an actual lease's completion fence with a finite GPU read."""
        runtime = self.slackkv_runtime
        ticket = runtime.tickets[lane]
        assert ticket.plan.refs and ticket.fence_recorded
        assert not getattr(self, "slackkv_validation_reader", None)
        buffers = runtime.layer_buffers[lane]
        ref = ticket.plan.refs[0]
        lease = next(lease for lease in ticket.plan.leases if lease.ref == ref)
        mirror = buffers.mirrors[ref]
        assert mirror.version == lease.allocation_version
        name = buffers.layer_names[0]
        page = mirror.pages[0][0]
        cache = runtime.caches()[name]
        ids = torch.tensor([page.page_id * buffers.ratio + j
                            for j in range(buffers.ratio)], dtype=torch.int64,
                           device=self.device)
        stream = runtime.streams[lane]
        # Complete preceding work before recording the expected real content;
        # the test then measures a new genuinely pending device fence.
        stream.synchronize()
        expected = cache.index_select(0, ids).cpu()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            torch.cuda._sleep(cycles)
            actual = cache.index_select(0, ids)
            event = torch.cuda.Event()
            event.record(stream)
        ticket.event = event
        if ticket.state is not None and ticket.state.completion is not None:
            ticket.state.completion = event
        ticket.fence_recorded = True
        self.slackkv_validation_reader = {
            "event": event, "actual": actual, "expected": expected,
            "ids": ids, "ref": ref, "lease": lease, "page": page,
        }
        observed_ready = event.query()
        assert not observed_ready, "finite GPU reader completed before pending observation"
        return {"lane": lane, "step_id": ticket.plan.step_id,
                "ref": {"request_id": ref.request_id, "epoch": ref.epoch},
                "allocation_version": lease.allocation_version,
                "page_id": page.page_id, "generation": page.generation,
                "cycles": cycles, "event_ready_at_observation": observed_ready}

    @torch.inference_mode()
    def slackkv_verify_layer_reader(self):
        reader = self.slackkv_validation_reader
        assert reader is not None and reader["event"].query()
        assert torch.equal(reader["actual"].cpu(), reader["expected"])
        result = {"status": "passed", "actual_reader_equal": True,
                  "page_id": reader["page"].page_id,
                  "generation": reader["page"].generation,
                  "allocation_version": reader["lease"].allocation_version}
        self.slackkv_validation_reader = None
        return result
