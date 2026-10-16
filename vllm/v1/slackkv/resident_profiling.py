# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real FA2 resource probes using the same LayerPage layout as execution."""

from dataclasses import replace

import numpy as np
import torch

from vllm.forward_context import set_forward_context
from vllm.utils.mem_utils import memory_profiling
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups, get_kv_cache_groups,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.slackkv.lane_state import LaneInputs, LaneState
from vllm.v1.slackkv.layer_metadata import build_metadata
from vllm.v1.slackkv.protocol import (
    LaneStepPlan, LayerMappingUpdate, PageHandle, ProtectedStepInput,
    RequestRef, StepLease,
)
from vllm.v1.worker.workspace import use_workspace_lane


def _shapes(runtime):
    config = runtime.config.scheduler_config
    max_reqs = config.max_num_seqs
    max_tokens = config.max_num_batched_tokens
    context = runtime.config.model_config.max_model_len
    if max_reqs < 2 or max_tokens < 2:
        raise ValueError("dual-lane profiling needs at least two requests/tokens")

    def fill_p(tokens, rows, balanced=False):
        if balanced:
            count = min(rows, tokens)
            base, remainder = divmod(tokens, count)
            return tuple(min(context, base + (i < remainder)) for i in range(count))
        result = []
        while tokens and len(result) < rows:
            query = min(tokens, context)
            result.append(query)
            tokens -= query
        return tuple(result)

    full = min(max_tokens, context)
    reqs = min(max_reqs, max_tokens)
    skewed = (min(context, max_tokens - reqs + 1),) + (1,) * (reqs - 1)
    result = [
        ("p-only-max-query", (full,), 0, False),
        ("p-only-skewed-prefill", skewed, 0, False),
        ("p-only-balanced-prefill", fill_p(max_tokens, reqs, True), 0, False),
        ("p-only-long-context-decode", (1,) * reqs, 0, False),
        ("e-only-max-query", (), full, False),
        ("e-only-long-context-decode", (), 1, True),
        ("long-context-decode", (1,) * (reqs - 1), 1, True),
    ]
    seen = set()
    for name, e_query in (
        ("p-heavy-prefill", 1),
        ("e-heavy-prefill", min(context, max_tokens - 1)),
        ("balanced-prefill", min(context, max_tokens // 2)),
    ):
        p_queries = fill_p(max_tokens - e_query, max_reqs - 1)
        if (p_queries, e_query) not in seen:
            seen.add((p_queries, e_query))
            result.append((name, p_queries, e_query, False))
    return result


def _plan(runtime, lane, queries, blocks_per_req):
    """Install real non-overlapping layer pages for the temporary envelope."""
    buffers = runtime.layer_buffers[lane]
    context = runtime.config.model_config.max_model_len
    version = max((m.version for m in buffers.mirrors.values()), default=0) + 1
    refs, updates, leases, inputs = [], [], [], []
    layers = len(buffers.layer_names)
    for row, query in enumerate(queries):
        owner = row if lane == "P" else runtime.runner.max_num_reqs - 1
        ref = RequestRef(f"__slackkv_profile_{lane}_{row}", 0)
        pages = tuple(
            tuple(PageHandle(1 + (owner * layers + layer) * blocks_per_req + i, 1)
                  for i in range(blocks_per_req))
            for layer in range(layers)
        )
        refs.append(ref)
        current = buffers.mirrors.get(ref)
        updates.append(LayerMappingUpdate(
            ref, "RESET", 0 if current is None else current.version,
            version, 0, blocks_per_req, pages,
        ))
        leases.append(StepLease(ref, version, lane, version))
        inputs.append(ProtectedStepInput(
            ref, (0,) * query, context - query, context, context,
        ))
    counts = {ref.request_id: query for ref, query in zip(refs, queries)}
    output = replace(SchedulerOutput.make_empty(), num_scheduled_tokens=counts,
                     total_num_scheduled_tokens=sum(queries),
                     num_common_prefix_blocks=[0])
    return LaneStepPlan(lane, version, tuple(refs), output, tuple(inputs),
                        tuple(updates), tuple(leases))


def _protected_inputs(runtime, queries, blocks_per_req):
    runner = runtime.runner
    context = runtime.config.model_config.max_model_len
    n, r = sum(queries), len(queries)
    runner.input_ids.np[:n] = 0
    runner.input_ids.copy_to_gpu(n)
    positions = np.concatenate([np.arange(context - q, context) for q in queries])
    runner.query_pos.np[:n] = positions
    runner.positions[:n].copy_(runner.query_pos.cpu[:n], non_blocking=True)
    runner.query_start_loc.np[:r + 1] = [0, *np.cumsum(queries)]
    runner.query_start_loc.copy_to_gpu(r + 1)
    runner.optimistic_seq_lens_cpu[:r].fill_(context)
    runner.seq_lens[:r].copy_(runner.optimistic_seq_lens_cpu[:r], non_blocking=True)
    buffers = runtime.layer_buffers["P"]
    plan = _plan(runtime, "P", queries, blocks_per_req)
    prepared = buffers.prepare(
        plan, plan.refs, runner.positions[:n], runner.query_start_loc.gpu[:r + 1], n,
    )
    first = buffers.layer_names[0]
    common = CommonAttentionMetadata(
        num_actual_tokens=n, num_reqs=r, max_query_len=max(queries),
        query_start_loc=runner.query_start_loc.gpu[:r + 1],
        query_start_loc_cpu=runner.query_start_loc.cpu[:r + 1],
        max_seq_len=context, seq_lens=runner.seq_lens[:r],
        _seq_lens_cpu=runner.optimistic_seq_lens_cpu[:r],
        block_table_tensor=prepared.tables[first], slot_mapping=prepared.slots[first],
        causal=True, positions=runner.positions[:n],
    )
    builder = runner.attn_groups[0][0].get_metadata_builder(0)
    return n, build_metadata(common, builder, prepared), prepared.slots


def _elastic_inputs(runtime, query, blocks_per_req):
    context = runtime.config.model_config.max_model_len
    views = runtime.input_slot.acquire_tokens([0] * query, context - query)
    plan = _plan(runtime, "E", (query,), blocks_per_req)
    buffers = runtime.layer_buffers["E"]
    prepared = buffers.prepare(plan, plan.refs, views["positions"], views["starts"], query)
    first = buffers.layer_names[0]
    common = CommonAttentionMetadata(
        num_actual_tokens=query, num_reqs=1, max_query_len=query,
        query_start_loc=views["starts"],
        query_start_loc_cpu=runtime.input_slot.host["starts"],
        max_seq_len=context, seq_lens=views["lengths"],
        _seq_lens_cpu=runtime.input_slot.host["lengths"],
        block_table_tensor=prepared.tables[first], slot_mapping=prepared.slots[first],
        causal=True, positions=views["positions"],
    )
    metadata = build_metadata(common, runtime.elastic_metadata_builder, prepared)
    return LaneInputs(
        (), 0, 0, views["ids"], views["positions"], views["lengths"],
        metadata, prepared.slots, runtime.greedy,
        tuple(runtime.input_slot.host.values()),
    )


def _run_case(runtime, case, blocks_per_req):
    from vllm.v1.slackkv.dual_lane_runtime import greedy_metadata

    name, p_queries, e_query, segmented = case
    runner, config = runtime.runner, runtime.config
    context = config.model_config.max_model_len
    inputs, state = None, None
    p_logits = p_hidden = p_result = p_metadata = p_host = None
    p_n = 0
    if e_query:
        with torch.cuda.stream(runtime.streams["E"]), use_workspace_lane(1):
            inputs = _elastic_inputs(runtime, e_query, blocks_per_req)
            state = LaneState(inputs)
            if segmented:
                with runtime.elastic._context(inputs):
                    runtime.adapter.begin(state)
                    runtime.adapter.run_layers(state, runtime.segment_layers)

    # P logits and the E boundary activation stay alive across both streams.
    if p_queries:
        with torch.cuda.stream(runtime.streams["P"]), use_workspace_lane(0):
            p_n, metadata, slots = _protected_inputs(runtime, p_queries, blocks_per_req)
            with set_forward_context(
                metadata, config, num_tokens=p_n, slot_mapping=slots, skip_compiled=True,
            ):
                p_hidden = runner._model_forward(
                    input_ids=runner.input_ids.gpu[:p_n], positions=runner.positions[:p_n],
                )
                indices = runner.query_start_loc.gpu[1:len(p_queries) + 1] - 1
                p_sample = p_hidden[indices]
                p_logits = runner.model.compute_logits(p_sample)
            p_metadata = greedy_metadata([], len(p_queries))
            p_result = runner.sampler(p_logits, p_metadata)
            p_host = runtime.outputs["P"][:len(p_queries)]
            p_host.copy_(p_result.sampled_token_ids, non_blocking=True)

    if e_query:
        with torch.cuda.stream(runtime.streams["E"]), use_workspace_lane(1):
            with runtime.elastic._context(inputs):
                if segmented:
                    runtime.adapter.run_layers(state, runtime.adapter.body.end_layer)
                    runtime.adapter.finish(state)
                else:
                    state.hidden = runner._model_forward(
                        input_ids=inputs.input_ids, positions=inputs.positions,
                    )
                    state.logits = runner.model.compute_logits(state.hidden[-1:])
                e_result = runtime.elastic.sampler(state.logits, inputs.sampling_metadata)
                runtime.outputs["E"].copy_(e_result.sampled_token_ids, non_blocking=True)
    for stream in runtime.streams.values():
        stream.synchronize()
    if e_query:
        runtime.input_slot.release()
    return {
        "name": name, "p_query_lengths": list(p_queries), "e_query_tokens": e_query,
        "total_query_tokens": p_n + e_query,
        "requests": len(p_queries) + bool(e_query), "max_history_tokens": context,
        "p_logits_retained": bool(p_queries), "e_segment_boundary_retained": segmented,
        "attention_executed": True, "per_layer_nonoverlapping_pages": True,
        "absent_lanes_skipped": True,
    }


@torch.inference_mode()
def profile_resident_lanes(runtime):
    runner, worker = runtime.runner, runtime.worker
    config = runtime.config
    specs = runner.get_kv_cache_spec()
    groups = get_kv_cache_groups(config, specs)
    if len(groups) != 1:
        raise ValueError("resident profiling needs one actual KV cache group")
    block_size = groups[0].kv_cache_spec.block_size
    blocks_per_req = (config.model_config.max_model_len + block_size - 1) // block_size
    blocks = 1 + config.scheduler_config.max_num_seqs * blocks_per_req
    saved_override = config.cache_config.num_gpu_blocks_override
    try:
        config.cache_config.num_gpu_blocks_override = blocks
        temporary = get_kv_cache_config_from_groups(config, groups, available_memory=0)
    finally:
        config.cache_config.num_gpu_blocks_override = saved_override
    temporary_bytes = sum(t.size for t in temporary.kv_cache_tensors)
    free = torch.cuda.mem_get_info(worker.device)[0]
    if temporary_bytes >= free:
        raise RuntimeError(
            "real dual-lane resource profiling cannot allocate the legal history "
            f"envelope: temporary KV requires {temporary_bytes} bytes, free={free}; "
            "no reduced-context or metadata=None fallback is allowed"
        )
    initialized = False
    try:
        runner.initialize_kv_cache(temporary, is_profiling=True)
        initialized = True
        config.cache_config.num_gpu_blocks = temporary.num_blocks
        runtime.configure_cache()
        runner._init_kv_zero_meta()
        caches = runtime.caches()
        unique_caches = {cache.untyped_storage().data_ptr(): cache
                         for cache in caches.values()}
        for cache in unique_caches.values():
            cache.zero_()
        del cache
        initialized_event = torch.cuda.Event()
        initialized_event.record(torch.cuda.current_stream())
        initialized_event.synchronize()
        for stream in runtime.streams.values():
            stream.wait_event(initialized_event)
        cases = _shapes(runtime)
        for case in cases:
            _run_case(runtime, case, blocks_per_req)
        actual_kv_bytes = sum(t.untyped_storage().nbytes() for t in unique_caches.values())
        with memory_profiling(
            worker.init_snapshot, weights_memory=int(runner.model_memory_usage),
        ) as phase:
            rows = [_run_case(runtime, case, blocks_per_req) for case in cases]
        dummy_bytes = (runner.slackkv_dummy_table.numel() * 4
                       + runner.slackkv_dummy_slots.numel() * 8)
        runtime.resource_profile = {
            "resource_profile_only": True, "attention_executed": True,
            "kv_layout": "layer_page_arena",
            "legal_global_token_request_shapes": True,
            "temporary_kv_bytes": actual_kv_bytes,
            "temporary_kv_blocks": temporary.num_blocks,
            "temporary_layer_pages": runner.slackkv_kv_layout.num_pages,
            "temporary_unique_storages": len(unique_caches),
            "temporary_kv_phase_peak_allocated": phase.after_profile.torch_peak,
            "temporary_kv_phase_peak_reserved": torch.cuda.max_memory_reserved(),
            "non_kv_cache_memory": phase.non_kv_cache_memory - actual_kv_bytes,
            "persistent_non_kv_memory": phase.total_consumed - actual_kv_bytes,
            "transient_peak_headroom": phase.transient_peak_headroom,
            "allocation_block_size": block_size,
            "kernel_block_size": runner._kernel_block_sizes[0],
            "layer_metadata": {lane: buffers.manifest()
                               for lane, buffers in runtime.layer_buffers.items()},
            "dummy_metadata_bytes": dummy_bytes,
            "cases": rows, "numerical_acceptance": False,
        }
        del caches, unique_caches
        return runtime.resource_profile
    finally:
        if initialized:
            for stream in runtime.streams.values():
                stream.synchronize()
            runtime.clear_cache_metadata()
            runner._cleanup_profiling_kv_cache()
            if hasattr(runner, "_kv_block_zeroer"):
                del runner._kv_block_zeroer
