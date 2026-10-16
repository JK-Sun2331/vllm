# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-host dual-lane execution with leased LayerPage KV mappings."""

from dataclasses import dataclass, replace
import time

import torch

from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.slackkv.elastic_runner import ElasticRunner
from vllm.v1.slackkv.lane_state import LaneInputs, LaneState
from vllm.v1.slackkv.model_adapter import Qwen3Adapter
from vllm.v1.slackkv.protocol import LaneCompletion, LaneStepPlan
from vllm.v1.slackkv.layer_metadata import LayerMetadataBuffers, build_metadata
from vllm.v1.slackkv.resident_buffers import ElasticInputSlot
from vllm.v1.worker.workspace import (
    current_workspace_manager, lock_workspace, use_workspace_lane,
)


def greedy_metadata(output_ids, num_reqs=1):
    return SamplingMetadata(
        temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(num_reqs, device="cuda"),
        presence_penalties=torch.zeros(num_reqs, device="cuda"),
        repetition_penalties=torch.ones(num_reqs, device="cuda"),
        output_token_ids=[list(output_ids) for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
    )


@dataclass
class WorkerTicket:
    plan: LaneStepPlan
    event: torch.cuda.Event
    state: LaneState | None = None
    sampler_output: object | None = None
    host_tokens: torch.Tensor | None = None
    direct_output: ModelRunnerOutput | None = None
    cancelled: bool = False
    fence_recorded: bool = False
    failed: bool = False
    prepared: object | None = None
    applied_versions: tuple = ()
    cleanup_acks: tuple = ()

    @property
    def control(self):
        return self.plan.output.total_num_scheduled_tokens == 0


class DualLaneRuntime:
    def __init__(self, worker):
        self.worker = worker
        self.runner = worker.model_runner
        self.config = worker.vllm_config
        self.streams = {"P": torch.cuda.Stream(priority=-1),
                        "E": torch.cuda.Stream(priority=0)}
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        for stream in self.streams.values():
            stream.wait_event(ready)
        self.adapter = Qwen3Adapter(self.runner.model)
        self.outputs = {
            "P": torch.empty((self.runner.max_num_reqs, 1), dtype=torch.int32,
                             device="cpu", pin_memory=True),
            "E": torch.empty((1, 1), dtype=torch.int32,
                             device="cpu", pin_memory=True),
        }
        self.output_busy = {"P": False, "E": False}
        self.input_slot = None
        self.elastic = ElasticRunner(self.adapter, self.config, self.streams["E"],
                                     self.outputs["E"])
        self.tickets: dict[str, WorkerTicket] = {}
        self.events: list[dict] = []
        self.enable_event_log = self.config.additional_config["slackkv"].get(
            "enable_event_log", True)
        self.resource_profile = {}
        self.hold_after_layer: int | None = None
        self.hold_step = 1
        self.hold_target = None
        self.hold_manual = False
        self.hold_release = False
        self.hold_used = False
        self.p_submissions = 0
        self.segment_layers = 5
        self.segment_cuts = None
        self.workspace_locked = False
        self.started = False
        self.closed = False
        self.startup_event = None
        self.greedy = None
        self.submitted_p_tokens = {}
        self.layer_buffers = {}
        self.elastic_metadata_builder = None
        self.p_native_ref_by_id = {}
        self.failed = False

    def emit(self, event, lane=None, step_id=None, **values):
        if self.enable_event_log:
            self.events.append({"event": event, "host_ns": time.monotonic_ns(),
                                "lane": lane, "step_id": step_id, **values})

    def caches(self):
        layers = self.config.compilation_config.static_forward_context
        result = {name: layer.kv_cache for name, layer in layers.items()
                  if hasattr(layer, "kv_cache")}
        assert len(result) == 36
        return result

    @torch.inference_mode()
    def profile(self):
        from vllm.v1.slackkv.resident_profiling import profile_resident_lanes

        return profile_resident_lanes(self)

    def configure_cache(self):
        groups = self.runner.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("resident_dual_lane requires one native KV cache group")
        block_size = groups[0].kv_cache_spec.block_size
        kernel_size = self.runner._kernel_block_sizes[0]
        if any(t.shape[2] != kernel_size for t in self.caches().values()):
            raise ValueError("resident FA2 arena does not match its kernel block size")
        layout = self.runner.slackkv_kv_layout
        self.layer_buffers = {
            lane: LayerMetadataBuffers(
                self.worker.device, layout.layer_names, rows,
                self.runner.max_num_tokens, self.config.model_config.max_model_len,
                block_size, kernel_size, layout.num_pages,
            )
            for lane, rows in (("P", self.runner.max_num_reqs), ("E", 1))
        }
        self.runner.slackkv_layer_buffers = self.layer_buffers["P"]
        group = self.runner.attn_groups[0][0]
        with torch.cuda.stream(self.streams["E"]), use_workspace_lane(1):
            self.elastic_metadata_builder = group.backend.get_builder_cls()(
                group.kv_cache_spec.copy_with_new_block_size(kernel_size),
                group.layer_names, self.config, self.worker.device,
            )
        if self.input_slot is None:
            self.input_slot = ElasticInputSlot(
                self.worker.device, self.runner.max_num_tokens,
                self.config.model_config.max_model_len, block_size, kernel_size,
            )
        elif (self.input_slot.block_size, self.input_slot.kernel_size) != (
                block_size, kernel_size):
            raise ValueError("profiling and actual KV block layouts differ")
        if self.greedy is None:
            with torch.cuda.stream(self.streams["E"]):
                self.greedy = greedy_metadata([])

    def clear_cache_metadata(self):
        for buffers in self.layer_buffers.values():
            buffers.clear()
        self.layer_buffers.clear()
        self.elastic_metadata_builder = None
        self.runner.slackkv_layer_buffers = None
        self.runner.slackkv_prepared_layers = None
        self.runner.slackkv_step_plan = None

    def _begin_ticket(self, plan):
        assert self.started and not self.closed and not self.failed
        lane = plan.lane
        assert lane not in self.tickets and not self.output_busy[lane]
        if (not plan.output.total_num_scheduled_tokens
                and (plan.refs or plan.leases or plan.inputs)):
            raise RuntimeError("cleanup-only plan must not hold request leases")
        ticket = WorkerTicket(plan, torch.cuda.Event())
        # Register before the first metadata DMA, zero or model kernel launch.
        self.tickets[lane] = ticket
        self.output_busy[lane] = True
        return ticket

    def _submission_failed(self, ticket):
        ticket.failed = self.failed = True
        try:
            ticket.event.record(self.streams[ticket.plan.lane])
            ticket.fence_recorded = True
        except Exception:
            # An unusable CUDA context is released by process exit, never recycled.
            ticket.fence_recorded = False
        self.emit("submission_failed", ticket.plan.lane, ticket.plan.step_id,
                  fence_recorded=ticket.fence_recorded)

    def _cleanup(self, plan):
        finished = set()
        acks = self.layer_buffers[plan.lane].cleanup(plan.cleanup_keys)
        for key in plan.cleanup_keys:
            if key.scope != "request":
                continue
            self.submitted_p_tokens.pop(key.ref, None)
            if plan.lane == "P":
                installed = self.p_native_ref_by_id.get(key.ref.request_id)
                if (installed is not None and installed[0] == key.ref
                        and installed[1] <= key.retired_allocation_version):
                    finished.add(key.ref.request_id)
                    del self.p_native_ref_by_id[key.ref.request_id]
        return acks, finished

    def _zero_new_pages(self, plan):
        if plan.zero_pages:
            if self.runner._kv_block_zeroer._meta is None:
                raise RuntimeError("LayerPage production zeroer has no arena metadata")
            page_ids = [page.page_id for page in plan.zero_pages]
            self.runner._zero_block_ids(page_ids)
            if self.enable_event_log:
                self.emit("pages_zeroed", plan.lane, plan.step_id, pages=page_ids)

    @staticmethod
    def _applied_versions(plan, prepared):
        actual = {version.ref: version for version in prepared.versions}
        expected = {lease.ref: lease.allocation_version for lease in plan.leases}
        if ({ref: version.allocation_version for ref, version in actual.items()}
                != expected):
            raise RuntimeError("worker applied mappings do not match step leases")
        return tuple(actual[lease.ref] for lease in plan.leases)

    def finish_startup(self):
        # The host also reuses native pinned warmup sources in execute_model.
        # Stream waits alone cannot protect those CPU writes from pending DMA.
        event = torch.cuda.Event(blocking=True)
        event.record(torch.cuda.current_stream())
        event.synchronize()
        for stream in self.streams.values():
            stream.wait_event(event)
        self.startup_event = event
        self._lock()
        self.started = True
        self.emit("startup_ready")

    def _lock(self):
        if not self.workspace_locked:
            lock_workspace()
            self.workspace_locked = True

    @torch.inference_mode()
    @with_gpu_sync_check
    def submit_protected_step(self, plan):
        ticket = self._begin_ticket(plan)
        try:
            with torch.cuda.stream(self.streams["P"]), use_workspace_lane(0):
                ticket.cleanup_acks, finished = self._cleanup(plan)
                self._zero_new_pages(plan)
                self.runner.slackkv_step_plan = plan
                self.runner.slackkv_layer_buffers = self.layer_buffers["P"]
                try:
                    direct = self.worker.execute_model(
                        replace(plan.output, finished_req_ids=finished))
                    if not ticket.control:
                        ticket.prepared = self.runner.slackkv_prepared_layers
                        if ticket.prepared is None:
                            raise RuntimeError("P forward did not consume layer mappings")
                        ticket.applied_versions = self._applied_versions(
                            plan, ticket.prepared)
                        for version in ticket.applied_versions:
                            self.p_native_ref_by_id[version.ref.request_id] = (
                                version.ref, version.allocation_version)
                    if direct is None:
                        state = self.runner.execute_model_state
                        assert state is not None and not ticket.control
                        sampled = self.runner._sample(
                            state.logits, state.spec_decode_metadata)
                        audit = getattr(self.runner, "slackkv_audit", None)
                        if audit is not None:
                            audit.finish(state.hidden_states, state.logits)
                        host = self.outputs["P"][:sampled.sampled_token_ids.shape[0]]
                        host.copy_(sampled.sampled_token_ids, non_blocking=True)
                        ticket.sampler_output = sampled
                        ticket.host_tokens = host
                    else:
                        ticket.direct_output = direct
                finally:
                    self.runner.slackkv_step_plan = None
                    self.runner.slackkv_prepared_layers = None
                ticket.event.record(self.streams["P"])
                ticket.fence_recorded = True
        except Exception:
            self._submission_failed(ticket)
            raise
        if plan.output.total_num_scheduled_tokens:
            self.p_submissions += 1
        for source in plan.inputs:
            self.submitted_p_tokens[source.ref] = source.computed
        if self.enable_event_log:
            self.emit("step_submitted", "P", plan.step_id,
                      requests=[r.request_id for r in plan.refs])
        return plan.step_id

    def _elastic_inputs(self, plan):
        source = plan.elastic
        assert source is not None
        request = Request(source.ref.request_id, list(source.prompt_ids),
                          source.sampling_params, None)
        request.append_output_token_ids(list(source.output_ids))
        request.num_computed_tokens = source.computed
        views = self.input_slot.acquire_tokens(source.input_ids, source.computed)
        prepared, metadata = self.prepare_elastic_layers(plan, views)
        self.tickets["E"].prepared = prepared
        self.tickets["E"].applied_versions = self._applied_versions(plan, prepared)
        return LaneInputs((request,), plan.step_id, source.ref.epoch,
                          views["ids"], views["positions"], views["lengths"],
                          metadata, prepared.slots,
                          self.greedy, tuple(self.input_slot.host.values()))

    def prepare_elastic_layers(self, plan, views):
        source = plan.elastic
        n = len(source.input_ids)
        length = source.computed + n
        prepared = self.layer_buffers["E"].prepare(
            plan, plan.refs, views["positions"], views["starts"], n)
        name = self.runner.slackkv_kv_layout.layer_names[0]
        common = CommonAttentionMetadata(
            query_start_loc=views["starts"],
            query_start_loc_cpu=self.input_slot.host["starts"],
            seq_lens=views["lengths"], num_reqs=1, num_actual_tokens=n,
            max_query_len=n, max_seq_len=length,
            block_table_tensor=prepared.tables[name],
            slot_mapping=prepared.slots[name],
            seq_lens_cpu_upper_bound=self.input_slot.host["lengths"],
            positions=views["positions"],
        )
        metadata = build_metadata(common, self.elastic_metadata_builder, prepared)
        return prepared, metadata

    def _audit_finish(self, state):
        audit = getattr(self.runner, "slackkv_audit", None)
        if audit is not None:
            audit.finish(state.hidden, state.logits)

    @torch.inference_mode()
    @with_gpu_sync_check
    def start_elastic_step(self, plan):
        ticket = self._begin_ticket(plan)
        try:
            with torch.cuda.stream(self.streams["E"]), use_workspace_lane(1):
                ticket.cleanup_acks, _ = self._cleanup(plan)
                if not ticket.control:
                    self._zero_new_pages(plan)
                    inputs = self._elastic_inputs(plan)
                    ready = torch.cuda.Event()
                    ready.record(self.streams["E"])
                    audit = getattr(self.runner, "slackkv_audit", None)
                    if audit is not None:
                        audit.begin_elastic(plan, ticket.prepared)
                    if plan.elastic.computed < len(plan.elastic.prompt_ids):
                        state = LaneState(inputs, next_layer=36)
                        self.elastic.state = state
                        ticket.state = state
                        with self.elastic._context(inputs):
                            state.hidden = self.runner._model_forward(
                                input_ids=inputs.input_ids, positions=inputs.positions)
                            state.logits = self.runner.model.compute_logits(
                                state.hidden[-1:])
                        self.elastic.submit_sample(self._audit_finish)
                    else:
                        ticket.state = self.elastic.begin_step(inputs, ready)
                ticket.event.record(self.streams["E"])
                ticket.fence_recorded = True
        except Exception:
            self._submission_failed(ticket)
            raise
        if self.enable_event_log:
            self.emit("step_submitted", "E", plan.step_id,
                      requests=[r.request_id for r in plan.refs])
        return plan.step_id

    @torch.inference_mode()
    @with_gpu_sync_check
    def advance_elastic(self, limit=1):
        assert not self.closed
        assert limit == 1
        ticket = self.tickets.get("E")
        if (ticket is None or ticket.control or ticket.failed or ticket.cancelled
                or not ticket.fence_recorded or ticket.state.completion is not None):
            return False
        if not ticket.event.query():
            return False
        state = ticket.state
        if (self.hold_after_layer is not None and not self.hold_used and
                ticket.plan.step_id == self.hold_step and
                state.next_layer >= self.hold_after_layer):
            if self.hold_target is None:
                candidates = [s for t in self.tickets.values()
                              if t.plan.lane == "P" for s in t.plan.inputs
                              if s.computed >= s.prompt_length]
                if not self.hold_manual and not candidates:
                    return False
                self.hold_target = (candidates[0].ref, candidates[0].computed + 1
                                    ) if candidates else (None, None)
                self.emit("test_gate_entered", "E", ticket.plan.step_id,
                          next_layer=state.next_layer,
                          target_request=(None if self.hold_target[0] is None
                                          else self.hold_target[0].request_id),
                          target_epoch=(None if self.hold_target[0] is None
                                        else self.hold_target[0].epoch),
                          target_computed=self.hold_target[1])
            target_ref, target_computed = self.hold_target
            if self.hold_manual:
                released = self.hold_release
            else:
                released = (self.submitted_p_tokens.get(target_ref, -1)
                            >= target_computed)
            if not released:
                return False
            self.hold_used = True
            self.emit("test_gate_released", "E", ticket.plan.step_id,
                      p_submissions=self.p_submissions)
        audit = getattr(self.runner, "slackkv_audit", None)
        if audit is not None:
            audit.resume_elastic(ticket.plan)
        end = (min(36, state.next_layer + self.segment_layers)
               if self.segment_cuts is None
               else next(cut for cut in self.segment_cuts if cut > state.next_layer))
        ticket.fence_recorded = False
        try:
            self.elastic.advance(end)
            if end == 36:
                self.elastic.submit_finish(self._audit_finish)
            with torch.cuda.stream(self.streams["E"]):
                ticket.event.record(self.streams["E"])
                ticket.fence_recorded = True
        except Exception:
            self._submission_failed(ticket)
            raise
        self.emit("segment_submitted", "E", ticket.plan.step_id, end_layer=end)
        return True

    def cancel_request(self, ref):
        for ticket in self.tickets.values():
            if ref in ticket.plan.refs:
                if ticket.plan.lane == "E":
                    ticket.cancelled = True
                self.emit("cancel_received", ticket.plan.lane, ticket.plan.step_id,
                          request_id=ref.request_id, epoch=ref.epoch)

    def poll_ready(self):
        # Empty polls only inspect host state and query a nonblocking event.
        # Keep all tensor/result handling inside the checked ready path.
        result = []
        for lane, ticket in list(self.tickets.items()):
            if ticket.failed or not ticket.fence_recorded:
                continue
            event = (ticket.state.completion or ticket.event
                     if lane == "E" and not ticket.control else ticket.event)
            if not event.query():
                continue
            if (lane == "E" and not ticket.control
                    and ticket.state.completion is None and not ticket.cancelled):
                continue
            result.append(self._collect_ready(lane, ticket))
        return result

    @torch.inference_mode()
    @with_gpu_sync_check
    def _collect_ready(self, lane, ticket):
        self.emit("output_ready" if not ticket.cancelled else "drain_ready",
                  lane, ticket.plan.step_id)
        if ticket.control:
            output = ticket.direct_output
        elif lane == "P":
            output = ticket.direct_output
            if output is None:
                with torch.cuda.stream(self.streams["P"]), use_workspace_lane(0):
                    output = self.runner.sample_tokens(
                        None, ready_sample=(ticket.sampler_output,
                                            ticket.host_tokens.tolist()))
        elif ticket.cancelled:
            output = None
        else:
            source = ticket.plan.elastic
            sampled = ticket.state.sampled_cpu.tolist()
            if source.computed + len(source.input_ids) < source.known:
                sampled = [[]]
            output = ModelRunnerOutput(
                req_ids=[source.ref.request_id],
                req_id_to_index={source.ref.request_id: 0},
                sampled_token_ids=sampled, logprobs=None,
                prompt_logprobs_dict={},
            )
        completion = LaneCompletion(
            lane, ticket.plan.step_id, ticket.plan.refs, output,
            ticket.cancelled, ticket.plan.leases, ticket.applied_versions,
            ticket.cleanup_acks,
        )
        if lane == "E" and not ticket.control:
            self.elastic.state = None
            self.input_slot.release()
        self.output_busy[lane] = False
        del self.tickets[lane]
        return completion

    def snapshot(self):
        caches = self.caches()
        storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                   for t in caches.values()}
        manager = current_workspace_manager()
        return {
            "events": list(self.events),
            "pending_lanes": list(self.tickets),
            "tickets": {lane: {"step_id": t.plan.step_id,
                               "cancelled": t.cancelled,
                               "last_event_ready": (t.fence_recorded
                                                    and t.event.query()),
                               "fence_recorded": t.fence_recorded,
                               "control": t.control,
                               "failed": t.failed,
                               "next_layer": (None if t.state is None
                                              else t.state.next_layer),
                               "completion_ready": (None if t.state is None
                                     or t.state.completion is None
                                     else t.state.completion.query())}
                        for lane, t in self.tickets.items()},
            "elastic_state_present": self.elastic.state is not None,
            "protected_execute_state_present": (
                self.runner.execute_model_state is not None),
            "output_slot_busy": dict(self.output_busy),
            "output_host_addresses": {k: t.data_ptr() for k, t in self.outputs.items()},
            "input_slot": (None if self.input_slot is None
                           else self.input_slot.snapshot()),
            "workspace_bytes": [0 if t is None else t.numel() * t.element_size()
                                for t in manager._current_workspaces],
            "workspace_addresses": [None if t is None else t.data_ptr()
                                    for t in manager._current_workspaces],
            "actual_kv_bytes": sum(storage.values()),
            "torch_allocated": torch.cuda.memory_allocated(),
            "torch_reserved": torch.cuda.memory_reserved(),
            "torch_peak_allocated": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved": torch.cuda.max_memory_reserved(),
            "startup_ready": self.started,
            "initial_resource_profile": self.resource_profile,
            "kv_layout": self.runner.slackkv_kv_layout.manifest(
                next(iter(caches.values()))),
            "layer_metadata": {lane: buffers.manifest()
                               for lane, buffers in self.layer_buffers.items()},
            "host_request_states": {
                "p_native_refs": len(self.p_native_ref_by_id),
                "submitted_p_tokens": len(self.submitted_p_tokens),
            },
        }

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        # Exit only: include any work queued before a failed submission created
        # its ticket, and do not release weights or staging during pending DMA.
        for stream in self.streams.values():
            stream.synchronize()
        self.runner.execute_model_state = None
        self.tickets.clear()
        self.elastic.state = None
        self.elastic = None
        self.adapter = None
        self.input_slot = None
        self.greedy = None
        self.outputs.clear()
        self.output_busy = {"P": False, "E": False}
        self.submitted_p_tokens.clear()
        self.p_native_ref_by_id.clear()
        self.clear_cache_metadata()
        self.startup_event = None
        self.streams.clear()
        self.runner = None
        self.worker = None
