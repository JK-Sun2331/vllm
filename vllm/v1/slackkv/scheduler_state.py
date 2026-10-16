# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident lane ownership, mapping leases and identity-safe cleanup."""

from copy import copy
from dataclasses import dataclass
import time

from vllm.v1.slackkv.config import SlackKVConfig
from vllm.v1.slackkv.protocol import (
    AppliedVersion, CleanupKey, LaneCompletion, LaneStepPlan, ProtectedStepInput,
    RequestRef, RequestStepInput, select_output,
)


@dataclass
class PendingStep:
    plan: LaneStepPlan
    requests: tuple


class ResidentSchedulerState:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.enable_event_log = SlackKVConfig.from_additional_config(
            scheduler.vllm_config.additional_config).enable_event_log
        self.refs: dict[str, RequestRef] = {}
        self.roles: dict[RequestRef, str] = {}
        self.pending: dict[str, PendingStep] = {}
        self.steps = {"P": 0, "E": 0}
        self.completed_steps = {"P": -1, "E": -1}
        self.epoch = 0
        self.events: list[dict] = []
        self.cleanup: set[CleanupKey] = set()
        self.terminal: set[RequestRef] = set()
        self.blocked_lanes: set[str] = set()
        self.allocation_failures: list[dict] = []
        self.manager = scheduler.kv_cache_manager
        self.manager.set_ref_resolver(self.refs.__getitem__)
        self.manager.arena.add_observer(self._page_event)

    def _page_event(self, action, handles):
        if not self.enable_event_log:
            return
        arena = self.manager.arena
        self.emit(
            "layer_page_" + action,
            pages=[{
                "page_id": handle.page_id, "generation": handle.generation,
                "state": arena.pages[handle.page_id].state,
                "owner": (None if arena.pages[handle.page_id].owner is None else {
                    "request_id": arena.pages[handle.page_id].owner.ref.request_id,
                    "epoch": arena.pages[handle.page_id].owner.ref.epoch,
                    "layer": arena.pages[handle.page_id].owner.layer,
                    "logical_block": arena.pages[handle.page_id].owner.logical_block,
                }),
            } for handle in handles],
            capacity=arena.snapshot(),
        )

    def emit(self, event, **values):
        if not self.enable_event_log:
            return
        self.events.append({"event": event, "host_ns": time.monotonic_ns(), **values})

    def register(self, request):
        params = request.sampling_params
        if (
            params is None or params.temperature != 0 or
            params.frequency_penalty != 0 or params.presence_penalty != 0 or
            params.repetition_penalty != 1 or params.min_tokens != 0 or
            params.logprobs is not None or params.prompt_logprobs is not None or
            params.logprob_token_ids is not None or
            params.thinking_token_budget is not None or
            params.allowed_token_ids is not None or params.bad_words or
            params.logit_bias or request.use_structured_output or
            request.has_encoder_inputs or request.lora_request is not None or
            request.prompt_embeds is not None or request.resumable
        ):
            raise ValueError("resident_dual_lane requires plain greedy text requests")
        role = (params.extra_args or {}).get("slackkv_lane", "P")
        if role not in ("P", "E"):
            raise ValueError("slackkv_lane must be P or E")
        self.epoch += 1
        ref = RequestRef(request.request_id, self.epoch)
        self.refs[request.request_id] = ref
        self.roles[ref] = role
        self.emit("request_registered", lane=role, request_id=ref.request_id,
                  epoch=ref.epoch)

    def role(self, request_id):
        return self.roles[self.refs[request_id]]

    def eligible(self, request, *, waiting=False):
        role = self.role(request.request_id)
        if role in self.pending or (waiting and role in self.blocked_lanes):
            return False
        if role == "E":
            return not any(
                r is not request and not r.is_finished() and
                self.role(r.request_id) == "E"
                for r in self.scheduler.running
            )
        return True

    def begin_schedule(self):
        self.blocked_lanes.clear()
        self.allocation_failures.clear()

    def allocation_blocked(self, request, queue, num_tokens):
        ref = self.refs[request.request_id]
        lane = self.roles[ref]
        self.blocked_lanes.add(lane)
        failure = {
            "lane": lane, "request_id": ref.request_id, "epoch": ref.epoch,
            "queue": queue, "input_tokens": num_tokens,
            "computed_tokens": request.num_computed_tokens,
            "known_tokens": request.num_tokens,
            "free_pages": self.manager.arena.counts["FREE"],
            "allocation_check": self.manager.last_failure,
            "held_input_tokens": self.in_flight_tokens,
        }
        self.allocation_failures.append(failure)
        self.emit("allocation_blocked", **failure)

    def check_capacity_progress(self, output, preempted):
        # Native P preemption can release pages for the next complete pass.
        # Pending tickets, including cancelled tickets, must drain before we
        # can conclude that no real GPU work can make pages available.
        if (
            not self.allocation_failures or output.total_num_scheduled_tokens or
            self.pending or self.cleanup or output.finished_req_ids or preempted or
            self.scheduler.pause_state.name != "UNPAUSED"
        ):
            return
        state = {
            "free_pages": self.manager.arena.counts["FREE"],
            "p_pending": "P" in self.pending,
            "e_pending": "E" in self.pending,
            "held_input_tokens": self.in_flight_tokens,
        }
        self.emit("capacity_blocked", failures=list(self.allocation_failures), **state)
        raise RuntimeError(
            "resident_dual_lane KV allocation blocked: no runnable token plan, "
            "in-flight ticket, cleanup, or newly preempted P can release pages; "
            f"state={state!r}; failures={self.allocation_failures!r}"
        )

    def in_flight(self, request):
        return any(request in p.requests for p in self.pending.values())

    @property
    def in_flight_tokens(self):
        return sum(p.plan.output.total_num_scheduled_tokens
                   for p in self.pending.values())

    @property
    def draining_members(self):
        return sum(r.is_finished() for p in self.pending.values() for r in p.requests)

    def finished(self, request):
        ref = self.refs[request.request_id]
        role = self.roles[ref]
        self.terminal.add(ref)
        self.cleanup.add(CleanupKey(ref, self.manager.latest_version(ref), "request"))
        # Bare native IDs cannot safely identify an old epoch's worker state.
        self.scheduler.finished_req_ids.discard(request.request_id)
        self.emit("request_finished", lane=role, request_id=ref.request_id,
                  epoch=ref.epoch, status=request.status.name)

    def retire(self, request):
        ref = self.refs[request.request_id]
        allocation = self.manager.allocation(ref)
        version = self.manager.detach_request(ref)
        scope = "request" if request.is_finished() else "allocation"
        if scope == "request" or version:
            self.cleanup.add(CleanupKey(
                ref, version or self.manager.latest_version(ref), scope,
            ))
        if allocation is not None and self.enable_event_log:
            deferred = allocation.lease is not None
            self.emit(
                "pages_deferred" if deferred else "pages_released",
                lane=self.roles[ref], request_id=ref.request_id, epoch=ref.epoch,
                allocation_version=version,
                pages=[p.page_id for p in allocation.retired_pages],
            )

    def _prune(self, ref):
        if (
            ref not in self.terminal
            or any(key.ref == ref for key in self.cleanup)
            or any(lease.ref == ref for lease in self.manager.leases)
        ):
            return
        if self.refs.get(ref.request_id) == ref:
            del self.refs[ref.request_id]
        self.roles.pop(ref, None)
        self.terminal.remove(ref)

    def plans(self, output):
        if output.kv_cache_block_copies or output.kv_connector_metadata:
            raise RuntimeError("resident_dual_lane received unsupported KV work")
        result = []
        lane_refs = {"P": [], "E": []}
        for rid in output.num_scheduled_tokens:
            ref = self.refs[rid]
            lane_refs[self.roles[ref]].append(ref)
        for lane in ("P", "E"):
            refs = tuple(lane_refs[lane])
            ids = {ref.request_id for ref in refs}
            if lane in self.pending:
                if ids:
                    raise RuntimeError(f"native scheduler selected busy lane {lane}")
                continue
            cleanup = tuple(sorted(
                (key for key in self.cleanup
                 if self.roles[key.ref] == lane and self.manager.cleanup_ready(key)),
                key=lambda key: (
                    key.ref.epoch, key.retired_allocation_version, key.scope,
                ),
            ))
            if not ids and not cleanup:
                continue
            requests = tuple(self.scheduler.requests[ref.request_id] for ref in refs)
            selected = select_output(output, ids)
            inputs = []
            for ref, request in zip(refs, requests):
                start = request.num_computed_tokens - selected.num_scheduled_tokens[
                    request.request_id]
                input_ids = tuple(request.all_token_ids[
                    start:request.num_computed_tokens])
                if lane == "P":
                    inputs.append(ProtectedStepInput(
                        ref, input_ids, start, request.num_tokens,
                        request.num_prompt_tokens,
                    ))
                else:
                    inputs.append(RequestStepInput(
                        ref, tuple(request.prompt_token_ids),
                        tuple(request.output_token_ids), input_ids, start,
                        request.num_tokens, copy(request.sampling_params),
                    ))
            if lane == "E" and inputs:
                assert len(inputs) == 1
            step_id = self.steps[lane]
            # No lease exists for a selection revoked earlier in schedule().
            leases = tuple(
                self.manager.acquire_step(ref, lane, step_id) for ref in refs
            )
            updates = tuple(update for ref in refs
                            if (update := self.manager.mapping_update(ref)) is not None)
            zeros = tuple(page for lease in leases
                          for page in self.manager.zero_pages_for(lease))
            plan = LaneStepPlan(
                lane, step_id, refs, selected, tuple(inputs), updates, leases,
                cleanup, zeros,
            )
            self.steps[lane] += 1
            self.pending[lane] = PendingStep(plan, requests)
            if lane == "P":
                self.scheduler.prev_step_scheduled_req_ids = set(ids)
            if self.enable_event_log:
                self.emit(
                    "step_planned", lane=lane, step_id=plan.step_id,
                    requests=[{"id": r.request_id, "epoch": r.epoch} for r in refs],
                    tokens=selected.total_num_scheduled_tokens,
                    mapping_versions=[lease.allocation_version for lease in leases],
                    zero_pages=[page.page_id for page in zeros],
                    cleanup_count=len(cleanup),
                )
            result.append(plan)
        return result

    def complete(self, completion: LaneCompletion):
        # One in-flight step per lane and monotonically allocated step IDs
        # make a scalar retirement watermark sufficient. An old notification
        # must not touch a newer pending step, token counters, or freed blocks.
        if completion.lane not in self.completed_steps:
            raise RuntimeError("unknown resident completion lane")
        if 0 <= completion.step_id <= self.completed_steps[completion.lane]:
            self.emit("completion_duplicate_ignored", lane=completion.lane,
                      step_id=completion.step_id)
            return {}
        pending = self.pending.get(completion.lane)
        if pending is None:
            raise RuntimeError("resident completion identity mismatch")
        plan = pending.plan
        expected_versions = tuple(
            AppliedVersion(lease.ref, lease.allocation_version) for lease in plan.leases
        )
        if (
            (completion.step_id, completion.refs) != (plan.step_id, plan.refs)
            or completion.leases != plan.leases
            or completion.applied_versions != expected_versions
            or completion.cleanup_acks != plan.cleanup_keys
        ):
            raise RuntimeError("resident completion identity mismatch")
        # Validate the entire envelope before mutating any lease or request.
        for lease in plan.leases:
            allocation = self.manager.leases.get(lease)
            if (
                allocation is None or allocation.lease != lease
                or allocation.ref != lease.ref
                or allocation.version != lease.allocation_version
            ):
                raise RuntimeError("resident completion lost its allocation lease")
        if any(key not in self.cleanup or not self.manager.cleanup_ready(key)
               for key in completion.cleanup_acks):
            raise RuntimeError("resident completion cleanup identity mismatch")
        for lease in plan.leases:
            self.manager.ack_mapping(lease)
        for lease in plan.leases:
            allocation = self.manager.leases[lease]
            retired_pages = allocation.retired_pages if allocation.retired else ()
            self.manager.release_step(lease)
            if retired_pages and self.enable_event_log:
                self.emit(
                    "pages_released", lane=plan.lane, step_id=plan.step_id,
                    request_id=lease.ref.request_id, epoch=lease.ref.epoch,
                    allocation_version=lease.allocation_version,
                    pages=[page.page_id for page in retired_pages],
                )
        valid = set()
        previous_outputs = {}
        for ref, request in zip(plan.refs, pending.requests):
            if (self.refs.get(ref.request_id) == ref and
                    self.scheduler.requests.get(ref.request_id) is request and
                    not request.is_finished() and not completion.cancelled):
                valid.add(ref.request_id)
                if self.enable_event_log:
                    previous_outputs[ref.request_id] = request.num_output_tokens
            else:
                request.num_in_flight_tokens -= plan.output.num_scheduled_tokens[
                    ref.request_id]
        outputs = {}
        if valid:
            assert completion.output is not None
            selected = (plan.output if len(valid) == len(plan.refs)
                        else select_output(plan.output, valid))
            outputs = self.scheduler.update_from_output(
                selected, completion.output)
        if self.enable_event_log:
            accepted = [
                {
                    "request_id": ref.request_id, "epoch": ref.epoch,
                    "tokens": list(request.output_token_ids[
                        previous_outputs[ref.request_id]:]),
                    "computed_tokens": request.num_computed_tokens,
                    "in_flight_tokens": request.num_in_flight_tokens,
                    "known_tokens": request.num_tokens,
                    "status": request.status.name,
                }
                for ref, request in zip(plan.refs, pending.requests)
                if ref.request_id in valid
            ]
            self.emit("completion_accepted", lane=plan.lane, step_id=plan.step_id,
                      valid_requests=sorted(valid), cancelled=completion.cancelled,
                      accepted=accepted)
        for key in completion.cleanup_acks:
            self.cleanup.remove(key)
            self.manager.cleanup_ack(key)
            self.emit(
                "cleanup_acked", lane=plan.lane, step_id=plan.step_id,
                request_id=key.ref.request_id, epoch=key.ref.epoch,
                allocation_version=key.retired_allocation_version, scope=key.scope,
            )
            self._prune(key.ref)
        del self.pending[completion.lane]
        self.completed_steps[plan.lane] = plan.step_id
        return outputs
