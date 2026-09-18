# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, process-local Motivation2 events; never changes scheduling decisions."""

import atexit
import json
import os
import queue
import threading
import time
from pathlib import Path


class EventWriter:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.stream = (Path(directory) / f"engine-{os.getpid()}.jsonl").open("x")
        self.queue = queue.Queue(maxsize=65536)
        self.seq = 0
        self.dropped = 0
        self.error = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def emit(self, event_type, **fields):
        with self.lock:
            self.seq += 1
            event = dict(event_type=event_type, seq=self.seq, pid=os.getpid(),
                         mono_ns=time.monotonic_ns(), wall_ns=time.time_ns(),
                         dropped=self.dropped, **fields)
            try:
                self.queue.put_nowait(event)
            except queue.Full:
                self.dropped += 1

    def _write(self):
        try:
            while True:
                event = self.queue.get()
                if event is None:
                    break
                self.stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                if self.queue.empty():
                    self.stream.flush()
        except Exception as exc:
            self.error = type(exc).__name__

    def close(self):
        if self.stream.closed:
            return
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join()
        self.stream.write(json.dumps(dict(event_type="writer_end", seq=self.seq + 1,
                                         dropped=self.dropped,
                                         error=self.error)) + "\n")
        self.stream.close()


_writer = None


def emit(event_type, **fields):
    global _writer
    directory = os.environ.get("SLACKKV_PROFILE_DIR")
    if not directory:
        return
    if _writer is None:
        _writer = EventWriter(directory)
    _writer.emit(event_type, **fields)


def close():
    if _writer is not None:
        _writer.close()


class SchedulerProfile:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.step = 0
        self.checked = set()
        self.failures = {}
        self.tables = {}
        self.started_requests = set()
        self.allocation_seq = 0
        self.failure_events = []
        self.blocks = {}
        self.generations = {}
        self.pool = scheduler.kv_cache_manager.block_pool
        emit("pool_init", total_blocks=len(self.pool.blocks),
             null_blocks=[b.block_id for b in self.pool.blocks if b.is_null],
             block_size=scheduler.block_size,
             kv_tensor_bytes=sum(t.size for t in
                                 scheduler.kv_cache_config.kv_cache_tensors),
             kv_group_count=len(scheduler.kv_cache_config.kv_cache_groups))
        self._install_pool_hooks()

    def _block_changes(self, blocks, allocated=False):
        changed = []
        for block in blocks:
            bid = block.block_id
            if allocated:
                self.generations[bid] = self.generations.get(bid, 0) + 1
            state = (block.ref_cnt, block.block_hash is not None, block.is_null,
                     self.generations.get(bid, 0))
            if self.blocks.get(bid) != state:
                self.blocks[bid] = state
                changed.append([bid, *state])
        if changed:
            emit("block_delta", step_id=self.step, blocks=changed,
                 allocatable=self.pool.get_num_free_blocks())

    def _install_pool_hooks(self):
        # Observe the actual mutation boundaries, including deferred offload frees.
        original_new = self.pool.get_new_blocks
        def allocate(num_blocks):
            result = original_new(num_blocks)
            self._block_changes(result, allocated=True)
            return result
        self.pool.get_new_blocks = allocate
        for name in ("touch", "free_blocks"):
            original = getattr(self.pool, name)
            def mutate(blocks, original=original):
                blocks = list(blocks)
                result = original(blocks)
                self._block_changes(blocks)
                return result
            setattr(self.pool, name, mutate)
        original_insert = self.pool._insert_block_hash
        def insert(key, block, num_tokens):
            result = original_insert(key, block, num_tokens)
            self._block_changes([block])
            return result
        self.pool._insert_block_hash = insert
        original_remove = self.pool._remove_cached_block_hashes
        def remove(block):
            result = original_remove(block)
            self._block_changes([block])
            return result
        self.pool._remove_cached_block_hashes = remove
        original_reset = self.pool.reset_prefix_cache
        def reset():
            result = original_reset()
            if result:
                self._block_changes(self.pool.blocks)
            return result
        self.pool.reset_prefix_cache = reset

    def ownership(self):
        current = {}
        for rid in self.scheduler.requests:
            groups = self.scheduler.kv_cache_manager.get_blocks(rid).blocks
            table = tuple(tuple(b.block_id for b in group if not b.is_null)
                          for group in groups)
            current[rid] = table
            if self.tables.get(rid) != table:
                emit("request_blocks", step_id=self.step, request_id=rid,
                     groups=table)
        for rid in self.tables.keys() - current.keys():
            emit("request_blocks", step_id=self.step, request_id=rid, groups=[])
        self.tables = current

    def begin(self):
        self.step += 1
        self.loop_stop_reason = "none"
        self.wait_details = {}
        self.stop_events = []
        self.last_failure = {}
        self.checked.clear()
        self.failures.clear()
        self.failure_events.clear()
        emit("step_start", step_id=self.step)

    def defer(self, request, reason, **details):
        self.wait_details[request.request_id] = dict(
            cause=reason, checked=True, **details)

    def stop(self, phase, reason, request=None, **details):
        """Record a traversal stop, not a capacity claim about unvisited requests."""
        self.loop_stop_reason = reason
        trigger = request.request_id if request is not None else None
        event = dict(phase=phase, cause=reason, trigger_request_id=trigger,
                     trigger_event_id=self.last_failure.get(trigger), **details)
        queues = ([self.scheduler.running] if phase == "running" else
                  [self.scheduler.waiting, self.scheduler.skipped_waiting])
        affected = []
        for queue in queues:
            for candidate in queue:
                rid = candidate.request_id
                if rid in self.checked or candidate.is_finished():
                    continue
                affected.append(rid)
                self.wait_details[rid] = dict(event, checked=False)
        self.stop_events.append(dict(event, unchecked_request_ids=affected))
        if request is not None:
            self.defer(request, reason, phase=phase,
                       trigger_event_id=self.last_failure.get(trigger))

    def request_state(self, request):
        computed = request.num_computed_tokens
        prompt = request.num_prompt_tokens
        return dict(status=request.status.name, computed_tokens=computed,
                    prompt_tokens=prompt, output_tokens=request.num_output_tokens,
                    decode_entered=request.num_output_tokens > 0,
                    phase="prefill_or_recompute" if computed < prompt else "decode")

    def selected(self, output):
        # Final schedule, before _update_after_schedule advances token counters.
        stamp = time.monotonic_ns()
        states = {}
        for rid, tokens in output.num_scheduled_tokens.items():
            request = self.scheduler.requests[rid]
            state = self.request_state(request)
            prompt_remaining = max(0, request.num_prompt_tokens -
                                   request.num_computed_tokens)
            state["scheduled_tokens"] = tokens
            state["scheduled_prompt_tokens"] = min(tokens, prompt_remaining)
            state["selection_kind"] = (
                "first" if rid not in self.started_requests else
                "resumed" if rid in output.scheduled_cached_reqs.resumed_req_ids
                else "continued")
            states[rid] = state
            if rid not in self.started_requests:
                emit("request_first_selected", request_id=rid,
                     selected_mono_ns=stamp, step_id=self.step)
        emit("schedule_selected", step_id=self.step, selected_mono_ns=stamp,
             requests=states, failure_events=list(self.failure_events))

    def allocation(self, request, needed, required, available, phase):
        failed = required > available
        self.allocation_seq += 1
        if failed:
            self.ownership()
        physical_free = self.pool.get_num_free_blocks()
        physical_deficit = max(0, needed - physical_free)
        self.failures[request.request_id] = (
            "kv_allocation_failed" if physical_deficit
            else "kv_reservation_or_watermark"
        ) if failed else None
        event_id = f"{os.getpid()}:{self.allocation_seq}"
        if failed:
            self.last_failure[request.request_id] = event_id
            self.failure_events.append(event_id)
            emit("contention_snapshot", step_id=self.step,
                 contention_event_id=event_id, request_id=request.request_id,
                 running=[r.request_id for r in self.scheduler.running],
                 waiting=[r.request_id for r in self.scheduler.waiting],
                 skipped_waiting=[r.request_id for r in self.scheduler.skipped_waiting],
                 requests={rid: self.request_state(r)
                           for rid, r in self.scheduler.requests.items()
                           if not r.is_finished()})
        emit("allocation_check", step_id=self.step, request_id=request.request_id,
             contention_event_id=event_id,
             needed_blocks=needed, required_blocks=required,
             available_blocks=available, deficit_blocks=max(0, required-available),
             physical_free_blocks=physical_free,
             physical_deficit_blocks=physical_deficit,
             watermark_blocks=required-needed, reserved_blocks=physical_free-available,
             phase=phase, failed=failed)

    def end(self, output, token_budget, input_budget, draft_slots):
        self.ownership()
        scheduled = output.num_scheduled_tokens
        unscheduled = {}
        attribution = {}
        for rid, request in self.scheduler.requests.items():
            if rid in scheduled or request.is_finished():
                continue
            reason = "other" if rid in self.checked else "not_checked"
            if "REMOTE_KV" in request.status.name:
                reason = "kv_transfer"
            elif self.failures.get(rid):
                reason = self.failures[rid]
            unscheduled[rid] = reason
            detail = self.wait_details.get(rid)
            if reason not in ("not_checked", "other"):
                detail = dict(cause=reason, checked=rid in self.checked,
                              trigger_request_id=rid,
                              trigger_event_id=self.last_failure.get(rid))
            attribution[rid] = detail or dict(
                cause="unknown_unchecked" if reason == "not_checked" else
                      "unknown_checked", checked=rid in self.checked)

        stop = self.loop_stop_reason
        if unscheduled and token_budget <= 0 and stop == "none":
            stop = "token_budget"
        first = sorted(set(scheduled) - self.started_requests)
        self.started_requests.update(scheduled)
        emit("step_end", step_id=self.step, scheduled=dict(scheduled),
             first_scheduled=first,
             preempted=list(output.preempted_req_ids or []),
             resumed=list(output.scheduled_cached_reqs.resumed_req_ids),
             unscheduled=unscheduled, loop_stop_reason=stop,
             wait_attribution=attribution, traversal_stops=self.stop_events,
             running=[r.request_id for r in self.scheduler.running],
             waiting=[r.request_id for r in self.scheduler.waiting],
             skipped_waiting=[r.request_id for r in self.scheduler.skipped_waiting],
             allocatable=self.pool.get_num_free_blocks())


def get_profile(scheduler):
    if not os.environ.get("SLACKKV_PROFILE_DIR"):
        return None
    profile = getattr(scheduler, "_slackkv_profile", None)
    if profile is None:
        profile = SchedulerProfile(scheduler)
        scheduler._slackkv_profile = profile
        scheduler.kv_cache_manager._slackkv_profile = profile
    return profile
