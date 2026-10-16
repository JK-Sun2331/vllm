# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plans for the single-GPU resident dual-lane runtime."""

from dataclasses import dataclass, replace

import msgspec

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput


@dataclass(frozen=True)
class RequestRef:
    request_id: str
    epoch: int


class PageHandle(msgspec.Struct, frozen=True):
    page_id: int
    generation: int


@dataclass(frozen=True)
class LayerMappingUpdate:
    ref: RequestRef
    kind: str
    base_version: int
    target_version: int
    append_start: int
    target_blocks: int
    layer_pages: tuple[tuple[PageHandle, ...], ...]


@dataclass(frozen=True)
class StepLease:
    ref: RequestRef
    allocation_version: int
    lane: str
    step_id: int


@dataclass(frozen=True)
class CleanupKey:
    ref: RequestRef
    retired_allocation_version: int
    scope: str


@dataclass(frozen=True)
class AppliedVersion:
    ref: RequestRef
    allocation_version: int


@dataclass(frozen=True)
class ProtectedStepInput:
    ref: RequestRef
    input_ids: tuple[int, ...]
    computed: int
    known: int
    prompt_length: int


@dataclass(frozen=True)
class RequestStepInput:
    ref: RequestRef
    prompt_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    computed: int
    known: int
    sampling_params: SamplingParams

    @property
    def prompt_length(self):
        return len(self.prompt_ids)


@dataclass(frozen=True)
class LaneStepPlan:
    lane: str
    step_id: int
    refs: tuple[RequestRef, ...]
    output: SchedulerOutput
    inputs: tuple[RequestStepInput | ProtectedStepInput, ...] = ()
    layer_updates: tuple[LayerMappingUpdate, ...] = ()
    leases: tuple[StepLease, ...] = ()
    cleanup_keys: tuple[CleanupKey, ...] = ()
    zero_pages: tuple[PageHandle, ...] = ()

    @property
    def elastic(self):
        return self.inputs[0] if self.lane == "E" and self.inputs else None


@dataclass
class LaneCompletion:
    lane: str
    step_id: int
    refs: tuple[RequestRef, ...]
    output: ModelRunnerOutput | None
    cancelled: bool = False
    leases: tuple[StepLease, ...] = ()
    applied_versions: tuple[AppliedVersion, ...] = ()
    cleanup_acks: tuple[CleanupKey, ...] = ()


def select_output(
    output: SchedulerOutput,
    ids: set[str],
    finished: set[str] | None = None,
    zero_blocks: list[int] | None = None,
) -> SchedulerOutput:
    """Copy the per-request containers without changing the original plan."""
    cached = output.scheduled_cached_reqs
    indices = [i for i, rid in enumerate(cached.req_ids) if rid in ids]
    selected = CachedRequestData(
        req_ids=[cached.req_ids[i] for i in indices],
        resumed_req_ids=cached.resumed_req_ids & ids,
        new_token_ids=([cached.new_token_ids[i] for i in indices]
                       if cached.new_token_ids else []),
        all_token_ids={k: list(v) for k, v in cached.all_token_ids.items()
                       if k in ids},
        new_block_ids=[cached.new_block_ids[i] for i in indices],
        num_computed_tokens=[cached.num_computed_tokens[i] for i in indices],
        num_output_tokens=[cached.num_output_tokens[i] for i in indices],
    )
    counts = {k: v for k, v in output.num_scheduled_tokens.items() if k in ids}
    return replace(
        output,
        scheduled_new_reqs=[r for r in output.scheduled_new_reqs if r.req_id in ids],
        scheduled_cached_reqs=selected,
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set() if finished is None else set(finished),
        new_block_ids_to_zero=zero_blocks,
    )
