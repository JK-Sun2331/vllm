# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""S1 resident E step; scheduling and asynchronous result collection are S2."""

from contextlib import contextmanager

import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.sample.sampler import Sampler
from vllm.v1.slackkv.lane_state import LaneInputs, LaneState
from vllm.v1.slackkv.model_adapter import LayerObserver, Qwen3Adapter
from vllm.v1.worker.workspace import use_workspace_lane


class ElasticRunner:
    def __init__(self, adapter: Qwen3Adapter, config: VllmConfig,
                 stream: torch.cuda.Stream | None = None,
                 output_buffer: torch.Tensor | None = None):
        self.adapter = adapter
        self.config = config
        self.stream = torch.cuda.Stream() if stream is None else stream
        self.sampler = Sampler()
        self.state: LaneState | None = None
        self.output_buffer = output_buffer

    @contextmanager
    def _context(self, inputs: LaneInputs):
        with (
            torch.cuda.stream(self.stream),
            use_workspace_lane(1),
            set_forward_context(
                inputs.attn_metadata, self.config,
                num_tokens=inputs.input_ids.numel(),
                slot_mapping=inputs.slot_mapping, skip_compiled=True,
            ),
        ):
            yield

    @torch.inference_mode()
    def begin_step(self, inputs: LaneInputs,
                   inputs_ready: torch.cuda.Event | None = None) -> LaneState:
        assert len(inputs.members) == 1
        assert self.state is None or self.state.completion is not None
        if self.state is not None:
            assert self.state.completion.query()
        if inputs_ready is None:
            self.stream.wait_stream(torch.cuda.current_stream())
        else:
            self.stream.wait_event(inputs_ready)
        state = self.state = LaneState(inputs)
        with self._context(inputs):
            self.adapter.begin(state)
        return state

    @torch.inference_mode()
    def advance(self, end: int, observer: LayerObserver | None = None) -> None:
        assert self.state is not None
        with self._context(self.state.inputs):
            self.adapter.run_layers(self.state, end, observer)

    @torch.inference_mode()
    def submit_finish(self, before_output=None) -> LaneState:
        assert self.state is not None
        state = self.state
        with self._context(state.inputs):
            self.adapter.finish(state)
        return self.submit_sample(before_output)

    @torch.inference_mode()
    def submit_sample(self, before_output=None) -> LaneState:
        assert self.state is not None
        state = self.state
        with self._context(state.inputs):
            state.sampled_gpu = self.sampler(
                state.logits, state.inputs.sampling_metadata).sampled_token_ids
            if before_output is not None:
                before_output(state)
            if self.output_buffer is None:
                # S1's serial entry owns its output per step.
                state.sampled_cpu = torch.empty(
                    state.sampled_gpu.shape, dtype=state.sampled_gpu.dtype,
                    device="cpu", pin_memory=True)
            else:
                assert state.sampled_gpu.shape == self.output_buffer.shape
                assert state.sampled_gpu.dtype == self.output_buffer.dtype
                state.sampled_cpu = self.output_buffer
            state.sampled_cpu.copy_(state.sampled_gpu, non_blocking=True)
            state.completion = torch.cuda.Event()
            state.completion.record(self.stream)
        return state

    @torch.inference_mode()
    def finish_step(self) -> LaneState:
        # Preserve the S1 serial entry; S2 only uses the submission methods.
        state = self.submit_finish()
        state.completion.synchronize()
        state.sampled_token = int(state.sampled_cpu.item())
        request = state.inputs.members[0]
        request.num_computed_tokens += state.inputs.input_ids.numel()
        request.append_output_token_ids(state.sampled_token)
        check_stop(request, self.config.model_config.max_model_len)
        return state
