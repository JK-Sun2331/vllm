# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invoke the loaded Qwen3 modules without owning another weight copy."""

from collections.abc import Callable

import torch

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.v1.slackkv.lane_state import LaneState

LayerObserver = Callable[[int, torch.Tensor, torch.Tensor], None]


class Qwen3Adapter:
    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.body = model.model
        assert self.body.start_layer == 0
        assert self.body.end_layer == len(self.body.layers) == 36

    def begin(self, state: LaneState) -> None:
        state.hidden = self.model.embed_input_ids(state.inputs.input_ids)

    def run_layers(
        self, state: LaneState, end: int, observer: LayerObserver | None = None
    ) -> None:
        assert state.next_layer < end <= self.body.end_layer
        for index in range(state.next_layer, end):
            state.hidden, state.residual = self.body.layers[index](
                state.inputs.positions, state.hidden, state.residual
            )
            state.next_layer = index + 1
            if observer is not None:
                observer(index, state.hidden, state.residual)

    def finish(self, state: LaneState) -> None:
        assert state.next_layer == self.body.end_layer
        state.hidden, _ = self.body.norm(state.hidden, state.residual)
        state.logits = self.model.compute_logits(state.hidden[-1:])
