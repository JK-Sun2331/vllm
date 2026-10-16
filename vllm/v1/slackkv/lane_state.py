# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned inputs and intermediate state for a resident segmented step."""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.request import Request
from vllm.v1.sample.metadata import SamplingMetadata


@dataclass(frozen=True)
class LaneInputs:
    members: tuple[Request, ...]
    step_id: int
    owner_epoch: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    attn_metadata: dict[str, FlashAttentionMetadata]
    slot_mapping: dict[str, torch.Tensor]
    sampling_metadata: SamplingMetadata
    # Keep pinned H2D sources owned until this lane's completion/drain event.
    cpu_sources: tuple[torch.Tensor, ...] = ()


@dataclass
class LaneState:
    inputs: LaneInputs
    next_layer: int = 0
    hidden: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    sampled_token: int | None = None
    completion: torch.cuda.Event | None = None
    sampled_gpu: torch.Tensor | None = None
    sampled_cpu: torch.Tensor | None = None
