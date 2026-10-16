# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leased E token inputs; LayerMetadataBuffers owns all KV indexing."""

import numpy as np
import torch

class ElasticInputSlot:
    def __init__(self, device, max_tokens, max_context, block_size, kernel_size):
        assert block_size % kernel_size == 0
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.ratio = block_size // kernel_size
        self.max_tokens = max_tokens
        self.max_context = max_context
        self.busy = False
        shapes = {
            "ids": ((max_tokens,), torch.int64),
            "positions": ((max_tokens,), torch.int64),
            "lengths": ((1,), torch.int32),
            "starts": ((2,), torch.int32),
        }
        self.host = {
            name: torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            for name, (shape, dtype) in shapes.items()
        }
        self.device = {
            name: torch.empty_like(tensor, device=device)
            for name, tensor in self.host.items()
        }

    def acquire_tokens(self, ids, computed):
        assert not self.busy
        n = len(ids)
        length = computed + n
        if not 0 < n <= self.max_tokens or length > self.max_context:
            raise ValueError("E input exceeds its configured input/context capacity")
        pos = np.arange(computed, length, dtype=np.int64)
        self.host["ids"].numpy()[:n] = ids
        self.host["positions"].numpy()[:n] = pos
        self.host["lengths"].numpy()[0] = length
        self.host["starts"].numpy()[:] = [0, n]
        views = {
            "ids": self.device["ids"][:n],
            "positions": self.device["positions"][:n],
            "lengths": self.device["lengths"],
            "starts": self.device["starts"],
        }
        for name, view in views.items():
            source = self.host[name]
            if name in ("ids", "positions"):
                source = source[:n]
            view.copy_(source, non_blocking=True)
        self.busy = True
        return views

    def release(self):
        assert self.busy
        self.busy = False

    def snapshot(self):
        return {
            "busy": self.busy,
            "allocation_block_size": self.block_size,
            "kernel_block_size": self.kernel_size,
            "host_bytes": sum(t.numel() * t.element_size()
                              for t in self.host.values()),
            "device_bytes": sum(t.numel() * t.element_size()
                                for t in self.device.values()),
            "host_addresses": {k: t.data_ptr() for k, t in self.host.items()},
            "device_addresses": {k: t.data_ptr() for k, t in self.device.items()},
        }
