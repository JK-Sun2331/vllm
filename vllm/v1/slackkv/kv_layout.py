# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable, homogeneous LayerPage storage with native FA2 page contents."""

from dataclasses import dataclass
from math import prod

import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache


@dataclass(frozen=True)
class LayerPageLayout:
    layer_names: tuple[str, ...]
    spec: AttentionSpec
    backend: type
    native_num_blocks: int
    kernel_size: int
    cache_dtype: str

    @classmethod
    def from_config(cls, config, attn_groups, kernel_sizes, cache_dtype):
        if len(config.kv_cache_groups) != 1 or len(attn_groups) != 1:
            raise ValueError("LayerPage requires one homogeneous KV group")
        if len(attn_groups[0]) != 1 or len(kernel_sizes) != 1:
            raise ValueError("LayerPage requires one homogeneous attention group")
        group = attn_groups[0][0]
        spec = group.kv_cache_spec
        if group.backend is not FlashAttentionBackend:
            raise ValueError("LayerPage S3 requires the native FlashAttention backend")
        if not isinstance(spec, AttentionSpec) or spec.dtype is not torch.bfloat16:
            raise ValueError("LayerPage S3 requires BF16 attention KV")
        if (spec.kv_quant_mode != KVQuantMode.NONE
                or spec.storage_block_size != spec.block_size
                or spec.page_size_padded is not None):
            raise ValueError("LayerPage S3 does not support compressed or padded KV")
        if any(t.block_stride for t in config.kv_cache_tensors):
            raise ValueError("LayerPage S3 cannot reinterpret packed native KV pools")
        names = tuple(config.kv_cache_groups[0].layer_names)
        if set(names) != set(group.layer_names) or len(names) != len(set(names)):
            raise ValueError("LayerPage layer ownership differs from the KV group")
        if config.num_blocks <= 0 or spec.block_size % kernel_sizes[0]:
            raise ValueError("Invalid LayerPage capacity or allocation/kernel ratio")
        result = cls(names, spec, group.backend, config.num_blocks,
                     kernel_sizes[0], cache_dtype)
        if sum(t.size for t in config.kv_cache_tensors) != result.arena_bytes:
            raise ValueError("LayerPage arena does not equal the native KV budget")
        if result.num_pages * result.ratio > 2**31:
            raise ValueError("LayerPage kernel page IDs exceed int32 metadata")
        return result

    @property
    def block_size(self):
        return self.spec.block_size

    @property
    def ratio(self):
        return self.block_size // self.kernel_size

    @property
    def num_pages(self):
        return len(self.layer_names) * self.native_num_blocks

    @property
    def page_bytes(self):
        return self.spec.page_size_bytes

    @property
    def arena_bytes(self):
        return self.num_pages * self.page_bytes

    def allocate(self, device):
        backing = torch.zeros(self.arena_bytes, dtype=torch.int8, device=device)
        shape = self.backend.get_kv_cache_shape(
            self.num_pages * self.ratio, self.kernel_size,
            self.spec.num_kv_heads, self.spec.head_size,
            cache_dtype_str=self.cache_dtype,
        )
        order = self.backend.get_kv_cache_stride_order()
        if order[0] != 0 or prod(shape) * 2 != self.arena_bytes:
            raise ValueError("LayerPage requires a native blocks-first BF16 view")
        view = _reshape_attention_kv_cache(
            backing, self.spec, shape, order, shape[0], packing=None,
        )
        return backing, {name: view for name in self.layer_names}

    def manifest(self, cache):
        return {
            "kv_layout": "layer_page_arena", "layers": len(self.layer_names),
            "native_equivalent_blocks": self.native_num_blocks,
            "layer_pages": self.num_pages, "system_null_pages": 1,
            "allocation_block_size": self.block_size,
            "kernel_block_size": self.kernel_size, "kernel_ratio": self.ratio,
            "page_bytes": self.page_bytes, "arena_bytes": self.arena_bytes,
            "unique_storage_bytes": cache.untyped_storage().nbytes(),
            "shape": list(cache.shape), "stride": list(cache.stride()),
            "storage_offset": cache.storage_offset(),
            "kv_heads": self.spec.num_kv_heads, "head_size": self.spec.head_size,
            "dtype": str(cache.dtype),
        }
