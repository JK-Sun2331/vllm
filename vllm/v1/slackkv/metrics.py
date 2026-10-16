# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in S0 worker inspection via the native worker-extension/RPC interface."""

import inspect
import sys

import torch


class SlackKVAuditWorker:
    def slackkv_runtime_manifest(self) -> dict:
        from vllm import _custom_ops as ops
        from vllm.model_executor.models.qwen2 import Qwen2Model
        from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        runner = self.model_runner
        assert type(runner).__module__ == "vllm.v1.worker.gpu_model_runner"
        config = self.vllm_config
        layers = config.compilation_config.static_forward_context
        caches = []
        for name, layer in layers.items():
            if not hasattr(layer, "kv_cache"):
                continue
            tensor = layer.kv_cache
            assert isinstance(tensor, torch.Tensor)
            caches.append({
                "layer": name,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "page_bytes": tensor[0].numel() * tensor.element_size(),
                "backend": layer.attn_backend.get_name(),
                "fa_version": layer.impl.vllm_flash_attn_version,
            })
        assert len(caches) == 36
        assert all(
            c["backend"] == "FLASH_ATTN" and c["fa_version"] == 2 for c in caches
        )
        # An explicit stream exercises the compiled DMA operator in both directions.
        source = torch.arange(4096, dtype=torch.int64).pin_memory()
        target = torch.empty_like(source, device="cuda")
        restored = torch.empty_like(source, pin_memory=True)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for src, dst in ((source, target), (target, restored)):
                ops.swap_blocks_batch(
                    torch.tensor([src.data_ptr()], dtype=torch.int64),
                    torch.tensor([dst.data_ptr()], dtype=torch.int64),
                    torch.tensor(
                        [source.numel() * source.element_size()], dtype=torch.int64
                    ),
                )
        stream.synchronize()
        assert torch.equal(source, restored)
        assert issubclass(Qwen3Model, Qwen2Model)
        result = {
            "runner": f"{type(runner).__module__}.{type(runner).__name__}",
            "model": f"{type(runner.model).__module__}.{type(runner.model).__name__}",
            "qwen2_forward": str(inspect.signature(Qwen2Model.forward)),
            "qwen3_decoder": str(inspect.signature(Qwen3DecoderLayer.forward)),
            "qwen3_model_bases": [c.__name__ for c in Qwen3Model.__mro__],
            "loaded_extensions": {
                name: module.__file__ for name, module in sys.modules.items()
                if name.startswith("vllm")
                and str(getattr(module, "__file__", "")).endswith(".so")
            },
            "kv_layout": get_kv_cache_layout(),
            "kv_tensors": caches,
            "dma_schema": str(torch.ops._C_cache_ops.swap_blocks_batch.default._schema),
            "dma_roundtrip_bytes": source.numel() * source.element_size(),
            "weights_bytes": int(runner.model_memory_usage),
            "available_kv_bytes": int(self.available_kv_cache_memory_bytes),
            "allocated_kv_bytes": sum(t.size for t in
                                      runner.kv_cache_config.kv_cache_tensors),
            "num_blocks": runner.kv_cache_config.num_blocks,
            "cpu_kv_bytes": 0,
            "cache_config": str(config.cache_config),
            "scheduler_config": str(config.scheduler_config),
        }
        if runner.slackkv_layer_page_enabled:
            cache = layers[runner.slackkv_kv_layout.layer_names[0]].kv_cache
            result["layer_page_arena"] = runner.slackkv_kv_layout.manifest(cache)
            result["layer_metadata"] = {
                lane: buffers.manifest()
                for lane, buffers in self.slackkv_runtime.layer_buffers.items()
            }
            result["dummy_metadata_bytes"] = (
                runner.slackkv_dummy_table.numel()
                * runner.slackkv_dummy_table.element_size()
                + runner.slackkv_dummy_slots.numel()
                * runner.slackkv_dummy_slots.element_size())
        return result
