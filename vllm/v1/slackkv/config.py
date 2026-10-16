# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration gates for the implemented SlackKV execution paths."""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@dataclass(frozen=True)
class SlackKVConfig:
    enabled: bool = False
    implementation_version: int = 1
    runner: str = "mrv1"
    max_elastic_running: int = 1
    mode: str = "fixed_layers"
    offload_pattern: str = "RRRO"
    staging_stripes: int = 2
    prefetch_lead_layers: int = 3
    e_submit_depth: int = 1
    e_prefetch_inflight: int = 2
    cpu_kv_bytes: int = 0
    migration_boundary: str = "decode_token"
    max_demotions_per_model_call: int = 1
    trigger: str = "kv_allocation_blocked"
    policy_source: str = "offline_label"
    copy_backend: str = "dma"
    enable_promotion: bool = False
    enable_event_log: bool = True

    @classmethod
    def from_additional_config(cls, additional_config) -> "SlackKVConfig":
        if (
            not isinstance(additional_config, dict)
            or "slackkv" not in additional_config
        ):
            return cls()
        raw = additional_config["slackkv"]
        if not isinstance(raw, dict):
            raise ValueError("additional_config.slackkv must be an object")
        defaults = cls()
        allowed = {f.name for f in fields(cls)}
        if unknown := raw.keys() - allowed:
            raise ValueError(f"Unknown SlackKV fields: {sorted(unknown)}")
        for key, value in raw.items():
            if type(value) is not type(getattr(defaults, key)):
                raise ValueError(f"SlackKV {key} has invalid type")
        result = cls(**raw)
        for key in sorted(allowed - {
            "enabled", "mode", "cpu_kv_bytes", "enable_event_log",
        }):
            if getattr(result, key) != getattr(defaults, key):
                raise ValueError(f"Unsupported SlackKV {key}: {getattr(result, key)!r}")
        if result.cpu_kv_bytes < 0:
            raise ValueError("SlackKV cpu_kv_bytes must be nonnegative")
        if result.mode not in ("fixed_layers", "resident_dual_lane"):
            raise ValueError(f"Unsupported SlackKV mode: {result.mode!r}")
        return result


def validate_slackkv_config(config: "VllmConfig") -> None:
    slackkv = SlackKVConfig.from_additional_config(config.additional_config)
    if not slackkv.enabled:
        return
    model = config.model_config
    parallel = config.parallel_config
    problems = []
    if config.use_v2_model_runner:
        problems.append("MRV1 required (VLLM_USE_V2_MODEL_RUNNER=0)")
    if any(
        getattr(parallel, key) != 1
        for key in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "decode_context_parallel_size", "prefill_context_parallel_size",
        )
    ):
        problems.append("TP=PP=DP=DCP=PCP=1 required")
    if parallel.enable_dbo:
        problems.append("DBO must be disabled")
    if slackkv.mode == "resident_dual_lane":
        # None is resolved to True later in VllmConfig.__post_init__.
        if config.scheduler_config.async_scheduling is not False:
            problems.append("async scheduling must be explicitly disabled")
    elif config.scheduler_config.async_scheduling:
        problems.append("async scheduling must be disabled")
    if config.speculative_config is not None:
        problems.append("speculation must be disabled")
    if config.cache_config.enable_prefix_caching:
        problems.append("prefix caching must be disabled")
    if (
        config.kv_transfer_config is not None
        or config.cache_config.kv_offloading_size is not None
    ):
        problems.append("KV connector / CPU prefix offload must be disabled")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        problems.append("BF16 KV required")
    if (
        config.attention_config.backend is None
        or config.attention_config.backend.name != "FLASH_ATTN"
        or config.attention_config.flash_attn_version != 2
    ):
        problems.append("explicit FLASH_ATTN / FA2 required")
    if model is None:
        problems.append("Qwen3-8B model required")
    else:
        hf = model.hf_config
        if (
            getattr(hf, "model_type", None) != "qwen3"
            or getattr(hf, "hidden_size", None) != 4096
            or getattr(hf, "num_hidden_layers", None) != 36
        ):
            problems.append("Qwen3-8B architecture required")
        if str(model.dtype) != "torch.bfloat16" or model.quantization is not None:
            problems.append("BF16 unquantized weights required")
        if not model.enforce_eager:
            problems.append("enforce_eager required for mechanism validation")
    if slackkv.mode == "resident_dual_lane":
        from importlib.metadata import entry_points

        from vllm import envs
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        if slackkv.cpu_kv_bytes != 0:
            problems.append("resident_dual_lane requires cpu_kv_bytes=0")
        if config.observability_config.kv_cache_metrics:
            problems.append("resident_dual_lane does not support native KV block metrics")
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            problems.append("resident_dual_lane does not support sync NaN diagnostics")
        if config.device_config.device_type != "cuda":
            problems.append("resident_dual_lane requires a CUDA device")
        if config.scheduler_config.max_num_seqs < 2:
            problems.append("resident_dual_lane requires max_num_seqs>=2")
        if config.scheduler_config.max_num_batched_tokens < 2:
            problems.append("resident_dual_lane requires max_num_batched_tokens>=2")
        if Executor.get_class(config) is not UniProcExecutor:
            problems.append("resident_dual_lane requires UniProcExecutor")
        if config.scheduler_config.get_scheduler_cls() is not Scheduler:
            problems.append("resident_dual_lane requires the native Scheduler")
        if parallel.worker_cls not in ("auto", "vllm.v1.worker.gpu_worker.Worker"):
            problems.append("resident_dual_lane requires the native CUDA Worker")
        if (
            config.offload_config.uva.cpu_offload_gb != 0
            or config.offload_config.prefetch.offload_group_size != 0
        ):
            problems.append("resident_dual_lane requires resident model weights")
        if config.weight_transfer_config is not None:
            problems.append("resident_dual_lane does not support weight transfer")
        if model is not None:
            if model.runner_type != "generate" or model.convert_type != "none":
                problems.append("resident_dual_lane requires a generation model")
            if model.model_impl not in ("auto", "vllm"):
                problems.append("resident_dual_lane requires the native vLLM model")
            if model.enable_sleep_mode:
                problems.append("resident_dual_lane does not support sleep mode")
            if model.enable_return_routed_experts:
                problems.append("resident_dual_lane does not support routed experts")
            if model.logits_processors:
                problems.append("resident_dual_lane does not support custom processors")
        plugins = entry_points(group="vllm.logits_processors")
        if plugins:
            problems.append(
                "resident_dual_lane does not support logits processor plugins: "
                + ", ".join(sorted(plugin.name for plugin in plugins))
            )
    if problems:
        raise ValueError("Incompatible SlackKV configuration: " + "; ".join(problems))
    if slackkv.mode == "resident_dual_lane":
        return
    raise ValueError(
        "SlackKV S0 only: elastic execution is not implemented; "
        "set additional_config.slackkv.enabled=false for native execution"
    )


def resident_mode(config: "VllmConfig") -> bool:
    slackkv = SlackKVConfig.from_additional_config(config.additional_config)
    return slackkv.enabled and slackkv.mode == "resident_dual_lane"


def validate_resident_processors(runner: "GPUModelRunner") -> None:
    """Reject loaded processors outside the plain greedy resident scope."""
    if not resident_mode(runner.vllm_config):
        return

    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinPLogitsProcessor,
        MinTokensLogitsProcessor,
    )

    builtin_types = {
        MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor,
    }
    unsupported = sorted({
        f"{type(processor).__module__}.{type(processor).__qualname__}"
        for processor in runner.input_batch.logitsprocs.all
        if type(processor) not in builtin_types
    })
    if unsupported:
        raise ValueError(
            "resident_dual_lane loaded unsupported logits processors: "
            + ", ".join(unsupported)
        )
