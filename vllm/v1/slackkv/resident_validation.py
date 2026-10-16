# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GPU S1 reference/segment comparison in an otherwise idle MRV1 worker."""

import torch

from vllm.forward_context import is_forward_context_available, set_forward_context
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.slackkv.elastic_runner import ElasticRunner
from vllm.v1.slackkv.lane_state import LaneInputs
from vllm.v1.slackkv.model_adapter import Qwen3Adapter
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    use_workspace_lane,
)


def sampling_state(request: Request) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None, all_greedy=True, all_random=False,
        top_p=None, top_k=None, generators={}, max_num_logprobs=None,
        no_penalties=True, prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device="cuda"),
        presence_penalties=torch.zeros(1, device="cuda"),
        repetition_penalties=torch.ones(1, device="cuda"),
        output_token_ids=[list(request.output_token_ids)],
        allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def make_inputs(request, page_ids, layer_names, step_id):
    computed = request.num_computed_tokens
    tokens = (request.prompt_token_ids if computed == 0
              else [request.output_token_ids[-1]])
    ids = torch.tensor(tokens, dtype=torch.int64, device="cuda")
    positions = torch.arange(computed, computed + len(tokens), device="cuda")
    seq_lens = torch.tensor([computed + len(tokens)], dtype=torch.int32,
                            device="cuda")
    table = torch.tensor([page_ids], dtype=torch.int32, device="cuda")
    slots = table[0].long()[positions // 16] * 16 + positions % 16
    metadata = FlashAttentionMetadata(
        num_actual_tokens=len(tokens), max_query_len=len(tokens),
        query_start_loc=torch.tensor([0, len(tokens)], dtype=torch.int32,
                                     device="cuda"),
        max_seq_len=computed + len(tokens), seq_lens=seq_lens,
        block_table=table, slot_mapping=slots, use_cascade=False,
        common_prefix_len=0, cu_prefix_query_lens=None,
        prefix_kv_lens=None, suffix_kv_lens=None,
    )
    return LaneInputs(
        members=(request,), step_id=step_id, owner_epoch=0,
        input_ids=ids, positions=positions, seq_lens=seq_lens,
        attn_metadata={name: metadata for name in layer_names},
        slot_mapping={name: slots for name in layer_names},
        sampling_metadata=sampling_state(request),
    )


def native_forward(runner, inputs, capture):
    handles = []
    if capture is not None:
        for index, layer in enumerate(runner.model.model.layers):
            def record(module, args, output, index=index):
                capture[index] = tuple(t.clone() for t in output)
            handles.append(layer.register_forward_hook(record))
    try:
        with (
            use_workspace_lane(0),
            set_forward_context(
                inputs.attn_metadata, runner.vllm_config,
                num_tokens=inputs.input_ids.numel(),
                slot_mapping=inputs.slot_mapping, skip_compiled=True,
            ),
        ):
            hidden = runner._model_forward(
                input_ids=inputs.input_ids, positions=inputs.positions)
            logits = runner.model.compute_logits(hidden[-1:])
    finally:
        for handle in handles:
            handle.remove()
    return hidden, logits


class S1ValidationWorker:
    @torch.inference_mode()
    def slackkv_validate_resident(self, cases: list[dict]) -> dict:
        runner = self.model_runner
        assert type(runner).__module__ == "vllm.v1.worker.gpu_model_runner"
        assert runner.input_batch.num_reqs == 0
        assert runner.execute_model_state is None
        batch_identity = id(runner.input_batch)
        torch.cuda.synchronize()
        # This named validation entry runs before any requests are submitted.
        # Keep production workspace initialization unchanged until S2 integration.
        init_workspace_manager(self.device, num_lanes=2)
        workspace_views = []
        for lane in (0, 1):
            with use_workspace_lane(lane):
                workspace_views.append(current_workspace_manager().get_simultaneous(
                    ((1024,), torch.float32))[0])
        assert workspace_views[0].data_ptr() != workspace_views[1].data_ptr()
        adapter = Qwen3Adapter(runner.model)
        elastic = ElasticRunner(adapter, self.vllm_config)
        native_sampler = Sampler()
        layers = self.vllm_config.compilation_config.static_forward_context
        caches = {name: layer.kv_cache for name, layer in layers.items()
                  if hasattr(layer, "kv_cache")}
        assert len(caches) == 36
        assert all(tuple(t.shape[1:]) == (8, 16, 256) for t in caches.values())
        weights_before = {name: p.untyped_storage().data_ptr()
                          for name, p in runner.model.named_parameters()}
        assert weights_before == {
            name: p.untyped_storage().data_ptr()
            for name, p in adapter.model.named_parameters()}
        result = {"status": "running", "rtol": 0, "atol": 0,
                  "weight_tensors": len(weights_before), "cases": []}
        for case in cases:
            row = {"name": case["name"], "steps": [], "comparisons": 0,
                   "max_abs_error": 0.0}
            result["cases"].append(row)

            def equal(left, right, label):
                error = float((left.float() - right.float()).abs().max().item())
                row["max_abs_error"] = max(row["max_abs_error"], error)
                row["comparisons"] += 1
                assert torch.equal(left, right), f"{case['name']} {label}: {error}"

            prompt = case["prompt_ids"]
            count = (len(prompt) + case["max_tokens"] + 15) // 16
            p_pages = list(range(1, 1 + count))
            e_pages = list(range(1 + count, 1 + 2 * count))
            assert max(e_pages) < runner.kv_cache_config.num_blocks
            for tensor in caches.values():
                tensor[p_pages + e_pages] = 0
            requests = []
            for lane in ("P", "E"):
                params = SamplingParams(temperature=0,
                                        max_tokens=case["max_tokens"])
                params.update_from_generation_config({}, case.get("eos_token_id"))
                requests.append(Request(lane, prompt.copy(), params, None))
            p_request, e_request = requests
            for step in range(case["max_tokens"]):
                p_inputs = make_inputs(p_request, p_pages, caches, step)
                e_inputs = make_inputs(e_request, e_pages, caches, step)
                assert p_inputs.input_ids.data_ptr() != e_inputs.input_ids.data_ptr()
                assert p_inputs.seq_lens.data_ptr() != e_inputs.seq_lens.data_ptr()
                assert p_inputs.attn_metadata[next(iter(caches))] is not \
                    e_inputs.attn_metadata[next(iter(caches))]
                before_context = is_forward_context_available()
                p_layers = {}
                p_hidden, p_logits = native_forward(runner, p_inputs, p_layers)
                assert is_forward_context_available() == before_context
                p_token = int(native_sampler(
                    p_logits.clone(), p_inputs.sampling_metadata
                ).sampled_token_ids.item())
                p_request.num_computed_tokens += p_inputs.input_ids.numel()
                p_request.append_output_token_ids(p_token)
                check_stop(p_request, self.vllm_config.model_config.max_model_len)
                e_layers = {}
                if step == 0:
                    # Both histories are established by original prefill.
                    e_hidden, e_logits = native_forward(runner, e_inputs, e_layers)
                    e_token = int(elastic.sampler(
                        e_logits.clone(), e_inputs.sampling_metadata
                    ).sampled_token_ids.item())
                    e_request.num_computed_tokens += e_inputs.input_ids.numel()
                    e_request.append_output_token_ids(e_token)
                    check_stop(e_request, self.vllm_config.model_config.max_model_len)
                else:
                    def observe(index, hidden, residual):
                        e_layers[index] = (hidden.clone(), residual.clone())
                    elastic.begin_step(e_inputs)
                    for end in case["cuts"]:
                        elastic.advance(end, observe)
                        assert is_forward_context_available() == before_context
                    state = elastic.finish_step()
                    e_hidden, e_logits, e_token = (
                        state.hidden, state.logits, state.sampled_token)
                for index in range(36):
                    for kind, left, right in zip(
                        ("hidden", "residual"), p_layers[index], e_layers[index]
                    ):
                        equal(left, right, f"step {step} layer {index} {kind}")
                equal(p_hidden, e_hidden, "final hidden")
                equal(p_logits, e_logits, "logits")
                valid = p_request.num_computed_tokens
                for name, cache in caches.items():
                    views = [cache[pages].permute(0, 2, 1, 3).reshape(-1, 8, 256)
                             [:valid] for pages in (p_pages, e_pages)]
                    equal(*views, f"{name} valid KV")
                assert p_token == e_token
                assert p_request.num_computed_tokens == e_request.num_computed_tokens
                assert p_request.num_tokens == p_request.num_computed_tokens + 1
                assert p_request.status == e_request.status
                assert id(runner.input_batch) == batch_identity
                assert runner.execute_model_state is None
                row["steps"].append({"step": step, "token": p_token,
                                     "valid_kv_tokens": valid,
                                     "known_tokens": p_request.num_tokens})
                if p_request.status in (RequestStatus.FINISHED_STOPPED,
                                        RequestStatus.FINISHED_LENGTH_CAPPED):
                    break
            row["stop_status"] = p_request.status.name
            if case.get("eos_token_id") is not None:
                assert p_request.output_token_ids[-1] == case["eos_token_id"]
                assert p_request.status == RequestStatus.FINISHED_STOPPED
            else:
                assert p_request.num_output_tokens == case["max_tokens"]
                assert p_request.status == RequestStatus.FINISHED_LENGTH_CAPPED
            for tensor in caches.values():
                tensor[p_pages + e_pages] = 0
        assert weights_before == {
            name: p.untyped_storage().data_ptr()
            for name, p in runner.model.named_parameters()}
        torch.cuda.synchronize()
        result.update(status="passed", input_batch_preserved=True,
                      weights_shared=True, workspace_lanes=2,
                      allocated_gpu_bytes=torch.cuda.memory_allocated())
        return result
