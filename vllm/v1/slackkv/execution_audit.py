# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in logical KV/numerical snapshots, excluded from resource/performance runs."""

import torch


class ExecutionAudit:
    def __init__(self, runner):
        self.runner = runner
        self.records = []
        self.elastic_records = {}
        self.active = None
        self.handles = []
        self.kv_update_originals = []
        self.layer_names = tuple(runner.kv_cache_config.kv_cache_groups[0].layer_names)
        context = runner.vllm_config.compilation_config.static_forward_context
        for name in self.layer_names:
            impl = context[name].impl
            original = impl.do_kv_cache_update

            def update(*args, original=original, name=name, **kwargs):
                assert self.active is not None
                counts = self.active["kv_update_counts"]
                counts[name] = counts.get(name, 0) + 1
                return original(*args, **kwargs)

            self.kv_update_originals.append((impl, original))
            impl.do_kv_cache_update = update
        for index, layer in enumerate(runner.model.model.layers):
            def capture(module, args, output, index=index):
                assert self.active is not None
                self.active["layers"][index] = tuple(t.clone() for t in output)
            self.handles.append(layer.register_forward_hook(capture))

    @staticmethod
    def label(params, request_id):
        return (params.extra_args or {}).get("slackkv_label", request_id)

    def _begin(self, lane, members):
        record = {"lane": lane, "members": members, "layers": {},
                  "kv": {}, "cpu_sources": [], "kv_update_counts": {}}
        self.records.append(record)
        self.active = record
        return record

    def begin_native(self, output):
        members = []
        prepared = getattr(self.runner, "slackkv_prepared_layers", None)
        for index, request_id in enumerate(self.runner.input_batch.req_ids):
            request = self.runner.requests[request_id]
            member = {
                "label": self.label(request.sampling_params, request_id),
                "computed": int(self.runner.input_batch.num_computed_tokens_cpu[index]),
                "q": output.num_scheduled_tokens[request_id],
                "blocks": list(request.block_ids[0]),
            }
            if prepared is not None:
                self._capture_indices(member, prepared, index)
            members.append(member)
        self._begin("P", members)

    def _capture_indices(self, member, prepared, row):
        valid = member["computed"] + member["q"]
        kernel_size = self.runner._kernel_block_sizes[0]
        width = (valid + kernel_size - 1) // kernel_size
        # Own these indices: the lane's persistent table will change next step.
        member["layer_indices"] = {
            name: prepared.tables[name][row, :width].to(torch.int64)
            for name in self.layer_names
        }

    def begin_elastic(self, plan, prepared):
        source = plan.elastic
        member = {
            "label": self.label(source.sampling_params, source.ref.request_id),
            "computed": source.computed, "q": len(source.input_ids),
            "blocks": [],
        }
        self._capture_indices(member, prepared, 0)
        record = self._begin("E", [member])
        self.elastic_records[plan.step_id] = record

    def resume_elastic(self, plan):
        self.active = self.elastic_records[plan.step_id]

    def finish(self, hidden, logits):
        record = self.active
        assert record is not None and len(record["layers"]) == 36
        assert "done" not in record
        assert record["kv_update_counts"] == {name: 1 for name in self.layer_names}
        record["hidden"] = hidden.clone()
        record["logits"] = logits.clone()
        config = self.runner.kv_cache_config
        group = config.kv_cache_groups[0]
        block_size = group.kv_cache_spec.block_size
        kernel_size = self.runner._kernel_block_sizes[0]
        ratio = block_size // kernel_size
        layers = self.runner.vllm_config.compilation_config.static_forward_context
        for member in record["members"]:
            indices = None
            if "layer_indices" not in member:
                kernel_ids = [b * ratio + j for b in member["blocks"]
                              for j in range(ratio)]
                source = torch.tensor(kernel_ids, dtype=torch.int64, pin_memory=True)
                indices = source.to(device=self.runner.device, non_blocking=True)
                record["cpu_sources"].append(source)
            valid = member["computed"] + member["q"]
            values = []
            for name in group.layer_names:
                cache = layers[name].kv_cache
                layer_indices = (member["layer_indices"][name]
                                 if "layer_indices" in member else indices)
                # FA2 NHD packed cache. Compare logical valid history, not padding.
                values.append(cache.index_select(0, layer_indices).permute(0, 2, 1, 3)
                              .reshape(-1, cache.shape[1], cache.shape[-1])[:valid].clone())
            record["kv"][member["label"]] = values
        record["done"] = torch.cuda.Event()
        record["done"].record(torch.cuda.current_stream())
        self.active = None

    def collect(self):
        result = []
        for record in self.records:
            assert record["done"].query(), "audit collection requires drained engine"
            result.append({
                "members": [{k: v for k, v in m.items()
                             if k not in ("blocks", "layer_indices")}
                            for m in record["members"]],
                "layers": {i: tuple(t.cpu() for t in pair)
                           for i, pair in record["layers"].items()},
                "hidden": record["hidden"].cpu(), "logits": record["logits"].cpu(),
                "kv": {label: [t.cpu() for t in values]
                       for label, values in record["kv"].items()},
                "kv_update_counts": dict(record["kv_update_counts"]),
            })
        self.records.clear()
        self.elastic_records.clear()
        self.active = None
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for impl, original in self.kv_update_originals:
            impl.do_kv_cache_update = original
        self.kv_update_originals.clear()
        self.records.clear()
        self.elastic_records.clear()
        self.active = None
