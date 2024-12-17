import logging
from typing import Dict, List

import torch

from .model import LLMModel


class SVDProcessor:
    def __init__(self, model: LLMModel, config):
        self.model = model
        self.config = config

        # Mapping dictionary for linear names to their corresponding LoRA attributes
        self._mapping_dict = {
            "q_proj": "wq_",
            "k_proj": "wk_",
            "v_proj": "wv_",
            "o_proj": "wo_",
            "gate_proj": "w1_",
            "down_proj": "w2_",
            "up_proj": "w3_",
        }

        # Possible linear types
        self._attn_linears = ["wq_", "wk_", "wv_", "wo_"]
        self._mlp_linears = ["w1_", "w2_", "w3_"]

    def process(self) -> List[List[Dict[str, List[float]]]]:
        target_linears_list = self._mapping(self._keys_extraction(self.config))

        if self.config.moe_flag:
            return self._moe_weight_traverse(target_linears_list)
        else:
            return self._lora_weight_traverse(target_linears_list)

    def _keys_extraction(self, config) -> List[Dict[str, List[str]]]:
        name = config.adapter_name
        target_modules: Dict[str, bool] = config.target_modules
        true_keys = [key for key, value in target_modules.items() if value]
        return [{name: true_keys}]

    def _mapping(
        self, keys_list: List[Dict[str, List[str]]]
    ) -> List[Dict[str, List[str]]]:
        mapped_list = [
            {
                name: [
                    self._mapping_dict[key] for key in keys if key in self._mapping_dict
                ]
            }
            for item in keys_list
            for name, keys in item.items()
        ]
        return mapped_list

    def _moe_weight_calculate(
        self, loading: List[float], lora_weights: List[torch.Tensor]
    ) -> torch.Tensor:
        return sum(weight * tensor for weight, tensor in zip(loading, lora_weights))

    def _perform_svd_analysis(
        self,
        p_weight: torch.Tensor,
        f_weight: torch.Tensor,
        n: int = 9,
        device: str = "cuda:0",
    ) -> List[float]:
        p_weight = p_weight.to(torch.float32).to(device)
        f_weight = f_weight.to(torch.float32).to(device)

        # SVD decomposition
        p_u, _, _ = torch.linalg.svd(p_weight, full_matrices=False)
        f_u, _, _ = torch.linalg.svd(f_weight, full_matrices=False)

        # Pick top-n singular vectors
        n_min = min(n, p_u.shape[1], f_u.shape[1])
        p_top_n = p_u[:, :n_min]
        f_top_n = f_u[:, :n_min]

        # Compute cosine similarities
        similarity = torch.mm(p_top_n.T, f_top_n)
        p_norms = torch.norm(p_top_n.T, dim=1, keepdim=True)
        f_norms = torch.norm(f_top_n, dim=0, keepdim=True)
        similarity = similarity / (p_norms * f_norms)

        return similarity.diagonal().tolist()

    def _process_lora_block(
        self,
        layer_idx: int,
        linear: str,
        adapter_name: str,
        loras_dict: Dict[str, torch.nn.Module],
        moe_profile: torch.Tensor = None,
    ) -> Dict[str, List[float]]:
        if moe_profile is not None:
            # MoE scenario
            tuned_expert_value_lists = []

            for expert_adapter in loras_dict.values():
                p_weight = expert_adapter.base_layer_.weight
                lora_a_weight = expert_adapter.lora_a_.weight
                lora_b_weight = expert_adapter.lora_b_.weight
                t_weight = lora_b_weight @ lora_a_weight + p_weight
                tuned_expert_value_lists.append(t_weight)

            tuned_weights = self._moe_weight_calculate(
                moe_profile, tuned_expert_value_lists
            )
            linear_key = linear.rstrip("_")
            logging.info(
                f"Layer [{layer_idx}], linear [{linear_key}]: performing MoE SVD analysis..."
            )
            return {linear_key: self._perform_svd_analysis(p_weight, tuned_weights)}

        else:
            # Normal LoRA scenario
            adapter = loras_dict.get(adapter_name, None)
            if adapter is not None:
                p_weight = adapter.base_layer_.weight
                lora_a_weight = adapter.lora_a_.weight
                lora_b_weight = adapter.lora_b_.weight
                t_weight = lora_b_weight @ lora_a_weight + p_weight

                linear_key = linear.rstrip("_")
                logging.info(
                    f"Layer [{layer_idx}], linear [{linear_key}]: performing SVD analysis..."
                )
                return {linear_key: self._perform_svd_analysis(p_weight, t_weight)}

        return {}

    def _lora_weight_traverse(
        self, target_linears_list: List[Dict[str, List[str]]]
    ) -> List[List[Dict[str, List[float]]]]:
        final_result = []

        for layer_idx, layer in enumerate(self.model.model_.layers_):
            layer_result = []
            for item in target_linears_list:
                for adapter_name, linear_lst in item.items():
                    for linear in linear_lst:
                        if linear in self._attn_linears:
                            # For attention linears
                            loras_dict = getattr(layer.self_attn_, linear).loras_
                            analysis_result = self._process_lora_block(
                                layer_idx,
                                linear,
                                adapter_name,
                                loras_dict,
                                moe_profile=None,
                            )
                            if analysis_result:
                                layer_result.append(analysis_result)

                        elif linear in self._mlp_linears:
                            # For MLP linears
                            loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            analysis_result = self._process_lora_block(
                                layer_idx,
                                linear,
                                adapter_name,
                                loras_dict,
                                moe_profile=None,
                            )
                            if analysis_result:
                                layer_result.append(analysis_result)
                        else:
                            raise ValueError(f"Invalid linear name: {linear}")

            final_result.append(layer_result)

        return final_result

    def _moe_weight_traverse(
        self, target_linears_list: List[Dict[str, List[str]]]
    ) -> List[List[Dict[str, List[float]]]]:
        final_result = []

        for layer_idx, layer in enumerate(self.model.model_.layers_):
            layer_result = []
            for item in target_linears_list:
                for adapter_name, linear_lst in item.items():
                    for linear in linear_lst:
                        if linear in self._attn_linears:
                            # MoE in attention linears
                            block = getattr(layer.self_attn_, linear)
                            loras_dict = block.loras_
                            moes_dict = getattr(block, "moes_", {})

                            if adapter_name in moes_dict and block.moes_:
                                profile_matrix = moes_dict[adapter_name].profiler_
                                analysis_result = self._process_lora_block(
                                    layer_idx,
                                    linear,
                                    adapter_name,
                                    loras_dict,
                                    moe_profile=profile_matrix,
                                )
                                if analysis_result:
                                    layer_result.append(analysis_result)
                            else:
                                # Normal scenario if no MoE
                                analysis_result = self._process_lora_block(
                                    layer_idx,
                                    linear,
                                    adapter_name,
                                    loras_dict,
                                    moe_profile=None,
                                )
                                if analysis_result:
                                    layer_result.append(analysis_result)

                        elif linear in self._mlp_linears:
                            # MoE in MLP linears
                            block = getattr(layer.mlp_.mlp_, linear)
                            loras_dict = block.loras_
                            moes_dict = getattr(layer.mlp_, "moes_", {})

                            if adapter_name in moes_dict and layer.mlp_.moes_:
                                profile_matrix = moes_dict[adapter_name].profiler_
                                analysis_result = self._process_lora_block(
                                    layer_idx,
                                    linear,
                                    adapter_name,
                                    loras_dict,
                                    moe_profile=profile_matrix,
                                )
                                if analysis_result:
                                    layer_result.append(analysis_result)
                            else:
                                # Normal scenario if no MoE
                                analysis_result = self._process_lora_block(
                                    layer_idx,
                                    linear,
                                    adapter_name,
                                    loras_dict,
                                    moe_profile=None,
                                )
                                if analysis_result:
                                    layer_result.append(analysis_result)

                        else:
                            raise ValueError(f"Invalid linear name: {linear}")

            final_result.append(layer_result)

        return final_result
