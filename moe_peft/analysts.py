import heapq
import logging
from collections import Counter
from typing import Dict, List, Optional, Union

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
        self._single_expert_mood = True

    def process(
        self, all_result: Optional[bool] = False
    ) -> Union[List[List[Dict[str, List[float]]]], List[Dict[str, List[float]]]]:
        target_linears_list = self._mapping(self._keys_extraction(self.config))

        # Return the total svd analysis results
        if self.config.moe_flag and all_result:
            return self._moe_weight_traverse(
                target_linears_list, self._single_expert_mood
            )
        elif all_result:
            return self._lora_weight_traverse(target_linears_list)

        # Return svd analysis features
        elif self.config.moe_flag and not all_result:
            return self._analyze_svd_data(
                self._moe_weight_traverse(
                    target_linears_list, self._single_expert_mood
                ),
                self.config,
            )
        else:
            return self._analyze_svd_data(
                self._lora_weight_traverse(target_linears_list), self.config
            )

    def _keys_extraction(self, config) -> List[Dict[str, List[str]]]:
        name = config.adapter_name
        target_modules: Dict[str, bool] = config.target_modules
        true_keys = [key for key, value in target_modules.items() if value]
        return [{name: true_keys}]

    def _analyze_svd_data(
        self, data: List[List[Dict[str, List[float]]]], config
    ) -> Dict:
        avg_similarities = []
        for layer_idx, layer in enumerate(data):
            for linear_data in layer:
                for linear_name, similarities in linear_data.items():
                    avg_similarity = sum(abs(sim) for sim in similarities) / len(
                        similarities
                    )
                    avg_similarities.append((avg_similarity, layer_idx, linear_name))

        lowest_avg_similarities = heapq.nsmallest(
            15, avg_similarities, key=lambda x: x[0]
        )

        all_similarities = []
        for layer_idx, layer in enumerate(data):
            for linear_data in layer:
                for linear_name, similarities in linear_data.items():
                    for vector_idx, similarity in enumerate(similarities):
                        all_similarities.append(
                            (abs(similarity), layer_idx, linear_name, vector_idx)
                        )

        lowest_similarities = heapq.nsmallest(15, all_similarities, key=lambda x: x[0])

        avg_layer_counter = Counter([entry[1] for entry in lowest_avg_similarities])
        avg_linear_counter = Counter([entry[2] for entry in lowest_avg_similarities])

        avg_statistics = {
            "layer_distribution": {
                k: v
                for k, v in sorted(
                    {
                        layer_idx: count / len(lowest_avg_similarities)
                        for layer_idx, count in avg_layer_counter.items()
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "linear_name_distribution": {
                k: v
                for k, v in sorted(
                    {
                        linear_name: count / len(lowest_avg_similarities)
                        for linear_name, count in avg_linear_counter.items()
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }

        individual_layer_counter = Counter([entry[1] for entry in lowest_similarities])
        individual_linear_counter = Counter([entry[2] for entry in lowest_similarities])
        individual_vector_counter = Counter([entry[3] for entry in lowest_similarities])

        individual_statistics = {
            "layer_distribution": {
                k: v
                for k, v in sorted(
                    {
                        layer_idx: count / len(lowest_similarities)
                        for layer_idx, count in individual_layer_counter.items()
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "linear_name_distribution": {
                k: v
                for k, v in sorted(
                    {
                        linear_name: count / len(lowest_similarities)
                        for linear_name, count in individual_linear_counter.items()
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "vector_index_distribution": {
                k: v
                for k, v in sorted(
                    {
                        vector_idx: count / len(lowest_similarities)
                        for vector_idx, count in individual_vector_counter.items()
                    }.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
        }

        result = {
            "adapter_name": config.adapter_name,
            "lowest_avg_similarities": [
                {
                    "avg_similarity": avg,
                    "layer_index": layer_idx,
                    "linear_name": linear_name,
                }
                for avg, layer_idx, linear_name in lowest_avg_similarities
            ],
            "lowest_avg_statistics": {
                "layer_distribution": avg_statistics["layer_distribution"],
                "linear_name_distribution": avg_statistics["linear_name_distribution"],
            },
            "lowest_individual_similarities": [
                {
                    "similarity": sim,
                    "layer_index": layer_idx,
                    "linear_name": linear_name,
                    "vector_index": vector_idx,
                }
                for sim, layer_idx, linear_name, vector_idx in lowest_similarities
            ],
            "lowest_individual_statistics": {
                "layer_distribution": individual_statistics["layer_distribution"],
                "linear_name_distribution": individual_statistics[
                    "linear_name_distribution"
                ],
                "vector_index_distribution": individual_statistics[
                    "vector_index_distribution"
                ],
            },
        }

        return result

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
        n: int = 100,
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
        single_expert_flag: bool = False,
    ) -> Dict[str, List[float]]:
        if moe_profile is not None and not single_expert_flag:
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
                f"Layer [{layer_idx}], Linear [{linear_key}]: performing MoE SVD analysis..."
            )
            return {linear_key: self._perform_svd_analysis(p_weight, tuned_weights)}

        elif moe_profile is not None and single_expert_flag:
            stage_result = {}
            linear_key = linear.rstrip("_")
            stage_result[linear_key] = []
            for expert_idx, expert_adapter in enumerate(loras_dict.values()):
                p_weight = expert_adapter.base_layer_.weight
                lora_a_weight = expert_adapter.lora_a_.weight
                lora_b_weight = expert_adapter.lora_b_.weight
                t_weight = lora_b_weight @ lora_a_weight + p_weight

                logging.info(
                    f"Layer [{layer_idx}], Linear [{linear_key}], Expert[{expert_idx}]: "
                    "performing single expert MoE SVD analysis..."
                )

                stage_result[linear_key].append(
                    self._perform_svd_analysis(p_weight, t_weight)
                )

            return stage_result

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
                    f"Layer [{layer_idx}], Linear [{linear_key}]: performing LoRA SVD analysis..."
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
        self, target_linears_list: List[Dict[str, List[str]]], sigle_expert_mood: str
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

                            if not moes_dict:
                                moes_dict = block.moes_

                            if adapter_name in moes_dict and (
                                layer.mlp_.moes_ or block.moes_
                            ):
                                profile_matrix = moes_dict[adapter_name].profiler_
                                analysis_result = self._process_lora_block(
                                    layer_idx,
                                    linear,
                                    adapter_name,
                                    loras_dict,
                                    moe_profile=profile_matrix,
                                    single_expert_flag=self._single_expert_mood,
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
