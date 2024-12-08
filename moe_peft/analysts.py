import torch
from typing import List, Tuple


class SVDProcessor:
    def __init__(self, model, config):
        """
        初始化 LLMProcessor
        :param model: LLMModel 实例
        :param config: 配置对象，包含 adapter_name、target_modules 和 moe_flag 等属性
        """
        self.model = model
        self.config = config

    def keys_extraction(self) -> list:
        """
        提取目标 keys
        :return: 包含 adapter 名称及其 keys 的列表
        """
        result = []
        name = self.config.adapter_name
        target_modules = self.config.target_modules
        true_keys = [key for key, value in target_modules.items() if value]
        result.append({name: true_keys})
        return result

    @staticmethod
    def mapping(keys_list: list) -> list:
        """
        将 keys 映射为标准化名称
        :param keys_list: 提取的 keys 列表
        :return: 映射后的 keys 列表
        """
        mapping_dict = {
            "q_proj": "wq_",
            "k_proj": "wk_",
            "v_proj": "wv_",
            "o_proj": "wo_",
            "gate_proj": "w1_",
            "down_proj": "w2_",
            "up_proj": "w3_"
        }

        return [
            {name: [mapping_dict[key] for key in keys if key in mapping_dict]}
            for item in keys_list
            for name, keys in item.items()
        ]

    @staticmethod
    def moe_weight_caculate(loading: list, lora_weights: list) -> torch.Tensor:
        """
        计算加权的 MOE 权重
        :param loading: 加载 profile_matrix
        :param lora_weights: 各个 expert 的权重
        :return: 最终加权权重
        """
        return sum(weight * tensor for weight, tensor in zip(loading, lora_weights))

    def weight_traverse(self, target_linears_list, is_moe: bool = False) -> Tuple[List, List]:
        """
        遍历权重
        :param target_linears_list: 提取的线性层列表
        :param is_moe: 是否为 MOE 模式
        :return: (预训练权重, 微调权重)
        """
        attn_linears = ['wq_', 'wk_', 'wv_', 'wo_']
        mlp_linears = ['w1_', 'w2_', 'w3_']

        pretrained_layers_weights = []
        tuned_layers_weights = []

        for layer in self.model.model_.layers_:
            pretrained_layer_weights = []
            tuned_layer_weights = []
            for item in target_linears_list:
                for adapter_name, linear_lst in item.items():
                    for linear in linear_lst:
                        try:
                            # 判断是注意力层还是 MLP 层
                            if linear in attn_linears:
                                loras_dict = getattr(layer.self_attn_, linear).loras_
                            elif linear in mlp_linears:
                                loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            else:
                                raise ValueError(f"Invalid linear name: {linear}")

                            adapter = loras_dict.get(adapter_name, None)

                            if adapter:
                                p_weight = getattr(adapter, 'base_layer_').weight
                                lora_a_weight = getattr(adapter, 'lora_a_').weight
                                lora_b_weight = getattr(adapter, 'lora_b_').weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight

                                linear_key = linear.rstrip('_')
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})

                            # MOE 特定逻辑
                            if is_moe and hasattr(layer.mlp_, 'moes_') and adapter_name in layer.mlp_.moes_:
                                profile_matrix = layer.mlp_.moes_[adapter_name].profiler_
                                expert_value_lists = loras_dict.values()
                                tuned_expert_value_lists = []
                                total_base_layer = getattr(layer.mlp_.mlp_, linear).base_layer_.weight

                                for value in expert_value_lists:
                                    p_weight = value.base_layer_.weight
                                    lora_a_weight = value.lora_a_.weight
                                    lora_b_weight = value.lora_b_.weight
                                    t_weight = lora_b_weight @ lora_a_weight + p_weight
                                    tuned_expert_value_lists.append(t_weight)

                                final_tuned_weights = self.moe_weight_caculate(profile_matrix, tuned_expert_value_lists)
                                pretrained_layer_weights.append({linear_key: total_base_layer})
                                tuned_layer_weights.append({linear_key: final_tuned_weights})

                        except AttributeError as e:
                            raise AttributeError(f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}")

            pretrained_layers_weights.append(pretrained_layer_weights)
            tuned_layers_weights.append(tuned_layer_weights)

        return pretrained_layers_weights, tuned_layers_weights

    @staticmethod
    def svd_analysis(p_weights: list, f_weights: list, n: int = 9, device: str = 'cuda:0') -> List:
        """
        对比分析 SVD 分解的权重
        :param p_weights: 预训练权重
        :param f_weights: 微调权重
        :param n: SVD 分解的 top N 奇异值
        :param device: 计算设备
        :return: 分析结果
        """
        results = []

        for layer_idx, (p_layer, f_layer) in enumerate(zip(p_weights, f_weights)):
            layer_results = []
            for key in p_layer.keys():
                p_tensor = p_layer[key].to(device) if isinstance(p_layer[key], torch.Tensor) else torch.tensor(p_layer[key], device=device)
                f_tensor = f_layer[key].to(device) if isinstance(f_layer[key], torch.Tensor) else torch.tensor(f_layer[key], device=device)

                p_u, _, _ = torch.linalg.svd(p_tensor, full_matrices=False)
                f_u, _, _ = torch.linalg.svd(f_tensor, full_matrices=False)

                n_min = min(n, p_u.shape[1], f_u.shape[1])
                p_top_n = p_u[:, :n_min]
                f_top_n = f_u[:, :n_min]

                similarity = torch.mm(p_top_n.T, f_top_n)
                p_norms = torch.norm(p_top_n.T, dim=1, keepdim=True)
                f_norms = torch.norm(f_top_n, dim=0, keepdim=True)
                similarity = similarity / (p_norms * f_norms)
                avg_similarity = similarity.mean().item()

                layer_results.append({key: avg_similarity})
            results.append(layer_results)

        return results

    def process(self) -> List:
        """
        执行处理流程
        :return: SVD 分析结果
        """
        keys_list = self.keys_extraction()
        mapped_keys = self.mapping(keys_list)

        if self.config.moe_flag:
            weights = self.weight_traverse(mapped_keys, is_moe=True)
        else:
            weights = self.weight_traverse(mapped_keys)

        return self.svd_analysis(weights[0], weights[1])