from typing import Dict, List, Tuple
import logging
import torch

from .model import LLMModel


def keys_extraction(config) -> list:
    result = []

    name = config.adapter_name
    target_modules = config.target_modules
    true_keys = [key for key, value in target_modules.items() if value]
    result.append({name: true_keys})

    return result


def mapping(keys_list) -> list:
    mapping_dict = {
        "q_proj": "wq_",
        "k_proj": "wk_",
        "v_proj": "wv_",
        "o_proj": "wo_",
        "gate_proj": "w1_",
        "down_proj": "w2_",
        "up_proj": "w3_",
    }

    mapped_list = [
        {name: [mapping_dict[key] for key in keys if key in mapping_dict]}
        for item in keys_list
        for name, keys in item.items()
    ]

    return mapped_list


def moe_weight_caculate(loading: list, lora_weights: list) -> torch.Tensor:
    return sum(weight * tensor for weight, tensor in zip(loading, lora_weights))


def lora_weight_traverse(model, target_linears_list) -> Tuple[Dict, Dict]:
    attn_linears = ["wq_", "wk_", "wv_", "wo_"]
    mlp_linears = ["w1_", "w2_", "w3_"]

    pretrained_layers_weights = []
    tuned_layers_weights = []

    for layer in model.model_.layers_:
        pretrained_layer_weights = []
        tuned_layer_weights = []
        for item in target_linears_list:
            for adapter_name, linear_lst in item.items():
                for linear in linear_lst:
                    if linear in attn_linears:
                        try:
                            loras_dict = getattr(layer.self_attn_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)

                            if adapter is not None:
                                p_weight = getattr(adapter, "base_layer_").weight
                                lora_a_weight = getattr(adapter, "lora_a_").weight
                                lora_b_weight = getattr(adapter, "lora_b_").weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight

                                linear_key = linear.rstrip("_")
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(
                                f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}"
                            )

                    elif linear in mlp_linears:
                        try:
                            loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)

                            if adapter is not None:
                                p_weight = getattr(adapter, "base_layer_").weight
                                lora_a_weight = getattr(adapter, "lora_a_").weight
                                lora_b_weight = getattr(adapter, "lora_b_").weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight

                                linear_key = linear.rstrip("_")
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(
                                f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}"
                            )

                    else:
                        raise ValueError(f"Invalid linear name: {linear}")

        pretrained_layers_weights.append(pretrained_layer_weights)
        tuned_layers_weights.append(tuned_layer_weights)

    return pretrained_layers_weights, tuned_layers_weights


def moe_weight_traverse_and_processing(model, target_linears_list) -> None:
    attn_linears = ["wq_", "wk_", "wv_", "wo_"]
    mlp_linears = ["w1_", "w2_", "w3_"]
    final_result = []

    for layer_idx, layer in enumerate(model.model_.layers_):  # layer: single decoder layer
        layer_result = []
        for item in target_linears_list:
            for adapter_name, linear_lst in item.items():
                for linear in linear_lst:  # qkvoudg
                    if linear in attn_linears:
                        try:
                            loras_dict = getattr(layer.self_attn_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)

                            if getattr(layer.self_attn_, linear).moes_:
                                profile_matrix = getattr(layer.self_attn_, linear).moes_[
                                    adapter_name
                                ].profiler_
                                expert_value_lists = loras_dict.values()
                                tuned_expert_value_lists = []

                                for value in expert_value_lists:
                                    p_weight = value.base_layer_.weight
                                    lora_a_weight = value.lora_a_.weight
                                    lora_b_weight = value.lora_b_.weight
                                    t_weight = lora_b_weight @ lora_a_weight + p_weight
                                    tuned_expert_value_lists.append(t_weight)

                                tuned_weights = moe_weight_caculate(
                                    profile_matrix, tuned_expert_value_lists
                                )

                                linear_key = linear.rstrip("_")
                                logging.info(f"start to analysis layer:{layer_idx}, linear:{linear_key}'s svd decomposition result...")
                                layer_result.append(
                                    {
                                        linear_key: svd_analysis(
                                            p_weight, tuned_weights
                                        )
                                    }  # layer result
                                )

                            else:
                                if adapter is not None:
                                    p_weight = getattr(adapter, "base_layer_").weight
                                    lora_a_weight = getattr(adapter, "lora_a_").weight
                                    lora_b_weight = getattr(adapter, "lora_b_").weight
                                    t_weight = lora_b_weight @ lora_a_weight + p_weight

                                    linear_key = linear.rstrip("_")
                                    logging.info(f"start to analysis layer:{layer_idx}, linear:{linear_key}'s svd decomposition result...")
                                    layer_result.append(
                                        {
                                            linear_key: svd_analysis(p_weight, t_weight)
                                        }  # linear result
                                    )

                        except AttributeError as e:
                            raise AttributeError(
                                f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}"
                            )

                    elif linear in mlp_linears:
                        try:
                            loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)

                            if layer.mlp_.moes_:
                                profile_matrix = layer.mlp_.moes_[
                                    adapter_name
                                ].profiler_
                                expert_value_lists = loras_dict.values()
                                tuned_expert_value_lists = []

                                for value in expert_value_lists:
                                    p_weight = value.base_layer_.weight
                                    lora_a_weight = value.lora_a_.weight
                                    lora_b_weight = value.lora_b_.weight
                                    t_weight = lora_b_weight @ lora_a_weight + p_weight
                                    tuned_expert_value_lists.append(t_weight)

                                tuned_weights = moe_weight_caculate(
                                    profile_matrix, tuned_expert_value_lists
                                )

                                linear_key = linear.rstrip("_")
                                logging.info(f"start to analysis layer:{layer_idx}, linear:{linear_key}'s svd decomposition result...")
                                layer_result.append(
                                    {
                                        linear_key: svd_analysis(
                                            p_weight, tuned_weights
                                        )
                                    }  # layer result
                                )

                            else:  # 普通lora微调的逻辑
                                if adapter is not None:
                                    p_weight = getattr(adapter, "base_layer_").weight
                                    lora_a_weight = getattr(adapter, "lora_a_").weight
                                    lora_b_weight = getattr(adapter, "lora_b_").weight
                                    t_weight = lora_b_weight @ lora_a_weight + p_weight

                                    linear_key = linear.rstrip("_")
                                    logging.info(f"start to analysis layer:{layer_idx}, linear:{linear_key}'s svd decomposition result...")
                                    layer_result.append(
                                        {
                                            linear_key: svd_analysis(p_weight, t_weight)
                                        }  # linear result
                                    )

                        except AttributeError as e:
                            raise AttributeError(
                                f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}"
                            )

                    else:
                        raise ValueError(f"Invalid linear name: {linear}")

        final_result.append(layer_result)

    return final_result


def svd_analysis(
    p_weight: torch.Tensor, f_weight: torch.Tensor, n: int = 9, device: str = "cuda:0"
) -> List[float]:
    try:
        # 将张量转换为 float32，并移动到指定设备
        p_weight = p_weight.to(torch.float32).to(device)
        f_weight = f_weight.to(torch.float32).to(device)

        # 进行SVD分解
        p_u, _, _ = torch.linalg.svd(p_weight, full_matrices=False)
        f_u, _, _ = torch.linalg.svd(f_weight, full_matrices=False)

        # 获取前n个奇异向量
        n_min = min(n, p_u.shape[1], f_u.shape[1])
        p_top_n = p_u[:, :n_min]
        f_top_n = f_u[:, :n_min]

        # 计算余弦相似度
        similarity = torch.mm(p_top_n.T, f_top_n)  # 点积
        p_norms = torch.norm(p_top_n.T, dim=1, keepdim=True)  # 计算 p_top_n 的范数
        f_norms = torch.norm(f_top_n, dim=0, keepdim=True)  # 计算 f_top_n 的范数
        similarity = similarity / (p_norms * f_norms)  # 标准化为余弦相似度

        # 转为 Python 标量列表
        cos_similarities = similarity.diagonal().tolist()

        return cos_similarities

    except RuntimeError as e:
        print(f"SVD 计算时出错: {e}")
        return []


def process(model: LLMModel, config):
    if config.moe_flag:
        ana_result = moe_weight_traverse_and_processing(
            model, mapping(keys_extraction(config))
        )
        return ana_result

    else:
        ana_result = lora_weight_traverse(model, mapping(keys_extraction(config)))
        return ana_result
