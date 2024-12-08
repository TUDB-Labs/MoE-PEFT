import logging
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity

from .model import LLMModel

def keys_extraction(config) -> list:
    result = []

    name = config.adapter_name
    target_modules = config.target_modules
    true_keys = [key for key, value in target_modules.items() if value]
    result.append({name: true_keys})

    return result  # 冗余数据结构设计待优化

def mapping(keys_list) -> list:
    mapping_dict = {
        "q_proj": "wq_",
        "k_proj": "wk_",
        "v_proj": "wv_",
        "o_proj": "wo_",
        "gate_proj": "w1_",
        "down_proj": "w2_",
        "up_proj": "w3_"
    }

    mapped_list = [
        {name: [mapping_dict[key] for key in keys if key in mapping_dict]}
        for item in keys_list
        for name, keys in item.items()
    ]

    return mapped_list

def lora_weight_traverse(model, target_linears_list) -> Tuple[Dict, Dict]:
    attn_linears = ['wq_', 'wk_', 'wv_', 'wo_']
    mlp_linears = ['w1_', 'w2_', 'w3_']
    
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
                                p_weight = getattr(adapter, 'base_layer_').weight
                                lora_a_weight = getattr(adapter, 'lora_a_').weight
                                lora_b_weight = getattr(adapter, 'lora_b_').weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight
                                
                                linear_key = linear.rstrip('_')
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}")
                    
                    elif linear in mlp_linears:
                        try:
                            loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)
                            
                            if adapter is not None:
                                p_weight = getattr(adapter, 'base_layer_').weight
                                lora_a_weight = getattr(adapter, 'lora_a_').weight
                                lora_b_weight = getattr(adapter, 'lora_b_').weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight
                                
                                linear_key = linear.rstrip('_')
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}")
                    
                    else:
                        raise ValueError(f"Invalid linear name: {linear}")
        
        pretrained_layers_weights.append(pretrained_layer_weights)
        tuned_layers_weights.append(tuned_layer_weights)

    return pretrained_layers_weights, tuned_layers_weights

def moe_weight_traverse(model, target_linears_list) -> Tuple[Dict, Dict]:
    attn_linears = ['wq_', 'wk_', 'wv_', 'wo_']
    mlp_linears = ['w1_', 'w2_', 'w3_']
    
    pretrained_layers_weights = []
    tuned_layers_weights = []

    for layer in model.model_.layers_:  # layer: single layer
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
                                p_weight = getattr(adapter, 'base_layer_').weight
                                lora_a_weight = getattr(adapter, 'lora_a_').weight
                                lora_b_weight = getattr(adapter, 'lora_b_').weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight
                                
                                linear_key = linear.rstrip('_')
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}")
                    
                    elif linear in mlp_linears:
                        try:
                            loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                            adapter = loras_dict.get(adapter_name, None)  # 获取adapter_name
                            if layer.mlp_.moes_:
                                profile_matrix = layer.mlp_.moes_[adapter_name].profiler_

                            if adapter is not None:
                                p_weight = getattr(adapter, 'base_layer_').weight
                                lora_a_weight = getattr(adapter, 'lora_a_').weight
                                lora_b_weight = getattr(adapter, 'lora_b_').weight
                                t_weight = lora_b_weight @ lora_a_weight + p_weight
                                
                                linear_key = linear.rstrip('_')
                                pretrained_layer_weights.append({linear_key: p_weight})
                                tuned_layer_weights.append({linear_key: t_weight})
                        except AttributeError as e:
                            raise AttributeError(f"Error accessing attributes for linear '{linear}' in adapter '{adapter_name}': {e}")
                    
                    else:
                        raise ValueError(f"Invalid linear name: {linear}")
                    
        pretrained_layers_weights.append(pretrained_layer_weights)
        tuned_layers_weights.append(tuned_layer_weights)
    
    return pretrained_layers_weights, tuned_layers_weights

def svd_analysis(p_weights: list, f_weights: list, n: int = 9):
    results = []

    for layer_idx, (p_layer, f_layer) in enumerate(zip(p_weights, f_weights)):
        layer_results = []
        for key in p_layer.keys():
            p_tensor = np.array(p_layer[key])
            f_tensor = np.array(f_layer[key])
            
            p_u, _, _ = np.linalg.svd(p_tensor, full_matrices=False)
            f_u, _, _ = np.linalg.svd(f_tensor, full_matrices=False)
            
            p_top_n = p_u[:, :n]
            f_top_n = f_u[:, :n]
            
            similarity = cosine_similarity(p_top_n.T, f_top_n.T)
            avg_similarity = np.mean(similarity)
            
            layer_results.append({key: avg_similarity})
        results.append(layer_results)
    
    return results


def process(model: LLMModel, config):
    if config.moe_flag:
        return moe_weight_traverse(model, mapping(keys_extraction(config)))

    else:
        return lora_weight_traverse(model, mapping(keys_extraction(config)))
