import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def keys_extraction(config) -> list:
    result = []

    for lora_config in config.get("lora", []):
        name = lora_config.get("name", "unknown")
        target_modules = lora_config.get("target_modules", {})
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
        "up_proj": "w3_"
    }

    mapped_list = [
        {name: [mapping_dict[key] for key in keys if key in mapping_dict]}
        for item in keys_list
        for name, keys in item.items()
    ]

    return mapped_list

def traverse(model, target_linears_list):
    attn_linears = ['wq_', 'wk_', 'wv_', 'wo_']
    mlp_linears = ['w1_', 'w2_', 'w3_']

    for layer in model.model_.layers_:  # Decoder Layer
        for item in target_linears_list:  # 含adapter_name
            for adapter_name, linear_lst in item.items():  # 键值分离
                for linear in linear_lst:
                    if linear in attn_linears:
                        loras_dict = getattr(layer.self_attn_, linear).loras_
                        adapter = loras_dict.get(adapter_name, None)
                        if adapter is not None:
                            p_weight = getattr(adapter, 'base_layer_').weight
                            lora_a_weight = getattr(adapter, 'lora_a_').weight
                            lora_b_weight = getattr(adapter, 'lora_b_').weight
                            t_weight = lora_b_weight @ lora_a_weight + p_weight
                            svd_analysis(p_weight, t_weight, n=9)

                    elif linear in mlp_linears:
                        loras_dict = getattr(layer.mlp_.mlp_, linear).loras_
                        adapter = loras_dict.get(adapter_name, None)
                        if adapter is not None:
                            p_weight = getattr(adapter, 'base_layer_').weight
                            lora_a_weight = getattr(adapter, 'lora_a_').weight
                            lora_b_weight = getattr(adapter, 'lora_b_').weight
                            t_weight = lora_b_weight @ lora_a_weight + p_weight
                            svd_analysis(p_weight, t_weight, n=9)
                    else:
                        raise ValueError(f"Invalid linear name: {linear}")

def svd_analysis(p_weight: list, f_weight: list, n: int=9):
    logging.info("Start SVD analysis...")
    for p_point, f_point in zip(p_weight, f_weight):
        # Perform SVD on pre-training weight matrix
        U_p, Sigma_p, Vt_p = np.linalg.svd(p_point, full_matrices=False)
        
        # Perform SVD on fine-tuning weight matrix
        U_f, Sigma_f, Vt_f = np.linalg.svd(f_point, full_matrices=False)
        
        # Calculate cosine similarity between corresponding singular vectors in U
        cosine_sim_U = [cosine_similarity(U_p[:, i].reshape(1, -1), U_f[:, i].reshape(1, -1))[0, 0]
                        for i in range(min(U_p.shape[1], U_f.shape[1]))]
        
        # Calculate cosine similarity between corresponding singular vectors in V
        cosine_sim_V = [cosine_similarity(Vt_p[i, :].reshape(1, -1), Vt_f[i, :].reshape(1, -1))[0, 0]
                        for i in range(min(Vt_p.shape[0], Vt_f.shape[0]))]
        
        # Display results
        print("Cosine similarity of left singular vectors (U):", cosine_sim_U)
        print("Cosine similarity of right singular vectors (V):", cosine_sim_V)
        print("Singular values of pre-training matrix (Sigma_p):", Sigma_p)
        print("Singular values of fine-tuning matrix (Sigma_f):", Sigma_f)
    
    return np.array(cosine_sim_U), np.array(cosine_sim_V), Sigma_p, Sigma_f


# Function to perform SVD and calculate cosine similarity between singular vectors
def process(model, config):
    ft_inform = mapping(keys_extraction(config))
    traverse(model, ft_inform)
















    





# import json
# import logging
# import torch
# import numpy as np

# def cosine_similarity(vec1, vec2):
#     dot_product = torch.dot(vec1, vec2)
#     norm_vec1 = torch.norm(vec1)
#     norm_vec2 = torch.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)


# def svd_analysis(p_weight: list, t_weight: list, n: int=3):
#     logging.info("Start SVD analysis...")
#     for p_point, t_point in zip(p_weight, t_weight):
#         # 对两个矩阵进行 SVD 分解
#         U_p, sigma_p, Vh_p = torch.linalg.svd(p_point)
#         U_t, sigma_t, Vh_t = torch.linalg.svd(t_point)

#         # 提取前 n 个奇异值和对应的列（U）和行（V^T）
#         sigma_p_top = sigma_p[:n]
#         sigma_t_top = sigma_t[:n]

#         U_p_top = U_p[:, :n]  # p_point 的前 n 列左奇异向量
#         U_t_top = U_t[:, :n]  # t_point 的前 n 列左奇异向量

#         V_p_top = Vh_p[:n, :]  # p_point 的前 n 行右奇异向量 (V^T)
#         V_t_top = Vh_t[:n, :]  # t_point 的前 n 行右奇异向量 (V^T)

#         # 计算奇异向量的余弦相似度
#         U_cos_similarities = [cosine_similarity(U_p_top[:, i], U_t_top[:, i]) for i in range(n)]
#         V_cos_similarities = [cosine_similarity(V_p_top[i, :], V_t_top[i, :]) for i in range(n)]

#         # 打印结果
#         print("Top-n Singular Values for P:", sigma_p_top)
#         print("Top-n Singular Values for T:", sigma_t_top)
#         print("\nCosine Similarities of Left Singular Vectors (U):", U_cos_similarities)
#         print("Cosine Similarities of Right Singular Vectors (V^T):", V_cos_similarities)