import heapq
import json


def analyze_svd_data(json_file):
    # 读取JSON数据
    with open(json_file, "r") as f:
        data = json.load(f)

    # 1️⃣ 计算每个线性层的前9个奇异向量的平均余弦相似度
    avg_similarities = []
    for layer_idx, layer in enumerate(data):
        for linear_data in layer:
            for linear_name, similarities in linear_data.items():
                avg_similarity = sum(abs(sim) for sim in similarities) / len(
                    similarities
                )
                avg_similarities.append((avg_similarity, layer_idx, linear_name))

    # 找出15个最低平均余弦相似度的层索引和线性层名称
    lowest_avg_similarities = heapq.nsmallest(15, avg_similarities, key=lambda x: x[0])

    # 2️⃣ 找出全部余弦相似度中最低的15个
    all_similarities = []
    for layer_idx, layer in enumerate(data):
        for linear_data in layer:
            for linear_name, similarities in linear_data.items():
                for vector_idx, similarity in enumerate(similarities):
                    all_similarities.append(
                        (abs(similarity), layer_idx, linear_name, vector_idx)
                    )

    # 找出15个最低的余弦相似度
    lowest_similarities = heapq.nsmallest(15, all_similarities, key=lambda x: x[0])

    # 格式化输出
    lowest_avg_result = [
        {"avg_similarity": avg, "layer_index": layer_idx, "linear_name": linear_name}
        for avg, layer_idx, linear_name in lowest_avg_similarities
    ]

    lowest_similarity_result = [
        {
            "adapter_name": "arc_c_1",
            "similarity": sim,
            "layer_index": layer_idx,
            "linear_name": linear_name,
            "vector_index": vector_idx,
        }
        for sim, layer_idx, linear_name, vector_idx in lowest_similarities
    ]

    return lowest_avg_result, lowest_similarity_result


# 使用示例
if __name__ == "__main__":
    input_json_file = "svd_result: arc_c_1.json"  # 替换为你的JSON文件路径
    lowest_avg, lowest_sims = analyze_svd_data(input_json_file)

    # 打印结果
    print("1️⃣ 最低平均余弦相似度的15个线性层:")
    for entry in lowest_avg:
        print(entry)

    print("\n2️⃣ 全部余弦相似度中最低的15个:")
    for entry in lowest_sims:
        print(entry)
