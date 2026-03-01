import numpy as np

def re_ranking(q_f, g_f, k1=20, k2=6, lambda_value=0.3):
    """
    使用 k-互反最近邻 (k-reciprocal nearest neighbors) 进行特征重排
    
    参数:
        q_f: 查询特征矩阵, numpy array, 形状为 [N_query, D]
        g_f: 底库特征矩阵, numpy array, 形状为 [N_gallery, D]
        k1: k-reciprocal 邻居的数量 (控制基础候选集大小)
        k2: 局部查询扩展 (Local Query Expansion) 的邻居数量
        lambda_value: 原始距离和 Jaccard 距离的权重惩罚系数
    
    返回:
        final_dist: 重排后的距离矩阵, 形状为 [N_query, N_gallery]
    """
    # 合并特征以计算全局距离矩阵
    q_f = q_f.astype(np.float32)
    g_f = g_f.astype(np.float32)
    query_num = q_f.shape[0]
    all_num = query_num + g_f.shape[0]
    feat = np.concatenate([q_f, g_f], axis=0)
    
    # 1. 计算初始距离矩阵 (假设特征已做 L2 归一化，这里使用余弦距离的等价欧氏距离形式)
    # 欧氏距离^2 = 2 - 2 * (A * B^T)
    original_dist = np.dot(feat, feat.T)
    original_dist = 2.0 - 2.0 * original_dist
    np.maximum(original_dist, 0.0, original_dist) # 避免由于浮点误差出现负数
    
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # 2. 寻找 k-互反最近邻并计算 Jaccard 距离
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]
    
    # 3. 局部查询扩展 (Local Query Expansion)
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    
    del initial_rank
    
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        
        jaccard_dist[i] = 1.0 - temp_min / (2.0 - temp_min)
    
    # 4. 融合最终距离矩阵
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    
    # 仅返回 Query 到 Gallery 的距离矩阵
    final_dist = final_dist[:query_num, query_num:]
    return final_dist