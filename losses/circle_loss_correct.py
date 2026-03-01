"""
Circle Loss - 标准论文实现 (CVPR 2020)
Sun et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization"

修正要点：
1. 移除之前离谱的 +1 操作（破坏余弦相似度语义）
2. 正确理解论文中的 O_p, O_n, Delta_p, Delta_n
3. 正确的动态权重计算
4. 使用 logsumexp 确保数值稳定性（gamma 可以很大）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss 标准实现
    
    参数：
        m (float): relaxation margin，论文推荐 0.25
        gamma (float): scale factor，推荐 256 或 128
    """
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        
        # 论文中的关键定义（非常重要！）
        self.O_p = 1 + m        # 正样本操作点
        self.O_n = -m           # 负样本操作点
        self.Delta_p = 1 - m    # 正样本 margin
        self.Delta_n = m        # 负样本 margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, feat_dim] - 特征向量（会进行 L2 归一化）
            targets: [batch_size] - 样本标签
        
        Returns:
            loss: 标量 Circle Loss 值
        """
        # 1. L2 归一化特征，确保余弦相似度在 [-1, 1]
        inputs = F.normalize(inputs, p=2, dim=1)
        sim_mat = torch.matmul(inputs, inputs.t())  # [n, n]
        
        n = inputs.size(0)
        targets = targets.view(-1)
        
        # 生成分类掩码：同类为True，异类为False
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        loss = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        valid_samples = 0
        
        # 2. 逐 anchor 计算 Circle Loss
        for i in range(n):
            # 获取正负样本掩码
            pos_mask = mask[i].clone()
            pos_mask[i] = False  # 排除自身（anchor 不能和自身配对）
            neg_mask = ~mask[i]
            
            # 获取相似度
            pos_sim = sim_mat[i][pos_mask]
            neg_sim = sim_mat[i][neg_mask]
            
            # 如果没有正样本或负样本，跳过
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue
            
            valid_samples += 1
            
            # 3. 【核心】动态计算权重 alpha
            # 论文公式：
            # alpha_p = max(0, O_p - s_p)  # 如果 s_p 太低，权重大；如果 s_p ≈ 1，权重小
            # alpha_n = max(0, s_n - O_n)  # 如果 s_n 太高，权重大；如果 s_n ≈ -1，权重小
            
            # 使用 detach() 确保权重对原始特征不计梯度
            # （只有损失项对梯度有影响，权重本身不获得梯度）
            alpha_p = F.relu(self.O_p - pos_sim.detach())
            alpha_n = F.relu(neg_sim.detach() - self.O_n)
            
            # 4. 【损失项】带加权 margin 的余弦距离
            # pos 项: -gamma * alpha_p * (s_p - Delta_p)
            #   当 s_p 接近 1 时，(s_p - Delta_p) 接近 m，损失小（好）
            #   当 s_p 接近 -1 时，(s_p - Delta_p) 接近 -(1-m)，损失大（坏）
            #
            # neg 项: +gamma * alpha_n * (s_n - Delta_n)
            #   当 s_n 接近 -1 时，(s_n - Delta_n) 接近 -(1+m)，损失小（好）
            #   当 s_n 接近 1 时，(s_n - Delta_n) 接近 (1-m)，损失大（坏）
            
            pos_term = -self.gamma * alpha_p * (pos_sim - self.Delta_p)
            neg_term = self.gamma * alpha_n * (neg_sim - self.Delta_n)
            
            # 5. 【数值稳定计算】使用 logsumexp 处理潜在的大指数值
            # 这是关键：避免 exp(gamma * 大数) 导致的溢出
            # 论文公式(4)的核心：双重求和可分解为乘积
            # ∑∑ exp(P_i + N_j) = (∑exp(P_i)) · (∑exp(N_j))
            # log形式：log(∑e^P) + log(∑e^N) = log(∑e^P · ∑e^N)
            # 最终损失: softplus(pos_exp + neg_exp) = log(1 + ∑e^P · ∑e^N)
            pos_exp = torch.logsumexp(pos_term, dim=0)
            neg_exp = torch.logsumexp(neg_term, dim=0)
            loss += F.softplus(pos_exp + neg_exp)
        
        # 6. 返回有效样本的平均损失
        if valid_samples == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return loss / valid_samples


class CircleLossWithHardMining(nn.Module):
    """
    Circle Loss + Hard Example Mining
    在动态权重的基础上，进一步放大难例的梯度
    """
    def __init__(self, m=0.25, gamma=256, hard_mining_weight=2.0):
        super(CircleLossWithHardMining, self).__init__()
        self.circle_loss = CircleLoss(m=m, gamma=gamma)
        self.hard_mining_weight = hard_mining_weight

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        sim_mat = torch.matmul(inputs, inputs.t())
        
        n = inputs.size(0)
        targets = targets.view(-1)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        loss = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        valid_samples = 0
        
        for i in range(n):
            pos_mask = mask[i].clone()
            pos_mask[i] = False
            neg_mask = ~mask[i]
            
            pos_sim = sim_mat[i][pos_mask]
            neg_sim = sim_mat[i][neg_mask]
            
            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue
            
            valid_samples += 1
            
            alpha_p = F.relu(self.circle_loss.O_p - pos_sim.detach())
            alpha_n = F.relu(neg_sim.detach() - self.circle_loss.O_n)
            
            # 【新增】Hard mining：加大难例的权重
            # 硬正样本（距离最远的正样本）
            if len(pos_sim) > 1:
                hard_pos_idx = torch.argmin(pos_sim)  # 最小相似度
                alpha_p[hard_pos_idx] = alpha_p[hard_pos_idx] * self.hard_mining_weight
            
            # 硬负样本（距离最近的负样本）
            if len(neg_sim) > 1:
                hard_neg_idx = torch.argmax(neg_sim)  # 最大相似度
                alpha_n[hard_neg_idx] = alpha_n[hard_neg_idx] * self.hard_mining_weight
            
            pos_term = -self.circle_loss.gamma * alpha_p * (pos_sim - self.circle_loss.Delta_p)
            neg_term = self.circle_loss.gamma * alpha_n * (neg_sim - self.circle_loss.Delta_n)
            
            # 论文公式(4)的核心：正负样本交叉的两两组合（乘积）
            pos_exp = torch.logsumexp(pos_term, dim=0)
            neg_exp = torch.logsumexp(neg_term, dim=0)
            loss += F.softplus(pos_exp + neg_exp)
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return loss / valid_samples
