import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 骨干网络: DINOv2 ViT-Base Patch14
# ==========================================
class DINOv2_Backbone(nn.Module):
    def __init__(self):
        super(DINOv2_Backbone, self).__init__()
        # 自动从官方拉取 DINOv2 权重，无需手动下载
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.in_planes = 768  # ViT-Base 的特征维度
        # --- 新增：冻结浅层特征，防止预训练知识崩塌 ---
        # 冻结 patch_embed 层
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
            
        # 冻结前 6 层 Transformer blocks (ViT-Base 共有 12 层)
        for i in range(4): 
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = False
    def forward(self, x):
        # DINOv2 requires input shape [B, 3, 224, 224]
        # Use forward_features to get all tokens (CLS + patch tokens)
        
        # Forward pass through transformer backbone
        features = self.backbone.forward_features(x)  # Dict output
        
        # DINOv2 hub models expose keys like:
        # - x_prenorm: [B, 257, 768]
        # - x_norm_clstoken: [B, 768]
        # - x_norm_patchtokens: [B, 256, 768]
        if isinstance(features, dict):
            if 'x_prenorm' in features and isinstance(features['x_prenorm'], torch.Tensor):
                x = features['x_prenorm']  # [B, 257, 768]
            elif (
                'x_norm_clstoken' in features
                and 'x_norm_patchtokens' in features
                and isinstance(features['x_norm_clstoken'], torch.Tensor)
                and isinstance(features['x_norm_patchtokens'], torch.Tensor)
            ):
                cls = features['x_norm_clstoken'].unsqueeze(1)  # [B, 1, 768]
                patches = features['x_norm_patchtokens']        # [B, 256, 768]
                x = torch.cat([cls, patches], dim=1)            # [B, 257, 768]
            else:
                raise KeyError(
                    "Unexpected DINOv2 forward_features keys: "
                    f"{list(features.keys())}"
                )
        else:
            # If it's directly a tensor, assume it's already what we need
            x = features
        
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    """
    【融合版本】单个分类头，用于计算分类损失和特征
    
    融合了旧版本的灵活参数 + 新版本的bug修复（评估时总是返回logit）
    """
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, 
                 num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        
        # 构建特征处理流程
        add_block = []
        
        # 1. 可选的线性投影（bottleneck）
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        
        # 2. 可选的归一化
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        
        # 3. 可选的激活函数
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        
        # 4. 可选的正则化
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        
        # 构建特征块
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        
        # 构建分类器
        classifier = nn.Linear(num_bottleneck, class_num)
        classifier.apply(weights_init_classifier)
        
        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        """
        前向传播
        
        训练模式:
            - return_f=True: 返回 (logit, feat)
            - return_f=False: 返回 logit
        
        评估模式（修复bug）:
            - 总是返回 (logit, feat)，便于特征提取和检索指标计算
        """
        # 提取特征
        feat = self.add_block(x)
        
        # 计算分类输出
        logit = self.classifier(feat)
        
        # 【修复】评估时也返回 (logit, feat)，不再丢失 logit
        if self.training:
            if self.return_f:
                return logit, feat
            else:
                return logit
        else:
            # 评估模式：总是返回 (logit, feat)
            return logit, feat

# ==========================================
# 2. 组装网络: 双视图 (卫星 + 无人机) + 热力分组
# ==========================================
class two_view_net(nn.Module):
    def __init__(self, opt, class_num, block=3, return_f=False):
        super(two_view_net, self).__init__()
        self.return_f = return_f
        self.block = block
        
        # 加载 DINOv2 骨干网络
        self.backbone = DINOv2_Backbone()
        feat_dim = self.backbone.in_planes
        
        # 【FSRA风格】全局分类器（用于 CLS token）
        self.global_classifier = ClassBlock(feat_dim, class_num, return_f=return_f)
        
        # 【FSRA风格】如果 block > 1，添加多个局部分类器（用于热力分组的patch）
        if self.block > 1:
            for i in range(self.block):
                name = f'part_classifier_{i}'
                setattr(self, name, ClassBlock(feat_dim, class_num, return_f=return_f))

    def get_heatmap_pool(self, patch_features):
        """
        【参考FSRA】按热力对patch进行排序和分组
        
        Args:
            patch_features: [B, num_patch, D] - 除去CLS token的patch特征
        
        Returns:
            part_features: [B, D, block] - 分组后的特征
        """
        # 计算热力：对每个patch的特征求L2范数，作为重要性分数/改了
        heatmap = torch.mean(patch_features, dim=-1)  # [B, num_patch]
        
        # 按热力从高到低排序
        num_patches = patch_features.size(1)
        sorted_idx = torch.argsort(heatmap, dim=1, descending=True)  # [B, num_patch]
        
        # 用torch.gather完成向量化索引，避免for循环
        batch_size = patch_features.size(0)
        feature_dim = patch_features.size(2)
        
        # 索引扩展至特征维度: [B, num_patch, D]
        sorted_idx_expanded = sorted_idx.unsqueeze(-1).expand(-1, -1, feature_dim)
        x_sorted = torch.gather(patch_features, 1, sorted_idx_expanded)  # [B, num_patch, D]
        
        # 将排序后的patch均匀分组
        split_size = num_patches // self.block
        split_list = [split_size] * (self.block - 1)
        split_list.append(num_patches - sum(split_list))  # 处理余数
        
        # 对每组patch求平均
        part_features_list = []
        start_idx = 0
        for group_size in split_list:
            group = x_sorted[:, start_idx:start_idx+group_size, :]  # [B, group_size, D]
            group_mean = torch.mean(group, dim=1)  # [B, D]
            part_features_list.append(group_mean)
            start_idx += group_size
        
        # 堆叠成 [B, D, block]
        part_features = torch.stack(part_features_list, dim=-1)  # [B, D, block]
        return part_features

    def forward(self, x1, x2):
        """
        前向传播，返回多个分类输出和特征
        
        Returns:
            训练模式:
                if return_f: (([cls_0, cls_1, ...], [feat_0, feat_1, ...]), ...)
                else: ([cls_0, cls_1, ...], ...)
            
            评估模式（始终返回特征）:
                (([cls_0, cls_1, ...], [feat_0, feat_1, ...]), ...)
        """
        # 提取完整特征 [B, N, D]，其中 N = 1 + num_patches
        x1_features = self.backbone(x1)  # [B, 257, 768]
        x2_features = self.backbone(x2)  # [B, 257, 768]
        
        # 分离 CLS token 和 patch token
        cls_token_1 = x1_features[:, 0, :]  # [B, 768]
        patch_token_1 = x1_features[:, 1:, :]  # [B, 256, 768]
        
        cls_token_2 = x2_features[:, 0, :]  # [B, 768]
        patch_token_2 = x2_features[:, 1:, :]  # [B, 256, 768]
        
        # 【全局特征】用 CLS token 计算分类和特征
        global_output_1 = self.global_classifier(cls_token_1)
        global_output_2 = self.global_classifier(cls_token_2)
        
        # 处理全局输出：在评估模式下，ClassBlock 总是返回 (logit, feat)
        if isinstance(global_output_1, (tuple, list)):
            cls_global_1, feat_global_1 = global_output_1
            cls_global_2, feat_global_2 = global_output_2
        else:
            cls_global_1 = global_output_1
            cls_global_2 = global_output_2
            feat_global_1 = feat_global_2 = None
        
        # 如果只有1个block，只返回全局分类输出
        if self.block == 1:
            # 在评估模式下，需要保证返回特征
            if not self.training and feat_global_1 is None:
                raise RuntimeError("Expected features in eval mode")
            
            if feat_global_1 is not None:
                return (cls_global_1, feat_global_1), (cls_global_2, feat_global_2)
            return cls_global_1, cls_global_2
        
        # 【局部特征】按热力分组 patch
        part_features_1 = self.get_heatmap_pool(patch_token_1)  # [B, 768, block]
        part_features_2 = self.get_heatmap_pool(patch_token_2)  # [B, 768, block]
        
        # 对每个 block 计算分类输出
        cls_list_1 = []
        feat_list_1 = []
        cls_list_2 = []
        feat_list_2 = []
        
        for i in range(self.block):
            part_feat_1 = part_features_1[:, :, i]  # [B, 768]
            part_feat_2 = part_features_2[:, :, i]  # [B, 768]
            
            classifier_i = getattr(self, f'part_classifier_{i}')
            
            part_output_1 = classifier_i(part_feat_1)
            part_output_2 = classifier_i(part_feat_2)
            
            # 处理部分输出：在评估模式下，ClassBlock 总是返回 (logit, feat)
            if isinstance(part_output_1, (tuple, list)):
                cls_i_1, feat_i_1 = part_output_1
                cls_i_2, feat_i_2 = part_output_2
            else:
                cls_i_1 = part_output_1
                cls_i_2 = part_output_2
                feat_i_1 = feat_i_2 = None
            
            cls_list_1.append(cls_i_1)
            cls_list_2.append(cls_i_2)
            
            if feat_i_1 is not None:
                feat_list_1.append(feat_i_1)
                feat_list_2.append(feat_i_2)
        
        # 添加全局分类输出到列表末尾
        cls_list_1.append(cls_global_1)
        cls_list_2.append(cls_global_2)
        
        # 【修复】根据训练模式和特征可用性决定返回格式
        # 评估模式下总是返回特征（由 ClassBlock 保证）
        if feat_global_1 is not None:
            if feat_list_1:  # 如果有部分特征
                feat_list_1.append(feat_global_1)
                feat_list_2.append(feat_global_2)
            else:  # 如果没有部分特征，创建空列表
                feat_list_1 = [feat_global_1]
                feat_list_2 = [feat_global_2]
            return (cls_list_1, feat_list_1), (cls_list_2, feat_list_2)
        
        return cls_list_1, cls_list_2

# ==========================================
# 3. 工厂函数
# ==========================================
def make_model(opt):
    return_f = bool(opt.triplet_loss > 0)
    block = getattr(opt, 'block', 1)
    
    model = two_view_net(opt, opt.nclasses, block=block, return_f=return_f)
    
    return model
