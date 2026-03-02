import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. 骨干网络: DINOv2 ViT-Base Patch14
# ==========================================
class DINOv2_Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(DINOv2_Backbone, self).__init__()
        # 自动从官方拉取 DINOv2 权重，无需手动下载
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.in_planes = 768  # ViT-Base 的特征维度
        
        # 【新增】如果指定了pretrain_path，加载预训练的权重
        if pretrain_path and len(pretrain_path) > 0:
            # 文件存在性检查
            if not os.path.exists(pretrain_path):
                print(f"[!] Warning: pretrain_path file not found: {pretrain_path}")
                print(f"[!] Continuing with official DINOv2 hub weights...")
            else:
                print(f"[*] Loading pretrained DINOv2 backbone from: {pretrain_path}")
                try:
                    checkpoint = torch.load(pretrain_path, map_location='cpu')
                except Exception as e:
                    print(f"[!] Error loading checkpoint file: {e}")
                    print(f"[!] Continuing with official DINOv2 hub weights...")
                else:
                    # 处理不同的checkpoint格式
                    backbone_state = None
                    
                    if isinstance(checkpoint, dict):
                        # 情况1: 完整的两视图模型（key: backbone.backbone.*）
                        if any(k.startswith('backbone.backbone.') for k in checkpoint.keys()):
                            backbone_state = {k.replace('backbone.backbone.', ''): v 
                                             for k, v in checkpoint.items() 
                                             if k.startswith('backbone.backbone.')}
                            print(f"[*] Detected full two_view_net checkpoint format")
                        
                        # 情况2: backbone状态字典（key: blocks.*, patch_embed.*, ...）
                        elif any(k.startswith('blocks') or k.startswith('patch_embed') for k in checkpoint.keys()):
                            backbone_state = checkpoint
                            print(f"[*] Detected DINOv2_Backbone checkpoint format")
                        
                        # 情况3: 可能是包含backbone的更大字典（按照模块名查找）
                        elif 'backbone' in checkpoint:
                            if isinstance(checkpoint['backbone'], dict):
                                potential_state = checkpoint['backbone']
                                if any(k.startswith('blocks') or k.startswith('patch_embed') for k in potential_state.keys()):
                                    backbone_state = potential_state
                                    print(f"[*] Detected nested checkpoint format (backbone key)")
                    
                    if backbone_state is None:
                        print(f"[!] Could not find backbone weights in checkpoint")
                        print(f"[!] Available keys (first 20 of {len(checkpoint) if isinstance(checkpoint, dict) else 'unknown'}):")
                        if isinstance(checkpoint, dict):
                            for k in list(checkpoint.keys())[:20]:
                                print(f"    - {k}")
                        print(f"[!] Continuing with official DINOv2 hub weights...")
                    else:
                        try:
                            # 使用strict=False允许加载部分权重
                            incompatible_keys = self.backbone.load_state_dict(backbone_state, strict=False)
                            print(f"[✓] Successfully loaded {len(backbone_state)} backbone weight keys")
                            if incompatible_keys.missing_keys:
                                print(f"[!] Missing {len(incompatible_keys.missing_keys)} keys (will use random init)")
                            if incompatible_keys.unexpected_keys:
                                print(f"[!] {len(incompatible_keys.unexpected_keys)} unexpected keys (ignored)")
                        except Exception as e:
                            print(f"[!] Error loading backbone state: {e}")
                            print(f"[!] Continuing with partial weights...")
        
        # --- 冻结浅层特征，防止预训练知识崩塌 ---
        # 对于新数据集 (group_4)，冻结的层数改为更激进的方案
        # 冻结 patch_embed 层（保护初始特征提取）
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
            
        # 【改动】对于新数据集，仅冻结前 2 层 Transformer blocks (ViT-Base 共有 12 层)
        # 这样有更多层可以适应新数据集的分布
        # 原来冻结 4 层太保守了，导致模型不能很好地适应新数据
        for i in range(2): 
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
        前向传播（严格对齐 FSRA 原版 ClassBlock）
        
        训练模式:
            - return_f=True: 返回 (logit, feat)
            - return_f=False: 返回 logit
        
        评估模式:
            - 只返回 feat（512维 BN 后特征），用于检索
        """
        feat = self.add_block(x)
        
        if self.training:
            if self.return_f:
                logit = self.classifier(feat)
                return logit, feat
            else:
                logit = self.classifier(feat)
                return logit
        else:
            # 评估模式：只返回 BN 后的特征，不过分类器（FSRA 原版行为）
            return feat

# ==========================================
# 2. 组装网络: 双视图 (卫星 + 无人机) + 热力分组
# ==========================================
class two_view_net(nn.Module):
    def __init__(self, opt, class_num, block=3, return_f=False):
        super(two_view_net, self).__init__()
        self.return_f = return_f
        self.block = block
        
        # 加载 DINOv2 骨干网络（可选加载pretrain_path）
        pretrain_path = getattr(opt, 'pretrain_path', '')
        self.backbone = DINOv2_Backbone(pretrain_path=pretrain_path)
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
        前向传播（严格对齐 FSRA 原版 build_transformer.forward）
        
        Returns:
            训练模式:
                if return_f: (([cls_0, ..., cls_global], [feat_0, ..., feat_global]), ...)
                else: ([cls_0, ..., cls_global], ...)
            
            评估模式:
                (y1, y2) — 每个 y 是 [B, 512, block+1] 的3D张量
        """
        # 提取完整特征 [B, N, D]，其中 N = 1 + num_patches
        x1_features = self.backbone(x1)  # [B, 257, 768]
        x2_features = self.backbone(x2)  # [B, 257, 768]
        
        # 分离 CLS token 和 patch token
        cls_token_1 = x1_features[:, 0, :]  # [B, 768]
        patch_token_1 = x1_features[:, 1:, :]  # [B, 256, 768]
        
        cls_token_2 = x2_features[:, 0, :]  # [B, 768]
        patch_token_2 = x2_features[:, 1:, :]  # [B, 256, 768]
        
        # 【全局特征】CLS token → ClassBlock
        global_output_1 = self.global_classifier(cls_token_1)
        global_output_2 = self.global_classifier(cls_token_2)
        
        # block==1 时只有全局分支
        if self.block == 1:
            if self.training:
                return global_output_1, global_output_2
            else:
                # 评估模式：ClassBlock 返回纯 feat [B, 512]
                # reshape 成 [B, 512, 1] 保持统一的3D格式
                return global_output_1.view(global_output_1.size(0), -1, 1), \
                       global_output_2.view(global_output_2.size(0), -1, 1)
        
        # 【局部特征】按热力分组 patch → [B, 768, block]
        heat_pool_1 = self.get_heatmap_pool(patch_token_1)
        heat_pool_2 = self.get_heatmap_pool(patch_token_2)
        
        # 对每个 block 分支过 ClassBlock
        part_results_1 = []
        part_results_2 = []
        
        for i in range(self.block):
            part_feat_1 = heat_pool_1[:, :, i]  # [B, 768]
            part_feat_2 = heat_pool_2[:, :, i]  # [B, 768]
            classifier_i = getattr(self, f'part_classifier_{i}')
            part_results_1.append(classifier_i(part_feat_1))
            part_results_2.append(classifier_i(part_feat_2))
        
        if self.training:
            # 训练模式：收集 cls logits 和 features
            # 把全局分支追加到末尾（和 FSRA 一致：[part_0, ..., part_block-1, global]）
            y1 = part_results_1 + [global_output_1]
            y2 = part_results_2 + [global_output_2]
            
            if self.return_f:
                # 每个元素是 (logit, feat) 元组
                cls_list_1 = [item[0] for item in y1]
                feat_list_1 = [item[1] for item in y1]
                cls_list_2 = [item[0] for item in y2]
                feat_list_2 = [item[1] for item in y2]
                return (cls_list_1, feat_list_1), (cls_list_2, feat_list_2)
            else:
                # 每个元素是纯 logit
                return y1, y2
        else:
            # ========== 评估模式（FSRA 原版逻辑）==========
            # ClassBlock eval 返回纯 feat [B, 512]
            # 局部分支：stack 成 [B, 512, block]
            part_feats_1 = torch.stack(part_results_1, dim=2)  # [B, 512, block]
            part_feats_2 = torch.stack(part_results_2, dim=2)
            
            # 全局分支：reshape [B, 512] → [B, 512, 1]
            global_feat_1 = global_output_1.view(global_output_1.size(0), -1, 1)
            global_feat_2 = global_output_2.view(global_output_2.size(0), -1, 1)
            
            # 拼接：[B, 512, block+1]（和 FSRA 的 torch.cat([y, tranformer_feature], dim=2) 完全对应）
            y1 = torch.cat([part_feats_1, global_feat_1], dim=2)
            y2 = torch.cat([part_feats_2, global_feat_2], dim=2)
            
            return y1, y2

# ==========================================
# 3. 工厂函数
# ==========================================
def make_model(opt):
    return_f = bool(opt.triplet_loss > 0)
    block = getattr(opt, 'block', 1)
    
    model = two_view_net(opt, opt.nclasses, block=block, return_f=return_f)
    
    return model
