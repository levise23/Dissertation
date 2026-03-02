import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model, opt):
    """
    ViT/DINOv2 专用 AdamW 优化器构建器（4-param-group 高效版本）。

    为什么 AdamW 而非 SGD？
      ViT 的 Self-Attention + LayerNorm 使各参数梯度高度各向异性，SGD 全局 LR
      无法适应 → 收敛极慢。AdamW 自适应二阶矩 + 解耦 weight_decay，是 ViT/DINOv2
      的官方标准优化器（原版 DINOv2 / MAE / TransReID 全部用 AdamW）。

    参数分组策略（4组）：
      ① backbone  + decay：骨干 0.1×lr，weight_decay=0.01
      ② backbone  + no_decay：骨干 0.1×lr，weight_decay=0（bias/norm）
      ③ head      + decay：头部 lr，   weight_decay=0.01
      ④ head      + no_decay：头部 lr，   weight_decay=0（bias/norm）

    推荐 lr 范围（命令行 --lr）：
      5e-4 ~ 1e-3  → 骨干 5e-5 ~ 1e-4，头部 5e-4 ~ 1e-3
      （比 SGD 的 lr 小 1~2 个数量级，防止洗掉预训练权重）
    """
    def is_no_decay(name):
        return (
            "bias" in name
            or "LayerNorm" in name
            or "layer_norm" in name
            or name.endswith(".norm.weight")
            or name.endswith(".norm1.weight")
            or name.endswith(".norm2.weight")
        )

    backbone_decay, backbone_no_decay = [], []
    head_decay,     head_no_decay     = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            (backbone_no_decay if is_no_decay(name) else backbone_decay).append(param)
        else:
            (head_no_decay if is_no_decay(name) else head_decay).append(param)

    param_groups = [
        {"params": backbone_decay,    "lr": opt.lr * 0.1, "weight_decay": 0.01},
        {"params": backbone_no_decay, "lr": opt.lr * 0.1, "weight_decay": 0.0},
        # 【修复】分类头 weight_decay 从 0.01 降到 0.0005
        # 避免 AdamW 把 classifier.weight 压成近零向量（logits 趋均匀→CE≈随机基线）
        {"params": head_decay,        "lr": opt.lr,       "weight_decay": 0.0005},
        {"params": head_no_decay,     "lr": opt.lr,       "weight_decay": 0.0},
    ]
    # 过滤掉空组，避免 AdamW 对空 param_groups 报 warning
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer_ft = optim.AdamW(param_groups)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=opt.steps, gamma=0.1
    )

    return optimizer_ft, exp_lr_scheduler
