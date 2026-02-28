import torch.optim as optim
from torch.optim import lr_scheduler

def make_optimizer(model, opt):
    # 【修复】适配新的模型结构：model.backbone 而非 model.model_1.backbone
    # 直接提取 DINOv2 骨干网络的参数 ID
    ignored_params = list(map(id, model.backbone.parameters()))
    
    # 将参数划分为 extra_params (分类头+瓶颈层) 和 base_params (DINOv2 骨干网络)
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    
    # 组装 SGD 优化器
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},  # DINOv2 必须使用较小的学习率防止特征崩塌
        {'params': extra_params, 'lr': opt.lr}        # 新初始化的分类器使用正常设定的学习率
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # 学习率阶梯衰减
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=opt.steps, gamma=0.1)

    return optimizer_ft, exp_lr_scheduler