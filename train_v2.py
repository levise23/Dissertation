from __future__ import print_function, division
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import time
import os
import warnings
from tqdm import tqdm  # [新增] 引入进度条神器

# 确保你的这些自定义模块路径正确
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from dataset.datasets.dataset import make_dataset
from tool.utils_server import save_network, copyfiles2checkpoints
from losses.triplet_loss import Tripletloss, TripletLoss
from losses.circle_loss_correct import CircleLoss, CircleLossWithHardMining  # 【修正】正确的Circle Loss实现
from losses.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss

warnings.filterwarnings("ignore")
version = torch.__version__


# 【新增】辅助函数：从特征列表中选择全局特征（用于triplet loss）
def _select_global_feature(features):
    """
    从多个部分特征中选择全局特征（最后一个）
    参考FSRA的逻辑
    
    Args:
        features: 可能是张量或列表
            - 如果是张量 [B, D]：直接返回
            - 如果是列表 [feat_0, feat_1, ..., feat_global]：返回最后一个（全局特征）
    
    Returns:
        张量 [B, D]
    """
    if isinstance(features, list) and len(features) > 0:
        return features[-1]  # 返回全局特征（最后一个）
    return features


def compute_cmc_and_map(dist_matrix, query_labels, gallery_labels):
    """
    计算 CMC (Cumulative Matching Characteristic) 和 mAP
    
    Args:
        dist_matrix: [N_query, N_gallery] - 距离矩阵
        query_labels: [N_query] - 查询集标签
        gallery_labels: [N_gallery] - 底库集标签
    
    Returns:
        cmc: CMC 曲线
        mAP: 平均精度
    """
    num_query = dist_matrix.size(0)
    num_gallery = dist_matrix.size(1)
    
    # 初始化（确保设备一致）
    device = dist_matrix.device
    cmc = torch.zeros(num_gallery, dtype=torch.float32, device=device)
    all_precision = 0.0
    num_valid_query = 0
    
    for q_idx in range(num_query):
        # 获取查询样本的真实标签
        q_label = query_labels[q_idx]
        
        # 获取距离排序索引
        sorted_indices = torch.argsort(dist_matrix[q_idx])
        
        # 获取底库标签序列
        g_labels = gallery_labels[sorted_indices]
        
        # 匹配位置（第一个正样本出现的位置）
        matches = (g_labels == q_label).float()
        
        # 如果没有正样本，跳过
        if matches.sum() == 0:
            continue
        
        num_valid_query += 1
        
        # 累积 CMC
        cmc_curve = torch.cumsum(matches, dim=0)
        cmc_curve[cmc_curve > 1] = 1  # 限制在 0-1
        cmc += cmc_curve
        
        # 计算 AP（Average Precision）
        cum_matches = torch.cumsum(matches, dim=0)
        precision = cum_matches / torch.arange(1, num_gallery + 1, dtype=torch.float32, device=device)
        # 【修复】必须除以该 Query 对应的正样本总数
        num_relevant = matches.sum().item()
        if num_relevant > 0:
            all_precision += torch.sum(precision * matches).item() / num_relevant
    
    if num_valid_query > 0:
        cmc = cmc / num_valid_query
        mAP = all_precision / num_valid_query
    else:
        mAP = 0.0
    
    return cmc.cpu().numpy(), mAP


def validate_reid(model, val_loader, use_gpu=True, verbose=False):
    """
    验证函数（严格对齐 FSRA 原版 extract_feature + evaluate_gpu 流程）
    
    FSRA 评估流程：
    1. model.eval() 时 ClassBlock 只返回 BN 后的 512 维特征
    2. two_view_net 返回 [B, 512, block+1] 的 3D 张量
    3. 对 3D 特征做 dim=1 L2-norm × sqrt(block+1)，再 flatten
    4. 用余弦相似度（torch.mm）做 N×N 全局检索
    """
    model.eval()
    
    query_features = []
    query_labels = []
    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        loader = tqdm(val_loader, desc="Validating") if verbose else val_loader
        
        for data_s, data_d in loader:
            inputs_s, labels_s = data_s
            inputs_d, labels_d = data_d
            
            if use_gpu:
                inputs_s = inputs_s.cuda()
                inputs_d = inputs_d.cuda()
            
            with autocast(enabled=True):
                outputs_s, outputs_d = model(inputs_s, inputs_d)
                # eval 模式下 outputs 是 [B, 512, block+1] 的 3D 张量
                
                gallery_features.append(outputs_s.detach().float().cpu())
                gallery_labels.append(labels_s.cpu())
                
                query_features.append(outputs_d.detach().float().cpu())
                query_labels.append(labels_d.cpu())
    
    # 合并所有批次
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)
    
    print(f"[VAL] Queries: {len(query_labels)}, Gallery: {len(gallery_labels)}, "
          f"Feature shape: {query_features.shape}")
    
    # ========== FSRA 原版特征后处理 ==========
    # 参考 test_server.py extract_feature 中的 3D 特征归一化逻辑：
    #   fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(ff.size(-1))
    #   ff = ff.div(fnorm.expand_as(ff))
    #   ff = ff.view(ff.size(0), -1)
    
    def fsra_normalize(ff):
        """FSRA 原版3D特征归一化: L2-norm on dim=1, scale by sqrt(num_parts), flatten"""
        if len(ff.shape) == 3:
            # [B, 512, block+1]
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(ff.size(-1))
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)  # [B, 512*(block+1)]
        else:
            # [B, 512] 单 block 退化情况
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        return ff
    
    query_features = fsra_normalize(query_features)
    gallery_features = fsra_normalize(gallery_features)
    
    # ========== 余弦相似度检索（FSRA 用 torch.mm(gf, qf) 即相似度越大越好）==========
    if use_gpu:
        device = torch.device('cuda:0')
        query_features = query_features.to(device)
        gallery_features = gallery_features.to(device)
    
    # 用距离（1 - similarity）以便 argsort 升序 = 最相似在前
    dist_matrix = 1 - torch.mm(query_features, gallery_features.t())
    dist_matrix = dist_matrix.cpu()
    
    cmc, mAP = compute_cmc_and_map(dist_matrix, query_labels, gallery_labels)
    
    metrics = {
        'R@1': float(cmc[0]),
        'R@10': float(cmc[9]) if len(cmc) > 9 else 1.0,
        'mAP': float(mAP)
    }
    
    return metrics

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='1,2,3', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='test', type=str, help='output model name')
    
    parser.add_argument('--train_csv_path', default='/usr1/home/s125mdg43_07/remote/rebuild_UAV/train_pairs.csv', type=str, help='path to the training csv file')
    parser.add_argument('--val_csv_path', default='/usr1/home/s125mdg43_07/remote/rebuild_UAV_dataset/stride=4/val_pairs.csv', type=str, help='path to the val csv file')

    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', default=True, help='use color jitter in training')
    parser.add_argument('--num_worker', default=6, type=int, help='number of dataloader workers')
    parser.add_argument('--batchsize', default=224, type=int, help='batchsize')
    parser.add_argument('--pad', default=20, type=int, help='padding')
    
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    
    parser.add_argument('--views', default=2, type=int, help='the number of views (satellite & drone)')
    parser.add_argument('--erasing_p', default=0.2, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--DA', default=False, help='use Color Data Augmentation')
    parser.add_argument('--share', action='store_true', default=True, help='share weight between different view')
    
    parser.add_argument('--fp16', action='store_true', default=False, help='use apex fp16 (deprecated)' )
    parser.add_argument('--autocast', action='store_true', default=True, help='use native mix precision')
    
    parser.add_argument('--block', default=3, type=int, help='')
    parser.add_argument('--kl_loss', default=True, help='kl_loss')
    
    # 【新增】Loss 类型选择
    parser.add_argument('--loss_type', default='soft_triplet', type=str, 
                       choices=['soft_triplet', 'circle', 'circle_hard'],
                       help='loss type: soft_triplet (Soft Margin Triplet) or circle (Circle Loss)')
    
    parser.add_argument('--triplet_loss', default=0.2, type=float, help='triplet loss weight')
    parser.add_argument('--triplet_margin', default=0.1, type=float, help='triplet loss margin (distance threshold)')
    parser.add_argument('--hard_factor', default=0.2, type=float, help='hard factor for soft boundary (0=hard margin, >0=soft margin)')
    
    # Circle Loss 特有参数
    parser.add_argument('--circle_margin', default=0.25, type=float, help='Circle Loss margin (m parameter)')
    parser.add_argument('--circle_gamma', default=256, type=int, help='Circle Loss gamma (scale factor, 256 or 128)')
    
    parser.add_argument('--sample_num', default=2, type=int, help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=100, type=int, help='')
    parser.add_argument('--steps', default=[60, 80], type=int, help='')
    parser.add_argument('--backbone', default="VIT-S", type=str, help='')
    parser.add_argument('--pretrain_path', default="", type=str, help='Path to pretrained checkpoint (e.g., net_040.pth). Leave empty to train from scratch with official DINOv2 weights')
    
    # [新增] 动态 stride 配置：支持循环抽样 (Curriculum Learning)
    # 示例：每个 Epoch 换一个 offset，用不同的 subset 训练
    parser.add_argument('--train_stride', default=1, type=int, help='Data subsampling stride (e.g., 4 means use 1/4 data per epoch)')
    parser.add_argument('--train_offset', default=0, type=int, help='Data subsampling offset (start index)')

    # 【新增】验证频率控制：每 N 个 epoch 验证一次，避免频繁计算全量矩阵导致 OOM
    parser.add_argument('--val_freq', default=2, type=int, help='validation frequency (every N epochs)')

    opt = parser.parse_args()
    return opt

def train_model(model, opt, optimizer, scheduler, dataloaders_dict, log_path=None):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()
    
    warm_up = 0.1  
    # 动态获取暖机步数
    warm_iteration = len(dataloaders_dict['train']) * opt.warm_epoch  

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    
    # 【新增】根据 loss_type 参数选择 Loss 函数
    if opt.loss_type == 'soft_triplet':
        triplet_loss = Tripletloss(margin=opt.triplet_margin, hard_factor=opt.hard_factor)
        print(f"[*] Using Soft Margin Triplet Loss (margin={opt.triplet_margin}, hard_factor={opt.hard_factor})")
    elif opt.loss_type == 'circle':
        triplet_loss = CircleLoss(m=opt.circle_margin, gamma=opt.circle_gamma)
        print(f"[*] Using Circle Loss (m={opt.circle_margin}, gamma={opt.circle_gamma})")
    elif opt.loss_type == 'circle_hard':
        triplet_loss = CircleLossWithHardMining(m=opt.circle_margin, gamma=opt.circle_gamma, hard_mining_weight=2.0)
        print(f"[*] Using Circle Loss with Hard Mining (m={opt.circle_margin}, gamma={opt.circle_gamma})")
    else:
        raise ValueError(f"Unknown loss type: {opt.loss_type}") 

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        # ===== 训练阶段 =====
        print("Training Phase")
        print('-' * 30)
        model.train()  # 开启 Dropout 和 BatchNorm 更新

        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects_s = 0.0  
        running_corrects_d = 0.0  
        seen_samples = 0 # 记录当前 phase 跑了多少样本

        # 使用 tqdm 包装 DataLoader 展示进度条
        pbar = tqdm(dataloaders_dict['train'], desc="Training")
        
        for data_s, data_d in pbar:
            inputs_s, labels_s = data_s
            inputs_d, labels_d = data_d

            now_batch_size = inputs_s.size(0)

            # 训练阶段依然用原来的逻辑，drop last 是合理的
            if now_batch_size < opt.batchsize:
                continue
            seen_samples += now_batch_size
            
            if use_gpu:
                inputs_s = inputs_s.cuda()
                inputs_d = inputs_d.cuda()
                labels_s = labels_s.cuda()
                labels_d = labels_d.cuda()

            optimizer.zero_grad()

            with autocast(enabled=opt.autocast):
                outputs_s, outputs_d = model(inputs_s, inputs_d)
                
                # 第1步：先初始化设备和特征（最优先！）
                current_device = inputs_s.device
                features_s, features_d = None, None
                
                # 第2步：解析返回格式
                if isinstance(outputs_s, (list, tuple)) and len(outputs_s) == 2:
                    if isinstance(outputs_s[0], list) and isinstance(outputs_s[1], list):
                        outputs_s_list, features_s = outputs_s
                        outputs_d_list, features_d = outputs_d
                        outputs_s = outputs_s_list
                        outputs_d = outputs_d_list
                    else:
                        outputs_s, features_s = outputs_s
                        outputs_d, features_d = outputs_d
                
                # 第3步：计算分类Loss
                if isinstance(outputs_s, list):
                    cls_loss = torch.tensor(0.0, device=current_device)
                    num_outputs = len(outputs_s)
                    for cls_s, cls_d in zip(outputs_s, outputs_d):
                        cls_loss += cal_loss(cls_s, labels_s, criterion)
                        cls_loss += cal_loss(cls_d, labels_d, criterion)
                    cls_loss = cls_loss / num_outputs
                else:
                    cls_loss = cal_loss(outputs_s, labels_s, criterion) + cal_loss(outputs_d, labels_d, criterion)
                
                # 第4步：计算KL Loss
                if opt.kl_loss:
                    if isinstance(outputs_s, list):
                        kl_loss = torch.tensor(0.0, device=current_device)
                        num_outputs = len(outputs_s)
                        for cls_s, cls_d in zip(outputs_s, outputs_d):
                            kl_loss += cal_kl_loss(cls_s, cls_d, loss_kl)
                        kl_loss = kl_loss / num_outputs
                    else:
                        kl_loss = cal_kl_loss(outputs_s, outputs_d, loss_kl)
                else:
                    kl_loss = torch.tensor(0.0, device=current_device)
                
                # 第5步：计算Triplet Loss
                f_triplet_loss = torch.tensor(0.0, device=current_device)
                if opt.triplet_loss > 0 and features_s is not None:
                    split_num = opt.batchsize // opt.sample_num
                    if isinstance(features_s, list):
                        num_features = len(features_s)
                        for feat_s, feat_d in zip(features_s, features_d):
                            f_triplet_loss += cal_triplet_loss(feat_s, feat_d, labels_s, triplet_loss, split_num)
                        f_triplet_loss = f_triplet_loss / num_features
                    else:
                        f_triplet_loss = cal_triplet_loss(features_s, features_d, labels_s, triplet_loss, split_num)
                
                # 第6步：合并Loss
                loss = cls_loss + kl_loss + f_triplet_loss*opt.triplet_loss
                
                if epoch < opt.warm_epoch:
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

            # 反向传播和更新权重
            if opt.autocast:
                scaler.scale(loss).backward()
                # 新增：解缩放并截断梯度，防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scaler.step(optimizer) 
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # 统计数据
            # 【修复】应该记录加权后的triplet loss，而不是原始值
            # 这样日志中的train_triplet_loss才能反映实际进入loss的权重
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item() * now_batch_size
            running_triplet += (f_triplet_loss * opt.triplet_loss).item() * now_batch_size  # 【修复】乘以权重
            running_kl_loss += kl_loss.item() * now_batch_size

            if isinstance(outputs_s, list):
                preds_s = [torch.max(out.data, 1)[1] for out in outputs_s]
                preds_d = [torch.max(out.data, 1)[1] for out in outputs_d]
                running_corrects_s += sum([float(torch.sum(pred == labels_s.data)) for pred in preds_s]) / len(preds_s)
                running_corrects_d += sum([float(torch.sum(pred == labels_d.data)) for pred in preds_d]) / len(preds_d)
            else:
                _, preds_s = torch.max(outputs_s.data, 1)
                _, preds_d = torch.max(outputs_d.data, 1)
                running_corrects_s += float(torch.sum(preds_s == labels_s.data))
                running_corrects_d += float(torch.sum(preds_d == labels_d.data))

            # 更新进度条显示的 Loss 和 Acc
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc_S': f"{(running_corrects_s/seen_samples):.4f}"})

        # --- 训练阶段结束，计算平均指标 ---
        epoch_cls_loss = running_cls_loss / seen_samples
        epoch_kl_loss = running_kl_loss / seen_samples
        epoch_triplet_loss = running_triplet / seen_samples
        epoch_loss = running_loss / seen_samples
        epoch_acc_s = running_corrects_s / seen_samples
        epoch_acc_d = running_corrects_d / seen_samples

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
        lr_other = optimizer.state_dict()['param_groups'][1]['lr']

        print('=> TRAIN Loss: {:.4f} Cls_Loss:{:.4f} KL:{:.4f} Triplet:{:.4f} Sat_Acc: {:.4f} Drone_Acc: {:.4f} LR:{:.6f}'.format(
            epoch_loss, epoch_cls_loss, epoch_kl_loss, epoch_triplet_loss, epoch_acc_s, epoch_acc_d, lr_backbone))

        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
                    epoch, epoch_loss, epoch_cls_loss, epoch_kl_loss, epoch_triplet_loss,
                    epoch_acc_s, epoch_acc_d, lr_backbone, lr_other))
        
        scheduler.step()

        # ===== 验证阶段 =====
        # 【修复】仅在特定 epoch 进行验证以加速训练和避免 OOM
        if epoch % opt.val_freq == 0 or epoch == num_epochs - 1:
            print("\nValidation Phase (ReID Metrics)")
            print('-' * 30)
            val_metrics = validate_reid(model, dataloaders_dict['val'], use_gpu=use_gpu)
            
            print('=> VAL R@1: {:.4f} R@10: {:.4f} mAP: {:.4f}'.format(
                val_metrics['R@1'], val_metrics['R@10'], val_metrics['mAP']))
            
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(",{:.6f},{:.6f},{:.6f}\n".format(
                        val_metrics['R@1'], val_metrics['R@10'], val_metrics['mAP']))
        else:
            # 非验证 epoch，仅记录占位符
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(",,,\n")  # 空值占位

        # 保存模型
        if epoch == 119 or epoch == 20 or (epoch % 10 == 0 and epoch > 0):
            # 【修复】剥离 DataParallel 的外壳以保证单卡推理能够顺利加载权重
            model_to_save = model.module if hasattr(model, 'module') else model
            save_network(model_to_save, opt.name, epoch)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


if __name__ == '__main__':
    opt = get_parse()

    # [新增] 自动循环抽样 (Cyclic Subsampling)
    # 如果设置了 train_stride > 1 (比如4)，但没有设置 offset
    # 可以在这里根据 epoch 动态设置 opt.train_offset 吗？不行，make_dataset 只能调一次
    # 所以如果你要完全动态，需要在 main loop 里重新 make_dataset，这比较慢
    # 
    # [折中方案]：先保持静态配置，如果想玩动态，需要写个 shell 脚本外层循环调用
    # 或者把 make_dataset 放到 epoch 循环里（不推荐）。
    #
    # 最佳实践：直接在这里打印配置
    if opt.train_stride > 1:
        print(f"[*] Curriculum Learning Mode: Stride={opt.train_stride}, Offset={opt.train_offset}")

    str_ids = opt.gpu_ids.split(',')
    gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # 【修复】多卡时调整 num_workers 以加快数据加载
    # 【重要】不能乘以 num_gpus！DataParallel 下，batch_size 是总数，会被平分
    num_gpus = len(gpu_ids)
    if num_gpus > 1:
        # 增加 num_workers：更多的进程用来加载数据（避免 GPU 空闲等数据）
        opt.num_worker = min(opt.num_worker * 2, 16)  # 最多 16 个 worker
        print(f"[*] Multi-GPU mode detected: {num_gpus} GPUs")
        print(f"[*] Keep batch_size at {opt.batchsize} (will be split across {num_gpus} GPUs)")
        print(f"[*] Auto-adjusted num_worker to {opt.num_worker}")

    # 1. 加载数据
    dataloaders_dict, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)
    print(f"[*] Total number of classes (locations): {opt.nclasses}")

    # 2. 初始化模型与优化器
    print("\n" + "="*70)
    print("【模型初始化】")
    print("="*70)
    print(f"[DEBUG] opt.pretrain_path = '{opt.pretrain_path}'")
    if opt.pretrain_path and len(opt.pretrain_path) > 0:
        if os.path.exists(opt.pretrain_path):
            print(f"[✓] Pretrain checkpoint found: {opt.pretrain_path}")
        else:
            print(f"[!] WARNING: Pretrain checkpoint NOT found: {opt.pretrain_path}")
            print(f"[!] Will train from scratch with official DINOv2 weights")
    else:
        print(f"[*] No pretrain checkpoint specified (--pretrain_path is empty)")
        print(f"[*] Training from scratch with official DINOv2 weights")
    print("="*70 + "\n")
    
    model = make_model(opt)
    
    # 【修复重点】：真正把权重吃进模型里！
    if opt.pretrain_path and len(opt.pretrain_path) > 0:
        if os.path.exists(opt.pretrain_path):
            print(f"[✓] Loading weights from: {opt.pretrain_path}")
            try:
                # 加载权重字典
                state_dict = torch.load(opt.pretrain_path, map_location='cpu')
                
                # 兼容 DataParallel 保存的 'module.' 前缀
                if isinstance(state_dict, dict) and 'net_dict' in state_dict:
                    state_dict = state_dict['net_dict']
                
                # 清理 module 前缀（DataParallel 会加这个前缀）
                clean_state_dict = {}
                for k, v in state_dict.items():
                    clean_key = k.replace('module.', '') if k.startswith('module.') else k
                    clean_state_dict[clean_key] = v
                
                # 【修正】过滤策略：
                # 1. 跳过所有 classifier 权重（因为 offset 变化后 class_id → 物理位置映射已改变，
                #    强行加载旧 classifier 会在前几个 epoch 产生语义混乱）
                # 2. 跳过形状不匹配的权重（兼容 nclasses 不一致的情况）
                CLASSIFIER_KEYS = ('classifier', 'part_classifier', 'global_classifier')
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                skipped_classifier = []
                skipped_shape = []
                for k, v in clean_state_dict.items():
                    # 跳过所有 classifier 相关层
                    if any(ck in k for ck in CLASSIFIER_KEYS):
                        skipped_classifier.append(k)
                        continue
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            filtered_state_dict[k] = v
                        else:
                            skipped_shape.append(f"{k}: {v.shape} vs {model_state_dict[k].shape}")
                if skipped_classifier:
                    print(f"    [*] Skipped {len(skipped_classifier)} classifier layer(s) (class-ID remapping)")
                if skipped_shape:
                    for s in skipped_shape:
                        print(f"    [!] Shape mismatch skipped: {s}")

                # 加载权重，strict=False 允许大小不匹配（比如新增层）
                incompatible = model.load_state_dict(filtered_state_dict, strict=False)
                
                print(f"[✓] Pretrained weights loaded successfully (filtered)!")
                if incompatible.missing_keys:
                    print(f"    Warning: {len(incompatible.missing_keys)} keys not found in checkpoint")
                if incompatible.unexpected_keys:
                    print(f"    Warning: {len(incompatible.unexpected_keys)} unexpected keys in checkpoint")
            except Exception as e:
                print(f"[!] ERROR loading checkpoint: {e}")
                print(f"[!] Training from scratch instead...")
        else:
            print(f"[!] WARNING: Pretrain checkpoint NOT found: {opt.pretrain_path}")
            print(f"[!] Training from scratch with official DINOv2 weights")
    
    if use_gpu:
        model = model.cuda()

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)
    
    # 【修复】多 GPU 支持：仅当实际有多个 GPU 时才使用 DataParallel
    # 注意：DataParallel 的 batch 会被平分到各卡，所以不需要乘以 num_gpus
    if use_gpu and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[*] Using DataParallel with GPUs: {gpu_ids}")
    else:
        print(f"[*] Using single GPU: {gpu_ids[0] if gpu_ids else 'CPU'}")

    # 3. 日志准备
    log_path = os.path.join("./checkpoints", opt.name, "train_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.isfile(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_cls_loss,train_kl_loss,train_triplet_loss,train_acc_sat,train_acc_drone,lr_backbone,lr_other,val_r@1,val_r@10,val_mAP\n")
    
    print(f"[*] Using {len(gpu_ids)} GPU(s): {gpu_ids}")
            
    # 4. 启动训练
    model = train_model(
        model=model, 
        opt=opt, 
        optimizer=optimizer_ft, 
        scheduler=exp_lr_scheduler, 
        dataloaders_dict=dataloaders_dict,
        log_path=log_path
    )

    # 强制退出：避免多进程 DataLoader worker 的 atexit 清理卡住进程
    os._exit(0)