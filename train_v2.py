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
from losses.cal_loss import cal_kl_loss, cal_loss, cal_triplet_loss
from re_ranking import re_ranking
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
    验证函数：使用原始的配对 val_loader，分离 Query 和 Gallery 进行标准 1-to-N 检索评估
    
    Args:
        model: 训练好的模型
        val_loader: 验证集 DataLoader（返回 (data_s, data_d) 配对）
        use_gpu: 是否使用 GPU
        verbose: 是否打印进度条（默认 False 以加快速度）
    
    Returns:
        metrics_dict: 包含 R@1, R@10, mAP 的字典
    """
    model.eval()
    
    # 分开提取 Query (drone) 和 Gallery (satellite) 的特征
    query_features = []
    query_labels = []
    gallery_features = []
    gallery_labels = []
    
    with torch.no_grad():
        # 移除 tqdm 以加快验证速度（除非 verbose=True）
        loader = tqdm(val_loader, desc="Validating") if verbose else val_loader
        
        for data_s, data_d in loader:
            inputs_s, labels_s = data_s
            inputs_d, labels_d = data_d
            
            if use_gpu:
                inputs_s = inputs_s.cuda()
                inputs_d = inputs_d.cuda()
            
            with autocast(enabled=True):
                outputs_s, outputs_d = model(inputs_s, inputs_d)
                
                # 【关键】在评估模式下，model 总是返回特征（由 ClassBlock 保证）
                # 处理返回格式
                if isinstance(outputs_s, (tuple, list)) and len(outputs_s) == 2:
                    if isinstance(outputs_s[0], list) and isinstance(outputs_s[1], list):
                        # 多块格式：([cls_0, cls_1, ...], [feat_0, feat_1, ...])
                        features_s = outputs_s[1]  # [feat_0, feat_1, ..., feat_global]
                        features_d = outputs_d[1]
                    else:
                        # 单块格式：(logit, feat)
                        _, features_s = outputs_s
                        _, features_d = outputs_d
                else:
                    raise ValueError(f"Expected outputs with features in evaluate mode, got: {type(outputs_s)}")
                
                # 从特征列表中提取全局特征（最后一个），或直接使用张量
                features_s = _select_global_feature(features_s)
                features_d = _select_global_feature(features_d)
                
                # 【安全检查】确保特征是张量
                assert isinstance(features_s, torch.Tensor), f"Expected Tensor, got {type(features_s)}"
                
                # 保持特征在 CPU 上，避免显存溢出
                gallery_features.append(features_s.detach().cpu())
                gallery_labels.append(labels_s.cpu())
                
                query_features.append(features_d.detach().cpu())
                query_labels.append(labels_d.cpu())
    
    # 合并所有批次数据
    query_features = torch.cat(query_features, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0)
    
    # 计算距离矩阵留在 CPU 上，避免显存爆炸
    # 先归一化特征
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    
    # # 计算余弦距离矩阵（CPU 上进行）
    dist_matrix = 1 - torch.mm(query_features, gallery_features.t())
    
    # # 计算 CMC 和 mAP
    cmc, mAP = compute_cmc_and_map(dist_matrix, query_labels, gallery_labels)
    # === 之前的代码保持不变 ===
    # 先归一化特征
    # query_features = F.normalize(query_features, p=2, dim=1)
    # gallery_features = F.normalize(gallery_features, p=2, dim=1)
    
    # # --- 新增：使用重排算法 ---
    # # 转换为 numpy 数组
    # q_f_np = query_features.numpy()
    # g_f_np = gallery_features.numpy()
    
    # # 计算重排后的距离矩阵
    # dist_matrix_np = re_ranking(q_f_np, g_f_np, k1=20, k2=6, lambda_value=0.3)
    
    # # 转回 Tensor
    # dist_matrix = torch.from_numpy(dist_matrix_np).to(query_features.device)
    # # ---------------------------
    
    # # 计算 CMC 和 mAP (这部分保持不变)
    # cmc, mAP = compute_cmc_and_map(dist_matrix, query_labels, gallery_labels)
    # 计算 R@k
    metrics = {
        'R@1': float(cmc[0]),
        'R@10': float(cmc[9]) if len(cmc) > 9 else 1.0,
        'mAP': float(mAP)
    }
    
    return metrics

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='1,2', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='test', type=str, help='output model name')
    
    parser.add_argument('--train_csv_path', default='/usr1/home/s125mdg43_07/remote/rebuild_UAV/train_pairs.csv', type=str, help='path to the training csv file')
    parser.add_argument('--val_csv_path', default='/usr1/home/s125mdg43_07/remote/rebuild_UAV/val_pairs.csv', type=str, help='path to the val csv file')

    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', default=True, help='use color jitter in training')
    parser.add_argument('--num_worker', default=6, type=int, help='number of dataloader workers')
    parser.add_argument('--batchsize', default=96, type=int, help='batchsize')
    parser.add_argument('--pad', default=20, type=int, help='padding')
    
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    
    parser.add_argument('--views', default=2, type=int, help='the number of views (satellite & drone)')
    parser.add_argument('--erasing_p', default=0.2, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--DA', default=False, help='use Color Data Augmentation')
    parser.add_argument('--share', action='store_true', default=True, help='share weight between different view')
    
    parser.add_argument('--fp16', action='store_true', default=False, help='use apex fp16 (deprecated)' )
    parser.add_argument('--autocast', action='store_true', default=True, help='use native mix precision')
    
    parser.add_argument('--block', default=3, type=int, help='')
    parser.add_argument('--kl_loss', default=True, help='kl_loss')
    parser.add_argument('--triplet_loss', default=1, type=float, help='')
    
    parser.add_argument('--sample_num', default=2, type=int, help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=80, type=int, help='')
    parser.add_argument('--steps', default=[40, 60, 78], type=int, help='')
    parser.add_argument('--backbone', default="VIT-S", type=str, help='')
    parser.add_argument('--pretrain_path', default="", type=str, help='')
    
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
    triplet_loss = Tripletloss(margin=opt.triplet_loss) 

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
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item() * now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
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
        if epoch == 119 or epoch == 79 or (epoch % 10 == 0 and epoch > 0):
            # 【修复】剥离 DataParallel 的外壳以保证单卡推理能够顺利加载权重
            model_to_save = model.module if hasattr(model, 'module') else model
            save_network(model_to_save, opt.name, epoch)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


if __name__ == '__main__':
    opt = get_parse()
    
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    # 【优化】多卡时自动调整 batch_size 和 num_workers 以加快训练
    num_gpus = len(gpu_ids)
    if num_gpus > 1:
        # 扩大 batch_size：每多一张卡，batch 就翻倍增长（充分利用 GPU 显存）
        opt.batchsize = opt.batchsize * num_gpus
        # 增加 num_workers：更多的进程用来加载数据（避免 GPU 空闲等数据）
        opt.num_worker = min(opt.num_worker * 2, 16)  # 最多 16 个 worker
        print(f"[*] Multi-GPU mode detected: {num_gpus} GPUs")
        print(f"[*] Auto-adjusted batch_size to {opt.batchsize} (原{opt.batchsize // num_gpus}×{num_gpus})")
        print(f"[*] Auto-adjusted num_worker to {opt.num_worker}")

    # 1. 加载数据
    dataloaders_dict, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)
    print(f"[*] Total number of classes (locations): {opt.nclasses}")

    # 2. 初始化模型与优化器
    model = make_model(opt)
    if use_gpu:
        model = model.cuda()

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)
    
    # 【修复】多 GPU 支持：仅当实际有多个 GPU 时才使用 DataParallel
    # 注意：如果只指定 1 个 GPU，不使用 DataParallel（会增加同步开销）
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