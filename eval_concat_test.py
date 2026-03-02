"""
快速测试：对比 global-only 特征 vs all-parts-concat 特征的 R@1
使用已保存的检查点快速验证 concatenation fix 是否有效
"""
import sys, os
sys.path.insert(0, '/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation')

import torch
import torch.nn.functional as F
import argparse

from models.model import make_model
from dataset.datasets.dataset import make_dataset

def compute_metrics(query_f, gallery_f, query_labels, gallery_labels):
    """计算 R@1, R@10, mAP"""
    qf = F.normalize(query_f, p=2, dim=1)
    gf = F.normalize(gallery_f, p=2, dim=1)
    
    dist = 1 - torch.mm(qf.cuda(), gf.cuda().t())
    dist = dist.cpu()
    
    n_q = dist.size(0)
    n_g = dist.size(1)
    
    cmc = torch.zeros(n_g)
    ap_sum = 0.0
    valid = 0
    
    for i in range(n_q):
        q_lbl = query_labels[i]
        sorted_idx = torch.argsort(dist[i])
        g_lbls = gallery_labels[sorted_idx]
        matches = (g_lbls == q_lbl).float()
        
        if matches.sum() == 0:
            continue
        valid += 1
        cum = torch.cumsum(matches, 0)
        cum[cum > 1] = 1
        cmc += cum
        
        # AP
        n_rel = matches.sum().item()
        cum_matches = torch.cumsum(matches, 0)
        prec = cum_matches / torch.arange(1, n_g + 1).float()
        ap_sum += (prec * matches).sum().item() / n_rel
    
    cmc /= valid
    mAP = ap_sum / valid
    return float(cmc[0]), float(cmc[9]) if n_g > 9 else 1.0, mAP


def evaluate_checkpoint(ckpt_path, gpu_id=0):
    # 构建 opt
    class Opt:
        block = 3
        bn = True
        droprate = 0.5
        stride = 2
        pool = 'avg'
        views = 2
        return_f = True
        circle = True
        triplet_loss = 0.2
        h = 224
        w = 224
        pad = 0
        use_all_drones = False
        sample_num = 1
        erasing_p = 0
        color_jitter = False
        DA = False
        train_all = False
        num_worker = 0
        train_csv_path = '/usr1/home/s125mdg43_07/remote/rebuild_UAV/train_pairs.csv'
        val_csv_path   = '/usr1/home/s125mdg43_07/remote/rebuild_UAV/val_pairs.csv'
        batchsize = 64
        nclasses = 4011
    opt = Opt()

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)

    model = make_model(opt)
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'net_dict' in state_dict:
        state_dict = state_dict['net_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # 加载 val 数据
    dataloaders, _, _ = make_dataset(opt)
    val_loader = dataloaders['val']

    query_features_all = []   # 全 parts concat
    query_features_global = [] # 只取 global (last)
    query_labels = []
    gallery_features_all = []
    gallery_features_global = []
    gallery_labels = []

    from torch.cuda.amp import autocast
    with torch.no_grad():
        for data_s, data_d in val_loader:
            inputs_s, labels_s = data_s
            inputs_d, labels_d = data_d
            inputs_s = inputs_s.to(device)
            inputs_d = inputs_d.to(device)

            with autocast(enabled=True):
                outputs_s, outputs_d = model(inputs_s, inputs_d)

            # 解析格式
            if isinstance(outputs_s[0], list):
                features_s = outputs_s[1]  # list of tensors
                features_d = outputs_d[1]
            else:
                _, features_s = outputs_s
                _, features_d = outputs_d

            if isinstance(features_s, list):
                # concat all parts (raw, no per-part norm)
                # validate_reid 统一对拼接向量做 L2 归一化
                all_s = torch.cat(features_s, dim=1)
                all_d = torch.cat(features_d, dim=1)
                # global only (last)
                glob_s = features_s[-1]
                glob_d = features_d[-1]
            else:
                all_s = glob_s = features_s
                all_d = glob_d = features_d

            gallery_features_all.append(all_s.cpu())
            gallery_features_global.append(glob_s.detach().cpu())
            gallery_labels.append(labels_s.cpu())
            
            query_features_all.append(all_d.cpu())
            query_features_global.append(glob_d.detach().cpu())
            query_labels.append(labels_d.cpu())

    query_labels = torch.cat(query_labels)
    gallery_labels = torch.cat(gallery_labels)

    # 方案1: 只用 global
    qf_g = torch.cat(query_features_global)
    gf_g = torch.cat(gallery_features_global)
    r1_g, r10_g, map_g = compute_metrics(qf_g, gf_g, query_labels, gallery_labels)
    print(f"\n[Global-only  512-dim]  R@1={r1_g:.4f}  R@10={r10_g:.4f}  mAP={map_g:.4f}")

    # 方案2: concat all parts
    qf_a = torch.cat(query_features_all)
    gf_a = torch.cat(gallery_features_all)
    r1_a, r10_a, map_a = compute_metrics(qf_a, gf_a, query_labels, gallery_labels)
    feat_dim = qf_a.shape[1]
    print(f"[All-concat  {feat_dim}-dim]  R@1={r1_a:.4f}  R@10={r10_a:.4f}  mAP={map_a:.4f}")
    
    print(f"\n>>> R@1 提升: {(r1_a - r1_g)*100:.1f}pp ({r1_g:.4f} → {r1_a:.4f})")


if __name__ == '__main__':
    ckpt = '/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/circle_Adamw/net_010.pth'
    print(f"测试检查点: {ckpt}")
    evaluate_checkpoint(ckpt, gpu_id=0)
