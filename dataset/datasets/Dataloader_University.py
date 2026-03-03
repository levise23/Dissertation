import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms

class Dataloader_University(Dataset):
    def __init__(self, csv_file, transforms, mode, stride=1, offset=0):
        super(Dataloader_University, self).__init__()
        self.mode = mode
        self.stride = stride
        self.offset = offset
        
        if self.mode == 'train':
            self.transforms_drone = transforms['train']
            self.transforms_satellite = transforms['satellite']
        elif self.mode == 'val':
            self.transforms_drone = transforms['val']
            self.transforms_satellite = transforms['val']
        
        # 读取CSV
        df = pd.read_csv(csv_file)
        
        # 【所有模式都初始化】位置分组（用于训练集，或备用）
        self.locations = df.groupby('sate_path')['drone_path'].apply(list).to_dict()
        self.sate_paths_grouped = list(self.locations.keys())
        
        # [Cyclic Subsampling 核心逻辑]
        # 对训练集的位置列表进行切片
        if self.mode == 'train' and self.stride > 1:
            self.sate_paths_grouped = self.sate_paths_grouped[self.offset::self.stride]
            print(f"[*] Dataloader Subsampling: Stride={self.stride}, Offset={self.offset}, "
                  f"Subset Size={len(self.sate_paths_grouped)}") # Debug info
        
        if self.mode == 'val':
            # 【修复】验证集：保留CSV行的原始顺序（避免标签-特征错位）
            # 直接用CSV的行号作为index，保证一一对应的对应关系
            self.csv_pairs = []
            for idx, row in df.iterrows():
                self.csv_pairs.append({
                    'drone_path': row['drone_path'],
                    'sate_path': row['sate_path'],
                    'location_id': self.sate_paths_grouped.index(row['sate_path'])
                })
            self.sate_paths = [p['sate_path'] for p in self.csv_pairs]  # 保留原始顺序
        else:
            # 【保持不变】训练集：按位置分组（支持同位置多个drone）
            self.sate_paths = self.sate_paths_grouped  # 去重后的位置列表
            self.csv_pairs = None  # 训练集不用

    def __len__(self):
        return len(self.sate_paths)

    def __getitem__(self, index):
        if self.mode == 'val' and self.csv_pairs is not None:
            # 【修复】验证集：直接从csv_pairs取，保证顺序
            pair = self.csv_pairs[index]
            sate_path = pair['sate_path']
            drone_path = pair['drone_path']
            location_id = pair['location_id']
        else:
            # 【保持不变】训练集：仍然随机选择
            sate_path = self.sate_paths[index]
            drone_path_list = self.locations[sate_path]
            drone_path = np.random.choice(drone_path_list)
            location_id = index
        
        # 读图
        img_satellite = Image.open(sate_path).convert('RGB')
        img_drone = Image.open(drone_path).convert('RGB')
        
        # 数据增强
        if self.transforms_satellite:
            img_satellite = self.transforms_satellite(img_satellite)
        if self.transforms_drone:
            img_drone = self.transforms_drone(img_drone)
        
        return img_satellite, img_drone, location_id


class Sampler_University(torch.utils.data.Sampler):
    def __init__(self, data_source, batchsize=8, sample_num=2):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num
        
        # 强制安全检查：确保 Batch 内的正样本对不会被截断
        assert self.batchsize % self.sample_num == 0, \
            f"Batch size ({self.batchsize}) 必须是 sample_num ({self.sample_num}) 的整数倍!"

    def __iter__(self):
        # 1. 生成基于地点（卫星图数量）的索引并打乱
        indices = np.arange(0, self.data_len)
        np.random.shuffle(indices)
        
        # 2. 扩充索引，保证同一个地点在同一个 Batch 里连续出现 sample_num 次
        nums = np.repeat(indices, self.sample_num, axis=0)
        return iter(nums.tolist())

    def __len__(self):
        return self.data_len * self.sample_num


def train_collate_fn(batch):
    # batch 是 Dataset 返回的元组列表: [(img_s1, img_d1, id1), (img_s2, img_d2, id2), ...]
    img_s, img_d, ids = zip(*batch)
    
    # 转换格式
    ids = torch.tensor(ids, dtype=torch.int64)
    img_s_batch = torch.stack(img_s, dim=0)
    img_d_batch = torch.stack(img_d, dim=0)
    
    # 返回: [卫星图Batch, 对应ID], [无人机图Batch, 对应ID]
    return [img_s_batch, ids], [img_d_batch, ids]