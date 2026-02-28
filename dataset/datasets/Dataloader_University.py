import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class Dataloader_University(Dataset):
    def __init__(self, csv_file, transforms,mode):
        super(Dataloader_University, self).__init__()
        self.mode=mode
        if self.mode=='train':
            self.transforms_drone = transforms['train']
            self.transforms_satellite = transforms['satellite']
        elif self.mode == 'val':
            # 验证集通常统一使用 val 的 transform（只做 Resize 和 Normalize）
            self.transforms_drone = transforms['val']
            self.transforms_satellite = transforms['val']
        # 1. 直接读取你的 CSV 文件（Pandas 会自动识别表头 drone_img, sate_img...）
        df = pd.read_csv(csv_file)
        
        # 2. 核心分组逻辑（适配一对一，以及未来的一对多）
        # 我们用 sate_path (卫星图绝对路径) 作为"地点"的唯一标识
        # 把同一个 sate_path 对应的所有 drone_path 收集成一个列表
        self.locations = df.groupby('sate_path')['drone_path'].apply(list).to_dict()
        
        # 3. 把所有的卫星图路径提出来，存成一个列表，用于索引
        self.sate_paths = list(self.locations.keys())

    def __len__(self):
        # 数据集的大小就是独立卫星图（地点）的数量
        return len(self.sate_paths)

    def __getitem__(self, index):
        # 1. 拿到当前 index 对应的卫星图绝对路径
        sate_path = self.sate_paths[index]
        
        # 2. 拿到这个卫星图对应的所有无人机图路径列表
        drone_path_list = self.locations[sate_path]
        
        # 3. 从列表中随机抽一张无人机图的路径
        # (如果你现在是一对一，这个列表里就只有1个路径，抽出来就是它本身)
        drone_path = np.random.choice(drone_path_list)

        # 4. 直接读图！(不再需要 root_dir 和 os.path.join)
        img_satellite = Image.open(sate_path).convert('RGB')
        img_drone = Image.open(drone_path).convert('RGB')

        # 5. 数据增强
        if self.transforms_satellite:
            img_satellite = self.transforms_satellite(img_satellite)
        if self.transforms_drone:
            img_drone = self.transforms_drone(img_drone)

        return img_satellite, img_drone, index


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