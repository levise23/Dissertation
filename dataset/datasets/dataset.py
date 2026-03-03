from torchvision import transforms
from .Dataloader_University import Sampler_University, Dataloader_University, train_collate_fn
from .random_erasing import RandomErasing
from .autoaugment import ImageNetPolicy, CIFAR10Policy
import torch

def make_dataset(opt):
    ######################################################################
    # Load Data
    # ---------
    #
    
    # [新增] 动态 stride 配置：从 opt 读取，如果没设置默认为 1（不抽样）
    train_stride = getattr(opt, 'train_stride', 1) 
    # [新增] 初始偏移
    train_offset = getattr(opt, 'train_offset', 0)

    # ... transforms code ...
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10), # 模拟偏航角抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,
                                                       hue=0.1)] + transform_train_list
        transform_satellite_list = [transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,
                                                           hue=0.1)] + transform_satellite_list

    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}

    train_all = ''
    if opt.train_all:
        train_all = '_all'

    train_datasets = Dataloader_University(opt.train_csv_path, transforms=data_transforms, mode='train', 
                                          stride=train_stride, offset=train_offset)
    # [Old Logic Disabled]
    # train_datasets.image_list = train_datasets.image_list[::4] 
    
    train_samper = Sampler_University(train_datasets, batchsize=opt.batchsize, sample_num=opt.sample_num)
    train_dataloader =torch.utils.data.DataLoader(train_datasets, 
                                             batch_size=opt.batchsize,
                                             sampler=train_samper,num_workers=opt.num_worker, 
                                             pin_memory=True,
                                             drop_last=True,
                                             collate_fn=train_collate_fn,
                                             persistent_workers=True if opt.num_worker > 0 else False,
                                             prefetch_factor=2 if opt.num_worker > 0 else None)
    val_dataset = Dataloader_University(
        csv_file=opt.val_csv_path,  # 比如 'val.csv'
        transforms=data_transforms, 
        mode='val',                  # 开启验证模式，关闭随机增强
        stride=1, offset=0           # 验证集不进行循环抽样，保持全量
    )
    # 验证集不需要那个复杂的 Sampler，直接按顺序读，甚至不需要打乱 (shuffle=False)
    # 【关键】验证时必须使用 num_workers=0 保证顺序！多 worker 会导致 query-gallery 对应错位
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=opt.batchsize,
        shuffle=False,              # 验证集不需要打乱
        num_workers=0,              # 【重要】验证集强制单进程，保证数据顺序一致
        pin_memory=True,
        drop_last=False,            # 验证集一滴都不能少，不能丢弃最后的零星数据
        collate_fn=train_collate_fn)  # num_workers=0 时不能设置 prefetch_factor

    # 把两个 loader 打包返回
    dataloaders = {'train': train_dataloader, 'val': val_loader}
    # 1. 获取类别名（即所有的唯一地点，使用训练集的地点即可）
    class_names = train_datasets.sate_paths
    
    # 2. 计算每个 Epoch 的真实数据量
    # 训练集因为有 Sampler 的 repeat，所以真实跑的数据量是 地点数 * sample_num
    dataset_sizes = {
        'satellite': len(train_datasets) * opt.sample_num, 
        'drone': len(train_datasets) * opt.sample_num
    }
    
    # 3. 严格返回三个变量，与 main 函数完美对接
    return dataloaders, class_names, dataset_sizes
