# 配置A: Soft Margin Triplet（基准）
python train_v2.py --name v4_soft_triplet --loss_type soft_triplet --num_epochs 120 --name softtriplet_v1



# 先跑标准Circle Loss，看能提升多少
python train_v2.py --name v4_circle --loss_type circle --num_epochs 100 --name circle_baseline

# 再跑Hard Mining版本，对比效果
python train_v2.py --name v4circle_hard --loss_type circle_hard --num_epochs 120 --name circle_with_hardmining