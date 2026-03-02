# 配置A: Soft Margin Triplet（基准）
python train_v2.py  --loss_type soft_triplet --num_epochs 100 --name softtriplet_v1 --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/v3_new_dataset_loss_norank_triplet_loss=5_epoch=120/net_080.pth"



# 先跑标准Circle Loss，看能提升多少
python train_v2.py  --loss_type circle --num_epochs 100 --name circle_baseline2  --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/v3_new_dataset_loss_norank_triplet_loss=5_epoch=120/net_080.pth";python train_v2.py --name v4circle_hard --loss_type circle_hard --num_epochs 100 --name circle_with_hardmining --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/v3_new_dataset_loss_norank_triplet_loss=5_epoch=120/net_080.pth"

# 再跑Hard Mining版本，对比效果
python train_v2.py --name v4circle_hard --loss_type circle_hard --num_epochs 100 --name circle_with_hardmining --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/v3_new_dataset_loss_norank_triplet_loss=5_epoch=120/net_080.pth"

python train_v2.py \
    --name "aggressive_full_drones" \
    --use_all_drones True \
    --sample_num 3 \
    --batchsize 192 \
    --triplet_loss 1.5 \
    --circle_gamma 256 \
    --num_epochs 100

python train_v2.py \
    --name "circle_Adamw" \
    --sample_num 2 \
    --batchsize 128 \
    --triplet_loss 0.2 \
    --loss_type "circle" \
    --num_epochs 100 \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/circle_Adamw/net_010.pth"

python train_v2.py \
    --name "soft_triplet" \
    --gpu_ids '0,1' \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 50 