

# python train_v2.py \
#     --name "circle_Adamw" \
#     --sample_num 2 \
#     --batchsize 128 \
#     --triplet_loss 0.2 \
#     --loss_type "circle" \
#     --num_epochs 100 \
#     --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/circle_Adamw/net_010.pth"

# ===============================================
# 方案 A: 循环子采样 (Cyclic Subsampling)
# Epoch 0-4: 用 Set 1 (Offset=0)
# Epoch 5-9: 用 Set 2 (Offset=1)
# ...
# ===============================================
python train_v2.py \
    --name "curriculum_s4_o0" \
    --gpu_ids '0,1' \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 21 \
    --train_stride 4 \
    --train_offset 0
# 1. 跑 Offset=0
python train_v2.py \
    --name "curriculum_s4_o1" \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o0/net_020.pth" \
    --gpu_ids '0,1' \
    --warm_epoch 0 \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 21 \
    --train_stride 4 \
    --train_offset 1
python train_v2.py \
    --name "curriculum_s4_o2" \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o1/net_020.pth" \
    --gpu_ids '0,1' \
    --warm_epoch 0 \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 21 \
    --train_stride 4 \
    --train_offset 2
python train_v2.py \
    --name "curriculum_s4_o3" \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o2/net_020.pth" \
    --gpu_ids '0,1' \
    --warm_epoch 0 \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 21 \
    --train_stride 4 \
    --train_offset 3


python train_v2.py \
    --name "curriculum_s2_o0" \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o3/net_020.pth" \
    --gpu_ids '0,1' \
    --lr 0.0005 \
    --warm_epoch 0 \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 21 \
    --train_stride 2 \
    --train_offset 0
python train_v2.py \
    --name "curriculum_s2_o1" \
    --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s2_o0/net_020.pth" \
    --gpu_ids '0,1' \
    --lr 0.0005 \
    --warm_epoch 0 \
    --sample_num 2 \
    --triplet_loss 2 \
    --loss_type "soft_triplet" \
    --num_epochs 2 \
    --train_stride 2 \
    --train_offset 1
# 2. 跑 Offset=1 (接着 Offset=0 的权重跑)
# 注意：你需要手动指定上一步生成的 checkpoint 路径
# PREV_CKPT="/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o0/net_019.pth"
# python train_v2.py \
#     --name "curriculum_s4_o1" \
#     --gpu_ids '0,1' \
#     --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o0/net_010.pth" \
#     --sample_num 2 \
#     --triplet_loss 2 \
#     --loss_type "soft_triplet" \
#     --num_epochs 21 \
#     --train_stride 4 \
#     --train_offset 1
# python train_v2.py \
#     --name "curriculum_s4_o1" \
#     --gpu_ids '0,1' \
#     --pretrain_path "/usr1/home/s125mdg43_07/remote/rebuild_UAV_v2/Dissertation/checkpoints/curriculum_s4_o1/net_020.pth" \
#     --sample_num 2 \
#     --triplet_loss 2 \
#     --loss_type "soft_triplet" \
#     --num_epochs 21 \
#     --train_stride 4 \
#     --train_offset 2
# ===============================================
# 方案 B: 静态稀疏训练 (Static Sparse Training)
# 只用 1/4 数据，看能不能更快收敛且泛化更好
# ===============================================
# python train_v2.py \
#    --name "sparse_v1" \
#    --gpu_ids '0,1' \
#    --sample_num 2 \
#    --triplet_loss 2 \
#    --loss_type "soft_triplet" \
#    --num_epochs 50 \
#    --train_stride 4 \
#    --train_offset 0

# ===============================================
# 方案 C: 暴力全量基准 (Full Baseline)
# ===============================================
# python train_v2.py \
#     --name "soft_triplet_full" \
#     --gpu_ids '0,1' \
#     --sample_num 2 \
#     --triplet_loss 2 \
#     --loss_type "soft_triplet" \
#     --num_epochs 50 \
#     --train_stride 1 