import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image

# ----------------- 路径配置 -----------------
QUERY_FEAT_PATH = "query_features.pt"
GALLERY_FEAT_PATH = "gallery_features.pt"
CSV_PATH = "val_pairs.csv"

# ----------------- 数据加载与缓存 -----------------
@st.cache_resource
def load_pt(path):
    if not os.path.exists(path): return None
    return torch.load(path, map_location='cpu')

@st.cache_data
def load_csv(path):
    if not os.path.exists(path): return None
    return pd.read_csv(path)

@st.cache_data
def compute_global_metrics(_q_feats, _g_feats):
    dist_matrix = 1 - torch.mm(_q_feats, _g_feats.t()).cpu().numpy()
    num_query, num_gallery = dist_matrix.shape
    labels = np.arange(num_query) 
    
    cmc = np.zeros(num_gallery)
    all_precision = 0.0

    for q_idx in range(num_query):
        q_label = labels[q_idx]
        sorted_indices = np.argsort(dist_matrix[q_idx])
        matches = (sorted_indices == q_label).astype(float)
        if matches.sum() == 0: continue
            
        cmc_curve = np.cumsum(matches)
        cmc_curve[cmc_curve > 1] = 1
        cmc += cmc_curve
        
        cum_matches = np.cumsum(matches)
        precision = cum_matches / (np.arange(num_gallery) + 1)
        all_precision += np.sum(precision * matches)
        
    cmc = cmc / num_query
    mAP = all_precision / num_query
    return dist_matrix, cmc, mAP

# ----------------- 页面初始化 -----------------
st.set_page_config(page_title="特征检索验证", layout="wide")
st.title("🚁 无人机-卫星特征极速检索 Demo")

# 检查文件
for path in [QUERY_FEAT_PATH, GALLERY_FEAT_PATH, CSV_PATH]:
    if not os.path.exists(path):
        st.error(f"文件未找到，请检查路径: `{path}`")
        st.stop()

query_features = load_pt(QUERY_FEAT_PATH)
gallery_features = load_pt(GALLERY_FEAT_PATH)
df = load_csv(CSV_PATH)
if query_features is None or gallery_features is None or df is None: st.stop()

with st.spinner("加载全局特征距离矩阵中..."):
    dist_matrix, global_cmc, global_mAP = compute_global_metrics(query_features, gallery_features)

num_samples = query_features.shape[0]
# ----------------- 状态管理与顶部导航 -----------------
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0

num_samples = query_features.shape[0]

# 紧凑型顶部导航
col_l, col_r, col_space, col_info = st.columns([1, 1, 4, 3])
with col_l:
    if st.button("⬅️ 上一张", use_container_width=True):
        st.session_state.sample_idx = max(0, st.session_state.sample_idx - 1)
with col_r:
    if st.button("下一张 ➡️", use_container_width=True):
        st.session_state.sample_idx = min(num_samples - 1, st.session_state.sample_idx + 1)
with col_info:
    st.markdown(f"<h4 style='text-align: right; margin-top: 5px;'>进度: {st.session_state.sample_idx + 1} / {num_samples}</h4>", unsafe_allow_html=True)

st.markdown("---", unsafe_allow_html=True)

# ----------------- 数据读取与比对 -----------------
sample_idx = st.session_state.sample_idx
row = df.iloc[sample_idx]
uav_img_path = row.get("drone_path", "")
gt_sat_path = row.get("sate_path", "")

current_dists = dist_matrix[sample_idx]
topk_indices = np.argsort(current_dists)[:10]

is_r1 = (topk_indices[0] == sample_idx)
is_r10 = (sample_idx in topk_indices)

# ================= 终极紧凑排版：左右 3:7 分栏 =================
left_panel, right_panel = st.columns([3, 7])

# --- 左侧面板：Query、状态、真值 ---
with left_panel:
    st.markdown("### 📥 输入与真值")
    
    # 强行限制无人机图片高度/宽度，避免撑爆屏幕
    if os.path.exists(uav_img_path):
        st.image(Image.open(uav_img_path), caption="UAV Query", use_container_width=True)
    else:
        st.error(f"UAV 图缺失: {uav_img_path}")

    # 紧凑的状态提示
    if is_r1:
        st.success("🎉 R@1 完美命中！")
    elif is_r10:
        rank = np.where(topk_indices == sample_idx)[0][0] + 1
        st.warning(f"⚠️ 命中 Top-10 (排第 {rank})")
    else:
        st.error("❌ 未命中 Top-10")

    if is_r10 and os.path.exists(gt_sat_path):
        st.image(Image.open(gt_sat_path), caption=f"真值 (Dist: {current_dists[sample_idx]:.3f})", use_container_width=True)

# --- 右侧面板：Top-10 画廊矩阵 ---
with right_panel:
    st.markdown("### 🖼️ Top-10 检索矩阵")
    
    # 5列2行的高密度网格排版
    cols = st.columns(5)
    for i, idx in enumerate(topk_indices):
        col = cols[i % 5]
        with col:
            cand_path = df.iloc[idx].get("sate_path", "")
            is_gt = (idx == sample_idx)
            
            # 极简高亮
            if is_gt:
                st.markdown("🟢 **Hit**")
            else:
                st.markdown(f"**Top-{i+1}**")
                
            if cand_path and os.path.exists(cand_path):
                # 图片会自动适应这 1/5 栏的宽度，非常小巧
                st.image(Image.open(cand_path), use_container_width=True)
            else:
                st.error("图片缺失")
                
            # 用 caption 代替 markdown 显得字更小更紧凑
            st.caption(f"Dist: {current_dists[idx]:.3f}")