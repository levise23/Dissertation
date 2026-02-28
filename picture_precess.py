import os
import pandas as pd
import numpy as np
import math
import cv2
import rasterio
from rasterio.windows import Window
from pathlib import Path
from pyproj import Geod
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- 全局设置 ---
target_area_km2 = 0.2   
output_size = (512, 512) 
satellite_enhance_mode = 'none'
area_random_range = (0.5, 0.7)  
iou_filter_range = (0.1, 1.0)
group_num=1
BaseDir = Path("/usr1/home/s125mdg43_07/remote/UAV/UAV_VisLoc_dataset")
output_dir = Path("/usr1/home/s125mdg43_07/remote/rebuild_UAV")
alpha=0.2
train_output_dir = output_dir / "train"
val_output_dir = output_dir / "val"
test_output_dir = output_dir / "test"

train_query_dir = train_output_dir / "query_drone"
train_gallery_dir = train_output_dir / "gallery_sate"
val_query_dir = val_output_dir / "query_drone"
val_gallery_dir = val_output_dir / "gallery_sate"
test_query_dir = test_output_dir / "query_drone"
test_gallery_dir = test_output_dir / "gallery_sate"
for d in [train_query_dir, train_gallery_dir, val_query_dir, val_gallery_dir, test_query_dir, test_gallery_dir]:
    d.mkdir(parents=True, exist_ok=True)
#处理之前需要先打开好csv
def process_drone_image(src_path, dst_path, offset_angle):
    """处理无人机图片：中心裁剪 + Resize"""
    # 增加文件存在性检查，防止报错
    if not src_path.exists():
        return False
    # if dst_path.exists():
    #     return True
    try:
        img = cv2.imread(str(src_path))
        if img is None: return False
        h, w = img.shape[:2]
        if abs(offset_angle) > 1e-3:
            center = (w // 2, h // 2)
            # 这里的角度是逆时针为正，顺时针为负，请根据你 CSV 的定义调整
            M = cv2.getRotationMatrix2D(center, offset_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        min_side = min(h, w)
        start_x = (w - min_side) // 2
        start_y = (h - min_side) // 2
        img_square = img[start_y : start_y + min_side, start_x : start_x + min_side]
        img_final = cv2.resize(img_square, output_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_path), img_final)
        return True
    except Exception as e:
        print(f"处理无人机图片",src_path,"失败: {e}") 
        return False
#这个逻辑不对，给出tif 传入应该裁剪的原始像素数
def get_crop(src, center_x, center_y, side_len):
    # 1. 计算左上角起始坐标
    half_side = side_len // 2
    col_start = center_x - half_side
    row_start = center_y - half_side
    
    # 2. 定义读取窗口
    # 即使 col_start 或 row_start 为负数，Window 也能定义
    window = Window(col_start, row_start, side_len, side_len)
    
    # 3. 读取数据
    # boundless=True: 允许超出 TIF 范围读取，越界部分自动填充 fill_value
    # [1, 2, 3] 代表读取 RGB 通道
    img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    
    # 4. 格式转换
    # Rasterio 读取的是 (C, H, W)，需要转为 (H, W, C)
    img = img.transpose(1, 2, 0)
    
    # 从 RGB 转换为 OpenCV 习惯的 BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img_bgr


# ==========================================
# 核心处理任务 (用于多进程)
# ==========================================
#处理单文件夹内的任务
def process_single_task(dataset_type,folder):
    recorder=[]
    folder_csv = BaseDir / folder / f"{folder}.csv"
    total_sate_csv = BaseDir / "coordinates_with_all_info.csv"
    tif_path=BaseDir / folder / f"satellite{folder}.tif"
    # --- 1. 获取该文件夹对应的卫星图元数据 ---
    try:
        CsvSate = pd.read_csv(total_sate_csv, encoding='utf-8-sig')
        CsvSate.columns = CsvSate.columns.str.strip()
        
        # 修正筛选逻辑：找到 mapname 列等于 'satellite' 且 某列(假设叫 tif_name) 匹配的行
        # 或者如果你是想按条件取某一行的数据：
        sate_row = CsvSate[CsvSate['mapname'] == f"satellite{folder}.tif"].iloc[0] 
        #print(f"成功获取卫星图信息: {sate_row['mapname']}")
    except Exception as e:
        print(f" 读取总表失败或找不到匹配文件夹 {folder}: {e}")
        return []

    # --- 2. 读取当前文件夹的无人机数据表 ---
    try:
        CsvFolder = pd.read_csv(folder_csv, encoding='utf-8-sig')
        CsvFolder.columns = CsvFolder.columns.str.strip()
        #print(f"成功获取Drone_csv信息: {CsvFolder.iloc[0]}")
    except Exception as e:
        print(f" 读取文件夹CSV失败 {folder_csv}: {e}")
        return []
    lon_min,lon_max=sate_row['LT_lon_map'],sate_row['RB_lon_map']
    lat_min,lat_max=sate_row['RB_lat_map'],sate_row['LT_lat_map']
    img_w, img_h = sate_row['width'], sate_row['height']
    geod = Geod(ellps='WGS84')
    _, _, width_m = geod.inv(lon_min, lat_max, lon_max, lat_max)
    _, _, height_m = geod.inv(lon_min, lat_min, lon_min, lat_max)
    res_x, res_y = width_m / img_w, height_m / img_h
    
    target_side_m = math.sqrt(target_area_km2 * 1e6)
    crop_w = int(target_side_m / res_x)
    crop_h = int(target_side_m / res_y)
    side_len = min(crop_w,crop_h)
    print(crop_w,crop_h)


        # --- 3. 核心循环：读取 CsvFolder 的每一行 ---
    # 使用 iterrows()，index 是行索引，row 是这一行的数据对象
    temp_lat,temp_lon=[],[]
    with rasterio.open(tif_path) as src:
        lenth=len(CsvFolder)
        for index, row in CsvFolder.iterrows():
            # 通过列名直接取值
            drone_filename = row['filename']
            d_lat = row['lat']
            temp_lat.append(d_lat)
            d_lon = row['lon']
            temp_lon.append(d_lon)
            phi1 = float(row['Phi1'])
            phi2 = float(row['Phi2'])
            angle1 = phi1 + 360 if phi1 < 0 else phi1
            angle2 = phi2 + 360 if phi2 < 0 else phi2
            diff = angle2 - angle1
            if diff > 180: angle2 -= 360
            elif diff < -180: angle2 += 360
            offset_angle = (angle1 + angle2) / 2.0
            if offset_angle < 0: offset_angle += 360
            if not ((lon_min< d_lon) & (d_lon <lon_max) & (lat_min<d_lat) & (d_lat<lat_max)):
                print("地址越限",drone_filename,d_lat,d_lon)
                continue
            drone_src = BaseDir / folder / "drone" / drone_filename
            drone_dst_name = f"drone_{folder}_{index}.jpg"
            if dataset_type=="train":
                drone_dst = train_query_dir / drone_dst_name
                g_dir=train_gallery_dir
            elif dataset_type=="val":
                drone_dst = val_query_dir / drone_dst_name
                g_dir=val_gallery_dir
            else :
                drone_dst = test_query_dir / drone_dst_name
                g_dir=test_gallery_dir
            
            if not process_drone_image(drone_src, drone_dst,offset_angle):
                print(drone_filename,"not found")
                continue
            # 这里嵌套你的 group 逻辑
            if index%group_num ==0:
                
                sate_name=f"sate{folder}_{index}.jpg"
                save_path = g_dir / sate_name
                avg_lat=np.mean(temp_lat)
                avg_lon=np.mean(temp_lon)

                center_px_x = int((avg_lon - lon_min) / (lon_max - lon_min) * img_w)
                center_px_y = int((lat_max - avg_lat) / (lat_max - lat_min) * img_h)
                if index % 20 == 0:
                    percentage = index / lenth * 100
                    print(f"{percentage:.2f}%  Total: {lenth}")
                rand_range=int(side_len*alpha)
                rand_offset_x = random.randint(-rand_range, rand_range)
                rand_offset_y = random.randint(-rand_range, rand_range)
                img_bgr=get_crop(src,center_px_x+rand_offset_x ,center_px_y+rand_offset_y , side_len)
                img_bgr = cv2.resize(img_bgr, output_size)
                cv2.imwrite(str(save_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                temp_lat,temp_lon=[],[]
            
            recorder.append((drone_filename,sate_name,drone_dst,save_path,d_lat,d_lon,))
            

                

            #     ... 你的逻辑 ...
                
        return recorder
# ==========================================
# 主程序
# ==========================================
train_folders = []   #"01", "02", "03", "04", "05", "08"
val_folders   = ["12"]                       
test_folders  = ["13"] 

if __name__ == "__main__":
    all_train_recorder = []
    for fold in train_folders:

        fold_recorder=process_single_task(dataset_type="train",folder=fold)
        all_train_recorder.extend(fold_recorder)
    if all_train_recorder:
        columns = ['drone_img', 'sate_img', 'drone_path', 'sate_path', 'drone_lat', 'drone_lon']
        df = pd.DataFrame(all_train_recorder, columns=columns)
        output_csv_path = output_dir / "train_pairs.csv"
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 成功保存 CSV 文件至: {output_csv_path}")
        print(f"总计记录条数: {len(df)}")
    else:
        print("⚠️ all_train_recorder 为空，未生成 CSV。")

    all_val_recorder = []
    for fold in val_folders:

        fold_recorder=process_single_task(dataset_type="val",folder=fold)
        all_val_recorder.extend(fold_recorder)
    if all_val_recorder:
        columns = ['drone_img', 'sate_img', 'drone_path', 'sate_path', 'drone_lat', 'drone_lon']
        df = pd.DataFrame(all_val_recorder, columns=columns)
        output_csv_path = output_dir / "val_pairs.csv"
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 成功保存 CSV 文件至: {output_csv_path}")
        print(f"总计记录条数: {len(df)}")
    else:
        print("⚠️ all_val_recorder 为空，未生成 CSV。")

    all_test_recorder = []
    for fold in test_folders:

        fold_recorder=process_single_task(dataset_type="test",folder=fold)
        all_test_recorder.extend(fold_recorder)
    if all_test_recorder:
        columns = ['drone_img', 'sate_img', 'drone_path', 'sate_path', 'drone_lat', 'drone_lon']
        df = pd.DataFrame(all_test_recorder, columns=columns)
        output_csv_path = output_dir / "test_pairs.csv"
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 成功保存 CSV 文件至: {output_csv_path}")
        print(f"总计记录条数: {len(df)}")
    else:
        print("⚠️ all_test_recorder 为空，未生成 CSV。")

    