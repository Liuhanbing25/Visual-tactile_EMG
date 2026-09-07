import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os

# ==================== 1. 用户自定义参数 ====================
VIDEO_PATH = "vision1.mp4"       # 原始视频路径
# --- 统计计算区间 (手动选择的区间) ---
START_FRAME = 130                # 计算统计指标的起始帧
END_FRAME = 230                  # 计算统计指标的结束帧

# 物理与算法参数
TAU = 1.5                        # 接触阈值
PIXEL_TO_FORCE = 1.0            
Z_FORCE_SCALE = 10.0            

DEFAULT_RADIUS = 25
SPATIAL_RADIUS_RATIO = 2.5
SPATIAL_ITERATIONS = 10
# =========================================================

# ---------- 核心检测与处理函数 (完整保留) ----------

def detect_circles_with_area(img, enable_repair=False):
    """检测标记点并提取面积"""
    if len(img.shape) == 3:
        _, _, gray = cv2.split(img) # 使用单通道提高对比度
    else: gray = img
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(binary, kernel, iterations=3)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 20000: continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            pts.append([M["m10"]/M["m00"], M["m01"]/M["m00"], area])
            
    res = np.array(pts, dtype=np.float32)
    if enable_repair and len(res) > 10:
        res = repair_grid_holes(res, img.shape)
    return res if len(res) > 0 else np.empty((0, 3))

def repair_grid_holes(data, img_shape):
    """自动修复/补全缺失的网格点"""
    if len(data) == 0: return data
    pts = data[:, :2]; tree = cKDTree(pts)
    dist = np.median(tree.query(pts, k=2)[0][:, 1])
    area_m = np.median(data[:, 2])
    dirs = np.array([[dist, 0], [-dist, 0], [0, dist], [0, -dist]])
    n_list = data.tolist()
    
    for _ in range(2):
        c_pts = np.array([p[:2] for p in n_list]); tree = cKDTree(c_pts)
        cands = []
        for p in c_pts:
            for d in dirs:
                tp = p + d
                if 20 < tp[0] < img_shape[1]-20 and 20 < tp[1] < img_shape[0]-20:
                    if tree.query(tp, k=1)[0] > dist * 0.6: 
                        cands.append(tp)
        for c in cands:
            if not any(np.linalg.norm(c - np.array(e[:2])) < dist * 0.5 for e in n_list):
                n_list.append([c[0], c[1], area_m])
    return np.array(n_list, dtype=np.float32)

def match_circles(c1, c2, max_dist=DEFAULT_RADIUS*2.0):
    """参考点与当前点匹配"""
    if len(c1) == 0 or len(c2) == 0: return []
    matches = []; used = set()
    for i, p in enumerate(c1):
        ds = np.linalg.norm(c2[:, :2] - p[:2], axis=1)
        j = int(np.argmin(ds))
        if ds[j] < max_dist and j not in used:
            matches.append((i, j)); used.add(j)
    return matches

def smooth_3d(pts, dx, dy, dz, radius, target_idx):
    """空间平滑处理，减少噪声"""
    tree = cKDTree(pts); neighbors = tree.query_ball_point(pts, r=radius)
    cx, cy, cz = dx.copy(), dy.copy(), dz.copy()
    for _ in range(SPATIAL_ITERATIONS):
        nx, ny, nz = cx.copy(), cy.copy(), cz.copy()
        for i in target_idx:
            nb = neighbors[i]
            if nb: nx[i], ny[i], nz[i] = np.mean(cx[nb]), np.mean(cy[nb]), np.mean(cz[nb])
        cx, cy, cz = nx, ny, nz
    return cx, cy, cz

# ---------- 2. 分析主程序 ----------

def run_csi_cv_analysis():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print("Video error: 无法打开视频文件"); return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化参考帧（通常是视频前几帧的均值）
    print("正在初始化参考帧...")
    frames = []
    for _ in range(int(fps)):
        ret, f = cap.read()
        if ret: frames.append(f)
    if not frames: print("错误：无法读取视频帧"); return
    
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    ref_data = detect_circles_with_area(median_frame, enable_repair=True)
    ref_coords = ref_data[:, :2]
    grid_sp = np.median(cKDTree(ref_coords).query(ref_coords, k=2)[0][:, 1])

    d_t_series = [] # 存储全视频每一帧的平均变形量
    
    print(f"正在处理全视频 ({total_frames} 帧)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        curr_data = detect_circles_with_area(frame, False)
        matches = match_circles(ref_data, curr_data)

        if matches:
            m_ref = [m[0] for m in matches]; m_curr = [m[1] for m in matches]
            dx_k = curr_data[m_curr, 0] - ref_data[m_ref, 0]
            dy_k = curr_data[m_curr, 1] - ref_data[m_ref, 1]
            dz_k = np.maximum(0, (curr_data[m_curr, 2] / np.maximum(ref_data[m_ref, 2], 1.0)) - 1.0)

            fx, fy, fz = np.zeros(len(ref_data)), np.zeros(len(ref_data)), np.zeros(len(ref_data))
            fx[m_ref], fy[m_ref], fz[m_ref] = dx_k, dy_k, dz_k
            
            missing = list(set(range(len(ref_data))) - set(m_ref))
            if missing and len(m_ref) > 3:
                for field in [fx, fy, fz]:
                    field[missing] = griddata(ref_coords[m_ref], field[m_ref], ref_coords[missing], method='nearest')
            
            fx, fy, fz = smooth_3d(ref_coords, fx, fy, fz, grid_sp*SPATIAL_RADIUS_RATIO, missing)
            mags = np.sqrt((fx*PIXEL_TO_FORCE)**2 + (fy*PIXEL_TO_FORCE)**2 + (fz*Z_FORCE_SCALE)**2)
            
            S_indices = np.where(mags > TAU)[0]
            d_t_series.append(np.mean(mags[S_indices]) if len(S_indices) > 0 else 0.0)
        else:
            d_t_series.append(0.0)

    cap.release()
    d_t_series = np.array(d_t_series)

    # --- 计算手动选择区间的统计指标 ---
    s_idx = max(0, START_FRAME)
    e_idx = min(len(d_t_series), END_FRAME)
    analysis_data = d_t_series[s_idx : e_idx]
    active_in_zone = analysis_data[analysis_data > 0]

    if len(active_in_zone) > 0:
        mean_d = np.mean(active_in_zone)
        std_d = np.std(active_in_zone)
        cv_val = std_d / mean_d if mean_d != 0 else 0 
        
        print("\n" + "="*35)
        print(f"WINDOW RESULTS (Frames: {s_idx}-{e_idx})")
        print(f"Mean: {mean_d:.4f}")
        print(f"Std:  {std_d:.4f}")
        print(f"CV:   {cv_val:.4f}")
        print("="*35)

        # --- 绘图显示 (白色背景 + 大字体) ---
        with plt.style.context('seaborn-v0_8-whitegrid'): 
            plt.figure(figsize=(14, 7))
            
            # 字体大小配置
            TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 28, 26, 24, 24

            # 1. 绘制全过程曲线 (背景浅灰色)
            plt.plot(d_t_series, color='lightgray', alpha=0.7, label='Full Video Data')
            
            # 2. 高亮显示分析窗口
            plt.plot(range(s_idx, e_idx), analysis_data, color='tab:blue', lw=3, label='Stability Analysis Window')
            
            # 3. 绘制区间均值线 (Mean)
            plt.hlines(y=mean_d, xmin=s_idx, xmax=e_idx, color='red', linestyle='--', 
           lw=2, label=f'Mean: {mean_d:.2f}')
            # plt.axhline(y=mean_d, color='red', linestyle='--', label=f'Mean: {mean_d:.2f}') 
            
            # 4. 绘制区间标准差阴影 (Std)
            plt.fill_between(range(s_idx, e_idx), 
                             mean_d - std_d, 
                             mean_d + std_d, 
                             color='red', alpha=0.15, 
                             label=f'Std: {std_d:.2f}')
            
            # 5. 绘制窗口边界辅助垂线
            plt.axvline(x=s_idx, color='gray', linestyle=':', alpha=0.6)
            plt.axvline(x=e_idx, color='gray', linestyle=':', alpha=0.6)

            # 设置标题、标签和刻度
            plt.title(f"Stability Analysis (Window CSI = {cv_val:.4f})", fontsize=TITLE_SIZE, pad=25, color='black')
            plt.xlabel("Frame Index", fontsize=LABEL_SIZE, labelpad=15)
            plt.ylabel("Magnitude", fontsize=LABEL_SIZE, labelpad=15)
            
            plt.xticks(fontsize=TICKS_SIZE)
            plt.yticks(fontsize=TICKS_SIZE)
            plt.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)
            
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            
            # 先保存，后显示
            plt.savefig('Final_Stability_Analysis_Report.png', dpi=300)
            plt.show()
    else:
        print("错误：手动选择的区间内没有检测到有效的接触数据。")

if __name__ == "__main__":
    run_csi_cv_analysis()
