import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os

# ==================== 1. 用户自定义参数 ====================
VIDEO_PATH = "vision1.mp4"       # 输入视频路径
# --- 统计计算区间 (手动选择) ---
START_FRAME = 130                # 分析起始帧
END_FRAME = 230                  # 分析结束帧

# 物理与算法参数
TAU = 1.5                        # 接触阈值 (用于定义集合 S)
PIXEL_TO_FORCE = 1.0            
Z_FORCE_SCALE = 10.0            

DEFAULT_RADIUS = 25
SPATIAL_RADIUS_RATIO = 2.5
SPATIAL_ITERATIONS = 10
# =========================================================

# ---------- 核心检测与处理函数 (保持原逻辑) ----------

def detect_circles_with_area(img, enable_repair=False):
    if len(img.shape) == 3:
        _, _, gray = cv2.split(img)
    else: gray = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray); blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.erode(binary, kernel, iterations=3)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = [[cv2.moments(cnt)["m10"]/cv2.moments(cnt)["m00"], 
            cv2.moments(cnt)["m01"]/cv2.moments(cnt)["m00"], 
            cv2.contourArea(cnt)] for cnt in contours if 2000 < cv2.contourArea(cnt) < 20000]
    res = np.array(pts, dtype=np.float32)
    return res if len(res) > 0 else np.empty((0, 3))

def match_circles(c1, c2, max_dist=DEFAULT_RADIUS*2.0):
    if len(c1) == 0 or len(c2) == 0: return []
    matches, used = [], set()
    for i, p in enumerate(c1):
        ds = np.linalg.norm(c2[:, :2] - p[:2], axis=1)
        j = int(np.argmin(ds)); 
        if ds[j] < max_dist and j not in used:
            matches.append((i, j)); used.add(j)
    return matches

# ---------- 2. CDU (Normalized) 分析主程序 ----------

def run_normalized_cdu_analysis():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print("Video error"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化参考
    print("Initializing reference...")
    frames = []
    for _ in range(int(fps)):
        ret, f = cap.read()
        if ret: frames.append(f)
    ref_data = detect_circles_with_area(np.median(frames, axis=0).astype(np.uint8))
    ref_coords = ref_data[:, :2]
    
    # 存储全视频归一化 CDU
    cdu_series = [] 
    
    print(f"Processing full video ({total_frames} frames)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        curr_data = detect_circles_with_area(frame)
        matches = match_circles(ref_data, curr_data)

        if matches:
            m_ref, m_curr = [m[0] for m in matches], [m[1] for m in matches]
            dx = curr_data[m_curr, 0] - ref_data[m_ref, 0]
            dy = curr_data[m_curr, 1] - ref_data[m_ref, 1]
            dz = np.maximum(0, (curr_data[m_curr, 2] / np.maximum(ref_data[m_ref, 2], 1.0)) - 1.0)
            
            # 计算每个标记点的合力模长 ||vi||
            mags = np.sqrt((dx*PIXEL_TO_FORCE)**2 + (dy*PIXEL_TO_FORCE)**2 + (dz*Z_FORCE_SCALE)**2)
            
            # 识别接触区域 S(t)
            S_mags = mags[mags > TAU]
            
            # --- 核心修改：计算归一化空间变异度 (Spatial CV) ---
            if len(S_mags) > 1:
                # CDU = std / mean (空间分布的离散程度)
                cdu_val = np.std(S_mags) / np.mean(S_mags)
                cdu_series.append(cdu_val)
            else:
                cdu_series.append(0.0)
        else:
            cdu_series.append(0.0)

    cap.release()
    cdu_series = np.array(cdu_series)

    # --- 截取分析窗口指标 ---
    s_idx, e_idx = max(0, START_FRAME), min(len(cdu_series), END_FRAME)
    analysis_data = cdu_series[s_idx : e_idx]
    active_in_zone = analysis_data[analysis_data > 0]

    if len(active_in_zone) > 0:
        mean_cdu_window = np.mean(active_in_zone)
        
        print("\n" + "="*35)
        print(f"NORMALIZED CDU RESULTS (Frames: {s_idx}-{e_idx})")
        print(f"Window Mean CDU: {mean_cdu_window:.4f}")
        print("="*35)

        # --- 绘图显示 (完全对齐 CSI 风格) ---
        with plt.style.context('seaborn-v0_8-whitegrid'): 
            plt.figure(figsize=(14, 7))
            TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 28, 26, 24, 24

            # 1. 全视频背景
            plt.plot(cdu_series, color='lightgray', alpha=0.7, label='Full Video Process')
            
            # 2. 高亮分析区间
            plt.plot(range(s_idx, e_idx), analysis_data, color='tab:blue', lw=3, label='CDU Analysis Window')
            
            # 3. 绘制区间均值线 (Mean CDU) - 使用 hlines 严格对齐
            plt.hlines(y=mean_cdu_window, xmin=s_idx, xmax=e_idx, color='red', linestyle='--', 
                       lw=2.5, label=f'Mean CDU: {mean_cdu_window:.3f}')
            
            # 4. 辅助垂线
            plt.axvline(x=s_idx, color='gray', linestyle=':', alpha=0.6)
            plt.axvline(x=e_idx, color='gray', linestyle=':', alpha=0.6)

            # 设置标题和标签 (去掉方差单位，现在是比例指标)
            plt.title(f"Contact Distribution Uniformity (Window CDU = {mean_cdu_window:.4f})", 
                      fontsize=TITLE_SIZE, pad=25)
            plt.xlabel("Frame Index", fontsize=LABEL_SIZE, labelpad=15)
            plt.ylabel("Spatial Variation (std/mean)", fontsize=LABEL_SIZE, labelpad=15)
            
            plt.xticks(fontsize=TICKS_SIZE)
            plt.yticks(fontsize=TICKS_SIZE)
            plt.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)
            
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            
            plt.savefig('Normalized_CDU_Analysis_Report.png', dpi=300)
            plt.show()
    else:
        print("No active contact detected in selected range.")

if __name__ == "__main__":
    run_normalized_cdu_analysis()
