import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os

# ==================== 1. 用户自定义参数 ====================
VIDEO_PATH = "vision1.mp4"       # 输入视频路径
EMG_PATH = "emg_log_dev15_1.csv" # 输入EMG路径
EMG_OFFSET = 9                 # EMG同步偏移量 (秒)

# 视触觉计算参数 (保持原逻辑不变)
TAU = 1.5                        
PIXEL_TO_FORCE = 1.0            
Z_FORCE_SCALE = 10.0            
DEFAULT_RADIUS = 25
SPATIAL_RADIUS_RATIO = 2.5
SPATIAL_ITERATIONS = 10
# =========================================================

# ---------- 视触觉核心处理函数 (保持不变) ----------
def detect_circles_with_area(img, enable_repair=False):
    if len(img.shape) == 3:
        _, _, gray = cv2.split(img)
    else: gray = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
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
        j = int(np.argmin(ds))
        if ds[j] < max_dist and j not in used:
            matches.append((i, j)); used.add(j)
    return matches

# ---------- 主分析程序 ----------

def run_combined_ete_analysis():
    # 1. 配置加载与检查
    if not os.path.exists(VIDEO_PATH) or not os.path.exists(EMG_PATH):
        print("Error: 找不到文件，请检查路径。"); return

    # 2. 处理视触觉数据
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Initializing tactile reference...")
    frames = []
    for _ in range(int(fps)):
        ret, f = cap.read(); 
        if ret: frames.append(f)
    if not frames: print("错误：无法读取视频帧"); return
    
    ref_data = detect_circles_with_area(np.median(frames, axis=0).astype(np.uint8))
    ref_coords = ref_data[:, :2]
    
    d_t_series = []
    print(f"Processing tactile video ({total_frames} frames)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        curr_data = detect_circles_with_area(frame)
        matches = match_circles(ref_data, curr_data)
        if matches:
            m_ref, m_curr = [m[0] for m in matches], [m[1] for m in matches]
            dx = curr_data[m_curr, 0] - ref_data[m_ref, 0]
            dy = curr_data[m_curr, 1] - ref_data[m_ref, 1]
            dz = np.maximum(0, (curr_data[m_curr, 2] / np.maximum(ref_data[m_ref, 2], 1.0)) - 1.0)
            mags = np.sqrt((dx*PIXEL_TO_FORCE)**2 + (dy*PIXEL_TO_FORCE)**2 + (dz*Z_FORCE_SCALE)**2)
            active = mags[mags > TAU]
            d_t_series.append(np.mean(active) if len(active) > 0 else 0.0)
        else: d_t_series.append(0.0)
    cap.release()
    d_t_series = np.array(d_t_series)
    tactile_time = np.arange(len(d_t_series)) / fps

    # 3. 加载并同步 EMG 数据
    df_emg = pd.read_csv(EMG_PATH, comment='#')
    # 同步：将 EMG 时间转为相对于视频开始的时间
    df_emg['synced_time'] = df_emg['EMG_t_s'] - EMG_OFFSET

    # --- 核心修改：仅保留视频持续区间的数据 ---
    video_duration = tactile_time[-1]
    mask_emg = (df_emg['synced_time'] >= 0) & (df_emg['synced_time'] <= video_duration)
    # 这行代码会剔除前面 8.7s 和视频结束后的数据
    df_emg_sync = df_emg.loc[mask_emg].copy()

    # 4. 寻找窗口 (1s 窗口)
    t_peak = df_emg_sync.loc[df_emg_sync['ch1_env_uV'].idxmax(), 'synced_time']
    w_start, w_end = t_peak - 1, t_peak + 1
    
    # 5. 计算 ETE
    emg_window = df_emg_sync[(df_emg_sync['synced_time'] >= w_start) & (df_emg_sync['synced_time'] <= w_end)]
    mean_emg_w = emg_window['ch1_env_uV'].mean()
    
    tactile_window = d_t_series[(tactile_time >= w_start) & (tactile_time <= w_end)]
    mean_tactile_w = np.mean(tactile_window) if len(tactile_window) > 0 else 0
    
    ete_score = mean_tactile_w / mean_emg_w if mean_emg_w > 0 else 0
    
    print("\n" + "="*35)
    print(f"ETE ANALYSIS RESULTS (Window: 1s)")
    print(f"Peak Time (Synced): {t_peak:.2f} s")
    print(f"Tactile Mean (W):   {mean_tactile_w:.4f}")
    print(f"EMG Mean (W):       {mean_emg_w:.4f}")
    print(f"ETE Score:          {ete_score:.6f}")
    print("="*35)

    # 6. 绘图 (完美对齐专业报告风格)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, ax1 = plt.subplots(figsize=(14, 7))
        TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 28, 24, 26, 26
        
        # 左轴：触觉 D(t)
        ax1.plot(tactile_time, d_t_series, color='tab:blue', lw=2, label='Tactile Magnitude D(t)')
        ax1.set_xlabel('Time Since Video Start (seconds)', fontsize=LABEL_SIZE, labelpad=15)
        ax1.set_ylabel('Tactile Magnitude', color='tab:blue', fontsize=LABEL_SIZE, labelpad=15)
        
        # 开启网格，左轴刻度
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.tick_params(axis='x', labelsize=TICKS_SIZE)
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=TICKS_SIZE)

        # 右轴：EMG
        ax2 = ax1.twinx()
        ax2.plot(df_emg_sync['synced_time'], df_emg_sync['ch1_env_uV'], color='tab:red', lw=1.5, alpha=0.7, label='EMG Envelope (RMS)')
        ax2.set_ylabel('EMG Envelope (uV)', color='tab:red', fontsize=LABEL_SIZE, labelpad=15)
        # 右轴网格关闭
        ax2.grid(False)
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=TICKS_SIZE)
        
        # 阴影标出 ETE 窗口
        plt.axvspan(w_start, w_end, color='yellow', alpha=0.15, label='ETE Window (1s)')
        # 垂线标出峰值时刻
        plt.axvline(x=t_peak, color='black', linestyle=':', alpha=0.4)

        # 标题和图例 (带阴影白底)
        plt.title(f'EMG-Tactile Efficiency Analysis (Window ETE = {ete_score:.6f})', fontsize=TITLE_SIZE, pad=25)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # 注意：Twin axes 需要合并图例才能完整显示
        handles = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(handles, labels, loc='upper right',  fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)

        plt.tight_layout()
        # 先保存，后显示
        plt.savefig('ETE_Combined_Sync_Report.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    run_combined_ete_analysis()
