"""
visuotactile_selective_3d_combined.py

功能：视触觉主生成程序 (3D合力版)
- [x] 保持原有 XY 处理逻辑：选择性平滑、网格补全。
- [x] 新增 Z 轴检测：通过标志点面积变化率计算。
- [x] 视觉升级：Heatmap 和 3D 变形展示的是 XYZ 三向合力。
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# ------------------ 参数配置 ------------------
DEFAULT_RADIUS = 25                
PIXEL_TO_FORCE = 1.0               
GRID_RES = 80                     
#OUTPUT_DIR = 'output_dashboard_3d_combined2'
#if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- Z轴受力参数 ---
# 法向力系数：因为面积变化率数值通常较小，为了体现其对合力的贡献，scale 设大一些
Z_FORCE_SCALE = 20.0 

# --- 空间平滑参数 ---
SPATIAL_RADIUS_RATIO = 2.5 
SPATIAL_ITERATIONS = 10
# --------------------

MIN_FORCE_THRESHOLD = 0.005
POINT_SIZE = 10          
FIXED_ARROW_LEN = 200     
ARROW_WIDTH = 0.013      
# -----------------------------------------

# ---------- 1. 核心检测 (增加面积提取) ----------
def detect_circles_with_area(img, enable_repair=False):
    # --- 修改部分：从提取红通道改为转换为灰度图 ---
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # -------------------------------------------

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (9, 9), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_binary = cv2.erode(binary, kernel_erode, iterations=3)

    contours, _ = cv2.findContours(eroded_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_points = []
    min_area = 2000 
    max_area = 20000 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.4: continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            # 返回 [x, y, area]
            detected_points.append([cX, cY, area])

    points_np = np.array(detected_points, dtype=np.float32)
    
    if enable_repair and len(points_np) > 10:
        points_np = repair_grid_holes_with_area(points_np, img.shape)
        
    return points_np

def repair_grid_holes_with_area(data, img_shape):
    if len(data) == 0: return data
    points = data[:, :2]
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2) 
    grid_spacing = np.median(dists[:, 1])
    if grid_spacing < 10: return data

    median_area = np.median(data[:, 2])
    directions = np.array([[grid_spacing, 0], [-grid_spacing, 0], [0, grid_spacing], [0, -grid_spacing]])
    new_points_list = data.tolist()
    
    for _ in range(3): 
        current_points = np.array([p[:2] for p in new_points_list])
        tree = cKDTree(current_points)
        candidates = []
        for p in current_points:
            for d in directions:
                target_pos = p + d
                x, y = target_pos
                if x < 20 or x > img_shape[1]-20 or y < 20 or y > img_shape[0]-20: continue
                dist, _ = tree.query(target_pos, k=1)
                if dist > grid_spacing * 0.5: candidates.append(target_pos)
        if not candidates: break
        for cand in candidates:
            is_dup = False
            for exist in new_points_list:
                if np.linalg.norm(cand - np.array(exist[:2])) < grid_spacing * 0.5: is_dup=True; break
            if not is_dup: new_points_list.append([cand[0], cand[1], median_area])

    return np.array(new_points_list, dtype=np.float32)

# ---------- 2. 计算位移 & 选择性平滑 (同步处理 Z) ----------

def apply_selective_smoothing_3d(points, dx, dy, dz, radius, target_indices, iterations=1):
    if len(target_indices) == 0:
        return dx, dy, dz
        
    tree = cKDTree(points)
    indices_list = tree.query_ball_point(points, r=radius)
    
    curr_dx, curr_dy, curr_dz = dx.copy(), dy.copy(), dz.copy()
    
    for _ in range(iterations):
        next_dx, next_dy, next_dz = curr_dx.copy(), curr_dy.copy(), curr_dz.copy()
        for i in target_indices:
            neighbors = indices_list[i]
            if not neighbors: continue
            next_dx[i] = np.mean(curr_dx[neighbors])
            next_dy[i] = np.mean(curr_dy[neighbors])
            next_dz[i] = np.mean(curr_dz[neighbors])
        curr_dx, curr_dy, curr_dz = next_dx, next_dy, next_dz
        
    return curr_dx, curr_dy, curr_dz

def compute_fill_and_selective_smooth_3d(ref_data, curr_data, matches, grid_spacing):
    matched_ref_indices = [m[0] for m in matches]
    matched_curr_indices = [m[1] for m in matches]
    
    p_ref_matched = ref_data[matched_ref_indices]
    p_curr_matched = curr_data[matched_curr_indices]
    
    # 原有的 XY 位移
    dx_known = p_curr_matched[:, 0] - p_ref_matched[:, 0]
    dy_known = p_curr_matched[:, 1] - p_ref_matched[:, 1]
    
    # 新增 Z 轴计算：面积增加量 (归一化)
    # 逻辑：(当前面积 / 初始面积) - 1.0
    dz_known = (p_curr_matched[:, 2] / np.maximum(p_ref_matched[:, 2], 1.0)) - 1.0
    dz_known = np.maximum(0, dz_known) # 只保留下压的力
    
    full_dx = np.zeros(len(ref_data))
    full_dy = np.zeros(len(ref_data))
    full_dz = np.zeros(len(ref_data))
    
    full_dx[matched_ref_indices] = dx_known
    full_dy[matched_ref_indices] = dy_known
    full_dz[matched_ref_indices] = dz_known
    
    all_ref_indices = set(range(len(ref_data)))
    missing_indices = list(all_ref_indices - set(matched_ref_indices))
    
    # 1. 初步插值 (三向同步)
    if len(missing_indices) > 0 and len(p_ref_matched) > 3:
        p_ref_coords = p_ref_matched[:, :2]
        p_missing_coords = ref_data[missing_indices, :2]
        try:
            for field_known, field_full in zip([dx_known, dy_known, dz_known], [full_dx, full_dy, full_dz]):
                filled = griddata(p_ref_coords, field_known, p_missing_coords, method='linear')
                mask_nan = np.isnan(filled)
                if np.any(mask_nan):
                    filled[mask_nan] = griddata(p_ref_coords, field_known, p_missing_coords[mask_nan], method='nearest')
                field_full[missing_indices] = filled
        except:
            pass 

    # 2. 选择性平滑 (三向同步)
    smoothing_radius = grid_spacing * SPATIAL_RADIUS_RATIO
    full_dx, full_dy, full_dz = apply_selective_smoothing_3d(
        ref_data[:, :2], full_dx, full_dy, full_dz, 
        radius=smoothing_radius, 
        target_indices=missing_indices, 
        iterations=SPATIAL_ITERATIONS
    )

    return full_dx, full_dy, full_dz

# ---------- 3. 渲染 (保持原有框架) ----------
def fig_to_bgr(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return img

def render_dashboard_frame(frame, ref_points, dx, dy, dz, grid_x, grid_y, grid_z, max_val, figsize=(18, 6)):
    h, w = frame.shape[:2]
    # XY 平面模长 (用于 Arrow)
    mags_2d = np.hypot(dx, dy) * PIXEL_TO_FORCE
    
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=figsize)
        cmap = plt.get_cmap('jet') 

        # 1. Vector (2D Force)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("2D Force Vector (XY)", fontsize=14, color='white', pad=10)
        ax1.set_xlim(0, w); ax1.set_ylim(h, 0); ax1.set_aspect('equal')
        ax1.scatter(ref_points[:,0], ref_points[:,1], c='white', s=POINT_SIZE, alpha=0.3, edgecolors='none')
        
        mask = mags_2d > MIN_FORCE_THRESHOLD
        if np.any(mask):
            ax1.scatter(ref_points[mask,0], ref_points[mask,1], c='white', s=POINT_SIZE, alpha=0.8, edgecolors='none')
            norm = np.hypot(dx[mask], dy[mask])
            norm[norm == 0] = 1.0
            u = (dx[mask] / norm) * FIXED_ARROW_LEN
            v = (dy[mask] / norm) * FIXED_ARROW_LEN
            ax1.quiver(ref_points[mask,0], ref_points[mask,1], u, v, mags_2d[mask], angles='xy', scale_units='xy', scale=1,
                       cmap=cmap, width=ARROW_WIDTH, alpha=0.9, clim=(0, max_val))
        ax1.axis('off')

        # 2. Heatmap (Combined Force XYZ)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Combined Force (XYZ) Heatmap", fontsize=14, color='white', pad=10)
        extent = (0, w, h, 0)
        ax2.imshow(grid_z, extent=extent, origin='upper', cmap=cmap, vmin=0, vmax=max_val)
        ax2.axis('off')

        # 3. 3D Deformation (Combined Force XYZ)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.set_title("3D Deformation (Combined)", fontsize=14, color='white', pad=10)
        ax3.set_facecolor('black'); ax3.grid(False); ax3.axis('off')
        ax3.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, vmin=0, vmax=max_val,
                         rstride=2, cstride=2, linewidth=0.1, edgecolors='white', antialiased=True)
        ax3.view_init(elev=50, azim=-70)
        ax3.set_zlim(0, max_val * 1.5 if max_val > 1 else 10)

        plt.tight_layout()
        out = fig_to_bgr(fig)
        plt.close(fig)
        return out

def interpolate_magnitude(pts, mags, img_shape, grid_res=GRID_RES):
    h, w = img_shape[:2]
    grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, grid_res), np.linspace(0, h-1, grid_res))
    try:
        grid_z = griddata(pts, mags, (grid_x, grid_y), method='cubic')
        grid_z = np.nan_to_num(grid_z, nan=0.0)
    except:
        grid_z = np.zeros_like(grid_x)
    return grid_x, grid_y, grid_z

def get_grid_spacing(points):
    if len(points) < 2: return 30.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return np.median(dists[:, 1])

# ---------- 主处理流程 ----------
def process_video(video_path, output_path, ref_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("Initializing reference...")
    ref_frames_count = int(fps * ref_seconds)
    frames_buffer = []
    for _ in range(ref_frames_count):
        ret, f = cap.read()
        if ret: frames_buffer.append(f)
    if not frames_buffer: return
    median_frame = np.median(np.array(frames_buffer), axis=0).astype(np.uint8)
    
    # 获取参考点 [x, y, area]
    ref_data = detect_circles_with_area(median_frame, enable_repair=True)
    ref_coords = ref_data[:, :2]
    grid_spacing = get_grid_spacing(ref_coords)
    print(f"Ref Grid: {len(ref_data)} points. Spacing: {grid_spacing:.1f}")

    max_force_global = 30.0 
    print(f"Max Force Limit: {max_force_global}")

    print(f"Rendering (XYZ Combined Force)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_w, out_h = 1800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_data = detect_circles_with_area(frame, enable_repair=False)
        curr_coords = curr_data[:, :2] if len(curr_data) > 0 else np.array([])
        matches = match_circles(ref_coords, curr_coords)
        
        if matches:
            # 1. 计算选择性平滑后的 XYZ 位移/形变
            final_dx, final_dy, final_dz = compute_fill_and_selective_smooth_3d(ref_data, curr_data, matches, grid_spacing)
            
            # 2. 计算合力：sqrt((dx*scale_xy)^2 + (dy*scale_xy)^2 + (dz*scale_z)^2)
            # 这里保持 xy 原有的 PIXEL_TO_FORCE，给 dz 单独一个较大的 scale
            combined_mags = np.sqrt(
                (final_dx * PIXEL_TO_FORCE)**2 + 
                (final_dy * PIXEL_TO_FORCE)**2 + 
                (final_dz * Z_FORCE_SCALE)**2
            )
        else:
            final_dx = final_dy = final_dz = np.zeros(len(ref_data))
            combined_mags = np.zeros(len(ref_data))
            
        # 3. 生成 Heatmap 和 3D 用的插值格点 (使用合力数据)
        if np.max(combined_mags) > 0.1:
            gx, gy, gz = interpolate_magnitude(ref_coords, combined_mags, frame.shape)
        else:
            gx, gy = np.meshgrid(np.linspace(0, w-1, GRID_RES), np.linspace(0, h-1, GRID_RES))
            gz = np.zeros_like(gx)

        # 4. 渲染
        dashboard_img = render_dashboard_frame(frame, ref_coords, final_dx, final_dy, final_dz, gx, gy, gz, max_force_global, figsize=(18, 6))
        
        if dashboard_img is not None:
            if dashboard_img.shape[1] != out_w or dashboard_img.shape[0] != out_h:
                dashboard_img = cv2.resize(dashboard_img, (out_w, out_h))
            writer.write(dashboard_img)
        
        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"Processing frame {frame_idx}")

    writer.release()
    cap.release()
    print(f"Done! Saved to {output_path}")

def match_circles(c1, c2, max_dist=DEFAULT_RADIUS*2.0):
    if len(c1) == 0 or len(c2) == 0: return []
    matches = []
    used_j = set()
    for i, p in enumerate(c1):
        dist, j = 0, 0
        if len(c2) > 0:
            dists = np.linalg.norm(c2 - p, axis=1)
            j = int(np.argmin(dists))
            dist = dists[j]
            if dist > max_dist: continue
            if j in used_j: continue 
            matches.append((i, j))
            used_j.add(j)
    return matches

if __name__ == "__main__":
    input_video = "vision1.mp4" 
    output_video = "tactile1.mp4"
    if os.path.exists(input_video):
        process_video(input_video, output_video)
    else:
        print("Input video not found.")
