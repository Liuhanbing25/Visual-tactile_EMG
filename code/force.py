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
TARGET_TIME_SEC = 5.0      # <--- 修改提取秒数
INPUT_VIDEO = "vision1.mp4" # <--- 输入视频文件名
OUTPUT_IMAGE = "tactile_analysis_result.png"

DEFAULT_RADIUS = 25               
PIXEL_TO_FORCE = 1.0              
GRID_RES = 80                     
Z_FORCE_SCALE = 10.0 
SPATIAL_RADIUS_RATIO = 2.5 
SPATIAL_ITERATIONS = 10
MIN_FORCE_THRESHOLD = 0.005
POINT_SIZE = 10         
FIXED_ARROW_LEN = 200     
ARROW_WIDTH = 0.013      
# ---------------------------------------------

# ---------- 1. 核心检测 ----------
def detect_circles_with_area(img, enable_repair=False):
    if len(img.shape) == 3:
        b, g, r_channel = cv2.split(img)
        gray = r_channel
    else:
        gray = img

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

# ---------- 2. 计算 & 平滑 ----------
def apply_selective_smoothing_3d(points, dx, dy, dz, radius, target_indices, iterations=1):
    if len(target_indices) == 0: return dx, dy, dz
    tree = cKDTree(points)
    indices_list = tree.query_ball_point(points, r=radius)
    curr_dx, curr_dy, curr_dz = dx.copy(), dy.copy(), dz.copy()
    for _ in range(iterations):
        next_dx, next_dy, next_dz = curr_dx.copy(), curr_dy.copy(), curr_dz.copy()
        for i in target_indices:
            neighbors = indices_list[i]
            if not neighbors: continue
            next_dx[i] = np.mean(curr_dx[neighbors]); next_dy[i] = np.mean(curr_dy[neighbors]); next_dz[i] = np.mean(curr_dz[neighbors])
        curr_dx, curr_dy, curr_dz = next_dx, next_dy, next_dz
    return curr_dx, curr_dy, curr_dz

def compute_fill_and_selective_smooth_3d(ref_data, curr_data, matches, grid_spacing):
    matched_ref_indices = [m[0] for m in matches]
    matched_curr_indices = [m[1] for m in matches]
    p_ref_matched = ref_data[matched_ref_indices]
    p_curr_matched = curr_data[matched_curr_indices]
    dx_known = p_curr_matched[:, 0] - p_ref_matched[:, 0]
    dy_known = p_curr_matched[:, 1] - p_ref_matched[:, 1]
    dz_known = (p_curr_matched[:, 2] / np.maximum(p_ref_matched[:, 2], 1.0)) - 1.0
    dz_known = np.maximum(0, dz_known)
    full_dx, full_dy, full_dz = np.zeros(len(ref_data)), np.zeros(len(ref_data)), np.zeros(len(ref_data))
    full_dx[matched_ref_indices], full_dy[matched_ref_indices], full_dz[matched_ref_indices] = dx_known, dy_known, dz_known
    missing_indices = list(set(range(len(ref_data))) - set(matched_ref_indices))
    if len(missing_indices) > 0 and len(p_ref_matched) > 3:
        p_ref_coords, p_missing_coords = p_ref_matched[:, :2], ref_data[missing_indices, :2]
        for field_k, field_f in zip([dx_known, dy_known, dz_known], [full_dx, full_dy, full_dz]):
            filled = griddata(p_ref_coords, field_k, p_missing_coords, method='linear')
            mask_nan = np.isnan(filled)
            if np.any(mask_nan): filled[mask_nan] = griddata(p_ref_coords, field_k, p_missing_coords[mask_nan], method='nearest')
            field_f[missing_indices] = filled
    smoothing_radius = grid_spacing * SPATIAL_RADIUS_RATIO
    return apply_selective_smoothing_3d(ref_data[:, :2], full_dx, full_dy, full_dz, smoothing_radius, missing_indices, SPATIAL_ITERATIONS)

# ---------- 3. 渲染 (已优化上移和白边) ----------
def render_analysis_image(frame, ref_points, dx, dy, dz, grid_x, grid_y, grid_z, max_val):
    h, w = frame.shape[:2]
    combined_mags = np.sqrt((dx * PIXEL_TO_FORCE)**2 + (dy * PIXEL_TO_FORCE)**2 + (dz * Z_FORCE_SCALE)**2)
    mags_2d = np.hypot(dx, dy) * PIXEL_TO_FORCE
    
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig = plt.figure(figsize=(24, 7.5))
        TITLE_SIZE = 46  
        LABEL_SIZE = 42
        cmap = plt.get_cmap('jet') 

        # 1. Force Vector
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("(a) Force Vector", fontsize=TITLE_SIZE, fontweight='bold', pad=25)
        ax1.set_xlim(0, w); ax1.set_ylim(h, 0); ax1.set_aspect('equal')
        ax1.scatter(ref_points[:,0], ref_points[:,1], c='lightgray', s=POINT_SIZE, alpha=0.5, edgecolors='none')
        mask = mags_2d > MIN_FORCE_THRESHOLD
        if np.any(mask):
            ax1.scatter(ref_points[mask,0], ref_points[mask,1], c='gray', s=POINT_SIZE, alpha=0.3)
            norm = np.maximum(np.hypot(dx[mask], dy[mask]), 1e-6)
            u, v = (dx[mask] / norm) * FIXED_ARROW_LEN, (dy[mask] / norm) * FIXED_ARROW_LEN
            q = ax1.quiver(ref_points[mask,0], ref_points[mask,1], u, v, combined_mags[mask], 
                           angles='xy', scale_units='xy', scale=1, cmap=cmap, 
                           width=ARROW_WIDTH, alpha=0.9, clim=(0, max_val))
        ax1.axis('off')

        # 2. Heatmap
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("(b) Heatmap", fontsize=TITLE_SIZE, fontweight='bold', pad=25)
        im = ax2.imshow(grid_z, extent=(0, w, h, 0), origin='upper', cmap=cmap, vmin=0, vmax=max_val)
        ax2.axis('off')

        # 3. 3D Deformation (手动调节位置上移)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        # 提高 y 坐标，防止上移后标题被切掉
        ax3.set_title("(c) 3D Deformation", fontsize=TITLE_SIZE, fontweight='bold', y=0.96)
        ax3.set_facecolor('white')
        ax3.grid(False); ax3.axis('off')
        surf = ax3.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, vmin=0, vmax=max_val,
                                 rstride=2, cstride=2, linewidth=0.1, edgecolors='gray', antialiased=True)
        ax3.view_init(elev=45, azim=-70)
        ax3.set_zlim(0, max_val * 1.5)
        
        # 【核心逻辑：强制上移 3D 子图】
        # y0 + 0.1 表示将底部高度向上推画布高度的 10%
        pos3 = ax3.get_position()
        ax3.set_position([pos3.x0-2, pos3.y0 + 2, pos3.width, pos3.height])

        # 4. 全局布局与 Colorbar
        fig.subplots_adjust(right=0.94, left=0.02, wspace=0.05) 
        cbar_ax = fig.add_axes([0.95, 0.2, 0.012, 0.6]) 
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Force Magnitude', fontsize=LABEL_SIZE, fontweight='bold')

        # 保存时 bbox_inches='tight' 会自动修剪外部所有白边
        plt.savefig(OUTPUT_IMAGE, dpi=200, bbox_inches='tight', pad_inches=0.1) 
        print(f"Result image saved to: {OUTPUT_IMAGE}")
        plt.close(fig)

def interpolate_magnitude(pts, mags, img_shape, grid_res=GRID_RES):
    h, w = img_shape[:2]
    grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, grid_res), np.linspace(0, h-1, grid_res))
    try:
        grid_z = griddata(pts, mags, (grid_x, grid_y), method='cubic')
        grid_z = np.nan_to_num(grid_z, nan=0.0)
    except: grid_z = np.zeros_like(grid_x)
    return grid_x, grid_y, grid_z

def match_circles(c1, c2, max_dist=DEFAULT_RADIUS*2.0):
    if len(c1) == 0 or len(c2) == 0: return []
    matches, used_j = [], set()
    for i, p in enumerate(c1):
        dists = np.linalg.norm(c2[:, :2] - p[:2], axis=1) if len(c2)>0 else []
        if len(dists) == 0: continue
        j = int(np.argmin(dists))
        if dists[j] < max_dist and j not in used_j: matches.append((i, j)); used_j.add(j)
    return matches

# ---------- 4. 执行流程 ----------
def extract_single_frame():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened(): print(f"Error: {INPUT_VIDEO}"); return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print("Initializing reference...")
    frames_buffer = []
    for _ in range(int(fps)):
        ret, f = cap.read()
        if ret: frames_buffer.append(f)
    if not frames_buffer: return
    median_frame = np.median(np.array(frames_buffer), axis=0).astype(np.uint8)
    ref_data = detect_circles_with_area(median_frame, enable_repair=True)
    ref_coords = ref_data[:, :2]
    
    tree = cKDTree(ref_coords)
    grid_spacing = np.median(tree.query(ref_coords, k=2)[0][:, 1]) if len(ref_data)>1 else 30.0
    
    target_frame_idx = int(TARGET_TIME_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    if not ret: return
    
    print(f"Processing frame at {TARGET_TIME_SEC} seconds...")
    curr_data = detect_circles_with_area(frame, enable_repair=False)
    matches = match_circles(ref_data, curr_data)
    
    if matches:
        final_dx, final_dy, final_dz = compute_fill_and_selective_smooth_3d(ref_data, curr_data, matches, grid_spacing)
        combined_mags = np.sqrt((final_dx * PIXEL_TO_FORCE)**2 + (final_dy * PIXEL_TO_FORCE)**2 + (final_dz * Z_FORCE_SCALE)**2)
    else:
        final_dx = final_dy = final_dz = combined_mags = np.zeros(len(ref_data))
        
    gx, gy, gz = interpolate_magnitude(ref_coords, combined_mags, frame.shape)
    
    render_analysis_image(frame, ref_coords, final_dx, final_dy, final_dz, gx, gy, gz, max_val=30.0)
    cap.release()

if __name__ == "__main__":
    extract_single_frame()
