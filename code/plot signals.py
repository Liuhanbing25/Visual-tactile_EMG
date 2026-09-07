import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据
# 自动忽略以 '#' 开头的注释行
filename = 'emg_log_dev15_1.csv'
df = pd.read_csv(filename, comment='#')

# 2. 准备绘图
# 创建 3 个子图分别展示：原始信号、滤波信号、包络线
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 绘制原始信号 (ch1_raw_uV)
ax1.plot(df['EMG_t_s'], df['ch1_raw_uV'], color='gray', alpha=0.6, label='Raw Signal')
ax1.set_title('Channel 1: EMG Signal Analysis')
ax1.set_ylabel('Raw (uV)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.7)

# 绘制滤波后信号 (ch1_filt_uV)
ax2.plot(df['EMG_t_s'], df['ch1_filt_uV'], color='blue', label='Filtered Signal')
ax2.set_ylabel('Filtered (uV)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.7)

# 绘制包络线/RMS (ch1_env_uV)
ax3.plot(df['EMG_t_s'], df['ch1_env_uV'], color='red', linewidth=1.5, label='Envelope (RMS)')
ax3.set_ylabel('Envelope (uV)')
ax3.set_xlabel('Time (seconds)')
ax3.legend(loc='upper right')
ax3.grid(True, linestyle='--', alpha=0.7)

# 3. 优化布局并显示/保存
plt.tight_layout()
plt.savefig('emg_plot1.png')
plt.show()

# --- 新增部分：以 CSI/CDU 风格独立绘制 Envelope (RMS) 曲线 ---

# 使用与 CSI/CDU 相同的白色带网格风格
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.figure(figsize=(14, 7))
    
    # 字体大小配置 (完全对齐 CSI/CDU 风格参数)
    TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 24, 20, 16, 16

    # 绘制包络线 (使用红线，保持 EMG 信号的辨识度)
    plt.plot(df['EMG_t_s'], df['ch1_env_uV'], color='red', lw=2, label='EMG Envelope (RMS)')

    # 设置标题、标签和刻度 (使用 CSI/CDU 的大字体和间距)
    plt.title('EMG Envelope Signal Analysis', fontsize=TITLE_SIZE, pad=25)
    plt.xlabel('Time (seconds)', fontsize=LABEL_SIZE, labelpad=15)
    plt.ylabel('Envelope Magnitude (uV)', fontsize=LABEL_SIZE, labelpad=15)
    
    # 设置坐标轴刻度字体
    plt.xticks(fontsize=TICKS_SIZE)
    plt.yticks(fontsize=TICKS_SIZE)
    
    # 设置图例 (带阴影和白底，与 CSI/CDU 一致)
    plt.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)
    
    # 设置网格 (浅灰色虚线)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存为高质量图片
    plt.savefig('emg_envelope_csi_style.png', dpi=300)
    plt.show()








import pandas as pd
import matplotlib.pyplot as plt
import os

filename = 'emg_log_dev15_1.csv' 

if not os.path.exists(filename):
    print(f"错误：找不到文件 {filename}，请检查文件名或路径是否正确。")
else:
    df = pd.read_csv(filename, comment='#')

    # ==================== 2. 用户自定义区间设置 ====================
    START_TIME = 9  # 设置你想截取的开始时间（秒）
    END_TIME = 25.7    # 设置你想截取的结束时间（秒）

    # 截取数据区间
    mask = (df['EMG_t_s'] >= START_TIME) & (df['EMG_t_s'] <= END_TIME)
    df_subset = df.loc[mask].copy() # 使用 copy 避免警告

    # --- 核心修改：计算相对时间，使横轴从 0 开始 ---
    df_subset['relative_time'] = df_subset['EMG_t_s'] - START_TIME
    # =============================================================

    # 3. 绘图显示 (对齐 CSI/CDU 的白色背景 + 大字体风格)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(14, 7))
        
        # 字体大小配置 (与 CSI/CDU 风格一致)
        TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 28, 26, 24, 24

        # 使用 relative_time 作为横坐标
        plt.plot(df_subset['relative_time'], df_subset['ch1_env_uV'], 
                 color='red', lw=2.5, label='EMG Envelope (RMS)')

        # 设置标题、标签和刻度
        plt.title(f'EMG Envelope Analysis', 
                  fontsize=TITLE_SIZE, pad=25)
        plt.xlabel('Time Offset (seconds)', fontsize=LABEL_SIZE, labelpad=15)
        plt.ylabel('Envelope Magnitude (uV)', fontsize=LABEL_SIZE, labelpad=15)
        
        # 设置坐标轴刻度字体
        plt.xticks(fontsize=TICKS_SIZE)
        plt.yticks(fontsize=TICKS_SIZE)
        
        # 设置图例 (带阴影和白底)
        plt.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('emg_envelope_relative_time.png', dpi=300)
        plt.show()








import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 加载数据
filename = 'emg_log_dev15_1.csv' 

if not os.path.exists(filename):
    print(f"错误：找不到文件 {filename}，请检查文件名或路径是否正确。")
else:
    df = pd.read_csv(filename, comment='#')

    # ==================== 2. 用户自定义区间设置 ====================
    START_TIME = 9   # 起始时间
    END_TIME = 25.7    # 结束时间

    # 截取数据区间
    mask = (df['EMG_t_s'] >= START_TIME) & (df['EMG_t_s'] <= END_TIME)
    df_subset = df.loc[mask].copy()

    # 计算相对时间，使横轴从 0 开始
    df_subset['relative_time'] = df_subset['EMG_t_s'] - START_TIME
    # =============================================================

    # 3. 绘图显示 (对齐 CSI/CDU 的白色背景 + 大字体风格)
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(14, 7))
        
        # 字体大小配置 (与 CSI/CDU 风格一致)
        TITLE_SIZE, LABEL_SIZE, TICKS_SIZE, LEGEND_SIZE = 38, 34, 32, 32

        # --- 核心修改：改为绘制过滤后的信号，但不使用 "filtered" 标签 ---
        plt.plot(df_subset['relative_time'], df_subset['ch1_filt_uV'], 
                 color='blue', lw=1.5, label='EMG Signal')

        # 修改标题和坐标轴标签，去掉 Envelope/Filtered 字样
        plt.title('(b) EMG Signal', fontsize=TITLE_SIZE, pad=25)
        plt.xlabel('Time Offset (seconds)', fontsize=LABEL_SIZE, labelpad=15)
        plt.ylabel('EMG Amplitude (uV)', fontsize=LABEL_SIZE, labelpad=15)
        
        # 设置坐标轴刻度字体
        plt.xticks(fontsize=TICKS_SIZE)
        plt.yticks(fontsize=TICKS_SIZE)
        
        # 设置图例
        plt.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True, facecolor='white', shadow=True)
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('emg_signal_relative_time.png', dpi=300)
        plt.show()
