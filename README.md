Code and Data Guide
EMG and Fingertip Visual-Tactile Analysis
This folder contains the paper-related analysis scripts, experimental data and visualization results.
1 Files in code/
File	Purpose
analysis.py	Tactile video analysis and dynamic visualization.
CSI.py / CDU.py / ETE.py	Calculate Contact Stability Index (CSI), Contact Distribution Uniformity (CDU), and EMG-Tactile Efficiency (ETE), respectively.
force.py	Generates vector, heatmap and 3D deformation plots.
plot signals.py	Plots EMG signals and RMS envelopes.
receive_signals.py	Earlier EMG and camera acquisition program.
vision1.mp4
emg_log_dev15_1.csv	Analysis inputs: tactile video and corresponding EMG data.
frame_10.png / frame_150.png	Extracted video frames for inspecting markers and deformation.
tactile1.mp4	Processed tactile visualization video.
Final_Stability_Analysis_Report.png	Saved CSI analysis plot.
Normalized_CDU_Analysis_Report.png	Saved CDU analysis plot.
ETE_Combined_Sync_Report.png	Saved joint ETE analysis plot.
tactile_analysis_result.png	Saved tactile visualization image.
2 Updated acquisition script in the parent folder
The separate receive signals script is the updated acquisition program. It adds the Unix_Time_s absolute timestamp for EMG–video time alignment. Use this version for new recordings; code/receive_signals.py is retained as the earlier version.

