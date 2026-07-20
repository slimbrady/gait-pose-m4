#!/usr/bin/env python3
"""
app.py - Streamlit UI for running gait analysis (M4 optimized)

Upload a running video → get joint angles, gait metrics, and downloadable CSV/JSON.
"""

import streamlit as st
import tempfile
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from inference import run_inference
from metrics import compute_metrics

st.set_page_config(
    page_title="Gait Pose – M4",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 Running Gait Analysis – Apple Silicon M4")
st.caption("RTMPose (ONNX) · COCO 17 keypoints · 10+ joint angles · cadence / speed / ROM")

with st.sidebar:
    st.header("⚙️ Settings")
    height_m = st.number_input("Subject height (m)", 1.2, 2.2, 1.75, 0.01,
                                help="Used for px→m auto-calibration")
    px_to_m = st.number_input("px→m scale (0 = auto)", 0.0, 0.05, 0.0, 0.0001, format="%.5f",
                               help="Leave 0 for auto-calibration from height")
    if px_to_m == 0.0:
        px_to_m = None
    det_thr = st.slider("Detection confidence", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown("**Model:** RTMPose-m (ONNX Runtime CPU)\n\n**Optimized for:** Apple Silicon M1/M2/M3/M4")
    st.markdown("No CUDA required – runs fast on M4 out of the box.")

uploaded = st.file_uploader("Upload running video (.mp4 / .mov)", type=["mp4", "mov", "m4v", "avi"])

if uploaded is None:
    st.info("👆 Upload a sagittal-plane running video to start. Treadmill or track, full body in frame works best.")
    st.stop()

# Save upload to temp file
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    video_path = tmpdir / uploaded.name
    video_path.write_bytes(uploaded.read())

    st.video(str(video_path))

    run_btn = st.button("🚀 Run Gait Analysis", type="primary", use_container_width=True)
    if not run_btn and 'results_done' not in st.session_state:
        st.stop()

    with st.spinner("Running RTMPose inference… (first run downloads model weights)"):
        try:
            kpts_json, overlay_mp4 = run_inference(
                str(video_path),
                out_json=str(tmpdir / "keypoints.json"),
                out_video=str(tmpdir / "overlay.mp4"),
                device='cpu',
                det_score_thr=det_thr,
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    with st.spinner("Computing biomechanics…"):
        try:
            df_angles, summary = compute_metrics(kpts_json, px_to_m=px_to_m, height_m=height_m)
        except Exception as e:
            st.error(f"Metrics failed: {e}")
            st.stop()

    # Cache results in session
    st.session_state['results_done'] = True
    st.session_state['df_angles'] = df_angles
    st.session_state['summary'] = summary
    st.session_state['overlay_mp4'] = Path(overlay_mp4).read_bytes()

# --- Display results ---
df_angles = st.session_state['df_angles']
summary = st.session_state['summary']
overlay_bytes = st.session_state['overlay_mp4']

st.success("✅ Analysis complete!")

# Video side-by-side
c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.video(uploaded)
with c2:
    st.subheader("Pose Overlay")
    st.video(overlay_bytes)

st.markdown("---")

# Metrics dashboard
st.subheader("📊 Gait Metrics")
g = summary['gait']
m1, m2, m3, m4 = st.columns(4)
m1.metric("Cadence", f"{g['cadence_spm']:.1f} spm" if g['cadence_spm'] else "n/a", help="Steps per minute")
m2.metric("Speed", f"{g['speed_kmh']:.2f} km/h" if g['speed_kmh'] else "n/a",
          help=f"{g['speed_ms']:.2f} m/s" if g['speed_ms'] else None)
m3.metric("Step length", f"{g['step_length_m']:.2f} m" if g['step_length_m'] else "n/a")
m4.metric("Stride time", f"{g['stride_time_s']:.3f} s" if g['stride_time_s'] else "n/a")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Contact time", f"{g['contact_time_s']*1000:.0f} ms")
m6.metric("L strikes", g['L_strikes'])
m7.metric("R strikes", g['R_strikes'])
m8.metric("px→m", f"{summary['px_to_m']:.5f}")

# ROM table
st.subheader("📐 Range of Motion")
rom = summary['rom_deg']
rom_df = pd.DataFrame([
    {"Joint": "Hip L", "ROM (°)": rom['L_hip_rom'], "Mean (°)": summary['mean_angles_deg']['L_hip']},
    {"Joint": "Hip R", "ROM (°)": rom['R_hip_rom'], "Mean (°)": summary['mean_angles_deg']['R_hip']},
    {"Joint": "Knee L", "ROM (°)": rom['L_knee_rom'], "Mean (°)": summary['mean_angles_deg']['L_knee']},
    {"Joint": "Knee R", "ROM (°)": rom['R_knee_rom'], "Mean (°)": summary['mean_angles_deg']['R_knee']},
    {"Joint": "Ankle L", "ROM (°)": rom['L_ankle_rom'], "Mean (°)": summary['mean_angles_deg']['L_ankle']},
    {"Joint": "Ankle R", "ROM (°)": rom['R_ankle_rom'], "Mean (°)": summary['mean_angles_deg']['R_ankle']},
    {"Joint": "Shoulder L", "ROM (°)": rom['L_shoulder_rom'], "Mean (°)": summary['mean_angles_deg']['L_shoulder']},
    {"Joint": "Shoulder R", "ROM (°)": rom['R_shoulder_rom'], "Mean (°)": summary['mean_angles_deg']['R_shoulder']},
    {"Joint": "Elbow L", "ROM (°)": rom['L_elbow_rom'], "Mean (°)": summary['mean_angles_deg']['L_elbow']},
    {"Joint": "Elbow R", "ROM (°)": rom['R_elbow_rom'], "Mean (°)": summary['mean_angles_deg']['R_elbow']},
    {"Joint": "Trunk lean", "ROM (°)": rom['trunk_lean_rom'], "Mean (°)": summary['mean_angles_deg']['trunk_lean']},
])
st.dataframe(rom_df.style.format({"ROM (°)": "{:.1f}", "Mean (°)": "{:.1f}"}), use_container_width=True, hide_index=True)

# Angle time series plots
st.subheader("📈 Joint Angles Over Time")

joint_groups = {
    "Lower Extremity": ['L_hip_deg', 'R_hip_deg', 'L_knee_deg', 'R_knee_deg', 'L_ankle_deg', 'R_ankle_deg'],
    "Upper Extremity": ['L_shoulder_deg', 'R_shoulder_deg', 'L_elbow_deg', 'R_elbow_deg'],
    "Trunk": ['trunk_lean_deg'],
}
group = st.radio("Joint group", list(joint_groups.keys()), horizontal=True)
cols_to_plot = joint_groups[group]

fig = go.Figure()
for col in cols_to_plot:
    if col in df_angles.columns:
        fig.add_trace(go.Scatter(
            x=df_angles['time_s'], y=df_angles[col],
            mode='lines', name=col.replace('_deg', ''),
        ))
fig.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Angle (°)",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# Downloads
st.subheader("⬇️ Downloads")
csv_bytes = df_angles.to_csv(index=False).encode('utf-8')
json_bytes = json.dumps(summary, indent=2).encode('utf-8')

d1, d2, d3 = st.columns(3)
d1.download_button("📄 Angles CSV", csv_bytes, "gait_angles.csv", "text/csv", use_container_width=True)
d2.download_button("📋 Summary JSON", json_bytes, "gait_summary.json", "application/json", use_container_width=True)
d3.download_button("🎞️ Overlay MP4", overlay_bytes, "gait_overlay.mp4", "video/mp4", use_container_width=True)

st.caption(f"Duration: {summary['duration_s']:.1f}s @ {summary['fps']:.1f} FPS · {summary['num_frames']} frames analyzed")
