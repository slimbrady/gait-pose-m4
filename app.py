#!/usr/bin/env python3
"""
app.py - Streamlit UI for running gait analysis (M4 optimized)
v0.2 – force + muscle activation + Google Sheets logging
"""
import streamlit as st
import tempfile, json, time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Local imports
from inference import run_inference
from metrics import compute_metrics

try:
    from biomech_force import compute_forces
    FORCE_AVAILABLE = True
except Exception: FORCE_AVAILABLE = False
try:
    from muscle_activation import compute_activations
    MUSCLE_AVAILABLE = True
except Exception: MUSCLE_AVAILABLE = False

try:
    from supabase_export_ui import render_supabase_export
    SUPABASE_EXPORT_AVAILABLE = True
except Exception:
    SUPABASE_EXPORT_AVAILABLE = False

st.set_page_config(page_title="Gait Pose – M4", page_icon="🏃", layout="wide")
st.title("🏃 Running Gait Analysis – Apple Silicon M4")
st.caption("RTMPose (ONNX) · Joint forces · Muscle activation · Sheets logging")

with st.sidebar:
    st.header("⚙️ Subject")
    mass_unit = st.radio("Mass unit", ["kg","lb"], horizontal=True, index=0)
    mass_default = 75.0 if mass_unit=="kg" else 165.0
    mass_max = 200.0 if mass_unit=="kg" else 440.0
    mass_min = 30.0 if mass_unit=="kg" else 66.0
    mass_input = st.number_input(f"Body mass ({mass_unit})", mass_min, mass_max, mass_default, 0.5)
    mass_kg = mass_input if mass_unit=="kg" else mass_input/2.20462

    height_m = st.number_input("Subject height (m)", 1.2, 2.2, 1.75, 0.01)
    sex = st.selectbox("Sex", ["unspecified","F","M"], help="anthropometric scaling")
    age = st.number_input("Age", 0, 100, 30)
    subject_id = st.text_input("Subject ID", "")
    foot_length_cm = st.number_input("Foot length (cm, optional)", 0.0, 35.0, 0.0, 0.5)

    st.divider()
    st.header("⚙️ Analysis")
    px_to_m = st.number_input("px→m scale (0 = auto)", 0.0, 0.05, 0.0, 0.0001, format="%.5f")
    if px_to_m == 0.0: px_to_m = None
    det_thr = st.slider("Detection confidence", 0.1, 0.9, 0.5, 0.05)
    log_to_sheets = st.checkbox("Log to Google Sheets", value=True,
        help="Requires gspread_service_account in secrets.toml")
    st.markdown("---")
    st.caption("Model: RTMPose-m · Apple Silicon M1/M2/M3/M4")

uploaded = st.file_uploader("Upload running video (.mp4 / .mov)", type=["mp4","mov","m4v","avi"])
if uploaded is None:
    st.info("👆 Upload a sagittal-plane running video to start.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    video_path = tmpdir / uploaded.name
    video_path.write_bytes(uploaded.read())
    st.video(str(video_path))
    run_btn = st.button("🚀 Run Gait Analysis", type="primary", use_container_width=True)
    if not run_btn and 'results_done' not in st.session_state:
        st.stop()
    t0 = time.time()
    with st.spinner("Running RTMPose inference…"):
        try:
            kpts_json, overlay_mp4 = run_inference(str(video_path),
                out_json=str(tmpdir/"keypoints.json"), out_video=str(tmpdir/"overlay.mp4"),
                device='cpu', det_score_thr=det_thr)
            with open(kpts_json) as f: kpts_data = json.load(f)
            st.session_state['kpts_np'] = np.array(kpts_data.get('frames', []), dtype=object)
            st.session_state['video_path_str'] = str(video_path)
        except Exception as e:
            st.error(f"Inference failed: {e}"); st.stop()
    with st.spinner("Computing biomechanics…"):
        try:
            df_angles, summary = compute_metrics(kpts_json, px_to_m=px_to_m, height_m=height_m)
            st.session_state['inference_fps'] = summary.get('fps',30.0)
        except Exception as e:
            st.error(f"Metrics failed: {e}"); st.stop()
    # --- Force + Muscle ---
    df_forces = pd.DataFrame(); forces_summary = {}
    df_act = pd.DataFrame(); act_summary = {}
    if FORCE_AVAILABLE:
        try:
            # Re-load keypoints for force module (px)
            with open(kpts_json) as f: kd = json.load(f)
            kpts = np.array([fr['keypoints'] for fr in kd['frames']], dtype=np.float32)
            scores = np.array([fr['scores'] for fr in kd['frames']], dtype=np.float32)
            KP = {'L_shoulder':5,'R_shoulder':6,'L_elbow':7,'R_elbow':8,'L_wrist':9,'R_wrist':10,
                  'L_hip':11,'R_hip':12,'L_knee':13,'R_knee':14,'L_ankle':15,'R_ankle':16}
            def kp(name, thr=0.3):
                xy = kpts[:,KP[name],:].copy(); conf = scores[:,KP[name]]; xy[conf<thr]=np.nan; return xy
            kp_dict = {n.replace('_','_') if True else n: kp(n) for n in KP}
            # rename to expected keys
            kp_dict = {k: kp(k) for k in KP}
            fps = summary['fps']
            pxm = summary['px_to_m']
            df_forces, forces_summary = compute_forces(kp_dict, fps, mass_kg, height_m, pxm)
        except Exception as e:
            st.warning(f"Force estimation failed: {e}")
    if MUSCLE_AVAILABLE and not df_forces.empty:
        try:
            df_act, act_summary = compute_activations(df_angles, df_forces, summary['fps'])
        except Exception as e:
            st.warning(f"Muscle activation failed: {e}")

    st.session_state['results_done'] = True
    st.session_state['df_angles'] = df_angles
    st.session_state['summary'] = summary
    st.session_state['df_forces'] = df_forces
    st.session_state['forces_summary'] = forces_summary
    st.session_state['df_act'] = df_act
    st.session_state['act_summary'] = act_summary
    st.session_state['overlay_mp4'] = Path(overlay_mp4).read_bytes()
    st.session_state['processing_time_s'] = time.time()-t0

# --- Display ---
df_angles = st.session_state['df_angles']
summary = st.session_state['summary']
df_forces = st.session_state.get('df_forces', pd.DataFrame())
forces_summary = st.session_state.get('forces_summary', {})
df_act = st.session_state.get('df_act', pd.DataFrame())
act_summary = st.session_state.get('act_summary', {})
overlay_bytes = st.session_state['overlay_mp4']

st.success("✅ Analysis complete!")
c1,c2 = st.columns(2)
with c1: st.subheader("Original"); st.video(uploaded)
with c2: st.subheader("Pose Overlay"); st.video(overlay_bytes)
st.markdown("---")

# Metrics dashboard
st.subheader("📊 Gait Metrics")
g = summary['gait']
m1,m2,m3,m4 = st.columns(4)
m1.metric("Cadence", f"{g['cadence_spm']:.1f} spm" if g['cadence_spm'] else "n/a")
m2.metric("Speed", f"{g['speed_kmh']:.2f} km/h" if g['speed_kmh'] else "n/a")
m3.metric("Step length", f"{g['step_length_m']:.2f} m" if g['step_length_m'] else "n/a")
m4.metric("Stride time", f"{g['stride_time_s']:.3f} s" if g['stride_time_s'] else "n/a")
m5,m6,m7,m8 = st.columns(4)
m5.metric("Contact time", f"{g['contact_time_s']*1000:.0f} ms")
m6.metric("L strikes", g['L_strikes']); m7.metric("R strikes", g['R_strikes']); m8.metric("px→m", f"{summary['px_to_m']:.5f}")

if forces_summary:
    st.subheader("💪 Force Summary")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Peak GRF L", f"{forces_summary.get('peak_grf_L_bw',0):.2f} ×BW")
    f2.metric("Peak GRF R", f"{forces_summary.get('peak_grf_R_bw',0):.2f} ×BW")
    f3.metric("Peak Knee Moment", f"{forces_summary.get('peak_knee_moment_L',0):.2f} Nm/kg")
    f4.metric("Peak Ankle Moment", f"{forces_summary.get('peak_ankle_moment_L',0):.2f} Nm/kg")

# Tabs
tab_angles, tab_forces, tab_muscles = st.tabs(["📐 Angles / ROM", "💪 Forces", "⚡ Muscle Activation"])

with tab_angles:
    rom = summary['rom_deg']
    rom_df = pd.DataFrame([
        {"Joint":"Hip L","ROM (°)":rom['L_hip_rom'],"Mean (°)":summary['mean_angles_deg']['L_hip']},
        {"Joint":"Hip R","ROM (°)":rom['R_hip_rom'],"Mean (°)":summary['mean_angles_deg']['R_hip']},
        {"Joint":"Knee L","ROM (°)":rom['L_knee_rom'],"Mean (°)":summary['mean_angles_deg']['L_knee']},
        {"Joint":"Knee R","ROM (°)":rom['R_knee_rom'],"Mean (°)":summary['mean_angles_deg']['R_knee']},
        {"Joint":"Ankle L","ROM (°)":rom['L_ankle_rom'],"Mean (°)":summary['mean_angles_deg']['L_ankle']},
        {"Joint":"Ankle R","ROM (°)":rom['R_ankle_rom'],"Mean (°)":summary['mean_angles_deg']['R_ankle']},
        {"Joint":"Shoulder L","ROM (°)":rom['L_shoulder_rom'],"Mean (°)":summary['mean_angles_deg']['L_shoulder']},
        {"Joint":"Shoulder R","ROM (°)":rom['R_shoulder_rom'],"Mean (°)":summary['mean_angles_deg']['R_shoulder']},
        {"Joint":"Elbow L","ROM (°)":rom['L_elbow_rom'],"Mean (°)":summary['mean_angles_deg']['L_elbow']},
        {"Joint":"Elbow R","ROM (°)":rom['R_elbow_rom'],"Mean (°)":summary['mean_angles_deg']['R_elbow']},
        {"Joint":"Trunk lean","ROM (°)":rom['trunk_lean_rom'],"Mean (°)":summary['mean_angles_deg']['trunk_lean']},
    ])
    st.dataframe(rom_df.style.format({"ROM (°)":"{:.1f}","Mean (°)":"{:.1f}"}), use_container_width=True, hide_index=True)
    joint_groups = {
        "Lower Extremity": ['L_hip_deg','R_hip_deg','L_knee_deg','R_knee_deg','L_ankle_deg','R_ankle_deg'],
        "Upper Extremity": ['L_shoulder_deg','R_shoulder_deg','L_elbow_deg','R_elbow_deg'],
        "Trunk": ['trunk_lean_deg'],
    }
    group = st.radio("Joint group", list(joint_groups.keys()), horizontal=True, key="angle_group")
    fig = go.Figure()
    for col in joint_groups[group]:
        if col in df_angles.columns:
            fig.add_trace(go.Scatter(x=df_angles['time_s'], y=df_angles[col], mode='lines', name=col.replace('_deg','')))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Angle (°)", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

with tab_forces:
    if df_forces.empty:
        st.info("Force estimation unavailable – install biomech_force.py")
    else:
        grf_cols = [c for c in df_forces.columns if 'grf_bw' in c.lower()]
        if grf_cols:
            fig_f = px.line(df_forces, x='time_s', y=grf_cols, title="Ground Reaction Force (%BW)")
            st.plotly_chart(fig_f, use_container_width=True)
        mom_cols = [c for c in df_forces.columns if 'moment_nmk' in c and c.count('_')<=3]
        if mom_cols:
            fig_m = px.line(df_forces, x='time_s', y=mom_cols[:6], title="Joint Moments (Nm/kg)")
            st.plotly_chart(fig_m, use_container_width=True)
        st.json(forces_summary)

with tab_muscles:
    if df_act.empty:
        st.info("Muscle activation unavailable – install muscle_activation.py")
    else:
        muscle_cols = [c for c in df_act.columns if c != 'time_s']
        fig_a = px.line(df_act, x='time_s', y=muscle_cols, title="Muscle Activation (0-1)")
        fig_a.update_yaxes(range=[0,1.05])
        st.plotly_chart(fig_a, use_container_width=True)
        st.json(act_summary)

# Sheets logging
if log_to_sheets:
    try:
        from sheets_logger import log_run
        log_run(summary, forces_summary, act_summary,
            engine="rtmpose", model_version="rtmpose-m",
            subject_meta={"mass_kg": mass_kg, "height_m": height_m, "sex": sex, "age": age, "subject_id": subject_id},
            perf={"inference_fps": summary['fps'], "processing_time_s": st.session_state.get('processing_time_s',0)},
            video_path=st.session_state.get('video_path_str'))
        st.toast("Logged to Google Sheets ✓")
    except Exception as e:
        st.caption(f"Sheets log skipped: {e}")

# Supabase export (optional)
if SUPABASE_EXPORT_AVAILABLE and 'kpts_np' in st.session_state:
    try:
        render_supabase_export(keypoints=st.session_state['kpts_np'],
            fps=st.session_state.get('inference_fps',30.0),
            video_path=st.session_state.get('video_path_str','video.mp4'),
            engine="rtmpose", model_version="rtmpose-m", default_view="sagittal")
    except Exception: pass

# Downloads
st.subheader("⬇️ Downloads")
csv_bytes = df_angles.to_csv(index=False).encode()
json_bytes = json.dumps(summary, indent=2).encode()
d1,d2,d3,d4,d5 = st.columns(5)
d1.download_button("📄 Angles CSV", csv_bytes, "gait_angles.csv", "text/csv", use_container_width=True)
d2.download_button("📋 Summary JSON", json_bytes, "gait_summary.json", "application/json", use_container_width=True)
d3.download_button("🎞️ Overlay MP4", overlay_bytes, "gait_overlay.mp4", "video/mp4", use_container_width=True)
if not df_forces.empty:
    d4.download_button("💪 Forces CSV", df_forces.to_csv(index=False).encode(), "gait_forces.csv", "text/csv", use_container_width=True)
if not df_act.empty:
    d5.download_button("⚡ Muscles CSV", df_act.to_csv(index=False).encode(), "gait_muscles.csv", "text/csv", use_container_width=True)

st.divider()
with st.expander("📚 Research & Citations"):
    st.markdown("""
**RTMPose / 2D pose-based gait validation**

1. Guo & Zhao, 2024 – *Gait analysis based on RTMPose using knee angle*
2. Menychtas et al., 2023 – *Gait analysis: 2D pose vs 3D marker-based* – Front. Rehabil. Sci. https://doi.org/10.3389/fresc.2023.1238134
3. Wade et al., 2022 – *Applications and limitations of markerless motion capture* – PeerJ https://doi.org/10.7717/peerj.12995
4. Tang et al., 2022 – *Joint Moment and Power: markerless vs marker-based running* – https://doi.org/10.3390/biomechanics9040574
5. Johnson et al., 2022 – *Foot and Tibia Angles During Running – markerless vs manual* – J Appl Biomech

---
**Full citation table:** [Google Sheet – Citations](https://docs.google.com/spreadsheets/d/1o4aA07t5ODfsXtLl5M0j6SudLRbGovgxpDUzHKFFOk8/edit#gid=1112676311)

**GitHub:** https://github.com/slimbrady/gait-pose-m4
""")

st.caption(f"Duration: {summary['duration_s']:.1f}s @ {summary['fps']:.1f} FPS · mass: {mass_kg:.1f} kg")
