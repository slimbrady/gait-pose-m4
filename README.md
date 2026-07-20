# gait-pose-m4 – Running Gait Analysis for Apple Silicon

2D pose-based running biomechanics analyzer optimized for MacBook M4 (Apple Silicon). Upload a running video → get joint angles, cadence, speed, step length, ROM for 10+ joints.

Built with **RTMPose via rtmlib / ONNX Runtime** – fast, CPU-only, no CUDA required. Perfect for M1/M2/M3/M4.

---

## Features

- **Pose estimation**: RTMPose-m, COCO 17 keypoints, ~75 FPS on M4 CPU
- **Joint angles tracked** (all in degrees):
  - Lower extremity: hip flexion, knee flexion, ankle dorsi/plantar
  - Upper extremity: shoulder flexion, elbow flexion
  - Trunk lean (forward/back from vertical)
- **Gait metrics**:
  - Cadence (steps/min)
  - Speed (m/s, km/h)
  - Step length (m)
  - Stride time (s)
  - Ground contact time (ms)
  - Left / Right strike counts
  - Range of motion (ROM) per joint
- **Outputs**: skeleton-overlay MP4, per-frame angles CSV, summary JSON
- **UI**: Streamlit web app with Plotly charts, side-by-side video, download buttons

---

## Install (MacBook M4)

### Option A: conda (recommended)

```bash
# Create isolated env
conda create -n gait python=3.11 -y
conda activate gait

# Install dependencies
pip install -r requirements.txt

# That's it – no CUDA, no PyTorch GPU builds, no Homebrew hacks.
# ONNX Runtime CPU uses Apple's Accelerate framework automatically.
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads RTMPose/RTMDet ONNX weights automatically (~20 MB).

---

## Quick Start

### CLI

```bash
# 1. Pose inference: video → keypoints + overlay
python inference.py run.mp4
# → run_keypoints.json, run_pose.mp4

# 2. Biomechanics: keypoints → angles + metrics
python metrics.py run_keypoints.json
# → run_angles.csv, run_summary.json
```

### Streamlit UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 → upload video → get interactive dashboard.

---

## Model Comparison – Apple Silicon M4

| Model | Backend | Input | Speed (M4) | Accuracy | Install Pain | Verdict |
|-------|---------|-------|------------|----------|-------------|---------|
| **RTMPose-m** | ONNX Runtime CPU | 192×256 | **~75 FPS** | COCO AP 75.8 | `pip install rtmlib` ✅ | **Pick this – fastest on M4** |
| ViTPose-B | ONNX / PyTorch | 256×192 | ~25 FPS | COCO AP 75.8 | Heavy, slow | Accurate but 3× slower |
| WHAM | PyTorch | 224×224 | ~8 FPS | 3D SMPL mesh | Huge deps, slow | Overkill for 2D gait angles |
| YOLOv8-Pose | Ultralytics | 640×640 | ~40 FPS | COCO AP 69.0 | Easy | Decent, less accurate than RTMPose |

**Why RTMPose for M4?**
- ONNX Runtime CPU is natively optimized, no Metal/GPU port needed
- Smallest model that still hits SOTA COCO accuracy
- `rtmlib` gives one-line inference with built-in RTMDet person tracker
- No CUDA/ROCm/CuDNN rabbit hole – just `pip install`

---

## Deployment

| Target | Notes |
|--------|-------|
| **Streamlit local** | `streamlit run app.py` – best for M4, real-time inference |
| **Streamlit Cloud** | Push to GitHub → deploy at share.streamlit.io – free tier works, slower CPU |
| **Docker** | Add `FROM python:3.11-slim` + `pip install -r requirements.txt` – portable |

Recommended: run locally on your M4 for speed, use Streamlit Cloud for sharing results.

---

## Accuracy Notes

- **2D only** – joint angles are in the camera/image plane. Sagittal (side) view gives best hip/knee/ankle accuracy.
- **px→m calibration** – auto-estimates from subject height (default 1.75 m). For best speed/step_length accuracy, pass `--px-to-m` with a known scale (e.g. measure a floor marker).
- **Gait events** – foot strikes detected via `scipy.signal.find_peaks` on ankle vertical velocity. Works well at 30–120 FPS. Treadmill or track, full-body in frame.
- **Contact time** – crude estimate (~200ms window around strikes). For clinical ground contact, use force plates / foot pods.
- **Occlusion** – low-confidence keypoints (< 0.3) are masked as NaN and smoothed with Savitzky-Golay.
- **Not a medical device** – for coaching / research / personal tracking only.

---

## File Map

```
gait-pose-m4/
├── inference.py      # RTMPose video → keypoints.json + skeleton MP4
├── metrics.py        # keypoints → joint angles, gait events, CSV/JSON
├── app.py            # Streamlit UI: upload, analyze, Plotly charts
├── requirements.txt  # rtmlib, opencv, streamlit, plotly, pandas, numpy, scipy
└── README.md         # this file
```

### Key functions

`inference.py`:
- `run_inference()` – RTMDet + RTMPose loop, draws skeleton, writes JSON/MP4
- `draw_skeleton()` – COCO skeleton overlay

`metrics.py`:
- `angle_deg(a,b,c)` – 3-point joint angle in degrees
- `trunk_lean_deg()` – shoulder_mid → hip_mid vs vertical
- `detect_foot_strikes()` – `find_peaks` on ankle y-velocity
- `compute_metrics()` – all angles + cadence / speed / step_length / ROM

---

## License

MIT – do what you want, just don't blame me if your marathon time doesn't improve.
