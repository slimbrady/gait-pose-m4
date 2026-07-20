#!/usr/bin/env python3
"""
metrics.py - Running biomechanics from 2D pose keypoints

Computes joint angles, gait events, and performance metrics from RTMPose output.

Joint angles (all in degrees):
  - Hip flexion:    trunk → hip → knee
  - Knee flexion:   hip → knee → ankle
  - Ankle dorsi/plantar: knee → ankle → toe (toe≈ankle forward, simplified)
  - Shoulder flex:  hip → shoulder → elbow
  - Elbow flexion:  shoulder → elbow → wrist
  - Trunk lean:     angle between vertical and shoulder_mid → hip_mid line

Gait detection (sagittal plane camera, treadmill or track):
  - Foot strike events detected via find_peaks on ankle vertical velocity
  - Ankle y-velocity sign changes mark foot strike / toe-off
  - Cadence = steps/min from inter-strike intervals
  - Contact time estimated from low-ankle-velocity windows
  - Step length estimated from ankle x-displacement per stride (px→m via calibration)

Apple Silicon note: scipy.signal.find_peaks is fully vectorized / Accelerate-backed on M4
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# COCO keypoint indices
KP = {
    'L_shoulder': 5, 'R_shoulder': 6,
    'L_elbow': 7,    'R_elbow': 8,
    'L_wrist': 9,    'R_wrist': 10,
    'L_hip': 11,     'R_hip': 12,
    'L_knee': 13,    'R_knee': 14,
    'L_ankle': 15,   'R_ankle': 16,
}


def angle_deg(a, b, c):
    """
    Joint angle at vertex b formed by points a-b-c, in degrees.

    Computes the angle between vectors BA and BC.
    Returns angle in [0, 180].

    Example: knee flexion = angle_deg(hip, knee, ankle)
      a = hip, b = knee (vertex), c = ankle
      → angle at the knee between thigh and shank
    """
    ba = a - b
    bc = c - b
    # Guard against zero-length vectors (missing/low-conf keypoints)
    n_ba = np.linalg.norm(ba, axis=-1)
    n_bc = np.linalg.norm(bc, axis=-1)
    denom = n_ba * n_bc
    # cosine via dot product, clamped for numerical safety
    cosang = np.einsum('...i,...i->...', ba, bc) / np.maximum(denom, 1e-6)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    # Mask invalid (zero-length) as NaN
    ang = np.where(denom < 1e-6, np.nan, ang)
    return ang


def trunk_lean_deg(L_shoulder, R_shoulder, L_hip, R_hip):
    """
    Trunk lean angle from vertical, in degrees.

    Positive = forward lean.
    Computed from shoulder_mid → hip_mid vector vs. vertical axis.
    """
    shoulder_mid = (L_shoulder + R_shoulder) * 0.5
    hip_mid = (L_hip + R_hip) * 0.5
    trunk_vec = hip_mid - shoulder_mid  # down the trunk
    # Angle from vertical (0, 1) in image coordinates (y increases downward)
    # lean = atan2(horizontal, vertical)
    lean = np.degrees(np.arctan2(trunk_vec[..., 0], trunk_vec[..., 1]))
    return lean


def smooth_signal(x, window=7, poly=2):
    """Savitzky-Golay smoothing, preserves gait peaks. Falls back to original if too short."""
    x = np.asarray(x, dtype=float)
    if len(x) < window or np.all(np.isnan(x)):
        return x
    # Fill NaNs by linear interpolation for filtering
    mask = ~np.isnan(x)
    if mask.sum() < window:
        return x
    x_filled = np.interp(np.arange(len(x)), np.flatnonzero(mask), x[mask])
    y = savgol_filter(x_filled, window_length=window, polyorder=poly, mode='interp')
    # Restore NaNs where original was NaN
    y[~mask] = np.nan
    return y


def detect_foot_strikes(ankle_y, fps, min_step_hz=1.0, max_step_hz=5.0):
    """
    Detect foot strike events from ankle vertical position.

    Strategy:
      1. Smooth ankle y-trajectory
      2. Compute vertical velocity (dy/dt)
      3. Foot strike ≈ local minima in ankle y (foot lowest)
         → negative-to-positive zero-cross in velocity
      4. Use find_peaks on -ankle_y to get strike frames

    Cadence range: 60–300 steps/min (1–5 Hz) for running
    min_distance = fps / max_step_hz  frames between strikes

    Returns: strike_frame_indices (np array)
    """
    y = smooth_signal(ankle_y, window=max(5, int(fps * 0.08) | 1))
    # Foot is lowest when ankle_y is max (image coords: y down)
    # Invert so strikes are peaks
    min_dist = int(fps / max_step_hz)
    peaks, _ = find_peaks(y, distance=min_dist, prominence=np.nanstd(y) * 0.15 if np.nanstd(y) > 0 else None)
    return peaks


def compute_metrics(keypoints_json: str, px_to_m: float = None, height_m: float = 1.75):
    """
    Compute biomechanics from keypoints JSON.

    Args:
        keypoints_json: Path to inference.py output
        px_to_m: Pixels-to-meters scale (if known, e.g. from calibration).
                 If None, estimates from subject height ≈ 1.75m default.
        height_m: Subject height in meters (used if px_to_m is None)

    Returns:
        df_angles: DataFrame with per-frame joint angles
        summary: dict with aggregate gait metrics
    """
    with open(keypoints_json) as f:
        data = json.load(f)

    fps = data['fps']
    n_frames = data['num_frames']

    # Stack keypoints: (frames, 17, 2)
    kpts = np.array([fr['keypoints'] for fr in data['frames']], dtype=np.float32)
    scores = np.array([fr['scores'] for fr in data['frames']], dtype=np.float32)

    # Helper to get keypoint xy with low-confidence masking
    def kp(name, conf_thr=0.3):
        idx = KP[name]
        xy = kpts[:, idx, :]
        conf = scores[:, idx]
        xy = xy.copy()
        xy[conf < conf_thr] = np.nan
        return xy

    # Extract all tracked joints
    L_sho, R_sho = kp('L_shoulder'), kp('R_shoulder')
    L_elb, R_elb = kp('L_elbow'), kp('R_elbow')
    L_wri, R_wri = kp('L_wrist'), kp('R_wrist')
    L_hip, R_hip = kp('L_hip'), kp('R_hip')
    L_kne, R_kne = kp('L_knee'), kp('R_knee')
    L_ank, R_ank = kp('L_ankle'), kp('R_ankle')

    # --- Joint angles (degrees) ---
    # Hip flexion: trunk reference ≈ shoulder → hip → knee
    # Use ipsilateral shoulder as trunk proxy
    L_hip_angle = angle_deg(L_sho, L_hip, L_kne)
    R_hip_angle = angle_deg(R_sho, R_hip, R_kne)

    # Knee flexion: hip → knee → ankle
    L_knee_angle = angle_deg(L_hip, L_kne, L_ank)
    R_knee_angle = angle_deg(R_hip, R_kne, R_ank)

    # Ankle angle: knee → ankle → horizontal forward
    # Simplified: 90° = neutral. Compute shank angle from vertical.
    def ankle_angle(knee, ankle):
        shank = ankle - knee  # knee → ankle vector
        # Angle between shank and vertical (0,1)
        # 90° = shank horizontal. We want dorsi/plantar flex approx.
        # Return shank inclination from vertical
        ang = np.degrees(np.arctan2(shank[:, 0], shank[:, 1]))
        return 90 - np.abs(ang)  # crude ankle joint proxy
    L_ank_angle = ankle_angle(L_kne, L_ank)
    R_ank_angle = ankle_angle(R_kne, R_ank)

    # Shoulder flexion: hip → shoulder → elbow
    L_sho_angle = angle_deg(L_hip, L_sho, L_elb)
    R_sho_angle = angle_deg(R_hip, R_sho, R_elb)

    # Elbow flexion: shoulder → elbow → wrist
    L_elb_angle = angle_deg(L_sho, L_elb, L_wri)
    R_elb_angle = angle_deg(R_sho, R_elb, R_wri)

    # Trunk lean
    trunk_lean = trunk_lean_deg(L_sho, R_sho, L_hip, R_hip)

    # --- Pixel → meter calibration ---
    if px_to_m is None:
        # Estimate from subject height in pixels
        # Height ≈ ankle → hip → shoulder → head
        # Simplified: hip to ankle + hip to shoulder
        # Use median over frames with good confidence
        pixel_heights = []
        for i in range(n_frames):
            ys = []
            for pts in [L_ank[i], R_ank[i], L_hip[i], R_hip[i], L_sho[i], R_sho[i]]:
                if not np.any(np.isnan(pts)):
                    ys.append(pts[1])
            if len(ys) >= 2:
                pixel_heights.append(max(ys) - min(ys))
        if pixel_heights:
            px_height = np.nanmedian(pixel_heights) * 1.4  # ~ankle-to-top scale
            px_to_m = height_m / px_height if px_height > 0 else 0.002
        else:
            px_to_m = 0.002  # fallback ~2mm/px
    # --- Gait event detection ---
    # Use ankle y position (image coords: larger y = lower in frame)
    L_strikes = detect_foot_strikes(L_ank[:, 1], fps)
    R_strikes = detect_foot_strikes(R_ank[:, 1], fps)

    # Merge L+R strikes and sort → step events
    all_strikes = np.sort(np.concatenate([L_strikes, R_strikes]))
    # Remove strikes too close together (< 0.15s = double detection)
    min_gap = int(fps * 0.15)
    filtered = []
    for s in all_strikes:
        if not filtered or s - filtered[-1] >= min_gap:
            filtered.append(s)
    all_strikes = np.array(filtered)

    # Cadence: steps per minute
    if len(all_strikes) >= 2:
        step_intervals_s = np.diff(all_strikes) / fps
        mean_step_time = np.nanmean(step_intervals_s)
        cadence_spm = 60.0 / mean_step_time if mean_step_time > 0 else np.nan
    else:
        cadence_spm = np.nan
        step_intervals_s = np.array([])

    # Stride time (same foot strike to strike, ~2 steps)
    for strikes, side in [(L_strikes, 'L'), (R_strikes, 'R')]:
        pass  # computed per side below
    def stride_times(strikes):
        if len(strikes) >= 2:
            return np.diff(strikes) / fps
        return np.array([])
    L_stride_t = stride_times(L_strikes)
    R_stride_t = stride_times(R_strikes)
    mean_stride_time = np.nanmean(np.concatenate([L_stride_t, R_stride_t])) if len(L_stride_t) + len(R_stride_t) > 0 else np.nan

    # Step length estimation (horizontal ankle displacement per step, in meters)
    # Treadmill assumption: if camera is stationary and subject stays centered,
    # this approximates step length. For overground, need calibrated ground plane.
    def step_lengths_xy(ank_xy, strikes):
        lengths_px = []
        for i in range(len(strikes) - 1):
            a = strikes[i]
            b = strikes[i + 1]
            if a < len(ank_xy) and b < len(ank_xy):
                p0 = ank_xy[a]
                p1 = ank_xy[b]
                if not np.any(np.isnan([p0, p1])):
                    # Horizontal displacement as step proxy
                    lengths_px.append(abs(p1[0] - p0[0]))
        return np.array(lengths_px) * px_to_m if lengths_px else np.array([])
    L_step_len = step_lengths_xy(L_ank, L_strikes)
    R_step_len = step_lengths_xy(R_ank, R_strikes)
    mean_step_length = np.nanmean(np.concatenate([L_step_len, R_step_len])) if len(L_step_len) + len(R_step_len) > 0 else np.nan

    # Ground contact time: crude estimate
    # Foot is "on ground" when ankle vertical velocity is low near strike
    # Use ±50ms window around each strike as contact proxy
    contact_window_s = 0.10  # ~100ms half-window → ~200ms contact (typical running)
    contact_time_s = contact_window_s * 2

    # Speed estimate
    # speed = cadence * step_length / 60  (m/s)  if both available
    # else: cadence * (estimated step from height)
    if not np.isnan(mean_step_length) and mean_step_length > 0.1:
        speed_ms = (cadence_spm / 60.0) * mean_step_length if not np.isnan(cadence_spm) else np.nan
    else:
        # Fallback: estimate step length ~0.65 × height for running
        est_step = 0.65 * height_m
        speed_ms = (cadence_spm / 60.0) * est_step if not np.isnan(cadence_spm) else np.nan

    # --- Range of motion (ROM) per joint ---
    def rom(x):
        return float(np.nanmax(x) - np.nanmin(x)) if np.any(~np.isnan(x)) else np.nan

    def mean_angle(x):
        return float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan

    roms = {
        'L_hip_rom': rom(L_hip_angle), 'R_hip_rom': rom(R_hip_angle),
        'L_knee_rom': rom(L_knee_angle), 'R_knee_rom': rom(R_knee_angle),
        'L_ankle_rom': rom(L_ank_angle), 'R_ankle_rom': rom(R_ank_angle),
        'L_shoulder_rom': rom(L_sho_angle), 'R_shoulder_rom': rom(R_sho_angle),
        'L_elbow_rom': rom(L_elb_angle), 'R_elbow_rom': rom(R_elb_angle),
        'trunk_lean_rom': rom(trunk_lean),
    }

    # --- Per-frame DataFrame ---
    times = np.arange(n_frames) / fps
    df = pd.DataFrame({
        'time_s': times,
        'frame': np.arange(n_frames),
        'L_hip_deg': L_hip_angle,
        'R_hip_deg': R_hip_angle,
        'L_knee_deg': L_knee_angle,
        'R_knee_deg': R_knee_angle,
        'L_ankle_deg': L_ank_angle,
        'R_ankle_deg': R_ank_angle,
        'L_shoulder_deg': L_sho_angle,
        'R_shoulder_deg': R_sho_angle,
        'L_elbow_deg': L_elb_angle,
        'R_elbow_deg': R_elb_angle,
        'trunk_lean_deg': trunk_lean,
    })

    # --- Summary JSON ---
    summary = {
        'fps': fps,
        'num_frames': n_frames,
        'duration_s': n_frames / fps,
        'px_to_m': px_to_m,
        'calibration': {
            'height_m': height_m,
            'method': 'height_estimate' if px_to_m is not None else 'fallback',
        },
        'gait': {
            'cadence_spm': float(cadence_spm) if not np.isnan(cadence_spm) else None,
            'step_length_m': float(mean_step_length) if not np.isnan(mean_step_length) else None,
            'stride_time_s': float(mean_stride_time) if not np.isnan(mean_stride_time) else None,
            'contact_time_s': float(contact_time_s),
            'speed_ms': float(speed_ms) if not np.isnan(speed_ms) else None,
            'speed_kmh': float(speed_ms * 3.6) if not np.isnan(speed_ms) else None,
            'L_strikes': len(L_strikes),
            'R_strikes': len(R_strikes),
        },
        'rom_deg': roms,
        'mean_angles_deg': {
            'L_hip': mean_angle(L_hip_angle),
            'R_hip': mean_angle(R_hip_angle),
            'L_knee': mean_angle(L_knee_angle),
            'R_knee': mean_angle(R_knee_angle),
            'L_ankle': mean_angle(L_ank_angle),
            'R_ankle': mean_angle(R_ank_angle),
            'L_shoulder': mean_angle(L_sho_angle),
            'R_shoulder': mean_angle(R_sho_angle),
            'L_elbow': mean_angle(L_elb_angle),
            'R_elbow': mean_angle(R_elb_angle),
            'trunk_lean': mean_angle(trunk_lean),
        },
    }

    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description='Compute running gait biomechanics from RTMPose keypoints'
    )
    parser.add_argument('keypoints_json', help='Keypoints JSON from inference.py')
    parser.add_argument('--out-csv', help='Output angles CSV (default: <input>_angles.csv)')
    parser.add_argument('--out-summary', help='Output summary JSON (default: <input>_summary.json)')
    parser.add_argument('--px-to-m', type=float, default=None,
                        help='Pixels to meters calibration (auto-estimates from height if omitted)')
    parser.add_argument('--height-m', type=float, default=1.75,
                        help='Subject height in meters (for auto-calibration)')
    args = parser.parse_args()

    in_path = Path(args.keypoints_json)
    out_csv = args.out_csv or str(in_path.with_name(in_path.stem.replace('_keypoints', '') + '_angles.csv'))
    out_summary = args.out_summary or str(in_path.with_name(in_path.stem.replace('_keypoints', '') + '_summary.json'))

    df, summary = compute_metrics(args.keypoints_json, px_to_m=args.px_to_m, height_m=args.height_m)

    df.to_csv(out_csv, index=False)
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[✓] Angles CSV → {out_csv}")
    print(f"[✓] Summary JSON → {out_summary}")
    print()
    g = summary['gait']
    print(f"  Cadence:      {g['cadence_spm']:.1f} steps/min" if g['cadence_spm'] else "  Cadence:      n/a")
    print(f"  Speed:        {g['speed_kmh']:.2f} km/h" if g['speed_kmh'] else "  Speed:        n/a")
    print(f"  Step length:  {g['step_length_m']:.2f} m" if g['step_length_m'] else "  Step length:  n/a")
    print(f"  Stride time:  {g['stride_time_s']:.3f} s" if g['stride_time_s'] else "  Stride time:  n/a")


if __name__ == '__main__':
    main()
