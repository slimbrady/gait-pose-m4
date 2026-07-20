#!/usr/bin/env python3
"""
inference.py - Running gait pose estimation for Apple Silicon M4

Pose estimation backend: RTMPose via rtmlib (ONNX Runtime)
  - Runs CPU-only, no CUDA required → perfect for Apple Silicon (M1/M2/M3/M4)
  - RTMPose is lightweight and real-time capable on MacBook M4
  - Model auto-downloads on first run via rtmlib

COCO 17 keypoint indices used:
  0: nose        5: L_shoulder  9: L_wrist   13: L_knee
  1: L_eye      6: R_shoulder 10: R_wrist   14: R_knee
  2: R_eye      7: L_elbow    11: L_hip     15: L_ankle
  3: L_ear      8: R_elbow    12: R_hip     16: R_ankle
  4: R_ear

Joint angles tracked: hips, knees, ankles, shoulders, elbows, trunk lean
"""

import argparse
import json
import cv2
from pathlib import Path
import numpy as np

try:
    from rtmlib import Body, PoseTracker, draw_skeleton as rtm_draw_skeleton
except ImportError:
    print("ERROR: rtmlib not installed. Run: pip install rtmlib onnxruntime")
    exit(1)

# COCO 17 keypoint names for reference
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Skeleton edges for drawing (COCO format)
SKELETON = [
    (5, 7), (7, 9),      # L arm: shoulder → elbow → wrist
    (6, 8), (8, 10),     # R arm
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),  # L leg: hip → knee → ankle
    (12, 14), (14, 16),  # R leg
]


def draw_skeleton_cv(frame, keypoints, scores, score_thr=0.3):
    """
    Draw pose skeleton overlay on a frame.
    keypoints: (17, 2) array of x,y coords
    scores: (17,) confidence scores
    """
    vis = frame.copy()

    # Draw bones
    for i, j in SKELETON:
        if i < len(scores) and j < len(scores) and scores[i] > score_thr and scores[j] > score_thr:
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in enumerate(keypoints):
        if idx < len(scores) and scores[idx] > score_thr:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 165, 255), -1, cv2.LINE_AA)

    return vis


def run_inference(
    video_path: str,
    out_json: str = None,
    out_video: str = None,
    device: str = 'cpu',
    det_score_thr: float = 0.5,
):
    """
    Run RTMPose on a running video.

    Apple Silicon notes:
      - device='cpu' uses ONNX Runtime CPU provider
      - On M4, ONNX Runtime leverages Apple's Accelerate framework automatically
      - No CUDA/ROCm needed. Just `pip install onnxruntime rtmlib`
      - For best M4 throughput: close other heavy apps, use conda-forge OpenCV

    Args:
        video_path: Input .mp4 / .mov running video
        out_json: Output keypoints JSON path (default: <video>_keypoints.json)
        out_video: Output skeleton-overlay MP4 path (default: <video>_pose.mp4)
        device: 'cpu' (recommended for M4) or 'cuda' if available
        det_score_thr: detection confidence threshold (passed to draw filter, PoseTracker handles det internally)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Default output paths
    if out_json is None:
        out_json = video_path.with_name(video_path.stem + '_keypoints.json')
    if out_video is None:
        out_video = video_path.with_name(video_path.stem + '_pose.mp4')

    print(f"[*] Loading RTMPose models (first run downloads weights)...")
    print(f"    Device: {device}  |  Apple Silicon M4: use 'cpu' (no CUDA needed)")

    # PoseTracker(Body) = YOLOX detector + RTMPose estimator, all-in-one
    # mode='balanced' ≈ RTMPose-m accuracy/speed tradeoff, good for M4
    # modes: 'performance' (bigger, slower), 'balanced', 'lightweight' (fastest)
    pose_tracker = PoseTracker(
        Body,
        mode='balanced',
        backend='onnxruntime',
        device=device
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[*] Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Video writer for skeleton overlay
    # Use 'avc1' for better browser/Streamlit compatibility on macOS
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_wr = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
    if not out_wr.isOpened():
        # Fall back to mp4v if avc1 codec unavailable
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_wr = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    frames_kpts = []
    frame_idx = 0

    print("[*] Running pose estimation...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # PoseTracker does detection + pose estimation in one call
        # Returns: keypoints (N, 17, 2), scores (N, 17)
        keypoints, scores = pose_tracker(frame)

        if keypoints is None or len(keypoints) == 0 or scores is None or len(scores) == 0:
            # No person detected: write empty keypoints
            kpts = np.zeros((17, 2), dtype=np.float32)
            scrs = np.zeros(17, dtype=np.float32)
            vis_frame = frame
        else:
            # Take first / largest person (PoseTracker usually returns 1)
            kpts = keypoints[0]
            scrs = scores[0]
            # Draw skeleton overlay
            try:
                # Try rtmlib's built-in draw first
                vis_frame = rtm_draw_skeleton(frame, keypoints, scores, kpt_thr=det_score_thr)
            except Exception:
                vis_frame = draw_skeleton_cv(frame, kpts, scrs, score_thr=det_score_thr)

        out_wr.write(vis_frame)

        # Store frame keypoints
        frames_kpts.append({
            'frame': frame_idx,
            'time_s': frame_idx / fps,
            'keypoints': kpts.tolist(),  # [[x, y], ...] × 17
            'scores': scrs.tolist(),
        })

        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            print(f"  {frame_idx}/{total_frames} frames")

    cap.release()
    out_wr.release()

    # Write keypoints JSON
    output = {
        'video': str(video_path),
        'fps': fps,
        'width': width,
        'height': height,
        'num_frames': frame_idx,
        'keypoint_names': COCO_KEYPOINTS,
        'frames': frames_kpts,
    }
    with open(out_json, 'w') as f:
        json.dump(output, f)

    print(f"[✓] Keypoints → {out_json}")
    print(f"[✓] Overlay video → {out_video}")

    return str(out_json), str(out_video)


def main():
    parser = argparse.ArgumentParser(
        description='RTMPose running gait inference (optimized for Apple Silicon M4)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('video', help='Input running video (.mp4 / .mov)')
    parser.add_argument('--out-json', help='Output keypoints JSON (default: <video>_keypoints.json)')
    parser.add_argument('--out-video', help='Output skeleton-overlay MP4 (default: <video>_pose.mp4)')
    parser.add_argument('--device', default='cpu',
                        help='onnxruntime device: cpu (recommended for M4) or cuda')
    parser.add_argument('--det-thr', type=float, default=0.5,
                        help='keypoint confidence threshold for visualization')
    args = parser.parse_args()

    run_inference(
        args.video,
        out_json=args.out_json,
        out_video=args.out_video,
        device=args.device,
        det_score_thr=args.det_thr,
    )


if __name__ == '__main__':
    main()
