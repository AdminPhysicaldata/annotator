from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _derivative(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.empty_like(x)
    dx[:] = np.nan
    if len(x) < 2:
        return dx
    dx[1:] = (x[1:] - x[:-1]) / dt
    dx[0] = dx[1]
    return dx


def build_sensor_features(
    aligned_trackers: pd.DataFrame,
    aligned_pince1: pd.DataFrame | None = None,
    aligned_pince2: pd.DataFrame | None = None,
    *,
    fps: float,
    include_quat: bool = False,
) -> pd.DataFrame:
    """Build frame-level features from aligned sensor streams.

    Inputs must already be aligned to the same timeline with a 'time_seconds' column.

    Trackers: expects tracker_{i}_{x,y,z} and optionally quats.
    Pinces: expects angle_deg column.

    Output: DataFrame with 'time_seconds' and engineered features:
      - per-tracker position, velocity, acceleration norms
      - optional quaternions
      - gripper angles + angular velocity
      - cross distances between trackers
    """
    dt = 1.0 / float(fps)

    feats = pd.DataFrame({"time_seconds": aligned_trackers["time_seconds"].to_numpy()})

    # Tracker features.
    trackers = [1, 2, 3]
    for i in trackers:
        for axis in ["x", "y", "z"]:
            col = f"tracker_{i}_{axis}"
            if col in aligned_trackers.columns:
                feats[col] = aligned_trackers[col].to_numpy(dtype=np.float64)

        # speed/acc norms
        pos_cols = [f"tracker_{i}_x", f"tracker_{i}_y", f"tracker_{i}_z"]
        if all(c in feats.columns for c in pos_cols):
            p = feats[pos_cols].to_numpy(dtype=np.float64)
            v = _derivative(p, dt)
            a = _derivative(v, dt)
            feats[f"tracker_{i}_speed"] = np.linalg.norm(v, axis=1)
            feats[f"tracker_{i}_acc"] = np.linalg.norm(a, axis=1)

        if include_quat:
            for q in ["qw", "qx", "qy", "qz"]:
                col = f"tracker_{i}_{q}"
                if col in aligned_trackers.columns:
                    feats[col] = aligned_trackers[col].to_numpy(dtype=np.float64)

    # Distances between trackers.
    def _dist(i: int, j: int) -> None:
        a = feats[[f"tracker_{i}_x", f"tracker_{i}_y", f"tracker_{i}_z"]].to_numpy(dtype=np.float64)
        b = feats[[f"tracker_{j}_x", f"tracker_{j}_y", f"tracker_{j}_z"]].to_numpy(dtype=np.float64)
        feats[f"dist_{i}_{j}"] = np.linalg.norm(a - b, axis=1)

    if all(f"tracker_{k}_x" in feats.columns for k in [1, 2, 3]):
        _dist(1, 2)
        _dist(1, 3)
        _dist(2, 3)

    # Grippers.
    def _add_pince(prefix: str, df: pd.DataFrame | None) -> None:
        if df is None:
            return
        if "angle_deg" not in df.columns:
            # allow different name: first numeric col besides time
            cols = [c for c in df.columns if c != "time_seconds"]
            if not cols:
                return
            angle_col = cols[0]
        else:
            angle_col = "angle_deg"

        ang = df[angle_col].to_numpy(dtype=np.float64)
        feats[f"{prefix}_angle_deg"] = ang
        feats[f"{prefix}_ang_vel"] = _derivative(ang, dt)

    _add_pince("pince1", aligned_pince1)
    _add_pince("pince2", aligned_pince2)

    # Global motion energy proxy.
    speed_cols = [c for c in feats.columns if c.endswith("_speed")]
    acc_cols = [c for c in feats.columns if c.endswith("_acc")]
    if speed_cols:
        feats["motion_speed_sum"] = feats[speed_cols].sum(axis=1)
    if acc_cols:
        feats["motion_acc_sum"] = feats[acc_cols].sum(axis=1)

    return feats


def build_video_features(
    video_path: str,
    timeline: "Timeline",
    *,
    resize: tuple[int, int] | None = (160, 90),
) -> pd.DataFrame:
    """Extract frame-level visual features from a video, aligned to a Timeline.

    For each frame of the timeline (by nearest video frame), computes:
      - video_brightness  : mean pixel intensity [0..255]
      - video_blur        : Laplacian variance (focus measure)
      - video_frame_diff  : mean absolute difference with previous frame

    Parameters
    ----------
    video_path:
        Path to the video file.
    timeline:
        The shared Timeline (from build_timeline).
    resize:
        Optional (width, height) to resize frames before computing features.
        Reduces computation time significantly.

    Returns
    -------
    DataFrame with 'time_seconds' matching the timeline and the three feature columns.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for build_video_features: pip install opencv-python") from e

    from pathlib import Path

    cap = cv2.VideoCapture(str(Path(video_path)))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS in video: {video_fps}")

    # Map each timeline timestamp to the nearest video frame index.
    frame_indices = np.clip(
        np.round(timeline.t * video_fps).astype(np.int64), 0, total_frames - 1
    )

    n = len(timeline.t)
    brightness = np.full(n, np.nan)
    blur = np.full(n, np.nan)
    frame_diff = np.full(n, np.nan)

    prev_gray: np.ndarray | None = None
    current_frame_idx = -1

    for out_idx, vid_idx in enumerate(frame_indices):
        # Seek only if needed (sequential read is faster when possible).
        if vid_idx != current_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(vid_idx))

        ret, frame = cap.read()
        if not ret:
            prev_gray = None
            current_frame_idx = vid_idx
            continue

        current_frame_idx = vid_idx

        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        brightness[out_idx] = float(gray.mean())
        blur[out_idx] = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        if prev_gray is not None:
            frame_diff[out_idx] = float(np.mean(np.abs(gray - prev_gray)))

        prev_gray = gray

    cap.release()

    return pd.DataFrame(
        {
            "time_seconds": timeline.t,
            "video_brightness": brightness,
            "video_blur": blur,
            "video_frame_diff": frame_diff,
        }
    )


def segment_level_features(
    frame_features: pd.DataFrame,
    segments: list[dict],
    *,
    exclude_cols: tuple[str, ...] = ("time_seconds",),
) -> np.ndarray:
    """Aggregate frame features into one vector per segment.

    Uses mean/std/min/max on each feature.
    """
    cols = [c for c in frame_features.columns if c not in exclude_cols]
    X = frame_features[cols].to_numpy(dtype=np.float64)

    out = []
    for s in segments:
        a, b = int(s["start_idx"]), int(s["end_idx"])
        a = max(a, 0)
        b = min(b, len(X) - 1)
        if b < a:
            out.append(np.zeros((len(cols) * 4,), dtype=np.float64))
            continue
        seg = X[a : b + 1]
        # handle nans
        m = np.nanmean(seg, axis=0)
        sd = np.nanstd(seg, axis=0)
        mn = np.nanmin(seg, axis=0)
        mx = np.nanmax(seg, axis=0)
        out.append(np.concatenate([m, sd, mn, mx], axis=0))

    return np.vstack(out)
