from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    w = np.ones(k, dtype=np.float64) / float(k)
    y = np.convolve(np.nan_to_num(x, nan=0.0), w, mode="same")
    return y


def heuristic_segments(
    frame_features: pd.DataFrame,
    *,
    fps: float,
    motion_col: str = "motion_speed_sum",
    grip_cols: tuple[str, ...] = ("pince1_ang_vel", "pince2_ang_vel"),
    smooth_ms: int = 120,
    thr_motion: float = 0.15,
    thr_grip: float = 2.0,
    min_action_ms: int = 200,
    min_gap_ms: int = 150,
) -> List[Dict]:
    """Bootstrap segmentation with a deterministic heuristic.

    Idea:
      - compute an activity score from motion + gripper angular velocity
      - threshold with hysteresis-like gap merging

    Returns list of segments with indices and timestamps.

    Notes:
      - thresholds are data-dependent. Tune on a few minutes then freeze.
    """
    dt = 1.0 / float(fps)
    n = len(frame_features)
    if n == 0:
        return []

    motion = frame_features[motion_col].to_numpy(dtype=np.float64) if motion_col in frame_features.columns else np.zeros(n)

    grip = np.zeros(n, dtype=np.float64)
    for c in grip_cols:
        if c in frame_features.columns:
            grip += np.abs(frame_features[c].to_numpy(dtype=np.float64))

    # Normalize-ish (robust) to make thresholds less fragile.
    def _robust_scale(a: np.ndarray) -> np.ndarray:
        a2 = np.nan_to_num(a, nan=0.0)
        p50 = np.percentile(a2, 50)
        p90 = np.percentile(a2, 90)
        denom = (p90 - p50) if (p90 - p50) > 1e-9 else 1.0
        return (a2 - p50) / denom

    motion_s = _robust_scale(motion)
    grip_s = _robust_scale(grip)

    k = max(1, int(round((smooth_ms / 1000.0) * fps)))
    score = _smooth(np.maximum(motion_s, 0.0) + 0.7 * np.maximum(grip_s, 0.0), k)

    active = (score > thr_motion) | (grip_s > thr_grip)

    # Build segments from active mask.
    segments = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        end = i - 1
        segments.append((start, end))

    if not segments:
        return []

    # Merge small gaps.
    min_gap = int(round((min_gap_ms / 1000.0) * fps))
    merged = [segments[0]]
    for (s, e) in segments[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= min_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Drop too-short segments.
    min_len = int(round((min_action_ms / 1000.0) * fps))
    merged2 = [(s, e) for (s, e) in merged if (e - s + 1) >= min_len]

    t = frame_features["time_seconds"].to_numpy(dtype=np.float64)
    out = []
    for idx, (s, e) in enumerate(merged2):
        out.append(
            {
                "segment_id": idx,
                "start_idx": int(s),
                "end_idx": int(e),
                "start_t": float(t[s]),
                "end_t": float(t[e]),
                "duration_s": float(t[e] - t[s]),
                "label": None,
                "score_mean": float(np.nanmean(score[s : e + 1])),
            }
        )

    return out


def cluster_segments_kmeans(
    seg_features: np.ndarray,
    segments: List[Dict],
    *,
    n_labels: int,
    random_state: int = 0,
) -> List[Dict]:
    """Assign X labels by clustering segment-level features.

    Mutates and returns the segments list (label becomes int in [0..n_labels-1]).
    """
    if n_labels <= 0:
        raise ValueError("n_labels must be > 0")
    if len(segments) == 0:
        return segments
    if n_labels > len(segments):
        # Cannot cluster into more clusters than samples.
        n_labels = len(segments)
    if seg_features.shape[0] != len(segments):
        raise ValueError("seg_features rows must match number of segments")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(np.nan_to_num(seg_features, nan=0.0))

    kmeans = KMeans(n_clusters=int(n_labels), n_init="auto", random_state=int(random_state))
    y = kmeans.fit_predict(Z)

    for s, lab in zip(segments, y.tolist()):
        s["label"] = int(lab)

    return segments


def export_segments_csv(segments: List[Dict], path: str) -> None:
    """Export segments to CSV for manual correction."""
    df = pd.DataFrame(segments)
    df.to_csv(path, index=False)
