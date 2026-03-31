from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


_PINCE_ANGLE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def load_trackers_csv(path: str | Path) -> pd.DataFrame:
    """Load Vive tracker CSV.

    Expected columns (as in your sample):
      - time_seconds (float)
      - frame_number (int)
      - tracker_{i}_{x,y,z,qw,qx,qy,qz} for i=1..3

    Returns a DataFrame sorted by time_seconds.
    """
    df = pd.read_csv(path)
    if "time_seconds" not in df.columns:
        raise ValueError("tracker CSV must contain 'time_seconds'")
    df = df.sort_values("time_seconds").reset_index(drop=True)
    return df


def load_pince_csv(path: str | Path, *, angle_col: str = "angle_deg") -> pd.DataFrame:
    """Load gripper (pince) CSV and parse angle from raw_data.

    Expected columns (as in your sample):
      - time_seconds
      - raw_data containing an angle (e.g. 'Angle relatif : -0.79 deg')

    Adds a numeric angle column (default: angle_deg) and returns sorted DataFrame.
    """
    df = pd.read_csv(path)
    if "time_seconds" not in df.columns:
        raise ValueError("pince CSV must contain 'time_seconds'")

    if "raw_data" not in df.columns:
        raise ValueError("pince CSV must contain 'raw_data'")

    def _parse_angle(x: str) -> float:
        if not isinstance(x, str):
            return float("nan")
        m = _PINCE_ANGLE_RE.search(x)
        return float(m.group(1)) if m else float("nan")

    df[angle_col] = df["raw_data"].map(_parse_angle)
    df = df.sort_values("time_seconds").reset_index(drop=True)
    return df[["time_seconds", angle_col]].copy()


def load_video(path: str | Path, *, t_offset: float = 0.0) -> pd.DataFrame:
    """Load a video file and return a DataFrame of frame timestamps.

    Requires OpenCV (cv2). Each row corresponds to one video frame.

    Parameters
    ----------
    path:
        Path to the video file (mp4, avi, mov, …).
    t_offset:
        Optional time offset in seconds to add to all timestamps.
        Use this to align the video clock to the sensor clock.

    Returns
    -------
    DataFrame with columns:
      - time_seconds  : absolute timestamp of each frame (float)
      - frame_number  : 0-based frame index (int)
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for load_video: pip install opencv-python") from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) in video: {path}")

    frame_numbers = np.arange(total_frames, dtype=np.int64)
    timestamps = frame_numbers / fps + t_offset

    return pd.DataFrame({"time_seconds": timestamps, "frame_number": frame_numbers})
