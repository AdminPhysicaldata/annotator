"""Helper utilities for VIVE Labeler."""

import numpy as np
from typing import Union, Tuple
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Format timestamp in seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize and expand path.

    Args:
        path: Input path

    Returns:
        Normalized Path object
    """
    path = Path(path)
    return path.expanduser().resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = normalize_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max.

    Args:
        value: Input value
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_linear(t: float, t0: float, t1: float,
                       v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Linear interpolation between two values at time t.

    Args:
        t: Target time
        t0: Time of first value
        t1: Time of second value
        v0: First value
        v1: Second value

    Returns:
        Interpolated value
    """
    if t1 == t0:
        return v0
    alpha = (t - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex digest of hash
    """
    import hashlib

    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe (reliable) with OpenCV fallback.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info (fps, frame_count, width, height, duration)
    """
    import cv2
    import subprocess
    import json

    # Try ffprobe first (more reliable for frame count)
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,r_frame_rate,width,height,duration",
                "-of", "json",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if "streams" in data and len(data["streams"]) > 0:
                stream = data["streams"][0]
                # Parse frame rate (e.g., "30/1" -> 30.0)
                fps_str = stream.get("r_frame_rate", "30/1")
                fps_parts = fps_str.split("/")
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

                frame_count = int(stream.get("nb_frames", 0))
                duration = float(stream.get("duration", 0))

                return {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": int(stream.get("width", 0)),
                    "height": int(stream.get("height", 0)),
                    "duration": duration
                }
    except Exception:
        pass  # Fallback to OpenCV

    # Fallback to OpenCV (less reliable for frame_count)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0.0

    cap.release()
    return info
