"""Session data loader for local recording sessions.

Loads data from a session directory containing:
- metadata.json: session configuration
- videos/head.mp4, videos/left.mp4, videos/right.mp4: camera videos
- videos/head.jsonl, videos/left.jsonl, videos/right.jsonl: frame timestamps
- tracker_positions.csv: VIVE tracker 3D positions + quaternions
  Columns: tracker_head_x/y/z/qw/qx/qy/qz, tracker_left_*, tracker_right_*
- gripper_left_data.csv, gripper_right_data.csv: gripper data
  Columns: timestamp, time_seconds, opening_mm, angle_deg, ...

All data streams are aligned to a common timestamp base derived from
the session start_time in metadata.json.
"""

import json
import logging
import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress noisy ffmpeg/libavcodec warnings (e.g. "mjpeg unable to decode APP fields")
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

# File-descriptor-level stderr suppression for native C library messages
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
_real_stderr_fd = os.dup(2)


def _suppress_stderr():
    """Redirect C-level stderr to /dev/null."""
    os.dup2(_devnull_fd, 2)


def _restore_stderr():
    """Restore C-level stderr."""
    os.dup2(_real_stderr_fd, 2)


@dataclass
class TrackerState:
    """State of all trackers at a given timestamp."""
    timestamp: float  # seconds since session start
    positions: Dict[str, np.ndarray]  # tracker_name -> [x, y, z]
    quaternions: Dict[str, np.ndarray]  # tracker_name -> [qw, qx, qy, qz]


@dataclass
class GripperState:
    """State of a gripper at a given timestamp."""
    timestamp: float
    opening_mm: float


@dataclass
class SessionMetadata:
    """Metadata for a recording session."""
    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    video_width: int
    video_height: int
    video_fps: int
    # camera_positions: ["head", "left", "right"] (ordered by camera ID)
    camera_positions: List[str]
    # tracker_names: ["head", "left", "right"] (or whatever trackers exist)
    tracker_names: List[str]
    # gripper_sides: ["left", "right"]
    gripper_sides: List[str]
    # scenario_name: optional human-readable scenario label from metadata.json
    scenario_name: str = ""


@dataclass
class CameraInfo:
    """Info about a loaded camera video."""
    position: str           # "head", "left", "right"
    capture: cv2.VideoCapture
    frame_count: int
    fps: float
    width: int
    height: int
    video_path: Optional[Path] = None   # absolute path to the video file
    video_offset: float = 0.0   # mono_offset_from_record: delay between session start and frame 0
    # frame_ms_offsets[i] = milliseconds from start of video file for frame i (0-based).
    # Loaded from videos/{position}.jsonl. None if jsonl is unavailable.
    frame_ms_offsets: Optional[np.ndarray] = None
    # frame_capture_ns[i] = absolute capture_time in nanoseconds for frame i.
    # Derived from jsonl capture_time (ms) * 1e6. None if jsonl is unavailable.
    frame_capture_ns: Optional[np.ndarray] = None


class SessionDataLoader:
    """Loads and synchronizes all data streams from a recording session."""

    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        self.metadata = self._load_metadata()

        try:
            self._ref_time = pd.Timestamp(self.metadata.start_time)
        except Exception as exc:
            logger.warning("Cannot parse start_time '%s': %s — using epoch", self.metadata.start_time, exc)
            self._ref_time = pd.Timestamp("1970-01-01T00:00:00")

        self.tracker_df = self._load_tracker_data()
        self.gripper_dfs: Dict[str, pd.DataFrame] = self._load_gripper_data()

        self._tracker_timestamps: Optional[np.ndarray] = None
        self._tracker_timestamps_ns: Optional[np.ndarray] = None
        self._tracker_arrays: Dict[str, np.ndarray] = {}  # "tracker_head_x" -> array
        self._prepare_tracker_arrays()

        self._gripper_timestamps: Dict[str, np.ndarray] = {}
        self._gripper_openings: Dict[str, np.ndarray] = {}
        self._prepare_gripper_arrays()

        self.cameras: Dict[str, CameraInfo] = self._load_cameras()

        # Use actual camera FPS instead of metadata FPS (which may be incorrect)
        camera_fps_values = [c.fps for c in self.cameras.values() if c.fps > 0]
        if camera_fps_values:
            # Use the most common FPS value from cameras
            self.fps = max(set(camera_fps_values), key=camera_fps_values.count)
        elif self.metadata.video_fps > 0:
            self.fps = self.metadata.video_fps
        else:
            self.fps = 30

        self.frame_count = self._compute_frame_count()
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

        logger.info(
            "Session %s loaded: %d frames @ %d FPS, %.2fs, %d cameras (%s), %d trackers (%s), %d grippers (%s)",
            self.metadata.session_id,
            self.frame_count,
            self.fps,
            self.duration,
            len(self.cameras),
            list(self.cameras.keys()),
            len(self.metadata.tracker_names),
            self.metadata.tracker_names,
            len(self.gripper_dfs),
            list(self.gripper_dfs.keys()),
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(self) -> SessionMetadata:
        meta_path = self.session_dir / "metadata.json"

        raw: dict = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            logger.warning("metadata.json not found at %s — using empty metadata", meta_path)
        except json.JSONDecodeError as exc:
            logger.error("metadata.json is malformed: %s — using empty metadata", exc)
        except Exception as exc:
            logger.error("Cannot read metadata.json: %s — using empty metadata", exc)

        if not isinstance(raw, dict):
            raw = {}

        # camera_anchors: {position: {mono_offset_from_record: float}}
        camera_anchors_raw = raw.get("camera_anchors", {})
        self._camera_anchors: Dict[str, float] = {}
        if isinstance(camera_anchors_raw, dict):
            for key, anchor in camera_anchors_raw.items():
                if isinstance(anchor, dict):
                    offset = anchor.get("mono_offset_from_record", 0.0)
                    try:
                        self._camera_anchors[str(key)] = float(offset)
                    except (TypeError, ValueError):
                        self._camera_anchors[str(key)] = 0.0

        # Cameras: extract position names from cameras dict
        # metadata.cameras = {"0": {"name": ..., "position": "head", ...}, ...}
        cameras_raw = raw.get("cameras", {}) if isinstance(raw.get("cameras"), dict) else {}
        camera_positions: List[str] = []
        self._camera_id_to_position: Dict[str, str] = {}  # "0" -> "head"
        for cam_id in sorted(cameras_raw.keys()):
            cam_info = cameras_raw[cam_id]
            if isinstance(cam_info, dict):
                pos = cam_info.get("position", cam_id)
            else:
                pos = cam_id
            self._camera_id_to_position[str(cam_id)] = str(pos)
            camera_positions.append(str(pos))

        # Trackers: extract tracker names from trackers dict keys or positions
        # metadata.trackers = {"1": {"serial": ..., "model": ...}, ...}
        # Tracker column names in CSV are tracker_head_*, tracker_left_*, tracker_right_*
        # We discover tracker names from the CSV columns at load time.
        # For now, store the raw tracker IDs — they'll be resolved against CSV columns.
        trackers_raw = raw.get("trackers", {}) if isinstance(raw.get("trackers"), dict) else {}
        tracker_names: List[str] = sorted(trackers_raw.keys())

        # Grippers: metadata.grippers = {"right": {...}, "left": {...}}
        grippers_raw = raw.get("grippers", {})
        if not isinstance(grippers_raw, dict):
            # Fallback: old "pinces" key
            grippers_raw = raw.get("pinces", {}) if isinstance(raw.get("pinces"), dict) else {}
        gripper_sides: List[str] = sorted(grippers_raw.keys()) if grippers_raw else []

        vc = raw.get("video_config", {}) if isinstance(raw.get("video_config"), dict) else {}

        return SessionMetadata(
            session_id=str(raw.get("session_id", "unknown")),
            start_time=str(raw.get("start_time", "1970-01-01T00:00:00")),
            end_time=str(raw.get("end_time", "")),
            duration_seconds=float(raw.get("duration_seconds", 0.0) or 0.0),
            video_width=int(vc.get("width", 1920) or 1920),
            video_height=int(vc.get("height", 1200) or 1200),
            video_fps=int(vc.get("fps", 30) or 30),
            camera_positions=camera_positions,
            tracker_names=tracker_names,
            gripper_sides=gripper_sides,
            scenario_name=str(raw.get("scenario") or raw.get("scenario_name") or ""),
        )

    # ------------------------------------------------------------------
    # Tracker CSV
    # ------------------------------------------------------------------

    def _load_tracker_data(self) -> pd.DataFrame:
        csv_path = self.session_dir / "tracker_positions.csv"
        if not csv_path.exists():
            logger.warning("tracker_positions.csv not found")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Cannot read tracker_positions.csv: %s", exc)
            return pd.DataFrame()

        if df.empty:
            return df

        if "time_seconds" in df.columns:
            try:
                df["t"] = pd.to_numeric(df["time_seconds"], errors="coerce")
                n_before = len(df)
                df = df.dropna(subset=["t"]).copy()
                if n_before - len(df) > 0:
                    logger.warning("tracker_positions.csv: dropped %d rows with invalid time_seconds", n_before - len(df))
            except Exception as exc:
                logger.error("Tracker time_seconds conversion failed: %s", exc)
                return pd.DataFrame()
        elif "timestamp" in df.columns:
            try:
                df["_abs_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["t"] = (df["_abs_time"] - self._ref_time).dt.total_seconds()
                n_before = len(df)
                df = df.dropna(subset=["t"]).copy()
                if n_before - len(df) > 0:
                    logger.warning("tracker_positions.csv: dropped %d rows with invalid timestamps", n_before - len(df))
            except Exception as exc:
                logger.error("Tracker timestamp conversion failed: %s", exc)
                return pd.DataFrame()
        else:
            logger.warning("tracker_positions.csv has neither 'time_seconds' nor 'timestamp' — skipping")
            return pd.DataFrame()

        # Discover tracker names from column names (tracker_{name}_{axis})
        tracker_names_discovered = set()
        for col in df.columns:
            if col.startswith("tracker_"):
                parts = col.split("_")
                # col = tracker_{name}_{axis} → name = parts[1], axis = parts[2]
                if len(parts) >= 3:
                    tracker_names_discovered.add(parts[1])

        if tracker_names_discovered:
            self.metadata.tracker_names = sorted(tracker_names_discovered)
            logger.info("Tracker names from CSV columns: %s", self.metadata.tracker_names)

        return df

    def _prepare_tracker_arrays(self) -> None:
        if self.tracker_df.empty:
            return
        try:
            self._tracker_timestamps = self.tracker_df["t"].to_numpy(dtype=np.float64)
            # Absolute nanosecond timestamps from the CSV (used for cross-stream alignment)
            if "timestamp_ns" in self.tracker_df.columns:
                self._tracker_timestamps_ns = pd.to_numeric(
                    self.tracker_df["timestamp_ns"], errors="coerce"
                ).to_numpy(dtype=np.float64)
            else:
                self._tracker_timestamps_ns = None
            for col in self.tracker_df.columns:
                if col.startswith("tracker_"):
                    arr = pd.to_numeric(self.tracker_df[col], errors="coerce").to_numpy(dtype=np.float64)
                    self._tracker_arrays[col] = arr
        except Exception as exc:
            logger.error("Failed to prepare tracker arrays: %s", exc)
            self._tracker_timestamps = None
            self._tracker_timestamps_ns = None
            self._tracker_arrays = {}

    # ------------------------------------------------------------------
    # Gripper CSV
    # ------------------------------------------------------------------

    def _load_gripper_data(self) -> Dict[str, pd.DataFrame]:
        """Load gripper CSVs. Supports both new (gripper_{side}_data.csv) and
        old (pince{id}_data.csv) naming conventions."""
        result = {}

        # New naming: gripper_left_data.csv, gripper_right_data.csv
        # Discovered from metadata.grippers sides, or scan directory
        sides_to_try = list(self.metadata.gripper_sides)
        if not sides_to_try:
            # Fallback: auto-discover from filesystem
            for p in self.session_dir.glob("gripper_*_data.csv"):
                # gripper_left_data.csv → "left"
                name = p.stem  # "gripper_left_data"
                parts = name.split("_")
                if len(parts) >= 2:
                    sides_to_try.append(parts[1])

        for side in sides_to_try:
            # Try new format first
            csv_path = self.session_dir / f"gripper_{side}_data.csv"
            if not csv_path.exists():
                # Fallback: old pince format
                csv_path = self.session_dir / f"pince{side}_data.csv"
            if not csv_path.exists():
                logger.warning("Gripper CSV not found for side '%s'", side)
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                logger.error("Cannot read %s: %s — skipping gripper '%s'", csv_path.name, exc, side)
                continue

            if df.empty or len(df) < 2:
                logger.warning("%s has fewer than 2 rows — skipping gripper '%s'", csv_path.name, side)
                continue

            if "t_ns" not in df.columns and "time_seconds" not in df.columns and "timestamp" not in df.columns:
                logger.warning("%s has neither 't_ns', 'time_seconds' nor 'timestamp' — skipping gripper '%s'", csv_path.name, side)
                continue

            # Detect "packed string" format: all data columns are empty but angle_deg
            # holds the full raw serial string, e.g.:
            #   "T=85524 ID=ARD-R-00001  SW=ON   Ouverture=  0.0 mm  Angle=  -0.26°"
            # In this case unpack all fields from that string first.
            packed_cols = ["t_ms", "t_ms_corrected_ns", "gripper_side", "sw", "opening_mm", "angle_deg"]
            is_packed = (
                "angle_deg" in df.columns
                and all(
                    col not in df.columns or pd.to_numeric(df[col], errors="coerce").notna().sum() == 0
                    for col in ["t_ms", "opening_mm"]
                )
                and df["angle_deg"].dropna().astype(str).str.contains("Ouverture=").any()
            )
            if is_packed:
                logger.info("%s: detected packed serial string in angle_deg — unpacking all fields", csv_path.name)
                parsed = df["angle_deg"].apply(self._parse_packed_serial)
                parsed_df = pd.DataFrame(list(parsed), index=df.index)
                for col in ["t_ms", "gripper_side", "sw", "opening_mm", "angle_deg"]:
                    if col in parsed_df.columns:
                        df[col] = parsed_df[col]

            # Resolve timestamp — priority: t_ns > time_seconds > timestamp
            if "t_ns" in df.columns:
                try:
                    ref_ns = self._ref_time.value  # ns since Unix epoch
                    df["t"] = (pd.to_numeric(df["t_ns"], errors="coerce") - ref_ns) / 1e9
                    n_before = len(df)
                    df = df.dropna(subset=["t"]).copy()
                    if n_before - len(df) > 0:
                        logger.warning("%s: dropped %d rows with invalid t_ns", csv_path.name, n_before - len(df))
                    logger.info("Gripper '%s': using t_ns column for timestamps", side)
                except Exception as exc:
                    logger.error("Gripper '%s' t_ns conversion failed: %s", side, exc)
                    continue
            elif "time_seconds" in df.columns:
                try:
                    df["t"] = pd.to_numeric(df["time_seconds"], errors="coerce")
                    n_before = len(df)
                    df = df.dropna(subset=["t"]).copy()
                    if n_before - len(df) > 0:
                        logger.warning("%s: dropped %d rows with invalid time_seconds", csv_path.name, n_before - len(df))
                except Exception as exc:
                    logger.error("Gripper '%s' time_seconds conversion failed: %s", side, exc)
                    continue
            else:
                try:
                    df["_abs_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df["t"] = (df["_abs_time"] - self._ref_time).dt.total_seconds()
                    n_before = len(df)
                    df = df.dropna(subset=["t"]).copy()
                    if n_before - len(df) > 0:
                        logger.warning("%s: dropped %d rows with invalid timestamps", csv_path.name, n_before - len(df))
                except Exception as exc:
                    logger.error("Gripper '%s' timestamp conversion failed: %s", side, exc)
                    continue

            # Resolve opening_mm: prefer already-resolved column, then ouverture_mm
            opening_resolved = False
            if "opening_mm" in df.columns:
                numeric = pd.to_numeric(df["opening_mm"], errors="coerce")
                if numeric.notna().sum() > 0:
                    df["opening_mm"] = numeric.fillna(0.0)
                    opening_resolved = True
            if not opening_resolved and "ouverture_mm" in df.columns:
                numeric = pd.to_numeric(df["ouverture_mm"], errors="coerce")
                if numeric.notna().sum() > 0:
                    df["opening_mm"] = numeric.fillna(0.0)
                    opening_resolved = True
            if not opening_resolved:
                df["opening_mm"] = 0.0

            if len(df) < 2:
                logger.warning("Gripper '%s' has no valid data after parsing — skipping", side)
                continue

            df = self._clean_gripper_df(df, side)
            if len(df) < 2:
                logger.warning("Gripper '%s': fewer than 2 rows after cleaning — skipping", side)
                continue

            result[side] = df
            logger.info("Gripper '%s': %d rows from %s", side, len(df), csv_path.name)

        return result

    def _clean_gripper_df(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """Clean and normalise a raw gripper DataFrame.

        Steps (applied in order):
        1. Drop rows with t < 0.
        2. Detect and discard pre-reset block (initial counter at 150+ that
           then resets to ~0 — identified by a backward jump > 50 s).
        3. Recalculate t from the raw ``timestamp`` column when available so
           that t always starts at 0 and is free of counter drift.
        4. Sort by t, deduplicate.
        5. Log warnings for abnormal timestamp jumps (continuity check).
        6. Apply centred rolling-mean smoothing (window=5) on opening_mm and
           angle_deg.
        7. Interpolate isolated outliers in opening_mm (IQR × 3 fence).
        """
        # 1. Remove negative-time rows
        n_before = len(df)
        df = df[df["t"] >= 0].copy()
        removed = n_before - len(df)
        if removed:
            logger.info("Gripper '%s': removed %d row(s) with t < 0", side, removed)

        if len(df) < 2:
            return df

        # 2. Detect and discard pre-reset block
        t_vals = df["t"].values
        reset_idx = None
        for i in range(1, len(t_vals)):
            if t_vals[i] < t_vals[i - 1] - 50.0:   # backward jump > 50 s
                reset_idx = i
                break
        if reset_idx is not None:
            logger.info(
                "Gripper '%s': time reset at index %d (%.1f → %.1f s) — "
                "discarding %d pre-reset row(s)",
                side, reset_idx, t_vals[reset_idx - 1], t_vals[reset_idx], reset_idx,
            )
            df = df.iloc[reset_idx:].copy()

        if len(df) < 2:
            return df

        # 3. Recalculate t from raw timestamp when available
        if "timestamp" in df.columns:
            ts_raw = pd.to_numeric(df["timestamp"], errors="coerce")
            valid_ts = ts_raw.dropna()
            if not valid_ts.empty:
                t0 = float(valid_ts.iloc[0])
                new_t = ts_raw - t0
                if (new_t.dropna() >= 0).all():
                    df = df.copy()
                    df["t"] = new_t
                    logger.info(
                        "Gripper '%s': t recalculated from timestamp (origin=%.3f)", side, t0
                    )

        # 4. Sort by t, remove duplicates
        df = (
            df.sort_values("t")
            .drop_duplicates(subset="t", keep="last")
            .reset_index(drop=True)
        )

        if len(df) < 2:
            return df

        # 5. Continuity check — log abnormal jumps
        dt = df["t"].diff().iloc[1:]
        if not dt.empty:
            median_dt = float(dt.median())
            threshold = max(median_dt * 20, 1.0)
            n_jumps = int((dt > threshold).sum())
            if n_jumps:
                logger.warning(
                    "Gripper '%s': %d abnormal timestamp jump(s) "
                    "(median_dt=%.4f s, threshold=%.2f s)",
                    side, n_jumps, median_dt, threshold,
                )

        # 6. Rolling-mean smoothing
        window = 5
        if "opening_mm" in df.columns and len(df) >= window:
            df["opening_mm"] = (
                df["opening_mm"]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )
        if "angle_deg" in df.columns and len(df) >= window:
            numeric_angle = pd.to_numeric(df["angle_deg"], errors="coerce")
            if numeric_angle.notna().sum() > 0:
                df["angle_deg"] = (
                    numeric_angle
                    .rolling(window=window, center=True, min_periods=1)
                    .mean()
                )

        # 7. Interpolate isolated outliers in opening_mm (IQR × 3 fence)
        if "opening_mm" in df.columns and len(df) > 4:
            col = df["opening_mm"]
            q25, q75 = float(col.quantile(0.25)), float(col.quantile(0.75))
            iqr = q75 - q25
            if iqr > 0:
                lo, hi = q25 - 3.0 * iqr, q75 + 3.0 * iqr
                mask = (col < lo) | (col > hi)
                n_out = int(mask.sum())
                if n_out:
                    df.loc[mask, "opening_mm"] = np.nan
                    df["opening_mm"] = (
                        df["opening_mm"]
                        .interpolate(method="linear")
                        .ffill()
                        .bfill()
                    )
                    logger.info(
                        "Gripper '%s': interpolated %d outlier(s) in opening_mm",
                        side, n_out,
                    )

        return df

    @staticmethod
    def _parse_packed_serial(raw: str) -> dict:
        """Parse all fields from a packed serial string into a dict.

        Input:  "T=85524 ID=ARD-R-00001  SW=ON   Ouverture=  0.0 mm  Angle=  -0.26°"
        Output: {"t_ms": 85524.0, "gripper_side": "ARD-R-00001", "sw": "ON",
                 "opening_mm": 0.0, "angle_deg": -0.26}
        """
        result: dict = {}
        s = str(raw)
        try:
            m = re.search(r"\bT=(\d+)", s)
            if m:
                result["t_ms"] = float(m.group(1))
        except Exception:
            pass
        try:
            m = re.search(r"\bID=([\w\-]+)", s)
            if m:
                result["gripper_side"] = m.group(1)
        except Exception:
            pass
        try:
            m = re.search(r"\bSW=(\w+)", s)
            if m:
                result["sw"] = m.group(1)
        except Exception:
            pass
        try:
            m = re.search(r"Ouverture=\s*([-+]?\d*\.?\d+)", s)
            if m:
                result["opening_mm"] = float(m.group(1))
        except Exception:
            pass
        try:
            m = re.search(r"Angle=\s*([-+]?\d*\.?\d+)", s)
            if m:
                result["angle_deg"] = float(m.group(1))
        except Exception:
            pass
        return result

    @staticmethod
    def _parse_ouverture_mm(raw: str) -> float:
        """Parse opening_mm from raw serial string.

        Handles: "T=85524 ID=ARD-R-00001  SW=ON   Ouverture=  0.0 mm  Angle=  -0.26°"
        Looks for 'Ouverture=' first, falls back to first numeric value.
        """
        try:
            s = str(raw)
            match = re.search(r"Ouverture=\s*([-+]?\d*\.?\d+)", s)
            if match:
                return float(match.group(1))
            match = re.search(r"[-+]?\d*\.?\d+", s)
            return float(match.group()) if match else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _parse_numeric(raw: str) -> float:
        try:
            match = re.search(r"[-+]?\d*\.?\d+", str(raw))
            return float(match.group()) if match else 0.0
        except Exception:
            return 0.0

    def _prepare_gripper_arrays(self) -> None:
        for side, df in self.gripper_dfs.items():
            try:
                # Tri chronologique + suppression des timestamps dupliqués
                # (doublons ou désordre dans le CSV causent des artefacts en dents de scie)
                df_clean = (
                    df[["t", "opening_mm"]]
                    .copy()
                    .pipe(lambda d: d.assign(opening_mm=pd.to_numeric(d["opening_mm"], errors="coerce").fillna(0.0)))
                    .sort_values("t")
                    .drop_duplicates(subset="t", keep="last")
                    .reset_index(drop=True)
                )
                self._gripper_timestamps[side] = df_clean["t"].to_numpy(dtype=np.float64)
                self._gripper_openings[side] = df_clean["opening_mm"].to_numpy(dtype=np.float64)
            except Exception as exc:
                logger.error("Failed to prepare gripper '%s' arrays: %s", side, exc)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    def _load_cameras(self) -> Dict[str, CameraInfo]:
        """Load cameras by position name from videos/ subfolder.

        Keys in returned dict are position names: "head", "left", "right".
        """
        cameras = {}
        videos_dir = self.session_dir / "videos"

        # Discover available video files
        positions_to_try = list(self.metadata.camera_positions)
        if not positions_to_try:
            # Auto-discover from filesystem
            if videos_dir.is_dir():
                for p in sorted(videos_dir.glob("*.mp4")):
                    positions_to_try.append(p.stem)
            else:
                # Fallback: old cam0/output.avi layout
                for cam_id in self._camera_id_to_position:
                    positions_to_try.append(self._camera_id_to_position[cam_id])

        for pos in positions_to_try:
            # Try new format: videos/{pos}.mp4
            video_path = videos_dir / f"{pos}.mp4"
            if not video_path.exists():
                # Try .avi
                video_path = videos_dir / f"{pos}.avi"
            if not video_path.exists():
                # Fallback: old cam layout — find the camera ID for this position
                for cam_id, cam_pos in self._camera_id_to_position.items():
                    if cam_pos == pos:
                        video_path = self.session_dir / f"cam{cam_id}" / "output.avi"
                        break
            if not video_path.exists():
                logger.warning("Video not found for position '%s'", pos)
                continue

            try:
                _suppress_stderr()
                cap = cv2.VideoCapture(str(video_path))
                _restore_stderr()
            except Exception as exc:
                _restore_stderr()
                logger.error("cv2.VideoCapture raised for %s: %s", video_path, exc)
                continue

            if not cap.isOpened():
                logger.error("Cannot open video: %s", video_path)
                cap.release()
                continue

            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            except Exception as exc:
                logger.error("Cannot read video properties for %s: %s", video_path, exc)
                cap.release()
                continue

            if fps <= 0:
                fps = float(self.metadata.video_fps) if self.metadata.video_fps > 0 else 30.0
                logger.warning("Camera '%s': FPS unreadable, using %s", pos, fps)
            if width <= 0 or height <= 0:
                width = self.metadata.video_width or 1920
                height = self.metadata.video_height or 1200
                logger.warning("Camera '%s': dimensions unreadable, using %dx%d", pos, width, height)

            # Get accurate frame count using ffprobe (reliable) with fallback
            frame_count = self._get_accurate_frame_count(video_path, fps)

            # camera_anchors keyed by position name ("head", "left", "right")
            video_offset = self._camera_anchors.get(pos, 0.0)

            # Load frame timestamps from .jsonl (same name as video, next to it)
            jsonl_path = video_path.with_suffix(".jsonl")
            frame_ms_offsets, capture_times_ms = self._load_jsonl_offsets(jsonl_path)
            # Absolute capture timestamps in nanoseconds (capture_time ms * 1e6)
            frame_capture_ns = capture_times_ms * 1e6 if capture_times_ms is not None else None

            # Note: jsonl may have fewer entries than actual frames (not authoritative)

            cameras[pos] = CameraInfo(
                position=pos,
                capture=cap,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                video_path=video_path,
                video_offset=video_offset,
                frame_ms_offsets=frame_ms_offsets,
                frame_capture_ns=frame_capture_ns,
            )
            jsonl_info = f", {len(frame_ms_offsets)} jsonl entries" if frame_ms_offsets is not None else ""
            logger.info("Camera '%s': %d frames (%.2fs) from %s%s", pos, frame_count, frame_count / fps if fps > 0 else 0, video_path.name, jsonl_info)

        return cameras

    def _get_accurate_frame_count(self, video_path: Path, fps: float) -> int:
        """Get accurate frame count using ffprobe (reliable) with OpenCV/metadata fallback.

        Args:
            video_path: Path to video file
            fps: Video FPS (for fallback calculation)

        Returns:
            Number of frames in the video
        """
        import subprocess
        import json

        # Try ffprobe first (most reliable)
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames",
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
                    nb_frames = data["streams"][0].get("nb_frames")
                    if nb_frames:
                        return int(nb_frames)
        except Exception:
            pass

        # Fallback to OpenCV CAP_PROP_FRAME_COUNT (may be unreliable)
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if frame_count > 0:
                    return frame_count
        except Exception:
            pass

        # Last resort: calculate from metadata duration
        if self.metadata.duration_seconds > 0 and fps > 0:
            return int(self.metadata.duration_seconds * fps)

        return 0

    def _compute_frame_count(self) -> int:
        # Prefer actual video frame counts over metadata duration
        counts = [c.frame_count for c in self.cameras.values() if c.frame_count > 0]
        if counts:
            return min(counts)
        # Fallback to metadata duration calculation
        if self.metadata.duration_seconds > 0 and self.fps > 0:
            return int(self.metadata.duration_seconds * self.fps)
        return 0

    # ------------------------------------------------------------------
    # JSONL loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl_offsets(jsonl_path: Path) -> Optional[np.ndarray]:
        """Load per-frame millisecond offsets from a .jsonl file.

        Each line: {"index": N, "capture_time": T}  where T is in ms.
        Returns a 0-based numpy array: offsets[i] = ms from start of video for frame i.
        Returns None if the file is missing or unreadable.
        """
        if not jsonl_path.exists():
            return None, None
        try:
            import json as _json
            entries = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(_json.loads(line))
                    except Exception:
                        continue

            if not entries:
                return None, None

            # Sort by index to ensure correct order
            entries.sort(key=lambda e: e.get("index", 0))
            capture_times = np.array([e["capture_time"] for e in entries], dtype=np.float64)
            # Offsets in ms from the first frame
            offsets = capture_times - capture_times[0]
            logger.debug("Loaded %d jsonl entries from %s", len(offsets), jsonl_path.name)
            return offsets, capture_times
        except Exception as exc:
            logger.warning("Could not load jsonl %s: %s", jsonl_path, exc)
            return None, None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, position: str, frame_index: int) -> Optional[np.ndarray]:
        """Seek to a specific frame index and decode it (precise scrub mode).

        Uses CAP_PROP_POS_FRAMES for reliable bidirectional seeking.
        Falls back to CAP_PROP_POS_MSEC (jsonl offsets or fps-based) if needed.

        NOT used during playback — use get_frame_sequential() for that.

        Args:
            position: Camera position ("head", "left", "right")
            frame_index: 0-based frame index

        Returns:
            BGR numpy array or None
        """
        cam = self.cameras.get(position)
        if cam is None:
            return None

        if frame_index < 0:
            frame_index = 0

        cap = cam.capture
        if cam.fps <= 0:
            return None

        _suppress_stderr()
        try:
            # Primary: seek by frame index — reliable for both forward and backward
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                # Fallback: seek by milliseconds using jsonl offsets or fps estimate
                if cam.frame_ms_offsets is not None and frame_index < len(cam.frame_ms_offsets):
                    target_ms = cam.frame_ms_offsets[frame_index]
                else:
                    target_ms = (frame_index / cam.fps) * 1000.0
                cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                ret, frame = cap.read()
        except Exception as exc:
            logger.error("Error reading frame %d from camera '%s': %s", frame_index, position, exc)
            ret, frame = False, None
        finally:
            _restore_stderr()

        return frame if ret else None

    def get_frame_sequential(self, position: str) -> Optional[np.ndarray]:
        """Read the next frame sequentially (no seek — playback mode).

        The caller is responsible for calling cap.set() once at playback
        start to position the capture at the right frame.

        Returns:
            BGR numpy array or None
        """
        cam = self.cameras.get(position)
        if cam is None:
            return None

        _suppress_stderr()
        try:
            ret, frame = cam.capture.read()
        except Exception as exc:
            logger.error("Sequential read error camera '%s': %s", position, exc)
            ret, frame = False, None
        finally:
            _restore_stderr()

        return frame if ret else None

    def seek_for_playback(self, frame_index: int) -> None:
        """Seek all camera captures to the given frame index for playback start."""
        for pos, cam in self.cameras.items():
            if cam.fps <= 0:
                continue
            _suppress_stderr()
            try:
                cam.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            except Exception as exc:
                logger.error("Seek error camera '%s': %s", pos, exc)
            finally:
                _restore_stderr()

    def get_all_frames(self, frame_index: int) -> Dict[str, np.ndarray]:
        """Seek + decode all cameras at a specific frame index (scrub mode)."""
        frames = {}
        for pos in self.cameras:
            try:
                f = self.get_frame(pos, frame_index)
                if f is not None:
                    frames[pos] = f
            except Exception as exc:
                logger.error("get_all_frames error camera '%s' frame %d: %s", pos, frame_index, exc)
        return frames

    def get_all_frames_sequential(self) -> Dict[str, np.ndarray]:
        """Read the next frame from all cameras sequentially (playback mode)."""
        frames = {}
        for pos in self.cameras:
            try:
                f = self.get_frame_sequential(pos)
                if f is not None:
                    frames[pos] = f
            except Exception as exc:
                logger.error("get_all_frames_sequential error camera '%s': %s", pos, exc)
        return frames

    def frame_timestamp(self, frame_index: int, position: str = None) -> float:
        """Convert frame index to time_seconds on the common CSV timeline.

        time_seconds = offset + frame_index / fps
        where offset = camera_anchors[position]["mono_offset_from_record"].
        """
        if self.fps <= 0:
            return 0.0
        if position is not None:
            cam = self.cameras.get(position)
        else:
            # Use first available camera
            cam = next(iter(self.cameras.values()), None) if self.cameras else None
        offset = cam.video_offset if cam is not None else 0.0
        return offset + frame_index / self.fps

    # ------------------------------------------------------------------
    # Sensor interpolation at arbitrary timestamp
    # ------------------------------------------------------------------

    def get_tracker_state(self, t: float) -> Dict[str, Dict[str, np.ndarray]]:
        """Get interpolated tracker positions and quaternions at time t.

        Returns:
            dict tracker_name -> {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
        """
        if self._tracker_timestamps is None or len(self._tracker_timestamps) < 2:
            return {}

        ts = self._tracker_timestamps
        result = {}
        for name in self.metadata.tracker_names:
            try:
                px = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_x"), t)
                py = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_y"), t)
                pz = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_z"), t)

                qw = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_qw"), t)
                qx = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_qx"), t)
                qy = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_qy"), t)
                qz = self._interp(ts, self._tracker_arrays.get(f"tracker_{name}_qz"), t)

                if px is not None:
                    result[name] = {
                        "position": np.array([px, py, pz]),
                        "quaternion": np.array([qw, qx, qy, qz]),
                    }
            except Exception as exc:
                logger.error("Tracker state interpolation error for '%s': %s", name, exc)

        return result

    def get_gripper_opening(self, side: str, t: float) -> Optional[float]:
        """Get interpolated gripper opening_mm at time t."""
        if side not in self._gripper_timestamps:
            return None
        try:
            ts = self._gripper_timestamps[side]
            openings = self._gripper_openings[side]
            return self._interp(ts, openings, t)
        except Exception as exc:
            logger.error("Gripper opening interpolation error for '%s': %s", side, exc)
            return None

    def get_all_tracker_positions(self) -> Dict[str, np.ndarray]:
        """Get full position arrays for all trackers (for trajectory display)."""
        if self._tracker_timestamps is None:
            return {}
        result = {}
        for name in self.metadata.tracker_names:
            try:
                x = self._tracker_arrays.get(f"tracker_{name}_x")
                y = self._tracker_arrays.get(f"tracker_{name}_y")
                z = self._tracker_arrays.get(f"tracker_{name}_z")
                if x is not None and y is not None and z is not None:
                    result[name] = np.column_stack([x, y, z])
            except Exception as exc:
                logger.error("Failed to build position array for tracker '%s': %s", name, exc)
        return result

    def get_tracker_timestamps(self) -> Optional[np.ndarray]:
        return self._tracker_timestamps

    def get_tracker_timestamps_ns(self) -> Optional[np.ndarray]:
        """Return absolute tracker timestamps in nanoseconds (from timestamp_ns column)."""
        return self._tracker_timestamps_ns

    def get_frame_capture_ns(self, position: str) -> Optional[np.ndarray]:
        """Return absolute capture timestamps in nanoseconds for each frame of a camera.

        Derived from jsonl capture_time (ms) * 1e6.
        Returns None if jsonl was not available for this camera.
        """
        cam = self.cameras.get(position)
        if cam is None:
            return None
        return cam.frame_capture_ns

    def get_timeline_ns(self) -> Tuple[float, float]:
        """Return (start_ns, end_ns) of the common absolute timeline across all streams.

        Uses the union of camera capture_ns and tracker timestamp_ns ranges.
        """
        bounds = []
        for cam in self.cameras.values():
            if cam.frame_capture_ns is not None and len(cam.frame_capture_ns) > 0:
                bounds.append((cam.frame_capture_ns[0], cam.frame_capture_ns[-1]))
        if self._tracker_timestamps_ns is not None and len(self._tracker_timestamps_ns) > 0:
            valid = self._tracker_timestamps_ns[np.isfinite(self._tracker_timestamps_ns)]
            if len(valid) > 0:
                bounds.append((valid[0], valid[-1]))
        if not bounds:
            return 0.0, 1.0
        start = min(b[0] for b in bounds)
        end = max(b[1] for b in bounds)
        return float(start), float(end)

    def get_frame_index_at_ns(self, position: str, t_ns: float) -> int:
        """Return the frame index closest to absolute timestamp t_ns for a camera.

        Falls back to fps-based estimate if jsonl is unavailable.
        """
        cam = self.cameras.get(position)
        if cam is None:
            return 0
        if cam.frame_capture_ns is not None and len(cam.frame_capture_ns) > 0:
            idx = int(np.searchsorted(cam.frame_capture_ns, t_ns, side="left"))
            return max(0, min(idx, len(cam.frame_capture_ns) - 1))
        # Fallback: fps-based estimate from first camera as reference
        if cam.fps > 0:
            # Use video_offset as best-effort anchor
            ref_ns = cam.frame_capture_ns[0] if cam.frame_capture_ns is not None else 0.0
            dt_s = (t_ns - ref_ns) / 1e9
            idx = int(dt_s * cam.fps)
            return max(0, min(idx, cam.frame_count - 1))
        return 0

    def get_tracker_index_at_ns(self, t_ns: float) -> int:
        """Return the tracker row index closest to absolute timestamp t_ns."""
        if self._tracker_timestamps_ns is None or len(self._tracker_timestamps_ns) == 0:
            return 0
        idx = int(np.searchsorted(self._tracker_timestamps_ns, t_ns, side="left"))
        return max(0, min(idx, len(self._tracker_timestamps_ns) - 1))

    def get_tracker_state_at_ns(self, t_ns: float) -> Dict[str, Dict[str, np.ndarray]]:
        """Get tracker positions and quaternions at absolute timestamp t_ns.

        Uses timestamp_ns column for exact alignment — avoids the time_seconds
        referential mismatch that occurs when passing Unix nanoseconds to
        get_tracker_state() which expects time_seconds offsets.

        Returns:
            dict tracker_name -> {"position": [x,y,z], "quaternion": [qw,qx,qy,qz]}
        """
        if self._tracker_timestamps_ns is None or len(self._tracker_timestamps_ns) == 0:
            # Fallback: no ns timestamps — use time_seconds interpolation
            return self.get_tracker_state(t_ns / 1e9)

        idx = int(np.searchsorted(self._tracker_timestamps_ns, t_ns, side="left"))
        n = len(self._tracker_timestamps_ns)
        # Pick the closest of idx-1 and idx
        if idx == 0:
            best = 0
        elif idx >= n:
            best = n - 1
        else:
            d_prev = abs(self._tracker_timestamps_ns[idx - 1] - t_ns)
            d_curr = abs(self._tracker_timestamps_ns[idx]     - t_ns)
            best = idx if d_curr <= d_prev else idx - 1

        result = {}
        for name in self.metadata.tracker_names:
            try:
                px = self._tracker_arrays.get(f"tracker_{name}_x")
                py = self._tracker_arrays.get(f"tracker_{name}_y")
                pz = self._tracker_arrays.get(f"tracker_{name}_z")
                qw = self._tracker_arrays.get(f"tracker_{name}_qw")
                qx = self._tracker_arrays.get(f"tracker_{name}_qx")
                qy = self._tracker_arrays.get(f"tracker_{name}_qy")
                qz = self._tracker_arrays.get(f"tracker_{name}_qz")
                if px is None or not np.isfinite(px[best]):
                    continue
                result[name] = {
                    "position":   np.array([px[best], py[best], pz[best]]),
                    "quaternion": np.array([qw[best], qx[best], qy[best], qz[best]]),
                }
            except Exception as exc:
                logger.error("Tracker state at ns error for '%s': %s", name, exc)
        return result

    def detect_axis_remap(self) -> str:
        """Détecte le mapping CSV→GL à partir des données.

        Heuristique : l'axe dont la moyenne absolue est la plus grande ET positive
        (hauteur du tracker au-dessus du sol) est l'axe vertical.
        Dans pyqtgraph GL, l'axe vertical est Z.

        Retourne une clé de Viewer3DWidget.AXIS_REMAPS.
        """
        if self._tracker_arrays is None:
            return "X  Y  Z  (identité)"

        # Calculer la moyenne des positions sur tous les trackers
        means = {"x": [], "y": [], "z": []}
        for name in self.metadata.tracker_names:
            for ax in ("x", "y", "z"):
                arr = self._tracker_arrays.get(f"tracker_{name}_{ax}")
                if arr is not None and len(arr) > 0:
                    valid = arr[np.isfinite(arr)]
                    if len(valid) > 0:
                        means[ax].append(float(np.median(valid)))

        avg = {ax: float(np.mean(vals)) if vals else 0.0 for ax, vals in means.items()}
        logger.info("detect_axis_remap: axis medians = %s", avg)

        # L'axe vertical est celui avec la valeur positive la plus grande
        vertical = max(avg, key=lambda a: avg[a] if avg[a] > 0 else -999)
        logger.info("detect_axis_remap: detected vertical axis = %s (mean=%.3f)", vertical, avg[vertical])

        # Mapping : axe vertical CSV → Z dans GL
        _MAP = {
            "z": "X  Y  Z  (identité)",    # Z déjà vertical → pas de remap
            "y": "X  Z  Y",                 # Y vertical → swap Y↔Z
            "x": "Z  Y  X",                 # X vertical → swap X↔Z
        }
        return _MAP.get(vertical, "X  Y  Z  (identité)")

    def get_gripper_timeseries(self, side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get full timeseries for a gripper (timestamps, opening_mm)."""
        if side not in self._gripper_timestamps:
            return None, None
        return self._gripper_timestamps[side], self._gripper_openings[side]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interp(timestamps: np.ndarray, values: Optional[np.ndarray], t: float) -> Optional[float]:
        """Simple linear interpolation with clamping."""
        try:
            if values is None or len(values) == 0 or len(timestamps) == 0:
                return None
            if t <= timestamps[0]:
                return float(values[0])
            if t >= timestamps[-1]:
                return float(values[-1])
            idx = np.searchsorted(timestamps, t, side="right") - 1
            idx = max(0, min(idx, len(timestamps) - 2))
            t0, t1 = timestamps[idx], timestamps[idx + 1]
            if t1 == t0:
                return float(values[idx])
            alpha = (t - t0) / (t1 - t0)
            return float(values[idx] * (1 - alpha) + values[idx + 1] * alpha)
        except Exception:
            return None

    def get_nan_report(self) -> Dict[str, str]:
        """Détecte les NaN dans les données de la session."""
        report: Dict[str, str] = {}

        for name in self.metadata.tracker_names:
            axes = ["x", "y", "z", "qw", "qx", "qy", "qz"]
            nan_axes = []
            total_rows = 0
            nan_rows = 0
            for ax in axes:
                col = f"tracker_{name}_{ax}"
                arr = self._tracker_arrays.get(col)
                if arr is None:
                    nan_axes.append(ax)
                    continue
                n_nan = int(np.isnan(arr).sum())
                if n_nan > 0:
                    nan_axes.append(ax)
                    nan_rows = max(nan_rows, n_nan)
                    total_rows = len(arr)
            if nan_axes:
                pct = f"{nan_rows / total_rows * 100:.1f}%" if total_rows > 0 else "?"
                report[f"tracker_{name}"] = (
                    f"Tracker {name} : données NaN sur {nan_rows} lignes ({pct}) — "
                    f"axes : {', '.join(nan_axes)}"
                )

        for side, openings in self._gripper_openings.items():
            n_nan = int(np.isnan(openings).sum())
            if n_nan > 0:
                total = len(openings)
                pct = f"{n_nan / total * 100:.1f}%" if total > 0 else "?"
                report[f"gripper_{side}"] = (
                    f"Pince {side} : {n_nan} valeurs NaN ({pct})"
                )

        return report

    def release(self) -> None:
        """Release all video captures."""
        for cam in self.cameras.values():
            try:
                cam.capture.release()
            except Exception:
                pass
        self.cameras.clear()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

