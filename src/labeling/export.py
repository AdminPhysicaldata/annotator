"""Export annotations to various formats."""

import csv
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .label_manager import LabelManager, LabelType

logger = logging.getLogger(__name__)


class AnnotationExporter:
    """Export annotations to different formats."""

    def __init__(self, label_manager: LabelManager, dataset_info: Optional[Dict] = None):
        """Initialize exporter.

        Args:
            label_manager: LabelManager instance with annotations
            dataset_info: Optional dataset metadata (episode_id, duration, etc.)
        """
        self.label_manager = label_manager
        self.dataset_info = dataset_info or {}

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def export_to_json(
        self,
        output_path: Path,
        include_metadata: bool = True,
    ) -> None:
        """Export to JSON format.

        Raises:
            RuntimeError: on any I/O or serialisation error.
        """
        try:
            data = self.label_manager.to_dict()

            if include_metadata:
                data["metadata"] = {
                    "export_timestamp": datetime.now().isoformat(),
                    "dataset_info": self.dataset_info,
                    "statistics": self.label_manager.get_statistics(),
                }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                # default=str handles non-serialisable types (numpy scalars, etc.)
                json.dump(data, f, indent=2, default=str)

            logger.info("Exported JSON annotations to %s", output_path)

        except Exception as exc:
            raise RuntimeError(f"Export JSON échoué vers {output_path} : {exc}") from exc

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_to_csv(
        self,
        output_path: Path,
        format_type: str = "frame_based",
    ) -> None:
        """Export to CSV format.

        Args:
            output_path: Output file path
            format_type: 'frame_based' or 'annotation_based'

        Raises:
            RuntimeError: on any I/O error.
        """
        if format_type not in ("frame_based", "annotation_based"):
            raise ValueError(f"Unknown format_type: {format_type}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if format_type == "frame_based":
                self._export_frame_based_csv(output_path)
            else:
                self._export_annotation_based_csv(output_path)
            logger.info("Exported CSV annotations (%s) to %s", format_type, output_path)
        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Export CSV échoué vers {output_path} : {exc}") from exc

    def _export_frame_based_csv(self, output_path: Path) -> None:
        """Export CSV with one row per frame."""
        max_frame = 0
        for ann in self.label_manager.annotations:
            try:
                if ann.annotation_type == LabelType.FRAME:
                    max_frame = max(max_frame, ann.frame_index)
                else:
                    max_frame = max(max_frame, ann.end_frame)
            except Exception:
                pass

        label_names = []
        try:
            label_names = sorted([label.name for label in self.label_manager.labels.values()])
        except Exception as exc:
            logger.warning("Could not retrieve label names: %s", exc)

        try:
            fps = float(self.dataset_info.get("fps", 30.0) or 30.0)
        except (TypeError, ValueError):
            fps = 30.0

        rows = []
        for frame_idx in range(max_frame + 1):
            try:
                active_labels = self.label_manager.get_labels_at_frame(frame_idx)
            except Exception:
                active_labels = set()

            row: Dict[str, Any] = {
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0.0,
            }
            for label_name in label_names:
                row[label_name] = 1 if label_name in active_labels else 0
            rows.append(row)

        fieldnames = ["frame_index", "timestamp"] + label_names
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _export_annotation_based_csv(self, output_path: Path) -> None:
        """Export CSV with one row per annotation."""
        rows = []
        for ann in self.label_manager.annotations:
            try:
                row = {
                    "annotation_id": ann.id,
                    "label_id": ann.label_id,
                    "label_name": ann.label_name,
                    "type": ann.annotation_type.value,
                    "frame_index": ann.frame_index if ann.annotation_type == LabelType.FRAME else "",
                    "start_frame": ann.start_frame if ann.annotation_type == LabelType.INTERVAL else "",
                    "end_frame": ann.end_frame if ann.annotation_type == LabelType.INTERVAL else "",
                    "confidence": ann.confidence,
                }
                rows.append(row)
            except Exception as exc:
                logger.warning("Skipping malformed annotation during CSV export: %s", exc)

        fieldnames = [
            "annotation_id", "label_id", "label_name", "type",
            "frame_index", "start_frame", "end_frame", "confidence",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # LeRobot v3.0
    # ------------------------------------------------------------------

    def export_to_lerobot_format(
        self,
        output_dir: Path,
        episode_index: int = 0,
        session=None,
        vflip: bool = False,
    ) -> None:
        """Export as a full LeRobot v3.0 dataset directory.

        Layout produced:
            <output_dir>/
            ├── meta/
            │   ├── info.json
            │   ├── stats.json
            │   ├── tasks.parquet
            │   └── episodes/
            │       └── chunk-000/
            │           └── file-000.parquet
            ├── data/
            │   └── chunk-000/
            │       └── file-000.parquet
            └── videos/
                └── <camera_key>/
                    └── chunk-000/
                        └── file-000.mp4

        Raises:
            RuntimeError: on any I/O or serialisation error.
        """
        try:
            self._export_to_lerobot_v3_impl(output_dir, episode_index, session, vflip=vflip)
            logger.info("Exported LeRobot v3.0 dataset to %s", output_dir)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Export LeRobot v3 échoué vers {output_dir} : {exc}") from exc

    def _export_to_lerobot_v3_impl(
        self,
        output_dir: Path,
        episode_index: int,
        session,
        vflip: bool = False,
    ) -> None:
        output_dir = Path(output_dir)

        try:
            fps = float(self.dataset_info.get("fps", 30.0) or 30.0)
        except (TypeError, ValueError):
            fps = 30.0

        frame_count = int(self.dataset_info.get("frame_count", 0) or 0)
        if frame_count == 0:
            for ann in self.label_manager.annotations:
                try:
                    if ann.annotation_type == LabelType.FRAME:
                        frame_count = max(frame_count, ann.frame_index + 1)
                    else:
                        frame_count = max(frame_count, ann.end_frame + 1)
                except Exception:
                    pass

        task_desc = self.dataset_info.get("session_id", "annotation_task")

        # ----------------------------------------------------------------
        # 1. Build main data parquet
        # ----------------------------------------------------------------
        # observation.state = flat float32 vector [pos×3 + quat×4] × n_trackers
        # observation.gripper.{side} = float32 scalar
        # annotation.{label} = bool
        # Standard cols: index, episode_index, frame_index, timestamp,
        #                task_index, is_first, is_last, is_terminal

        state_dim = 0
        if session is not None:
            try:
                state_dim = len(session.metadata.tracker_names) * 7
            except Exception:
                state_dim = 0

        rows = []
        for frame_idx in range(frame_count):
            t = frame_idx / fps if fps > 0 else 0.0
            try:
                active_labels = self.label_manager.get_labels_at_frame(frame_idx)
            except Exception:
                active_labels = set()

            row: Dict[str, Any] = {
                "index": frame_idx + episode_index * frame_count,  # global frame index
                "episode_index": episode_index,
                "frame_index": frame_idx,
                "timestamp": round(t, 6),
                "task_index": 0,
                "is_first": frame_idx == 0,
                "is_last": frame_idx == frame_count - 1,
                "is_terminal": frame_idx == frame_count - 1,
            }

            for label in self.label_manager.labels.values():
                try:
                    row[f"annotation.{label.name}"] = label.name in active_labels
                except Exception:
                    row[f"annotation.{label.name}"] = False

            if session is not None:
                # observation.state: flat float32 vector (NaN-free)
                try:
                    state_vec = self._get_state_vector(session, t)
                    row["observation.state"] = [float(v) for v in state_vec]
                except Exception as exc:
                    logger.warning("State vector failed at frame %d: %s", frame_idx, exc)
                    row["observation.state"] = [0.0] * state_dim

                # observation.gripper.{side}: scalar float32
                try:
                    for side in session.metadata.gripper_sides:
                        opening = session.get_gripper_opening(side, t)
                        row[f"observation.gripper.{side}"] = (
                            float(opening) if opening is not None else 0.0
                        )
                except Exception as exc:
                    logger.warning("Gripper reading failed at frame %d: %s", frame_idx, exc)

            rows.append(row)

        df = pd.DataFrame(rows)

        # action = next-frame state (imitation learning convention):
        # action[i] = state[i+1], last frame repeats itself.
        if "observation.state" in df.columns and state_dim > 0:
            states = df["observation.state"].tolist()
            # action[i] = state[i+1], action[-1] = state[-1]
            actions = states[1:] + [states[-1]]
            df["action"] = actions

        # Enforce dtypes for LeRobot compatibility
        int_cols = ["index", "episode_index", "frame_index", "task_index"]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype("int64")
        df["timestamp"] = df["timestamp"].astype("float32")
        for col in ["is_first", "is_last", "is_terminal"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        data_dir = output_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / "file-000.parquet", index=False)

        # ----------------------------------------------------------------
        # 2. Copy/transcode videos — v3 layout: videos/{cam_key}/chunk-000/file-000.mp4
        # ----------------------------------------------------------------
        if session is not None:
            for cam_key, cam_info in session.cameras.items():
                try:
                    video_src = session.session_dir / "videos" / f"{cam_key}.mp4"
                    if not video_src.exists():
                        video_src = session.session_dir / "videos" / f"{cam_key}.avi"
                    if video_src.exists():
                        video_dir = (
                            output_dir
                            / "videos"
                            / f"observation.images.{cam_key}"
                            / "chunk-000"
                        )
                        video_dir.mkdir(parents=True, exist_ok=True)
                        video_dst = video_dir / "file-000.mp4"
                        if vflip:
                            self._ffmpeg_vflip(video_src, video_dst)
                        else:
                            shutil.copy2(str(video_src), str(video_dst))
                except Exception as exc:
                    logger.warning("Could not copy video for camera '%s': %s", cam_key, exc)

        # ----------------------------------------------------------------
        # 3. meta/tasks.parquet
        # ----------------------------------------------------------------
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        tasks_df = pd.DataFrame([{"task_index": 0, "task": task_desc}])
        tasks_df["task_index"] = tasks_df["task_index"].astype("int64")
        tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)

        # ----------------------------------------------------------------
        # 4. meta/episodes/chunk-000/file-000.parquet
        # ----------------------------------------------------------------
        ep_dir = meta_dir / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True, exist_ok=True)

        ep_df = pd.DataFrame([{
            "episode_index": episode_index,
            "task_index": 0,
            "length": frame_count,
            "data_chunk_index": 0,
            "data_file_index": 0,
            "video_chunk_index": 0,
            "video_file_index": 0,
            "dataset_from_index": episode_index * frame_count,
            "dataset_to_index": episode_index * frame_count + frame_count,
        }])
        for col in ep_df.select_dtypes("object").columns:
            ep_df[col] = ep_df[col].astype("int64")
        ep_df = ep_df.astype("int64")
        ep_df.to_parquet(ep_dir / "file-000.parquet", index=False)

        # ----------------------------------------------------------------
        # 5. meta/info.json
        # ----------------------------------------------------------------
        features: Dict[str, Any] = {
            "index":         {"dtype": "int64",   "shape": [1]},
            "episode_index": {"dtype": "int64",   "shape": [1]},
            "frame_index":   {"dtype": "int64",   "shape": [1]},
            "timestamp":     {"dtype": "float32", "shape": [1]},
            "task_index":    {"dtype": "int64",   "shape": [1]},
            "is_first":      {"dtype": "bool",    "shape": [1]},
            "is_last":       {"dtype": "bool",    "shape": [1]},
            "is_terminal":   {"dtype": "bool",    "shape": [1]},
        }

        for label in self.label_manager.labels.values():
            features[f"annotation.{label.name}"] = {"dtype": "bool", "shape": [1]}

        if session is not None:
            try:
                if state_dim > 0:
                    state_names = self._state_vector_names(session)
                    features["observation.state"] = {
                        "dtype": "float32",
                        "shape": [state_dim],
                        "names": state_names,
                    }
                    # action mirrors observation.state (next-frame imitation convention)
                    features["action"] = {
                        "dtype": "float32",
                        "shape": [state_dim],
                        "names": state_names,
                    }
                for side in session.metadata.gripper_sides:
                    features[f"observation.gripper.{side}"] = {
                        "dtype": "float32",
                        "shape": [1],
                        "names": ["opening_mm"],
                    }
                for cam_key, cam_info in session.cameras.items():
                    features[f"observation.images.{cam_key}"] = {
                        "dtype": "video",
                        "shape": [cam_info.height, cam_info.width, 3],
                        "video_info": {
                            "video.fps": float(cam_info.fps),
                            "video.height": int(cam_info.height),
                            "video.width": int(cam_info.width),
                            "video.channels": 3,
                            "video.codec": "h264",
                            "video.pix_fmt": "yuv420p",
                            "video.vflip": vflip,
                        },
                    }
            except Exception as exc:
                logger.warning("Could not build full feature map: %s", exc)

        total_chunks = 1
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.dataset_info.get("robot_type", "vive_tracker"),
            "fps": fps,
            "total_episodes": 1,
            "total_frames": frame_count,
            "total_tasks": 1,
            "total_chunks": total_chunks,
            "chunks_size": max(frame_count, 1),
            "splits": {"train": f"0:{1}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "episodes_path": "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": features,
        }
        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)

        # ----------------------------------------------------------------
        # 6. meta/stats.json
        # ----------------------------------------------------------------
        self._write_stats_json(df, features, meta_dir)

        # ----------------------------------------------------------------
        # 7. meta/annotations.json — simple label list with frame ranges
        # ----------------------------------------------------------------
        self._write_annotations_json(meta_dir)

        # ----------------------------------------------------------------
        # 8. meta/quality.json — quality rating and flags
        # ----------------------------------------------------------------
        self._write_quality_json(meta_dir)

    @staticmethod
    def _ffmpeg_vflip(src: Path, dst: Path) -> None:
        """Re-encode *src* with a vertical flip into *dst* using ffmpeg.

        Uses stream-copy for audio (if any) and libx264 for video so the
        output is always a valid MP4 regardless of the source container.

        Raises:
            RuntimeError: if ffmpeg is not found or returns a non-zero exit code.
        """
        import subprocess as _sp

        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", "vflip",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            str(dst),
        ]
        try:
            result = _sp.run(
                cmd,
                stdout=_sp.PIPE,
                stderr=_sp.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {result.returncode}:\n"
                    + result.stderr.decode(errors="replace")[-2000:]
                )
            logger.info("vflip applied: %s -> %s", src.name, dst.name)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg introuvable — installez ffmpeg pour utiliser le vflip vidéo."
            )

    def _state_vector_names(self, session) -> List[str]:
        """Return human-readable names for each element of observation.state."""
        names: List[str] = []
        axes_pos = ["x", "y", "z"]
        axes_quat = ["qw", "qx", "qy", "qz"]
        for name in session.metadata.tracker_names:
            for ax in axes_pos:
                names.append(f"tracker_{name}_{ax}")
            for ax in axes_quat:
                names.append(f"tracker_{name}_{ax}")
        return names

    def _write_annotations_json(self, meta_dir: Path) -> None:
        """Write meta/annotations.json with simple label, start_frame, end_frame format.

        This provides a human-readable list of all annotations applied to the episode.
        """
        annotations_list = []

        for ann in self.label_manager.annotations:
            try:
                ann_data = {
                    "label": ann.label_name,
                    "type": ann.annotation_type.value,
                }

                if ann.annotation_type == LabelType.FRAME:
                    ann_data["frame"] = ann.frame_index
                    ann_data["start_frame"] = ann.frame_index
                    ann_data["end_frame"] = ann.frame_index
                else:  # INTERVAL
                    ann_data["start_frame"] = ann.start_frame
                    ann_data["end_frame"] = ann.end_frame

                if hasattr(ann, 'confidence') and ann.confidence is not None:
                    ann_data["confidence"] = ann.confidence

                annotations_list.append(ann_data)
            except Exception as exc:
                logger.warning("Could not serialize annotation: %s", exc)

        # Sort by start_frame for readability
        annotations_list.sort(key=lambda x: x.get("start_frame", 0))

        output = {
            "annotations": annotations_list,
            "total_count": len(annotations_list),
            "export_timestamp": datetime.now().isoformat(),
        }

        try:
            with open(meta_dir / "annotations.json", "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info("Wrote annotations.json with %d annotations", len(annotations_list))
        except Exception as exc:
            logger.warning("Could not write annotations.json: %s", exc)

    def _write_quality_json(self, meta_dir: Path) -> None:
        """Write meta/quality.json with quality rating and flags.

        This provides quality assessment information for the episode.
        """
        quality_data = {
            "rating": self.dataset_info.get("quality_rating"),
            "flags": self.dataset_info.get("quality_flags", []),
            "export_timestamp": datetime.now().isoformat(),
        }

        # Only write if rating is present
        if quality_data["rating"] is not None:
            try:
                with open(meta_dir / "quality.json", "w", encoding="utf-8") as f:
                    json.dump(quality_data, f, indent=2, default=str)
                logger.info(
                    "Wrote quality.json: rating=%s, flags=%s",
                    quality_data["rating"],
                    quality_data["flags"],
                )
            except Exception as exc:
                logger.warning("Could not write quality.json: %s", exc)

    @staticmethod
    def _write_stats_json(df: "pd.DataFrame", features: Dict, meta_dir: Path) -> None:
        """Compute and write meta/stats.json for all numeric/bool columns.

        NaN values are replaced with None so the output is valid JSON
        (NaN is not a legal JSON literal and is rejected by strict parsers).
        """
        import numpy as np

        def _nan_to_none(val):
            """Replace float NaN/Inf with None for valid JSON serialisation."""
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val

        def _clean_list(lst):
            return [_nan_to_none(v) for v in lst]

        stats: Dict[str, Any] = {}

        for col, feat in features.items():
            dtype = feat.get("dtype", "")
            if dtype == "video":
                continue
            if col not in df.columns:
                continue
            try:
                series = df[col]
                # observation.state is stored as list-of-floats per row → explode
                if dtype in ("float32", "float64") and series.dtype == object:
                    arr = np.array(series.tolist(), dtype=np.float64)
                    # Compute per-dimension stats ignoring NaN
                    mean_vals = np.nanmean(arr, axis=0)
                    std_vals = np.nanstd(arr, axis=0)
                    min_vals = np.nanmin(arr, axis=0)
                    max_vals = np.nanmax(arr, axis=0)
                    # Dimensions where ALL values were NaN → keep as None
                    all_nan_mask = np.all(np.isnan(arr), axis=0)
                    mean_vals[all_nan_mask] = np.nan
                    std_vals[all_nan_mask] = np.nan
                    min_vals[all_nan_mask] = np.nan
                    max_vals[all_nan_mask] = np.nan
                    stats[col] = {
                        "mean": _clean_list(mean_vals.tolist()),
                        "std": _clean_list(std_vals.tolist()),
                        "min": _clean_list(min_vals.tolist()),
                        "max": _clean_list(max_vals.tolist()),
                    }
                elif dtype in ("float32", "float64", "bool", "int64"):
                    arr = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
                    valid = arr[~np.isnan(arr)]
                    if len(valid) == 0:
                        continue
                    stats[col] = {
                        "mean": [float(valid.mean())],
                        "std": [float(valid.std())],
                        "min": [float(valid.min())],
                        "max": [float(valid.max())],
                    }
            except Exception as exc:
                logger.warning("Stats computation failed for column '%s': %s", col, exc)

        try:
            with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write stats.json: %s", exc)

    @staticmethod
    def _get_state_vector(session, t: float) -> list:
        """Build a flat state vector from all trackers at time t.

        Missing or NaN tracker values are replaced with 0.0 to produce a
        well-defined, NaN-free float32 vector suitable for LeRobot training.
        """
        import math
        state: List[float] = []
        try:
            tracker_state = session.get_tracker_state(t)
            for tid in session.metadata.tracker_names:
                if tid in tracker_state:
                    pos = tracker_state[tid]["position"]
                    quat = tracker_state[tid]["quaternion"]
                    for v in pos.tolist():
                        state.append(0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
                    for v in quat.tolist():
                        state.append(0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
                else:
                    state.extend([0.0] * 7)
        except Exception as exc:
            logger.warning("_get_state_vector error at t=%.3f: %s", t, exc)
        return state

    # ------------------------------------------------------------------
    # Chunk format  (chunk_000.{session_id}/)
    # ------------------------------------------------------------------

    def export_to_chunk_format(
        self,
        output_dir: Path,
        episode_index: int = 0,
        annotator: str = "",
        quality_rating: Optional[int] = None,
        quality_flags: Optional[List[str]] = None,
        episode_refused: bool = False,
        episode_refused_comment: str = "",
        scenario_name: str = "",
        scenario_action: str = "",
    ) -> None:
        """Export annotations in the chunk folder format.

        Produces inside *output_dir*:
            annotator_info.json
            quality_check.json          (only if episode is refused)
            episode_{N:06d}_subtitle.json
            episode_{N:06d}_video_quality.json

        The subtitle file contains all INTERVAL annotations sorted by
        start_frame, formatted as::

            [{"start": <int>, "end": <int>, "text": "<label_name>"}, ...]

        Raises:
            RuntimeError: on any I/O error.
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # ── annotator_info.json ──────────────────────────────────
            annotator_info = {
                "submited_at": datetime.now().isoformat(),
                "annotator": annotator,
                "scenario_name": scenario_name,
                "scenario_action": scenario_action,
            }
            with open(output_dir / "annotator_info.json", "w", encoding="utf-8") as f:
                json.dump(annotator_info, f, indent=4)

            # ── episode_subtitle.json ────────────────────────────────
            gripper_left: List[Dict[str, Any]] = []
            gripper_right: List[Dict[str, Any]] = []
            for ann in self.label_manager.annotations:
                try:
                    if ann.annotation_type == LabelType.INTERVAL:
                        is_fail = ann.metadata.get("fail", False) if ann.metadata else False
                        entry = {
                            "start": ann.start_frame,
                            "end": ann.end_frame,
                            "text": ("[fail] " if is_fail else "") + ann.label_name,
                        }
                        hand = ann.metadata.get("hand", "")
                        if hand == "left":
                            gripper_left.append(entry)
                        else:
                            gripper_right.append(entry)
                except Exception as exc:
                    logger.warning("Skipping annotation in subtitle export: %s", exc)

            gripper_left.sort(key=lambda x: x["start"])
            gripper_right.sort(key=lambda x: x["start"])

            subtitles = {
                "gripper_left": gripper_left,
                "gripper_right": gripper_right,
            }
            with open(output_dir / "episode_subtitle.json", "w", encoding="utf-8") as f:
                json.dump(subtitles, f, indent=4)

            # ── episode_quality.json ─────────────────────────────────
            episode_quality: Dict[str, Any] = {
                "flag": quality_flags or [],
                "note": quality_rating,
                "refused": episode_refused,
                "refused_comment": episode_refused_comment if episode_refused else "",
            }
            with open(output_dir / "episode_quality.json", "w", encoding="utf-8") as f:
                json.dump(episode_quality, f, indent=4)

            logger.info(
                "Exported chunk format to %s (%d subtitles)",
                output_dir, len(subtitles),
            )

        except Exception as exc:
            raise RuntimeError(f"Export chunk échoué vers {output_dir} : {exc}") from exc

    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------

    def export_to_coco_format(
        self,
        output_path: Path,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> None:
        """Export in COCO-like format for temporal action detection.

        Raises:
            RuntimeError: on any I/O error.
        """
        try:
            categories = []
            for idx, label in enumerate(self.label_manager.labels.values()):
                try:
                    categories.append({
                        "id": idx + 1,
                        "name": label.name,
                        "supercategory": "action",
                    })
                except Exception:
                    pass

            label_id_to_cat_id = {}
            try:
                label_id_to_cat_id = {
                    label.id: idx + 1
                    for idx, label in enumerate(self.label_manager.labels.values())
                }
            except Exception as exc:
                logger.warning("Could not build label→category map: %s", exc)

            annotations = []
            for ann in self.label_manager.annotations:
                try:
                    coco_ann: Dict[str, Any] = {
                        "id": ann.id,
                        "category_id": label_id_to_cat_id.get(ann.label_id, 0),
                        "category_name": ann.label_name,
                    }
                    if ann.annotation_type == LabelType.FRAME:
                        coco_ann["frame_index"] = ann.frame_index
                        coco_ann["temporal_extent"] = [ann.frame_index, ann.frame_index]
                    else:
                        coco_ann["start_frame"] = ann.start_frame
                        coco_ann["end_frame"] = ann.end_frame
                        coco_ann["temporal_extent"] = [ann.start_frame, ann.end_frame]
                    coco_ann["confidence"] = ann.confidence
                    annotations.append(coco_ann)
                except Exception as exc:
                    logger.warning("Skipping malformed annotation in COCO export: %s", exc)

            coco_data = {
                "info": {
                    "description": "VIVE Labeler Export",
                    "version": "1.0",
                    "date_created": datetime.now().isoformat(),
                },
                "video": {
                    "id": 1,
                    "width": image_width,
                    "height": image_height,
                    "fps": self.dataset_info.get("fps", 30.0),
                    "duration": self.dataset_info.get("duration", 0.0),
                },
                "categories": categories,
                "annotations": annotations,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(coco_data, f, indent=2, default=str)

            logger.info("Exported COCO annotations to %s", output_path)

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Export COCO échoué vers {output_path} : {exc}") from exc

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def export_summary_report(self, output_path: Path) -> None:
        """Export a human-readable summary report.

        Raises:
            RuntimeError: on any I/O error.
        """
        try:
            stats = self.label_manager.get_statistics()

            lines = [
                "VIVE Labeler Annotation Report",
                "=" * 50,
                "",
                f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]

            if self.dataset_info:
                lines.extend(["Dataset Information:", "-" * 50])
                for key, value in self.dataset_info.items():
                    lines.append(f"  {key}: {value}")
                lines.append("")

            lines.extend([
                "Annotation Statistics:",
                "-" * 50,
                f"  Total Labels: {stats.get('total_labels', 0)}",
                f"  Total Annotations: {stats.get('total_annotations', 0)}",
                f"    - Frame annotations: {stats.get('frame_annotations', 0)}",
                f"    - Interval annotations: {stats.get('interval_annotations', 0)}",
                "",
                "Annotations by Label:",
                "-" * 50,
            ])

            for label_name, count in stats.get("annotations_by_label", {}).items():
                lines.append(f"  {label_name}: {count}")

            lines.append("")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            logger.info("Exported summary report to %s", output_path)

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Export rapport échoué vers {output_path} : {exc}") from exc
