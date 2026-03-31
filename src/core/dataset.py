"""Main dataset abstraction for VIVE Labeler.

Provides a unified interface for accessing video, sensor data, and VIVE tracking
with automatic synchronization.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .data_loader import LeRobotDataLoader
from .synchronizer import DataSynchronizer, SyncedFrame
from .transforms import Transform3D
from ..utils.helpers import get_video_info


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    episode_index: int
    episode_id: str
    frame_count: int
    duration: float
    fps: float
    camera_keys: List[str]
    sensor_keys: List[str]
    has_vive_tracking: bool


class ViVEDataset:
    """Main dataset class combining video, sensor, and VIVE tracking data."""

    def __init__(
        self,
        dataset_path: str,
        episode_index: int = 0,
        use_streaming: bool = False,
        cache_dir: Optional[Path] = None,
        interpolation_method: str = "linear"
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to LeRobot dataset or HuggingFace repo ID
            episode_index: Index of episode to load
            use_streaming: If True, stream data without downloading
            cache_dir: Cache directory for downloaded data
            interpolation_method: Method for temporal interpolation
        """
        self.dataset_path = dataset_path
        self.episode_index = episode_index
        self.interpolation_method = interpolation_method

        # Load dataset
        self.loader = LeRobotDataLoader(
            dataset_path=dataset_path,
            use_streaming=use_streaming,
            cache_dir=cache_dir
        )

        # Episode data
        self.current_episode: Optional[Dict] = None
        self.episode_metadata: Optional[EpisodeMetadata] = None

        # Video data
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        self.video_info: Dict[str, Dict] = {}

        # Synchronization
        self.synchronizers: Dict[str, DataSynchronizer] = {}

        # Load episode
        self.load_episode(episode_index)

    def load_episode(self, episode_index: int) -> None:
        """Load a specific episode.

        Args:
            episode_index: Index of episode to load
        """
        self.episode_index = episode_index

        # Release previous video captures
        self._release_videos()

        # Load episode data from loader
        self.current_episode = self.loader.get_episode(episode_index)
        episode_info = self.current_episode['episode_info']
        samples = self.current_episode['samples']

        # Extract metadata
        camera_keys = self.loader.get_camera_keys()
        sensor_keys = self.loader.get_state_keys()

        # Load videos
        for camera_key in camera_keys:
            video_path = self.loader.get_video_path(camera_key, episode_index)
            if video_path and video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    self.video_captures[camera_key] = cap
                    self.video_info[camera_key] = get_video_info(video_path)

        # Extract sensor data and timestamps
        timestamps = np.array([s['timestamp'].item() for s in samples])

        sensor_data = {}
        for key in sensor_keys:
            values = []
            for s in samples:
                if key in s:
                    val = s[key]
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    values.append(val)
            if values:
                sensor_data[key] = np.array(values)

        # Check for VIVE tracking data
        vive_keys = [k for k in sensor_keys if 'vive' in k.lower() or 'tracker' in k.lower()]
        has_vive_tracking = len(vive_keys) > 0

        # Build synchronizers for each camera
        for camera_key, info in self.video_info.items():
            self.synchronizers[camera_key] = DataSynchronizer(
                video_fps=info['fps'],
                video_frame_count=info['frame_count'],
                sensor_timestamps=timestamps,
                sensor_data=sensor_data,
                interpolation_method=self.interpolation_method
            )

        # Create metadata
        primary_camera = camera_keys[0] if camera_keys else None
        fps = self.video_info[primary_camera]['fps'] if primary_camera else 30.0
        frame_count = self.video_info[primary_camera]['frame_count'] if primary_camera else len(samples)
        duration = frame_count / fps if fps > 0 else 0.0

        self.episode_metadata = EpisodeMetadata(
            episode_index=episode_index,
            episode_id=episode_info.get('episode_id', f'episode_{episode_index:06d}'),
            frame_count=frame_count,
            duration=duration,
            fps=fps,
            camera_keys=camera_keys,
            sensor_keys=sensor_keys,
            has_vive_tracking=has_vive_tracking
        )

    def get_frame(
        self,
        frame_index: int,
        camera_key: Optional[str] = None,
        include_sensor_data: bool = True
    ) -> Dict[str, Any]:
        """Get a specific frame with synchronized data.

        Args:
            frame_index: Frame index
            camera_key: Camera to use (if None, use first available)
            include_sensor_data: If True, include synchronized sensor data

        Returns:
            Dictionary with frame data
        """
        if camera_key is None:
            camera_key = self.episode_metadata.camera_keys[0]

        if camera_key not in self.video_captures:
            raise KeyError(f"Camera '{camera_key}' not available")

        # Read video frame
        cap = self.video_captures[camera_key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_index} from camera '{camera_key}'")

        # Get synchronized sensor data
        synced_frame = None
        if include_sensor_data and camera_key in self.synchronizers:
            synced_frame = self.synchronizers[camera_key].get_synced_frame(
                frame_index=frame_index,
                video_frame=frame
            )

        return {
            'frame_index': frame_index,
            'timestamp': frame_index / self.episode_metadata.fps,
            'camera_key': camera_key,
            'frame': frame,
            'sensor_data': synced_frame.sensor_data if synced_frame else {},
            'metadata': synced_frame.metadata if synced_frame else {},
        }

    def get_vive_transform(
        self,
        frame_index: int,
        tracker_name: str = "vive",
        camera_key: Optional[str] = None
    ) -> Optional[Transform3D]:
        """Get VIVE tracker transform at a specific frame.

        Args:
            frame_index: Frame index
            tracker_name: Name of VIVE tracker in sensor data
            camera_key: Camera to sync with (if None, use first available)

        Returns:
            Transform3D object or None if not available
        """
        if camera_key is None:
            camera_key = self.episode_metadata.camera_keys[0]

        if camera_key not in self.synchronizers:
            return None

        synced_frame = self.synchronizers[camera_key].get_synced_frame(frame_index)

        # Look for position and rotation data
        position_key = None
        rotation_key = None

        for key in synced_frame.sensor_data.keys():
            key_lower = key.lower()
            if tracker_name.lower() in key_lower:
                if 'position' in key_lower or 'pos' in key_lower:
                    position_key = key
                elif 'rotation' in key_lower or 'quat' in key_lower or 'orient' in key_lower:
                    rotation_key = key

        if position_key is None:
            return None

        position = synced_frame.sensor_data[position_key]

        # Check if we have rotation
        rotation = None
        rotation_format = "quaternion"
        if rotation_key is not None:
            rotation = synced_frame.sensor_data[rotation_key]
            # Infer format based on size
            if len(rotation) == 4:
                rotation_format = "quaternion"
            elif len(rotation) == 3:
                rotation_format = "euler"
            elif len(rotation) == 9:
                rotation_format = "matrix"
                rotation = rotation.reshape(3, 3)

        return Transform3D(position=position, rotation=rotation, rotation_format=rotation_format)

    def get_trajectory(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        tracker_name: str = "vive"
    ) -> np.ndarray:
        """Get 3D trajectory of VIVE tracker over a range of frames.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (if None, use last frame)
            tracker_name: Name of VIVE tracker

        Returns:
            Array of shape (N, 3) with 3D positions
        """
        if end_frame is None:
            end_frame = self.episode_metadata.frame_count

        positions = []
        for i in range(start_frame, end_frame):
            transform = self.get_vive_transform(i, tracker_name=tracker_name)
            if transform is not None:
                positions.append(transform.position)

        return np.array(positions) if positions else np.zeros((0, 3))

    def get_num_episodes(self) -> int:
        """Get total number of episodes in dataset."""
        return self.loader.num_episodes

    def get_frame_count(self) -> int:
        """Get number of frames in current episode."""
        return self.episode_metadata.frame_count if self.episode_metadata else 0

    def get_duration(self) -> float:
        """Get duration of current episode in seconds."""
        return self.episode_metadata.duration if self.episode_metadata else 0.0

    def _release_videos(self) -> None:
        """Release all video captures."""
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()

    def __del__(self):
        """Cleanup resources."""
        self._release_videos()

    def __len__(self) -> int:
        """Get number of frames."""
        return self.get_frame_count()

    def __repr__(self) -> str:
        """String representation."""
        if self.episode_metadata:
            return (
                f"ViVEDataset(\n"
                f"  episode={self.episode_metadata.episode_id},\n"
                f"  frames={self.episode_metadata.frame_count},\n"
                f"  duration={self.episode_metadata.duration:.2f}s,\n"
                f"  cameras={self.episode_metadata.camera_keys},\n"
                f"  vive_tracking={self.episode_metadata.has_vive_tracking}\n"
                f")"
            )
        return f"ViVEDataset(not loaded)"
