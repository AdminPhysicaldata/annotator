"""Synchronization module for video and sensor data.

Handles temporal alignment between:
- Video frames (from MP4 files)
- Sensor/state data (from Parquet files)
- VIVE tracking data (position, rotation)

Provides interpolation when exact timestamps don't match.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.interpolate import interp1d

from .transforms import quaternion_slerp


@dataclass
class SyncedFrame:
    """Synchronized frame containing video and sensor data."""
    frame_index: int
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    sensor_data: Dict[str, np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.sensor_data is None:
            self.sensor_data = {}
        if self.metadata is None:
            self.metadata = {}


class DataSynchronizer:
    """Synchronizes video frames with sensor data."""

    def __init__(
        self,
        video_fps: float,
        video_frame_count: int,
        sensor_timestamps: np.ndarray,
        sensor_data: Dict[str, np.ndarray],
        interpolation_method: str = "linear",
        max_time_diff_ms: float = 50.0
    ):
        """Initialize synchronizer.

        Args:
            video_fps: Video framerate
            video_frame_count: Total number of video frames
            sensor_timestamps: Timestamps for sensor data (in seconds)
            sensor_data: Dictionary mapping sensor keys to data arrays
            interpolation_method: Interpolation method ('linear', 'nearest', 'cubic')
            max_time_diff_ms: Maximum acceptable time difference in milliseconds
        """
        self.video_fps = video_fps
        self.video_frame_count = video_frame_count
        self.sensor_timestamps = np.array(sensor_timestamps)
        self.sensor_data = sensor_data
        self.interpolation_method = interpolation_method
        self.max_time_diff_ms = max_time_diff_ms

        # Compute video timestamps
        self.video_timestamps = np.arange(video_frame_count) / video_fps

        # Build interpolators for each sensor
        self._build_interpolators()

        # Compute synchronization quality metrics
        self._compute_sync_metrics()

    def _build_interpolators(self) -> None:
        """Build interpolation functions for each sensor channel."""
        self.interpolators = {}

        for key, data in self.sensor_data.items():
            if len(data.shape) == 1:
                # Scalar or 1D data
                if self.interpolation_method == "nearest":
                    kind = "nearest"
                elif self.interpolation_method == "cubic":
                    kind = "cubic"
                else:
                    kind = "linear"

                self.interpolators[key] = interp1d(
                    self.sensor_timestamps,
                    data,
                    kind=kind,
                    bounds_error=False,
                    fill_value=(data[0], data[-1])
                )
            else:
                # Multi-dimensional data (e.g., 3D position, quaternions)
                # Interpolate each dimension separately
                interpolators_list = []
                for i in range(data.shape[1]):
                    if self.interpolation_method == "nearest":
                        kind = "nearest"
                    elif self.interpolation_method == "cubic":
                        kind = "cubic"
                    else:
                        kind = "linear"

                    interp = interp1d(
                        self.sensor_timestamps,
                        data[:, i],
                        kind=kind,
                        bounds_error=False,
                        fill_value=(data[0, i], data[-1, i])
                    )
                    interpolators_list.append(interp)

                self.interpolators[key] = interpolators_list

    def _compute_sync_metrics(self) -> None:
        """Compute synchronization quality metrics."""
        # Check time coverage
        self.video_start_time = self.video_timestamps[0]
        self.video_end_time = self.video_timestamps[-1]
        self.sensor_start_time = self.sensor_timestamps[0]
        self.sensor_end_time = self.sensor_timestamps[-1]

        # Check for gaps
        self.has_full_coverage = (
            self.sensor_start_time <= self.video_start_time and
            self.sensor_end_time >= self.video_end_time
        )

        # Compute average sampling rates
        self.video_dt = 1.0 / self.video_fps
        sensor_dts = np.diff(self.sensor_timestamps)
        self.sensor_dt_mean = np.mean(sensor_dts)
        self.sensor_dt_std = np.std(sensor_dts)

        # Check if sensor sampling is regular
        self.sensor_is_regular = self.sensor_dt_std < 0.001  # 1ms threshold

    def get_synced_frame(
        self,
        frame_index: int,
        video_frame: Optional[np.ndarray] = None
    ) -> SyncedFrame:
        """Get synchronized data for a specific video frame.

        Args:
            frame_index: Video frame index
            video_frame: Optional video frame data (image)

        Returns:
            SyncedFrame object with interpolated sensor data
        """
        if frame_index < 0 or frame_index >= self.video_frame_count:
            raise IndexError(f"Frame index {frame_index} out of range")

        timestamp = self.video_timestamps[frame_index]

        # Interpolate sensor data at this timestamp
        sensor_data = {}
        for key, interpolator in self.interpolators.items():
            if isinstance(interpolator, list):
                # Multi-dimensional data
                values = np.array([interp(timestamp) for interp in interpolator])
                sensor_data[key] = values
            else:
                # Scalar data
                sensor_data[key] = float(interpolator(timestamp))

        # Check if we need to warn about time difference
        closest_sensor_idx = np.argmin(np.abs(self.sensor_timestamps - timestamp))
        time_diff_ms = abs(self.sensor_timestamps[closest_sensor_idx] - timestamp) * 1000

        metadata = {
            'interpolated': time_diff_ms > 1.0,  # 1ms threshold
            'time_diff_ms': time_diff_ms,
            'needs_warning': time_diff_ms > self.max_time_diff_ms,
        }

        return SyncedFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            video_frame=video_frame,
            sensor_data=sensor_data,
            metadata=metadata
        )

    def get_synced_frames(
        self,
        start_frame: int,
        end_frame: int,
        include_video: bool = False
    ) -> List[SyncedFrame]:
        """Get multiple synchronized frames.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            include_video: If True, include video frame data (requires separate loading)

        Returns:
            List of SyncedFrame objects
        """
        frames = []
        for i in range(start_frame, end_frame):
            frame = self.get_synced_frame(i, video_frame=None)
            frames.append(frame)
        return frames

    def interpolate_quaternion(
        self,
        timestamp: float,
        quaternion_key: str
    ) -> np.ndarray:
        """Interpolate quaternion using SLERP.

        Args:
            timestamp: Target timestamp
            quaternion_key: Key for quaternion data in sensor_data

        Returns:
            Interpolated quaternion [w, x, y, z]
        """
        if quaternion_key not in self.sensor_data:
            raise KeyError(f"Quaternion key '{quaternion_key}' not found in sensor data")

        quaternions = self.sensor_data[quaternion_key]
        timestamps = self.sensor_timestamps

        # Find bracketing timestamps
        idx_after = np.searchsorted(timestamps, timestamp)

        if idx_after == 0:
            return quaternions[0]
        elif idx_after >= len(timestamps):
            return quaternions[-1]

        idx_before = idx_after - 1
        t0 = timestamps[idx_before]
        t1 = timestamps[idx_after]
        q0 = quaternions[idx_before]
        q1 = quaternions[idx_after]

        # Compute interpolation parameter
        alpha = (timestamp - t0) / (t1 - t0) if t1 > t0 else 0.0

        # Use SLERP
        return quaternion_slerp(q0, q1, alpha)

    def find_nearest_sensor_index(self, timestamp: float) -> int:
        """Find index of sensor data closest to given timestamp.

        Args:
            timestamp: Target timestamp in seconds

        Returns:
            Index of nearest sensor sample
        """
        return np.argmin(np.abs(self.sensor_timestamps - timestamp))

    def get_time_range(self) -> Tuple[float, float]:
        """Get time range covered by synchronized data.

        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        start = max(self.video_start_time, self.sensor_start_time)
        end = min(self.video_end_time, self.sensor_end_time)
        return (start, end)

    def get_sync_report(self) -> Dict[str, Any]:
        """Get synchronization quality report.

        Returns:
            Dictionary with synchronization metrics
        """
        return {
            'video_fps': self.video_fps,
            'video_frame_count': self.video_frame_count,
            'video_duration': self.video_end_time - self.video_start_time,
            'sensor_sample_count': len(self.sensor_timestamps),
            'sensor_duration': self.sensor_end_time - self.sensor_start_time,
            'sensor_mean_dt': self.sensor_dt_mean,
            'sensor_std_dt': self.sensor_dt_std,
            'sensor_is_regular': self.sensor_is_regular,
            'has_full_coverage': self.has_full_coverage,
            'time_range': self.get_time_range(),
            'max_time_diff_ms': self.max_time_diff_ms,
        }

    def __repr__(self) -> str:
        """String representation."""
        report = self.get_sync_report()
        return (
            f"DataSynchronizer(\n"
            f"  video: {report['video_frame_count']} frames @ {report['video_fps']} fps\n"
            f"  sensor: {report['sensor_sample_count']} samples @ {1/report['sensor_mean_dt']:.1f} Hz\n"
            f"  coverage: {'full' if report['has_full_coverage'] else 'partial'}\n"
            f"  time_range: {report['time_range'][0]:.2f}s - {report['time_range'][1]:.2f}s\n"
            f")"
        )
