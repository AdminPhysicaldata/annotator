"""3D transformation utilities for VIVE tracking data.

Handles conversion between different rotation representations:
- Quaternions (w, x, y, z)
- Euler angles (roll, pitch, yaw)
- Rotation matrices (3x3)

Also provides utilities for position transformations and trajectory computation.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy.spatial.transform import Rotation as R


class Transform3D:
    """Represents a 3D transformation (position + rotation)."""

    def __init__(
        self,
        position: np.ndarray,
        rotation: Optional[Union[np.ndarray, R]] = None,
        rotation_format: str = "quaternion"
    ):
        """Initialize 3D transform.

        Args:
            position: 3D position vector [x, y, z]
            rotation: Rotation as quaternion [w,x,y,z], euler angles [r,p,y],
                     or scipy Rotation object
            rotation_format: Format of rotation ('quaternion', 'euler', 'matrix')
        """
        self.position = np.array(position, dtype=np.float64)

        if rotation is None:
            self.rotation = R.identity()
        elif isinstance(rotation, R):
            self.rotation = rotation
        else:
            rotation = np.array(rotation, dtype=np.float64)
            if rotation_format == "quaternion":
                # scipy expects [x, y, z, w], but we use [w, x, y, z]
                self.rotation = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
            elif rotation_format == "euler":
                self.rotation = R.from_euler('xyz', rotation, degrees=False)
            elif rotation_format == "matrix":
                self.rotation = R.from_matrix(rotation)
            else:
                raise ValueError(f"Unknown rotation format: {rotation_format}")

    @property
    def quaternion(self) -> np.ndarray:
        """Get rotation as quaternion [w, x, y, z]."""
        q = self.rotation.as_quat()  # [x, y, z, w] from scipy
        return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w, x, y, z]

    @property
    def euler(self) -> np.ndarray:
        """Get rotation as euler angles [roll, pitch, yaw] in radians."""
        return self.rotation.as_euler('xyz', degrees=False)

    @property
    def matrix(self) -> np.ndarray:
        """Get rotation as 3x3 rotation matrix."""
        return self.rotation.as_matrix()

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get full 4x4 transformation matrix."""
        mat = np.eye(4)
        mat[:3, :3] = self.matrix
        mat[:3, 3] = self.position
        return mat

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Apply transformation to a 3D point.

        Args:
            point: 3D point [x, y, z]

        Returns:
            Transformed point
        """
        return self.rotation.apply(point) + self.position

    def inverse(self) -> "Transform3D":
        """Get inverse transformation."""
        inv_rot = self.rotation.inv()
        inv_pos = -inv_rot.apply(self.position)
        return Transform3D(inv_pos, inv_rot)

    def __mul__(self, other: "Transform3D") -> "Transform3D":
        """Compose two transformations."""
        new_rot = self.rotation * other.rotation
        new_pos = self.transform_point(other.position)
        return Transform3D(new_pos, new_rot)


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw].

    Args:
        quaternion: Quaternion as [w, x, y, z]

    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    # Convert to scipy format [x, y, z, w]
    q = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=False)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles [roll, pitch, yaw] to quaternion [w, x, y, z].

    Args:
        euler: Euler angles [roll, pitch, yaw] in radians

    Returns:
        Quaternion as [w, x, y, z]
    """
    r = R.from_euler('xyz', euler, degrees=False)
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # Convert to [w, x, y, z]


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: First quaternion [w, x, y, z]
        q1: Second quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    # Convert to scipy format [x, y, z, w]
    q0_scipy = [q0[1], q0[2], q0[3], q0[0]]
    q1_scipy = [q1[1], q1[2], q1[3], q1[0]]

    r0 = R.from_quat(q0_scipy)
    r1 = R.from_quat(q1_scipy)

    # Use scipy's Slerp
    from scipy.spatial.transform import Slerp
    times = [0, 1]
    rotations = R.concatenate([r0, r1])
    slerp = Slerp(times, rotations)

    r_interp = slerp(t)
    q_interp = r_interp.as_quat()  # [x, y, z, w]

    return np.array([q_interp[3], q_interp[0], q_interp[1], q_interp[2]])


def compute_trajectory(positions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute smoothed trajectory from position sequence.

    Args:
        positions: Array of shape (N, 3) with 3D positions
        window_size: Size of smoothing window

    Returns:
        Smoothed trajectory of same shape
    """
    if len(positions) < window_size:
        return positions

    # Simple moving average
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(positions, size=window_size, axis=0, mode='nearest')
    return smoothed


def compute_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute velocity from position sequence.

    Args:
        positions: Array of shape (N, 3) with 3D positions
        timestamps: Array of shape (N,) with timestamps in seconds

    Returns:
        Velocity vectors of shape (N, 3) in m/s
    """
    if len(positions) < 2:
        return np.zeros_like(positions)

    # Compute finite differences
    dt = np.diff(timestamps)
    dp = np.diff(positions, axis=0)

    # Avoid division by zero
    dt = np.maximum(dt, 1e-6)

    velocity = dp / dt[:, np.newaxis]

    # Pad to maintain shape (repeat first velocity)
    velocity = np.vstack([velocity[0:1], velocity])

    return velocity


def compute_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D positions.

    Args:
        pos1: First position [x, y, z]
        pos2: Second position [x, y, z]

    Returns:
        Distance in meters
    """
    return np.linalg.norm(pos2 - pos1)


def rotation_angle_between(q0: np.ndarray, q1: np.ndarray) -> float:
    """Compute rotation angle between two quaternions.

    Args:
        q0: First quaternion [w, x, y, z]
        q1: Second quaternion [w, x, y, z]

    Returns:
        Rotation angle in radians
    """
    # Convert to scipy format
    q0_scipy = [q0[1], q0[2], q0[3], q0[0]]
    q1_scipy = [q1[1], q1[2], q1[3], q1[0]]

    r0 = R.from_quat(q0_scipy)
    r1 = R.from_quat(q1_scipy)

    # Compute relative rotation
    r_rel = r1 * r0.inv()

    # Get rotation angle
    return r_rel.magnitude()
