"""Tests for 3D transformation utilities."""

import pytest
import numpy as np
from src.core.transforms import (
    Transform3D,
    quaternion_to_euler,
    euler_to_quaternion,
    quaternion_slerp,
    compute_distance,
    rotation_angle_between
)


class TestTransform3D:
    """Tests for Transform3D class."""

    def test_identity_transform(self):
        """Test identity transformation."""
        t = Transform3D(position=np.array([0, 0, 0]))

        # Check identity quaternion
        q = t.quaternion
        assert np.allclose(q, [1, 0, 0, 0])

        # Check identity euler angles
        euler = t.euler
        assert np.allclose(euler, [0, 0, 0])

        # Check identity matrix
        mat = t.matrix
        assert np.allclose(mat, np.eye(3))

    def test_transform_point(self):
        """Test point transformation."""
        # Translation only
        t = Transform3D(position=np.array([1, 2, 3]))
        point = np.array([0, 0, 0])
        result = t.transform_point(point)
        assert np.allclose(result, [1, 2, 3])

    def test_inverse_transform(self):
        """Test inverse transformation."""
        t = Transform3D(
            position=np.array([1, 2, 3]),
            rotation=np.array([0, 0, np.pi/4]),
            rotation_format="euler"
        )
        t_inv = t.inverse()

        # Compose with inverse should give identity
        identity_pos = t.transform_point(t_inv.position)
        assert np.allclose(identity_pos, [0, 0, 0], atol=1e-6)

    def test_transform_composition(self):
        """Test composition of transformations."""
        t1 = Transform3D(position=np.array([1, 0, 0]))
        t2 = Transform3D(position=np.array([0, 1, 0]))

        t_composed = t1 * t2
        expected_pos = np.array([1, 1, 0])
        assert np.allclose(t_composed.position, expected_pos)


class TestQuaternionConversions:
    """Tests for quaternion conversion functions."""

    def test_euler_to_quaternion_identity(self):
        """Test identity conversion."""
        euler = np.array([0, 0, 0])
        q = euler_to_quaternion(euler)
        assert np.allclose(q, [1, 0, 0, 0])

    def test_quaternion_to_euler_identity(self):
        """Test identity conversion."""
        q = np.array([1, 0, 0, 0])
        euler = quaternion_to_euler(q)
        assert np.allclose(euler, [0, 0, 0])

    def test_round_trip_conversion(self):
        """Test euler -> quaternion -> euler round trip."""
        original_euler = np.array([0.1, 0.2, 0.3])
        q = euler_to_quaternion(original_euler)
        result_euler = quaternion_to_euler(q)
        assert np.allclose(original_euler, result_euler, atol=1e-6)


class TestQuaternionSlerp:
    """Tests for quaternion SLERP."""

    def test_slerp_endpoints(self):
        """Test SLERP at endpoints."""
        q0 = np.array([1, 0, 0, 0])
        q1 = euler_to_quaternion(np.array([0, 0, np.pi/2]))

        # At t=0, should return q0
        result = quaternion_slerp(q0, q1, 0.0)
        assert np.allclose(result, q0)

        # At t=1, should return q1
        result = quaternion_slerp(q0, q1, 1.0)
        assert np.allclose(result, q1, atol=1e-6)

    def test_slerp_midpoint(self):
        """Test SLERP at midpoint."""
        q0 = np.array([1, 0, 0, 0])
        q1 = euler_to_quaternion(np.array([0, 0, np.pi/2]))

        result = quaternion_slerp(q0, q1, 0.5)

        # Should be approximately halfway
        result_euler = quaternion_to_euler(result)
        expected_euler = np.array([0, 0, np.pi/4])
        assert np.allclose(result_euler, expected_euler, atol=0.01)


class TestDistanceAndAngles:
    """Tests for distance and angle computations."""

    def test_compute_distance(self):
        """Test Euclidean distance."""
        pos1 = np.array([0, 0, 0])
        pos2 = np.array([1, 1, 1])
        distance = compute_distance(pos1, pos2)
        assert np.isclose(distance, np.sqrt(3))

    def test_rotation_angle_identity(self):
        """Test rotation angle for identity."""
        q = np.array([1, 0, 0, 0])
        angle = rotation_angle_between(q, q)
        assert np.isclose(angle, 0.0)

    def test_rotation_angle_90deg(self):
        """Test rotation angle for 90 degree rotation."""
        q0 = np.array([1, 0, 0, 0])
        q1 = euler_to_quaternion(np.array([0, 0, np.pi/2]))
        angle = rotation_angle_between(q0, q1)
        assert np.isclose(angle, np.pi/2, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__])
