"""
Geometry utilities for point cloud processing.

This module provides common geometric operations:
- Rotation matrix creation from Euler angles
- Point transformations
- Distance calculations

Functions:
    rotation_matrix_from_euler: Create rotation matrix from roll, pitch, yaw
    transform_points: Apply rigid transformation to points
    homogeneous_transform: Apply 4x4 transformation matrix

TODO: In Phase 3, extract duplicate functions from:
    - python/icp_3d/point_to_point_utils_3d.py
    - python/icp_3d/point_to_plane_utils_3d.py
    - python/pytorch/rotate.py
"""

import numpy as np
from typing import Union, Tuple


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create 3D rotation matrix from Euler angles.

    Args:
        roll: Rotation around X-axis (radians)
        pitch: Rotation around Y-axis (radians)
        yaw: Rotation around Z-axis (radians)

    Returns:
        3x3 rotation matrix

    Example:
        >>> R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        >>> R.shape
        (3, 3)
    """
    # Rotation around X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation around Y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: Union[np.ndarray, list] = None
) -> np.ndarray:
    """
    Apply rigid transformation (rotation + translation) to points.

    Args:
        points: (N, 3) array of 3D points
        rotation: (3, 3) rotation matrix
        translation: (3,) translation vector or None

    Returns:
        (N, 3) array of transformed points

    Example:
        >>> points = np.random.rand(100, 3)
        >>> R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        >>> t = np.array([1.0, 2.0, 3.0])
        >>> transformed = transform_points(points, R, t)
    """
    # Apply rotation
    transformed = (rotation @ points.T).T

    # Apply translation if provided
    if translation is not None:
        translation = np.asarray(translation)
        transformed = transformed + translation

    return transformed


def homogeneous_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 homogeneous transformation matrix to points.

    Args:
        points: (N, 3) array of 3D points
        matrix: (4, 4) homogeneous transformation matrix

    Returns:
        (N, 3) array of transformed points

    Example:
        >>> points = np.random.rand(100, 3)
        >>> T = np.eye(4)
        >>> T[:3, :3] = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        >>> T[:3, 3] = [1, 2, 3]
        >>> transformed = homogeneous_transform(points, T)
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = (matrix @ points_homogeneous.T).T

    # Convert back to 3D
    transformed = transformed_homogeneous[:, :3]

    return transformed


def euler_from_rotation_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians

    Example:
        >>> R = rotation_matrix_from_euler(0.1, 0.2, 0.3)
        >>> roll, pitch, yaw = euler_from_rotation_matrix(R)
    """
    # TODO: Implement in Phase 3
    # Extract from existing implementations
    raise NotImplementedError("To be implemented in Phase 3")


# TODO: Phase 3 - Add more functions extracted from projects:
# - quaternion_from_rotation_matrix
# - rotation_matrix_from_quaternion
# - axis_angle_from_rotation_matrix
# - inverse_transform
# - compose_transforms
