"""
Evaluation metrics for point clouds.

This module provides common metrics for evaluating point cloud alignment,
registration, and neural network performance.

Functions:
    mean_squared_error: MSE between point sets
    root_mean_squared_error: RMSE
    chamfer_distance: Chamfer distance metric
    hausdorff_distance: Hausdorff distance

TODO: In Phase 3, consolidate metrics from:
    - python/pytorch/metrics.py
    - python/pytorch_geometric/metrics.py
    - python/pytorch_voxel/metrics.py
"""

import numpy as np
from typing import Optional
from scipy.spatial import distance_matrix


def mean_squared_error(source: np.ndarray, target: np.ndarray) -> float:
    """
    Compute mean squared error between two point sets.

    Args:
        source: (N, D) array of points
        target: (N, D) array of points (same shape as source)

    Returns:
        MSE value

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = source + np.random.rand(100, 3) * 0.1
        >>> error = mean_squared_error(source, target)
    """
    return np.mean(np.sum((source - target) ** 2, axis=1))


def root_mean_squared_error(source: np.ndarray, target: np.ndarray) -> float:
    """
    Compute root mean squared error between two point sets.

    Args:
        source: (N, D) array of points
        target: (N, D) array of points (same shape as source)

    Returns:
        RMSE value

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = source + np.random.rand(100, 3) * 0.1
        >>> error = root_mean_squared_error(source, target)
    """
    return np.sqrt(mean_squared_error(source, target))


def chamfer_distance(source: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point sets.

    Chamfer distance measures:
    - For each point in source, distance to nearest point in target
    - For each point in target, distance to nearest point in source
    - Average of both directions

    Args:
        source: (N, D) array of points
        target: (M, D) array of points

    Returns:
        Chamfer distance

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = np.random.rand(120, 3)
        >>> cd = chamfer_distance(source, target)
    """
    # Compute pairwise distances
    dists_src_to_tgt = distance_matrix(source, target)
    dists_tgt_to_src = distance_matrix(target, source)

    # Find nearest neighbor distances
    min_dists_src_to_tgt = np.min(dists_src_to_tgt, axis=1)
    min_dists_tgt_to_src = np.min(dists_tgt_to_src, axis=1)

    # Chamfer distance is average of both directions
    chamfer = (np.mean(min_dists_src_to_tgt) + np.mean(min_dists_tgt_to_src)) / 2.0

    return chamfer


def hausdorff_distance(source: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two point sets.

    Hausdorff distance is the maximum distance from a point in one set
    to the nearest point in the other set.

    Args:
        source: (N, D) array of points
        target: (M, D) array of points

    Returns:
        Hausdorff distance

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = np.random.rand(120, 3)
        >>> hd = hausdorff_distance(source, target)
    """
    # Compute pairwise distances
    dists_src_to_tgt = distance_matrix(source, target)
    dists_tgt_to_src = distance_matrix(target, source)

    # Find nearest neighbor distances
    min_dists_src_to_tgt = np.min(dists_src_to_tgt, axis=1)
    min_dists_tgt_to_src = np.min(dists_tgt_to_src, axis=1)

    # Hausdorff distance is maximum of both directions
    hausdorff = max(np.max(min_dists_src_to_tgt), np.max(min_dists_tgt_to_src))

    return hausdorff


def point_to_point_distance(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute point-to-point distances (for aligned clouds with correspondences).

    Args:
        source: (N, D) array of points
        target: (N, D) array of points

    Returns:
        (N,) array of distances

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = source + 0.1
        >>> distances = point_to_point_distance(source, target)
    """
    return np.linalg.norm(source - target, axis=1)


# TODO: Phase 3 - Add more metrics:
# - f1_score for classification
# - accuracy metrics
# - confusion_matrix utilities
# - iou (intersection over union) for 3D
# - precision_recall for detection
