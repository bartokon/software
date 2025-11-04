"""
Visualization utilities for point clouds.

This module provides common plotting functions for 2D and 3D point clouds.

Functions:
    plot_2d_points: Plot 2D point clouds
    plot_3d_points: Plot 3D point clouds
    plot_3d_arrows: Plot 3D arrows (normals, vectors)

TODO: In Phase 3, consolidate duplicate plotting code from:
    - python/icp_2d/*.py
    - python/icp_3d/draw_only.py
    - python/icp_3d/arrow_3d.py
    - python/pytorch/arrow_3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union


def plot_2d_points(
    *point_clouds,
    labels: Optional[list] = None,
    colors: Optional[list] = None,
    title: str = "2D Point Clouds",
    show: bool = True
):
    """
    Plot multiple 2D point clouds.

    Args:
        *point_clouds: Variable number of (N, 2) numpy arrays
        labels: List of labels for each point cloud
        colors: List of colors for each point cloud
        title: Plot title
        show: Whether to call plt.show()

    Example:
        >>> source = np.random.rand(100, 2)
        >>> target = np.random.rand(100, 2)
        >>> plot_2d_points(source, target, labels=['Source', 'Target'])
    """
    plt.figure(figsize=(8, 8))

    default_colors = ['blue', 'red', 'green', 'orange', 'purple']
    default_labels = [f'Cloud {i+1}' for i in range(len(point_clouds))]

    if labels is None:
        labels = default_labels
    if colors is None:
        colors = default_colors

    for i, points in enumerate(point_clouds):
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=colors[i % len(colors)],
            label=labels[i % len(labels)],
            alpha=0.6,
            s=20
        )

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if show:
        plt.show()


def plot_3d_points(
    *point_clouds,
    labels: Optional[list] = None,
    colors: Optional[list] = None,
    title: str = "3D Point Clouds",
    show: bool = True
):
    """
    Plot multiple 3D point clouds.

    Args:
        *point_clouds: Variable number of (N, 3) numpy arrays
        labels: List of labels for each point cloud
        colors: List of colors for each point cloud
        title: Plot title
        show: Whether to call plt.show()

    Example:
        >>> source = np.random.rand(100, 3)
        >>> target = np.random.rand(100, 3)
        >>> plot_3d_points(source, target, labels=['Source', 'Target'])
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    default_colors = ['blue', 'red', 'green', 'orange', 'purple']
    default_labels = [f'Cloud {i+1}' for i in range(len(point_clouds))]

    if labels is None:
        labels = default_labels
    if colors is None:
        colors = default_colors

    for i, points in enumerate(point_clouds):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[i % len(colors)],
            label=labels[i % len(labels)],
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()


def plot_3d_arrows(
    start_points: np.ndarray,
    vectors: np.ndarray,
    scale: float = 1.0,
    color: str = 'red',
    title: str = "3D Arrows",
    show: bool = True
):
    """
    Plot 3D arrows (useful for normals, motion vectors).

    Args:
        start_points: (N, 3) array of arrow start positions
        vectors: (N, 3) array of arrow directions
        scale: Scale factor for arrow length
        color: Arrow color
        title: Plot title
        show: Whether to call plt.show()

    Example:
        >>> points = np.random.rand(50, 3)
        >>> normals = np.random.rand(50, 3)
        >>> normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        >>> plot_3d_arrows(points, normals, scale=0.1)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        start_points[:, 2],
        c='blue',
        alpha=0.6,
        s=20
    )

    # Plot arrows
    for i in range(len(start_points)):
        ax.quiver(
            start_points[i, 0],
            start_points[i, 1],
            start_points[i, 2],
            vectors[i, 0] * scale,
            vectors[i, 1] * scale,
            vectors[i, 2] * scale,
            color=color,
            alpha=0.7,
            arrow_length_ratio=0.3
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if show:
        plt.show()


# TODO: Phase 3 - Add more functions:
# - plot_with_correspondences (show matching lines)
# - plot_convergence (ICP iteration plots)
# - plot_rotation_angles (angle visualization)
# - save_figure (utility to save plots)
