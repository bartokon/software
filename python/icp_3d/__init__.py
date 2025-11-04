"""
3D Iterative Closest Point (ICP) implementations.

This package provides comprehensive ICP implementations for 3D point cloud alignment,
including both point-to-point and point-to-plane variants.

Available modules:
    Point-to-Point:
        - point_to_point_least_squares_3d: SVD-based
        - point_to_point_pytorch_3d: PyTorch optimization
        - point_to_point_pytorch_batch_3d: Batched processing

    Point-to-Plane:
        - point_to_plane_least_squares_3d: Linear system
        - point_to_plane_pytorch_3d: PyTorch optimization
        - point_to_plane_pytorch_batch_3d: Batched processing

    Utilities:
        - point_to_point_utils_3d: Transformations, rotations
        - point_to_plane_utils_3d: Normal estimation
        - arrow_3d: 3D visualization
        - draw_angles: Angle visualization
        - draw_only: Plotting utilities

Example:
    from python.icp_3d import point_to_point_utils_3d as utils
    import numpy as np

    source = np.random.rand(1000, 3)
    target = np.random.rand(1000, 3)
    # Run 3D ICP alignment
"""

__version__ = "0.1.0"
