"""
2D Iterative Closest Point (ICP) implementations.

This package provides multiple ICP algorithm implementations for 2D point cloud registration.

Available modules:
    - point_to_point_least_squares_2d: SVD-based ICP
    - point_to_point_pytorch_2d: PyTorch-based ICP
    - point_to_point_pytorch_single_weight_2d: Weighted ICP
    - point_to_point_utils_2d: Utility functions

Example:
    from python.icp_2d import point_to_point_utils_2d as utils
    import numpy as np

    source = np.random.rand(100, 2)
    target = np.random.rand(100, 2)
    # Run ICP alignment
"""

__version__ = "0.1.0"
