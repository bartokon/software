"""
Common utilities shared across projects.

This package provides shared functionality to avoid code duplication:
- geometry: Rotation matrices, transformations, point operations
- visualization: 2D/3D plotting utilities
- metrics: Evaluation metrics for point clouds

Modules:
    - geometry: 3D geometry operations
    - visualization: Plotting functions
    - metrics: Evaluation metrics

Example:
    from python.common.geometry import rotation_matrix_from_euler
    from python.common.visualization import plot_3d_points
    from python.common.metrics import mean_squared_error

    import numpy as np

    # Create rotation
    R = rotation_matrix_from_euler(0.1, 0.2, 0.3)

    # Transform points
    points = np.random.rand(100, 3)
    transformed = (R @ points.T).T

    # Visualize
    plot_3d_points(points, title="Original")
    plot_3d_points(transformed, title="Transformed")

    # Compute error
    error = mean_squared_error(points, transformed)
"""

__version__ = "0.1.0"

# Import key functions for easier access
try:
    from .geometry import rotation_matrix_from_euler, transform_points
    from .visualization import plot_2d_points, plot_3d_points
    from .metrics import mean_squared_error, chamfer_distance

    __all__ = [
        'rotation_matrix_from_euler',
        'transform_points',
        'plot_2d_points',
        'plot_3d_points',
        'mean_squared_error',
        'chamfer_distance',
    ]
except ImportError:
    # Modules not yet created, imports will work after Phase 3
    pass
