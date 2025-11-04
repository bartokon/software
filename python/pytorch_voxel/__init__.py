"""
Voxel-based deep learning for 3D point clouds.

This package contains large-scale experimental pipelines for voxel-based
and point-based neural network training.

Available modules:
    Training Scripts (Large Files):
        - main_voxel (6,491 lines): Voxel model training pipeline
        - main_points (6,544 lines): Point-based model training

    Datasets (Large Files):
        - dataset_voxel (9,293 lines): Voxel dataset loader
        - dataset_points (7,937 lines): Point dataset loader

    Utilities:
        - metrics: Evaluation metrics
        - utils_voxel: Voxel utilities
        - scipy_tests: Testing utilities

Note:
    Files are intentionally large to keep experiment context together.
    See table of contents at top of each file for navigation.

Example:
    # See individual file documentation
    # These are complete training pipelines, not libraries
    python main_voxel.py --data_path ./data --epochs 100
"""

__version__ = "0.1.0"
