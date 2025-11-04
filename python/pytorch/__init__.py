"""
PyTorch models and training scripts for point cloud processing.

This package contains:
- PointNet and custom architectures
- Dataset loaders
- Training scripts
- Utilities for point cloud manipulation

Available modules:
    Models:
        - pointnet_model: PointNet architecture
        - mymodel_0, mymodel_1, mymodel_2: Custom models
        - model, model_pn: Base models
        - histogram_model: Histogram-based features

    Datasets:
        - dataset: Main dataset class
        - dataset_pytorch: PyTorch dataset wrapper
        - histogram_dataset: Histogram features

    Training:
        - main: Basic training
        - main_gcn: GCN training
        - main_gcn_ConvNet: ConvNet training
        - main_gcn_pointnet: PointNet training

    Utilities:
        - rotate: Point cloud rotation
        - arrow_3d: Visualization
        - mesh_io: Mesh I/O
        - point_information: Point cloud statistics

Example:
    from python.pytorch import pointnet_model
    import torch

    model = pointnet_model.PointNet(num_classes=10)
    points = torch.rand(32, 1024, 3)
    output = model(points)
"""

__version__ = "0.1.0"
