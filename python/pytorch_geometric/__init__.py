"""
Graph neural network models using PyTorch Geometric.

This package contains models for processing point clouds as graphs,
including voxel representations and transformers.

Available modules:
    Models:
        - model_dense_voxel: Dense voxel CNN
        - model_res_voxel: ResNet-style voxel model
        - model_point_transformer_voxel: Transformer architecture
        - my_voxel: Custom voxel implementation

    Training:
        - main_voxel: Training script

    Data:
        - dataset_torchstudio_small_voxel: Dataset loader

    Utilities:
        - metrics: Evaluation metrics
        - rotate: Rotation utilities
        - scipy_tests: Testing utilities

Example:
    from python.pytorch_geometric import model_dense_voxel
    import torch

    model = model_dense_voxel.DenseVoxelCNN()
    voxels = torch.rand(32, 1, 32, 32, 32)
    output = model(voxels)
"""

__version__ = "0.1.0"
