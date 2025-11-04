# PyTorch Geometric Models

Graph neural network models for 3D point cloud processing using PyTorch Geometric.

## Purpose

Implements graph-based and voxel-based models:
- Dense voxel CNNs
- ResNet-style voxel models
- Point transformer architectures
- Custom voxel representations

## Files

### Models
- `model_dense_voxel.py` - Dense voxel CNN
- `model_res_voxel.py` - ResNet-style voxel model
- `model_point_transformer_voxel.py` - Transformer architecture
- `my_voxel.py` - Custom voxel implementation

### Training
- `main_voxel.py` - Training script

### Data
- `dataset_torchstudio_small_voxel.py` - Dataset loader

### Utilities
- `metrics.py` - Evaluation metrics
- `rotate.py` - Rotation utilities
- `scipy_tests.py` - Testing utilities

## Dependencies

```bash
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install scipy>=1.10.0
```

## Usage

```bash
python main_voxel.py --data_path ./data --epochs 100
```

## Models

### Dense Voxel CNN
3D convolutional network on voxel grids.

### ResNet Voxel
Residual connections for deeper voxel models.

### Point Transformer
Transformer architecture adapted for voxel/point data.

## Learning Objectives

- PyTorch Geometric library usage
- Graph neural networks
- Voxel representations
- Transformer architectures for 3D

For detailed model descriptions, see [docs/projects-overview.md](../../docs/projects-overview.md).
