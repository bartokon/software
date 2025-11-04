# Python Projects

This directory contains Python projects for point cloud processing, ICP algorithms, and deep learning.

## Projects

| Project | Purpose | ML/DL | Visualization | LOC |
|---------|---------|-------|---------------|-----|
| [common](common/) | Shared utilities (geometry, visualization, metrics) | No | Yes | TBD |
| [icp_2d](icp_2d/) | 2D Iterative Closest Point implementations | No | Yes | ~800 |
| [icp_3d](icp_3d/) | 3D point cloud alignment algorithms | No | Yes | ~2,500 |
| [pytorch](pytorch/) | PyTorch models and datasets | Yes | Yes | ~3,000 |
| [pytorch_geometric](pytorch_geometric/) | Graph neural networks for point clouds | Yes | Yes | ~1,500 |
| [pytorch_voxel](pytorch_voxel/) | Voxel-based CNN models | Yes | Yes | ~30,000+ |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NumPy, Matplotlib, SciPy

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Package Structure

After setup, you can import from any project:
```python
from python.icp_2d import point_to_point_least_squares_2d
from python.common.geometry import rotation_matrix_from_euler
from python.pytorch import pointnet_model
```

## Getting Started

1. **Install package**:
   ```bash
   pip install -e .
   ```

2. **Run examples**:
   ```bash
   cd icp_2d
   python point_to_point_least_squares_2d.py
   ```

See individual project READMEs for detailed usage.

## Common Utilities

The `common/` directory provides shared functions:
- **geometry**: Rotation matrices, transformations
- **visualization**: 2D/3D plotting utilities
- **metrics**: Evaluation functions

This eliminates code duplication across projects.

For more details, see [docs/projects-overview.md](../docs/projects-overview.md).
