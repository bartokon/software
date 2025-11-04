# 3D Iterative Closest Point (ICP)

Comprehensive 3D point cloud alignment implementations including point-to-point and point-to-plane variants.

## Purpose

Research and educational implementations of:
- Point-to-point ICP (minimize point distances)
- Point-to-plane ICP (minimize point-to-plane distances)
- Multiple optimization methods
- Batch processing
- Visualization and analysis tools

## Files

### Point-to-Point Implementations
- `point_to_point_least_squares_3d.py` - SVD-based
- `point_to_point_pytorch_3d.py` - PyTorch optimization
- `point_to_point_pytorch_batch_3d.py` - Batched processing
- `point_to_point_pytorch_single_weight_3d.py` - Weighted

### Point-to-Plane Implementations
- `point_to_plane_least_squares_3d.py` - Linear system
- `point_to_plane_pytorch_3d.py` - PyTorch optimization
- `point_to_plane_pytorch_batch_3d.py` - Batched processing

### Utilities
- `point_to_point_utils_3d.py` - Transformations, rotations
- `point_to_plane_utils_3d.py` - Normal estimation
- `arrow_3d.py` - 3D arrow visualization
- `draw_angles.py` - Angle visualization
- `draw_only.py` - Plotting utilities

## Algorithms

### Point-to-Point ICP
Minimizes sum of point-to-point distances:
```
E = Σ ||Rp_i + t - q_i||²
```

**Best for:**
- Dense point clouds
- Similar point densities
- Good initial alignment

### Point-to-Plane ICP
Minimizes sum of point-to-plane distances:
```
E = Σ (n_i · (Rp_i + t - q_i))²
```

Where n_i is the normal at target point q_i.

**Advantages:**
- Faster convergence (typically 2-5x)
- Better with varying densities
- More robust to sampling differences

## Usage

### Basic Example

```bash
python point_to_point_least_squares_3d.py
```

### Python API

```python
from python.icp_3d import point_to_point_utils_3d as utils
import numpy as np

# Create 3D point clouds
source = np.random.rand(1000, 3)

# Apply known transformation
R = utils.rotation_matrix_from_euler(0.1, 0.2, 0.3)
t = np.array([1.0, 2.0, 3.0])
target = (R @ source.T).T + t

# Run ICP
aligned, R_est, t_est = utils.icp_point_to_point(
    source, target,
    max_iterations=50
)

# Verify result
rotation_error = np.linalg.norm(R - R_est)
translation_error = np.linalg.norm(t - t_est)
print(f"Rotation error: {rotation_error:.6f}")
print(f"Translation error: {translation_error:.6f}")
```

### Point-to-Plane Example

```python
from python.icp_3d import point_to_plane_utils_3d as plane_utils

# Estimate normals for target
normals = plane_utils.estimate_normals(target, k_neighbors=10)

# Run point-to-plane ICP
aligned, R, t = plane_utils.icp_point_to_plane(
    source, target, normals,
    max_iterations=30  # Converges faster!
)
```

### Batch Processing

```python
from python.icp_3d import point_to_point_pytorch_batch_3d as batch_icp

# Multiple point clouds
sources = [cloud1, cloud2, cloud3]  # List of numpy arrays
targets = [target1, target2, target3]

# Process in parallel
results = batch_icp.batch_align(sources, targets)

for i, (aligned, R, t) in enumerate(results):
    print(f"Cloud {i}: aligned successfully")
```

## 3D Transformations

### Euler Angles
Represent rotations as (roll, pitch, yaw):

```python
from python.icp_3d import point_to_point_utils_3d as utils

# Create rotation matrix
R = utils.rotation_matrix_from_euler(
    roll=0.1,   # Rotation around X
    pitch=0.2,  # Rotation around Y
    yaw=0.3     # Rotation around Z
)

# Apply to points
transformed = (R @ points.T).T + translation
```

### Rotation Representations
- Euler angles (used in most scripts)
- Rotation matrices (for computation)
- Quaternions (planned)
- Axis-angle (planned)

## Visualization

### 3D Point Cloud Plot
```python
from python.icp_3d import draw_only

draw_only.plot_point_clouds(
    source, target, aligned,
    title="ICP Alignment"
)
```

### Normal Visualization
```python
from python.icp_3d import arrow_3d

arrow_3d.plot_with_normals(
    points, normals,
    scale=0.1
)
```

### Angle Visualization
```python
from python.icp_3d import draw_angles

draw_angles.visualize_rotation(
    R_true, R_estimated
)
```

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch  # For PyTorch variants
import scipy  # For normal estimation
```

## Parameters

### Point-to-Point
- `max_iterations`: Maximum iterations (default: 50)
- `tolerance`: Convergence threshold (default: 1e-6)
- `visualize`: Show 3D plots (default: True)

### Point-to-Plane
- `max_iterations`: Maximum iterations (default: 30)
- `k_neighbors`: Neighbors for normal estimation (default: 10)
- `normal_method`: 'pca' or 'cross_product'

## Performance Comparison

| Method | Iterations | Time (1K points) | Robustness |
|--------|-----------|------------------|------------|
| Point-to-Point | 30-100 | ~0.5s | Medium |
| Point-to-Plane | 10-30 | ~0.3s | High |
| Batch | Varies | ~0.2s per cloud | Medium |
| Weighted | 20-50 | ~0.7s | High |

Point-to-plane converges 2-3x faster!

## Convergence Criteria

ICP stops when:
1. Max iterations reached, OR
2. Error change < tolerance, OR
3. Transformation change < threshold

## Troubleshooting

**Slow convergence:**
- Use point-to-plane for faster convergence
- Check initialization quality
- Increase tolerance for early stopping

**Poor alignment:**
- Verify sufficient overlap (>30%)
- Check for symmetry issues
- Try different initializations

**Memory issues with batch:**
- Reduce batch size
- Process sequentially instead
- Downsample point clouds

## Learning Objectives

- 3D transformations and rotations
- Euler angles and rotation matrices
- Normal estimation techniques
- Point-to-point vs point-to-plane trade-offs
- PyTorch for gradient-based optimization
- 3D visualization with Matplotlib

## Advanced Topics

### Normal Estimation
```python
# Using PCA on local neighborhood
def estimate_normals(points, k=10):
    normals = []
    for point in points:
        neighbors = find_k_nearest(points, point, k)
        cov = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        normal = eigenvectors[:, eigenvalues.argmin()]
        normals.append(normal)
    return np.array(normals)
```

### Covariance-Based Convergence
More robust convergence criteria using covariance of correspondences.

### Robust Cost Functions
- Huber loss for outliers
- Truncated distances
- M-estimators

## References

- [ICP Algorithm](https://en.wikipedia.org/wiki/Iterative_closest_point)
- [Point-to-Plane ICP](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf)
- Original ICP: Besl & McKay, 1992
- Point-to-Plane: Chen & Medioni, 1992
- [PCL Documentation](https://pointclouds.org/documentation/classpcl_1_1_iterative_closest_point.html)
