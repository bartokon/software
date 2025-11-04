# Common Utilities

Shared utilities to avoid code duplication across projects.

## Modules

- **geometry.py** - 3D geometry operations (rotations, transformations)
- **visualization.py** - Plotting functions for 2D/3D point clouds
- **metrics.py** - Evaluation metrics (MSE, RMSE, Chamfer distance)

## Usage

```python
from python.common.geometry import rotation_matrix_from_euler, transform_points
from python.common.visualization import plot_3d_points
from python.common.metrics import chamfer_distance
import numpy as np

# Create rotation matrix
R = rotation_matrix_from_euler(roll=0.1, pitch=0.2, yaw=0.3)

# Transform points
points = np.random.rand(100, 3)
transformed = transform_points(points, R, translation=[1, 2, 3])

# Visualize
plot_3d_points(points, transformed, labels=['Original', 'Transformed'])

# Compute error
error = chamfer_distance(points, transformed)
print(f"Chamfer distance: {error:.6f}")
```

## Status

**Phase 2 Complete**: Basic utilities created with placeholder implementations.

**Phase 3 TODO**: Extract duplicate functions from existing projects:
- Rotation functions from icp_3d projects
- Visualization code from icp_2d and icp_3d
- Metrics from pytorch projects

## Implementation Plan

See [REORGANIZATION_PLAN_REVISED.md](../../REORGANIZATION_PLAN_REVISED.md) Phase 3 for details.
