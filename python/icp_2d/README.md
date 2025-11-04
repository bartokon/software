# 2D Iterative Closest Point (ICP)

Multiple implementations of the ICP algorithm for 2D point cloud registration.

## Purpose

Educational implementations demonstrating:
- Classic ICP with SVD
- PyTorch-based differentiable ICP
- Weighted ICP for robust matching
- Visualization and analysis

## Files

- `point_to_point_least_squares_2d.py` - SVD-based ICP
- `point_to_point_pytorch_2d.py` - PyTorch ICP
- `point_to_point_pytorch_single_weight_2d.py` - Weighted ICP
- `point_to_point_utils_2d.py` - Utility functions
- `experimental.py` - Experimental features

## Algorithm

ICP iteratively:
1. Find nearest neighbors (correspondences)
2. Compute optimal transformation (rotation + translation)
3. Apply transformation
4. Repeat until convergence

## Usage

### Basic Example

```bash
python point_to_point_least_squares_2d.py
```

### Python API

```python
from python.icp_2d import point_to_point_least_squares_2d as icp
import numpy as np

# Create point clouds
source = np.random.rand(100, 2)
target = source @ rotation_matrix(0.1) + [1, 2]  # Transform

# Run ICP
aligned, R, t = icp.align(source, target, max_iterations=50)

# Check convergence
error = np.mean(np.linalg.norm(aligned - target, axis=1))
print(f"Final error: {error:.6f}")
```

## Implementations

### 1. Least Squares ICP (SVD)
Classic approach using Singular Value Decomposition.

**Pros:**
- Closed-form solution
- Fast and stable
- No learning required

**Cons:**
- Sensitive to outliers
- Requires good initialization

**Complexity:** O(n) per iteration

### 2. PyTorch ICP
Differentiable implementation using PyTorch.

**Pros:**
- Gradient-based optimization
- Can integrate with deep learning
- GPU acceleration

**Cons:**
- Slower than SVD
- May need hyperparameter tuning

### 3. Weighted ICP
Per-point weights for robust matching.

**Pros:**
- Handles outliers better
- Adaptive weighting
- More robust convergence

**Cons:**
- Requires weight estimation
- Slightly slower

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
import torch  # For PyTorch variants
```

## Visualization

All scripts include visualization:
- Source point cloud (blue)
- Target point cloud (red)
- Aligned result (green)
- Convergence plot

## Parameters

Common parameters:
- `max_iterations`: Max ICP iterations (default: 50)
- `tolerance`: Convergence threshold (default: 1e-6)
- `visualize`: Show plots (default: True)

## Performance

Typical convergence:
- **Iterations**: 10-50
- **Time**: < 1 second for 1000 points
- **Accuracy**: Sub-pixel alignment

## Example Output

```
Iteration 0: error = 2.453
Iteration 1: error = 1.234
Iteration 2: error = 0.567
...
Iteration 15: error = 0.001
Converged!

Final transformation:
Rotation: 0.1 radians
Translation: [1.0, 2.0]
```

## Troubleshooting

**Not converging:**
- Check initialization (should be reasonably close)
- Increase max_iterations
- Verify point clouds have overlap

**Poor alignment:**
- May have local minimum
- Try different initialization
- Use weighted ICP for robustness

## Learning Objectives

- Understanding ICP algorithm
- Point cloud registration
- SVD for optimal transformation
- PyTorch for differentiable algorithms
- Visualization techniques

## References

- [ICP Algorithm](https://en.wikipedia.org/wiki/Iterative_closest_point)
- [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- Original paper: Besl & McKay, 1992
