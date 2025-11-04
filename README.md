# Software - Research & Learning Repository

A collection of experimental implementations for 3D point cloud processing, GPU computing, and deep learning algorithms. This repository contains both C++ and Python projects focused on computer graphics, parallel computing, and machine learning research.

## Overview

This repository serves as a research and learning platform for exploring:

- **Point Cloud Processing**: ICP (Iterative Closest Point) algorithms in 2D and 3D
- **GPU Computing**: CUDA, HIP (AMD ROCm), and SYCL implementations
- **Deep Learning**: PyTorch-based models for point clouds and voxel representations
- **Algorithm Implementations**: Custom data structures and sorting algorithms

## Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **C++ Compiler**: GCC 11+ with C++20 support
- **GPU Computing** (optional):
  - CUDA 11.8+ (for NVIDIA GPUs)
  - ROCm 6.0.0+ (for AMD GPUs)
  - Intel OneAPI 2024.0+ (for Intel GPUs/CPUs)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bartokon/software.git
   cd software
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Install package in editable mode
   pip install -e .
   ```

3. **Verify installation**
   ```bash
   # Test Python imports
   python -c "from python.icp_2d import point_to_point_utils_2d; print('Success!')"
   ```

### Running Examples

**ICP 2D Example:**
```bash
cd python/icp_2d
python point_to_point_least_squares_2d.py
```

**ICP 3D Example:**
```bash
cd python/icp_3d
python point_to_point_least_squares_3d.py
```

**C++ Point Cloud Matching:**
```bash
cd cpp/point_cloud_matching
make
./main.elf
```

**CUDA Vector Addition:**
```bash
cd cpp/cuda
make
./vadd.elf
```

## Project Structure

```
software/
├── cpp/                    # C++ Projects
│   ├── bubble_pointer_sort/    # Sorting algorithm implementation
│   ├── cuda/                   # CUDA/HIP/SYCL vector addition example
│   ├── cuda_sum/               # CUDA reduction/sum kernel
│   ├── point_cloud_matching/   # Point cloud alignment algorithms
│   └── string/                 # Custom string class implementation
│
├── python/                 # Python Projects
│   ├── common/                 # Shared utilities (geometry, visualization, metrics)
│   ├── icp_2d/                 # 2D Iterative Closest Point implementations
│   ├── icp_3d/                 # 3D point cloud alignment algorithms
│   ├── pytorch/                # PyTorch models and datasets
│   ├── pytorch_geometric/      # Graph neural networks for point clouds
│   └── pytorch_voxel/          # Voxel-based CNN models
│
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── data/                   # Data directory (gitignored)
```

## Technologies

### C++
- **Standard**: C++20
- **Compilers**: GCC 11+, NVCC (CUDA 11.8), hipcc (ROCm 6.0.0), icpx (Intel OneAPI)
- **GPU**: CUDA, HIP, SYCL
- **Build System**: Make

### Python
- **Version**: 3.8+
- **Core Libraries**:
  - PyTorch 2.0+
  - PyTorch Geometric 2.3+
  - NumPy 1.24+
  - Matplotlib 3.7+
  - SciPy 1.10+

## Key Features

### Point Cloud Processing
- **ICP Algorithms**: Point-to-point and point-to-plane variants
- **Implementations**: Least squares, PyTorch-based, weighted, batched
- **Visualization**: 2D and 3D plotting utilities

### GPU Computing
- **Multi-Platform**: CUDA (NVIDIA), HIP (AMD), SYCL (Intel)
- **Examples**: Vector operations, reduction algorithms
- **Translation**: Includes CUDA→HIP→SYCL conversion examples

### Deep Learning
- **Architectures**: PointNet, custom CNNs, graph neural networks
- **Representations**: Point clouds, voxels, graphs
- **Training**: Full training pipelines with visualization

## Documentation

- [Getting Started Guide](docs/getting-started.md) - Detailed setup instructions
- [Projects Overview](docs/projects-overview.md) - Description of all projects
- [GPU Setup](docs/gpu-setup.md) - CUDA/HIP/SYCL configuration

Each project directory contains its own README with specific usage instructions.

## Usage Examples

### Python: Import and Use ICP

```python
from python.icp_2d import point_to_point_least_squares_2d as icp_2d
from python.common.geometry import rotation_matrix_from_euler
from python.common.visualization import plot_2d_points
import numpy as np

# Create sample point clouds
source = np.random.rand(100, 2)
target = np.random.rand(100, 2)

# Run ICP alignment
aligned, transformation = icp_2d.align(source, target)

# Visualize results
plot_2d_points(source, target, aligned)
```

### Python: Using Shared Utilities

```python
from python.common.geometry import rotation_matrix_from_euler, transform_points
from python.common.visualization import plot_3d_points
from python.common.metrics import mean_squared_error
import numpy as np

# Create rotation
R = rotation_matrix_from_euler(roll=0.1, pitch=0.2, yaw=0.3)

# Transform points
points = np.random.rand(100, 3)
transformed = transform_points(points, R, translation=[1, 2, 3])

# Visualize
plot_3d_points(points, title="Original")
plot_3d_points(transformed, title="Transformed")

# Compute error
error = mean_squared_error(points, transformed)
```

### C++: Point Cloud Matching

```cpp
#include "Point_Cloud.hpp"
#include "Point_Matcher.hpp"
#include "BFTree.hpp"

int main() {
    // Create point clouds
    Point_Cloud<double> source, target;

    // Load or generate points
    // ...

    // Create matcher
    Point_Matcher matcher;

    // Find correspondences
    auto matches = matcher.match(source, target);

    return 0;
}
```

## Building C++ Projects

Each C++ project has its own Makefile:

```bash
cd cpp/[project-name]
make              # Build the project
make clean        # Clean build artifacts
./[executable]    # Run the program
```

For GPU projects with multiple targets:
```bash
cd cpp/cuda
make cuda         # Build CUDA version
make hip          # Build HIP version (AMD)
make dpc          # Build SYCL version (Intel)
```

## Development

### Code Organization

- **C++**: Header-only libraries for data structures, compiled executables for examples
- **Python**: Package-based structure with `__init__.py` for clean imports
- **Shared Code**: Common utilities in `python/common/` to avoid duplication

### Adding New Projects

1. Create project directory under `cpp/` or `python/`
2. Add README.md explaining the project
3. For Python: Create `__init__.py` and export main functions
4. Update this README with project description

## Repository Philosophy

This is a **research and learning repository**. Code prioritizes:

- **Experimentation**: Flexibility to try new approaches
- **Learning**: Clear implementations over optimization
- **Reusability**: Shared utilities to avoid duplication
- **Documentation**: Understanding what and why

Large training scripts and notebooks are intentional - they keep experiment context together.

## GPU Setup Notes

For GPU computing projects, you'll need appropriate drivers and toolkits:

- **NVIDIA**: Install CUDA Toolkit 11.8+
- **AMD**: Install ROCm 6.0.0+
- **Intel**: Install Intel OneAPI Toolkit 2024.0+

See [docs/gpu-setup.md](docs/gpu-setup.md) for detailed instructions.

## Contributing

This is primarily a personal research repository. If you find it useful:

1. Feel free to fork and adapt for your needs
2. Issues and suggestions are welcome
3. Pull requests should focus on bug fixes or documentation improvements

## License

GNU General Public License v3.0 (GPL-3.0)

See [LICENSE](LICENSE) file for details.

## Author

**bartokon**

## Acknowledgments

This repository contains implementations inspired by various research papers and tutorials in computer vision, graphics, and machine learning.

---

## Project Status

🔬 **Active Research** - This repository is actively being developed and experimented with. Expect frequent updates and changes.

## Additional Resources

- **ICP Algorithms**: Classic point cloud registration techniques
- **GPU Computing**: Examples of heterogeneous computing across NVIDIA, AMD, and Intel platforms
- **Deep Learning**: Modern architectures for 3D data processing

For questions or discussions, please open an issue on GitHub.
