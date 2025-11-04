# Projects Overview

This document provides detailed descriptions of all projects in the repository, organized by language and category.

## Table of Contents

- [C++ Projects](#c-projects)
  - [Algorithms](#algorithms)
  - [GPU Computing](#gpu-computing)
  - [Data Structures](#data-structures)
- [Python Projects](#python-projects)
  - [Point Cloud Processing](#point-cloud-processing)
  - [Deep Learning](#deep-learning)
- [Project Dependencies](#project-dependencies)
- [Technology Matrix](#technology-matrix)

---

## C++ Projects

### Algorithms

#### bubble_pointer_sort/
**Purpose**: Implementation of bubble sort using pointers

- **Location**: `cpp/bubble_pointer_sort/`
- **Language**: C
- **Lines of Code**: ~100
- **Build**: `build.sh` script
- **Complexity**: O(n²)

**Description**:
A classic sorting algorithm implementation demonstrating pointer manipulation in C. Uses pointer arithmetic to swap elements in an array.

**Key Concepts**:
- Pointer-based array manipulation
- Bubble sort algorithm
- Memory management

**Usage**:
```bash
cd cpp/bubble_pointer_sort
./build.sh
./main
```

**Learning Objectives**:
- Understanding pointer arithmetic
- Basic sorting algorithms
- C programming fundamentals

---

#### point_cloud_matching/
**Purpose**: Point cloud alignment using spatial data structures

- **Location**: `cpp/point_cloud_matching/`
- **Language**: C++20
- **Lines of Code**: ~400
- **Build**: Makefile
- **Key Files**:
  - `BFTree.hpp` - Brute force / KD-tree implementation
  - `Point_3D.hpp` - 3D point class
  - `Point_Cloud.hpp` - Point cloud container
  - `Point_Matcher.hpp` - Matching algorithm

**Description**:
A comprehensive point cloud processing library implementing spatial data structures and matching algorithms. Uses template-based design for flexibility.

**Features**:
- 3D point representation
- Point cloud container with template support
- Brute force nearest neighbor search
- KD-tree spatial data structure
- Point matching algorithms

**Key Concepts**:
- Template metaprogramming
- Spatial data structures
- Nearest neighbor search
- C++20 features

**Usage**:
```bash
cd cpp/point_cloud_matching
make
./main.elf
```

**API Example**:
```cpp
#include "Point_Cloud.hpp"
#include "Point_Matcher.hpp"

Point_Cloud<double> source, target;
// Load points...
Point_Matcher matcher;
auto matches = matcher.match(source, target);
```

**Learning Objectives**:
- Advanced C++ templates
- Spatial data structures
- Point cloud processing
- Header-only library design

---

### GPU Computing

#### cuda/
**Purpose**: Heterogeneous computing demonstration (CUDA/HIP/SYCL)

- **Location**: `cpp/cuda/`
- **Language**: CUDA C++
- **Lines of Code**: ~150
- **Build**: Makefile with multiple targets
- **Supported Platforms**:
  - CUDA (NVIDIA)
  - HIP (AMD ROCm)
  - SYCL (Intel OneAPI)

**Description**:
A vector addition kernel demonstrating portable GPU programming. The same algorithm is implemented across three GPU programming models, showing translation patterns.

**Key Files**:
- `src/vadd.cu` - CUDA vector addition kernel
- `src/vadd.hpp` - Header file
- `src/main_vadd.cpp` - Host code
- `Makefile` - Multi-target build system

**Features**:
- Cross-platform GPU code
- CUDA → HIP translation via hipify-clang
- CUDA → SYCL translation via dpct
- Performance comparison across platforms

**Build Targets**:
```bash
make cuda    # NVIDIA GPUs (NVCC)
make hip     # AMD GPUs (hipcc)
make dpc     # Intel GPUs (icpx)
```

**Learning Objectives**:
- GPU parallel programming basics
- Heterogeneous computing
- Cross-platform GPU development
- Translation tools usage

**Performance Notes**:
- Vector addition is memory-bound
- Good for learning GPU basics
- Not computationally intensive

---

#### cuda_sum/
**Purpose**: CUDA reduction algorithm (parallel summation)

- **Location**: `cpp/cuda_sum/`
- **Language**: CUDA C++
- **Lines of Code**: ~200
- **Build**: Makefile

**Description**:
Implementation of parallel reduction algorithm for computing array sum on GPU. Demonstrates optimization techniques like shared memory and reduction patterns.

**Key Files**:
- `src/sum.cu` - Reduction kernel
- `src/sum.hpp` - Header
- `src/main_sum.cpp` - Test harness

**Features**:
- Parallel reduction algorithm
- Shared memory optimization
- Thread synchronization
- Performance benchmarking

**Algorithm**:
```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
Step 1: [3, 7, 11, 15]  (pairwise sum)
Step 2: [10, 26]
Step 3: [36]  (result)
```

**Usage**:
```bash
cd cpp/cuda_sum
make
./cuda/main_sum.elf
```

**Learning Objectives**:
- GPU reduction patterns
- Shared memory usage
- Thread synchronization
- GPU optimization techniques

**Complexity**: O(log n) parallel time, O(n) work

---

### Data Structures

#### string/
**Purpose**: Custom string class implementation (vs std::string)

- **Location**: `cpp/string/`
- **Language**: C++20
- **Lines of Code**: ~4,200 (string.hpp)
- **Build**: Makefile

**Description**:
A comprehensive custom string class implementing most std::string functionality from scratch. Educational project demonstrating C++ concepts.

**Key Files**:
- `src/string.hpp` - Custom string class (4,209 lines!)
- `src/main.cpp` - Comparison with std::string

**Features Implemented**:
- Dynamic memory management
- Copy/move semantics
- String operations (concatenation, substring, search)
- Iterators
- Comparison operators
- Memory optimization

**Why So Large?**:
This is a learning project that implements many string operations to understand:
- Memory management
- Rule of five (copy/move constructors and assignments, destructor)
- Operator overloading
- Template functions
- STL-like interface design

**Usage**:
```bash
cd cpp/string
make
./string_test.elf
```

**Comparison with std::string**:
```cpp
// Custom string
MyString s1 = "Hello";
MyString s2 = "World";
MyString s3 = s1 + " " + s2;

// Should behave like std::string
std::string s4 = "Hello";
std::string s5 = "World";
std::string s6 = s4 + " " + s5;
```

**Learning Objectives**:
- Deep understanding of std::string internals
- Advanced C++ memory management
- Operator overloading patterns
- STL design principles

---

## Python Projects

### Point Cloud Processing

#### icp_2d/
**Purpose**: 2D Iterative Closest Point algorithm implementations

- **Location**: `python/icp_2d/`
- **Language**: Python 3.8+
- **Dependencies**: NumPy, PyTorch, Matplotlib
- **Lines of Code**: ~800

**Description**:
Multiple implementations of the ICP algorithm for 2D point cloud registration. Compares different approaches: analytical, iterative, and deep learning-based.

**Key Files**:
- `point_to_point_least_squares_2d.py` - Classic ICP with SVD
- `point_to_point_pytorch_2d.py` - PyTorch-based ICP
- `point_to_point_pytorch_single_weight_2d.py` - Weighted ICP
- `point_to_point_utils_2d.py` - Utility functions
- `experimental.py` - Experimental features

**Algorithms**:
1. **Least Squares ICP**: Uses SVD for optimal transformation
2. **PyTorch ICP**: Differentiable implementation
3. **Weighted ICP**: Per-point weights for robust matching

**Usage**:
```bash
cd python/icp_2d
python point_to_point_least_squares_2d.py
```

**Key Concepts**:
- Point cloud registration
- Singular Value Decomposition (SVD)
- Iterative optimization
- Visualization

**Performance**: Converges in 10-50 iterations typically

---

#### icp_3d/
**Purpose**: 3D point cloud alignment algorithms

- **Location**: `python/icp_3d/`
- **Language**: Python 3.8+
- **Dependencies**: NumPy, PyTorch, Matplotlib
- **Lines of Code**: ~2,500
- **Files**: 14 Python scripts

**Description**:
Comprehensive 3D ICP implementations including both point-to-point and point-to-plane variants. Includes visualization and analysis tools.

**Key Files**:

**Point-to-Point:**
- `point_to_point_least_squares_3d.py`
- `point_to_point_pytorch_3d.py`
- `point_to_point_pytorch_batch_3d.py`

**Point-to-Plane:**
- `point_to_plane_least_squares_3d.py`
- `point_to_plane_pytorch_3d.py`
- `point_to_plane_pytorch_batch_3d.py`

**Utilities:**
- `point_to_point_utils_3d.py` - Transformations, rotations
- `point_to_plane_utils_3d.py` - Normal estimation
- `arrow_3d.py` - 3D arrow visualization
- `draw_angles.py` - Angle visualization
- `draw_only.py` - Plotting utilities

**Algorithms**:
1. **Point-to-Point ICP**: Minimizes point-to-point distances
2. **Point-to-Plane ICP**: Minimizes point-to-plane distances (faster convergence)
3. **Batched Versions**: Process multiple point clouds simultaneously

**Key Features**:
- Euler angle rotation representations
- Normal estimation for point-to-plane
- Batch processing support
- Comprehensive visualization

**Usage**:
```bash
cd python/icp_3d
python point_to_point_least_squares_3d.py
python point_to_plane_pytorch_3d.py
```

**Learning Objectives**:
- 3D transformations and rotations
- Normal estimation
- ICP convergence behavior
- PyTorch autodifferentiation

---

### Deep Learning

#### pytorch/
**Purpose**: General PyTorch models and datasets for point clouds

- **Location**: `python/pytorch/`
- **Language**: Python 3.8+
- **Dependencies**: PyTorch, PyTorch Geometric, NumPy, Matplotlib
- **Lines of Code**: ~3,000
- **Files**: 23 Python files

**Description**:
A collection of deep learning models and training scripts for point cloud processing. Includes PointNet, custom architectures, and graph neural networks.

**Model Files**:
- `pointnet_model.py` - PointNet architecture for point clouds
- `mymodel_0.py`, `mymodel_1.py`, `mymodel_2.py` - Custom architectures
- `model.py`, `model_pn.py` - Base models
- `histogram_model.py` - Histogram-based features

**Dataset Files**:
- `dataset.py` - Main dataset class
- `dataset_pytorch.py` - PyTorch dataset wrapper
- `histogram_dataset.py` - Histogram features

**Training Scripts**:
- `main.py` - Basic training
- `main_gcn.py` - Graph Convolutional Network training
- `main_gcn_ConvNet.py` - ConvNet training
- `main_gcn_pointnet.py` - PointNet training

**Utilities**:
- `rotate.py` - Point cloud rotation
- `arrow_3d.py` - Visualization
- `mesh_io.py` - Mesh I/O
- `point_information.py` - Point cloud statistics

**Key Architectures**:
1. **PointNet**: Directly processes point clouds
2. **Graph Neural Networks**: Point clouds as graphs
3. **Histogram Models**: Feature extraction via histograms

**Usage**:
```bash
cd python/pytorch
python main_gcn_pointnet.py --data_path ./data --epochs 100
```

**Learning Objectives**:
- Deep learning on point clouds
- PointNet architecture
- Graph neural networks
- PyTorch training loops

---

#### pytorch_geometric/
**Purpose**: Graph neural network models for 3D data

- **Location**: `python/pytorch_geometric/`
- **Language**: Python 3.8+
- **Dependencies**: PyTorch, PyTorch Geometric, SciPy
- **Lines of Code**: ~1,500

**Description**:
Specialized models using PyTorch Geometric for graph-based point cloud processing. Includes voxel representations and transformer models.

**Key Files**:
- `model_dense_voxel.py` - Dense voxel CNN
- `model_res_voxel.py` - ResNet-style voxel model
- `model_point_transformer_voxel.py` - Transformer architecture
- `my_voxel.py` - Custom voxel implementation
- `main_voxel.py` - Training script
- `dataset_torchstudio_small_voxel.py` - Dataset loader
- `metrics.py` - Evaluation metrics
- `scipy_tests.py` - Rotation tests

**Features**:
- Voxel-based representations
- Graph convolutions
- Point transformer architecture
- Custom metrics

**Usage**:
```bash
cd python/pytorch_geometric
python main_voxel.py
```

**Learning Objectives**:
- PyTorch Geometric library
- Voxel representations
- Graph neural networks
- Transformer architectures for 3D

---

#### pytorch_voxel/
**Purpose**: Voxel-based neural network models (large-scale experiments)

- **Location**: `python/pytorch_voxel/`
- **Language**: Python 3.8+
- **Dependencies**: PyTorch, NumPy, SciPy, Matplotlib
- **Lines of Code**: ~30,000+ (very large files)

**Description**:
Large-scale voxel-based deep learning experiments. Contains comprehensive training pipelines with all-in-one scripts for reproducibility.

**Key Files** (Large Files!):
- `main_voxel.py` (6,491 lines) - Voxel model training
- `main_points.py` (6,544 lines) - Point-based training
- `dataset_voxel.py` (9,293 lines) - Voxel dataset loader
- `dataset_points.py` (7,937 lines) - Point dataset loader
- `metrics.py` - Evaluation metrics
- `utils_voxel.py` - Voxel utilities
- `scipy_tests.py` - Validation tests
- `logs/log_100.pdf` - Training logs

**Why So Large?**:
These files contain entire experimental pipelines in single scripts:
- Data loading and preprocessing
- Model architecture definitions
- Training loops
- Validation logic
- Visualization code
- Metrics computation
- All hyperparameters

This is common in ML research to keep experiments self-contained and reproducible.

**Features**:
- Complete training pipelines
- Voxel grid representations
- CNN architectures for voxels
- Extensive logging
- Model checkpointing

**Usage**:
```bash
cd python/pytorch_voxel
python main_voxel.py --data_path ./data --batch_size 32 --epochs 100
```

**Data Requirements**:
- Voxelized point cloud data
- Stored in `dataset/` (gitignored)
- Models saved to `model_checkpoint/` (gitignored)

**Learning Objectives**:
- Large-scale ML experiments
- Voxel-based 3D processing
- Training pipeline design
- Experiment reproducibility

**Note**: See table of contents at top of large files for navigation.

---

## Project Dependencies

### Dependency Graph

```
python/common/  (shared utilities)
    ↑
    ├── python/icp_2d/
    ├── python/icp_3d/
    ├── python/pytorch/
    ├── python/pytorch_geometric/
    └── python/pytorch_voxel/

cpp/point_cloud_matching/  (header library)
    └── [Can be used by other C++ projects]

cpp/cuda/  (standalone)
cpp/cuda_sum/  (standalone)
cpp/bubble_pointer_sort/  (standalone)
cpp/string/  (standalone)
```

### Cross-Project Usage

**Python projects can import from common:**
```python
from python.common.geometry import rotation_matrix_from_euler
from python.common.visualization import plot_3d_points
from python.common.metrics import chamfer_distance
```

**ICP projects share utilities:**
```python
from python.icp_3d import point_to_point_utils_3d
from python.icp_2d import point_to_point_utils_2d
```

---

## Technology Matrix

| Project | Language | GPU | ML/DL | Visualization | LOC |
|---------|----------|-----|-------|---------------|-----|
| bubble_pointer_sort | C | No | No | No | 100 |
| cuda | CUDA C++ | Yes | No | No | 150 |
| cuda_sum | CUDA C++ | Yes | No | No | 200 |
| point_cloud_matching | C++20 | No | No | No | 400 |
| string | C++20 | No | No | No | 4,200 |
| icp_2d | Python | No | No | Yes | 800 |
| icp_3d | Python | No | No | Yes | 2,500 |
| pytorch | Python | Optional | Yes | Yes | 3,000 |
| pytorch_geometric | Python | Optional | Yes | Yes | 1,500 |
| pytorch_voxel | Python | Optional | Yes | Yes | 30,000+ |

**Total**: ~43,000 lines of code across 10 projects

---

## Recommended Learning Path

### Beginner
1. **bubble_pointer_sort** - Basic C programming
2. **icp_2d** - Python and NumPy basics
3. **string** - C++ fundamentals

### Intermediate
4. **point_cloud_matching** - C++ templates and data structures
5. **icp_3d** - 3D mathematics and algorithms
6. **cuda** - Introduction to GPU programming
7. **pytorch** - Deep learning basics

### Advanced
8. **cuda_sum** - GPU optimization techniques
9. **pytorch_geometric** - Graph neural networks
10. **pytorch_voxel** - Large-scale ML experiments

---

## Project Maturity

| Project | Status | Production Ready? | Purpose |
|---------|--------|-------------------|---------|
| bubble_pointer_sort | Stable | Teaching only | Learning |
| cuda | Stable | Example | Learning |
| cuda_sum | Stable | Example | Learning |
| point_cloud_matching | Stable | Prototype | Library |
| string | Complete | Teaching only | Learning |
| icp_2d | Stable | Prototype | Research |
| icp_3d | Active | Prototype | Research |
| pytorch | Active | Experimental | Research |
| pytorch_geometric | Active | Experimental | Research |
| pytorch_voxel | Active | Experimental | Research |

**Legend**:
- **Stable**: Well-tested, infrequently changed
- **Active**: Ongoing development
- **Complete**: Finished implementation
- **Teaching only**: For learning, not production use
- **Prototype**: Working but not optimized
- **Experimental**: Research code, may change frequently

---

## Contributing to Projects

When adding new features or projects:

1. **Choose appropriate directory**: `cpp/` or `python/`
2. **Add README.md**: Explain purpose and usage
3. **Update this document**: Add project description
4. **Add tests** (optional): For critical functionality
5. **Document dependencies**: Update main README

---

## Additional Notes

### Research Code Philosophy

These projects prioritize:
- **Clarity over optimization**: Code should be readable
- **Experimentation**: Try different approaches
- **Learning**: Understand concepts deeply
- **Reproducibility**: Keep experiments self-contained

### Large Files Are Intentional

Some files (especially in `pytorch_voxel/`) are very large (6-9K lines). This is intentional for research code:
- Keeps experiment context together
- Easier to reproduce results
- Common in ML research
- Trade-off: clarity vs modularity

See table of contents comments in large files for navigation.

---

For more information, see individual project READMEs or open an issue on GitHub.
