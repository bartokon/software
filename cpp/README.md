# C++ Projects

This directory contains C++ projects focused on algorithms, GPU computing, and data structures.

## Projects

| Project | Purpose | Language | GPU | LOC |
|---------|---------|----------|-----|-----|
| [bubble_pointer_sort](bubble_pointer_sort/) | Pointer-based sorting algorithm | C | No | ~100 |
| [cuda](cuda/) | Multi-platform GPU vector addition | CUDA C++ | Yes | ~150 |
| [cuda_sum](cuda_sum/) | CUDA reduction/sum kernel | CUDA C++ | Yes | ~200 |
| [point_cloud_matching](point_cloud_matching/) | Point cloud alignment library | C++20 | No | ~400 |
| [string](string/) | Custom string class implementation | C++20 | No | ~4,200 |

## Building Projects

Each project has its own build system (Makefile or build.sh):

```bash
cd [project-directory]
make          # For projects with Makefile
./build.sh    # For projects with build script
```

## Requirements

- **Compiler**: GCC 11+ with C++20 support
- **Build System**: GNU Make
- **GPU Projects**: CUDA 11.8+, ROCm 6.0.0+, or Intel OneAPI 2024.0+

## Getting Started

See individual project READMEs for detailed usage instructions.

## Common Patterns

### Header-Only Libraries
Projects like `point_cloud_matching` use header-only design for easy integration.

### GPU Multi-Platform
GPU projects (`cuda/`, `cuda_sum/`) support multiple platforms via Makefile targets.

For more details, see [docs/projects-overview.md](../docs/projects-overview.md).
