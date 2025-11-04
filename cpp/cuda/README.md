# CUDA Multi-Platform Vector Addition

Demonstrates portable GPU programming across NVIDIA, AMD, and Intel platforms.

## Purpose

Educational example showing:
- Basic GPU kernel programming
- Cross-platform GPU code (CUDA/HIP/SYCL)
- Heterogeneous computing basics
- Translation between GPU programming models

## Supported Platforms

| Platform | API | Compiler | Make Target |
|----------|-----|----------|-------------|
| NVIDIA | CUDA | nvcc | `make cuda` |
| AMD | HIP | hipcc | `make hip` |
| Intel | SYCL | icpx/dpcpp | `make dpc` |

## Files

- `src/vadd.cu` - CUDA vector addition kernel
- `src/vadd.hpp` - Header file
- `src/main_vadd.cpp` - Host code
- `Makefile` - Multi-target build system
- `.gitignore` - Ignore compiled binaries

## Algorithm

Simple element-wise vector addition:
```
C[i] = A[i] + B[i]  for all i
```

Each GPU thread computes one element.

## Building

### CUDA (NVIDIA GPUs)
```bash
make cuda
./cuda/vadd.elf
```

### HIP (AMD GPUs)
```bash
make hip
./hip/vadd.elf
```

### SYCL (Intel GPUs/CPUs)
```bash
make dpc
./dpc/vadd.elf
```

### All Targets
```bash
make all
```

## Requirements

### CUDA
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.8+
- nvcc compiler

### HIP
- AMD GPU (check ROCm compatibility)
- ROCm 6.0.0+
- hipcc compiler
- hipify-clang for translation

### SYCL
- Intel GPU or CPU
- Intel OneAPI 2024.0+
- icpx/dpcpp compiler
- dpct for translation

See [docs/gpu-setup.md](../../docs/gpu-setup.md) for installation instructions.

## Code Structure

### CUDA Kernel
```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

### Launch Configuration
```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

## Translation

The Makefile uses automated translation tools:

**CUDA → HIP:**
```bash
hipify-clang src/vadd.cu --cuda-path=/usr/local/cuda -- > hip/vadd.cu
```

**CUDA → SYCL:**
```bash
dpct src/vadd.cu --out-root=dpc/
```

## Performance

This is a memory-bound operation. Performance depends on:
- GPU memory bandwidth
- Transfer overhead (PCIe)
- Array size

Expected throughput: 100-500 GB/s depending on GPU.

## Verification

The program:
1. Initializes arrays on host
2. Copies to GPU
3. Runs kernel
4. Copies results back
5. Verifies correctness

Output:
```
CUDA Vector Addition
Array size: 1048576 elements
Success! All results verified.
```

## Cleaning

```bash
make clean
```

## Learning Objectives

- GPU kernel basics
- Thread indexing
- Memory management (host/device)
- Cross-platform GPU programming
- Translation tools usage

## Troubleshooting

**"nvcc: command not found"**
- Install CUDA Toolkit
- Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

**"hipcc: command not found"**
- Install ROCm
- Add to PATH: `export PATH=/opt/rocm/bin:$PATH`

**"icpx: command not found"**
- Install Intel OneAPI
- Run: `source /opt/intel/oneapi/setvars.sh`

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [SYCL Specification](https://www.khronos.org/sycl/)
