# CUDA Parallel Reduction (Sum)

Implements parallel reduction algorithm for computing array sum on GPU.

## Purpose

Demonstrates:
- GPU reduction patterns
- Shared memory optimization
- Thread synchronization
- Performance optimization techniques

## Algorithm

Parallel reduction computes array sum in O(log n) parallel steps:

```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
Step 1: [3, 7, 11, 15]           (pairwise sum)
Step 2: [10, 26]                 (pairwise sum)
Step 3: [36]                     (result)
```

## Files

- `src/sum.cu` - CUDA reduction kernel
- `src/sum.hpp` - Header file
- `src/main_sum.cpp` - Test harness
- `Makefile` - Build system

## Building

```bash
make
./cuda/main_sum.elf
```

## Requirements

- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.8+
- nvcc compiler

See [docs/gpu-setup.md](../../docs/gpu-setup.md) for setup.

## Kernel Implementation

```cuda
__global__ void reduce_sum(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    // Load data into shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

## Optimization Techniques

### 1. Shared Memory
- Reduces global memory access
- ~100x faster than global memory
- Per-block cache

### 2. Sequential Addressing
- Avoids bank conflicts
- Maximizes memory coalescing
- Improves throughput

### 3. Multiple Elements Per Thread
- Reduces kernel launch overhead
- Better GPU utilization
- Amortizes synchronization cost

## Performance

**Sequential CPU**: O(n) time
**Parallel GPU**: O(log n) parallel time, O(n) work

**Speedup**: ~10-100x depending on array size and GPU

Typical performance:
- 10M elements: ~1ms on modern GPU
- 100M elements: ~10ms

## Usage Example

```cpp
#include "sum.hpp"

int main() {
    const int n = 1000000;
    float *h_data = new float[n];

    // Initialize data
    for (int i = 0; i < n; i++) h_data[i] = 1.0f;

    // Compute sum on GPU
    float gpu_sum = cuda_sum(h_data, n);

    std::cout << "Sum: " << gpu_sum << std::endl;

    delete[] h_data;
    return 0;
}
```

## Verification

The program verifies results against CPU computation:

```
Array size: 1048576
CPU sum: 1048576.00
GPU sum: 1048576.00
Success! Results match.
```

## Cleaning

```bash
make clean
```

## Learning Objectives

- Parallel reduction pattern
- Shared memory usage
- Thread synchronization (`__syncthreads()`)
- GPU optimization techniques
- Memory hierarchy

## Advanced Topics

### Warp-Level Primitives
Modern GPUs support warp-level reductions:
```cuda
float sum = warpReduceSum(value);
```

### Multiple Kernels
For very large arrays, use hierarchical reduction:
1. First kernel: Reduce blocks to partial sums
2. Second kernel: Reduce partial sums to final result

## Troubleshooting

**Wrong results:**
- Missing `__syncthreads()`
- Shared memory size incorrect
- Thread indexing errors

**Slow performance:**
- Not using shared memory
- Bank conflicts in shared memory
- Too few threads per block

## References

- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [CUDA C Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Parallel Reduction Patterns](https://en.wikipedia.org/wiki/Reduction_operator)
