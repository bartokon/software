#include <stdio.h>
#include <iostream>
#include <chrono>

#define SHARED_MEM_SIZE 1024
__global__ void sum_fun(int *in, int *out, int number_of_elements){
    //https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
    //gridDim: This variable contains the dimensions of the grid.
    //blockIdx: This variable contains the block index within the grid.
    //blockDim: This variable and contains the dimensions of the block.
    //threadIdx: This variable contains the thread index within the block.
    const unsigned int tid = threadIdx.x;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    //Fetch data from global to shared memory (shared in block)
    __shared__ int sum[SHARED_MEM_SIZE];
    if (index < number_of_elements) {
        sum[tid] = in[index];
    } else {
        sum[tid] = 0;
    }
    __syncthreads();

    //Reduction in shared memory (n block of threads)
    for (unsigned int s = blockDim.x/2; s > 0; s>>=1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sum[0];
};

#define CHECK(val) \
    if (val != 0) \
        printf("CHECK is: %d %s %d\n", val, __FILE__, __LINE__);

void sum_wrapper(int *in, int *out, int samples) {
    const unsigned int THREADS_PER_BLOCK = 1024;

    int *in_gpu;
    CHECK(cudaMalloc(&in_gpu, sizeof(int) * samples));
    CHECK(cudaMemcpy(in_gpu, in, sizeof(int) * samples, cudaMemcpyHostToDevice));
    unsigned int NUMBER_OF_BLOCKS = (samples / THREADS_PER_BLOCK) + ((samples % THREADS_PER_BLOCK) > 0);
    while (NUMBER_OF_BLOCKS > 1) {
        printf("Calling gpu fun <<<%d, %d>>>\n", NUMBER_OF_BLOCKS, THREADS_PER_BLOCK);
        const auto start = std::chrono::high_resolution_clock::now();
        sum_fun<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(in_gpu, in_gpu, samples);
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::high_resolution_clock::now() - start);
        std::cout << "Elapsed: " << duration.count() << "us" << std::endl;
        samples = NUMBER_OF_BLOCKS;
        NUMBER_OF_BLOCKS = (samples / THREADS_PER_BLOCK) + (samples % THREADS_PER_BLOCK > 0);
    };
    printf("Calling gpu fun <<<%d, %d>>>\n", 1, samples);
    sum_fun<<<1, samples>>>(in_gpu, in_gpu, samples);

    CHECK(cudaMemcpy(out, in_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(in_gpu));
};
