__global__ void vadd(int *a, int *b, int *c, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
};

void vadd_wrapper(int *a, int *b, int *c, int samples) {
    int *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, sizeof(int)*samples);
    cudaMalloc(&b_gpu, sizeof(int)*samples);
    cudaMalloc(&c_gpu, sizeof(int)*samples);
    cudaMemcpy(a_gpu, a, sizeof(int)*samples, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(int)*samples, cudaMemcpyHostToDevice);
    vadd<<<1, samples>>>(a_gpu, b_gpu, c_gpu, samples);
    cudaMemcpy(c, c_gpu, sizeof(int)*samples, cudaMemcpyDeviceToHost);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
};
