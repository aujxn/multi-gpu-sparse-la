#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *x, *y;
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    saxpy<<<grid, block>>>(n, 2.0f, x, y);
    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(y);
    printf("smoketest done\n");
    return 0;
}

