// @fake compile=pass refval=pass cases=12
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    const int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    vector_add_kernel<<<grid, BLOCK>>>(a, b, c, n);
}
