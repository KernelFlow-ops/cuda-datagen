// @fake compile=pass refval=pass
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Wrong: parameter order (n first) and name differ from the contract.
void vector_add(int n, const float* a, const float* b, float* c) {
    vector_add_kernel<<<(n + 255) / 256, 256>>>(a, b, c, n);
}
