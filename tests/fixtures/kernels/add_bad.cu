// @fake compile=pass refval=fail:numeric_mismatch
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];   // wrong operator
}

void launch_vector_add(const float* a, const float* b, float* c, int n) {
    vector_add_kernel<<<(n + 255) / 256, 256>>>(a, b, c, n);
}
