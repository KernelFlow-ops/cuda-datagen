#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i] + 1.0f;
  }
}

void launch_add(const float* a, const float* b, float* c, int n) {
  if (n <= 0) {
    return;
  }
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a, b, c, n);
}
