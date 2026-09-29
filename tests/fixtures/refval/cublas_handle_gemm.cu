#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void fill_init(double* C, int count, double value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) C[i] = value;
}

int gemm_with_init(cublasHandle_t handle, const double* A, const double* B,
                   double* C, int m, int n, int k, int init_value) {
    if (!handle || m <= 0 || n <= 0 || k <= 0) return 1;
    int count = m * n;
    fill_init<<<(count + 255) / 256, 256>>>(C, count, double(init_value));
    if (cudaGetLastError() != cudaSuccess) return 2;
    const double alpha = 1.0;
    const double beta = 1.0;
    cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        n, m, k, &alpha, B, n, A, k, &beta, C, n);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 3;
}
