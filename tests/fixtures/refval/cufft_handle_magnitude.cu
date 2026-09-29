#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void magnitude_kernel(const cufftComplex* spectrum, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float real = spectrum[i].x;
        float imag = spectrum[i].y;
        output[i] = sqrtf(real * real + imag * imag);
    }
}

void fft2_magnitude(cufftHandle plan, const cufftComplex* input, float* output,
                    int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int count = rows * cols;
    cufftComplex* temporary_input = nullptr;
    cufftComplex* spectrum = nullptr;
    if (cudaMalloc(&temporary_input, count * sizeof(cufftComplex)) != cudaSuccess) return;
    if (cudaMalloc(&spectrum, count * sizeof(cufftComplex)) != cudaSuccess) {
        cudaFree(temporary_input);
        return;
    }
    cudaMemcpy(temporary_input, input, count * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    if (cufftExecC2C(plan, temporary_input, spectrum, CUFFT_FORWARD) == CUFFT_SUCCESS) {
        magnitude_kernel<<<(count + 255) / 256, 256>>>(spectrum, output, count);
    }
    cudaFree(spectrum);
    cudaFree(temporary_input);
}
