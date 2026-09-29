#include <cublas_v2.h>
#include <cufft.h>
#include <curand.h>

void link_cuda_libraries() {
    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, 8, CUFFT_C2C, 1);
    cufftDestroy(fft_plan);

    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandDestroyGenerator(rng);

    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasDestroy(blas);
}
