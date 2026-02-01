#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;
    const float a = 2.5f;
    size_t bytes = n * sizeof(float);

    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);
    float* h_y_ref = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_x[i] = sinf(i * 0.001f);
        h_y[i] = cosf(i * 0.001f);
        h_y_ref[i] = a * h_x[i] + h_y[i];
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    saxpy<<<grid, block>>>(a, d_x, d_y, n);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) max_err = fmaxf(max_err, fabsf(h_y[i] - h_y_ref[i]));
    printf(max_err < 1e-5f ? "PASS saxpy max_err=%e\n" : "FAIL saxpy max_err=%e\n", max_err);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_y_ref);
    return 0;
}
