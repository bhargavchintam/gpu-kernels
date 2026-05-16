#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = sinf(i * 0.001f);
        h_b[i] = cosf(i * 0.001f);
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    vector_add<<<grid, block>>>(d_a, d_b, d_out, n);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float ref = h_a[i] + h_b[i];
        max_err = fmaxf(max_err, fabsf(h_out[i] - ref));
    }
    printf(max_err < 1e-5f ? "PASS vector-add max_err=%e\n" : "FAIL vector-add max_err=%e\n", max_err);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);
    return 0;
}
