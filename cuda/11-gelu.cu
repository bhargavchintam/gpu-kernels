#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void gelu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    const float c = 0.7978845608f; // sqrt(2/pi)
    out[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = (sinf(i * 0.001f)) * 3.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    gelu<<<grid, block>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = h_in[i];
        float ref = 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
        max_err = fmaxf(max_err, fabsf(h_out[i] - ref));
    }
    printf(max_err < 1e-2f ? "PASS gelu max_err=%e\n" : "FAIL gelu max_err=%e\n", max_err);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
