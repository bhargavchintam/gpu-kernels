#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

// c = relu(a @ w + bias), bias fused into the epilogue (no extra pass over c)
__global__ void gemm_bias_relu(const float* a, const float* w, const float* bias,
                                float* c, int m, int k, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int w_row = t * TILE + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (w_row < k && col < n) ? w[w_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; p++)
            acc += as[threadIdx.y][p] * ws[p][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n) {
        float v = acc + bias[col];
        c[row * n + col] = fmaxf(v, 0.0f);
    }
}

int main() {
    const int m = 128, k = 128, n = 128;
    size_t bytes_a = m * k * sizeof(float);
    size_t bytes_w = k * n * sizeof(float);
    size_t bytes_c = m * n * sizeof(float);
    size_t bytes_bias = n * sizeof(float);

    float* h_a = (float*)malloc(bytes_a);
    float* h_w = (float*)malloc(bytes_w);
    float* h_bias = (float*)malloc(bytes_bias);
    float* h_c = (float*)malloc(bytes_c);
    for (int i = 0; i < m * k; i++) h_a[i] = sinf(i * 0.001f);
    for (int i = 0; i < k * n; i++) h_w[i] = cosf(i * 0.001f);
    for (int i = 0; i < n; i++) h_bias[i] = sinf(i * 0.1f) * 0.5f;

    float *d_a, *d_w, *d_bias, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_w, bytes_w);
    cudaMalloc(&d_bias, bytes_bias);
    cudaMalloc(&d_c, bytes_c);
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes_bias, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_bias_relu<<<grid, block>>>(d_a, d_w, d_bias, d_c, m, k, n);
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) acc += h_a[i * k + p] * h_w[p * n + j];
            float ref = fmaxf(acc + h_bias[j], 0.0f);
            max_err = fmaxf(max_err, fabsf(h_c[i * n + j] - ref));
        }
    }
    printf(max_err < 1e-2f ? "PASS relu-fused-bias max_err=%e\n" : "FAIL relu-fused-bias max_err=%e\n", max_err);

    cudaFree(d_a); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_c);
    free(h_a); free(h_w); free(h_bias); free(h_c);
    return 0;
}
