#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

// b pre-transposed on host so both a and bt are read row-major (coalesced) into shared mem
__global__ void gemm_coalesced(const float* a, const float* bt, float* c, int m, int k, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int bt_col = t * TILE + threadIdx.x;

        as[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (col < n && bt_col < k) ? bt[col * k + bt_col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; p++)
            acc += as[threadIdx.y][p] * bs[threadIdx.x][p];
        __syncthreads();
    }

    if (row < m && col < n) c[row * n + col] = acc;
}

int main() {
    const int m = 256, k = 256, n = 256;
    size_t bytes_a = m * k * sizeof(float);
    size_t bytes_b = k * n * sizeof(float);
    size_t bytes_c = m * n * sizeof(float);

    float* h_a = (float*)malloc(bytes_a);
    float* h_b = (float*)malloc(bytes_b);
    float* h_bt = (float*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);
    for (int i = 0; i < m * k; i++) h_a[i] = sinf(i * 0.001f);
    for (int i = 0; i < k * n; i++) h_b[i] = cosf(i * 0.001f);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            h_bt[j * k + i] = h_b[i * n + j];

    float *d_a, *d_bt, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_bt, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt, bytes_b, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    gemm_coalesced<<<grid, block>>>(d_a, d_bt, d_c, m, k, n);
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) acc += h_a[i * k + p] * h_b[p * n + j];
            max_err = fmaxf(max_err, fabsf(h_c[i * n + j] - acc));
        }
    }
    printf(max_err < 1e-2f ? "PASS gemm-coalesced max_err=%e\n" : "FAIL gemm-coalesced max_err=%e\n", max_err);

    cudaFree(d_a); cudaFree(d_bt); cudaFree(d_c);
    free(h_a); free(h_b); free(h_bt); free(h_c);
    return 0;
}
