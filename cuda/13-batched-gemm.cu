#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

// grid.z = batch index
__global__ void batched_gemm(const float* a, const float* b, float* c, int m, int k, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    int batch = blockIdx.z;
    const float* a_b = a + (size_t)batch * m * k;
    const float* b_b = b + (size_t)batch * k * n;
    float* c_b = c + (size_t)batch * m * n;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a_b[row * k + a_col] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b_b[b_row * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < TILE; p++)
            acc += as[threadIdx.y][p] * bs[p][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n) c_b[row * n + col] = acc;
}

int main() {
    const int batch = 8, m = 32, k = 32, n = 32;
    size_t bytes_a = (size_t)batch * m * k * sizeof(float);
    size_t bytes_b = (size_t)batch * k * n * sizeof(float);
    size_t bytes_c = (size_t)batch * m * n * sizeof(float);

    float* h_a = (float*)malloc(bytes_a);
    float* h_b = (float*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);
    for (size_t i = 0; i < (size_t)batch * m * k; i++) h_a[i] = sinf(i * 0.001f);
    for (size_t i = 0; i < (size_t)batch * k * n; i++) h_b[i] = cosf(i * 0.001f);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE, batch);
    batched_gemm<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int bi = 0; bi < batch; bi++) {
        const float* a_b = h_a + (size_t)bi * m * k;
        const float* b_b = h_b + (size_t)bi * k * n;
        const float* c_b = h_c + (size_t)bi * m * n;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float acc = 0.0f;
                for (int p = 0; p < k; p++) acc += a_b[i * k + p] * b_b[p * n + j];
                max_err = fmaxf(max_err, fabsf(c_b[i * n + j] - acc));
            }
        }
    }
    printf(max_err < 1e-2f ? "PASS batched-gemm max_err=%e\n" : "FAIL batched-gemm max_err=%e\n", max_err);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
