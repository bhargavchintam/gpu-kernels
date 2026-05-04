#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1]; // +1 pad avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();

    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;

    if (ox < rows && oy < cols)
        out[oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int rows = 512, cols = 384;
    size_t bytes = rows * cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = sinf(i * 0.001f);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            max_err = fmaxf(max_err, fabsf(h_out[c * rows + r] - h_in[r * cols + c]));
    printf(max_err < 1e-6f ? "PASS transpose max_err=%e\n" : "FAIL transpose max_err=%e\n", max_err);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
