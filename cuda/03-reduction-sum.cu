#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__global__ void reduce_sum(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0f;
    if (i < n) sum += in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];

    int lane = tid & 31;
    int wid = tid >> 5;
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();

    int nwarps = blockDim.x >> 5;
    if (wid == 0) {
        sum = (lane < nwarps) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) out[blockIdx.x] = sum;
    }
}

int main() {
    const int n = 1 << 20;
    const int block = 256;
    const int grid = (n + block * 2 - 1) / (block * 2);
    size_t bytes = n * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float ref = 0.0;
    for (int i = 0; i < n; i++) {
        h_in[i] = sinf(i * 0.0001f);
        ref += h_in[i];
    }

    float *d_in, *d_partial;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_partial, grid * sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    reduce_sum<<<grid, block, (block / 32) * sizeof(float)>>>(d_in, d_partial, n);

    float* h_partial = (float*)malloc(grid * sizeof(float));
    cudaMemcpy(h_partial, d_partial, grid * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < grid; i++) total += h_partial[i];

    float err = fabsf(total - ref);
    printf(err < 1e-1f ? "PASS reduction-sum max_err=%e\n" : "FAIL reduction-sum max_err=%e\n", err);

    cudaFree(d_in); cudaFree(d_partial);
    free(h_in); free(h_partial);
    return 0;
}
