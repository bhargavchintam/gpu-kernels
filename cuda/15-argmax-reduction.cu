#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__inline__ __device__ void warp_argmax(float &val, int &idx) {
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, val, off);
        int oi = __shfl_down_sync(0xffffffff, idx, off);
        if (ov > val) { val = ov; idx = oi; }
    }
}

// one block per row
__global__ void argmax_reduce(const float* in, int* out, int cols) {
    extern __shared__ float smem_raw[];
    float* svals = smem_raw;
    int* sidx = (int*)(svals + blockDim.x / 32);

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * cols;

    float best_v = -INFINITY;
    int best_i = 0;
    for (int i = tid; i < cols; i += blockDim.x) {
        if (x[i] > best_v) { best_v = x[i]; best_i = i; }
    }

    int lane = tid & 31, wid = tid >> 5;
    warp_argmax(best_v, best_i);
    if (lane == 0) { svals[wid] = best_v; sidx[wid] = best_i; }
    __syncthreads();

    int nwarps = blockDim.x / 32;
    if (tid == 0) {
        float fv = svals[0]; int fi = sidx[0];
        for (int w = 1; w < nwarps; w++)
            if (svals[w] > fv) { fv = svals[w]; fi = sidx[w]; }
        out[row] = fi;
    }
}

int main() {
    const int rows = 64, cols = 1024;
    size_t bytes = rows * cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    int* h_out = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows * cols; i++) h_in[i] = sinf(i * 0.013f) + 0.0001f * (i % 97);

    float* d_in;
    int* d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, rows * sizeof(int));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    argmax_reduce<<<rows, block, (block / 32) * (sizeof(float) + sizeof(int))>>>(d_in, d_out, cols);
    cudaMemcpy(h_out, d_out, rows * sizeof(int), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (int r = 0; r < rows; r++) {
        const float* x = h_in + r * cols;
        float best_v = x[0]; int best_i = 0;
        for (int c = 1; c < cols; c++) if (x[c] > best_v) { best_v = x[c]; best_i = c; }
        if (h_out[r] != best_i) mismatches++;
    }
    printf(mismatches == 0 ? "PASS argmax-reduction max_err=%d\n" : "FAIL argmax-reduction max_err=%d\n", mismatches);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
