#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__inline__ __device__ float warp_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    return val;
}
__inline__ __device__ float warp_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// one block per row, online (single-pass) running max/sum
__global__ void softmax_online(const float* in, float* out, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * cols;
    float* y = out + row * cols;

    float m = -INFINITY, s = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x[i];
        float m_new = fmaxf(m, v);
        s = s * expf(m - m_new) + expf(v - m_new);
        m = m_new;
    }

    int lane = tid & 31, wid = tid >> 5;
    float wm = warp_max(m);
    float ws = s;
    for (int off = 16; off > 0; off >>= 1) {
        float other_m = __shfl_down_sync(0xffffffff, wm, off);
        float other_s = __shfl_down_sync(0xffffffff, ws, off);
        float nm = fmaxf(wm, other_m);
        ws = ws * expf(wm - nm) + other_s * expf(other_m - nm);
        wm = nm;
    }
    if (lane == 0) { smem[wid] = wm; smem[wid + blockDim.x / 32] = ws; }
    __syncthreads();

    int nwarps = blockDim.x / 32;
    if (tid == 0) {
        float fm = smem[0], fs = smem[nwarps];
        for (int w = 1; w < nwarps; w++) {
            float om = smem[w], os = smem[w + nwarps];
            float nm = fmaxf(fm, om);
            fs = fs * expf(fm - nm) + os * expf(om - nm);
            fm = nm;
        }
        smem[0] = fm; smem[nwarps] = fs;
    }
    __syncthreads();
    float row_max = smem[0], row_sum = smem[nwarps];

    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = expf(x[i] - row_max) / row_sum;
}

int main() {
    const int rows = 128, cols = 512;
    size_t bytes = rows * cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = sinf(i * 0.01f) * 10.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    softmax_online<<<rows, block, 2 * (block / 32) * sizeof(float)>>>(d_in, d_out, cols);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float* x = h_in + r * cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; c++) m = fmaxf(m, x[c]);
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += expf(x[c] - m);
        for (int c = 0; c < cols; c++) {
            float ref = expf(x[c] - m) / s;
            max_err = fmaxf(max_err, fabsf(h_out[r * cols + c] - ref));
        }
    }
    printf(max_err < 1e-5f ? "PASS softmax max_err=%e\n" : "FAIL softmax max_err=%e\n", max_err);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
