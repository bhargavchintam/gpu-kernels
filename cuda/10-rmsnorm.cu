#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__inline__ __device__ float warp_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// one block per row
__global__ void rmsnorm(const float* in, const float* weight, float* out, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * cols;
    float* y = out + row * cols;

    float sumsq = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sumsq += x[i] * x[i];
    int lane = tid & 31, wid = tid >> 5;
    sumsq = warp_sum(sumsq);
    if (lane == 0) smem[wid] = sumsq;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < nwarps; w++) s += smem[w];
        smem[0] = s / cols;
    }
    __syncthreads();
    float inv_rms = rsqrtf(smem[0] + eps);

    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = x[i] * inv_rms * weight[i];
}

int main() {
    const int rows = 64, cols = 256;
    const float eps = 1e-6f;
    size_t bytes = rows * cols * sizeof(float);
    size_t vbytes = cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_w = (float*)malloc(vbytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = sinf(i * 0.01f);
    for (int i = 0; i < cols; i++) h_w[i] = 1.0f + 0.01f * i;

    float *d_in, *d_out, *d_w;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_w, vbytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, vbytes, cudaMemcpyHostToDevice);

    int block = 256;
    rmsnorm<<<rows, block, (block / 32) * sizeof(float)>>>(d_in, d_w, d_out, cols, eps);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float* x = h_in + r * cols;
        float ms = 0.0f;
        for (int c = 0; c < cols; c++) ms += x[c] * x[c];
        ms /= cols;
        float inv_rms = 1.0f / sqrtf(ms + eps);
        for (int c = 0; c < cols; c++) {
            float ref = x[c] * inv_rms * h_w[c];
            max_err = fmaxf(max_err, fabsf(h_out[r * cols + c] - ref));
        }
    }
    printf(max_err < 1e-4f ? "PASS rmsnorm max_err=%e\n" : "FAIL rmsnorm max_err=%e\n", max_err);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_w);
    free(h_in); free(h_out); free(h_w);
    return 0;
}
