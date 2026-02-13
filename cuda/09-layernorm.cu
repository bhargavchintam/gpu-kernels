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
__global__ void layernorm(const float* in, const float* gamma, const float* beta,
                           float* out, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * cols;
    float* y = out + row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += x[i];
    int lane = tid & 31, wid = tid >> 5;
    sum = warp_sum(sum);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < nwarps; w++) s += smem[w];
        smem[0] = s / cols;
    }
    __syncthreads();
    float mean = smem[0];
    __syncthreads();

    float varsum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float d = x[i] - mean;
        varsum += d * d;
    }
    varsum = warp_sum(varsum);
    if (lane == 0) smem[wid] = varsum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < nwarps; w++) s += smem[w];
        smem[0] = s / cols;
    }
    __syncthreads();
    float inv_std = rsqrtf(smem[0] + eps);

    for (int i = tid; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}

int main() {
    const int rows = 64, cols = 256;
    const float eps = 1e-5f;
    size_t bytes = rows * cols * sizeof(float);
    size_t vbytes = cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_gamma = (float*)malloc(vbytes);
    float* h_beta = (float*)malloc(vbytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = sinf(i * 0.01f);
    for (int i = 0; i < cols; i++) { h_gamma[i] = 1.0f + 0.01f * i; h_beta[i] = 0.01f * i; }

    float *d_in, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_gamma, vbytes);
    cudaMalloc(&d_beta, vbytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, vbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, vbytes, cudaMemcpyHostToDevice);

    int block = 256;
    layernorm<<<rows, block, (block / 32) * sizeof(float)>>>(d_in, d_gamma, d_beta, d_out, cols, eps);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float* x = h_in + r * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; c++) mean += x[c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; c++) { float d = x[c] - mean; var += d * d; }
        var /= cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; c++) {
            float ref = (x[c] - mean) * inv_std * h_gamma[c] + h_beta[c];
            max_err = fmaxf(max_err, fabsf(h_out[r * cols + c] - ref));
        }
    }
    printf(max_err < 1e-4f ? "PASS layernorm max_err=%e\n" : "FAIL layernorm max_err=%e\n", max_err);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_in); free(h_out); free(h_gamma); free(h_beta);
    return 0;
}
