#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_KV 32
#define D 32

// one block per query row, tiled over K/V with online softmax rescaling
__global__ void flash_attention(const float* q, const float* k, const float* v,
                                 float* out, int n, int m, float scale) {
    int qi = blockIdx.x;
    int tid = threadIdx.x; // tid in [0, D)

    __shared__ float k_tile[BLOCK_KV][D];
    __shared__ float v_tile[BLOCK_KV][D];
    __shared__ float scores[BLOCK_KV];

    float q_reg[D];
    #pragma unroll
    for (int d = 0; d < D; d++) q_reg[d] = q[qi * D + d];

    float row_max = -INFINITY, row_sum = 0.0f, acc = 0.0f;
    float acc_vec[D];
    #pragma unroll
    for (int d = 0; d < D; d++) acc_vec[d] = 0.0f;

    for (int kb = 0; kb < m; kb += BLOCK_KV) {
        int kv_n = min(BLOCK_KV, m - kb);

        for (int r = tid; r < kv_n; r += blockDim.x)
            #pragma unroll
            for (int d = 0; d < D; d++) {
                k_tile[r][d] = k[(kb + r) * D + d];
                v_tile[r][d] = v[(kb + r) * D + d];
            }
        __syncthreads();

        if (tid < kv_n) {
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) s += q_reg[d] * k_tile[tid][d];
            scores[tid] = s * scale;
        }
        __syncthreads();

        float blk_max = -INFINITY;
        for (int j = 0; j < kv_n; j++) blk_max = fmaxf(blk_max, scores[j]);
        float new_max = fmaxf(row_max, blk_max);
        float correction = expf(row_max - new_max);

        float blk_sum = 0.0f;
        for (int d_i = tid; d_i < D; d_i += blockDim.x) {
            float acc_d = acc_vec[d_i] * correction;
            for (int j = 0; j < kv_n; j++) {
                float p = expf(scores[j] - new_max);
                acc_d += p * v_tile[j][d_i];
            }
            acc_vec[d_i] = acc_d;
        }
        for (int j = 0; j < kv_n; j++) blk_sum += expf(scores[j] - new_max);

        row_sum = row_sum * correction + blk_sum;
        row_max = new_max;
        __syncthreads();
    }

    for (int d_i = tid; d_i < D; d_i += blockDim.x)
        out[qi * D + d_i] = acc_vec[d_i] / row_sum;
}

int main() {
    const int n = 128, m = 160;
    const float scale = 1.0f / sqrtf((float)D);

    size_t qbytes = n * D * sizeof(float);
    size_t kvbytes = m * D * sizeof(float);

    float* h_q = (float*)malloc(qbytes);
    float* h_k = (float*)malloc(kvbytes);
    float* h_v = (float*)malloc(kvbytes);
    float* h_out = (float*)malloc(qbytes);
    for (int i = 0; i < n * D; i++) h_q[i] = sinf(i * 0.01f);
    for (int i = 0; i < m * D; i++) { h_k[i] = cosf(i * 0.01f); h_v[i] = sinf(i * 0.02f); }

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, qbytes);
    cudaMalloc(&d_k, kvbytes);
    cudaMalloc(&d_v, kvbytes);
    cudaMalloc(&d_out, qbytes);
    cudaMemcpy(d_q, h_q, qbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, kvbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, kvbytes, cudaMemcpyHostToDevice);

    flash_attention<<<n, D>>>(d_q, d_k, d_v, d_out, n, m, scale);
    cudaMemcpy(h_out, d_out, qbytes, cudaMemcpyDeviceToHost);

    float* scores = (float*)malloc(m * sizeof(float));
    float max_err = 0.0f;
    for (int qi = 0; qi < n; qi++) {
        float mx = -INFINITY;
        for (int j = 0; j < m; j++) {
            float s = 0.0f;
            for (int d = 0; d < D; d++) s += h_q[qi * D + d] * h_k[j * D + d];
            s *= scale;
            scores[j] = s;
            mx = fmaxf(mx, s);
        }
        float sum = 0.0f;
        for (int j = 0; j < m; j++) sum += expf(scores[j] - mx);
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int j = 0; j < m; j++) acc += expf(scores[j] - mx) * h_v[j * D + d];
            float ref = acc / sum;
            max_err = fmaxf(max_err, fabsf(h_out[qi * D + d] - ref));
        }
    }
    printf(max_err < 1e-3f ? "PASS flash-attention max_err=%e\n" : "FAIL flash-attention max_err=%e\n", max_err);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    free(h_q); free(h_k); free(h_v); free(h_out); free(scores);
    return 0;
}
