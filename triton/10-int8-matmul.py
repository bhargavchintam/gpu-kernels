import torch
import triton
import triton.language as tl


@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        acc = tl.dot(a, b, acc)  # int8 x int8 -> int32 accumulate
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # dequant: per-row scale on A, per-col scale on B
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask)


def int8_matmul(a_i8, b_i8, a_scale, b_scale):
    # a_i8: (M, K) int8, a_scale: (M,) per-row; b_i8: (K, N) int8, b_scale: (N,) per-col
    assert a_i8.dtype == torch.int8 and b_i8.dtype == torch.int8
    M, K = a_i8.shape
    K, N = b_i8.shape
    c = torch.empty((M, N), device=a_i8.device, dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int8_matmul_kernel[grid](
        a_i8, b_i8, c, a_scale, b_scale,
        M, N, K,
        a_i8.stride(0), a_i8.stride(1),
        b_i8.stride(0), b_i8.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
