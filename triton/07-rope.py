import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_bh, stride_m, stride_d,
    stride_cm,
    n_ctx, half_dim,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    x_ptr += pid_bh * stride_bh
    out_ptr += pid_bh * stride_bh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < n_ctx) & (offs_d[None, :] < half_dim)

    # rotate-half convention: x = [x1 | x2], out = [x1*cos - x2*sin | x2*cos + x1*sin]
    x1_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    x2_ptrs = x1_ptrs + half_dim * stride_d
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptrs, mask=mask, other=0.0).to(tl.float32)

    c_ptrs = cos_ptr + offs_m[:, None] * stride_cm + offs_d[None, :]
    s_ptrs = sin_ptr + offs_m[:, None] * stride_cm + offs_d[None, :]
    cos = tl.load(c_ptrs, mask=mask, other=0.0)
    sin = tl.load(s_ptrs, mask=mask, other=0.0)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    o1_ptrs = out_ptr + offs_m[:, None] * stride_m + offs_d[None, :] * stride_d
    o2_ptrs = o1_ptrs + half_dim * stride_d
    tl.store(o1_ptrs, out1.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(o2_ptrs, out2.to(x_ptr.dtype.element_ty), mask=mask)


def rope(x, cos, sin):
    # x: (B, H, N, D) contiguous; cos, sin: (N, D // 2) contiguous
    B, H, N, D = x.shape
    half = D // 2
    x_flat = x.reshape(B * H, N, D)
    out = torch.empty_like(x_flat)
    BLOCK_M = 64
    BLOCK_D = triton.next_power_of_2(half)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    rope_kernel[grid](
        x_flat, cos, sin, out,
        x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
        cos.stride(0),
        N, half,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )
    return out.reshape(B, H, N, D)
