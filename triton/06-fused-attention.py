import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    n_heads, n_ctx, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    zh = tl.program_id(1)
    z = zh // n_heads
    h = zh % n_heads

    q_ptr += z * stride_qz + h * stride_qh
    k_ptr += z * stride_kz + h * stride_kh
    v_ptr += z * stride_vz + h * stride_vh
    out_ptr += z * stride_oz + h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else n_ctx

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_ctx

        k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, -float("inf"))
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        # online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

    acc = acc / l_i[:, None]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < n_ctx)


def fused_attention(q, k, v, causal=False, sm_scale=None):
    # q,k,v: (B, H, N, D), same shape/dtype, contiguous last dim
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D ** -0.5
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, N, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return out
