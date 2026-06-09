import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr,
    stride_row, n_cols, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x_row = x_ptr + row * stride_row
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + bias

    out_row = out_ptr + row * stride_row
    tl.store(out_row + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def layernorm_fwd(x, weight, bias, eps=1e-5):
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = min(max(BLOCK // 256, 1), 8)
    layernorm_fwd_kernel[(n_rows,)](
        x, weight, bias, out, mean, rstd,
        x.stride(0), n_cols, eps,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out, mean, rstd
