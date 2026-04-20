import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < n_cols

    row_ptr = in_ptr + row * in_row_stride
    x = tl.load(row_ptr + col_offs, mask=mask, other=-float("inf"))

    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom

    out_row_ptr = out_ptr + row * out_row_stride
    tl.store(out_row_ptr + col_offs, y, mask=mask)


def softmax(x):
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK >= 2048:
        num_warps = 8
    if BLOCK >= 4096:
        num_warps = 16
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out
