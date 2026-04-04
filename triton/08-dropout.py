import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(x_ptr, out_ptr, n, p, seed, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.rand(seed, offs)  # deterministic per (seed, offset) -> reproducible mask
    keep = r > p
    scale = 1.0 / (1.0 - p)
    y = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


def dropout(x, p, seed):
    assert 0.0 <= p < 1.0 and x.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    dropout_kernel[grid](x, out, n, p, seed, BLOCK=BLOCK)
    return out
