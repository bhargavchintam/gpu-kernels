import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    gelu_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out
