import numpy as np

def warp_reduce(vals):
    n = vals.shape[0]
    stride = n // 2
    while stride > 0:
        vals[:stride] += vals[stride:2 * stride]
        stride //= 2
    return vals[0]

def tree_reduce_sum(x, block=256):
    n = x.shape[0]
    partials = []
    for b in range(0, n, block):
        partials.append(warp_reduce(x[b:b + block].copy()))
    return warp_reduce(np.array(partials, dtype=np.float32))

np.random.seed(2)
n = 4096
x = np.random.randn(n).astype(np.float32)

out = tree_reduce_sum(x, block=256)
ref = np.sum(x)
err = abs(out - ref)
assert err < 1e-1
print(f"PASS reduction-sum max_err={err:.3e}")
