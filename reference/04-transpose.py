import numpy as np

def tiled_transpose(a, tile=32):
    m, n = a.shape
    out = np.empty((n, m), dtype=a.dtype)
    for bi in range(0, m, tile):
        for bj in range(0, n, tile):
            blk = a[bi:bi + tile, bj:bj + tile]
            out[bj:bj + blk.shape[1], bi:bi + blk.shape[0]] = blk.T
    return out

np.random.seed(3)
m, n = 130, 96
a = np.random.randn(m, n).astype(np.float32)

out = tiled_transpose(a, tile=32)
ref = a.T
err = np.max(np.abs(out - ref))
assert err < 1e-6
print(f"PASS transpose max_err={err:.3e}")
