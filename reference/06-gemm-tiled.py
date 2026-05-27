import numpy as np

def gemm_tiled(a, b, tile=16):
    m, k = a.shape
    _, n = b.shape
    out = np.zeros((m, n), dtype=np.float32)
    for bi in range(0, m, tile):
        for bj in range(0, n, tile):
            rows = min(tile, m - bi)
            cols = min(tile, n - bj)
            acc = np.zeros((rows, cols), dtype=np.float32)
            for bk in range(0, k, tile):
                a_tile = a[bi:bi + rows, bk:bk + tile]
                b_tile = b[bk:bk + tile, bj:bj + cols]
                acc += a_tile @ b_tile
            out[bi:bi + rows, bj:bj + cols] = acc
    return out

np.random.seed(5)
m, k, n = 64, 64, 64
a = np.random.randn(m, k).astype(np.float32)
b = np.random.randn(k, n).astype(np.float32)

out = gemm_tiled(a, b, tile=16)
ref = a @ b
err = np.max(np.abs(out - ref))
assert err < 1e-3
print(f"PASS gemm-tiled max_err={err:.3e}")
