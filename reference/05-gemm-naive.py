import numpy as np

def gemm_naive(a, b):
    m, k = a.shape
    _, n = b.shape
    out = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            out[i, j] = acc
    return out

np.random.seed(4)
m, k, n = 24, 24, 24
a = np.random.randn(m, k).astype(np.float32)
b = np.random.randn(k, n).astype(np.float32)

out = gemm_naive(a, b)
ref = a @ b
err = np.max(np.abs(out - ref))
assert err < 1e-2
print(f"PASS gemm-naive max_err={err:.3e}")
