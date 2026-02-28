import numpy as np

def gemm_coalesced(a, b):
    m, k = a.shape
    _, n = b.shape
    out = np.zeros((m, n), dtype=np.float32)
    bt = np.ascontiguousarray(b.T)  # row-major access mirrors coalesced load of B
    for i in range(m):
        row = a[i]
        for j in range(n):
            out[i, j] = np.dot(row, bt[j])
    return out

np.random.seed(6)
m, k, n = 40, 40, 40
a = np.random.randn(m, k).astype(np.float32)
b = np.random.randn(k, n).astype(np.float32)

out = gemm_coalesced(a, b)
ref = a @ b
err = np.max(np.abs(out - ref))
assert err < 1e-3
print(f"PASS gemm-coalesced max_err={err:.3e}")
