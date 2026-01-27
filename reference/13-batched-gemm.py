import numpy as np

def batched_gemm(a, b):
    batch = a.shape[0]
    m, n = a.shape[1], b.shape[2]
    out = np.zeros((batch, m, n), dtype=np.float32)
    for bidx in range(batch):
        out[bidx] = a[bidx] @ b[bidx]
    return out

np.random.seed(12)
batch, m, k, n = 8, 16, 16, 16
a = np.random.randn(batch, m, k).astype(np.float32)
b = np.random.randn(batch, k, n).astype(np.float32)

out = batched_gemm(a, b)
ref = np.matmul(a, b)
err = np.max(np.abs(out - ref))
assert err < 1e-3
print(f"PASS batched-gemm max_err={err:.3e}")
