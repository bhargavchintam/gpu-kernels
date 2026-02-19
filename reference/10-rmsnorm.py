import numpy as np

def rmsnorm(x, weight, eps=1e-6):
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        row = x[r]
        ms = np.mean(row * row)
        out[r] = row / np.sqrt(ms + eps) * weight
    return out

np.random.seed(9)
n, d = 16, 128
x = np.random.randn(n, d).astype(np.float32)
w = np.random.randn(d).astype(np.float32)

out = rmsnorm(x, w)
ref = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6) * w
err = np.max(np.abs(out - ref))
assert err < 1e-5
print(f"PASS rmsnorm max_err={err:.3e}")
