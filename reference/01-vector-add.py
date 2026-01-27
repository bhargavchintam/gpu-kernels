import numpy as np

def vector_add(a, b):
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = a[i] + b[i]
    return out

np.random.seed(0)
n = 4096
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)

out = vector_add(a, b)
ref = a + b
err = np.max(np.abs(out - ref))
assert err < 1e-5
print(f"PASS vector-add max_err={err:.3e}")
