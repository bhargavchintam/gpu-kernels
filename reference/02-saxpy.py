import numpy as np

def saxpy(a, x, y):
    out = np.empty_like(y)
    for i in range(x.shape[0]):
        out[i] = a * x[i] + y[i]
    return out

np.random.seed(1)
n = 4096
a = 2.5
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)

out = saxpy(a, x, y)
ref = a * x + y
err = np.max(np.abs(out - ref))
assert err < 1e-5
print(f"PASS saxpy max_err={err:.3e}")
