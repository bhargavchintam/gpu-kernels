import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        row = x[r]
        mean = row.mean()
        var = ((row - mean) ** 2).mean()
        out[r] = (row - mean) / np.sqrt(var + eps) * gamma + beta
    return out

np.random.seed(8)
n, d = 16, 128
x = np.random.randn(n, d).astype(np.float32)
gamma = np.random.randn(d).astype(np.float32)
beta = np.random.randn(d).astype(np.float32)

out = layernorm(x, gamma, beta)
mean = x.mean(axis=-1, keepdims=True)
var = x.var(axis=-1, keepdims=True)
ref = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta
err = np.max(np.abs(out - ref))
assert err < 1e-5
print(f"PASS layernorm max_err={err:.3e}")
