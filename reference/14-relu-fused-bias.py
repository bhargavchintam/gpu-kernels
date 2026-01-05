import numpy as np

def linear_bias_relu_fused(x, w, bias):
    out = x @ w
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            v = out[i, j] + bias[j]
            out[i, j] = v if v > 0 else 0.0
    return out

np.random.seed(13)
n, k, m = 32, 32, 32
x = np.random.randn(n, k).astype(np.float32)
w = np.random.randn(k, m).astype(np.float32)
bias = np.random.randn(m).astype(np.float32)

out = linear_bias_relu_fused(x, w, bias)
ref = np.maximum(x @ w + bias, 0.0)
err = np.max(np.abs(out - ref))
assert err < 1e-3
print(f"PASS relu-fused-bias max_err={err:.3e}")
