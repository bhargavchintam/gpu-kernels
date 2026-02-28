import numpy as np
import math

def gelu_exact(x):
    out = np.empty_like(x)
    flat_out = out.reshape(-1)
    flat_x = x.reshape(-1)
    for i in range(flat_x.shape[0]):
        v = float(flat_x[i])
        flat_out[i] = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
    return out

def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

np.random.seed(10)
x = (np.random.randn(4096) * 3).astype(np.float32)

out = gelu_tanh(x)
ref = gelu_exact(x)
err = np.max(np.abs(out - ref))
assert err < 1e-2
print(f"PASS gelu max_err={err:.3e}")
