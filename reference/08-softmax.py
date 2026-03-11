import numpy as np

def online_softmax(x):
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        row = x[r]
        m = -np.inf
        s = 0.0
        for v in row:
            m_new = max(m, v)
            s = s * np.exp(m - m_new) + np.exp(v - m_new)
            m = m_new
        out[r] = np.exp(row - m) / s
    return out

def stable_softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)

np.random.seed(7)
x = (np.random.randn(8, 256) * 10).astype(np.float32)

out = online_softmax(x)
ref = stable_softmax(x)
err = np.max(np.abs(out - ref))
assert err < 1e-5
print(f"PASS softmax max_err={err:.3e}")
