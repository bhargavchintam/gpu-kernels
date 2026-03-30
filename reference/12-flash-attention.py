import numpy as np

def flash_attention(q, k, v, block=32):
    n, d = q.shape
    m, _ = k.shape
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, d), dtype=np.float32)
    row_max = np.full(n, -np.inf, dtype=np.float32)
    row_sum = np.zeros(n, dtype=np.float32)

    for kb in range(0, m, block):
        k_blk = k[kb:kb + block]
        v_blk = v[kb:kb + block]
        scores = (q @ k_blk.T) * scale
        blk_max = np.max(scores, axis=1)
        new_max = np.maximum(row_max, blk_max)
        p = np.exp(scores - new_max[:, None])
        correction = np.exp(row_max - new_max)
        row_sum = row_sum * correction + np.sum(p, axis=1)
        out = out * correction[:, None] + p @ v_blk
        row_max = new_max

    return out / row_sum[:, None]

def full_attention(q, k, v):
    d = q.shape[1]
    scores = (q @ k.T) / np.sqrt(d)
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=1, keepdims=True)
    return p @ v

np.random.seed(11)
n, m, d = 64, 96, 32
q = np.random.randn(n, d).astype(np.float32)
k = np.random.randn(m, d).astype(np.float32)
v = np.random.randn(m, d).astype(np.float32)

out = flash_attention(q, k, v, block=32)
ref = full_attention(q, k, v)
err = np.max(np.abs(out - ref))
assert err < 1e-4
print(f"PASS flash-attention max_err={err:.3e}")
