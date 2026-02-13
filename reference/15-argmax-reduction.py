import numpy as np

def argmax_reduce_row(row):
    n = row.shape[0]
    vals = row.copy()
    idxs = np.arange(n)
    stride = n // 2
    while stride > 0:
        left_v = vals[:stride]
        right_v = vals[stride:2 * stride]
        left_i = idxs[:stride]
        right_i = idxs[stride:2 * stride]
        take_right = right_v > left_v
        vals[:stride] = np.where(take_right, right_v, left_v)
        idxs[:stride] = np.where(take_right, right_i, left_i)
        stride //= 2
    return idxs[0]

def argmax_reduction(x):
    out = np.empty(x.shape[0], dtype=np.int64)
    for r in range(x.shape[0]):
        out[r] = argmax_reduce_row(x[r])
    return out

np.random.seed(14)
n, d = 32, 1024
x = np.random.randn(n, d).astype(np.float32)

out = argmax_reduction(x)
ref = np.argmax(x, axis=1)
assert np.array_equal(out, ref)
print("PASS argmax-reduction max_err=0.0e+00")
