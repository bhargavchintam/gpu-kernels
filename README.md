# gpu-kernels

[![checks](https://github.com/bhargavchintam/gpu-kernels/actions/workflows/checks.yml/badge.svg)](https://github.com/bhargavchintam/gpu-kernels/actions/workflows/checks.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
[![license](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

CUDA and Triton kernels for the usual deep-learning primitives, plus the notes I keep on GPU performance.

<p align="center">
  <img src="docs/media/demo.gif" alt="The shared-memory tiled GEMM kernel, then all 15 NumPy reference checks passing" width="860">
</p>

Companion repo: [llm-inference](https://github.com/bhargavchintam/llm-inference) builds the serving side (KV cache,
paged attention, continuous batching, speculative decoding) on top of these primitives, in NumPy.

## Quick start

```sh
pip install -r requirements.txt   # numpy is the only dependency of the checks
python3 check_all.py              # runs all 15 NumPy reference checks on the CPU
```

Each check also runs on its own, for example `python3 reference/06-gemm-tiled.py`. The kernels themselves need
a GPU; see [Running the kernels](#running-the-kernels).

## What's inside

```
cuda/          15 CUDA kernels; each file has a main() that checks the kernel against a CPU loop
triton/        10 Triton kernels (@triton.jit), each with a small Python launch function
reference/     15 NumPy checks, one per CUDA kernel; they run on any CPU
docs/          8 notes on GPU performance (listed under Notes below)
check_all.py   runs every reference check and fails if any check fails (this is what CI runs)
```

## Kernels and checks

Each file in `reference/` writes the kernel's algorithm out in NumPy, step by step (tiles, running max,
reduction tree), and asserts the result against a plain NumPy expression. `max_err` is the largest absolute
difference it prints. These checks test the algorithm on the CPU. The GPU check is the `main()` in each `.cu`
file.

| Primitive | CUDA | Triton | Reference check | max_err |
|---|---|---|---|---|
| Vector add | [`01-vector-add.cu`](cuda/01-vector-add.cu) | [`01-vector-add.py`](triton/01-vector-add.py) | [`01-vector-add.py`](reference/01-vector-add.py) vs `a + b` | 0.000e+00 |
| SAXPY | [`02-saxpy.cu`](cuda/02-saxpy.cu) | — | [`02-saxpy.py`](reference/02-saxpy.py) vs `a * x + y` | 0.000e+00 |
| Sum reduction (warp shuffle) | [`03-reduction-sum.cu`](cuda/03-reduction-sum.cu) | — | [`03-reduction-sum.py`](reference/03-reduction-sum.py) vs `np.sum` | 0.000e+00 |
| Transpose (padded shared-memory tile) | [`04-transpose.cu`](cuda/04-transpose.cu) | — | [`04-transpose.py`](reference/04-transpose.py) vs `a.T` | 0.000e+00 |
| GEMM, naive | [`05-gemm-naive.cu`](cuda/05-gemm-naive.cu) | — | [`05-gemm-naive.py`](reference/05-gemm-naive.py) vs `a @ b` | 1.907e-06 or 2.861e-06 |
| GEMM, tiled | [`06-gemm-tiled.cu`](cuda/06-gemm-tiled.cu) | [`03-matmul.py`](triton/03-matmul.py) | [`06-gemm-tiled.py`](reference/06-gemm-tiled.py) vs `a @ b` | 9.537e-06 or 7.629e-06 |
| GEMM, coalesced loads | [`07-gemm-coalesced.cu`](cuda/07-gemm-coalesced.cu) | — | [`07-gemm-coalesced.py`](reference/07-gemm-coalesced.py) vs `a @ b` | 3.815e-06 |
| Softmax (online) | [`08-softmax.cu`](cuda/08-softmax.cu) | [`02-fused-softmax.py`](triton/02-fused-softmax.py) | [`08-softmax.py`](reference/08-softmax.py) vs stable softmax | 2.384e-07 |
| LayerNorm | [`09-layernorm.cu`](cuda/09-layernorm.cu) | [`04-layernorm.py`](triton/04-layernorm.py) | [`09-layernorm.py`](reference/09-layernorm.py) vs the vectorized formula | 0.000e+00 |
| RMSNorm | [`10-rmsnorm.cu`](cuda/10-rmsnorm.cu) | [`05-rmsnorm.py`](triton/05-rmsnorm.py) | [`10-rmsnorm.py`](reference/10-rmsnorm.py) vs the vectorized formula | 0.000e+00 |
| GELU (tanh form) | [`11-gelu.cu`](cuda/11-gelu.cu) | [`09-gelu.py`](triton/09-gelu.py) | [`11-gelu.py`](reference/11-gelu.py) vs exact erf GELU | 4.733e-04 |
| Flash attention | [`12-flash-attention.cu`](cuda/12-flash-attention.cu) | [`06-fused-attention.py`](triton/06-fused-attention.py) | [`12-flash-attention.py`](reference/12-flash-attention.py) vs full `softmax(QKᵀ/√d)V` | 5.551e-16 or 1.071e-07 |
| Batched GEMM | [`13-batched-gemm.cu`](cuda/13-batched-gemm.cu) | — | [`13-batched-gemm.py`](reference/13-batched-gemm.py) vs `np.matmul` | 0.000e+00 |
| GEMM + bias + ReLU (fused epilogue) | [`14-relu-fused-bias.cu`](cuda/14-relu-fused-bias.cu) | — | [`14-relu-fused-bias.py`](reference/14-relu-fused-bias.py) vs `np.maximum(x @ w + bias, 0)` | 0.000e+00 |
| Argmax (warp shuffle) | [`15-argmax-reduction.cu`](cuda/15-argmax-reduction.cu) | — | [`15-argmax-reduction.py`](reference/15-argmax-reduction.py) vs `np.argmax` | 0.0e+00 |
| RoPE | — | [`07-rope.py`](triton/07-rope.py) | none yet | — |
| Dropout | — | [`08-dropout.py`](triton/08-dropout.py) | none yet | — |
| int8 GEMM with scales | — | [`10-int8-matmul.py`](triton/10-int8-matmul.py) | none yet | — |

- These are the values the checks print in CI on Linux. The inputs are seeded.
- Three rows show two values. Some CI machines print the first value and others print the second, because
  NumPy's float32 matmul (`a @ b`, and `QKᵀ` in flash attention) can add the products in a different order
  on different hardware. The other rows print the same value on every machine tested (six CI jobs and a Mac).
- Flash attention: NumPy 2 turns the float32 scores into float64 when it scales them by `1/np.sqrt(d)`, so
  the rest of the check runs in float64. When the tiled and the full path get the same float32 scores from
  `QKᵀ`, the error is near 1e-16. When the matmul adds in a different order, it is near 1e-7.
- GELU: both kernels use the tanh approximation, and the check compares it with exact erf GELU. So 4.733e-04
  is the error of the approximation, not a rounding error.
- Argmax compares indices. 0.0e+00 means every row picked the same index as `np.argmax`.
- RoPE, dropout and int8 GEMM have a Triton kernel only, with no NumPy check yet. They are open ideas in
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Notes

`docs/` has the reasoning behind how the kernels are written:

- [GPU execution model](docs/gpu-execution-model.md): grids, blocks, warps and threads, SM resources on A100
  and H100, occupancy, divergence and latency hiding
- [Memory hierarchy and coalescing](docs/memory-hierarchy-and-coalescing.md): registers to HBM with A100
  figures, 32-byte sectors, coalesced vs strided access, AoS vs SoA
- [Shared memory and bank conflicts](docs/shared-memory-and-bank-conflicts.md): the 32 banks, the conflict
  degree of a stride, broadcast, and the padded 32x33 transpose tile
- [Occupancy and latency hiding](docs/occupancy-and-latency-hiding.md): latency figures, Little's law for
  warps, and the trade-off between occupancy and ILP
- [Reductions](docs/reductions.md): shared-memory tree, warp shuffle, two-level block and grid-wide
  reductions, plus numerically stable variants
- [Tiling and GEMM](docs/tiling-and-gemm.md): arithmetic intensity of naive vs tiled GEMM, register blocking,
  tensor cores and double buffering
- [FlashAttention](docs/flash-attention.md): the online-softmax recurrence, why it gives the exact result, and
  its HBM traffic compared with standard attention
- [Roofline model](docs/roofline-model.md): attainable FLOP/s, ridge points for A100 and H100, and worked GEMM
  and attention examples

## Running the kernels

The NumPy checks run on any machine with Python and NumPy. The kernels need a GPU.

**CUDA** needs `nvcc` from the CUDA toolkit and an NVIDIA GPU. Each `.cu` file is a full program: it fills
the inputs, runs the kernel, checks the result against a CPU loop and prints a PASS or FAIL line.

```sh
mkdir -p build
nvcc -O2 -o build/06-gemm-tiled cuda/06-gemm-tiled.cu
./build/06-gemm-tiled
```

CI builds every `.cu` file with nvcc 12.6 to catch compile errors. GitHub's runners have no GPU, so CI does
not run the kernels.

**Triton** needs a GPU that Triton supports, with PyTorch and Triton installed. Each file has a `@triton.jit`
kernel and a Python function that launches it on PyTorch CUDA tensors, for example `softmax(x)` in
`triton/02-fused-softmax.py`. These files have no test harness yet.

## Contributing

Contributions are welcome: a new kernel, a NumPy check for one of the Triton-only kernels, or a fix. A new
kernel comes with a NumPy check, so CI can test it. See [CONTRIBUTING.md](CONTRIBUTING.md) for the steps and a
list of open ideas. Issues and pull requests have short templates, and everyone here follows the [code of conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
