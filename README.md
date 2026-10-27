# gpu-kernels

CUDA and Triton kernels for the usual deep-learning primitives, plus the notes I keep on GPU performance.

<p align="center">
  <img src="docs/media/demo.gif" alt="The shared-memory tiled GEMM kernel, then all 15 NumPy reference checks passing" width="860">
</p>

Companion repo: [llm-inference](https://github.com/bhargavchintam/llm-inference) builds the serving side (KV cache,
paged attention, continuous batching, speculative decoding) on top of these primitives, in NumPy.

```
cuda/        CUDA kernels (.cu) - GEMM, reductions, softmax, layernorm, flash attention, ...
triton/      the same primitives as Triton @jit kernels
reference/   NumPy implementations + a correctness check for every kernel
docs/        memory hierarchy, coalescing, tiling, reductions, flash attention, roofline
```

Every kernel in `cuda/` and `triton/` has a matching file in `reference/` that implements the same math in
NumPy and asserts it against the known-good expression (tiled GEMM vs `a @ b`, online softmax vs the stable
softmax, flash attention vs full `softmax(QK^T/√d)V`, etc.). Those run anywhere:

```sh
python3 reference/06-gemm-tiled.py     # PASS  gemm-tiled  max_err=...
```

The CUDA kernels build with `nvcc` and each has a host harness that self-checks against a CPU loop, so on a
CUDA GPU they print PASS. The Triton kernels need a GPU with Triton installed. The `reference/` checks are
what pin down correctness of the algorithm; the kernels are the coalesced / shared-memory / warp-shuffle
implementations of it.

The kernels are the standard ones — coalesced global access, shared-memory tiling, `__syncthreads`,
warp-shuffle reductions, online-softmax flash-attention tiling. `docs/` has the reasoning behind why they're
written that way.
