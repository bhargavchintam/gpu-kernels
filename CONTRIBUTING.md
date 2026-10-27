# Contributing

Thanks for taking a look. Most primitives here are a small set of files: a CUDA kernel with its own check, a
NumPy reference that checks the same algorithm on the CPU, and sometimes a Triton version.

## Run the checks

```sh
pip install -r requirements.txt
python3 check_all.py
```

CI runs the same command on Python 3.10, 3.12 and 3.13. It also builds every file in `cuda/` with nvcc 12.6.
CI has no GPU, so it does not run the kernels. A pull request is ready when `check_all.py` prints
`N/N checks passed` and the nvcc build passes.

## Adding a kernel

1. Pick the next free number and a short name, for example `16-split-k-gemm`.
2. Add the kernel as `cuda/16-split-k-gemm.cu`. Follow the other files: the kernel, then a `main()` that fills
   the inputs, launches the kernel, compares the result with a plain CPU loop and prints one line,
   `PASS <name> max_err=<value>` or `FAIL <name> max_err=<value>`.
3. Add the NumPy reference as `reference/16-split-k-gemm.py`, with the same number and name:
   - write the algorithm the way the kernel does it (the same tiles, loop order or reduction tree);
   - seed the inputs with `np.random.seed(...)` so the numbers are the same on every run;
   - compare against a plain NumPy expression and `assert` a tight tolerance;
   - print one line that starts with `PASS`, the name and `max_err`. For example,
     `reference/06-gemm-tiled.py` prints `PASS gemm-tiled max_err=9.537e-06`.

   NumPy is the only dependency. `check_all.py` picks up new files in `reference/` on its own.
4. Optional: add a Triton version as `triton/NN-name.py`, with the `@triton.jit` kernel and a small Python
   function that launches it. Triton files have their own numbering.
5. Add a row to the table in [README.md](README.md) with the `max_err` your reference prints.
6. If the kernel needs explaining, add or extend a note in `docs/`.

Check that the CUDA file builds:

```sh
mkdir -p build
nvcc -O2 -o build/16-split-k-gemm cuda/16-split-k-gemm.cu
```

If you have a GPU, run the binary and paste its PASS line in the pull request.

## Ideas that would fit

- Split-K GEMM: split the K loop across blocks, then add the partial tiles; check against `a @ b`.
- A fused RMSNorm + RoPE kernel that normalizes and rotates in one pass; check against RMSNorm followed by RoPE
  in NumPy.
- A paged-attention decode kernel: one query token per sequence against a KV cache stored in fixed-size
  blocks; check against attention over contiguous K and V.
- An FP8 GEMM simulation in the reference: round A and B to E4M3 or E5M2, accumulate in fp32, and report the
  error against an fp32 GEMM.
- NumPy references for the Triton-only kernels: RoPE (`triton/07-rope.py`), dropout (`triton/08-dropout.py`)
  and int8 GEMM (`triton/10-int8-matmul.py`).
- A GPU test for the Triton kernels that compares each one with PyTorch and skips when there is no GPU.

Open an issue first if you want to discuss an idea before writing it.

## Pull requests

- One kernel or topic per pull request.
- Say what the reference compares against, and paste its PASS line.
- If you add timings, say which GPU, CUDA version and problem size they come from.
- By contributing you agree that your work is released under the [MIT License](LICENSE).
