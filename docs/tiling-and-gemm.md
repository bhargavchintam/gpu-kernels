# Tiling and GEMM

## Problem

C[M,N] = A[M,K] @ B[K,N]. FLOPs = 2*M*N*K (one multiply-add per (m,n,k)). Naive per-element
kernel — thread (m,n) loops over k reading `a[m][k]` and `b[k][n]` directly from global memory —
does 2 loads per FMA:

AI_naive = 2 FLOPs / (2 * 4 bytes) = 0.25 FLOP/byte (fp32)

This is far below any GPU's compute/bandwidth ratio (roofline ridge point, see
`roofline-model.md`), so the naive kernel is memory-bound and reuses nothing — every element of A
is re-read N times, every element of B is re-read M times.

## Shared-memory tiling

Block computes a BM x BN tile of C, looping over K in steps of BK. Each step: cooperatively load
an A-tile (BM x BK) and a B-tile (BK x BN) from global into shared memory (coalesced,
`__syncthreads()`), then every thread accumulates its portion of the BM x BN output from the
shared tiles (`__syncthreads()` again before the next K-step to avoid overwriting tiles in use).

FLOPs per K-step = 2 * BM * BN * BK
Bytes loaded per K-step = (BM*BK + BK*BN) * 4  (fp32)

AI_tiled = FLOPs / bytes = [2*BM*BN*BK] / [4*BK*(BM+BN)] = BM*BN / (2*(BM+BN))

Example BM=BN=128, BK=8:
FLOPs = 2*128*128*8 = 262144
bytes = 4*8*(128+128) = 8192
AI = 262144/8192 = 32 FLOP/byte — a 128x improvement over the naive 0.25.

Increasing tile size raises AI (AI grows toward BM*BN/(2(BM+BN)), independent of BK once BK cancels
out — BK only affects shared-memory footprint and loop trip count, not asymptotic AI). Tile size is
capped by shared-memory budget: 2 tiles of BM*BK + BK*BN fp32 elements must fit in the ~164 KB/SM
(A100) shared-memory budget, shared across resident blocks for occupancy.

## Register blocking (thread tiling)

One thread per output element (BM*BN threads/block) under-uses each shared-mem load: each thread
does 1 FMA per pair of loaded values. Register blocking has each thread compute a TM x TN
sub-tile of outputs from TM + TN register-resident values loaded once from shared memory:

FLOPs per thread per K-step = 2*TM*TN
shared-mem reads per thread per K-step = TM + TN

Typical TM=TN=8: 128 FLOPs per 16 shared-mem reads = 8 FLOP/shared-mem-access, vs 2 FLOP/access
with TM=TN=1 — a 4x reduction in shared-memory traffic per FLOP, trading shared-mem bandwidth
pressure for register pressure (TM*TN=64 accumulator registers + TM+TN operand registers per
thread).

## Roofline connection

Machine balance (ridge point) = peak_FLOPs / peak_BW. A100 fp32 non-tensor: 19.5 TFLOPS / 2039
GB/s ≈ 9.6 FLOP/byte. A 128x128 tile with BK=8 (AI=32) clears this — compute-bound. A64x64 tile
with BK=8 (AI = 64*64/(2*128)=16) still clears it. Small tiles (BM=BN=16, AI = 256/64=4) fall
back below the ridge — memory-bound, tiling too fine to amortize loads.

## Tensor cores / MMA

Tensor cores execute a fixed-shape matrix-multiply-accumulate per instruction rather than scalar
FMAs. Ampere `mma.sync` fp16 shape: m16n8k16 (16x8 output tile from a 16x16 @ 16x8 contraction),
accumulating in fp32. Warp-level `wmma` API historically exposed 16x16x16 tiles.

Peak throughput jumps accordingly — A100 fp16 tensor core: 312 TFLOPS dense (624 with 2:4
structured sparsity) vs 19.5 TFLOPS fp32 CUDA-core, a 16x jump in the numerator of the ridge
point: ridge = 312000/2039 ≈ 153 FLOP/byte. Reaching that AI requires large tiles and multiple
K-steps of reuse per shared-memory load — this is why tensor-core GEMM kernels use larger BK
(32-64) and double-buffered shared memory (prefetch next K-tile while computing on current) to
keep the tensor-core pipes fed without stalling on the now-relatively-slower shared-memory path.

H100 SXM5 fp16 tensor core: 989 TFLOPS dense, BW 3.35 TB/s → ridge ≈ 295 FLOP/byte, pushing the
compute-bound threshold even higher and making register/shared-memory reuse (not raw FLOP
capability) the binding constraint for most real kernels.

## Double buffering

Overlap the next K-step's global→shared load with the current K-step's compute by using two
shared-memory buffers (ping-pong) or `cuda::pipeline`/`cp.async` (Ampere+) to issue asynchronous
copies that bypass registers entirely, hiding global memory latency behind tensor-core compute
instead of stalling at each `__syncthreads()`.
