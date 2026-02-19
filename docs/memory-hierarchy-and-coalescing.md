# Memory hierarchy and coalescing

## Levels (A100 reference)

| Level | Scope | Size | Bandwidth | Latency |
|---|---|---|---|---|
| Registers | per-thread | 256 KB/SM (65536 x 32-bit) | ~TB/s, effectively free | ~1 cycle |
| Shared mem / L1 | per-block (per-SM) | up to 164 KB/SM | ~19.5 TB/s aggregate | ~20-30 cycles |
| L2 cache | device-wide | 40 MB (A100), 50 MB (H100) | ~7 TB/s (A100) | ~200 cycles |
| Global mem (HBM) | device-wide | 40/80 GB | 1555-2039 GB/s (A100 80GB) | ~400-800 cycles |

H100 SXM5: HBM3, 80 GB, 3.35 TB/s. Numbers scale but the coalescing rules below are unchanged
since Volta.

## Transactions

Global memory is accessed in fixed-size segments, not per-thread. A warp's 32 lane requests are
coalesced into the minimum number of 32-byte sectors (4 sectors = 128 bytes = one cache line) that
cover all requested addresses. The transaction cost is paid per sector touched, not per byte
useful.

efficiency = useful_bytes_requested / bytes_actually_moved

## Coalesced (best case)

32 threads, each loads one contiguous 4-byte `float`, thread `i` at `base + 4*i`, base 128-byte
aligned. All 128 bytes requested fall in exactly one 128-byte line → 4 sectors moved, 128 bytes
moved for 128 bytes requested. efficiency = 1.0.

```
addr(tid) = base + tid*4     // stride 1 word — coalesced
```

## Strided / misaligned (bad case)

Same warp, stride 32 words (128 bytes) between threads — e.g. each thread reads column `col` of a
row-major matrix with row pitch 32 floats (`a[tid][col]`, tid indexing the row). Each thread's
4 bytes lands in its own 128-byte line → up to 32 separate sector-groups touched.

bytes_moved = 32 * 32 bytes (min sector granularity, best case one sector/thread) = 1024 bytes
useful = 32 * 4 = 128 bytes
efficiency = 128/1024 = 12.5%, and with 128-byte-line granularity instead of 32-byte sectors it's
128 bytes requested / 32*128 = 4096 bytes moved = 3.1%.

Effective bandwidth scales down by the same factor: on A100 (2039 GB/s peak), a fully
non-coalesced stride-32 access pattern delivers roughly 2039 * 0.031 ≈ 63 GB/s of useful
throughput.

## Misalignment

Even with stride-1 access, if `base` is not 128-byte aligned the warp's 128 bytes straddle two
cache lines, costing 2 line-fetches (256 bytes moved) instead of 1 → efficiency drops to 50%. This
is why row pitches in tiled kernels are padded to 128-byte (32-float or 16-double) boundaries.

## AoS vs SoA

Array-of-structs (`struct{float x,y,z;} p[N]`, warp reads `p[tid].x`) strides by `sizeof(struct)`
per thread — non-unit stride, poor coalescing (12 bytes stride for a 3-float struct, none of the
nice power-of-two alignment). Struct-of-arrays (`float x[N], y[N], z[N]`, warp reads `x[tid]`) is
unit-stride and fully coalesced. Prefer SoA for any warp-parallel access pattern.

## Shared memory as a staging buffer

Global loads are coalesced into shared memory once per tile, then re-read from shared memory
(bank-addressed, not sector-addressed — see `shared-memory-and-bank-conflicts.md`) many times per
tile. This is the mechanism behind tiled GEMM, transpose, and convolution kernels: pay the global
memory coalescing cost once, amortize over BK or BM*BN reuses.

## L2 residency

Working sets that fit in L2 (40 MB on A100) avoid HBM entirely on reuse — e.g. a 4096x4096 fp32
matrix is 64 MB, larger than L2, but a 2048x2048 tile (16 MB) fits with room for double buffering.
`cudaMemAdvise`/L2 persistence hints (`cudaAccessPolicyWindow`) let a kernel pin a working set in
L2 across grid launches for iterative algorithms (e.g. attention KV cache reused every decode
step).
