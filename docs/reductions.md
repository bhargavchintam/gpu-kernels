# Reduction patterns

## Tree reduction in shared memory

Classic pattern for reducing `n` elements per block (n = threads/block, e.g. 256):

```
__shared__ float s[256];
s[tid] = in[gid];
__syncthreads();
for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
}
if (tid == 0) out[bid] = s[0];
```

log2(256) = 8 steps, each with a `__syncthreads()` barrier (~20-30 cycle overhead) plus a
shared-mem round trip. Sequential addressing (`tid < stride`, `s[tid]+=s[tid+stride]`) keeps the
active-thread set contiguous each round, avoiding both warp divergence within active warps and
bank conflicts (stride is always a power of two ≤128, but active lanes read contiguous addresses
so no cross-lane same-bank collision on the read side).

Naive addressing (`if (tid % (2*stride) == 0)`) is worse: divergent modulo test every step, and
half the warps do zero useful work but still get scheduled.

## Warp-level: shuffle reduction

Since Kepler, `__shfl_down_sync` exchanges a register value directly between lanes of a warp — no
shared memory, no `__syncthreads()`, no bank-conflict risk.

```
unsigned mask = 0xffffffff;
for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(mask, val, off);
// lane 0 now holds the warp's sum
```

log2(32) = 5 steps to reduce a full warp, each a single instruction (register-to-register, latency
a few cycles, no synchronization barrier since a warp is always in lockstep at the instruction
issue level — `_sync` variants still require the mask for correctness under independent thread
scheduling, Volta+).

## Block-wide: two-level shuffle

For a block of 1024 threads (32 warps):

1. Each warp reduces its 32 values via shuffle → 5 steps, 32 partial sums (one per warp).
2. Each warp's lane 0 writes its partial sum to `shared[warp_id]` (32 shared-mem slots).
3. `__syncthreads()` once.
4. The first warp loads the 32 partials into shared and shuffle-reduces again → 5 more steps.

Total: 10 shuffle steps + 1 shared-mem write/read + 1 barrier, vs. a pure shared-mem tree
reduction's 10 barriers (log2(1024)=10) each paying the ~20-30 cycle sync cost. The two-level
shuffle approach removes 9 of the 10 barriers.

## Numeric comparison (1024-element block reduction)

| Method | Sync barriers | Shared-mem round trips | Shuffle ops |
|---|---|---|---|
| Naive tree (divergent) | 10 | 10 | 0 |
| Tree, sequential addressing | 10 | 10 | 0 |
| Two-level shuffle | 1 | 1 | 10 |

Shuffle instruction latency is on the order of the ALU pipe (~single-digit to ~20 cycles
depending on architecture), versus a barrier's fixed overhead plus the shared-memory load/store
latency (~20-30 cycles each) — the two-level scheme cuts total reduction latency roughly 3-5x on
large blocks.

## Grid-wide reduction

A single block reduction only handles one block's worth of data. For a full-array reduction across
`G` blocks:

- **Two-pass**: kernel 1 writes G partial sums to global memory, kernel 2 (single block) reduces
  the G values. Requires a kernel-launch round trip (~5-10 us).
- **Atomics**: each block's lane-0 does `atomicAdd(&out[0], block_sum)` — no second launch, but
  contends on one address; fine when G is small (hundreds) relative to atomic throughput.
- **Cooperative groups grid sync** (`cg::grid_group`, `this_grid().sync()`): all blocks
  synchronize within one launch, avoiding the second kernel launch, but requires
  `cudaLaunchCooperativeKernel` and caps grid size to the device's concurrent-block capacity.

## Numerically-stable variants

Sum reduction as above is exact-order-dependent (floating point non-associativity); for
mean/variance use Welford's online algorithm merged pairwise (same tree/shuffle structure, merging
`(mean, M2, count)` triples) to avoid catastrophic cancellation from naive sum-of-squares.
