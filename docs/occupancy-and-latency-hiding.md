# Occupancy and latency hiding

## Why occupancy exists

An SM hides stalls (memory latency, pipeline dependency latency) by switching to a different
ready warp every cycle — zero-cost context switch since every resident warp owns disjoint
registers. More resident warps = more chances one of them has an issuable instruction on any
given cycle. Occupancy (active_warps / max_warps_per_SM, max = 64 on A100/H100) is a proxy for
this, not the goal itself.

## Latency figures (A100-class)

| Operation | Latency |
|---|---|
| FP32 FMA (dependent chain) | ~4 cycles issue-to-use pipeline, extends to ~15-20 cycles observed due to pipe depth |
| Shared memory access | ~20-30 cycles |
| L2 hit | ~200 cycles |
| Global memory (HBM) miss | ~400-800 cycles |

## Little's law for warp scheduling

To keep an execution pipe fully utilized despite latency L (cycles) on operations issued at rate
1 per cycle per scheduler, the number of independent instructions in flight must reach L. Each
warp contributes at most 1 in-flight instruction of a given type at a time (scalar per-thread
model), so:

warps_needed ≈ L / (cycles between independent issues per warp)

### Worked example

Global load latency L ≈ 800 cycles. If each warp's next instruction depends on the load result
(no independent work to overlap), the warp stalls the full 800 cycles unless other warps fill the
gap. With 4 warp schedulers/SM issuing 1 instruction/cycle each when possible, keeping all 4 busy
across an 800-cycle stall needs roughly 800 warps of *aggregate* independent instruction supply
across the SM — but only 64 warps can be resident (2048 threads / 32). In practice full latency
hiding for a single dependent global load isn't reached by occupancy alone at typical values;
it's supplemented by:

- **Multiple loads in flight per warp** (unrolling, prefetching next tile while computing current)
- **ILP**: each thread doing independent work (e.g. accumulating 4 output elements) so a warp has
  4x the independent instructions per stall cycle, cutting warps_needed by 4x.

## Occupancy vs ILP tradeoff

Two ways to reach the same latency-hiding target:

- High occupancy, low ILP: many warps (e.g. 64, from small register/shared-mem footprint per
  thread), each doing scalar work — relies on warp-count alone to fill stalls.
- Low occupancy, high ILP: few warps (e.g. 16, occupancy 25%, from register-heavy per-thread
  tiling as in GEMM register blocking), each doing 4-8 independent FMAs per loop iteration —
  relies on per-warp instruction-level parallelism to fill stalls.

Register-blocked GEMM and FlashAttention kernels deliberately run at 25-50% occupancy because
each thread holds many accumulator registers (TM*TN in GEMM), trading occupancy for the higher
arithmetic intensity per shared-memory access described in `tiling-and-gemm.md`. Memory-bound
kernels (elementwise ops, reductions with light per-thread work) benefit more directly from raw
occupancy since they have little ILP to exploit.

## Occupancy calculator inputs

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` / Nsight Compute's occupancy view take
(registers/thread, shared mem/block, threads/block) and return achieved occupancy against the
three caps from `gpu-execution-model.md`. Nsight further reports "achieved occupancy" (measured,
accounts for tail effects — the last wave of blocks in a grid not fully filling the SM) versus
"theoretical occupancy" (the static calculation) — a kernel with a grid of, say, 109 blocks on
a 108-SM device leaves one SM's second wave running alone at low occupancy for its final block,
dragging down achieved occupancy even though theoretical occupancy is high.

## Diagnostic order

1. Check achieved occupancy in Nsight Compute. If low relative to theoretical, look for tail
   effects, load imbalance, or excessive `__syncthreads()` stalls.
2. If occupancy is already near the register/shared-mem-derived cap and the kernel is still
   latency-bound, increase ILP (unroll, multi-element-per-thread) rather than chasing more
   occupancy headroom that doesn't exist.
3. If occupancy is capped by shared memory and the kernel is memory-bound, reducing shared-mem
   footprint per block (smaller tiles) to admit more resident blocks trades tile-reuse (arithmetic
   intensity) for occupancy — check which one is the actual bottleneck via the roofline model
   before making this tradeoff.
