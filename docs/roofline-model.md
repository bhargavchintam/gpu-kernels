# Roofline model

## Definition

attainable_FLOPs = min(peak_FLOPs, peak_BW * arithmetic_intensity)

arithmetic_intensity (AI) = FLOPs / bytes_moved_from_HBM, a property of the algorithm/kernel and
its tiling, not the hardware. peak_FLOPs and peak_BW are hardware constants. Plotted log-log
(AI on x, attainable FLOP/s on y), this gives a flat "compute roof" and a rising "memory roof"
that intersect at the ridge point.

ridge_point = peak_FLOPs / peak_BW    (units: FLOP/byte)

Left of the ridge point: memory-bound (attainable throughput scales linearly with AI). Right of
the ridge point: compute-bound (attainable throughput is capped at peak_FLOPs regardless of AI).

## Reference hardware

| GPU | Peak FLOP/s | Peak BW | Ridge point |
|---|---|---|---|
| A100 fp32 (CUDA core) | 19.5 TFLOPS | 2039 GB/s | 9.6 FLOP/byte |
| A100 fp16 tensor (dense) | 312 TFLOPS | 2039 GB/s | 153 FLOP/byte |
| A100 fp16 tensor (2:4 sparse) | 624 TFLOPS | 2039 GB/s | 306 FLOP/byte |
| H100 SXM fp32 (CUDA core) | 67 TFLOPS | 3350 GB/s | 20 FLOP/byte |
| H100 SXM fp16 tensor (dense) | 989 TFLOPS | 3350 GB/s | 295 FLOP/byte |

Tensor cores push the ridge point up by an order of magnitude relative to CUDA cores — a kernel
that was comfortably compute-bound on fp32 cores can become memory-bound once ported to tensor
cores, because peak_FLOPs jumped 16-50x while peak_BW didn't move.

## Worked example: GEMM

M=N=K=4096, fp16 inputs, fp32 accumulate, on A100 tensor cores.

FLOPs = 2*M*N*K = 2*4096^3 ≈ 1.374e11 (137.4 GFLOP)

Memory traffic with good tiling (each element of A, B read once from HBM given a shared-mem tile
large enough to amortize reuse across the whole matrix; C written once):
bytes ≈ (M*K + K*N)*2 (fp16) + M*N*4 (fp32 out) = (4096*4096*2)*2 + 4096*4096*4
      ≈ 2*33.6MB + 67.1MB ≈ 134.4 MB

AI = 1.374e11 / 1.344e8 ≈ 1022 FLOP/byte

1022 >> 153 (A100 fp16 tensor ridge point) → deep in the compute-bound region. Attainable
throughput ≈ peak_FLOPs = 312 TFLOPS (modulo achieving near-100% tensor core utilization via
correct tiling and occupancy — see `tiling-and-gemm.md`).

## Worked example: attention (with vs without FlashAttention)

N=2048 (seq len), d=128 (head dim), fp16, one head, forward pass.

FLOPs (QK^T + P@V) = 2 * (2*N^2*d) = 4*N^2*d = 4*2048^2*128 ≈ 2.147e9 (2.15 GFLOP)

**Without FlashAttention** (materialize S, P as N x N fp16 matrices, 2 matrices, read+write each):
bytes ≈ 2*(N^2*2 bytes) reads + 2*(N^2*2 bytes) writes = 4*N^2*2 = 4*2048^2*2 ≈ 3.36e7 (33.6 MB)
(plus O(Nd) for Q,K,V,O, negligible next to the N^2 term)
AI ≈ 2.147e9 / 3.36e7 ≈ 64 FLOP/byte

64 FLOP/byte is below the A100 fp16 tensor ridge point (153) → memory-bound; attainable throughput
≈ peak_BW * AI = 2039 * 64 ≈ 130 TFLOPS, well under the 312 TFLOPS the hardware can do — the N x N
intermediate traffic is the bottleneck, not tensor core throughput.

**With FlashAttention** (S, P never touch HBM, only Q,K,V,O plus small per-block stats do):
bytes ≈ 4*N*d*2 bytes (Q,K,V,O) = 4*2048*128*2 ≈ 2.097e6 (2.1 MB)
AI ≈ 2.147e9 / 2.1e6 ≈ 1023 FLOP/byte

1023 >> 153 → compute-bound, attainable throughput ≈ 312 TFLOPS (tensor-core peak), a ~2.4x
theoretical throughput improvement purely from removing the N^2 memory traffic, matching the
qualitative speedups reported for FlashAttention at this problem size.

## Using the model

1. Compute AI for the kernel's actual memory traffic (not the naive "each element read once"
   estimate — account for reuse from tiling/caching, and any spilled intermediates).
2. Compare against the ridge point for the target precision/instruction class (fp32 core vs
   tensor core ridge points differ by 10-30x on the same chip).
3. Left of ridge → optimize for AI: better tiling, larger blocks, fusing kernels to avoid writing
   intermediates to HBM (exactly the FlashAttention move).
4. Right of ridge → optimize for FLOP throughput: better instruction mix, higher tensor-core
   utilization, reduced overhead per useful FLOP (loop unrolling, avoiding predication).
5. A kernel can be memory-bound on one GPU and compute-bound on another if peak_BW and peak_FLOPs
   scale unevenly between generations (tensor core FLOPs have grown faster than HBM bandwidth
   across Ampere→Hopper→Blackwell, pushing ridge points up and more workloads left of the roof).
