# FlashAttention: online softmax and IO complexity

## Standard attention

O = softmax(QK^T / sqrt(d)) V, with Q,K,V ∈ R^(N x d). Materializing S = QK^T (N x N) and
P = softmax(S) requires writing and reading two N x N matrices to/from HBM in addition to Q,K,V,O.

HBM accesses (standard): Θ(Nd + N^2). For N=4096, d=128: the N^2 term (16.8M elements) dominates
Nd (524K elements) by ~32x — the intermediate matrices, not the inputs/outputs, dominate memory
traffic.

## Online softmax recurrence

Softmax is shift-invariant: softmax(x) = softmax(x - c) for any constant c, which lets a running
max be maintained without a second pass over the data. Process K,V in blocks of size Bc; for each
Q-block (size Br) maintain running max m_i, running sum l_i, running (unnormalized) output O_i,
initialized m_i = -inf, l_i = 0, O_i = 0.

For each K,V block j, compute local scores and merge:

```
S_ij   = Q_i @ K_j^T / sqrt(d)          // Br x Bc
m_ij   = rowmax(S_ij)                    // Br
m_new  = max(m_i, m_ij)                  // Br
P_ij   = exp(S_ij - m_new)               // Br x Bc, broadcast subtract per row
correction = exp(m_i - m_new)            // Br, rescales old accumulators
l_new  = correction * l_i + rowsum(P_ij)
O_new  = correction * O_i + P_ij @ V_j
m_i, l_i, O_i = m_new, l_new, O_new
```

After the last K,V block: O_i = O_i / l_i (final normalize, one division per row instead of per
element of the never-materialized N x N matrix).

## Correctness argument

`correction = exp(m_old - m_new)` exactly rescales the previous (wrongly-normalized-so-far)
accumulators to be consistent with the new running max, so the recurrence is algebraically
identical to computing softmax over the full concatenated row and multiplying by V — no
approximation, this is exact softmax, just computed incrementally. This is the same trick as
numerically-stable log-sum-exp, extended to also accumulate a weighted matrix product (P@V)
alongside the normalizer.

## IO complexity

FlashAttention never writes S or P to HBM — each (Q-block, K/V-block) pair's S_ij, P_ij live
only in on-chip SRAM (registers/shared memory) for the duration of that block's processing.

HBM accesses (FlashAttention): Θ(N^2 d^2 / M), where M = SRAM size (shared memory capacity).
Block size is chosen Bc = M/(4d) (so a Bc x d block of K, V, and the Br x Bc score tile fit in
SRAM with room for Q_i and O_i). For d ≤ M ≤ Nd (the realistic regime), this is asymptotically
fewer HBM accesses than standard attention's Θ(Nd + N^2) by roughly a factor of M/d.

## Worked numbers

N=4096, d=128, A100 shared memory M ≈ 164 KB/SM ÷ 4 bytes (fp32 working set) ≈ 41000 elements.
Bc ≈ M/(4d) = 41000/512 ≈ 80; frameworks round to powers of two / hardware-friendly sizes (64 or
128 in practice, accounting for double buffering and Q/O tile coexistence, which eats into the
naive M/(4d) budget).

Standard attention HBM traffic ≈ 2*N^2 (S read+write) + 2*N^2 (P read+write) in fp16 ≈
4*4096^2*2 bytes ≈ 134 MB, per attention head, per forward pass.
FlashAttention HBM traffic ≈ O(Nd) for Q,K,V,O ≈ 4*4096*128*2 bytes ≈ 4.2 MB.
Ratio ≈ 32x less HBM traffic for this (N,d), consistent with the Θ(N^2d^2/M) vs Θ(N^2) scaling
once M >> d^2 doesn't hold — the exact constant depends on block size, but the qualitative result
(traffic collapses from O(N^2) to O(Nd) or O(N^2/√M)-ish depending on regime) is the source of the
measured speedup.

## Compute-memory tradeoff

FlashAttention recomputes S_ij and P_ij in the backward pass from stored (Q,K,V,O,m,l) rather than
storing the N x N attention matrix from the forward pass. This trades additional FLOPs (cheap,
GPUs are FLOP-rich) for HBM bandwidth (expensive, the binding constraint per the roofline model —
see `roofline-model.md`): recompute cost is O(N^2 d) extra FLOPs, negligible next to the
bandwidth saved by never storing O(N^2) intermediates.

Reported results (original paper, A100): ~2-4x wall-clock speedup on attention itself at
sequence lengths 512-4096, 15% end-to-end training speedup on BERT-large, ~3x on GPT-2, with
memory use linear in N instead of quadratic — enabling longer context lengths at the same memory
budget.
