# GPU execution model

## Hierarchy

Grid → thread blocks (CTAs) → warps → threads. A kernel launch `kernel<<<grid, block>>>(...)`
specifies grid dim (blocks) and block dim (threads/block). The hardware schedules blocks onto SMs;
each block stays resident on one SM for its entire lifetime (no migration).

Warp = 32 threads, the actual unit of scheduling and execution (SIMT: single instruction, multiple
thread). A block of 256 threads = 8 warps. All 32 lanes of a warp execute the same instruction each
cycle, on different data; a warp scheduler issues one instruction per warp per available cycle.

## SM resources (Ampere A100 / Hopper H100 as reference)

| | A100 | H100 SXM5 |
|---|---|---|
| SMs | 108 | 132 |
| FP32 cores/SM | 64 | 128 |
| Tensor cores/SM | 4 (3rd gen) | 4 (4th gen) |
| Warp schedulers/SM | 4 | 4 |
| Register file/SM | 65536 x 32-bit (256 KB) | 65536 x 32-bit (256 KB) |
| Max regs/thread | 255 | 255 |
| Max threads/SM | 2048 (64 warps) | 2048 (64 warps) |
| Max blocks/SM | 32 | 32 |
| Max threads/block | 1024 | 1024 |
| Shared mem/SM (max) | 164 KB (opt-in, 8 KB reserved) | 228 KB |
| Boost clock | ~1.41 GHz | ~1.98 GHz |

Each of the 4 warp schedulers per SM owns a disjoint subset of resident warps and issues 1
instruction/cycle to its FP32/INT32/tensor/LSU pipes when a warp has a ready instruction (no
stall on operand dependency, memory, or barrier).

## Occupancy

occupancy = active_warps_per_SM / max_warps_per_SM (64 on A100/H100)

Three independent caps determine active_warps_per_SM, and the binding one wins (min of all three):

1. **Registers**: blocks_by_reg = floor(regfile_size / (regs_per_thread * threads_per_block))
2. **Shared memory**: blocks_by_smem = floor(smem_per_SM / smem_per_block)
3. **Threads**: blocks_by_threads = floor(2048 / threads_per_block), also capped at 32 blocks/SM

resident_blocks = min(blocks_by_reg, blocks_by_smem, blocks_by_threads, 32)
active_warps = resident_blocks * (threads_per_block / 32)

### Worked example

Block = 256 threads (8 warps), 64 registers/thread, 8 KB shared mem/block, A100.

- blocks_by_reg = floor(65536 / (64*256)) = floor(65536/16384) = 4
- blocks_by_smem = floor(164 KB / 8 KB) = 20 (not binding)
- blocks_by_threads = floor(2048/256) = 8 (not binding)
- resident_blocks = min(4, 20, 8, 32) = 4
- active_warps = 4 * 8 = 32 → occupancy = 32/64 = 50%, register-limited.

Dropping to 32 regs/thread removes the register cap (blocks_by_reg = 8), raising occupancy to
8*8=64 warps = 100%. This is the standard `--maxrregcount` / `__launch_bounds__` tradeoff: fewer
registers per thread buys more resident warps at the cost of more register spills to local memory
if the kernel actually needed them.

## Divergence

Threads within a warp share one instruction stream. An `if/else` where lanes disagree executes
both paths serially with the disagreeing lanes predicated off (active mask), so worst case a
32-way divergent branch costs 32x the straight-line instruction count for that region. Divergence
across warps (different warps taking different branches) is free — each warp is independent.

## Latency hiding

An SM hides instruction and memory latency by switching to another ready warp every cycle instead
of stalling — zero-overhead warp switch, no register save/restore since each warp owns disjoint
registers. This is why occupancy matters: more resident warps → more chances the scheduler finds
one that isn't waiting on a ~400-800 cycle global load or ~4-20 cycle arithmetic dependency.
