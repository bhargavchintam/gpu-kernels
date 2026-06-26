# Shared memory and bank conflicts

## Layout

Shared memory is banked: 32 banks (one per lane in a warp), each bank 4 bytes wide, successive
4-byte words assigned to successive banks and wrapping every 128 bytes (32 banks * 4 B).

bank(addr) = (addr / 4) mod 32     // for the default 4-byte access mode

Each bank services one request per cycle. A warp's 32 shared-memory accesses complete in a single
cycle iff no two lanes hit the same bank with different addresses (or, on Volta+, take the
degraded path serialized by conflict degree).

## Conflict degree

For a warp accessing `smem[tid * s]` with word-stride `s`:

conflict_degree = 32 / gcd(s, 32)

- `s = 1`: gcd(1,32)=1 → degree 1, no conflict, full bandwidth.
- `s = 2`: gcd(2,32)=2 → degree 16, 16-way conflict.
- `s = 32`: gcd(32,32)=32 → degree 32, worst case, fully serialized.
- `s = 33`: gcd(33,32)=1 → degree 1, conflict-free (odd/coprime stride trick).

A conflicted access is replayed conflict_degree times, each replay consuming a cycle on that
pipe — an N-way conflict costs N cycles instead of 1, i.e. throughput divides by N.

## Broadcast (not a conflict)

If every active lane in the warp reads the exact same address, hardware broadcasts the single
value to all lanes in one cycle regardless of which bank it's in. Conflicts only arise from
*different* addresses mapping to the *same* bank.

## Bandwidth

Peak shared-mem bandwidth per SM = 32 banks * 4 bytes * clock.

A100: 32*4*1.41 GHz ≈ 180.5 GB/s/SM * 108 SMs ≈ 19.5 TB/s aggregate.
H100: 32*4*1.98 GHz ≈ 253.4 GB/s/SM * 132 SMs ≈ 33.4 TB/s aggregate.

An N-way conflict divides realized bandwidth by N on that access.

## Classic case: matrix transpose tile

Naive shared-mem tile for a 32x32 block transpose:

```
__shared__ float tile[32][32];
tile[ty][tx] = in[row*N + col];      // write: coalesced global read, stride-1 write to smem
__syncthreads();
out[...] = tile[tx][ty];             // read: column access
```

The read `tile[tx][ty]` fixes `ty` per-thread-column and varies `tx` across the warp (tx =
threadIdx.x = lane id here in the transposed indexing), so consecutive lanes read addresses
`tx*32*4 + ty*4` — word-stride between lanes is 32 (row pitch), giving `gcd(32,32)=32` → 32-way
conflict, worst case.

Fix: pad the row pitch so the stride is no longer a multiple of 32.

```
__shared__ float tile[32][33];       // pad to 33 words/row
```

Now the effective word-stride for the column read is 33: `gcd(33,32)=1` → conflict-free. Cost:
32*33*4 = 4224 bytes of shared mem instead of 4096, a ~3% capacity overhead to remove a 32x
bandwidth penalty.

## Wider access modes

Shared memory also supports an 8-byte banking mode (`cudaSharedMemBankSizeEightByte`) useful for
`double`-typed tiles, halving the effective bank count interactions for 8-byte words — same
gcd-based analysis applies with 16 banks of 8 bytes instead of 32 banks of 4 bytes.

## Practical rule

Any kernel indexing shared memory with the same stride as the tile's leading dimension (a very
common bug in tiled GEMM, transpose, and stencil kernels) should pad the leading dimension by one
element to break power-of-two stride alignment against the 32-bank modulus.
