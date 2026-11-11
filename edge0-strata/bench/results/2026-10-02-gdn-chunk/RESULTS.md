# v100/gdn-chunk: chunked GDN prompt recurrence and Volta recurrence variants (2026-10-02)

Measured on the V100-SXM2-16GB x2 (one card, CUDA 12.8, driver 570.x), build:
`-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`, Release.  `bench/gdn_chunk_bench`
(CUDA events, real GDN geometry S=128 / 16 q/k heads / 48 v heads / C=10240, one `gdn_recurrence`
call per side including the out norm, 5-20 iterations).

## GDN prompt recurrence kernel, one layer, one prompt chunk

| T     | pipe (old) | chain-split (default now) | sync-free warp | chunked FMA | chunked wmma |
|-------|-----------:|--------------------------:|---------------:|------------:|-------------:|
| 1024  | 1.056 ms   | 1.042 ms (+1%)            | 2.712 ms       | 6.67 ms     | 4.57 ms      |
| 4096  | 3.836 ms   | 3.600 ms (+6%)            | 9.34 ms        | 25.0 ms     | 17.7 ms      |
| 8192  | 7.703 ms   | 7.190 ms (+7%)            | 18.7 ms        | 50.2 ms     | 35.4 ms      |

- pipe = `gdn_rec_cols_pipe_kernel` (the old path, bit-exact reference).
- chain-split = `gdn_rec_cols_pipe_fast_kernel`: same structure and loads, the k^T W / q^T W
  accumulators split 4 ways (FP32-level, NOT bit-exact; `gdn_chunk_parity` bounds it).
- sync-free warp = `gdn_rec_cols_warp_kernel`: warp-shuffle reductions, no __syncthreads,
  BIT-EXACT vs pipe (asserted) - but 2.6x SLOWER: its per-thread row loads cost more than the
  barriers it removes.  Kept as the bitwise control.
- chunked = the FlashQLA-style chunked recurrence (cumsum / kkt / fwd, 64-token chunks).  Correct
  (parity below) but 4-6x SLOWER than the recurrence as implemented: with 128 threads and the
  ~84 KB tile staging a CTA only feeds ~4 warps to the tensor cores, and every phase is
  latency-bound (measured ablation: staging+Gram+U GEMMs alone = 2.4 ms of the wmma kernel's
  4.6 ms at T=1024).  Opt-in only (STRATA_GDN_CHUNK=1 wmma / 2 FMA).

## Error bounds (`gdn_chunk_parity`, vs a double-precision transcription of the recurrence)

| path                        | y max abs | state max abs | note |
|-----------------------------|----------:|--------------:|------|
| pipelined recurrence        | 1.1e-06   | 3.0e-08       | the FP32 floor |
| chain-split recurrence      | 7.2e-07   | 3.0e-08       | FP32-level, not bit-exact |
| sync-free warp recurrence   | 0         | 0             | bit-exact vs the pipelined kernel |
| chunked FMA                 | 3.1e-06   | 1.2e-07       | FP32-level (reordered sums) |
| chunked wmma                | 2.3e-03   | 8.9e-05       | q/k/v and the state staged f32->f16, f32 accumulate |
