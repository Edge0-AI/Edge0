# V100 / Swift IQ2_XS / 32k / --kv int8: the KV read path

Test bed: 2x Tesla V100-SXM2-16GB (SM 7.0), CUDA 12.8, one card. No full-engine run; kernel microbench only.

## 1. Audit - where int8 KV is decoded, and whether it is vectorized

The default decode path is the fused split-K attention (`g_fast_attn = true`, `src/core/layer.cpp:42`, chosen at
`layer.cpp:950-952`), which reads the pools directly - no gather:

- K decode: `load8_q8`, `src/kernels/cuda/qsa_decode_attn.cu:46-54`. One lane reads 8 int8 as a `uint2` (8 B) and
  one fp16 scale per 64-group, then scalar `(float)c * sc` x8 (I2F + FMUL each). The *load* is vectorized, the
  *decode* is scalar.
- V decode: `attn_chunk_kernel`, `qsa_decode_attn.cu:158-179`. Thread `t` owns dimension `t` for all 12 heads:
  one int8 load + one fp16 scale + one I2F+FMUL per cell, then 12 FMAs. 1 byte/thread/cell, coalesced to 256 B.
- MTP draft/verify use the same kernel by batch: `src/core/mtp.cpp:503`, `src/core/verify.cpp:647`
  (`qsa_decode_attn_batch`).

The non-fast fallback gathers int8 into an fp16 scratch first, then attends over it:

- `kv_gather_q8_kernel`, `src/kernels/cuda/kv_q8.cu:69-99`: `char4` load (4 int8) + 2 fp16 scales, 8 scalar
  converts, `ushort4` fp16 store. 4 values/thread.
- `qsa_attend_kernel` then reads the fp16 scratch: `qsa.cu:462-467` (K dot), `qsa.cu:502-505` (V), 2 B/value.

The KV-streaming copy is already wide: `copy_kernel`, `src/kernels/cuda/kv_stream.cu:161-171`, `uint4` (16 B).

## 2. Vectorization

The gather (`kv_gather_q8_kernel`) was 4 values/thread: two `char4` (4 B) loads + two `ushort4` (8 B) stores.  The
experimental build (`-DSTRATA_EXPERIMENTAL_SM60=ON`, Pascal/Volta) adds an 8-wide `kv_gather_q8_wide_kernel`: one
`uint2` (8 B) load of codes, one `uint4` (16 B) store of the dequantized fp16, one scale per thread instead of two.
The 8 values lie in one 64-value group, and the arithmetic is unchanged (`(float) code * scale -> f16_from_f32`,
packed two halfs per 32-bit word), so the output is byte-identical.  The ready-made engine keeps exactly the
4-wide kernel.

`kv_q8_parity`: OK, worst INT8-vs-FP16 error 0.590 steps (bitwise codes, scales and gather; unchanged).

Microbench (`kv_q8_parity --bench`, 2,051 cells / 2 KV heads / 256 dims, 32k pool, random cells, V100):

| | us/call | GB/s |
|---|---|---|
| before (4-wide) | 15.85 | 401 |
| after (8-wide)  | 14.05 | 452 |

+11.4% on the gather. The default decode path (`g_fast_attn`) does not gather; its decode (`load8_q8`,
`qsa_decode_attn.cu:46-54`) already reads 8 int8 with one `uint2` and is arithmetic-bound (8 I2F + 8 FMUL per
cell), which `__byte_perm`/half2 cannot improve while staying bit-exact. The gather is what the non-fast,
streaming and hybrid paths use.

## 3. int8 vs q4_0 vs k8v4 at 32k

TBD
