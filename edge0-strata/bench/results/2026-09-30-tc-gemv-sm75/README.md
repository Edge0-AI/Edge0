# BF16 GEMV on Turing tensor cores (sm_75): measured negative

i7-8700K + RTX 2070 8 GB (sm_75, 36 SMs), engine 0.1.27 tree (`bf16_gemv.cu` is unchanged through 0.1.30),
qwen3.8-flash-next (~125.7B MoE, GGUF-verified) Q2_0 in production. Isolated benchmark only -
[`bench_tc_gemv.cu`](bench_tc_gemv.cu) touches nothing in the inference path. Question: the QSA prompt attention
already rides the Turing tensor cores (`mma.sync.m16n8k8`, the #270 path); can the decode GEMVs do the same?

## Setup

- Baseline: the production `bf16_gemv` warp kernel, transcribed verbatim
- TC variant: `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` on BF16->FP16 (the sm_75 MMA has native f16 only).
  BF16->FP16 conversion is exact (0 / 2,000,000 values inexact - bf16's 7-bit mantissa fits in fp16's 10 bits); the
  A-fragment packing is the same one #270 validated (`a0` = row `gid`, k columns `2t..2t+1`; broadcast makes
  `a0 == a1 == {x[2t], x[2t+1]}`)
- Three TC variants to cover the conversion overhead from every angle: W preconverted to fp16 once at load
  (`pc`), W converted in-register every call (`in`), and a fused ks=1 form that writes the output directly and skips
  the reduce kernel (`fused`)
- The k-split across blocks (`ks`) is swept over all values that divide n_in, and the best is reported - the TC
  kernel gets its best occupancy
- Parity against an fp64 host reference; 2000 iterations x 3 reps, warmed CUDA events

## Results (best TC config per shape)

| Shape (decode, n_in = 2560, BF16 W) | Baseline us | Best TC us | Speedup | Config |
| --- | ---: | ---: | ---: | --- |
| ssm_alpha/beta [2560, 48] | 8.73 | 11.32 | 0.77x | pc, ks=16 |
| indexer.k_proj [2560, 128] | 7.14 | 12.09 | 0.59x | pc, ks=16 |
| indexer.q_proj / router [2560, 512] | 9.48 | 15.78 | 0.60x | pc, ks=4 |
| ple_value [2560, 2560] | 35.08 | 77.73 | 0.45x | pc, ks=40 |

- Numerics pass: TC max error vs fp64 reference 1.7e-6 .. 7.2e-6 (fp16-accumulation level), baseline
  0.7e-6 .. 2.3e-6. The TC variant computes the same thing, it is just slower.
- The warp kernel is already bandwidth-bound at 276-374 GB/s effective W bandwidth (374 GB/s on the 13 MB
  [2560, 2560] shape); the TC path reaches only 166-169 GB/s (4-byte lane loads, conversion instructions, in-tile
  launch overhead).
- A single-token GEMV broadcasts one token across the whole m16 MMA tile, so 15/16 of the MMA throughput is thrown
  away - but the baseline is not compute-bound, so the wasted throughput buys nothing. The k-split sweep and the
  fused variant do not recover it.
- One-time W conversion cost is negligible (2 ms for 13 MB), so preconverting W for the TC path does not help.

## How to rerun

```
nvcc -arch=sm_75 -O3 -o bench_tc_gemv.exe bench_tc_gemv.cu
./bench_tc_gemv.exe
```

Prints the table above, per-shape latency/parity and the conversion exactness scan. On a newer card, adjust
`-arch`; the packed-A layout comment in the source documents the fragment mapping it relies on.

## Conclusion

**Keep the warp kernel. Tensor cores are not a promising direction for the decode GEMVs on sm_75**, and this was a
measured loss (1.3-2.2x slower), not a failed attempt to optimize. The Turing MMA stays worthwhile where the input
really has m16 rows of work: the QSA prompt-attention tiles, which is where #270 uses it. Someone looking to
re-evaluate on future hardware (a card with native bf16 or denser GEMV batching) can rerun this instead of
rebuilding it.