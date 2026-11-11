# RX 6900 XT (gfx1030): rocBLAS's kernel choice for the FP16 prompt GEMMs, and a solution table (2026-10-05)

On gfx103x the prompt path's dense GEMMs run through rocBLAS as FP16 in -> FP16 out (#835). rocBLAS picks the kernel
for each product from its own table by shape, and for some shapes that pick is far from the best kernel the same library
holds. The clearest case is the GDN input projection, N = 10240, K = 2560: below T = 1152 tokens rocBLAS's choice runs at
5.2-5.3 TFLOPS (n = 682 ... 1120, measured with `rocblas_gemm_ex` on this card, 8.1 ms at n = 799), at T = 1152 it jumps
to 38 TFLOPS (1.6 ms), and the best of the 237 solutions `rocblas_gemm_ex_get_solutions` lists for n = 799 runs it in
1.36 ms. It is not an alignment effect (n = 800 is as slow as 799). In the engine's own phase timing this is why the
"gdn" phase of a 799-token prompt (445 ms) was nearly as long as that of an 8,874-token one (1,345 ms).

`tools/hip/tune_rocblas.cpp` enumerates rocBLAS's solutions per shape and token bucket, times them, checks the fastest
against the default kernel's output and writes the winners as a table; `Gemm::f16_inplace` runs the table's solution for
a shape in place of rocBLAS's choice (`STRATA_ROCBLAS_TUNING`; docs/AMD_HIP.md, "rocBLAS table"). The table shipped,
`tools/hip/gfx1030-rocblas-5.6.0.8d1ae90e.txt`, is calibrated against rocBLAS 5.6.0.8d1ae90e (ROCm 10.0.0,
`/opt/rocm/lib`); its indices are valid for that build only, and the engine refuses any other.

## Rig

- AMD Radeon RX 6900 XT 16 GB (gfx1030), PCIe 4.0 x8, one card of two (`HIP_VISIBLE_DEVICES` as setup sets it); AMD
  Ryzen 5 5600X (6 cores, AVX2), 128 GB DDR4-3200; Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0.0 (HIP 7.15.26333).
  Note that a HIP build on this Ubuntu links Ubuntu's own `librocblas5` (a 5.1 build, package 7.1.0-1ubuntu4) unless
  `lib_dirs` (`/opt/rocm/lib`) puts ROCm's first: the engine runs below had `lib_dirs` set, as setup's configuration does.
  rocBLAS 5.1's default kernel for the shape above is as slow (8.1 ms at n = 799, measured too); its solution indices
  differ, so it would need its own table.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 shards, `--native`), MTP draft, `--kv int8 --kv-resident 32768
  --max-context 131072 --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --adapt-every 100000` (the last the
  #884 workaround; it does not touch the prompt path).
- Engine: `6f32ec0` (0.1.39) + #835 + this change, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
  -DSTRATA_PREFILL_MMQ=ON`. #849 is not in this build, so the `qsa attn` phase is the slow pre-#849 kernel here (the same
  in both arms).

## Table

`tune_rocblas --tuning-out ...` with the defaults: the engine's 14 FP16 dense shapes (ldy = N) at T = 256, 512, 768,
1024, 1536, 2048, 4096, 8192; 112 rows, 76 with a solution that beat the default by at least 5% and also ran at T-1,
T-37, T-63 and T/2+1 (a kernel can refuse a token count it was not measured at), `default` for the other 36.
[tune-summary.txt](tune-summary.txt) is the tool's per-case output (default ms, best solution, gain). The large gains
are at small T: N = 10240 K = 2560 at T = 256 ... 1024, 7.0-7.8x; N = 12288 K = 2560 at T = 256, 5.8x; N = 6144 K =
2560 at T = 256 / 512, 3.6-3.8x; N = 2560 K = 6144 at T = 256, 4.5x; N = 320 K = 10240 (`hc read`), 2.5-8.7x across the
buckets. At T >= 2048 most shapes keep the default (1.0-1.2x where a solution still wins).

## Method

One server per arm, the same binary; `STRATA_PREFILL_TIMING=1` (the engine's prompt-phase timing),
`STRATA_ROCBLAS_VERBOSE=1`; the on arm with `STRATA_ROCBLAS_TUNING` pointing at the directory holding the table. Four
prompts from four different offsets of one text (no prefix reuse between them), "Reply with one word", `max_tokens` 8,
temperature 0; the answer text compared between arms. The GPU timeline is the engine's `strata prefill timing` line; the
per-phase numbers are its phases ([timing-off.txt](timing-off.txt), [timing-on.txt](timing-on.txt)). One run per arm; the off
arm repeated an earlier run of the same prompts the same day (with the table's first version) to within 10 ms on every
prompt, and the on arm's gains repeated within 100 ms.

## Results

| prompt tokens (read) | arm | GPU timeline ms | gdn | hc read | wait copy | answer |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 806 (799) | default | 3,329 | 443 | 270 | 1,376 | Estimated |
| 806 (799) | **table** | **2,985** | 177 | 77 | 1,384 | Estimated |
| 502 (495) | default | 2,477 | 365 | 81 | 1,077 | Incomplete |
| 502 (495) | **table** | **2,139** | 146 | 52 | 1,075 | Incomplete |
| 1,115 (1,108) | default | 3,391 | 598 | 158 | 1,144 | Incomplete |
| 1,115 (1,108) | table | 3,395 | 201 | 99 | 1,637 | Incomplete |
| 8,881 (8,874) | default | 11,978 | 1,346 | 1,200 | 1,683 | Strata |
| 8,881 (8,874) | **table** | **11,459** | 1,128 | 898 | 1,670 | Strata |

- 799 tokens: -10.3%; 495 tokens: -13.6%; 8,874 tokens: -4.3% (the 682-token tail chunk's projections, and `hc read`
  at both chunks).
- 1,108 tokens: no change. Its GDN time fell by 397 ms and `wait copy` rose by 493 ms: that prompt is bound by the
  experts' PCIe transfer, and a faster GEMM only brings the GPU to the wait sooner. Which prompts are copy-bound depends
  on how many distinct experts they touch, not on their length alone.
- The answers are the same in all four pairs. The tuner checked each solution it kept against the default kernel's FP16
  output (relative L2 within 2e-3, no element beyond 4 FP16 ulps of the largest output, nothing written outside the
  N x T result); in the engine's run no solution was refused (0 fallbacks over 52 shape-and-T combinations served).
- Decode does not use these GEMMs; not measured here.

## Limits

One machine, one card, one model; one run per arm, four prompts. The table is for one rocBLAS build; another build (or
ROCm's 5.1 packaged by Ubuntu) needs `tune_rocblas` run against it. Prompts above ~1,150 tokens gain only through their
tail chunk and `hc read`, and copy-bound prompts gain nothing. Whether other gfx103x cards (gfx1031 / gfx1032) share
rocBLAS's gfx1030 kernel table was not checked; the engine's architecture check would refuse this table on them.
