# RX 6900 XT (gfx1030): the prompt path's 16-bit GEMMs in FP16 in and out (2026-10-04)

On gfx103x (RDNA2) the prompt path ran its 16-bit GEMMs as rocBLAS FP16-in / FP32-out (`Gemm::f16`, `Gemm::native`) and
BF16-in / FP32-out (`Gemm::bf16`). rocBLAS on gfx1030 has tuned kernels for FP16-in / **FP16-out** only; the other two
fall back to generic kernels about 6.6x slower. This change runs those GEMMs FP16 in and out on gfx103x, writing the FP16
result into the start of each FP32 row of Y and widening it there (no extra buffer), and has the kernels that produce the
BF16 GEMMs' activation images (`gr_*`, `to_bf16`) write FP16 instead. `STRATA_HIP_PROMPT_F16=0` is the old path, so every
number below is the same binary with and without that switch.

## Rig

- 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), each PCIe 4.0 x8, no P2P used. AMD Ryzen 5 5600X, 128 GB DDR4-3200.
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0.0 (HIP 7.15.26333, rocBLAS 5.6).
- Engine: `origin/main` at `6f32ec0` (0.1.39) with this change, `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON`.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 shards, `--native`), MTP draft layer on, `--kv int8 --kv-resident 32768
  --max-context 131072`, `--expert-cache auto`. Not one of setup's models (see the limits).

## Why: rocBLAS on gfx1030

A 7.3K-token prompt on one card (rocprofv3, 0.1.38): rocBLAS `Cijk_..._HSS_..._MT64x32x8` (FP16 in, FP32 out, a fallback
tile) was 54% of the prompt's 15.2 s of GPU time, the BF16 products (`BSS`) another ~15%. Microbenchmark, N = 10240,
T = 7313, K = 2560:

| rocBLAS GEMM on gfx1030 | TFLOPS |
| --- | ---: |
| FP16 in, FP32 out (HSS) | 5.6 |
| BF16 in, any out | ~5.3 |
| **FP16 in, FP16 out (HH, fp32 accumulate)** | **37.7** |

## Prompt speed

One card, cold server per run, the first message of each chat below read by the batched prompt path; the
server's own line `strata serve: prompt N tokens = 0 reused + N read in T ms`. Three runs per path (two of them the
distribution check's), all within 0.3%:

| Prompt tokens | old path (`STRATA_HIP_PROMPT_F16=0`) | FP16 | gain | read time, old -> FP16 |
| ---: | ---: | ---: | ---: | ---: |
| 9,427 | 439 tok/s | 744 tok/s | +69% | 21.5 s -> 12.7 s |
| 34,659 | 466 tok/s | 915 tok/s | +96% | 74.4 s -> 37.9 s |
| 105,811 | 461 tok/s | 926 tok/s | +101% | 229.5 s -> 114.3 s |

Decode is unchanged within the noise (44-48 tok/s on one card): it does not use these GEMMs.

Both cards, cold server per run, one 8,275- and one 33,586-token prompt of repository text (128 tokens answered):

| | old path | FP16 | gain |
| --- | ---: | ---: | ---: |
| `--layer-split auto` (25 / 23 layers), 8.3K | 444 tok/s | 833 tok/s | +88% |
| `--layer-split auto`, 33.6K | 707 tok/s | 1,356 tok/s | +92% |
| one card + the second as expert helper (`--expert-cache-device1 auto --remote-expert-opt`), 8.3K | 432 tok/s | 798 tok/s | +85% |
| the same, 33.6K | 454 tok/s | 887 tok/s | +95% |

Decode is the same within the noise in both setups (these short answers: 51-58 tok/s split, 36-42 helper).

## Against the FP32 route (#1006), same card (2026-10-06)

#1006 takes the same gfx103x GEMMs the other way: both inputs widened to FP32 and the product as SGEMM (exact widening,
FP32 accumulate, `STRATA_RDNA2_SGEMM=0` for the old path). Built at its `63a92c2` and run next to this change's build
(`6add1a7`, `main` + this) on one RX 6900 XT: the same model and arguments as above, three prompts from one corpus, cold
server per arm, temperature 0, 8 tokens answered, `STRATA_PREFILL_TIMING=1`. The old path here is #1006's branch with
its switch off. One run per cell; a second cold run of this change's arm repeated it within 1.3%.

| prompt tokens | old path | #1006 (SGEMM) | this change (FP16 HGEMM) |
| ---: | ---: | ---: | ---: |
| 8,368 | 412 tok/s | 622 tok/s (+51%) | 730 tok/s (+77%) |
| 32,813 | 455 tok/s | 742 tok/s (+63%) | 900 tok/s (+98%) |
| 104,412 | 456 tok/s | 750 tok/s (+65%) | 904 tok/s (+98%) |

The 32,813-token prompt's GPU timeline, the phases these GEMMs sit in (old / SGEMM / FP16): gdn 22.0 -> 6.2 -> 4.0 s,
qsa proj 8.6 -> 3.2 -> 2.1 s, hc read 8.1 -> 6.1 -> 4.4 s, router+shared 4.0 -> 2.5 -> 0.7 s; `wait copy` is 1.0 s in
all three (PCIe 4.0 x8), so the GEMMs are not hidden behind the expert stream here. Per call this is the card's FP32
rate against its FP16 rate: the N 10240 x K 2560 projection at ~15 TFLOPS as SGEMM (#1006's 27.7 ms at T 8192) and
37.7 as HGEMM, about 2x on the GEMM itself, 17-21% on the prompt. What the SGEMM route has that this change does not:
no FP16 output rounding, `beta != 0` and N < 64 covered. The three arms gave the same answer on each prompt.

## Range check

FP16 ends at 65504 where BF16 reaches 3.4e38, so the precondition is that nothing the prompt path feeds these GEMMs or gets
out of them leaves FP16's range. The engine now counts it: `STRATA_F16_RANGE=1` records, per device, the largest |value|
and the count beyond 65504 (or not finite) at the three places a value enters or leaves FP16 - the activation images
(`hf_sat`, every image writer's one funnel), the BF16 weights converted to FP16 (`bf16_to_f16_rows`) and the FP16 GEMM
outputs as they are widened (`widen_rows_f16`; an output rocBLAS wrote beyond the range is Inf there, not saturated) -
and prints one line per prompt (`strata f16 range: ...`). Off, the cost is one flag read per value; on, a 34.4K-token
prompt read at 889.8 tok/s against 888.5 without it.

Measured on one card over 466K tokens in nine prompts: three 32-34K slices of the earlier corpus (English docs, Chinese
logs, C++), two prompts (118K and 64K tokens) built to push activations - delimiter floods (`<|im_start|>`, 200 newlines,
500 spaces), one token repeated 4,000 times, 6,000-character runs, 18-digit numbers, 20K characters of base64, emoji and
Japanese / Russian / Arabic / Hindi text, 2,000 SQL lines - and four batches (40-49K tokens) of real requests recorded
from this machine's API:

| | activation images, max abs | BF16 weights, max abs | FP16 GEMM outputs, max abs | beyond 65504 / not finite |
| --- | ---: | ---: | ---: | ---: |
| corpus slices (3) | 84.0 / 103.4 / 112.2 | 11.56 | 387.0 / 392.0 / 391.5 | 0 |
| built to push (2) | 76.6 / 99.0 | 11.56 | 409.5 / 399.2 | 0 |
| real requests (4) | 73.9 - 92.7 | 11.56 | 374.2 - 382.5 | 0 |

The largest output is 409.5, 160x below 65504; the largest activation 112, 580x below. The prompts built to push
activations did not: their maxima are below the ordinary corpus's. Offline, the pack's dense BF16 tensors (484 in the
first GGUF shard) peak at 8.56 (`hc_ffn_up`); 11.56 is in the second shard (the PLE). This is one model (GSQ-RCO IQ3_S);
the other models setup installs are fine-tunes of the same architecture, so the headroom is the same kind of number
there but has not been measured. Out-of-range activations would still be finite: the image writers saturate to
+-65504 (`hf_sat`), a NaN stays a NaN; an out-of-range GEMM output would be Inf, which is what the counter is for.
`to_f16` (the hc image for the FP16-weight GEMMs and the PLE key image) saturates and counts too now, so every FP16
image writer is under the same rule; three of the prompts rerun with it in gave the same maxima.

## Distribution check (teacher-forced)

The method of `bench/results/2026-10-03-v100-prompt-attn` (`docs/UNSLOTH_Q4.md`): `STRATA_LOGPOS=<file>
STRATA_LOGPOS_TOPK=256` on a serve engine (one card, `--short-read 640 --adapt-every 100000`, greedy). Each chat is a long
first message of this repository's docs and source (9,427 / 34,659 / 105,811 tokens, disjoint, read by the batched prompt
path - the code under test), a fixed assistant reply, and a last message of ~580 tokens of other repository text that is
read through the verify windows, where every position is scored: 578 / 519 / 481 positions. KL is P || Q over P's top 256
plus one bucket for the rest.

| comparison | KL mean (9K / 35K / 106K) | KL median | argmax same | top-10 overlap |
| --- | --- | --- | --- | --- |
| old path vs old path (a second run) | 0 / 0 / 0 | 0 | 100% | 100% |
| FP16 vs FP16 (a second run) | 0 / 0 / 0 | 0 | 100% | 100% |
| old path vs FP16 | 0.032 / 0.021 / 0.019 | 0.0056 / 0.0041 / 0.0038 | 94.5 / 95.0 / 96.5% | 91.4 / 92.8 / 91.5% |
| **reference vs old path** | 0.061 / 0.033 / 0.023 | 0.0092 / 0.0096 / 0.0055 | 93.1 / 93.3 / 93.8% | 90.6 / 92.8 / 91.9% |
| **reference vs FP16** | 0.020 / 0.015 / 0.028 | 0.0034 / 0.0044 / 0.0038 | 95.5 / 96.5 / 95.8% | 92.7 / 93.2 / 91.6% |

The reference is the old path with `STRATA_PREFILL_BF16X2=1` (every BF16 activation image carries its low part too, close
to fp32 activations; 393-411 tok/s). The FP16 path is closer to it than the old path in all three chats by median KL and
argmax agreement, and by mean KL in two of three (in the 106K chat the FP16 path's p99 is lower, 0.20 vs 0.24, and its mean is pulled up
by a few positions beyond that). That is the expected direction: an FP16 activation keeps 10 mantissa bits where BF16 keeps 7, which outweighs
rounding the GEMM's output to FP16 before it is widened.

A rounding-order control as in the V100 report did not work here: `--prefill 2048` instead of auto (8192; the smaller
chunk did take effect, 1,059 lent slots instead of the 8192-token ring) gave results **bit-identical** to the old path, so on
this path the chunking does not change the arithmetic and cannot serve as a control.

## Where the rounding moves

Round-to-nearest bounds per value rounded, for values in FP16's normal range (|x| ≥ 6.1e-5; below that FP16 is
subnormal and its relative error grows): BF16 keeps 8 significant bits (≤ 2^-8 ≈ 0.39%), FP16 keeps 11
(≤ 2^-11 ≈ 0.049%) but ends at 65504. Both paths accumulate in fp32.

| | old path | FP16 path |
| --- | --- | --- |
| BF16-weight products: hyper-connection down / up / inject, router, shared-expert gate, indexer, `ssm_alpha` / `ssm_beta`, PLE | W: the pack's BF16; **X rounded to BF16** (≤ 0.39% per element); Y fp32 | W converted to FP16 (exact in the normal range; 0.1-0.9% of these weights lie below it); X rounded to FP16 (≤ 0.049% per element); **Y rounded to FP16** (≤ 0.049%), then widened |
| quantized-weight products through `Gemm::native` / `Gemm::f16`, beta = 0 (X and W already FP16 in both paths) | Y fp32 | **Y rounded to FP16** (≤ 0.049%), then widened |

So the change removes most of the BF16 activation rounding and adds an FP16 rounding of every output. The two are not
directly comparable - an element's rounding enters a K-long dot product, where it partly cancels or adds up depending
on the data - which is why the distribution check above measures the net effect instead of arguing it. The reference
(`STRATA_PREFILL_BF16X2=1`) differs from the old path in exactly the term this change shrinks (it adds the BF16 low part
to every one of these activations), and the FP16 path comes out closer to it: by median KL and argmax agreement in all
three chats, by mean KL in two (the 106K chat's mean is pulled up by a few positions beyond its lower p99).

Not shown: the reference is not an fp32 forward pass, and this is one model, one corpus and 1,578 positions. A comparison
against an fp32 forward pass (e.g. llama.cpp on the CPU) is a follow-up for when time allows. Range is
the cost: the range check found nothing beyond 65504 here, but a model whose activations or BF16 weights reach that far
would need `STRATA_HIP_PROMPT_F16=0`.

## Output checks

- **Needle recall** (`tools/needle_bench.py --lengths 8k,32k,128k --depths 10,30,50,70,90`, greedy, thinking off, a fresh
  server per path, one card; prompts of 7.9K, 32.2K and 125.4K tokens): **15 of 15 found with the old path, 15 of 15 with
  FP16.** A 125K-token read takes 275 s on the old path, 139 s with FP16 (the later depths are shorter because the server
  reuses the checkpoint of the text before the needle).
- **Repeatability:** two runs of each path give bit-identical log-probabilities (table above), and no run of this report
  (17 server starts) stalled.

## What changed

- `Gemm::set_f16_io(bool)`: off by default, so every caller other than the prompt path (e.g. `gemm_bf16_parity`) runs what
  it ran. `Prefill::init` turns it on together with `set_act_f16` when `prompt_f16()` is true for its device.
- `prompt_f16()`: gfx103x, cached per device (a layer split can mix cards); `STRATA_HIP_PROMPT_F16=0/1` overrides.
- With it on: `Gemm::f16` / `Gemm::native` run HH into Y's own rows (`ldc = 2 ldy` halves) and `widen_rows_f16` widens each
  row from its end; `Gemm::bf16` converts W row slices to FP16 through the existing dequantization scratch (as
  `native()` does) and takes X as the FP16 image. `bf16x2_mode()` is 0 there (the BF16 low part has no FP16 meaning).
- No new device memory, no host sync, no fallback branch; CUDA builds are unchanged (`#if defined(__HIPCC__)`).
- `hip_prefill_gemm` (ctest) runs its shapes a second time with `set_f16_io(true)` on any HIP card: X as the FP16 image
  for the BF16 products, a scratch of 64 rows so a 96-row weight converts in two slices, `ldy > N`, an odd N, T = 1 and
  a `beta = 1` f16() call (which keeps the FP32-out GEMM). The FP16-out results are checked against the double reference
  at FP16's rounding (6e-4 relative per element, rel L2 < 1e-3); row padding and the guards around Y must be untouched.
  The test's BF16 inputs are now made by hand: `hip_bfloat16(float)` produced 0 in this build, so the two BF16 cases had
  passed on all-zero inputs (rel L2 printed 0); with real inputs they pass at 1.9e-07 (FP32 out) and 2.2e-04 (FP16 out).

## Limits

- One machine, one model (IQ3_S, packed from the GSQ-RCO GGUF, not one of setup's), one corpus for the distribution check.
- gfx1031/1032 (the same rocBLAS family) are enabled by the same `gfx103` prefix but were not run.
