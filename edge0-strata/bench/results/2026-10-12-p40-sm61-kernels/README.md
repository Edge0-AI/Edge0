# Tesla P40 (sm_61): wide dequantization, the dense GEMMs as cublasSgemm (opt-in), expert mode 8 and a matvec table (2026-10-12)

One Tesla P40 with IQ3_S. The card is the bottleneck of the prompt and of the decode, so the profile was read first
and four things were changed where it pointed. Each gives exactly the bits of what it replaces.

| What | Was on sm_61 | Now on sm_61 | Switch |
| --- | --- | --- | --- |
| Dequantization of the dense weights for the prompt GEMMs (Q4_K / Q5_K / Q6_K / IQ4_XS) | a thread per 32 values, byte loads, 2-byte stores 64 bytes apart | a thread per 8 values, 16-bit loads, one 16-byte store | on by default, `STRATA_DEQUANT_WIDE=0` |
| The large dense prompt GEMMs (qkv, query, gate and output projections) | `cublasGemmEx` on FP16 operands | `cublasSgemm` on FP32 copies | **opt-in**, `STRATA_DENSE_F32=1` |
| Routed experts in VRAM (`native_expert_grouped`) | the CUDA layout (mode 0) | mode 8 | on by default, `STRATA_SM61_TABLE=0` |
| Dense 1-4 column GEMVs and the head (`native_mmvq_il`) | `native_mmvq` | the interleaved kernel with a table read off the P40 (`kIlRows_61`, `kIl1Rows_61`) | on by default, `STRATA_SM61_TABLE=0` |

Through the server, medians of 3 requests, two runs per build alternated, every run started with the card at 65 C:

| | `a0116b2` | this change (defaults) | this change + `STRATA_DENSE_F32=1` |
| --- | ---: | ---: | ---: |
| 565-token prompt, tok/s | 203 / 205 and 205 / 210 | 256 / 255 | 254 / 252 |
| 7,047-token prompt, tok/s | 360-365 or 430-432, by the token count (below) | 369-374 or 437-443, by the token count | 464-474 at every count |
| decode after the short prompt, tok/s | 35.0 / 34.2 and 35.4 / 34.4 | 38.2 / 37.2 | 37.7 / 37.9 |
| decode after the long prompt, tok/s | 33.1 / 33.3 and 32.1 / 33.4 | 33.5 / 35.6 | 35.9 / 34.0 |

The decode gain is small (+6-10% after the short prompt) and inside the spread of single requests; it is the same
size in the GPU stage profile (49.1 -> 45.6 ms per window), which is the steadier measurement.

## Rig

- Tesla P40 24 GB (sm_61, PCIe gen3 x16, 250 W limit), driver 550.163.01. A Proxmox VM with the card passed through:
  20 vCPUs of a Xeon E5-2695 v4 (AVX2, no AVX-512), 98 GB RAM, Debian 13, kernel 6.12. The model is on a virtual
  disk that reads as rotational; every run had the files in the page cache. Nothing else used the card.
- CUDA 12.4.131 (Debian's toolkit; **cuBLAS 12.4.5.8**), GCC 14.2.
- Engine: `main` at `a0116b2` (0.1.43.1), `-DCMAKE_CUDA_ARCHITECTURES=61 -DSTRATA_EXPERIMENTAL_SM60=ON`, built
  unchanged ("base") and with this change. The server and everything else unchanged.
- Model: setup's IQ3_S (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, both shards, the native pack, the shipped
  expert profile and MTP head). Setup's config after `setup.sh --calibrate`: `--expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.70 --mtp --max-context 262144 --kv int8 --kv-resident 32768 --vision --vram-reserve-mib 700
  --pcie-frac 0.00 --conversation-cache-mib 8192 --conversation-cache-slots 4`, **with the experimental speed
  projection on** (layers 4-44, project mode). 8,225 expert slots in VRAM (15.6 GiB), 46.8 GiB of experts in RAM,
  19 pool workers, prompt chunks of up to 8,192 tokens.
- Requests: `bench.py` (here): a ~565-token prompt with 200 new tokens and a ~7,047-token prompt with 60, three
  each, greedy, `reasoning_effort` none, a random first line so nothing is reused. Speeds are the engine's own lines.

## Where the time goes on a P40 (base build)

`STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1` ([profile-base.txt](profile-base.txt)), and
one run under `nvprof` for the kernels' names.

- The prompt's GPU timeline equals its wall time (7,038 tokens: 16,005 of 16,070 ms). Waiting for expert copies is
  0.2% of it, the host's grouping 0.2%. The dense projections' GEMMs (GDN in and out, QSA) are 36%, the experts
  (MMQ: gate/up, down, combine) 28%, the hyper-connection read 13%, the attention 12%.
- A 559-token prompt (2,469 ms) spends 24% in the GDN input projections and 21% in the experts' gate/up, and about
  0.5 s of it in `dequant_kernel<Q6_K>`: every dense weight is dequantized once per chunk, whatever the chunk's
  length (768 calls and 2.72 s over six prompts under nvprof, 3.5 ms a call).
- A decode window is 49 ms of GPU stages with 1-3 ms of CPU experts (hit rate 99%): dense GEMVs ~15 ms, the experts
  in VRAM ~12.5 ms, the hyper-connection reads ~8.5 ms, the head 2.9 ms.

What that leaves alone: `cublasSgemm` already runs at 8.5-10 TFLOPS on this card (11.8 at its boost clock), ggml's
MMQ has its own Pascal dp4a configuration, and the decode GEMVs sit at 60-90% of the memory bandwidth the wide
dequant kernel reaches (227 GB/s). No larger kernel lever was found.

## Kernels (no engine)

**Dequantization**, `dequant_wide_parity --bench` ([dequant-wide-parity.txt](dequant-wide-parity.txt)): 360 cases
bitwise (FP16, BF16, FP32 out, a padded row stride, row slices; sane blocks and raw random bytes), FP16 out:

| Shape | per-group | wide | |
| --- | ---: | ---: | ---: |
| Q6_K 10240 x 2560 | 7.25 ms (10 GB/s) | 0.33 ms (227 GB/s) | 22.3x |
| Q6_K 2560 x 6144 | 4.45 ms | 0.20 ms | 22.5x |
| Q4_K 10240 x 2560 | 1.67 ms | 0.32 ms | 5.2x |
| Q5_K 10240 x 2560 | 1.24 ms | 0.33 ms | 3.8x |
| IQ4_XS 10240 x 2560 | 1.21 ms | 0.34 ms | 3.6x |

**The FP16-operand GEMM**, a sweep of the chunk size T ([gemm-sweep.txt](gemm-sweep.txt), `gemm_sweep.cu`, cuBLAS
only). cuBLAS 12.4 has two kernels for FP16 operands with FP32 sums on this card, `maxwell_sgemm_fp16_64x64_tn` and
`_32x128_tn`, and for most T it picks the slower one:

| Shape | `cublasGemmEx` FP16 operands | `cublasSgemm` FP32 copies |
| --- | --- | --- |
| qkv 10240 x 2560 | 7.6 TFLOPS at 27 of the 76 sizes sampled from T 256 (T 7,030-7,040 among them), **4.3-4.6 at the others** (49 -> 85 ms from T 7,040 to 7,041) | 6.8-10.0 from T 256, 8.4-9.6 from 512 (39-42 ms at 7K) |
| query 12288 x 2560 | 7.5 at 24 of the 76 sizes, 3.8-4.7 at the others (103 ms at 7,041) | 6.7-9.8 from 256, 8.4-9.8 from 512 (47-50 ms at 7K) |
| gate 6144 x 2560 | 7.3-7.8 at 26 of the 65 sizes from T 512 (up to ~1,100, and 7,030-7,040), 4.5-6.1 at the others | 8.2-9.7 from 512 |
| out 2560 x 6144 | 7-8 at 23 of the 47 sizes from T 4,096, 4.4-6.2 at the others | 8.3-9.2 from 4,096 |

So the base build reads a 7K-token prompt in 16.3 s when its chunk has 7,040 tokens or fewer and in 19.5 s when it
has 7,041 or more. The two results are bitwise equal wherever both run a Maxwell kernel; below T ~144 (6,144+ rows)
or ~408 (2,560 rows) cuBLAS takes another kernel family for one of them and the bits differ, so the route starts at
T 256 / 512 / 4,096 for the three groups of shapes. `gemm_f32_parity --bench`
([gemm-f32-parity.txt](gemm-f32-parity.txt)) runs `Gemm::f16` both ways: **5,201 chunk sizes bitwise** (every 7th T
to 8,192, every T of 7,000-7,100, around the powers of two and the activation slices' split), and with the
conversions included qkv at T 7,041 85.4 -> 44.7 ms, at 560 6.5 -> 4.4 ms; the gate projection at 560 and the
4,096-column output projection at 4,096 are 5-7% slower that way (2.9 -> 3.05 ms, 12.3 -> 13.3 ms).

**That equality is a property of this cuBLAS build**, not of the code: which kernel cuBLAS picks, and that both
add in the same order, was measured with 12.4.5.8 only. Hence opt-in. `gemm_f32_parity` is the check for another
library (CUDA 12.9's, the ready-made CUDA 12 engine's).

**Experts**, `native_expert_bench <shard1> 0,5,11,17,23,29,35,41,47 18 2 0 8` (18 groups, 2 tokens, the model's real
rows, bitwise on all nine layers; [native-expert-bench.txt](native-expert-bench.txt)):

| Layer gate/up / down | mode 0 | mode 8 |
| --- | ---: | ---: |
| IQ3_S / IQ4_NL | 390 us | 255 us |
| IQ3_XXS / IQ4_NL | 345 us | 252 us |
| IQ2_S / IQ4_NL | 290 us | 238 us |
| IQ4_XS / IQ4_NL | 328 us | 261 us |

Modes 2, 5 and 6 are slower than mode 0 here, mode 7 is 3-6% behind mode 8.

**Dense GEMVs**, `mmvq_il_parity --bench --emit-table`, three runs
([mmvq-il-bench-p40-run1.txt](mmvq-il-bench-p40-run1.txt), run2, run3). The three emitted tables are identical but
for two Q3_K cells (4 columns below 2,048 rows: 1, 1, 2 -> 1; one column at 2,048-4,095 rows: 2, 2, 0 -> 0). A few
cells of run 1:

| Shape | columns | `native_mmvq` | interleaved |
| --- | ---: | ---: | ---: |
| IQ4_XS 2560 x 10,240 | 2 | 111.8 us | 65.0 us (rows 4) |
| Q6_K 2560 x 12,288 | 4 | 189.3 us | 127.4 us (rows 2) |
| Q6_K 2560 x 10,240 | 1 | 92.6 us | 109.7 us (rows 2): the table keeps `native_mmvq` for one Q6_K column |
| Q5_K head 2560 x 248,320 | 3 | 2,545 us | 1,766 us (rows 2) |
| Q5_K head 2560 x 248,320 | 1 | 1,754 us | 1,507 us (rows 2) |

**Not taken**: the latency-hidden hyper-connection read ([fused-gr-bench.txt](fused-gr-bench.txt), bitwise): 82.0
-> 88.2 us at one token, 92.4 -> 102.6 at two, faster only from five (192 -> 181 us). sm_61 keeps the plain one.

## Engine

[engine-runs.txt](engine-runs.txt) has every request. Each run is one server start; before it the card cooled to
65 C (it ends a run at 82-84 C). Two builds of the change were run: without the GEMM route (what this change does by
default) and with it (`STRATA_DENSE_F32=1`).

Prompt read time by the prompt's token count, engine ms (a chunk has 7 tokens fewer):

| prompt tokens | base | defaults | + `STRATA_DENSE_F32=1` |
| ---: | --- | --- | --- |
| 7,043 | 16,374 / 16,370 | 15,905 | 14,845 |
| 7,046 | 16,352 / 16,334 | 15,963 | 15,065 |
| 7,047 | 16,325 / 16,318 | 16,136 | 15,191 / 14,863 |
| 7,048 | 19,540 / 19,384 / 19,600 / 19,489 / 19,334 | 19,081 / 18,902 / 18,856 | (no request of this size) |
| 7,049 | 19,533 | (no request of this size) | 14,861 / 14,914 |

GPU stages per verify window (`STRATA_VERIFY_PROFILE=1`; three requests after the short prompt 49.2 / 48.3 / 49.1 ->
46.8 / 45.6 / 45.0 ms, after the long one 48.2 / 49.5 / 49.0 -> 45.2 / 45.2 / 44.9 ms). VRAM hits 9.6 +
3.1 -> 7.5 + 2.5 ms, the GDN qkv / gate GEMVs 4.2 + 2.5 -> 3.9 + 2.2 ms, the QSA query GEMV 1.83 -> 1.60 ms, the
head 2.87 -> 2.45 ms; the other stages unchanged ([profile-base.txt](profile-base.txt),
[profile-new.txt](profile-new.txt)).

With `STRATA_DEQUANT_WIDE=0 STRATA_SM61_TABLE=0 STRATA_DENSE_F32=0` the changed build measures as the base one:
210 tok/s short, 360 / 431 / 365 tok/s long (7,048 / 7,047 / 7,048 tokens), decode 33.3 tok/s.

## Answers

Seven fixed prompts (`equal_check.py`: code, prose, arithmetic, JSON, Italian, a 2.6K-token summary, a needle in 3K
tokens), greedy, first requests after a start. With `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0
STRATA_PREFILL_CPU_SHARE=0 STRATA_ROUTE_TAIL_SKIP=0 STRATA_IQ_MT_MIN=1`, two runs of the base build and two of the
changed one (GEMM route on) give **byte-identical answers, 4 of 4 runs on 7 of 7 prompts** (2,900 characters each).

Without `STRATA_IQ_MT_MIN=1` that comparison cannot be made on this CPU: the same build gave two different answers
to the Italian prompt in two runs (drafts accepted 34 of 65 against 33 of 69). 60% of that prompt's experts run on
the CPU, where a lone token's rows round differently from a group's (`native_gu_mt_min`), and how many drafts a
window holds depends on timing.

## Two traps on such a card

- **Temperature.** This P40 reaches 87-89 C within minutes of work; at 89 C `nvidia-smi -q -d PERFORMANCE` shows
  `SW Thermal Slowdown: Active` and the clock goes from 1,531 to ~1,300 MHz. Runs made one after another without the
  cool-down gave the later arm 10-20% less on the long prompt.
- **The token count.** Before the GEMM route, a long prompt's speed has two values, and a random nonce in the prompt
  moves it across the 7,040-token line: what looks like 20% of run-to-run noise is one cuBLAS kernel choice.

## Not tested

- Any other sm_61 card (Tesla P4, GTX 10 series), Windows, two cards or a layer split, `--batch`.
- cuBLAS other than 12.4.5.8 (see above), the CUDA 13 build, HIP and SYCL: only the sm_61 CUDA 12.4 build (and a
  compile of the two libraries without `STRATA_EXPERIMENTAL_SM60`) was built here.
- Prompts above 7K tokens, the 32K / 128K sweep, long-context needles, a quality suite: one needle at 3K tokens and
  the byte comparison above are all there is.
- The engine A/B above was made with two builds of this code that differ from the commits here only in the GEMM
  route's default (absent in the first, on by default in the second, where it is `STRATA_DENSE_F32=1` here). On the
  commits as they are: the three parity tools on the P40 (`gemm_f32_parity --quick`, 134 sizes) and one engine pass
  each way, the last lines of `engine-runs.txt`: by default 261 tok/s short, 439-446 tok/s long (7,045-7,047 tokens,
  the fast side of the line), decode 38.1 / 36.0 tok/s; with `STRATA_DENSE_F32=1` 250 tok/s short, 465-473 long,
  decode 37.6 / 34.3.

## Files

`engine-runs.txt` (every request), `profile-base.txt`, `profile-new.txt` (the timing lines), the kernel outputs
named above, `nvprof-base.txt` (the base build's kernels over three short and three long prompts), `bench.py`,
`equal_check.py`, `gemm_sweep.cu`.
