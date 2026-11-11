# Community benchmark: RTX 5070 Ti, Ryzen 9 5900X, 64 GB DDR4-3600 (Windows 10)

Measured on 2026-10-06 by [obiscr](https://github.com/obiscr), on a native-Windows machine.
This tests Strata 0.1.39 (the `strata-windows-x64.zip` release build) with three Flash-Next
native packs — **IQ2_XS, IQ3_XXS and IQ3_S** — on one GPU at a 65,536-token context, and then
a six-configuration A/B of the settings that helped a similar machine in #832.

Headline, medians of three runs (greedy, reasoning off, 256-token cap, synthetic
code-explanation prompts, engine-reported timings):

| Model | prompt 4,096: prefill / decode | prompt 32,768: prefill / decode |
| --- | --- | --- |
| **IQ2_XS** | 1,992 tok/s / 102.0 tok/s | 2,817 tok/s / 101.3 tok/s |
| **IQ3_XXS** | 1,653 tok/s / 72.5 tok/s | 2,482 tok/s / 73.7 tok/s |
| **IQ3_S** | 1,439 tok/s / 62.9 tok/s | 2,261 tok/s / 66.1 tok/s |

**The A/B is the part worth reading:** the `--prefill auto:32768` cap is worth **+18–29%
prefill at 32K** on this machine (measured in two sessions; the baseline is the reproducible
side), while the `--kv q4_0` that is standard on #832's box costs **16–18% of prefill here**,
and the engine env levers that gave #832 +6.9%/+8.1% measure **zero** here. Same GPU,
different CPU, memory and PCIe generation — see the section below.

Main limitations: one machine, one context size (128K tests do not fit a 64K context), and
decode throughput drifted by up to 17% across the session while prefill stayed within ±2%.

## Hardware and software

- NVIDIA GeForce RTX 5070 Ti; 16,303 MiB reported VRAM; 300 W power limit. **PCIe Gen4 x16**
  (the platform caps the link at Gen4). GPU clocks were not fixed.
- AMD Ryzen 9 5900X; 12 cores / 24 threads, Zen 3. **AVX2, no AVX-512.** The engine used
  11 expert-pool workers plus its host thread.
- 64 GB installed RAM (Windows reports 63.9 GiB of physical memory); **4 modules, all at
  DDR4-3600 with XMP/DOCP enabled** (~57.6 GB/s theoretical dual-channel).
- Model files on `D:`; 2.83 GiB/s measured while loading experts, so an NVMe-class SSD.
- Windows 10 (build 19045); NVIDIA driver 591.86; CUDA 13.0.
- Strata commit `6f32ec0`; engine **0.1.39** from the release zip; no source build.
- The machine was otherwise idle during the runs; the benchmark drove the server over
  loopback with no browser open.

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`:

- `IQ2_XS/…-00001-of-00002.gguf`, `IQ2_XS/…-00002-of-00002.gguf`
- `IQ3_XXS/…-00001-of-00002.gguf`, `IQ3_XXS/…-00002-of-00002.gguf`
- `IQ3_S/…-00001-of-00002.gguf`, `IQ3_S/…-00002-of-00002.gguf`

Shard 2 is the same file for every size, so only shard 1 was downloaded again. All three
sizes were prepared by the unmodified installer. MTP was fetched by the installer from
`Qwen/Qwen3.8-Flash-Next` and is shared by all three.

Vision is **not installed**; the experimental speed projection is off; the shipped expert
profile was used without calibration.

The arms differ only in the settings below; everything else is the configuration in
[`strata-iq3_s.json`](strata-iq3_s.json) (arm A), **with its local paths replaced by `<repo>`
and `<data-dir>`**:

| arm | `--kv` | `--prefill` | engine env |
| --- | --- | --- | --- |
| **A** baseline | `int8` | `auto` | — |
| **B** | `q4_0` | `auto:32768` | `STRATA_PF_FUSED=1`, `STRATA_PF_FUSED_TILE=128`, `STRATA_GR_DOWN_MAX4=1` |
| **C** | `q4_0` | `auto` | the same three |
| **D** | `int8` | `auto` | the same three |
| **E** | `q4_0` | `auto` | — |
| **F** | `int8` | `auto:32768` | — |

Everything else is identical in all six: `--expert-cache auto --spec 4 --spec-min-p 0.5`,
`--max-context 65536 --kv-resident 32768`, MTP on, one GPU, the same pack and profile.

```text
strata.exe --pack <data-dir>\packs\iq3_s
  --native   <data-dir>\models\IQ3_S\…-IQ3_S-00001-of-00002.gguf
  --ple-gguf <data-dir>\models\IQ3_S\…-IQ3_S-00002-of-00002.gguf
  --expert-profile <repo>\data\expert-profile.bin
  --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5
  --mtp <data-dir>\mtp\rt
  --max-context 65536 --kv int8 --kv-resident 32768
```

## Method

The three model rows use the repository's community harness
([`benchmark.py`](../2026-09-30-community-rtx-5090/benchmark.py)) unchanged: deterministic
synthetic Python functions with a unique nonce at the front of every request (so nothing is
reused), greedy decoding (`temperature 0`, `reasoning_effort "none"`), a 256-token output cap,
streaming, three runs per length, serial, in increasing-length order on one loaded engine.
Throughput values are the engine's own counters read from `/metrics`, not client walls;
`prompt tok/s` divides freshly read tokens by `prompt_ms`, `decode tok/s` divides
`engine_generated` by `decode_ms`. Every run processed its whole prompt: **zero reused
tokens** in all of them.

The A/B uses the same harness and the same two prompt sizes. Arms were run in the order
A, B, B, A / C, A, C, A / D, E, A / F, A, so every comparison has a baseline arm adjacent to
it. Each arm is a fresh server start with its own config; loading time is excluded. The
`--prefill auto:32768` result was re-checked in a second session (the capped config against the
baseline, same two prompt sizes) after the setting was applied to the machine's own config
file. Both sessions carry a `session` field in `results.json`.

## Results

Each cell is the median of three runs; the range is across runs. Every run of every model and
arm is in [`results.json`](results.json) (one object per run, with the engine's own token,
timing and cache fields); the medians are in [`summary.json`](summary.json).

| Model, 64K context | Prompt tokens | Reused | Prompt tok/s [range] | Decode tok/s [range] | TTFT s | Total s |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| IQ2_XS | 4,096 | 0 | 1,992 [1,899–2,001] | 102.0 [83.3–103.0] | 2.10 | 4.60 |
| IQ2_XS | 32,768 | 0 | 2,817 [2,781–2,826] | 101.3 [92.8–104.6] | 11.72 | 14.29 |
| IQ3_XXS | 4,096 | 0 | 1,653 [1,600–1,666] | 72.5 [64.7–77.9] | 2.51 | 6.03 |
| IQ3_XXS | 32,768 | 0 | 2,482 [2,452–2,527] | 73.7 [73.1–77.6] | 13.29 | 16.57 |
| IQ3_S | 4,096 | 0 | 1,439 [1,360–1,442] | 62.9 [54.6–64.1] | 2.89 | 6.93 |
| IQ3_S | 32,768 | 0 | 2,261 [2,259–2,261] | 66.1 [64.9–68.0] | 14.59 | 18.45 |

Moving up a size costs ~27–29% of decode throughput (IQ3_XXS vs IQ2_XS) and ~35–38% (IQ3_S vs
IQ2_XS) on this machine, a little steeper than the ratios in `docs/MODELS.md` (~22% and ~33%
there). Prefill at 32K is higher than at 4K in every configuration, as expected once the
prompt path is fed longer batches.

### A/B: the settings that help a 9800X3D do not transfer here

| arm | 4K prefill (runs) | 32K prefill (runs) | 4K decode | 32K decode |
| --- | --- | --- | --- | --- |
| **A** baseline | **1,439** (1,362–1,454, n=6) | **2,256** (2,087–2,262, n=6) | 61.6 | 64.5 |
| **F** `auto:32768` | 1,450 (n=1) | **2,907** (n=1) | 62.1 | 64.7 |
| **D** env trio | 1,459 (n=1) | 2,282 (n=1) | 62.6 | 63.1 |
| **B** q4_0 + env + cap | 1,168 (n=2) | 2,427 (2,340–2,514, n=2) | 60.0 | 63.2 |
| **C** q4_0 + env | 1,186 (n=2) | 1,899 (1,889–1,908, n=2) | 62.6 | 64.2 |
| **E** q4_0 only | 1,180 (n=1) | 1,902 (n=1) | 60.6 | 65.0 |

Three findings:

1. **`--prefill auto:32768` is a large, free win at long prompts: +18–29% prefill at 32K,
   unchanged at 4K.** Arm F reads a 32,768-token prompt at 2,907 tok/s against the baseline's
   2,256 (+28.9%). Re-checked in a second session, with the cap applied to the machine's own
   config, the engine read 2,635 tok/s against that session's baseline of 2,236 (+17.9%): the
   direction reproduces, the size does not. The baseline is the reproducible side (32K prefill
   2,236–2,256 across both sessions) while the capped arm varied 2,635–2,907; 4K prefill and
   decode stayed inside the baseline's spread in both. It is the one setting from the #832
   appendix that clearly pays on this box. Its cost is a larger prompt-path VRAM borrow: 3,499
   cache slots (6.66 GiB) against 1,834 (3.47 GiB) in arm A, leaving 276 MiB free after load
   instead of 483 MiB (the arms' startup lines are in [`engine.log`](engine.log)).
2. **`--kv q4_0` costs 16–18% of prefill here and buys nothing on decode.** Arm E, which
   changes only the KV type, drops 4K prefill from 1,438 to 1,180 and 32K from 2,256 to 1,902,
   with no decode gain that survives the session's spread. This is the opposite of #832, where
   `q4_0` is part of the working configuration. The two machines share the GPU but not the
   host: 5900X/DDR4-3600/PCIe Gen4 here against 9800X3D/DDR5/PCIe Gen5 there.
3. **The env trio measures zero here.** Arm D lands inside the baseline's range on both
   prompt sizes (1,459 and 2,282 against 1,362–1,454 and 2,087–2,262). #832 measured +6.9%
   prefill and +8.1% long-context decode for the same three variables at 24K–130K prompt
   tokens; on a 64K-context machine with short prompts the same code path is not reached.

### Decode spread is the honest caveat

The same baseline configuration produced decode medians from **53.9 to 63.5 tok/s** across
six runs in one session (one run, `round3-3-A`, was low across every metric at once).
Prefill over the same six runs stayed inside 1,362–1,454 (4K) and 2,087–2,262 (32K). Prefill
conclusions above are therefore solid; decode differences under roughly 10% should not be
read as real without many more runs. Decode in this harness is also workload-shaped: these
prompts accept about 68% of MTP drafts overall (57–77% per run).

## Recall

`tools/needle_bench.py` at 32k with depths 10%, 50% and 90% found **all three needles for
each of the three models** (9/9), at actual prompt lengths of 32,171–32,172 tokens. Answers
exactly matched the expected code words. Results: [`needles.json`](needles.json), keyed by
model. The 128k stage the 5090 report used does not fit this machine's 65,536-token context.

## A Windows trap worth fixing in the harness

The community `benchmark.py` **cannot run on a non-UTF-8 Windows locale**. It reads the
tokenizer with `Path.read_text()` and no encoding, so on a Chinese Windows (cp936) its
248,320-entry vocabulary is silently mis-decoded and the run dies later with
`KeyError: BPE produced a token outside the vocabulary: '\u010a'` — a message that points
nowhere near the cause. Running the same script with `python -X utf8` fixes it without a code
change, and that is how every number above was produced.

This is the same class of bug already reported for `tools/hip/package_windows.py` in #881
(which also notes ~31 unprotected `read_text()` calls in `setup.py`); the harness is a third
site. The second commit of this PR adds `encoding="utf-8"` to its three `read_text()` calls;
every number above was measured before that change, running the unchanged harness with
`python -X utf8`.

## What this machine is not

Not isolated: it is a desktop that was otherwise idle during the runs, not a stripped test
host. One GPU, one context size, one engine version, one quantization family. The 128K and
164K cells of the #832 battery cannot run here at all. No same-workload baseline on another
engine or GPU was run, so the comparison against #832 and the 5090 report is indicative.
Clocks were not fixed, and the decode drift documented above is the reason to treat small
decode deltas as noise.
