# Community benchmark on Radeon 8060S (Strix Halo laptop, 70 W)

Measured on 2026-10-09 by nsbradley88. Strata v0.1.41 with the `docs/STRIX_HALO.md` §5 fast config on a **laptop** Strix
Halo that holds a 70 W package limit under load. Main limitation: prompt throughput is power-limited (see below); every run
is a fresh engine process (no prompt reuse, cold expert cache start).

## Hardware and software

- GPU and VRAM: Radeon 8060S (gfx1151), unified memory; 112 GiB GTT pool (`ttm.pages_limit=29360128`)
- CPU: Ryzen AI Max+ 395; RAM: 128 GB LPDDR5X; storage: NVMe SSD
- Machine: HP ZBook Ultra G1a 14" laptop, platform profile `performance`, on AC
- OS: Ubuntu 26.04, kernel 7.0.0-38, kernel args `amd_iommu=off amdgpu.lockup_timeout=10000,60000,10000,10000`
- ROCm: 7.14.1 (`therock-dist-linux-gfx1151-7.14.1.tar.gz`); hipBLASLt table `tools/hip/gfx1151-hipblaslt-100401.txt`
- Strata: tag v0.1.41 (`fb58e0d`), source build per `docs/STRIX_HALO.md` §2 (`-DSTRATA_PREFILL_MMQ=ON`, gfx1151 only)
- Background workloads: GNOME desktop session (idle); no other GPU jobs
- Power: package power median 66–69 W, **max 70 W** in every run; GPU clock median ~2,070 MHz (max 2,900); measured from
  amdgpu hwmon (`power1_average`, `freq1_input`) every 0.5 s

## Model and configuration

- Model: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ4_XS
  (`Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`); no vision encoder
- Pack: built with `tools/iq_pack.py --compat-bf16` (a pack rebuilt with v0.1.41's tool measured the same); MTP draft
  layer `mtp/rt`; expert profile `data/expert-profile.bin`
- Context: `--max-context` = prompt + 512; KV int8; no KV streaming; expert cache 24,576 (all experts on the GPU, 55.43 GiB)
- Prefill: `auto:16384`; MTP: `--spec 5 --spec-min-p 0.5 --mtp-window 8192`; greedy (engine `--stats`)
- Calibration / experimental speed projection: off

```text
STRATA_HIPBLASLT_TUNING=tools/hip/gfx1151-hipblaslt-100401.txt STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_HC_UPMIX=1 \
STRATA_PA_FAST=1 STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1 STRATA_HC_Q8=1 STRATA_PF_SWITCH_MIN_T=4096 STRATA_PREFILL_STREAM_MIN=128 \
build-halo/strata --pack <pack> --native <UD-IQ4_XS shard 1> --expert-profile data/expert-profile.bin \
  --expert-cache 24576 --resident-budget-gib 55 --prefill auto:16384 --spec 5 --spec-min-p 0.5 --mtp <mtp/rt> \
  --mtp-window 8192 --max-context <prompt+512> --kv int8 --max-new 256 --tokens-file <prompt tokens> --stats
```

## Method

- Prompts: the first 8,192 / 32,768 / 131,072 tokens of llama.cpp's `src/**/*.cpp` sources (C++), tokenized with
  `tools/strata_tokenizer.py`; 256 generated tokens (cap reached in every run).
- One fresh engine process per run; model loading excluded (the engine's own `prefill` / `decode` timing lines).
- Runs from two sessions on the same build and config (2026-10-09, after the reboot that set the kernel args); 4–6 runs
  per length, interleaved with other configurations. Per-run data: [runs.json](runs.json) (ms and tok/s from the engine).
- TTFT: not measured separately (engine-level runs; the prompt time is listed).
- Memory: `MemAvailable` sampled every 2 s; minimum 45–49 GiB during these runs.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| fast config, `--spec 5 --mtp-window 8192` | 8,192 | 0 | 256 | 6 | 969 (836–1,009) | 59.2 (58.5–59.6) | not measured (prompt 8.5 s) |
| same | 32,768 | 0 | 256 | 5 | 1,067 (1,041–1,067) | 57.8 (57.1–58.7) | not measured (prompt 30.7 s) |
| same | 131,072 | 0 | 256 | 4 | 1,031 (1,005–1,043) | 42.0 (38.0–44.6) | not measured (prompt 127 s) |

Other measurements on the same machine (engine-level, same prompts, medians):

| Change | Session | Prompt tok/s | Decode tok/s | Same-session baseline |
| --- | --- | ---: | ---: | --- |
| v0.1.41 defaults only (no §5 switches), 8K | A | 577–597 | 50.4–50.7 | fast config `--spec 4`: 862–873 / 55.0–55.3 |
| `--spec 5 --mtp-window 8192` vs `--spec 4`, 32K | A | 885–939 | 54.3–54.7 | `--spec 4`: 892–952 / 50.3–50.4 |
| `--spec 6`, 32K | A | 982 | 43.2 | `--spec 4`: 901–1,006 / 51.1–51.7 |
| `--prefill 16384 --spec 4 --lookup-chain 3 --mtp-q4 all` (§6 flags), 32K | B | 1,027–1,067 | 48.9–53.3 | this report's config: 1,044–1,067 / 57.8–58.7 |
| same v0.1.41 built against ROCm 10.2 (table `100500`), 32K | B | 1,037–1,049 | 55.7–59.1 | same |
| 104 GiB GTT + IOMMU on → 112 GiB + `amd_iommu=off`, 32K | A→B | 959–986 → 1,041–1,067 | 56.3–56.4 → 57.1–57.8 | before/after reboot, same script |

Session A: 104 GiB GTT, IOMMU on; session B: the kernel args of this report. Rows compare against a baseline from the
same session.

Server-level, 4 concurrent clients (distinct ~6K-token prompts, ≤512 out, aggregate output tok/s incl. prompt time):
one at a time 32.7 (first answer 3.9 s); `"parallel": 2` 28.2; `"parallel": 4` 27.8; `"parallel": 4` + `--batch-mtp`
30.6 (first answer 41 s).

Failure: with setup's memory arguments for this model (`--expert-cache auto`, no `--resident-budget-gib`) the run
caused a global OOM on this 128 GB machine; reported as issue #1715. All runs above use the explicit cache arguments.

## Correctness and limitations

- Needle: our own test (5 codenames at 10–90% depth of an ~83K-token code prompt, 3 seeds, OpenAI server): **15/15**,
  93–105 s per prompt. Agentic: 5 Go coding tasks (charmbracelet/bubbles) through opencode, hidden tests: **5/5**.
  `tools/needle_bench.py` was not run.
- Prompt throughput is ~20% below the §6 desktop table because of the 70 W limit (config, pack, ROCm runtime and kernel
  args were A/B'd; none closes it); decode matches or exceeds it.
- Greedy engine runs only; 128K has 4 runs (one low decode outlier at 38.0).
- Full write-up and configs: https://github.com/nsbradley88/halo-tuning (qwen-strata/)
