# Community benchmark: NVIDIA GeForce RTX 4090

Measured on 2026-10-08 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.40.3 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality. The same sweep with Russian-language prompts is in [2026-10-08-community-rtx4090-iq3xxs-200k-ru](../2026-10-08-community-rtx4090-iq3xxs-200k-ru/README.md).

This is the follow-up of [2026-10-07-community-rtx4090-iq3xxs-200k-code](../2026-10-07-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.40.2 on the same PC and the same config). Its short answer: **0.1.40.3 is not measurable on this machine** - every case lands inside the noise band of this method, which is what the release's own change list predicts for a Linux/NVIDIA build.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; nvcc release 13.4, V13.4.92.
- Strata commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` (tag `v0.1.40.3`), branch `main`; engine 0.1.40.3 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). No ready-made Linux engine is published for this tag either (the release publishes Windows archives only), so a source build is the only path here; it is the same build `./update.sh` runs. The previous version in the comparison was built the same way.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated. This server also serves an interactive agent session; no request from it was issued while the measured runs were running (see *Correctness and limitations*).
- PCIe link speed and width: not measured - `nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver, and `lspci` link status was not collected.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` with the expert cache in `auto` mode; GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model (`--expert-profile … --expert-profile-save … --expert-profile-save-every 10`), reused between restarts. It is a persistent artifact of this machine, not a fresh calibration, and the same file was in use for the 0.1.40 / 0.1.40.1 / 0.1.40.2 reports.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows).
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log - unchanged from 0.1.38 through 0.1.40.3, so these numbers are comparable with the earlier reports from this PC.
- The launch command is byte-identical to the one in the 0.1.40.2 report (compared line by line in the server log); the server config files were unchanged by the update.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

- Warm-up: a **full sweep** (4K, 32K, 128K, gen-only) was run first and discarded, then the benchmark's own 4K warm-up. This is required on this machine: in the discarded sweep the first 4K read ran at 1660.6 tok/s with a 2.416 s TTFT and the rest of that sweep at 1971-2021 tok/s, while the measured pass read 2146.1-2161.8 tok/s with 1.86-1.87 s TTFT. That difference is the cold expert cache and page cache after a service restart, not the version.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused token counts from the engine (0 in every run here).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). The model often stopped early on the repetitive text; actual generated lengths are in the table.
- The expert cache was warmed by the warm-up sweep and earlier sessions; the expert profile state is in the config copy.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 3971 | 0 | 256 | 3 | 2146.4 (range 2146.1-2161.8, n=3) | 128.2 (range 124.6-138.1, n=3) | 1.87 s (range 1.86-1.87, n=3) |
| 32768-prompt | 31307 | 0 | 256 | 3 | 3341.8 (range 3316.0-3350.9, n=3) | 138.4 (range 136.2-138.9, n=3) | 9.46 s (range 9.43-9.53, n=3) |
| 131072-prompt | 125011 | 0 | 142/256/256 | 3 | 3292.3 (range 3278.1-3313.8, n=3) | 121.7 (range 110.3-125.6, n=3) | 38.27 s (range 38.02-38.43, n=3) |
| gen-only | 163 | 0 | 1024 | 3 | 218.3 (range 211.6-225.4, n=3) | 143.3 (range 139.6-145.2, n=3) | 0.77 s (range 0.74-0.79, n=3) |

- Total latency (client, wall): 4096-prompt - 3.86 s (range 3.7-3.92, n=3); 32768-prompt - 11.29 s (range 11.26-11.39, n=3); 131072-prompt - 40.05 s (range 39.55-40.52, n=3); gen-only - 7.92 s (range 7.8-8.06, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 492/662; 32768-prompt 502/658; 131072-prompt 412/547; gen-only 2061/2624.
- Memory: peak VRAM 22026 MiB (unchanged from the start snapshot), peak RAM 6747076 KiB.
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json) - 6/6 found at 32K and 128K, depths 10/50/90.

## Comparison with engine 0.1.40.2 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192). Only the Strata version changed: 0.1.40.2 (commit `e8ca9af`) -> 0.1.40.3 (commit `d5ea713`), both compiled from source with CUDA 13.4 for `sm_89`.

Three reference runs are included so the comparison is self-contained:

- [runs-0.1.40.2-previous.json](runs-0.1.40.2-previous.json) - the previous version, measured 2026-10-07 with this same method;
- [runs-0.1.40-baseline.json](runs-0.1.40-baseline.json) - the 0.1.40 baseline, measured 2026-10-06;
- [runs-0.1.40.1-control.json](runs-0.1.40.1-control.json) - a full repeat of the sweep on **the identical engine binary** (0.1.40.1 changed only the Python server; `engine/strata` was not rebuilt). This is the noise band of the method on this machine, and it is what decides whether a difference is a change.

| Configuration | Prompt tok/s 0.1.40.2 -> 0.1.40.3 | Decode tok/s 0.1.40.2 -> 0.1.40.3 | TTFT s 0.1.40.2 -> 0.1.40.3 | Noise band (same binary): prompt / decode |
| --- | --- | --- | --- | --- |
| 4096-prompt | 2180.0 -> 2146.4 (-1.5%) | 133.5 -> 128.2 (-4.0%) | 1.84 -> 1.87 | -1.8% / -5.0% |
| 32768-prompt | 3367.3 -> 3341.8 (-0.8%) | 133.7 -> 138.4 (+3.5%) | 9.38 -> 9.46 | -2.0% / -3.4% |
| 131072-prompt | 3312.6 -> 3292.3 (-0.6%) | 128.1 -> 121.7 (-5.0%) | 38.03 -> 38.27 | -1.1% / -5.6% |
| gen-only | 224.6 -> 218.3 (-2.8%) | 141.7 -> 143.3 (+1.1%) | 0.74 -> 0.77 | -6.6% / -3.1% |

- **No measurable change from 0.1.40.2 to 0.1.40.3.** Every prompt delta (-0.6…-2.8%) is inside the same-binary band (-1.1…-6.6%), every decode delta (-5.0…+3.5%) is inside its band (-5.6…-3.1%), and TTFT moved by 0.03-0.24 s. The honest reading is "the same speed", not "-1%".
- **That is what the release predicts for this machine.** The 0.1.40.3 diff touches four engine files: `src/core/mtp.cpp` adds a `n_expert == 512 && K == 10` guard to the drafter's per-token router call - this model has `expert_count = 512` and `expert_used_count = 10` in its GGUF metadata, so the guarded native path is the one already taken; `src/program/generate.cpp` adds HIP/Windows-only diagnostics behind `#if defined(_WIN32) || defined(STRATA_USE_HIP)`; `src/kernels/native_multi_parity.cpp` adds a self-test, not runtime code; and the commit that made `STRATA_MMVQ_IL` opt-in is reverted in the same release, so the interleaved q8_1 projections stay on by default exactly as in 0.1.40.2. The rest of the release is Intel Arc setup, Windows AMD packaging, Docker, the web app and the tokenizer.
- **The 0.1.40 -> 0.1.40.2 prompt gain is still there.** Against the 0.1.40 baseline this build reads +8.5% at 4096, +6.6% at 32768, +5.6% at 131072 and +2.7% at the 163-token prompt, with TTFT 0.02-2.1 s lower. Nothing of it was lost in 0.1.40.3.
- **Attribution of that gain, corrected.** The 0.1.40.2 report suggested the prompt stager's wait change (#1057) as the cause. The 0.1.40.3 docs give numbers in the opposite direction: sleeping stager waits read prompts 5-6% slower on a Ryzen 9 7940HS + RTX 4070 laptop (while whole-machine CPU use falls from 77-90% to 21-25%), and on a desktop RTX 5070 spinning is 1.2% faster. So #1057 is a CPU-load feature, not a speed feature, and the cause of the prompt gain in 0.1.40.2 is **not identified** here - it is uniform across lengths, which is all this data says about it.
- Draft acceptance is unchanged in substance (71.7% -> 74.3% at 4096-prompt, 75.8% -> 76.3% at 32768-prompt, 78.5% -> 78.5% at gen-only). The 131072-prompt totals differ between runs only because the answers ended early at different points (163/142/142 tokens in 0.1.40.2, 142/256/256 here).
- Recall: 6/6 at 32K and 128K in both versions.
- Longer trend on this PC, same config and method: [2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.38) -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.39) -> [2026-10-07](../2026-10-07-community-rtx4090-iq3xxs-200k-code/README.md) (0.1.40.2) -> this report (0.1.40.3). Prompt throughput at 131072 tokens: 2962 -> 3085 -> 3119 (0.1.40, in runs-0.1.40-baseline.json) -> 3313 -> 3292 tok/s. The last step is inside the noise band: the jump happened at 0.1.40.2 and 0.1.40.3 keeps it.

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.
- This server also serves an interactive agent session. No request from it was issued during the measured runs; the measured sweeps ran as one detached process from start to finish.
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (every version here was built the same way on this machine), but they are not a comparison against published Linux binaries - there are none for this tag.
- Opt-ins from this release were **not** tested here and are not part of these numbers: `STRATA_PREFILL_CPU_SHARE`, `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`, `STRATA_STAGER_SLEEP=0`.
