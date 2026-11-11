# Community benchmark: NVIDIA GeForce RTX 4090 (Russian prompts)

Measured on 2026-10-08 by [Dmitry-B](https://github.com/Dmitry-B). This is the Russian-language counterpart of [2026-10-08-community-rtx4090-iq3xxs-200k-code](../2026-10-08-community-rtx4090-iq3xxs-200k-code/README.md): same PC, same model, same server config, same sweep, same method - only the prompt text differs (Russian prose instead of code-explanation text). It follows [2026-10-07-community-rtx4090-iq3xxs-200k-ru](../2026-10-07-community-rtx4090-iq3xxs-200k-ru/README.md) (0.1.40.2 on this machine).

Short answer: **0.1.40.3 is not measurable here either.** Prompt throughput moved by -0.2…+3.6% against a same-binary noise band of -3.8…+0.8%. The one number that looks large (+32% decode at 131072-prompt) is not a change: the answers ended at different lengths, and decode on this synthetic text follows draft acceptance, which is not comparable across different answer lengths.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed.
- Storage: models, packs and the expert profile are on a 1.9 TB NVMe drive (ADATA LEGEND 960, ext4, mounted at `/`). No network storage is in the measured path.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; nvcc release 13.4, V13.4.92.
- Strata commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c` (tag `v0.1.40.3`), branch `main`; engine 0.1.40.3 **compiled from source on this machine** (`engine/BUILD.json` records `"source": "local"`, `archs: [89]`, nvcc from `/usr/local/cuda`). The release publishes Windows archives only, so a source build is the only path on Linux; the version it is compared against was built the same way.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated. This server also serves an interactive agent session; no request from it was issued while the measured runs were running (see *Correctness and limitations*).
- PCIe link speed and width: not measured - `nvidia-smi -q -d BUS` answers "Failed to parse --display/-d flags" on this driver, and `lspci` link status was not collected.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; `--mmap-experts` with the expert cache in `auto` mode; GPU vision - see the config copy below.
- Expert profile: a profile learned online on this model, reused between restarts (`--expert-profile-save … --expert-profile-save-every 10`). It is a persistent artifact of this machine, and the same file was in use for the 0.1.40 / 0.1.40.1 / 0.1.40.2 reports.
- Experimental speed projection is enabled: `--control-vector-scaled …/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`. It was enabled in every earlier report from this PC.
- `--vram-reserve-mib 989`, `--pcie-frac 0.00`, `--pool-workers 10`, `--spec 4` with an MTP draft pack.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows). This matters for the decode column - see *What the Russian variant adds*.
- The engine's auto prompt chunk is `prompt chunk auto: 8192 tokens, a 96-slot ring` in the server log - unchanged from 0.1.38 through 0.1.40.3, so these numbers are comparable with the earlier reports from this PC.
- The launch command is byte-identical to the one in the 0.1.40.2 report; the server config files were unchanged by the update.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json; the bearer token in its `mcp_servers` block is replaced with `removed`), environment details: [env.json](env.json).

## Method

- Warm-up: a **full sweep** was run first and discarded, then the benchmark's own 4K warm-up. In the discarded sweep 4K read at 2070.1-2142.8 tok/s with a 1.909-1.974 s TTFT and 32K at 3165.3-3359.2 tok/s, while the measured pass read 2181.5-2204.6 and 3363.4-3382.8 tok/s. That difference is the cold expert cache and page cache after a service restart, not the version.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the reused column reports the actual counts from the engine (0 in every run here).
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). The model stopped early on the repetitive text in most runs; actual generated lengths are in the table, and they are the reason the decode column cannot be read as a version comparison.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.
- The measurement script is a local script (not part of this repository); it issues the requests, reads the engine's timing lines, and writes `runs.json` / `needles.json` in the format used by the earlier reports from this PC. Available on request.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 4028 | 0 | 142/142/256 | 3 | 2199.3 (range 2181.5-2204.6, n=3) | 102.0 (range 96.8-108.5, n=3) | 1.86 s (range 1.85-1.87, n=3) |
| 32768-prompt | 31828 | 0 | 123/237/256 | 3 | 3375.1 (range 3363.4-3382.8, n=3) | 98.5 (range 95.4-100.9, n=3) | 9.56 s (range 9.54-9.59, n=3) |
| 131072-prompt | 127228 | 0 | 118/118/118 | 3 | 3336.2 (range 3334.7-3340.9, n=3) | 118.9 (range 102.2-122.2, n=3) | 38.59 s (range 38.54-38.62, n=3) |
| gen-only | 228 | 0 | 721/421/906 | 3 | 320.4 (range 298.2-324.8, n=3) | 97.0 (range 95.7-99.6, n=3) | 0.73 s (range 0.71-0.78, n=3) |

- Total latency (client, wall): 4096-prompt - 3.26 s (range 3.16-4.49, n=3); 32768-prompt - 11.93 s (range 10.77-12.23, n=3); 131072-prompt - 39.58 s (range 39.57-39.68, n=3); gen-only - 8.24 s (range 5.11-9.81, n=3).
- Draft acceptance (accepted/total, all runs of a case): 4096-prompt 197/372; 32768-prompt 269/451; 131072-prompt 177/220; gen-only 753/1593.
- Memory: peak VRAM 22026 MiB (unchanged from the start snapshot), peak RAM 6771640 KiB.
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json) - 6/6 found at 32K and 128K, depths 10/50/90.

## Comparison with engine 0.1.40.2 on the same PC

Same PC, same model, same server config, same launch command, same `expert-profile-learned.bin`, same prompt chunk (8192). Only the Strata version changed: 0.1.40.2 (commit `e8ca9af`) -> 0.1.40.3 (commit `d5ea713`), both compiled from source with CUDA 13.4 for `sm_89`.

Three reference runs are included so the comparison is self-contained:

- [runs-0.1.40.2-previous.json](runs-0.1.40.2-previous.json) - the previous version, measured 2026-10-07 with this same method;
- [runs-0.1.40-baseline.json](runs-0.1.40-baseline.json) - the 0.1.40 baseline, measured 2026-10-06;
- [runs-0.1.40.1-control.json](runs-0.1.40.1-control.json) - a full repeat of the sweep on **the identical engine binary** (0.1.40.1 changed only the Python server; `engine/strata` was not rebuilt). This is the noise band of the method on this machine.

| Configuration | Prompt tok/s 0.1.40.2 -> 0.1.40.3 | Decode tok/s 0.1.40.2 -> 0.1.40.3 | TTFT s 0.1.40.2 -> 0.1.40.3 | Noise band (same binary): prompt / decode | Answer tokens 0.1.40.2 -> 0.1.40.3 |
| --- | --- | --- | --- | --- | --- |
| 4096-prompt | 2204.7 -> 2199.3 (-0.2%) | 108.1 -> 102.0 (-5.6%) | 1.86 -> 1.86 | +0.8% / +8.6% | 144/142/142 -> 142/142/256 |
| 32768-prompt | 3356.1 -> 3375.1 (+0.6%) | 93.7 -> 98.5 (+5.1%) | 9.61 -> 9.56 | -0.7% / -0.1% | 102/256/123 -> 123/237/256 |
| 131072-prompt | 3317.3 -> 3336.2 (+0.6%) | 89.9 -> 118.9 (+32.3%) | 38.82 -> 38.59 | -0.5% / +5.7% | 254/123/123 -> 118/118/118 |
| gen-only | 309.2 -> 320.4 (+3.6%) | 94.0 -> 97.0 (+3.2%) | 0.75 -> 0.73 | -3.8% / +8.3% | 640/720/286 -> 721/421/906 |

- **Prompt throughput: no measurable change.** -0.2…+3.6% against a same-binary band of -3.8…+0.8%. The ranges are tight in both versions (3334.7-3340.9 and 3259-3344 tok/s at 131072-prompt), which is what makes this a negative result rather than an inconclusive one.
- **Decode: no claim.** Three of four cases sit inside the noise band. The 131072-prompt +32.3% is not a version effect: the answers ended at 254/123/123 tokens in 0.1.40.2 and at 118/118/118 here, and draft acceptance moved from 209/360 (58%) to 177/220 (80%) - a different early-stop pattern, not a faster engine. On this synthetic text the decode column follows draft acceptance, so it is only comparable when the answer lengths match, and here they do not.
- **The 0.1.40 -> 0.1.40.2 prompt gain is still there.** Against the 0.1.40 baseline this build reads +16.1% at 4096, +8.0% at 32768, +6.4% at 131072 and +7.3% at the 228-token prompt, with TTFT 0.06-2.5 s lower. Nothing of it was lost in 0.1.40.3.
- **Attribution of that gain, corrected.** The 0.1.40.2 report suggested the prompt stager's wait change (#1057). The 0.1.40.3 docs give numbers in the opposite direction (sleeping stager waits read prompts 5-6% slower on a Ryzen 9 7940HS + RTX 4070 laptop while whole-machine CPU use falls from 77-90% to 21-25%; on a desktop RTX 5070 spinning is 1.2% faster), so #1057 is a CPU-load feature, not a speed feature. The cause of the 0.1.40.2 prompt gain is **not identified** by these measurements.
- Recall: 6/6 at 32K and 128K in both versions.
- Longer trend on this PC, same config and method: [2026-10-03](../2026-10-03-community-rtx4090-iq3xxs-200k-ru/README.md) (0.1.38) -> [2026-10-04](../2026-10-04-community-rtx4090-iq3xxs-200k-ru/README.md) (0.1.39) -> [2026-10-07](../2026-10-07-community-rtx4090-iq3xxs-200k-ru/README.md) (0.1.40.2) -> this report (0.1.40.3). Prompt throughput at 131072 tokens: 2984 -> 3126 -> 3136 (0.1.40, in runs-0.1.40-baseline.json) -> 3317 -> 3336 tok/s.

## What the Russian variant adds

Published as a second folder on purpose: it separates what depends on the engine from what depends on the text.

- **Prompt throughput is nearly script-independent.** At 32768 nominal tokens this build reads 3363.4-3382.8 tok/s on Russian text and 3316.0-3350.9 tok/s on code text - a 1-2% difference, inside the noise band. The same holds at 131072 (3334.7-3340.9 vs 3278.1-3313.8).
- **Decode is strongly script-dependent.** 95.4-100.9 tok/s here versus 136.2-138.9 tok/s in the code variant at 32768, and 95.7-99.6 versus 139.6-145.2 in the generation-only case. The gap tracks draft acceptance exactly: 47-80% accepted here versus 74-79% in the code variant.
- **Cyrillic text tokenizes denser.** The same nominal 32768-token prompt becomes 31828 tokens here and 31307 in the code variant; 131072 becomes 127228 and 125011. Any cross-script comparison has to compare actual token counts, not nominal lengths.
- For Cyrillic workloads the setting that matters is `draft_vocab=cyrillic`: it is what makes draft acceptance on Russian text reach 53-80% at all. The default `en` subset does not cover Cyrillic.

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated Russian prose with a random marker); real workloads will show different prefix reuse, early-stop behaviour and draft acceptance.
- This server also serves an interactive agent session. No request from it was issued during the measured runs; the measured sweeps ran as one detached process from start to finish.
- The engine is compiled from source with CUDA 13.4, while the release binaries are built with CUDA 13.0. Comparisons inside this report are unaffected (every version here was built the same way on this machine), but they are not a comparison against published Linux binaries - there are none for this tag.
- Opt-ins from this release were **not** tested here and are not part of these numbers: `STRATA_PREFILL_CPU_SHARE`, `STRATA_IO_PREFETCH` / `STRATA_IO_PF_STAGE`, `pin=N` (docs/RESEARCH_RUNS.md), `STRATA_SPEC_GUMBEL`, `STRATA_MMVQ_IL=0`, `STRATA_STAGER_SLEEP=0`.
