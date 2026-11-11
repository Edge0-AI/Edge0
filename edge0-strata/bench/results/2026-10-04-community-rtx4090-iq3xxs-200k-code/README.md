# Community benchmark: NVIDIA GeForce RTX 4090

Measured on 2026-10-04 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.39 with qwen3.8-flash-next-iq3_xxs and a 204800-token context. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed; storage layout: see env.json (lsblk output).
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; release 13.4, V13.4.92.
- Strata commit `2900da3ba8de27a467b4b3fcea6a4ca2a9061bea` (branch bench/2026-10-03-rtx4090-community); engine 0.1.39, release binary from the repository's engine/ directory.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_xxs (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 204800; INT8 KV; resident experts; GPU vision - see the config copy below.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows). This is wider than the default `en` subset; draft acceptance on English text was unaffected (78-80%), but it can cost a few percent of decode speed.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_xxs --native /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 204800 --kv int8 --mmap-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --pcie-frac 0.00 --spec-min-p 0.50 --pool-workers 10 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-200k.json), environment details: [env.json](env.json).

## Method

- Warm-up: one 4K-token prompt (64 generated tokens), excluded from the measurements.
- Every measured prompt carries a random marker, so the prompt-prefix cache is not reused; the table's reused column reports the actual reused token counts from the engine.
- Throughput comes from the engine's timing fields (prompt_per_second / predicted_per_second). TTFT is the time to the first non-empty streaming delta, ignoring keep-alives. Total latency (wall) is the whole request time measured at the client.
- temperature=0, reasoning_effort=none, a 256-token output cap (1024 for the generation-only case). The model often stopped early on the repetitive text; actual generated lengths are in the table.
- The expert cache was warmed by the warm-up run and earlier sessions; the expert profile state is in the config copy.
- Memory: peak VRAM/RAM sampled every 2 seconds during the measured runs, plus a start snapshot.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT s median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4096-prompt | 3971 | 0 | 256 | 3 | 1966.2 (range 1952.4-1967.7, n=3) | 123.6 (range 109.6-134.3, n=3) | 2.04 s (range 2.04-2.06, n=3) |
| 32768-prompt | 31307 | 0 | 256 | 3 | 2973.7 (range 2952.3-3130.1, n=3) | 134.9 (range 123.8-136.0, n=3) | 10.61 s (range 10.09-10.69, n=3) |
| 131072-prompt | 125011 | 0 | 183 | 3 | 3084.9 (range 3043.4-3109.1, n=3) | 120.3 (range 91.5-124.1, n=3) | 40.82 s (range 40.51-91.16, n=3) |
| gen-only | 163 | 0 | 1024 | 3 | 205.3 (range 204.7-219.0, n=3) | 135.8 (range 135.2-138.6, n=3) | 0.81 s (range 0.76-0.82, n=3) |

- Total latency (client, wall): 4096-prompt - 4.1 s (range 3.94-4.38, n=3); 32768-prompt - 12.48 s (range 11.97-12.75, n=3); 131072-prompt - 42.87 s (range 42.02-92.7, n=3); gen-only - 8.29 s (range 8.18-8.37, n=3).
- Memory: {"start_snapshot": {"vram_used_mib": 22031, "ram_used_kib": 5931132}, "peak_vram_used_mib": 22031, "peak_ram_used_kib": 6667592, "note": "пик — максимум опросов каждые 2 с во время замеров"}
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json).

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.
- One measured run is contaminated by the interactive session this machine was also serving: a 139,527-token request
  from another client was served between two 131072-prompt runs (the server answers one sequence at a time), which is
  why that case's TTFT and wall ranges reach 91 s and 93 s. The engine-side timings of that run are unaffected
  (125,011 tokens read at 3043 tok/s, the same as its two neighbours); only the client-side waiting is inflated.


## Comparison with engine 0.1.38

The same PC, the same model and the same run configuration as
[2026-10-03-community-rtx4090-iq3xxs-200k-code](../2026-10-03-community-rtx4090-iq3xxs-200k-code/README.md)
(engine 0.1.38, Strata commit `1d5e1ea`). Only the Strata version changed: 0.1.38 -> 0.1.39, source build, same
CUDA 13.4, same `sm_89`, same arguments, same `expert-profile-learned.bin`. The engine's auto prompt chunk is
unchanged in both (`prompt chunk auto: 8192` in the server log), so the byte-budget ring of #583 does not apply to
this configuration.

| Configuration | Prompt tok/s 0.1.38 -> 0.1.39 | Decode tok/s 0.1.38 -> 0.1.39 | TTFT s 0.1.38 -> 0.1.39 |
| --- | --- | --- | --- |
| 4096-prompt | 1844.3 -> 1966.2 (+6.6%) | 118.0 -> 123.6 (+4.7%) | 2.18 -> 2.04 |
| 32768-prompt | 2976.2 -> 2973.7 (-0.1%) | 127.9 -> 134.9 (+5.5%) | 10.6 -> 10.61 |
| 131072-prompt | 2962.3 -> 3084.9 (+4.1%) | 108.4 -> 120.3 (+11.0%) | 42.5 -> 40.82 |
| gen-only | 200.0 -> 205.3 (+2.7%) | 133.0 -> 135.8 (+2.1%) | 0.83 -> 0.81 |

- Prompt throughput is the comparable number: it comes from the engine's own timing, and the ranges are tight in both
  runs (2906-2965 and 3021-3110 tok/s at 131072-prompt).
- Decode medians are the least comparable column. Every measured prompt carries a random marker, and on this repetitive
  text the model stops early at a different point each run, so the generated lengths differ between the two reports
  (for example 142 vs 256 tokens at 4096-prompt). A shorter answer and a longer one do not have the same draft
  acceptance. The `gen-only` case, which always runs to the 1024-token cap, is the cleanest decode comparison of the
  four.
- Recall: 6/6 at 32K and 128K in both versions.
