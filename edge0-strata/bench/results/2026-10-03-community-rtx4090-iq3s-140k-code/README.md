# Community benchmark: NVIDIA GeForce RTX 4090

Measured on 2026-10-03 by [Dmitry-B](https://github.com/Dmitry-B). This tests Strata 0.1.38 with qwen3.8-flash-next-iq3_s and a 143360-token context. Prompts are code-explanation text; greedy decoding, a 256-token output cap, three runs per configuration; TTFT measured over streaming. These are synthetic workloads; they do not establish general answer quality.

## Hardware and software

- GPU: NVIDIA GeForce RTX 4090; 23028 MiB reported VRAM; 480.00 W power limit; PCIe bus 00000000:01:00.0; PCIe link speed and width: not measured. GPU clocks were not fixed.
- CPU: AMD Ryzen 9 7950X 16-Core Processor (32 logical CPUs).
- RAM: 46464 MiB installed; storage layout: see env.json (lsblk output).
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic; NVIDIA driver 610.57.04; release 13.4, V13.4.92.
- Strata commit `99f3dbd0b21d1401b3769e0c0d963913607f380b` (branch main); engine 0.1.38, release binary from the repository's engine/ directory.
- Background workloads: a dsh/Authentik/Caddy web stack and stock Ubuntu services; the GPU was dedicated to Strata but the operating system was not isolated.

## Model and configuration

- Model: qwen3.8-flash-next-iq3_s (the Strata server's model name); GGUF filenames, sizes, and modification times are in env.json (no hashes).
- Context 143360; INT8 KV; resident experts; GPU vision - see the config copy below.
- Draft vocabulary subset: `draft_vocab=cyrillic` (the English/code subset plus the whole Cyrillic script, ~106k rows). This is wider than the default `en` subset; draft acceptance on English text was unaffected (78-80%), but it can cost a few percent of decode speed.

```text
/home/dgbox/Strata/engine/strata --serve --pack /home/dgbox/Strata-data/packs/iq3_s --native /home/dgbox/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --ple-gguf /home/dgbox/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --expert-profile /home/dgbox/Strata/data/expert-profile-learned-iq3s.bin --expert-cache auto --prefill auto --spec 4 --mtp /home/dgbox/Strata-data/mtp/rt --max-context 143360 --kv int8 --resident-experts --vision --vram-reserve-mib 989 --control-vector-scaled /home/dgbox/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --spec-min-p 0.5 --expert-profile-save /home/dgbox/Strata/data/expert-profile-learned-iq3s.bin --expert-profile-save-every 10
```

Full server config: [config.json](config.json) (was at /home/dgbox/Strata/strata-iq3s-140k-resident.json), environment details: [env.json](env.json).

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
| 4096-prompt | 3971 | 0 | 105 | 3 | 2058.6 (range 2057.9-2061.4, n=3) | 77.5 (range 77.5-96.6, n=3) | 1.95 s (range 1.95-1.95, n=3) |
| 32768-prompt | 31307 | 0 | 103 | 3 | 2948.0 (range 2944.4-2971.6, n=3) | 97.2 (range 96.0-97.6, n=3) | 10.71 s (range 10.63-10.72, n=3) |
| 131072-prompt | 125011 | 0 | 103 | 3 | 3047.8 (range 3047.7-3050.2, n=3) | 84.4 (range 83.6-92.5, n=3) | 41.32 s (range 41.29-41.33, n=3) |
| gen-only | 163 | 0 | 1024 | 3 | 226.0 (range 218.7-236.0, n=3) | 88.7 (range 67.9-92.3, n=3) | 0.74 s (range 0.71-0.77, n=3) |

- Total latency (client, wall): 4096-prompt - 3.3 s (range 3.0-4.08, n=3); 32768-prompt - 11.77 s (range 11.7-11.77, n=3); 131072-prompt - 42.45 s (range 42.43-42.51, n=3); gen-only - 11.82 s (range 2.28-12.24, n=3).
- Memory: {"start_snapshot": {"vram_used_mib": 22116, "ram_used_kib": 46312940}, "peak_vram_used_mib": 22116, "peak_ram_used_kib": 46948172, "note": "peak = maximum of 2-second samples taken during the measured runs"}
- Every run with its draft statistics (accepted/total): [runs.json](runs.json).
- Recall check (needle): [needles.json](needles.json).

## Correctness and limitations

- Speed measurements do not establish general answer quality. The recall check passed 6/6 at 32K and 128K across depths 10/50/90; see needles.json.
- The GPU was not fully isolated: background services may have added small noise.
- Prompts are synthetic (repeated text with a random marker); real workloads will show different prefix reuse and draft acceptance.