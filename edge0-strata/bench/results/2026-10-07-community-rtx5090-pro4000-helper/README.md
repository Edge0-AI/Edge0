# Community benchmark: RTX 5090 + RTX PRO 4000 Blackwell, UD-Q4_K_XL on the helper-cache config, engine 0.1.40.3

Measured 2026-10-07 on a daily-driver coding-agent box. The config is the two-GPU helper cache: all 48 layers on the 5090, the PRO 4000 holding 5000 extra experts (`--expert-cache-device1` + `--remote-expert-opt`). Standard ladder: 4k / 32k / 128k fresh prompts (nonce first, nothing reused), one warm-up then 3 measured runs per length, `temperature 0`, 256 tokens out (reasoning counts toward the 256). Limits at the end.

## Hardware and software

- GPU 0: NVIDIA GeForce RTX 5090 32 GB (all layers, expert cache, draft head, vision). GPU 1: NVIDIA RTX PRO 4000 Blackwell 24 GB (helper expert cache + vision encoder). No NVLink; PCIe probe 57.8 GB/s host->device (`pcie_frac 0.55`)
- CPU: AMD Ryzen 7 9800X3D (8 cores / 16 threads). RAM: 128 GB. OS: Windows 11 Pro
- Engine: ready-made CUDA 13.0 build (`engine-BUILD.json`), repo at `d5ea713` (v0.1.40.3)
- Background workload: a vLLM embedding server holds about 12 GiB of GPU 1 permanently, which caps the helper cache at 5000 slots; GPU 0 is Strata-only. This is the machine's real state, not a clean bench rig

## Model and configuration

- Model: `unsloth/Qwen3.8-Flash-Next UD-Q4_K_XL`, native pack (`--pack` + `--native`), shipped expert profile, MTP runtime from `Strata-data/mtp/rt`
- Config as run (`strata-config.json`): `--expert-cache auto` (7674 slots, 22.40 GiB) `--expert-cache-device1 5000` (5000 experts, 14.62 GiB on GPU 1) `--remote-expert-opt --prefill auto --spec 4 --mtp --max-context 262144 --kv int8 --kv-resident 32768 --spec-min-p 0.5 --vision`; prompt chunk auto picks 8192
- Start: `serve/server.py --engine strata --config strata-config.json --idle-unload 0 --port 8081`
- No `STRATA_*` environment overrides; no calibration; reasoning at the server default

## Method

- `bench_ladder.py`: prompt = Strata source and docs cut to the target length plus an "explain and propose three refactorings" task, with a random nonce first so nothing is reused from the cache. One warm-up per length, then 3 measured runs; speeds are the engine's own `timings` from the response (`prompt_per_second`, `predicted_per_second`)
- Daily-use context: the same box runs a coding agent (Kilo) at 15K-160K-token depths between the ladder runs; the ladder numbers are cold-prompt, single-stream

## Results

### Standard ladder (3 measured runs per length after one warm-up)

| Context | Actual prompt tokens | Prompt tok/s median (range) | Decode tok/s median (range) | Draft accept |
| --- | ---: | ---: | ---: | ---: |
| 4k | 5083 | 1513.7 (1493.3-1523.6) | 103.3 (95.0-110.1) | 71.5 |
| 32k | 33008 | 2048.0 (2043.5-2048.6) | 92.2 (87.3-98.6) | 65.6 |
| 128k | 115048 | 2200.6 (2132.8-2220.6) | 93.9 (90.4-95.7) | 69.1 |

### Daily use (warm, coding-agent sessions, depth 15K-160K)

Warm decode (prompt checkpoint-reused, second and later requests in a session): 96.5-121.4 tok/s, median ~101, expert cache hit 92.5-95.3 %. Cold prompt reading of 147K-160K-token prompts with 0 reused: 2,090-2,131 tok/s. KV streaming hit VRAM 99.4-99.7 % up to about 47K depth, 95-96 % at 140-160K.

## Limits

- The vLLM server on GPU 1 caps the helper cache at 5000 slots; a clean rig could size it larger
- One model, one quant, one workload family; the daily-use numbers are production sessions (varying prompts and depths), not repeated identical runs
- The helper cache is chosen over the layer split because the pair is lopsided (the PRO 4000 in every verify window slows decode); the split was not benched in the ladder format here
