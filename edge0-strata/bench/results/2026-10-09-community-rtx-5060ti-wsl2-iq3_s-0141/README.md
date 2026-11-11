# Community benchmark on RTX 5060 Ti 16 GB (WSL2) + GSQ-RCO IQ3_S

Measured on 2026-10-09 by kgmkm. Single-GPU serve of the official
ISTA-DASLab GSQ-RCO IQ3_S on engine 0.1.41 under WSL2, with experimental
speed projection (ESP) enabled. Main limitations: one run per condition
(no median/range), TTFT and needle recall not measured, WSL2 + virtiofs
stack rather than native Linux.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5060 Ti 16 GB (16311 MiB), single-GPU
  serve. A second card (RTX A4000 16 GB) is present but idle (0 MiB).
  PCIe Gen4 x8, probed 14.1 GB/s host->device, so pcie_frac 0.39.
- CPU: AMD Ryzen 7 3700X, 8 cores / 16 threads, no AVX-512 (AVX2 expert
  kernels); 7 pool workers.
- Installed RAM: 128 GB host; WSL2 VM 78 GiB. Engine arena ~48 GB;
  about 50 GB used / 27 GB free during the runs.
- Storage: model served from a Windows NVMe drive (KLEVV CRAS C720 2 TB)
  mounted in WSL2 via virtiofs from a Windows partition on the same NVMe drive.
  Sequential read 1.1 GB/s;
  engine load 46.84 GiB at 1.75 GiB/s.
- OS: Windows 11 26H2 (build 26300); WSL components 3.0.1.0 with the
  Ubuntu distro running in WSL2 mode
  (kernel 6.18.40.1-microsoft-standard-WSL2), Ubuntu 24.04.2 LTS.
  Driver 616.92, nvidia-smi 615.71.08, CUDA UMD 13.4; CUDA toolkit 13.0
  for the build.
- Strata commit fb58e0db (v0.1.41), source build via wsl-build.py
  (135/135 objects); engine 0.1.41 (src 62eb4907c92a8cb7).
- Background workloads: none; the server had been idle (last request
  more than 10 hours earlier).

## Model and configuration

- Model repository: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, IQ3_S/
  (3.50 bpw). Shard sizes: 54,817,524,224 B + 28,800,138,432 B;
  mmproj BF16 907,543,008 B. Repository revision was not recorded at
  download time. Vision encoder disabled.
- Custom pack: Strata-data/packs/iq3_s (dense 1.5 GB), MTP rt/ runtime,
  24576-pair expert profile; prepared with
  `setup.py --setup --family qwen --model IQ3_S --experimental-speed-projection on`.
- Context 131072; KV int8; expert cache auto (3627 slots, 6.94 GiB VRAM,
  PROFILE-ranked, prefilled in 0.7 s); prefill auto; spec 4 on the
  command line (adapted to spec 6 / mtp_max 4 at runtime); spec_min_p 0.50.
- Reasoning: `reasoning_effort: none` per request; sampling and
  temperature left at defaults.
- ESP enabled: control vector `project`, `per-layer`, layers 4-44
  (41 steered), scale 1.0. A refusal A/B with ESP on/off is included below.
- No manual pool-worker, VRAM-reserve, or calibration tuning.

```text
serve/server.py --engine strata --config strata-iq3_s.json --port 8083
  --host 0.0.0.0 --gpu 1 --fit-max-tokens --api-key <redacted>
```

Full sanitized config: `config-stock.json`. Note the config file carries
`"gpu": [1, 0]` (a layer split written by setup), but the launcher's
`--gpu 1` overrode it: nvidia-smi showed GPU0 at 0 MiB and GPU1 at
~15.7 GiB, i.e. single-GPU serve. The engine log indexes the serving
card as "GPU 0: NVIDIA GeForce RTX 5060 Ti".

## Method

- `benchmark-iq3s.sh` (packaged, run 2026-10-09 01:14 JST): a warm-up
  ("Say hello.", 8 tokens), then each condition once with a unique prompt
  (0 reused tokens every time): two 1100-word generations (story about a
  lighthouse keeper, factual article on tides; max_tokens 1400), two
  prompt reads (unique nonsense-word documents of 3,820 and 19,196 actual
  tokens; max_tokens 16/24, reply limited to the first word), and one
  refusal probe (dark-fantasy execution scene) sent twice with
  `experimental_speed_projection` true and false (max_tokens 500).
- Timing boundaries: the engine log's own lines
  (`prompt N tokens = R reused + M read in X ms (Y tok/s),
  G generated in Z ms (W tok/s)`); client-side wall clock is also listed
  and includes ~2-4 s of HTTP/python overhead versus the engine sums.
- Engine 0.1.41 caveat: prompt chunks under 1024 tokens partly run on
  the CPU (`STRATA_PREFILL_CPU_SHARE` default), so the 47-48-token prompt
  reads below carry that caveat; long prompts and decode are unaffected.
- Memory: nvidia-smi plus a `/metrics` snapshot after the runs
  (`metrics-stock.json`).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| decode-story (1100-word lighthouse story) | 48 | 0 | 1400 (finish=length) | 1 | 45.4 (single run) | 48.2 (single run) | not measured |
| decode-explan (1100-word tides article) | 47 | 0 | 1284 (finish=stop) | 1 | 41.9 (single run) | 47.7 (single run) | not measured |
| prompt-4k (unique nonsense-word document) | 3820 | 0 | 5 | 1 | 343.0 (single run) | n/a (5 output tokens) | not measured |
| prompt-20k (same, longer document) | 19196 | 0 | 4 | 1 | 598.3 (single run) | n/a (4 output tokens) | not measured |

Per-request rows: `requests.csv`. Per-condition medians (each of one run):
`measurements.csv`. Engine timing excerpts: `engine-logs/`.

- Draft acceptance: story 737/1256 (58.7%), explan 728/1180 (61.7%).
- Decode expert cache hit rate: 86.1% best, 76.6% on the other run.
- ESP refusal A/B on the same dark-fantasy prompt: ESP on generated
  500 tokens (finish=length, scene written to the cap); ESP off stopped
  at 106 tokens (finish=stop) with a refusal text offering a milder
  rewrite. ESP demonstrably avoids the refusal on this aligned
  (non-uncensored) model.
- VRAM free: 379 MiB at measure time (~180 MiB later under normal desktop
  load; drifts with other VRAM users, as documented for this box).
- No failures, no skipped cases, no client-side retries in these runs.

## Correctness and limitations

- No needle recall test (not measured); vision not tested; no soak run;
  TTFT not measured; no temperature sampling.
- Single run per condition: no median/range and no repeatability
  assessment (in particular not for short-prompt reads under the 0.1.41
  CPU-share default).
- WSL2 + virtiofs: these numbers describe that stack, not native Linux
  on the same card.
- Untested: other quantizations, contexts above 131072, a layer split
  across the A4000, `--batch`, calibration sweeps, concurrent requests.
