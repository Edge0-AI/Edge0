# Community benchmark on RTX 5080 + Ryzen 9 9900X3D

Measured on 2026-10-05 by kalabaddon.

* **Machine:** RTX 5080 16 GB at **PCIe 5.0 x8**, Ryzen 9 9900X3D, 96 GiB DDR5-6000, Windows 11 25H2, Strata engine 0.1.39 (commit `6f32ec0`).
* **Tested:** four models in five configurations: IQ3_S, Unsloth UD-IQ4_XS (twice, before and after `START-HERE.bat --calibrate`), Unsloth UD-Q4_K_XL, and Coder IQ1_M.
* **Speed test:** 3 runs at each of 1k, 4k, 32k, 64k, 128k and 262k prompt tokens, 256-token output cap, greedy, reasoning off.
* **Needle test:** 9 cases (32k / 128k / 256k at 10 / 50 / 90% depth) on IQ3_S, calibrated UD-IQ4_XS, UD-Q4_K_XL and Coder IQ1_M.
* **Main caveat:** many answers stopped before the 256-token cap, so decode figures are less reliable than prompt figures.

## Summary

All four models found 9 of 9 needles. Speed, engine-reported medians (prompt range across 4k-262k, 1k shown separately because it is much lower for all of them; decode range across all six sizes):

| Configuration | Prompt tok/s, 4k-262k | Prompt tok/s, 1k | Decode tok/s | TTFT at 262k |
|---|---|---|---|---|
| IQ3_S | 2,344-3,603 | 674 | 80-98 | 80.6 s |
| UD-IQ4_XS, before calibration | 1,498-2,424 | 483 | 39-51 | 174.9 s |
| UD-IQ4_XS, calibrated | 1,480-2,410 | 479 | 52-63 | 174.8 s |
| UD-Q4_K_XL | 1,238-1,861 | 382 | 44-51 | 179.8 s |
| Coder IQ1_M (reduced quant, see below) | 3,175-4,001 | 1,431 | 86-117 | 73.6 s |

* **IQ3_S** was the fastest of the full models at every size, in prompt speed, decode speed and TTFT: from 4k up, about 1.5-2.2x the prompt speed and 1.4-2.5x the decode speed of the UD-IQ4_XS passes, and less than half their TTFT at 262k.
* **UD-Q4_K_XL** had the slowest prompt speed of the five at every size and the longest TTFT. Its decode speed was close to the uncalibrated UD-IQ4_XS and below the calibrated pass. It also had the largest expert arena (66,482 MiB) and the least RAM headroom (85.5 of 93.6 GiB in use, engine reading).
* **Calibration (UD-IQ4_XS)** changed only `--pcie-frac` (0.55 to 0.35). Decode medians rose 13-44%; prompt speed moved -2% to +8%.
* **Coder IQ1_M** is the Coder variant at the much smaller IQ1_M quant (expert arena 23,981 MiB, against 47,962-66,482 MiB for the others), aimed at more limited systems. It was tested in the hope of a large speed gain, but its lead over IQ3_S was modest: prompt speed about 9-11% higher from 32k up (35% at 4k, 112% at 1k), decode mixed (higher at most sizes, lower at 4k and 128k). It also differs from IQ3_S in `--spec-min-p` (0.50 vs 0.70) and expert profile, so even that gap is not down to the quant alone.

**Runs that wrote the full 256-token answer** (instead of stopping early, usually with a line like "I'll start by exploring the repository"). This is only a proxy for following the task: some full-length IQ3_S answers began by saying the file was truncated.

| Context | IQ3_S | UD-IQ4_XS, before calibration | UD-IQ4_XS, calibrated | UD-Q4_K_XL | Coder IQ1_M |
|---|---|---|---|---|---|
| 1k-64k (of 12) | 4 | 12 | 12 | 11 | 3 |
| 128k (of 3) | 1 | 2 | 1 of 2 \* | 1 | 2 |
| 262k (of 3) | 0 | 0 | 0 | 1 | 3 |
| Total (of 18) | 5 | 14 | 13 of 17 \* | 13 | 8 |

\* One calibrated 128k run failed with an engine error and is excluded (see Results).

**Conditions that limit the comparison:** PCIe 5.0 x8 rather than x16; only 249 MiB of VRAM free with IQ3_S loaded at 262,144 context; RAM nearly full with UD-IQ4_XS and UD-Q4_K_XL; one prompt per size; one failed request (calibrated UD-IQ4_XS, 128k); short answers make decode figures noisier. Details under Limitations.

## Hardware and software

* **GPU:** NVIDIA GeForce RTX 5080, 16 GB (16,303 MiB reported), the only GPU. The display is attached to it (`nvidia-smi` shows Disp.A On), so the desktop uses some VRAM.
* **PCIe link:** Gen 5 at **x8** (the card supports x16). `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max`, sampled once a second for 15 seconds during a request, read 5, 5, 8, 16 every time. Cause, per the owner: lane sharing with the NVMe drives (not independently verified).
* **CPU:** AMD Ryzen 9 9900X3D (AM5).
* **RAM:** 96 GiB DDR5 (2x48 GiB Corsair Vengeance, 2 of 4 slots) at 6000 MT/s per Task Manager; 93.6 GiB usable per the engine.
* **Storage:** Samsung 970 EVO Plus 2 TB NVMe, the boot drive and the only drive used; negotiated link speed not checked.
* **Software:** Windows 11 25H2 (build 26200.9457); NVIDIA driver 610.47 (WDDM), CUDA 13.3 per `nvidia-smi`; Python 3.14. Strata engine 0.1.39 (startup log "session is up (engine 0.1.39)", matching the checkout's `CMakeLists.txt`), commit `6f32ec0`; release binary or source build: not recorded; no changed build options.
* **Background:** Firefox and a few other apps open; Steam and other gaming services stopped; memory use not otherwise minimized.
* **Memory snapshot with UD-IQ4_XS loaded** (Task Manager, one snapshot, not a peak during inference): 88.6 GiB in use (1.2 GiB compressed), 5.1 GiB available, 5.0 GiB cached, 97 of 157 GiB committed, 2.4 GiB hardware reserved. `nvidia-smi` at about the same time: 15,692 of 16,303 MiB VRAM in use, 51 W of a 360 W cap. No Task Manager snapshot for the other configurations; engine RAM readings for all five are in the table under Model and configuration.
* **Also available:** the machine dual-boots Linux with a software RAID (believed RAID 0) of two Crucial T705 1 TB PCIe Gen5 NVMe SSDs (rated up to 13,600 MB/s) and lower OS overhead. Not used here; I can rerun there if anyone wants a faster-storage comparison.

## Model and configuration

Models (all Qwen3.8-Flash-Next):

* **IQ3_S:** the original model (not the Coder or Swift variant), 2 shards, the first entry in setup's model menu, as provided by Strata's setup. Served as `qwen3.8-flash-next-iq3_s`.
* **UD-IQ4_XS:** Unsloth family, 3 shards, the experimental entry in setup's menu, as provided by Strata's setup. `docs/MODELS.md` on main describes this family as UD-Q4_K_XL, so UD-IQ4_XS may be newer than that page. Served as `qwen3.8-flash-next-unsloth-ud-iq4_xs`.
* **UD-Q4_K_XL:** Unsloth family, 4 shards, the model `docs/MODELS.md` names for this family. Served as `qwen3.8-flash-next-unsloth-ud-q4_k_xl`.
* **Coder IQ1_M:** Coder variant, 2 shards, from Strata's official install list. Served as `qwen3.8-flash-next-coder-iq1_m`.

Model repository and revision: not recorded; all GGUFs were downloaded by Strata's setup. Packs under `<Strata-data>\packs\` were generated by setup (no hashes recorded); expert profiles are the ones shipped in `data\`. No custom draft vocabulary, vision encoder off.

Engine launch flags. Paths are relative to `<Strata-data>` (the data folder next to the Strata folder) or to the Strata folder.

| Flag | IQ3_S | UD-IQ4_XS, before calibration | UD-IQ4_XS, calibrated | UD-Q4_K_XL | Coder IQ1_M |
|---|---|---|---|---|---|
| `--pack` | `<Strata-data>\packs\iq3_s` | `<Strata-data>\packs\unsloth-ud-iq4_xs` | same | `<Strata-data>\packs\unsloth-ud-q4_k_xl` | `<Strata-data>\packs\coder-iq1_m` |
| `--native` | `<Strata-data>\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `<Strata-data>\models\unsloth-UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | same | `<Strata-data>\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | `<Strata-data>\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` |
| `--ple-gguf` | second shard of the model | not used | not used | not used | `<Strata-data>\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` |
| `--expert-profile` | `data\expert-profile.bin` (shipped) | `data\expert-profile.bin` (shipped) | same | `data\expert-profile.bin` (shipped) | `data\expert-profile-coder.bin` (shipped, Coder-specific) |
| `--expert-cache` | auto | auto | auto | auto | auto |
| `--prefill` | auto | auto | auto | auto | auto |
| `--spec` | 4 | 4 | 4 | 4 | 4 |
| `--mtp` | `<Strata-data>\mtp\rt` | `<Strata-data>\mtp\rt` | same | `<Strata-data>\mtp\rt` | `<Strata-data>\mtp\rt` |
| `--max-context` | 262144 | 262144 | 262144 | 262144 | 262144 |
| `--kv` | int8 | int8 | int8 | int8 | int8 |
| `--kv-resident` | 32768 | 32768 | 32768 | 32768 | 32768 |
| `--resident-budget-gib` | not set | 55 | 55 | 66 | not set |
| `--pcie-frac` | 0.20 | 0.55 † | 0.35 | 0.20 | 0.20 |
| `--spec-min-p` | 0.70 | 0.50 † | 0.50 | 0.70 | 0.50 |

Captured engine arguments for UD-Q4_K_XL, with local paths shortened (the other launches differ only as shown in the table):

```text
--pack <Strata-data>\packs\unsloth-ud-q4_k_xl
--native <Strata-data>\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
--expert-profile data\expert-profile.bin
--expert-cache auto --prefill auto --spec 4
--mtp <Strata-data>\mtp\rt
--max-context 262144 --kv int8 --kv-resident 32768
--resident-budget-gib 66 --pcie-frac 0.20 --spec-min-p 0.70
```

† Reported as in effect for that run (the flag was absent from its config); only the calibrated pass's flag list was captured directly. All other columns are the flag lists as launched.

Engine-reported state at the first 1k run of each configuration (from the metrics saved in each run's JSON):

| Engine-reported | IQ3_S | UD-IQ4_XS, before calibration | UD-IQ4_XS, calibrated | UD-Q4_K_XL | Coder IQ1_M |
|---|---|---|---|---|---|
| Expert cache slots (VRAM) | 4,010 (7,835 MiB) | 3,040 (7,009 MiB) | 2,941 (6,781 MiB) | 2,335 (6,967 MiB) | 3,973 (7,750 MiB) |
| Expert arena (RAM) | 47,962 MiB | 49,753 MiB | 49,981 MiB | 66,482 MiB | 23,981 MiB |
| VRAM free after load | 249 MiB | 99 MiB | 464 MiB | 176 MiB | 383 MiB |
| RAM in use (of 93.6 GiB) | 63.1 GiB | 68.8 GiB | 67.7 GiB | 85.5 GiB | 45.0 GiB |

Other settings: reasoning off (`reasoning_effort: none`), greedy (temperature 0), vision off, CPU workers 11 (engine-reported `pool_workers`; not set on the command line). Low-RAM mode and the experimental speed projection were not checked; the IQ3_S startup log has no control-vector line, which the Strata docs say appears when the projection is on.

## Method

* **Script:** `strata-bench.py` from `bench/results/2026-09-29-rtx3090-epyc-milan/data/` (stdlib only), run from the Strata repo root against the running server, once per configuration with its own `--label` and `--log`. The UD-Q4_K_XL command is shown; the others changed only `--label` and `--log` (labels `iq3_s`, `unsloth-ud-iq4_xs`, `unsloth-ud-iq4_xs_calibrated`, `coder-iq1_m`). Key removed:

```text
python bench\results\2026-09-29-rtx3090-epyc-milan\data\strata-bench.py --url http://127.0.0.1:8080 --api-key <API_KEY> --label unsloth-ud-q4_k_xl --contexts 1k,4k,32k,64k,128k,262k --log strata-unsloth-ud-q4_k_xl.log --out bench\results\2026-10-05-community-rtx5080-9900x3d
```

* **Prompt:** text from the Strata repository's own source and docs, cut by character count, followed by "explain what the last file above does, then propose one concrete improvement as a unified diff". Each run has a unique prefix, so the engine log shows 0 reused tokens in every run.
* **Output:** 256-token cap, streaming API, greedy.
* **Repetitions:** 3 per size; tables show median (range). No separate warm-up; the 1k runs came first. The expert cache adapts during a run and its state carried over between runs. Model loading is not timed.
* **Timing:** TTFT is client-side, from sending the request to the first content or reasoning token (with reasoning off, the first answer token). Total request time is client-side from send to the end of the stream, excluding model loading. Prompt and decode tok/s are the engine's own log-line rates (freshly read prompt tokens / ms, generated tokens / ms), not total request time divided by tokens. API-side figures are in each `matrix.json` as a cross-check.
* **Memory:** not logged during runs; only the snapshots above (Task Manager snapshot with UD-IQ4_XS loaded; engine RAM and VRAM readings at each configuration's first 1k run). No paging or out-of-memory failures were seen.
* **Units:** RAM in GiB (Task Manager's "GB" are binary; engine byte counts are converted to GiB), VRAM and expert memory in MiB.

## Results

Prompt tokens and generated tokens are the engine log counts for each run; reused tokens were 0 in every run, and a generated count of 256 means the run hit the output cap. Prompt and decode tok/s are engine-reported; TTFT and total request time are client-measured. Format: median (range). Every request completed except the one marked failed.

### IQ3_S

| Context | Prompt tokens | Generated tokens, runs 1 / 2 / 3 | Prompt tok/s | Decode tok/s | TTFT s | Total s |
|---|---|---|---|---|---|---|
| 1k | 1,042-1,143 | 174 / 8 / 256 | 673.8 (618.7-676.0) | 80.3 (59.9-94.7) | 1.71 (1.70-1.72) | 3.86 (1.83-4.41) |
| 4k | 4,184-4,450 | 256 / 256 / 96 | 2,344.3 (2,337.5-2,422.0) | 93.3 (87.9-97.2) | 1.84 (1.83-1.87) | 4.45 (2.91-4.59) |
| 32k | 31,513-31,601 | 112 / 83 / 49 | 3,603.3 (3,590.9-3,607.6) | 98.1 (96.9-105.0) | 8.86 (8.85-8.87) | 9.63 (9.34-10.01) |
| 64k | 64,376-67,076 | 256 / 87 / 94 | 3,596.1 (3,382.4-3,601.1) | 85.5 (83.5-104.3) | 18.08 (18.04-20.01) | 19.11 (18.93-22.99) |
| 128k | 129,580-132,302 | 108 / 57 / 256 | 3,488.3 (3,377.3-3,496.0) | 90.2 (78.6-98.8) | 37.43 (37.42-39.46) | 40.54 (38.05-40.67) |
| 262k | 257,832-261,112 | 183 / 53 / 77 | 3,255.5 (3,242.7-3,259.9) | 91.0 (86.3-98.1) | 80.63 (80.08-80.78) | 81.46 (81.30-82.18) |

### UD-IQ4_XS, before calibration

| Context | Prompt tokens | Generated tokens, runs 1 / 2 / 3 | Prompt tok/s | Decode tok/s | TTFT s | Total s |
|---|---|---|---|---|---|---|
| 1k | 1,042-1,144 | 256 / 256 / 256 | 482.9 (198.4-508.1) | 48.3 (46.8-49.2) | 2.42 (2.27-5.33) | 7.59 (7.55-10.76) |
| 4k | 4,185-4,451 | 256 / 256 / 256 | 1,503.5 (702.2-1,518.1) | 51.3 (49.9-51.3) | 2.84 (2.83-6.40) | 7.94 (7.79-11.36) |
| 32k | 31,487-31,600 | 256 / 256 / 256 | 2,214.5 (1,474.8-2,239.1) | 39.0 (38.6-47.6) | 14.37 (14.21-21.48) | 20.75 (19.72-28.08) |
| 64k | 64,377-67,074 | 256 / 256 / 256 | 2,423.8 (1,563.4-2,451.0) | 47.0 (46.3-48.8) | 26.75 (26.51-43.10) | 32.25 (31.73-48.53) |
| 128k | 129,581-132,301 | 87 / 256 / 256 | 2,285.0 (1,707.0-2,318.0) | 39.6 (36.1-43.0) | 57.00 (56.36-77.82) | 64.08 (62.80-79.81) |
| 262k | 257,822-261,110 | 94 / 125 / 67 | 1,497.5 (1,497.1-1,813.4) | 50.6 (44.1-53.2) | 174.89 (142.77-175.01) | 176.13 (144.87-177.46) |

### UD-IQ4_XS, calibrated

| Context | Prompt tokens | Generated tokens, runs 1 / 2 / 3 | Prompt tok/s | Decode tok/s | TTFT s | Total s |
|---|---|---|---|---|---|---|
| 1k | 1,043-1,142 | 256 / 256 / 256 | 479.4 (248.9-500.6) | 58.7 (47.9-60.8) | 2.42 (2.32-4.25) | 6.65 (6.60-9.56) |
| 4k | 4,184-4,449 | 256 / 256 / 256 | 1,480.1 (680.3-1,499.1) | 59.2 (50.9-59.8) | 2.87 (2.87-6.60) | 7.16 (7.12-11.61) |
| 32k | 31,507-31,604 | 256 / 256 / 256 | 2,385.0 (1,466.1-2,395.0) | 56.3 (54.9-57.0) | 13.39 (13.35-21.64) | 17.87 (17.85-26.27) |
| 64k | 64,374-67,074 | 256 / 256 / 256 | 2,410.4 (1,508.3-2,417.0) | 53.2 (50.6-57.6) | 26.98 (26.83-44.67) | 31.62 (31.40-49.71) |
| 128k \* | 129,577-132,307 | failed / 83 / 256 | 2,264.1 (2,238.0-2,290.1) | 52.1 (49.9-54.2) | 58.25 (56.97-59.54) | 61.57 (61.05-62.08) |
| 262k | 257,730-261,118 | 96 / 54 / 113 | 1,499.1 (1,494.9-1,763.3) | 63.4 (58.5-65.3) | 174.78 (146.94-175.15) | 175.59 (148.55-176.91) |

\* Run 1 failed: the engine recorded the request as finishing with an error after reading 132,308 prompt tokens (1,674.4 tok/s) and generating 51 tokens (49.6 tok/s) in 80.4 s, and the API reply carried no usage block (`matrix.json` shows `prompt_tokens: null`, 17 completion tokens, 16.0 tok/s API decode). It is excluded from this row's medians and from every summary figure; its values are listed here and in `data/speed-128k-1.json`.

### UD-Q4_K_XL

| Context | Prompt tokens | Generated tokens, runs 1 / 2 / 3 | Prompt tok/s | Decode tok/s | TTFT s | Total s |
|---|---|---|---|---|---|---|
| 1k | 1,041-1,143 | 256 / 256 / 256 | 382.0 (225.7-403.4) | 49.2 (35.3-52.5) | 3.02 (2.86-4.69) | 8.20 (7.71-11.90) |
| 4k | 4,185-4,441 | 256 / 256 / 256 | 1,237.9 (714.3-1,283.6) | 48.1 (47.6-52.7) | 3.43 (3.33-6.28) | 8.73 (8.68-11.11) |
| 32k | 31,511-31,602 | 256 / 256 / 256 | 1,619.2 (1,584.5-1,728.5) | 51.0 (46.2-51.9) | 19.63 (18.40-20.01) | 24.54 (23.90-25.00) |
| 64k | 64,378-67,073 | 256 / 95 / 256 | 1,680.6 (1,426.7-1,998.7) | 45.2 (43.7-48.3) | 38.47 (32.42-47.23) | 43.75 (34.57-52.86) |
| 128k | 129,581-132,301 | 53 / 57 / 256 | 1,860.6 (1,565.0-1,883.0) | 46.8 (37.2-54.9) | 69.93 (69.26-84.89) | 76.12 (70.95-85.99) |
| 262k | 257,822-261,112 | 107 / 256 / 113 | 1,455.8 (1,446.0-1,527.7) | 44.3 (37.8-44.7) | 179.84 (169.47-181.17) | 182.37 (172.25-186.87) |

### Coder IQ1_M

| Context | Prompt tokens | Generated tokens, runs 1 / 2 / 3 | Prompt tok/s | Decode tok/s | TTFT s | Total s |
|---|---|---|---|---|---|---|
| 1k | 1,041-1,145 | 41 / 50 / 256 | 1,430.8 (1,318.8-1,447.5) | 104.9 (92.5-134.5) | 0.82 (0.81-0.83) | 1.29 (1.12-3.56) |
| 4k | 4,183-4,447 | 256 / 256 / 52 | 3,175.1 (3,140.9-3,180.2) | 89.2 (76.6-100.9) | 1.36 (1.35-1.46) | 3.88 (2.02-4.31) |
| 32k | 31,507-31,602 | 52 / 55 / 49 | 3,939.5 (3,931.3-3,947.4) | 108.1 (98.3-112.2) | 8.11 (8.09-8.13) | 8.61 (8.55-8.61) |
| 64k | 64,373-67,071 | 24 / 48 / 48 | 4,000.9 (3,829.8-4,002.2) | 116.8 (99.8-129.1) | 16.26 (16.25-17.67) | 16.66 (16.61-17.90) |
| 128k | 129,581-132,308 | 100 / 256 / 256 | 3,805.2 (3,789.7-3,847.9) | 86.2 (81.9-118.2) | 34.56 (33.96-35.05) | 37.07 (35.89-37.51) |
| 262k | 257,824-261,113 | 256 / 256 / 256 | 3,574.8 (3,555.9-3,577.7) | 94.1 (90.6-98.6) | 73.63 (72.62-73.96) | 76.33 (75.43-76.53) |

### What calibration changed (UD-IQ4_XS, medians)

| Context | Decode tok/s, before -> after | Prompt tok/s, before -> after |
|---|---|---|
| 1k | 48.3 -> 58.7 (+22%) | 482.9 -> 479.4 (-1%) |
| 4k | 51.3 -> 59.2 (+15%) | 1,503.5 -> 1,480.1 (-2%) |
| 32k | 39.0 -> 56.3 (+44%) | 2,214.5 -> 2,385.0 (+8%) |
| 64k | 47.0 -> 53.2 (+13%) | 2,423.8 -> 2,410.4 (-1%) |
| 128k \* | 39.6 -> 52.1 (+32%) | 2,285.0 -> 2,264.1 (-1%) |
| 262k | 50.6 -> 63.4 (+25%) | 1,497.5 -> 1,499.1 (0%) |

\* Calibrated 128k uses the two successful runs.

### Expert streaming

From the engine log lines across each configuration's successful runs (18, or 17 for the calibrated pass) ("N more read by the GPU over PCIe", and the decode expert cache hit rate):

| | IQ3_S | UD-IQ4_XS, before calibration | UD-IQ4_XS, calibrated | UD-Q4_K_XL | Coder IQ1_M |
|---|---|---|---|---|---|
| Routed experts read over PCIe | 2-6% (median 3%) | 14-25% (median 18%) | 10-14% (median 11%) | 5.5-8.0% (median 6.2%) | 0.3-3.6% (median 2.1%) |
| Expert cache hit rate | ~50-80% | ~50-80% | ~50-80% | 39-59% | 67-92% |

Per-run data and engine log lines are in `iq3_s/`, `unsloth-ud-iq4_xs/`, `unsloth-ud-iq4_xs_calibrated/`, `unsloth-ud-q4_k_xl/` and `coder-iq1_m/` (each has `matrix.json`, `matrix.md` and `data/`, with each run's actual prompt and generated token counts).

## Needle recall

`tools/needle_bench.py`, thinking off, greedy. All four found 9 of 9, with no errors or skipped cases. Prompts were identical across models: 32,171-32,172 tokens ("32k"), 125,449-125,451 ("128k") and 250,574-250,575 ("256k"). Results: `needles-iq3_s.json`, `needles-unsloth-ud-iq4_xs_calibrated.json`, `needles-unsloth-ud-q4_k_xl.json`, `needles-coder-iq1_m.json`. There is no needle run for the uncalibrated UD-IQ4_XS pass.

Command (UD-Q4_K_XL shown; the others changed only the `--out` file name; key removed):

```text
python tools\needle_bench.py --url http://127.0.0.1:8080 --api-key <API_KEY> --lengths 32k,128k,256k --depths 10,50,90 --out bench\results\2026-10-05-community-rtx5080-9900x3d\needles-unsloth-ud-q4_k_xl.json
```

Request time in seconds at depth 10 / 50 / 90%:

| Context | IQ3_S | UD-IQ4_XS, calibrated | UD-Q4_K_XL | Coder IQ1_M |
|---|---|---|---|---|
| 32k | 9.2 / 9.1 / 4.8 | 29.0 / 13.5 / 7.4 | 17.1 / 16.5 / 9.0 | 8.4 / 8.6 / 4.4 |
| 128k | 37.1 / 37.4 / 33.4 | 79.2 / 55.8 / 47.9 | 89.3 / 71.3 / 66.0 | 33.6 / 33.7 / 29.5 |
| 256k | 73.4 / 73.9 / 74.5 | 106.0 / 130.0 / 131.9 | 130.4 / 157.2 / 152.9 | 67.3 / 67.7 / 67.6 |

These are whole-request client times, and later depths probably reuse part of an earlier prompt's cached prefix, so use the speed tables for speed. Each needle run used the same server launch as that configuration's speed run (assumed for IQ3_S).

## Limitations

* **Short answers.** The benchmark prompt cuts the last file mid-function, and models often commented on that or stopped with a one-liner instead of writing 256 tokens (counts in the Summary). Examples: IQ3_S 1k run 2 answered "I can't help with that."; one Coder 64k run emitted a raw tool-call fragment. Decode tok/s from short answers is less reliable, so comparisons across configurations mix answer lengths. The UD-IQ4_XS before/after comparison is the cleanest, since all 12 runs at 1k-64k are full-length in both passes.
* **Task-following is not a quality measure.** At 1k-64k the UD-IQ4_XS passes and UD-Q4_K_XL nearly always wrote full answers while IQ3_S and Coder IQ1_M mostly did not; at 128k and 262k every configuration except Coder IQ1_M mostly stopped early, and calibration did not change the pattern. This is one prompt per size with greedy decoding, so it shows behaviour on this prompt only. All four models found every needle, so recall did not separate them.
* **Configurations differ in more than the model.** UD-IQ4_XS vs IQ3_S also differs in `--pcie-frac` (0.35 vs 0.20), `--spec-min-p` (0.50 vs 0.70) and `--resident-budget-gib 55`. UD-Q4_K_XL matches IQ3_S on `--pcie-frac` (0.20) and `--spec-min-p` (0.70) but sets `--resident-budget-gib 66`. Coder IQ1_M differs from IQ3_S in quant, `--spec-min-p` and expert profile, and was not re-run back to back with IQ3_S.
* **Warm-up.** In every UD-IQ4_XS run set, the first run at each size read its prompt more slowly than runs 2 and 3 (1k before calibration: 198 vs 483 and 508 tok/s), which looks like expert-cache warm-up; medians hide it, ranges show it. At 262k the first run was the faster one in both passes (about 1,800 vs 1,500 tok/s); I do not know why.
* **PCIe x8.** Strata streams experts over PCIe, and existing reports (for example engine 0.1.26 on an RTX 5070) list PCIe 5.0 x16, so absolute speeds here may be lower than the same card at x16. The calibrated `--pcie-frac` of 0.35 (0.55 before) is consistent with a narrower link, but untested.
* **Tight VRAM.** With IQ3_S at 262,144 context the engine reported 249 MiB of VRAM free and warned requests may stall; none did. A smaller `--max-context` or larger `--vram-reserve-mib` was not tested. Windows refused large pages for the expert arena, so it used 4 KB pages; whether granting them changes speed was not tested.
* **Tight RAM.** UD-IQ4_XS ran with 5.1 GiB available (Task Manager) and UD-Q4_K_XL with 85.5 of 93.6 GiB in use (engine), with Firefox and other apps open. The Strata docs list low free RAM as a cause of slow output, so those speeds may be lower than on a cleaner system.
* **Prompt lengths** vary by a few thousand tokens between runs of the same size, since the script builds prompts by character count.
* **Scope:** one set of 3 runs per size per configuration on one machine; no continuous memory, power or thermal logging.

