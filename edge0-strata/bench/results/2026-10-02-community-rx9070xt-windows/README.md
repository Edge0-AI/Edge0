# Community results: RX 9070 XT 16 GB on Windows 11, the ready-made AMD engine (0.1.35), IQ3_S

A run of the release's `strata-windows-x64-hip.zip` on a discrete card. [AMD_HIP.md](../../../docs/AMD_HIP.md#windows)
says the ready-made zip "has not run a model on a discrete card yet - please report"; this is that report, measured
the way [COMMUNITY_BENCHMARKS.md](../../../docs/COMMUNITY_BENCHMARKS.md) suggests.

## Machine

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB (gfx1201), Windows driver version 32.0.31041.3013 |
| CPU | AMD Ryzen Threadripper 3960X, 24 cores / 48 threads (AVX2, no AVX-512) |
| RAM | 128 GiB, 8x DDR4-3200 (quad channel) |
| OS | Windows 11 Pro 10.0.26200 |
| Storage | model files on drive F: (local disk) |
| PCIe | engine probe: 28.6 GB/s host to device |

## Software and settings

- Strata `main` at `d9ab843` (engine 0.1.35), installed with
  `START-HERE.bat --yes --family qwen --model IQ3_S --no-start --data-dir <folder>\Strata-data`.
  Setup downloaded the ready-made AMD engine ("ready-made AMD engine 0.1.35 for gfx1100, gfx1101, gfx1102, gfx1200,
  gfx1201, gfx1030 (ROCm 10.2.0a20260930)"); nothing was compiled.
- The engine loaded the bundled HIP runtime from `engine\` (`strata generate: HIP runtime <folder>\Strata\engine\amdhip64_7.dll`),
  so the #468 placement took effect.
- Model: Qwen3.8-Flash-Next GSQ-RCO **IQ3_S** (ISTA-DASLab), both GGUFs, pack and MTP layer prepared by setup.
- Config as written by setup: [strata-iq3_s.json](strata-iq3_s.json) (paths shortened to `<folder>`; no API key).
  Setup picked a **65,536-token context** ("recommended for your GPU"), KV int8, 32,768 KV cells resident in VRAM,
  `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, MTP on, hipBLASLt tuning table
  `tools\hip\gfx1201-hipblaslt-100500.txt`. Images off (not available on Windows AMD yet). No calibration, no speed
  projection.
- Server: `.venv\Scripts\python.exe serve\server.py --engine strata --config strata-iq3_s.json --port 8080`, started
  fresh for this run, nothing else using it.

## Workload

[bench.py](bench.py) (standard library only) against the OpenAI endpoint, streaming, temperature 0,
`"reasoning_effort": "none"`. Every request starts with a unique nonce, so no request reuses a cached prefix (the
engine log confirms `0 reused` for every measured prompt). One warm-up request (38 prompt tokens, 8 generated), then
three runs of each test:

- **decode**: a short coding prompt (72 tokens), 256 tokens out (all three runs produced the full 256).
- **prefill 4K / 32K**: deterministic synthetic Python code, 3,927 / 31,059 prompt tokens, 1 token out.

Per-run client results: [data/runs.json](data/runs.json). Engine log of the whole session:
[data/strata-iq3_s.log](data/strata-iq3_s.log).

## Results (engine timing lines; median [range] of 3 runs)

| Test | Prompt tokens (reused) | Prompt read tok/s | Decode tok/s | Drafts accepted | Decode expert cache hits |
|---|---:|---:|---:|---:|---:|
| decode | 72 (0) | - | **45.2** [37.1-46.4] | 84.5% [84.1-86.3] | 84.6% [67.0-86.2] |
| prefill 4K | 3,927 (0) | **370.4** [366.8-370.9] | - | - | - |
| prefill 32K | 31,059 (0) | **601.5** [595.4-602.7] | - | - | - |

- The first decode run (37.1 tok/s, cache hits 67.0%) is the expert cache adapting after the warm-up; runs 2 and 3
  give 45.2 and 46.4 tok/s at 84.6% / 86.2% hits.
- The client-side numbers in `runs.json` agree within 1%: decode 45.2 [37.2-46.5], prefill 367.9 [364.4-368.4] and
  599.5 [593.4-600.9] tok/s.
- Time to first token (client): 10.7 s for the 4K prompt, 51.7-52.3 s for the 32K prompt.
- **Short prompts read slowly:** the 72-token decode prompt was read at 14.7 / 22.8 / 21.9 tok/s (engine), so the
  first token took 3.2-4.9 s (client). On the RTX 2080 Ti machine below the same prompt's first token took 1.6 s.
- KV streaming: 99.4% of block reads hit VRAM while decoding, 80.5% (4K) and 73.1-73.3% (32K) during prompt reads.

## Memory

- VRAM: expert cache 4,983 slots in 9.48 GiB, MTP draft layer 835 MiB; the server reports "462 MiB of VRAM free with
  everything loaded". The prompt path borrows 2,385 cache slots (4.50 GiB) while reading. Peak usage not recorded
  separately.
- RAM: expert arena 46.84 GiB, loaded at 3.25 GiB/s. **Windows refused large pages** for it:
  `large pages refused for 50295996416 B (GetLargePageMinimum=2097152, VirtualAlloc error 1314); using 4 KB pages`.
  Error 1314 is `ERROR_PRIVILEGE_NOT_HELD`: the account has no "Lock pages in memory" right (the Windows default).
  The engine continued with 4 KB pages; whether granting the right changes the speed was not tested.

## Answers to the AMD_HIP.md checklist

- Card and driver: above.
- `strata-device --list-devices` / `--selftest` (with `engine\rocm\bin` on the PATH): the card is listed as device 0,
  gfx1201, 15.9 GiB, wave32 (engine compiled for gfx1100, gfx1101, gfx1102, gfx1200, gfx1201, gfx1030), HIP runtime
  `engine\amdhip64_7.dll`, and `strata-device selftest OK`. Full output:
  [data/strata-device.log](data/strata-device.log).
- End of `strata-iq3_s.log` and the speed lines: in [data/strata-iq3_s.log](data/strata-iq3_s.log); the session ran
  without errors from setup to the last request.

## Notes

- **Setup first downloaded the model again.** The Strata folder was a fresh clone with the model files copied into
  `Strata-data` next to it (with their `.done` marks). `%APPDATA%\Strata\settings.json` still had a `data_dir` from an
  earlier install on this PC, and setup used that folder instead, so the copied files were not seen and the 54.8 GB
  IQ3_S download started. `--data-dir <folder>\Strata-data` fixed it ("already downloaded" for both GGUFs). Behaviour
  as designed (the remembered folder wins), but a check of `Strata-data` next to the Strata folder for complete files
  before downloading would have avoided the download.
- For orientation only, not a like-for-like comparison: the same CPU and RAM with an RTX 2080 Ti 11 GB on Linux
  (engine 0.1.34, CUDA, 262,144-token context) measured 56.0 tok/s decode and 839 / 783 tok/s prompt reads with the
  same `bench.py` workload. OS, backend, engine version and context setting all differ, so this does not isolate the
  card.
- Not measured: correctness checks (needle test), contexts above 32K, model load time, power.
