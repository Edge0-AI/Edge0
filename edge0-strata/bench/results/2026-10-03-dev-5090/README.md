# IQ3_S single-GPU results, RTX 5090 / Ryzen 9 9950X3D

Issue: [#165](https://github.com/Niko1221/Strata/issues/165)
Engine: three passes on identical hardware and config —
v0.1.34 (prebuilt), v0.1.38 (source build, sm_120), v0.1.38 with `STRATA_PF_FUSED=1`.
Status: submission-shaped; fills the "IQ3_S single-GPU results remain outstanding" gap from the 3090/EPYC notes.

## Hardware

See [`hw.txt`](hw.txt). Summary: RTX 5090 32 GB (450 W configured cap), Ryzen 9 9950X3D, 89 GiB RAM, native Ubuntu, CUDA 13.3.

## Method

Native Linux setup via the shipped setup path, IQ3_S pack + native experts, `--expert-cache auto`, `--prefill auto`, MTP spec decode (`--spec 4`, `--spec-min-p 0.70`), int8 KV with 32768 resident, `--pcie-frac 0.00`, conversation cache 8192 MiB / 4 slots, max-context 131072. Identical serve args across all three passes; only the engine binary and the fused-kernel env flag differ.

The local API harness is the same one used in the 3090/EPYC run (`strata-bench.py`, patched copy in `data/`): public Strata source tree as prompt corpus, generated tasks, `temperature: 0`, `reasoning_effort: none`, `max_tokens: 256`, unique request-id prefixes per cell, three runs per speed cell, medians summarized, actual `usage.prompt_tokens` recorded.

Cleanliness: every cell in the final matrices was run with no other client on the engine, and every engine log line shows `0 reused` cold prefill. Two earlier passes were contaminated by a second client sharing the engine and were fully overwritten. Engine/API TTFT agreement is within ~1% on every retained cell.

## Harness fix worth upstreaming

The shipped `strata-bench.py` sizes prompts with a 3.4 chars/token estimate. On this corpus the real ratio is ~3.0, so the 128k cell overshot: a 129,792-token target tokenized to 148,496 tokens and the server returned HTTP 400 (`prompt + max tokens exceeds the context; requests are never truncated`). The patched copy parses the exact prompt token count from the 400 body, rescales the char budget, and retries up to three times until the prompt fits (a single rescale still overshoots ~1% of the time). Suggested upstream change: derive the chars/token ratio from a short probe request per corpus instead of a constant, or keep the 400-rescale loop.

## Results (decode tok/s engine, median of 3)

| ctx | v0.1.34 | v0.1.38 | v0.1.38 + `STRATA_PF_FUSED=1` |
|---|---|---|---|
| 1k   | 176.5 | 168.0 | 190.9 |
| 4k   | 178.3 | 192.7 | 189.2 |
| 32k  | 160.9 | 191.7 | 181.3 |
| 128k | 143.6 | 152.3 | 164.9 |

| ctx | TTFT s (med): 0.1.34 / 0.1.38 / fused | prefill tok/s (med): 0.1.34 / 0.1.38 / fused |
|---|---|---|
| 1k   | 0.7 / 0.7 / 0.7   | 1699 / 1753 / 1723 |
| 4k   | 1.0 / 0.9 / 0.8   | 4308 / 4846 / 5361 |
| 32k  | 6.0 / 5.3 / 5.0   | 5301 / 6036 / 6324 |
| 128k | 23.5 / 22.5 / 21.7 | 5607 / 5862 / 6092 |

Observations:

- v0.1.38 over v0.1.34: prefill gains at every context (largest at 4k/32k, +13%/+14%), decode mixed but median up at 32k/128k. Draft acceptance counts per run are in `data/` (#463 territory).
- `STRATA_PF_FUSED=1` on IQ3_S: prefill up again at 4k/32k/128k (+11%/+5%/+4% over default 0.1.38). Decode medians move within run-to-run spread (per-run decode spread is 139-214 tok/s in every pass), so no decode claim is made either way.
- Run-to-run decode spread is wide (±15%) at fixed config; medians of 3 are the honest unit of comparison here.
- 128k prefill holds ~5,900-6,100 tok/s with int8 KV and 32768 resident KV; the 131072 cap is the binding limit for prompt sizing.
- No needle runs, no calibration comparison, no under-load power capture in this pass.

## Files

- `IQ3_S-1x5090/` — v0.1.34 matrix + per-run data
- `IQ3_S-1x5090-0138/` — v0.1.38 default matrix + per-run data
- `IQ3_S-1x5090-0138-fused/` — v0.1.38 `STRATA_PF_FUSED=1` matrix + per-run data
- `strata-bench.py` — patched harness (400-rescale loop)
- `hw.txt` — hardware capture
- `BUILD-0.1.34.json`, `BUILD-0.1.38.json` — engine build stamps
- `strata-iq3_s.json`, `strata-iq3_s.fused.json` — serve configs (default / fused)
- `engine-0.1.34.log`, `engine-0.1.38.log` — engine log tails
