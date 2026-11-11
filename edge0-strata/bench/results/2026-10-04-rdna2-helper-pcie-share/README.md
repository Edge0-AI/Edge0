# 2x RX 6900 XT (gfx1030): a helper GPU's experts and the PCIe share (2026-10-04)

With a second GPU holding experts (`--expert-cache-device1`, `--remote-expert-opt`), the verify window's GPU plan counted
the experts that helper holds among the layer's misses and moved part of them - the PCIe share, `pcie_frac` of the misses
- to the primary card over PCIe, where the primary computed them while the helper sat idle for them. The plan runs before
`RemoteExperts::begin()` claims its rows, and `begin()` only takes the rows the plan left alone. #854 makes the plan treat
a helper's experts as it treats the peer tier's: not a miss, never in the PCIe share. The numbers below are the engine's
own `STRATA_DECODE_TIMING=1` report per request.

## Rig

- 2x AMD Radeon RX 6900 XT 16 GB (gfx1030), each PCIe 4.0 x8 through the CPU; no P2P. The engine's probe measured 14.1
  GB/s host to device and set `pcie_frac` to 0.39 (0.55 is the default without a probe). AMD Ryzen 5 5600X (6 cores,
  AVX2, no AVX-512; 5 expert-pool workers plus the host thread), 128 GB DDR4-3200.
- Ubuntu 26.04, kernel 7.0.0-38-generic, ROCm 10.0.0 (HIP 7.15.26333).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S (2 shards, `--native`), MTP draft layer on, `--kv int8 --kv-resident 32768
  --max-context 131072`. Configuration [strata-helper.json](strata-helper.json) (`/m` the model directory, `/work` the
  build; `HIP_VISIBLE_DEVICES=1,0` makes the card on PCI bus 0f the primary): `--expert-cache auto
  --expert-cache-device1 auto --remote-expert-opt --spec 4 --spec-min-p 0.5`. The primary's cache holds 4,861 experts
  (9.26 GiB), the helper 8,163 more (15.46 GiB); 24,576 experts in all.
- Engines, all `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DSTRATA_PREFILL_MMQ=ON`:
  - *baseline*: `origin/main` at `6f32ec0` (0.1.39) with #835 (`6add1a7`; #835 changes the prompt path's GEMMs only,
    not the decode path measured here - its prompt tok/s are therefore higher than the fix's below);
  - *baseline, same base* (2026-10-05): `6f32ec0` unchanged;
  - *fix*: `6f32ec0` with #854 (`ffca1b0`).

## Method

A fresh server per arm. Two requests of about 5,000 prompt tokens each (two offsets into one text), "Summarize the
document above in about 300 words", sampled (temperature 1.0, top-p 0.95, top-k 20), up to 600 tokens; the engine's
`strata decode timing` line per request. Milliseconds per verify window and its parts are the measure; tok/s also
depends on how many drafts a sampled answer accepts and varies more.

## Results

| arm | ms / window | of which: GPU-reach wait | per-layer host | PCIe entries / layer-window | CPU entries / layer-window | decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (`6add1a7`), probed share 0.39 | 59.6 / 58.5 | 43.4 / 43.7 | 8.6 / 8.3 | 4.00 / 3.99 | 0.83 / 0.43 | 40.6 / 39.3 |
| baseline, same base (`6f32ec0`), probed share 0.39 | 60.4 / 57.8 | 43.7 / 43.0 | 8.8 / 8.3 | 4.07 / 3.91 | 0.95 / 0.38 | 41.0 / 37.5 |
| **fix, probed share 0.39** | **37.2 / 33.8** | 19.5 / 18.6 | 9.9 / 9.1 | 0.26 / 0.06 | 1.40 / 0.57 | **65.6 / 67.7** |
| fix, probed share 0.39, run 2 (2026-10-05) | 35.2 / 33.8 | 18.8 / 18.7 | 9.3 / 9.1 | 0.20 / 0.06 | 1.17 / 0.56 | 63.0 / 64.4 |
| fix, `--pcie-frac 0` | 36.5 / 35.1 | 18.8 / 18.8 | 10.1 / 9.5 | 0 / 0 | 1.61 / 0.84 | 66.9 / 67.1 |
| baseline (`6add1a7`), `--pcie-frac 0` (the workaround) | 34.6 / 34.1 | 18.2 / 18.7 | 9.5 / 9.3 | 0 / 0 | - | 60.5 / 67.7 |

"GPU-reach wait" is the host waiting for the GPU to reach the next layer: with the share on, the primary spent 24 ms more
per window reading and computing about 4 experts per layer-window that the helper also held (the "PCIe entries"
column), and the helper waited. With the fix the share keeps only experts no GPU holds: 0.06-0.26 per layer-window here,
and the probed share costs nothing against `--pcie-frac 0` while taking a little off the CPU (1.40 -> 1.61 CPU entries
without it). Before the fix, `--pcie-frac 0` was the way around it.

## Limits

One machine, one model, two prompts per arm; one server run per arm, two for the fix with the probed share (one on each day). The change
is host code and backend-independent, but only HIP and only one helper were run; three or four helpers go through the same
loop and were not tested. The prompt path does not use this plan and is unaffected by the change.
