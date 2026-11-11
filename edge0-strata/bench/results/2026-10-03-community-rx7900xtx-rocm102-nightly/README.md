# Community benchmark on AMD Radeon RX 7900 XTX 24 GB — ROCm 7.1.1 vs ROCm 10.2 nightly (serve path, Qwen3.8-Flash-Next IQ3_S)

Measured 2026-10-03/04 by xyzzing. What was tested: the same serving config on two ROCm stacks —
the packaged ROCm 7.1.1 and a ROCm 10.2 nightly SDK — measuring warm steady-state decode on a
repeated identical request. Main limitation: the nightly build has no matching gfx1100 hipBLASLt
tuning table (see the confound under Results), and per-request raw server logs were lost to a log
truncation on restart — the decode values below were captured at request time from the API's
`timings` field.

## Hardware and software

- GPU: AMD Radeon RX 7900 XTX 24 GB (gfx1100, RDNA3); CPU: Ryzen 9 7900X (12C/24T); RAM: 96 GB DDR5; storage: NVMe ext4; PCIe Gen4 x16:
- OS: Fedora 44 (kernel 7.2.7); driver: amdgpu (in-tree); ROCm: packaged 7.1.1 vs a ROCm 10.2 nightly SDK (`libamdhip64.so.7.17.26392`, verified loaded via /proc/PID/maps):
- Strata: HIP source builds of the xyzzing integration line — engine 0.1.38 (`99f3dbd` base) for both stacks (one binary per ROCm stack):
- Background workloads: KDE/Wayland desktop + browser (the reason for the VRAM reserve below); no power limits:

## Model and configuration

- Model: Qwen3.8-Flash-Next GSQ-RCO (ISTA), IQ3_S GGUF (2 shards, hash `692a40d0…` shard 1):
- No vision; expert cache `auto` pre-filled from the shipped routing profile; `--vram-reserve-mib 3072`:
- Context 65536; KV int8 with `--kv-resident 32768`; prefill chunk 8192; MTP (`--mtp`, draft layer loaded); sampling: greedy (temperature 0); no calibration; no low-RAM mode:

```text
# serve (identical args on both stacks; only ROCM_PATH/LD_LIBRARY_PATH differ)
<venv python> serve/server.py --engine strata --config <cfg>.json --port <8081|8082> --host 127.0.0.1
# cfg args: --pack packs/iq3_s --native <shard1> --ple-gguf <shard2> --mtp mtp/rt
#   --expert-profile expert-profile.bin --expert-cache auto --spec 3 --kv int8 --kv-resident 32768
#   --prefill 8192 --vram-reserve-mib 3072 --max-context 65536
#   env: STRATA_PREFILL_RING=96 STRATA_GR_V3=1 STRATA_SELECT_WMMA=1 (see confound)
# request: POST /v1/chat/completions {"messages":[{"role":"user","content":
#   "Name the capital of France in one word."}],"max_tokens":200,"temperature":0}
```

## Method

One identical request repeated back-to-back per stack (≈61 prompt tokens incl. template, 200
generated tokens, greedy), served over loopback by `serve/server.py`; decode tok/s read from the
response's `timings.predicted_per_second` field (committed tokens / decode interval — not derived
from total request time). The request sequence intentionally measures the warm-up curve: request 1
is cold (fresh engine + cold expert tier), later requests are warm. Stack order: all ROCm 7
requests, then engine stop, then all ROCm 10.2 requests. No prompt reuse beyond the engine's own
cache (the same short prompt each time). Memory: engine VRAM 22.8/24 GiB with the 3072 MiB
desktop reserve (compositor + browser co-resident). **Raw server logs were truncated by a restart
between stacks; the decode values below were captured at request time** (see `runs.json`, which
also marks one stalled request).

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Decode tok/s median and range |
| --- | ---: | ---: | ---: | ---: | --- |
| ROCm 7.1.1, warm (requests 4–9) | ~61 | not recorded per run | 200 | 6 | 80.4, 78.8–81.6 |
| ROCm 7.1.1, request 1 (cold-warm-up) | ~61 | 0 | 200 | 1 | 59.3 |
| ROCm 10.2 nightly, warm (requests 6–10) | ~61 | not recorded per run | 200 | 5 | 88.4, 83.8–88.9 |
| ROCm 10.2 nightly, request 1 (cold restart) | ~61 | 0 | 200 | 1 | 58.6 |
| → toolchain delta at warm steady state | | | | | **+10.0% (88.4 vs 80.4)** |

Secondary, 128K-context one-shot (131071 prompt tokens, 256 generated, one run each — reported for
completeness, not a median): decode 62.0 (ROCm 7.1.1) vs 61.0 (nightly) tok/s — decode is O(1) in
context on this model and toolchain-neutral at this shape. **Prefill at 128K is confounded**: the
nightly's hipBLASLt is version 100500 and no gfx1100 table ships for it (100100/100200 only), so
the engine falls back to plain hipBLAS (prefill 921 vs 1535 tok/s; the version-mismatch mechanism
is documented in the AMD HIP docs). Prompt throughput 1K/4K tiers on 0.1.38: 850 / 1,366 tok/s
(earlier sweep, engine 0.1.31; `runs.json` has the 128K pair).

Failures: one stalled request on the nightly (29.0 tok/s, request 2 after a cold restart — the
documented background-GPU stall pattern; marked and excluded, kept in `runs.json`).

## Correctness and limitations

Correctness: every request returned the identical greedy answer ("Paris", finish=stop, 200-token
cap) on both stacks. Limitations: single GPU; a desktop co-resident (that is the point of the VRAM
reserve); the request is short (~10 content tokens + template), so absolute tok/s reflects decode
of a chat-sized answer, not long-completion behavior — decode is flat in context for this model
(GDN linear attention), measured 1K→128K elsewhere. Untested: k8v4 KV (incompatible with
`--kv-resident` on this line), the rocWMMA select arm (compiled out of the nightly build), long
multi-turn conversations, and prompt-processing comparisons at other hipBLASLt versions.
