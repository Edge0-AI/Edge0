# Community benchmark: Tesla V100-PCIE-32GB + Tesla P100-PCIE-16GB, P100 as expert-helper card (mixed Volta/Pascal)

Measured on 2026-10-07 by aleesposito85 (headless Proxmox container, LXC, LAN only).
This is the first mixed Volta/Pascal (sm_70 + sm_60) report for Strata: the P100
serves as the **expert-helper cache for the V100** (`--expert-cache-device1 auto
--remote-expert-opt`), not a layer split. Engine 0.1.40.2, Flash-Next IQ3_XXS at
a 262,144-token context, Qwen3.8-Flash-Next IQ3_XXS.

Median prompt throughput was **1,436.9 tok/s at 17,971 prompt tokens** and
**1,446.4 tok/s at 44,011 prompt tokens**; median decode throughput was around
**73 tok/s** at a 256-token output cap. A single 119,731-token run read at
1,410.5 tok/s. These are synthetic filler prompts with a fixed explanation task,
greedy decoding, `reasoning_effort: none`; they do not establish general answer
quality or performance on other workloads.

## Hardware and software

- **CUDA0 (primary):** NVIDIA Tesla V100-PCIE-32GB (PG500-216, compute
  capability 7.0), 32,768 MiB, 250 W power limit. PCIe **Gen3 8.0 GT/s x8**
  (electrical x8 riser limit; both cards).
- **CUDA1 (helper):** NVIDIA Tesla P100-PCIE-16GB (GP100GL, compute capability
  6.0), 16,384 MiB, 250 W power limit. Used as the expert-helper cache
  (`--expert-cache-device1 auto`): 9,188 additional experts / 15.01 GiB, results
  return through pinned host rows. The cards sit on different CPU sockets
  (separate root complexes) and there is **no P2P**.
- **CPU:** Intel Xeon E5-2699 v3 @ 2.30 GHz (36 threads). The engine selected
  35 expert-pool workers; the host thread drains too (`--host-core first`).
  This CPU has **no AVX-512** — the engine used its AVX-2 expert kernels
  (logged at startup).
- **RAM:** 128.8 GB installed; engine process holds ~46 GB resident + page
  cache. Storage: rotational SAS RAID5 (the model files, 28.8 GB PLE shard and
  the expert arena are read through the file cache; `--ple-io ram`).
- **OS:** Debian 12 container (Proxmox host kernel 6.14.11-9-pve), NVIDIA
  driver 580.95.05, CUDA 12.8 (nvcc 12.8.61). Engine built from source on this
  machine (`-DCMAKE_CUDA_ARCHITECTURES="70;60" -DSTRATA_EXPERIMENTAL_SM60=ON`;
  CUDA 13 dropped Volta, so a CUDA 12.x toolkit is required).
- **Strata:** commit `e8ca9af` (v0.1.40.2), engine 0.1.40.2, server 0.1.40.2.
  Engine binary SHA-256
  `40626ec2045c3a8f30e20e344969a9fd642eb8c55776dd411969f49b67c78baa`.
- **Background workloads:** none on the GPUs; the LAN server was otherwise idle
  during the test window. Nothing else shares VRAM (492 MiB free with
  everything loaded).

## Model and configuration

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_XXS:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (47,039,860,096 B),
  SHA-256 `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (28,800,138,432 B,
  the PLE n-gram shard), SHA-256
  `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
  (byte-identical to the IQ3_S report's shard — hardlinked here)
- `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 B), SHA-256
  `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`

See [model-sha256.txt](model-sha256.txt). The model and pack were prepared by
the unmodified installer (v0.1.39-era `setup.py`; pack format v1). MTP draft
files under `/mnt/llama/strata/mtp/rt`: experts.bin (707,788,800 B, SHA-256
`09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f`), dense.bin
(116,099,072 B, `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a`),
draft_vocab.bin (425,196 B, `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`).
The bundled expert profile was used without calibration (SHA-256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`).

- Context **262,144**; INT8 KV, 32,768 KV cells per attention layer resident on
  GPU (`--kv int8 --kv-resident 32768`).
- Expert cache `auto`: **15,352 slots / 24.90 GiB on the V100**, profile-
  prefilled, policy PROFILE; **CUDA1 helper: 9,188 additional experts /
  15.01 GiB**. Warm decode expert-cache hit rate ~100%; ~7–18% of routed
  experts are read from the helper per request.
- Prefill `auto:32768` (chosen deliberately; see below), 384-slot ring; the
  prompt path borrows 8,148 CUDA0 cache slots (13.27 GiB).
- MTP `--spec 4 --spec-min-p 0.5` with `--remote-expert-opt`.
- **`STRATA_PREFILL_CPU_SHARE=auto`** (new in 0.1.40.2, opt-in): for prompt
  chunks under 1,024 tokens the CPU pool computes the least-routed experts.
- Vision encoder enabled (GPU), warmed at 1,024 image tokens; not used by these
  benchmark prompts.
- Reasoning disabled (`reasoning_effort: none`), temperature 0, 256 generated
  tokens per speed run (all runs reached the cap: `predicted_n` 256).

```text
# engine args (from the server config; env: CUDA_VISIBLE_DEVICES=0,1,
# STRATA_PREFILL_CPU_SHARE=auto)
--serve --pack /mnt/llama/strata/packs/iq3_xxs \
  --native /mnt/llama/strata/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /mnt/llama/strata/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --expert-profile /opt/strata-src/data/expert-profile.bin \
  --expert-cache auto --prefill auto:32768 --spec 4 --spec-min-p 0.5 \
  --mtp /mnt/llama/strata/mtp/rt --max-context 262144 --kv int8 \
  --kv-resident 32768 --vision --vram-reserve-mib 700 --ple-io ram \
  --expert-cache-device1 auto --remote-expert-opt
```

The complete server config is [strata-iq3_xxs.json](strata-iq3_xxs.json).

## Method

- Script: [bench_report.py](bench_report.py) (also runnable unchanged against
  any Strata server). It sends fresh salted prompts (a `[tag-nanoseconds]`
  prefix guarantees **zero prompt reuse**; the engine logs confirm
  `0 reused + N read` on every run), output cap 256 tokens, temperature 0,
  `reasoning_effort: none`, streaming on.
- Prompt construction: a technical-report filler sentence repeated to the
  target size, plus a fixed ~220-word explanation task. Actual prompt token
  counts from the engine: **809 / 17,971 / 44,011 / 119,731** tokens.
- Sizes and repetitions: ~550 (3 runs), ~13.5K (3 runs), ~33K (3 runs),
  ~90K (1 run — marked as single-run in the table). One warm-up request
  (not counted) precedes the battery; the server had already been running with
  a warmed 100% expert cache, and every measured run is fully cold on prompt
  tokens (no reuse).
- TTFT = seconds from request send to the first content delta (streaming);
  reasoning was off, so the first token is answer text.
- Timings are the engine's own `timings` block (prompt_ms / predicted_ms);
  wall time from the client. No model loading is included (server pre-loaded).
- Memory: VRAM free after load **492 MiB** (reported by the engine);
  host RAM ~46 GB resident for the engine + page cache.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| V100+P100 helper, IQ3_XXS, ~550-token prompt | 809 | 0 | 256 | 3 | 424.5 (422.3–425.1) | 77.0 (70.8–81.6) | 1.94 (1.93–1.95) |
| … ~17.9K-token prompt | 17,971 | 0 | 256 | 3 | 1,436.9 (1,436.5–1,438.2) | 72.8 (72.8–73.3) | 12.66 (12.64–12.67) |
| … ~44K-token prompt | 44,011 | 0 | 256 | 3 | 1,446.4 (1,445.9–1,446.7) | 74.2 (73.3–76.0) | 30.79 (30.76–30.81) |
| … ~120K-token prompt (single run) | 119,731 | 0 | 256 | 1 | 1,410.5 | 69.5 | 85.79 |

Decode varies with MTP draft acceptance (higher acceptance = more tokens per
verify pass); accepted-draft fractions in these runs were 59–72%. Per-run JSON
with all engine timing blocks: [results.json](results.json); the engine's own
timing lines: [engine-timings.txt](engine-timings.txt).

**Total latency** (wall, client-side, excludes loading): 5.2 s at 809 tokens,
16.1 s at 17,971, 34.2 s at 44,011, 89.5 s at 119,731.

## Correctness and limitations

- **Needle-in-a-haystack** (`tools/needle_bench.py`, the report's own check):
  32k: found at depths 10%, 50%, 90%; 128k: found at depths 10%, 50%, 90% —
  see [needles.json](needles.json).
- This is one machine with **no P2P between the cards** (cross-socket
  host-bridge path); decode includes the helper round-trip per request.
- `--prefill auto:32768` is a deliberate non-default: with the default 8192
  chunk cap, this rig's prompt chunks ≥1,024 tokens each trigger a full
  "stream every non-resident expert" pass, and mid/long prompts lose 50–100%
  prompt speed (measured: ~20.5K 413→827 tok/s, ~33.5K 499→965; also reported
  in issue #1079). At 32768 the same prompts read in a single pass and the
  numbers above follow. This interaction is worth checking on other low-VRAM /
  helper-cache rigs.
- `STRATA_PREFILL_CPU_SHARE=auto` (0.1.40.2) measured **−23% time-to-first-token
  on sub-1,024-token prompts** on this rig (off→auto→off interleaved: 1,905/1,880
  ms → 1,462/1,440 ms), with visible output byte-identical in a fixed-prompt
  parity check; ≥2K prompts unchanged. It is enabled for the runs above.
- The helper cache costs a few percent of sustained decode versus a hypothetical
  all-resident setup; the trade was accepted for +14–25% decode versus the V100
  alone (measured in an earlier local A/B, not part of this report).
- Background: single-sequence serving (one request at a time); no `--batch`
  slots. All numbers are warm (the P100 helper fills on the first large
  request; the first ~2 big requests after a restart run 20–30% slower while it
  fills).
- Reported previously in [issue #1079](https://github.com/Niko1221/Strata/issues/1079)
  (V100+P100 helper, 0.1.39 → 0.1.40 decode +6–11%).
