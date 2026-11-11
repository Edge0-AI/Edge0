# Community benchmark on 4x AMD Radeon RX 7900 XT

Measured on 2026-10-09 by Cass (ubt26, a 4-card ROCm test rig). Strata 0.1.41, ISTA-DASLab Flash-Next GSQ-RCO
IQ3_S, 262,144-token context, layer split across all four cards. Two arms on the same machine, image, model and
prompts, differing only in five documented opt-in environment switches: **opt-ins off** (Strata defaults for those
switches) and **opt-ins on** (this machine's run profile). Each arm: fresh engine process, one cold first request,
two warm-ups, then three runs each at 4,096 / 32,768 / 128,000 prompt tokens with a 256-token cap and reasoning
off, plus six needle-in-a-haystack recall checks. Main limitations: one model on one box; no coding-quality check
beyond recall; four cards on an asymmetric Gen3 PCIe layout (below).

## Hardware and software

- GPU and VRAM: 4x AMD Radeon RX 7900 XT (gfx1100), 20 GiB (21.46 GB) each, all four used (`HIP_VISIBLE_DEVICES=0,1,2,3`)
- CPU: Intel Core i9-9900K (8 cores, 16 threads), ASUS WS Z390 PRO; RAM: 60.4 GiB visible to the OS (installed size not recorded)
- Storage: models, pack and MTP head on a Sabrent 1 TB NVMe SSD
- PCIe (read at the bridge ports, not the GPUs' internal links): three cards sit behind a PLX PEX 8747 Gen3
  switch at Gen3 x16, x8 and x8, sharing one Gen3 x16 uplink to the CPU; the fourth card hangs off the
  chipset at Gen3 x4
- Power caps (hwmon `power1_cap`): 304 W, 324 W, 304 W on the three switch-attached cards; the chipset-attached
  card reads 0 W (max 324 W), so its effective limit is unknown
- OS: Linux, kernel 7.0.0-34-generic; ROCm 10.0.0 inside the runner container (TheRock gfx110X build)
- Strata: tag `v0.1.41` (`fb58e0d`), clean tree, built from source in a container: `-DSTRATA_ENABLE_HIP=ON
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON`, `Release`, gfx1100
- Background workloads: none from clients during the runs (the engine logs show only the benchmark's requests,
  one at a time); other GPU workloads were not sampled.

## Model and configuration

- Model: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  folder `IQ3_S`; files identical to revision `ed59f92` (byte sizes and LFS sha256 match; downloaded 2026-10-07)
- GGUF files, sha256 checked against the Hub's LFS hashes (read with `O_DIRECT`):
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, 54,817,524,224 bytes, sha256 `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`, matches
  - `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf`, 28,800,138,432 bytes, sha256 `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`, matches (`--ple-gguf`)
- Strata pack `packs/iq3_s` (9 files, 1.5 GB; experts read in place from shard 1), MTP head `mtp/rt` (4 files,
  787 MB), expert profile `data/expert-profile.bin` as shipped in `v0.1.41` (sha256 `8f59b4aa…`). Manifest
  sha256 (per-file sha256 list, sorted by path): pack `d0293e9b…`, MTP `6425cce0…`. The exact pack and MTP
  preparation commands were not recorded.
- Vision encoder: enabled in the launch config (`--vision`, `--vram-reserve-mib 700`); no images were sent
- Context 262,144; KV int8; expert cache auto; prefill auto; layer split `12,24,36`; resident budget 36 GiB;
  `--remote-expert-opt` set but inactive (it acts only with helper caches, none here); MTP draft `--spec 4 --spec-min-p 0.5`; calibration and speed projection not used

```text
engine args (both arms, credentials: none)
--pack /mnt/spare/build/strata-data/packs/iq3_s --native <shard-1 GGUF> --ple-gguf <shard-2 GGUF>
--expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp /mnt/spare/build/strata-data/mtp/rt --kv int8
--resident-budget-gib 36 --remote-expert-opt --max-context 262144 --vision --vram-reserve-mib 700
layer_split 12,24,36   gpu 0,1,2,3   env STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100401.txt

opt-ins off: no further STRATA_* variables
opt-ins on:  STRATA_SPLIT_OWN=1 STRATA_STAGE_TRIM=1 STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096
```

The five switches are documented opt-ins. `STRATA_SPLIT_OWN=1` gives every card its own prompt buffers on a
layer split; `STRATA_STAGE_TRIM=1` makes each card load only its own layers' dense weights (PR #639).
`STRATA_PF_FUSED=1`, `STRATA_PF_GEMM=1` and `STRATA_PF_SWITCH_MIN_T=4096` put the prompt experts and FP16
projections on the matrix cores, as listed in [`docs/STRIX_HALO.md`](../../../docs/STRIX_HALO.md). The container
environment of each arm was read back with `docker inspect` to confirm which switches were set.

## Method

- Scripts in this folder:
  - [`gen.py`](gen.py) builds the prompts with Strata's own tokenizer (the pack's `tokenizer/`), in the runner's
    Python. Text: this repository at `v0.1.41`, built the way `tools/needle_bench.py` builds its haystack. Each
    prompt is `Run id: <uuid>`, a slice of that text, and one question ("Summarize the material above…"). The slice
    is binary-searched so the server-side prompt (content plus the chat template's 12 tokens) hits the target.
    Index of the prompts actually sent (labels, offsets, token counts, run ids): [`prompts-index.jsonl`](prompts-index.jsonl).
  - [`bench.py`](bench.py) sends them one at a time: streaming `/v1/chat/completions`, `temperature 0` (greedy),
    `reasoning_effort: "none"`, `max_tokens 256`. After each request it reads the engine's own timings
    (`last_timings`) and RAM/VRAM from `/v1/status`.
  - [`arms2.py`](arms2.py) runs the two arms. Opt-ins off: the live runner container is stopped and a clone is
    started from its own `docker inspect` (same image, command, devices, mounts, network, 1 GiB shm) with only the
    five variables dropped. Opt-ins on: the live container is started again. Both are fresh engine processes.
- Order per arm: cold first request (32,768 tokens, first request after the process started), warm-up at 4,096,
  warm-up at 32,768 (neither counted), then 3 runs at 4,096, 3 at 32,768, 3 at 128,000, then the needle checks.
  Opt-ins off ran 02:39:37–02:45:26 UTC, opt-ins on 02:49:35–02:52:54 UTC.
- Reuse and cache state: `cache_n = 0` (no reused prompt tokens) in all 24 benchmark requests; every request has
  a unique first line. The expert cache is pre-filled from the shipped profile at startup (below), so "cold"
  means a fresh process, not an empty expert cache.
- Output: every request produced answer text and no reasoning text, and every one hit the 256-token cap
  (`finish_reason: length`). TTFT is client-side, from sending the request to the first streamed answer token,
  so it includes prompt processing. Model loading is excluded (fresh process to loaded: 52 s and 34 s).
- Prompt and decode throughput are the engine's `prompt_per_second` and `predicted_per_second`; they agree with
  the per-request `strata serve: prompt …` lines in the engine log excerpts
  ([off](engine-optins-off.txt), [on](engine-optins-on.txt)).
- Memory: VRAM per card from `rocm-smi` every second for the whole run ([`vram.jsonl`](vram.jsonl)); RAM from
  `/v1/status` after every request.

## Results

Per-run data: [`matrix-optins-off.jsonl`](matrix-optins-off.jsonl), [`matrix-optins-on.jsonl`](matrix-optins-on.jsonl).

| Arm | Prompt tokens | Reused | Generated | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range | MTP drafts accepted |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| opt-ins on | 4,096 | 0 | 256 | 3 | 2,052 (2,051–2,100) | 60.0 (59.8–62.1) | 2.03 (1.99–2.03) | 72–76% |
| opt-ins on | 32,768–32,773 | 0 | 256 | 3 | 3,804 (3,787–3,811) | 52.7 (50.8–53.2) | 8.68 (8.65–8.71) | 65–73% |
| opt-ins on | 128,005 | 0 | 256 | 3 | 4,307 (4,282–4,319) | 50.4 (49.7–50.5) | 29.87 (29.81–30.03) | 67–69% |
| opt-ins off | 4,096 | 0 | 256 | 3 | 756 (743–757) | 59.3 (55.8–60.3) | 5.45 (5.45–5.55) | 67–75% |
| opt-ins off | 32,768–32,773 | 0 | 256 | 3 | 1,722 (1,720–1,724) | 52.0 (51.4–53.7) | 19.17 (19.11–19.21) | 64–69% |
| opt-ins off | 128,005 | 0 | 256 | 3 | 2,433 (2,416–2,454) | 49.6 (48.9–51.0) | 52.98 (52.31–53.12) | 64–68% |

Cold first request (32,768 tokens, first request of a fresh process, single run): opt-ins on 3,430 tok/s prompt,
60.9 decode, TTFT 9.6 s; opt-ins off 963 tok/s prompt, 55.1 decode, TTFT 34.1 s.

- The opt-ins change prompt throughput, not decode: prompt 2.7x at 4K, 2.2x at 32K, 1.8x at 128K; decode within
  run-to-run spread at every size.
- Prompt sizes hit the targets except the 32K run 3 (32,773) and the 128K runs (128,005): the server's
  tokenization of the slice edge differs from the local count by a few tokens.

Memory (GiB):

| Arm | Card 0 | Card 1 | Card 2 | Card 3 | Total | RAM used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| opt-ins on, loaded (before first request) | 15.66 | 14.77 | 14.32 | 17.70 | 62.4 | 8.2 after load |
| opt-ins on, peak during requests | 15.69 | 14.81 | 14.34 | 17.74 | 62.6 | 8.7–9.5 |
| opt-ins off, loaded (before first request) | 17.26 | 16.40 | 15.87 | 19.32 | 68.9 | 10.6 after load |
| opt-ins off, peak during requests | 17.30 | 16.44 | 15.90 | 19.37 | 69.0 | 9.7–10.6 |

Card 3 (layers 36–47) has the least headroom: 19.37 of 19.98 GiB with the opt-ins off. In both arms the expert
cache holds 6,144 slots per card, all 6,144 of each card's profiled pairs (10.82 / 11.33 / 12.21 / 12.49 GiB on
cards 0–3); with the opt-ins off, card 0's automatic size came to 5,859 slots but it pre-filled all 6,144. No paging
or out-of-memory failures.

Needle recall ([off](needles-optins-off.json), [on](needles-optins-on.json); logs alongside):

| Arm | Length | Prompt tokens | Depth 10% | Depth 50% | Depth 90% |
| --- | --- | ---: | --- | --- | --- |
| opt-ins on | 32K | 33,286–33,287 | found, 9.2 s | found, 9.1 s | found, 5.6 s |
| opt-ins on | 128K | 126,638–126,640 | found, 29.8 s | found, 30.2 s | found, 27.0 s |
| opt-ins off | 32K | 33,286–33,287 | found, 20.4 s | found, 20.0 s | found, 14.1 s |
| opt-ins off | 128K | 126,638–126,640 | found, 52.6 s | found, 52.7 s | found, 47.7 s |

12 of 12 found. In each arm the two depth-90% checks reused 16,384 prompt tokens from the previous check at the
same length (engine log); the other checks read the whole prompt. No failed, cancelled, or skipped requests.

## Correctness and limitations

- Correctness: needle recall only. The speed runs produce answer text, but each answer is cut at 256 tokens and
  was not graded. No coding or tool-call check.
- One model, one machine, three runs per cell, one process start per arm; no variance study across reboots.
- Prompts are repository text with a unique first line; real agent traffic reuses long prefixes, which these
  runs deliberately avoid.
- PCIe is asymmetric: one card on a chipset Gen3 x4 link, three on a shared Gen3 switch uplink. Decode speed on
  this four-way layer split was not compared with fewer cards.
- Not tested: other quants, contexts above 128K, multi-user load, the vision encoder.
