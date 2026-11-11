# Community benchmark on Tesla V100-PCIE-32GB with 16 GB of system RAM

Measured on 2026-10-04 by christopherrobertbrooks-tech. The Coder IQ1_M on one Volta card (the experimental CUDA 12
engine) in a PC with only 16 GB of RAM, below the 32 GB the Coder normally asks for. Main limitation: one machine, one
model size; the V100 sits on a PCIe 3.0 x4 link.

## Hardware and software

- GPU: Tesla V100-PCIE-32GB (sm_70), PCIe 3.0 **x4** (the 16-lane slot holds an RTX 4070 12 GB that drives the display
  and was idle during the runs); power limit 250 W (default). CPU: Intel Core i7-13700KF (24 threads, AVX2, no
  integrated GPU). RAM: **16 GB DDR5** (15 GiB visible). Model files on a SATA SSD (Samsung 870 EVO 500 GB).
- OS: Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic; NVIDIA driver 580.173.02; CUDA 12.9 toolkit.
- Strata main at 6f32ec0, engine 0.1.39. Engine built by setup.sh from source (CUDA 12, `-DSTRATA_EXPERIMENTAL_SM60=ON`,
  sm_70; the build took ~45 s). One local change in `serve/frontend.py` (images inside Anthropic tool_result blocks, PR #819);
  it does not touch the engine or these OpenAI-API requests.
- Transparent huge pages: `enabled = madvise`, `defrag = madvise`.
- Background: the server ran behind llama-swap (requests went through its proxy on the same machine); no other GPU work.

## Model and configuration

- Qwen3.8-Flash-Next GSQ-RCO **Coder IQ1_M** (setup's `--family coder`):
  `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`, `-00002-of-00002.gguf`. Vision encoder on
  (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, on the GPU). Expert profile `data/expert-profile-coder.bin` (shipped).
- Context 131,072; KV int8 in VRAM (setup turned KV streaming off: "it needs ~1.8 GB of RAM beside the ~32 GB IQ1_M uses,
  and this PC has 15"); expert cache auto; prefill auto; `--resident-experts` (setup's low-RAM choice, `--low-ram on`).
- At start: "resident RAM mode: 4.61 GiB of experts in RAM (page-locked), **11650 in the GPU cache**"; "672 MiB of VRAM
  free with everything loaded". Without the vision encoder all **12,288** experts were in the GPU cache (3.40 GiB in RAM).
- MTP `--spec 4 --spec-min-p 0.5`; reasoning `none` for these runs; temperature 0; no calibration, no speed projection.
  The config also sets `"reasoning_budget_tokens": 8192` (irrelevant here with reasoning off).

```text
./setup.sh --yes --family coder --gpu 1 --cuda 12 --low-ram on --vision gpu --data-dir /mnt/steam/Strata-data --no-start
engine args: --pack .../packs/coder-iq1_m --native ...-00001-of-00002.gguf --ple-gguf ...-00002-of-00002.gguf
  --expert-profile data/expert-profile-coder.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp .../mtp/rt --max-context 131072 --kv int8 --resident-experts --vision --vram-reserve-mib 700
```

## Method

The `benchmark.py` from `bench/results/2026-10-03-community-2x-mi50/` (same model), unchanged: serial requests with fresh
prompts (0 reused tokens), one warm-up, three measured runs at 4,096, 32,768 and 128,000 prompt tokens, output capped at
256 tokens, `reasoning_effort: none`, temperature 0. Prompt and decode rates are the engine's own timings; TTFT and total
time are measured at the client, through llama-swap, with the model already loaded. Per-run JSON is in this folder.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Coder IQ1_M, V100 only | 4,096 | 0 | 256 | 3 | 1,233 (1,232–1,234) | 69.3 (67.2–77.1) | 3.35 (3.35–3.36) |
| Coder IQ1_M, V100 only | 32,768 | 0 | 256 | 3 | 1,580 (1,580–1,581) | 68.6 (64.6–69.0) | 20.80 (20.80–20.81) |
| Coder IQ1_M, V100 only | 128,000 | 0 | 256 | 3 | 1,394 (1,393–1,395) | 67.1 (65.5–71.6) | 92.01 (91.93–92.06) |

- Total request time (client): 7.0 s, 24.5 s and 95.8 s (medians).
- GPU expert-cache hit rate 0.997–0.999; **0.0 MB read from the model files** in every run; PCIe share 0.0 (engine log).
- MTP draft acceptance 0.63–0.78.
- Memory: V100 31.5–32.2 GB used while serving; system RAM ~8–9 GB used of 15 GiB. No paging, no out-of-memory failures.
- Start-up through llama-swap (load to first answer): ~80 s, model files on the SATA SSD.
- PCIe, measured separately with `nvidia-smi dmon -s t` during agent work: 120–150 MB/s into the card, 3–19 MB/s out --
  about 4% of the x4 link, so the narrow slot did not limit inference with the model resident.

## Correctness and limitations

- Long-context recall, `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`: **6 of 6 found** (32,171-32,172 and
  125,449-125,451 prompt tokens; 11-22 s and 81-90 s per check). Results in `needles.json`.
- HumanEval (164, greedy, thinking off, our own harness, not part of this repo): **158/164**, 7.1 min.
- As the builder of a local coding agent (Claude Agent SDK -> `/v1/messages`, medium effort), 4 real tasks in an existing
  Python/GTK project, 3 rounds: 12/12 finished with all acceptance tests and hidden layout checks passing.
- For reference, the README's Coder rows (other engine versions and settings, so only roughly comparable): RTX 5070
  12 GB with 64 GB RAM, 55 / 43 tok/s decode (short / 128K) and 2,180 tok/s prompt; this V100 holds the whole Coder in
  VRAM and decodes 69 / 67 tok/s, with slower prompt reading (1,580 at 32K).
- Not tested: other sizes, a layer split with the second card, thinking on for the speed runs, more than three runs.
- Observed once in agent use (not in this benchmark): a single thinking block ran to the 32K output cap. A
  `reasoning_budget_tokens` cap bounds that: on a prompt built to think at length, no cap gave no answer
  (`finish_reason: length`), a 400-token cap gave the server's wrap-up line and a full answer. In later agent runs with
  an 8K cap the cap never fired (longest thought ~4K tokens). See #710/#728.

The report was drafted with an AI assistant (Claude) from our measurements, and checked by me.
