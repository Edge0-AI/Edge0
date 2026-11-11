# Community benchmark on AMD Radeon RX 6800M (gfx1031)

Measured on 2026-10-06 by [Hugua700](https://github.com/Hugua700). This is the low end of what runs
Strata: one mobile RDNA2 card (12 GB) with 32 GB of system RAM, on Windows, using a **self-built HIP
engine** (the ready-made Windows HIP zip has no gfx1031 code). Three prompt sizes at two engine
settings, three runs each, plus six recall checks.

The numbers are far below the published ones and that is the point of this report: on this machine
**prefill is 50–114 tok/s and decode is 9–27 tok/s**, against 2,650 / 76 tok/s for Q2_0 on the cards in
[docs/MODELS.md](../..//docs/MODELS.md). Two reasons, both visible in the engine's own startup lines:
the experts do not fit in RAM and are streamed from the GGUF, and the CPU has no AVX-512 so the
expert kernels run AVX-2.

**Main limitation:** it is not a 4K/32K/128K sweep. The server was configured for a 65,536-token
context but the longest prompt measured here is 8,727 tokens. The machine also has a known
power-delivery fault (four spontaneous power-offs on 2026-10-03/05/06 under sustained load), so the
run was deliberately split into short segments rather than one long sweep.

## Hardware and software

- **GPU:** AMD Radeon RX 6800M (Navi 22, `1002:73df`), 12,272 MiB usable VRAM (the engine reports
  `Windows budgets 11474 of this card's 12272 MiB for this process`), PCIe probe **7.0 GB/s**
  host→device (the engine's own probe; 384 GB/s memory bandwidth is the spec figure, not measured here)
- **CPU:** AMD Ryzen 9 5900HX, 8 cores / 16 threads, 3.3 GHz base; **no AVX-512** — the engine logs
  `this CPU has no AVX-512: the expert kernels run on AVX-2`
- **RAM:** 31.4 GB (2 × 16 GB Micron DDR4-3200)
- **Storage:** Samsung SSD 980 1 TB NVMe
- **OS:** Windows 11 Home, build 26300
- **Driver:** AMD 32.0.21045.5002 (2026-08-17)
- **ROCm:** 10.2.0a20260930 (TheRock wheels)
- **Strata:** `1735d64` (v0.1.40), engine `0.1.40`, **source build** — `STRATA_HIP_ARCHS=gfx1031`,
  `tools/hip/build_windows.bat`, gfx1031 only; CPU image encoder also built (`vision_src b8ba8d7bfed20e5b`)
- **Power:** on AC, ASUS "Performance" power plan; nothing else was using the GPU during the runs.
  The integrated Radeon iGPU was idle and is not used.

## Model and configuration

- **Model:** [`SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF`](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
  quantization **Q2_0**, shard 1 `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf`
  (38,021,379,872 bytes, sha256 `1620faf2…4647e`, verified against the Hugging Face LFS id), shard 2
  `…-00002-of-00002.gguf` (28,800,138,432 bytes, sha256 `316b46f3…1e113`). Shard 2 is shared with the
  original Coder IQ1_M download and is a hard link to it on this machine.
  This is a community abliterated fine-tune of the same base; it was chosen because it is what this
  machine's users wanted to run, and its quant ladder is the subject of one of the notes below.
- **Vision encoder:** `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes), `--vision`, CPU
  (12,272 MiB card, so the encoder runs on the CPU). Not exercised by these runs; `/health` reports
  `"images": true` and a separate three-colour test image was read correctly.
- **Pack/profile:** `tools/iq_pack.py --gguf <shard 1> --out <pack>` (no `--experts-bin`; 1.4 GiB
  pack, 303 tensors served natively); `data/expert-profile.bin` (the 48×512 profile — the Coder
  variant needs the 48×256 one); MTP draft layer from `tools/mtp_*` at `…\mtp\rt`.
- **Context:** 65,536; **KV int8** with `--kv-resident 32768` (KV streaming: 32,768 of 65,536 cells
  per QSA layer in VRAM, K/V in 0.77 GiB of pinned RAM).
- **Cache / low-RAM:** `--expert-cache auto --prefill auto --resident-experts`; the engine settles on
  **4,026 GPU cache slots (5.18 GiB)** and **16.77 GiB of experts in RAM (pageable)**, with the rest
  read from the GGUF through the OS file cache.
- **MTP:** on (`--spec 4 --spec-min-p 0.5`); draft head 835 MiB of VRAM. This is a suffix/MTP draft
  path and it accepts a good share of drafts, so decode speed includes it — it is not a
  draft-free measurement.
- **Sampling:** greedy (`temperature 0`), `chat_template_kwargs: {"enable_thinking": false}`.
  Calibration and experimental speed projection were **not** used.

```text
strata.exe --serve --pack F:\Strata-data\packs\sc117-q2_0 ^
  --native F:\Strata-data\models\sc117-q2_0\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf ^
  --ple-gguf F:\Strata-data\models\sc117-q2_0\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00002-of-00002.gguf ^
  --expert-profile F:\Strata\data\expert-profile.bin --expert-cache auto --prefill auto ^
  --spec 4 --spec-min-p 0.5 --mtp F:\Strata-data\mtp\rt ^
  --max-context 65536 --kv int8 --kv-resident 32768 --resident-experts --vision
```

`data/server-config.json` is the exact server config, `data/engine-startup-default.txt` and
`data/engine-startup-f16.txt` are the engine's own startup lines for the two settings.

## Method

Requests went through `/v1/chat/completions`, **one at a time**. Speed numbers are the server's own
`timings` object (`prompt_per_second`, `predicted_per_second`, `prompt_ms` as TTFT), the same source
the existing reports use; `strata-bench.py` records the whole object per run. Model loading is not
included in any timing.

- **Warm-up:** one short request is sent and discarded before each configuration. The engine's
  first request after a load is cold — the expert cache starts empty and fills from the profile
  (`pre-filled 4026 of 4026 slots from the profile in 17.0 s`) — and it is several times slower.
- **Prompts** are built from this repository's own source files (`tools/iq_pack.py`,
  `tools/make_profile.py`, `tools/bench_prefill.py`, `setup.py`, `serve/server.py`), so anyone can
  rebuild them. The ~512-token prompt is a short instruction; the 2K and 8K prompts are a block of
  that source followed by "summarize what the code above does".
- **Prefix reuse is defeated on purpose.** Strata reuses a cached conversation prefix, so a repeated
  prompt makes runs 2..N read only a handful of new tokens and their prompt speed is not comparable
  to run 1. Every measured run therefore carries a unique leading `# benchmark run <nonce>` line;
  the "Reused" column below is 0 for every row as a result. `data/*.json` keeps the raw
  `timings.cache_n` for each run.
- **Output cap:** 256 tokens, all runs reached it (`finish_reason: length`).
- **Recall checks** put `# RECORD-KEY: XYLOPHONE-7742` at 25 %, 50 % and 75 % of the 2K and 8K
  prompts and ask for the value; the answer is checked for the needle.

## Results

Server setting: `default` = the launch command above; `f16` = the same with
**`STRATA_HIP_PROMPT_F16=1`** in the environment (the engine suggests it on gfx1031:
*"reads long prompts about 2x faster here"*).

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| default, short | 556 | 0 | 256 | 3 | 50.4 (37.5–57.8) | 19.3 (16.5–22.8) | 11.0 (9.6–14.8) |
| default, ~2K | 2,309 | 0 | 256 | 3 | 59.6 (53.3–61.7) | 9.1 (8.6–13.7) | 38.7 (37.4–43.3) |
| default, ~8K | 8,726 | 0 | 256 | 3 | 92.2 (89.7–95.6) | 14.7 (14.1–14.7) | 94.6 (91.3–97.2) |
| f16, short | 558 | 0 | 256 | 3 | 64.9 (38.9–66.0) | 26.5 (24.1–26.7) | 8.6 (8.5–14.3) |
| f16, ~2K | 2,311 | 0 | 256 | 3 | 63.0 (52.5–64.3) | 14.5 (13.2–14.9) | 36.7 (35.9–44.1) |
| f16, ~8K | 8,727 | 0 | 256 | 3 | 113.5 (105.9–120.2) | 14.1 (13.8–14.3) | 76.9 (72.6–82.4) |

Per-run data: `data/strata-bench-default.json`, `data/strata-bench-default8k.json`,
`data/strata-bench-f16.json` (every run's full `timings` object, wall time and answer head),
`data/strata-bench-recall.json`.

**Prefill improves with prompt length, decode does not.** Prompt speed rises 50 → 60 → 92 tok/s as
the prompt grows: the fixed per-request cost is amortized. Decode is 9–27 tok/s and does not follow a
clean trend — it was 9.1 tok/s at 2K and 14.7 tok/s at 8K in the default runs, which is the opposite
of the usual expectation; the spread is larger than the effect, so treat decode as "about 10–25 tok/s
on this machine" rather than reading the ordering.

### ⭐ `STRATA_HIP_PROMPT_F16=1`: +6 % to +29 % here, against +69 % to +101 % on gfx1030

| Prompt tokens | default prompt tok/s | f16 prompt tok/s | change |
| ---: | ---: | ---: | --- |
| 556 | 50.4 | 64.9 | **+29 %** |
| 2,309 | 59.6 | 63.0 | +6 % |
| 8,726 | 92.2 | 113.5 | **+23 %** |

TTFT falls by the same factor (94.6 → 76.9 s at 8K). This is the same switch as
[`bench/results/2026-10-04-rdna2-fp16-prompt`](../2026-10-04-rdna2-fp16-prompt/README.md) (RX 6900 XT,
gfx1030), and the two results do not agree — that is the interesting part:

| Rig | 9–9.5K prompt, old path → FP16 |
| --- | --- |
| 2x RX 6900 XT 16 GB, **128 GB RAM**, IQ3_S fits in RAM | 439 → 744 tok/s (**+69 %**) |
| RX 6800M 12 GB, **32 GB RAM**, Q2_0 does **not** fit (16.77 GiB resident, the rest streamed) | 92 → 114 tok/s (**+23 %**) |

The likely reason: on this machine prefill is not bound by the prompt GEMM. The experts do not fit in
RAM, so a long prompt also drives a lot of expert streaming from the GGUF and AVX-2 CPU kernels, and
that floor sits underneath the GEMM win. The engine's per-request line shows the file tier working
during the 8K runs (`files 91798 blobs 106863.7 MB read`), and prompt speed here is 5–8x below the
gfx1030 rig even before the switch. **Consistent conclusion across both reports: the switch is worth
enabling on gfx103x, but its size depends on what else is bound.**

**Decode deltas here are noise, not an effect.** The table above shows decode moving 19.3 → 26.5 tok/s
at short and 9.1 → 14.5 at 2K, then not at all at 8K (14.7 → 14.1). The gfx1030 report measured decode
as unchanged, with the explanation that decode does not use these GEMMs at all — so the movement here
is run-to-run spread, and this report does not claim a decode gain. The decode column should be read as
"about 10–25 tok/s on this machine".

**A caveat, stated plainly:** the flag is documented as *"not bit-identical to the default"*. The
recall checks below were run in the **default** setting only, so the `f16` rows above have no
correctness check behind them — they are speed measurements of a configuration that is known to
round differently. Anyone turning it on should re-run their own accuracy checks.

## Correctness and limitations

**Recall — 6/6 correct** (default setting, needle `XYLOPHONE-7742`):

| Context | Depth | Prompt tokens | Found |
| ---: | ---: | ---: | --- |
| ~2K | 25 % / 50 % / 75 % | 2,311 / 2,311 / 2,310 | yes, yes, yes |
| ~8K | 25 % / 50 % / 75 % | 8,727 / 8,727 / 8,726 | yes, yes, yes |

All six answers were exactly `XYLOPHONE-7742`. The three ~2K checks were the first requests after an
engine restart and show the cold-cache effect (11.3 / 14.1 / 33.5 prompt tok/s), which is why the
warm-up exists in the speed runs.

**Other checks, not part of the table:** the same build serves `/v1/chat/completions` with `tools`
and returns proper `tool_calls`; the CPU image encoder reads a test image correctly; `/health`
reports `max_context 65536`, `images: true`.

**Limitations**

- **Longest prompt 8,727 tokens** against a configured 65,536. No 32K or 128K measurement. The
  engine's own numbers show prompt speed still improving at 8K, so this report says nothing about
  where it peaks or how it degrades.
- **One model, one quantization.** No second quant or the original (non-abliterated) weights were
  measured on this machine, so nothing here compares fine-tunes.
- **Decode includes the MTP/suffix draft path**, which is on by default in this configuration; it is
  not a draft-free decoding number.
- **RAM is saturated during the runs**: 0.1–0.2 GB free, pagefile in use. The engine says so
  explicitly:
  `WARNING: the whole resident RAM mode does not fit (FileExpertSource: resident complement 26.46 GiB exceeds available RAM (16.66 GiB) minus the 4 GiB safety headroom); 12.41 GiB of the experts the GPU does not hold … are kept in RAM and the rest are read from the model folder through the OS file cache`.
  So a large part of the decode cost here is disk streaming, and these numbers should not be read as
  what a 12 GB card can do when the model fits in RAM.
- **Power plan was "Performance".** ASUS's balanced plan is a documented mitigation for this
  machine's power fault; it was not used, so the measured speeds are the faster of the two.
- **Not measured:** latency percentiles, concurrency (one request at a time), long-run stability.

**Two notes for other low-RAM AMD users**, both found while setting this up:

1. **Quantization availability.** The abliterated repository above has **no IQ1_M**; its ladder
   starts at Q2_0 / IQ2_XS. On a 32 GB machine that moves the model out of the "fits" column —
   `START-HERE.bat --check` correctly reports `Q2_0 needs ~48 GB RAM: does not fit`, and the engine
   falls back to the partial resident set quoted above. It works, it is just slower than the Coder
   IQ1_M it replaced. Pick the quant knowing this.
2. **`--expert-profile` follows the model family, not the base model.** The Coder variant has 256
   experts per layer and the plain one 512, so reusing a profile gives a hard stop:
   `read_expert_profile: …expert-profile-coder.bin is 48x256 but this model is 48x512 - it is a profile for a different artifact`.
   The message is clear; the point is only that `data/expert-profile.bin` (512) and
   `data/expert-profile-coder.bin` (256) are not interchangeable when switching fine-tunes.
