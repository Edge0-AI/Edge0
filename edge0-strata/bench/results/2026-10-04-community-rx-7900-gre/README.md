# Community benchmark: RX 7900 GRE (gfx1100, 16 GB), Ryzen 7 5700X3D

Measured on 2026-10-04 by [jase100k](https://github.com/jase100k) on Linux.
Three Flash-Next quantizations were measured on one card with settings that are
identical apart from the quantization: the Coder IQ1_M pack, the original
Flash-Next IQ2_XS and Flash-Next IQ3_XXS. Each arm ran the full speed benchmark
three times and one 32k needle recall pass.

Median decode on fresh prompts was **65.3 tok/s for IQ2_XS, 59.1 tok/s for
IQ3_XXS and 54.7 tok/s for Coder IQ1_M** at 8,830 prompt tokens; prompt
throughput was 1,140-1,216 tok/s for all three. The Coder arm is a different
fine-tune with fewer expert-cache slots, so its row is not a quantization-only
comparison against the other two. This is one machine, one card and one small
synthetic workload; nothing here establishes answer quality or performance on
other hardware.

## Hardware and software

- GPU: AMD Radeon RX 7900 GRE, gfx1100, 16 GB. The amdgpu sysfs counters report
  17,163,091,968 bytes (15.98 GiB) of VRAM. The desktop display is attached to
  this card, so `--vram-reserve-mib 3072` keeps its share free.
- CPU: AMD Ryzen 7 5700X3D, 8 cores / 16 threads. The engine reported "no
  AVX-512: the expert kernels run on AVX-2" and chose 7 expert-pool workers plus
  its host thread.
- Installed RAM: 64 GB, `MemTotal` 65,760,384 kB (62.7 GiB). Storage: NVMe
  Kingston SKC3000D2048G, 1.9 TB. Swap use was not recorded.
- PCIe link speed and width were not read from the card. The engine's own
  startup transfer probe measured 28.0-28.1 GB/s host-to-device on all three
  arms, and `pcie_frac` was left at the probed default of 0.55.
- OS: NixOS with a CachyOS kernel, 7.2.4-cachyos, amdgpu/KFD. ROCm 7.2.3 from
  the TheRock `gfx110X-dgpu` wheels in `~/.local/opt/rocm` (no system ROCm);
  hipBLASLt 1.2.2 (100202). The repository tuning table
  `tools/hip/gfx1100-hipblaslt-100202.txt` was active for every arm; the engine
  logged "hipBLASLt tuning enabled (26 rows, gfx1100, version 100202)".
- Strata commit `99f3dbd`, engine 0.1.38, local source build in `build-hip`:
  Release, `STRATA_PREFILL_MMQ=ON`. No calibration and no experimental speed
  projection were used.
- GPU power limits and clocks were neither set nor measured. Background work on
  the same machine during the measurements: this agent session (opencode, not
  served by Strata) and the desktop compositor. No other inference server ran.

## Model and configuration

The three configs are in `data/cfg-*.json`. They differ only in the pack, the two
GGUF files, the expert profile and the model name. Everything else is identical:

```text
--expert-cache auto --prefill auto --spec 5 --spec-min-p 0.5
--mtp /home/jason/Projects/Strata-data/mtp/rt
--max-context 65536 --kv int8 --kv-resident 32768 --vram-reserve-mib 3072
```

Server settings: `backend: hip`, `draft_vocab: en`, `fit_max_tokens: true`,
loopback port 8080, environment `STRATA_HIPBLASLT_TUNING` (table above) and
`STRATA_ADAPT_NOWAIT=1`. Requests used temperature 0, top-k 1, top-p 1, min-p 0,
seed 42, `reasoning_effort: none` and a 128-token output cap. No vision encoder
was loaded and low-RAM mode was off.

The KV cache is INT8 with 32,768 cells per layer resident on the GPU; cells above
that stream from host memory. The engine's closing line for each run reported
98.9-99.4% of KV block reads hitting VRAM, so streaming was rarely used at these
prompt lengths.

GGUF filenames, as used by each config:

```text
Coder IQ1_M      Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
Flash-Next IQ2_XS  Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
Flash-Next IQ3_XXS  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
                 Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
```

The download revision of these files was not recorded, so no claim is made about
which upstream revision they came from. `data/artifact-hashes.json` records the
byte size and SHA-256 of all six GGUFs, of the two expert profiles and the
English draft vocabulary, of the four MTP draft files, and of the small metadata
and tokenizer files inside each pack. Pack tensors of 64 MiB or more are listed
by name and size in that file but are not hashed.

Notes on the choices:

- The context limit is 65,536 for all three arms. The Coder config normally runs
  at 262,144; it was lowered here so that the only difference between the arms is
  the quantization. Throughput at 262,144 was not measured.
- `expert_profile_save` was left off, so every start used the shipped expert
  profile and no run inherited a profile learned during an earlier run. The
  Coder pack uses `data/expert-profile-coder.bin` (12,288 ranked pairs); the two
  Flash-Next arms use `data/expert-profile.bin` (24,576 ranked pairs).
- The MTP draft vocabulary is the English one and the draft weights come from
  `Strata-data/mtp/rt` for all three arms.
- The engine asked for 2 MiB hugepages for the host expert arena and could not get
  them (`vm.nr_hugepages=0`), so the arena used 4 KB pages. That was true for
  every arm and is a property of this machine's configuration.

What the engine reported at startup:

| Quant | GGUF total size | Native weights | Expert cache slots | VRAM for experts | Host arena (2 MiB pages asked) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coder IQ1_M | 54.40 GiB | 2,018.88 MiB | 3,783 | 7.21 GiB | 11,992 (23.4 GiB) |
| Flash-Next IQ2_XS | 63.35 GiB | 1,474.85 MiB | 5,959 | 7.99 GiB | 16,907 (33.0 GiB) |
| Flash-Next IQ3_XXS | 70.63 GiB | 1,781.11 MiB | 4,608-4,621 | 7.48-7.50 GiB | 20,464 (40.0 GiB) |

## Method

`tools/hip/bench_prefill.py` from the repository, unchanged. It sends one warm-up
request ("Reply READY."), then four fresh prompts of 4,210 / 8,830 / 4,210 /
8,830 tokens, each built from synthetic JavaScript functions
(`export function ruleN(x) ...`) followed by "Review this source and describe its
behavior precisely in a paragraph." Each fresh prompt is followed by one
follow-up that asks for the most relevant boundary case and the smallest useful
regression test, which reuses the previous prompt prefix. The output cap is 128
tokens.

The script requires exactly one completed engine timing line per request and
refuses a fresh prompt that reused any cached token. The engine was restarted
before each of the three runs per arm, so every one of the 36 fresh prompts in
this report processed its whole prompt: zero reused tokens. Model loading is not
included in any timing; `wall_s` is the client-side duration of the request.
Every measured request generated the full 128 tokens and stopped at the cap.
**Time to first token was not measured**: `bench_prefill.py` reads the
non-streaming completion endpoint and records one wall-clock duration per
request, so no TTFT column is reported here.

The needle pass ran on the run-3 server, so its prefix cache was already warm;
its latencies are therefore not comparable with the fresh-prompt table.

Host memory and VRAM were sampled once per second by
`scripts/sample_mem.py` during each run, reading the amdgpu sysfs counters and
`/proc/meminfo`. These are whole-machine numbers, not the engine's accounting.
Brief spikes between samples can be missed.

Reproduce one arm with `scripts/run-one-quant.sh coder-iq1_m` (also `iq2_xs`,
`iq3_xxs`), which writes into `scripts/out/`, then aggregate with
`python scripts/summarize.py --write`. `ROOT` inside that script and the paths
inside `data/cfg-*.json` are this machine's, so change them before reuse.

## Results

Fresh prompts, no prefix reuse. Each cell is the median **[minimum-maximum]** of
six requests (three runs, two prompts of that size per run). Prefill is freshly
read prompt tokens per second from the engine timing line, decode is generated
tokens per second from the same line.

| Quant | Prompt tokens | Reused | Prompt tok/s | Decode tok/s | Total seconds |
| --- | ---: | ---: | --- | --- | --- |
| Coder IQ1_M | 4,210 | 0 | 1,224.1 [1,210.4-1,273.1] | 48.5 [46.3-52.4] | 6.1 [5.8-6.3] |
| Coder IQ1_M | 8,830 | 0 | 1,213.2 [1,181.5-1,229.1] | 54.7 [51.9-55.2] | 9.6 [9.6-9.8] |
| Flash-Next IQ2_XS | 4,210 | 0 | 1,145.5 [1,119.0-1,171.9] | 64.5 [55.0-71.2] | 5.7 [5.4-6.1] |
| Flash-Next IQ2_XS | 8,830 | 0 | 1,154.7 [1,152.6-1,159.4] | 65.3 [63.6-68.2] | 9.6 [9.5-9.7] |
| Flash-Next IQ3_XXS | 4,210 | 0 | 1,147.5 [1,118.9-1,174.7] | 54.2 [46.9-60.8] | 6.1 [5.7-6.5] |
| Flash-Next IQ3_XXS | 8,830 | 0 | 1,140.1 [1,138.4-1,143.9] | 59.1 [56.8-61.9] | 9.9 [9.8-10.0] |

Follow-up requests, which reuse the previous prompt prefix. The number of reused
tokens is not identical in every run, because the length of the assistant's reply
decides it; these rows are a separate condition from the table above.

| Quant | Prompt tokens | Reused tokens | Freshly read | Decode tok/s | Total seconds |
| --- | ---: | ---: | ---: | --- | --- |
| Coder IQ1_M | 4,451 | 4,337 | 114 | 47.8 [45.3-50.5] | 3.4 [3.3-3.5] |
| Coder IQ1_M | 9,071 | 8,957 | 114 | 47.8 [43.4-49.9] | 3.5 [3.3-3.7] |
| Flash-Next IQ2_XS | 4,451 | 4,203-4,338 | 113-248 | 66.3 [56.1-72.8] | 2.7 [2.5-2.9] |
| Flash-Next IQ2_XS | 9,071 | 8,957-8,958 | 113-114 | 73.0 [62.2-75.4] | 2.4 [2.3-2.7] |
| Flash-Next IQ3_XXS | 4,451 | 4,203-4,338 | 113-248 | 51.9 [47.9-53.3] | 3.5 [3.4-3.6] |
| Flash-Next IQ3_XXS | 9,071 | 8,823 | 248 | 53.0 [51.3-59.3] | 3.5 [3.2-3.5] |

Follow-up *prompt* throughput is not comparable across arms, because the amount of
reused text differs per run; it is left out of the conclusions.

Speculative decoding and cache behaviour, summed over each arm's whole session
(speed runs and needle pass), from `data/<quant>.engine.log`:

| Quant | Drafts accepted | Acceptance | Decode expert-cache hit rate |
| --- | ---: | ---: | ---: |
| Coder IQ1_M | 1,862 / 2,729 | 68.2% | 84.0% |
| Flash-Next IQ2_XS | 2,059 / 2,918 | 70.6% | 88.9% |
| Flash-Next IQ3_XXS | 2,057 / 3,027 | 68.0% | 83.4% |

Memory, sampled once per second during each run and the needle pass. VRAM is the
peak of the amdgpu counters; host RAM used is `MemTotal - MemAvailable`, so it
includes the desktop, this agent session and everything else on the machine.

| Quant | Peak VRAM | Peak host RAM used | Samples per run |
| --- | ---: | ---: | ---: |
| Coder IQ1_M | 14.58-14.61 GiB | 31.9-32.4 GiB | 42-46 s of 1 s samples |
| Flash-Next IQ2_XS | 14.57-14.58 GiB | 41.1 GiB | 42 s |
| Flash-Next IQ3_XXS | 14.60-14.62 GiB | 48.0-48.2 GiB | 47 s |

No out-of-memory error and no failed request occurred in any arm.

## Recall

`tools/needle_bench.py --lengths 32k --depths 10,50,90` found the needle in all
three depths for all three arms:

| Quant | Found | Prompt tokens | Seconds per case |
| --- | ---: | ---: | --- |
| Coder IQ1_M | 3 of 3 | 31,969-31,970 | 24.2 [12.5-25.2] |
| Flash-Next IQ2_XS | 3 of 3 | 31,969-31,970 | 25.9 [13.3-26.3] |
| Flash-Next IQ3_XXS | 3 of 3 | 31,969-31,970 | 25.3 [13.0-25.8] |

This measures recall on those three inputs only, not overall model quality.

## What these numbers do and do not show

- Between the two arms that differ only in quantization (the same Flash-Next
  model), IQ2_XS decoded faster than IQ3_XXS: 65.3 against 59.1 tok/s median on
  8,830-token fresh prompts, while prompt throughput was nearly equal
  (1,154.7 against 1,140.1 tok/s). The smaller files did not cost prefill here
  and did not win decode at this size.
- The Coder IQ1_M arm decoded slowest of the three, but it is a different
  fine-tune: it was given fewer expert-cache slots (3,783 against 5,959), filled
  them from a profile with half the ranked pairs, and reached a lower decode
  cache hit rate (84.0% against 88.9%). Its row cannot be read as "the smallest
  quantization is slowest"; the expert cache sizing and the profile differ along
  with the model.
- Peak VRAM is nearly the same for all three arms (14.6 GiB) because
  `--expert-cache auto` fills whatever the reserve leaves free. The cost of the
  larger quantizations showed up in host RAM instead: 32 GiB for Coder IQ1_M,
  41 GiB for IQ2_XS and 48 GiB for IQ3_XXS, with a 40.0 GiB host arena for
  IQ3_XXS against 23.4 GiB for the Coder pack.

## Limitations

- One machine, one card, one workload. No other GPU, engine or build was
  measured, so these numbers are not a controlled comparison with the figures in
  the repository README.
- The Coder IQ1_M arm is a different model from the other two. Only the IQ2_XS
  against IQ3_XXS comparison changes just the quantization.
- 65,536-token context limit, 128-token outputs, greedy decoding, reasoning off.
  Longer contexts, longer outputs, sampled decoding, thinking, vision, tool use,
  concurrency and a sustained thermal run were not evaluated.
- GPU clocks and power limits were not fixed or recorded; the desktop shares the
  measured card.
- 4 KB pages backed the host expert arena on this machine; a machine with 2 MiB
  hugepages configured may see different host-to-device behaviour.
- Follow-up rows reuse between 4,203 and 8,958 tokens depending on the run, so
  their prompt throughput is not a controlled measurement.
- Time to first token was not measured (see Method).
- The needle pass ran on a warm prefix cache and covers three inputs per arm.
- No answer-quality claim beyond that needle check.

## Files

- `data/cfg-coder-iq1_m.json`, `data/cfg-iq2_xs.json`, `data/cfg-iq3_xxs.json` -
  the exact server configs, one per arm.
- `data/<quant>-run1.json` ... `run3.json` - per-request records from
  `tools/hip/bench_prefill.py`, including the generated text, the engine timing
  values and the prefix reuse counts.
- `data/<quant>-needles.json` - needle results.
- `data/<quant>.engine.log` - the engine's own log for that arm, including the
  startup lines quoted above, the PCIe probe, cache sizing and the per-window
  draft and cache statistics.
- `data/memory/<quant>-mem-*.csv` - the 1 Hz memory samples behind the table.
- `data/artifact-hashes.json` - sizes and SHA-256 sums of the model, profile,
  draft and pack files used here.
- `data/summary.json` - every aggregate in this report, produced by
  `scripts/summarize.py --write`.
- `scripts/run-one-quant.sh`, `scripts/sample_mem.py`, `scripts/summarize.py` -
  the runner and the two analysis scripts.
