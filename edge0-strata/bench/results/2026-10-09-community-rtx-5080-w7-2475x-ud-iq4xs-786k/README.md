# Community benchmark: RTX 5080 16 GB, Xeon w7-2475X, 128 GB DDR5, Unsloth UD-IQ4_XS at a 786,432-token context (YaRN x3)

Measured on 2026-10-09 by enkynakamura. One consumer Blackwell card with 16 GB, all experts resident in RAM
(`--resident-budget-gib 55`), and the context window set to three times the model's native 262,144 tokens with
`--rope-scaling yarn --rope-scale 3` (the opt-in from #84). Reported per `docs/COMMUNITY_BENCHMARKS.md`: three runs
per configuration on the standard 4K / 32K / 128K sweep, medians and ranges, prompt and decode kept apart, the
needle check with the repository's `tools/needle_bench.py` up to 512K tokens, plus two 352K-token measurements
(one synthetic and reproducible, one from a real agent conversation that is not shareable). Engine 0.1.41, greedy
everywhere.

Main limitation: a single machine and a single quantization; the real-conversation row is one private prompt and
cannot be repeated by anyone else, it is reported only because it is the workload this configuration is used for
daily, and because it reads 15-28% slower than synthetic prompts of the same length (see "What the synthetic sweep
does not show").

## Hardware and software

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 16 GB (Palit GamingPro), driver 595.97, WDDM, power limit 360 W; VRAM clock raised to 2125 MHz (about 1,088 GB/s against 960 GB/s stock, submitter's measurement) |
| PCIe | x16; `nvidia-smi` reported gen 2 at idle, the generation under load was not recorded |
| CPU | Intel Xeon w7-2475X, 20 cores / 40 threads, all-core 4.5 GHz (manual overclock), AVX-512 |
| RAM | 128 GB: 4x 32 GB Kingston FURY Renegade Pro DDR5 RDIMM ECC at JEDEC 4800 MT/s, one DIMM per channel (quad-channel) |
| Storage | Samsung 9100 PRO 2 TB NVMe SSD (model files, pack and MTP on it) |
| OS | Windows 11, build 26200.9457 |
| CUDA | nvcc 13.3 (CUDA toolkit 13.3), source build; runtime 13.2 per `nvidia-smi` |
| Strata | tag `v0.1.41` (commit `fb58e0d`), built from source with `build.bat` in this folder: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120`, MSVC 19.44, Ninja |

Background workloads: a desktop chat app, two browsers and Docker Desktop were open and idle; no other GPU
compute. The engine ran as a normal process (not Task Scheduler).

## Model and configuration

Qwen3.8-Flash-Next **Unsloth UD-IQ4_XS** from `unsloth/Qwen3.8-Flash-Next-GGUF` (revision not recorded; files dated
2026-09-13), three GGUF shards, 93.7 GB in all:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

Pack `unsloth-ud-iq4_xs` built by setup (1.50 GB, experts served natively from the GGUF in place, `--mmap-experts`),
MTP draft layer `Strata-data/mtp/rt` (824 MB), the repository's `data/expert-profile.bin`, vision encoder on
(`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 908 MB, run on the CPU with 20 threads; no image was sent during these runs).
No calibration, no experimental speed projection, no custom profile.

```
strata.exe --pack C:\Strata-data\packs\unsloth-ud-iq4_xs
  --native C:\Strata-data\models\UD-IQ4_XS\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
  --expert-profile data\expert-profile.bin --expert-cache auto --prefill auto
  --spec 5 --spec-min-p 0.8 --mtp C:\Strata-data\mtp\rt
  --max-context 786432 --rope-scaling yarn --rope-scale 3
  --kv int8 --kv-resident 32768 --resident-budget-gib 55
  --vision --vram-reserve-mib 1017 --pcie-frac 0.33 --suffix-draft 0
env: STRATA_IQ_MT_MIN=1 STRATA_IQ_GATHER=1 STRATA_IQ_PREFETCH=4096 STRATA_STAGER_THREADS=8 STRATA_PREFILL_CPU_SHARE=0
```

The full server config is `strata-ud-iq4_xs-786k.json` (local paths, no credentials), launched by `run-chat.bat`
through `serve/server.py` on port 8080. Settings worth noting:

- `--expert-cache auto` gave 2,671 slots (6.0 GiB of VRAM); `--prefill auto` picked 8,192-token chunks.
- `--pcie-frac 0.33` and `STRATA_STAGER_THREADS=8` were chosen by earlier measurements on this machine
  (the autotuned 0.55 and the engine's 32 stager threads were slower here); `STRATA_IQ_*` are the IQ4_XS CPU
  pool switches, `STRATA_IQ_MT_MIN=1` makes the output reproducible run to run.
- `STRATA_PREFILL_CPU_SHARE=0` turns off the 0.1.41 default CPU share for short prompt chunks, so the answers are
  bit-for-bit those of 0.1.40.x. Several prompts here end with a chunk under 1,024 tokens, so with the default on
  the prompt numbers would differ slightly; not measured.
- The config's own sampling block is temperature 1.0 / top_p 0.95 / top_k 20 (daily use); every request in this
  report overrides it with `temperature: 0` and `enable_thinking: false`.
- Resident RAM at start, from the engine log: 49.6 GiB of experts page-locked in RAM, the K/V in 9.3 GiB of pinned
  RAM (32,768 of 786,432 cells per layer in VRAM, the rest streamed). RAM and VRAM peaks were not measured.

## Method

Everything was measured through the running server, with the engine's own `timings` (prompt tok/s = fresh prompt
tokens / `prompt_ms`; decode tok/s = `predicted_per_second`; TTFT = wall time minus `predicted_ms`, non-streaming
request, so it is the prompt time plus HTTP overhead). The server's cumulative request counter (`/metrics`,
`guard.py`) was read before and after each block to prove no other client touched the port; every block was clean.
Model loading is never included. Order of the blocks, on one engine start (14:50 local): the real conversation
once (first request after start), the synthetic sweep, the synthetic 352K prompt, the real conversation again with
three warm runs, then the needle check. Between the first and the second real-conversation reads the conversation
had been evicted from the prompt cache (`cache_n 0`), so both are full reads; the difference between them is the
expert cache's state.

- **Synthetic sweep** (`bench_sweep.py`, adapted from the 2026-10-08 TITAN RTX report's `benchmark.py`): a fixed
  paragraph repeated to the target length, with a revision marker that differs per run and per length so no run
  reuses another's prefix (`cache_n` is 0 in all nine), followed by a request for a 600-word essay so the 256-token
  output cap is always reached (`finish_reason: length` in all nine). Three runs per length, no warm-up beyond the
  earlier block. Prompt tokens are the server's count, not the target.
- **Synthetic 352K**: the same script with `--lengths 352130 --runs 1`, so a single run, kept for the comparison
  with the real conversation of the same length.
- **Real conversation** (`thread_bench.py`): a Jan conversation export (109 messages, 1,009,433 characters,
  352,130 tokens through the chat template: Italian, code, tool calls and their results), plus a question asking
  for a 600-word summary, 512-token cap, greedy. The first request reads the whole prompt (reported as prompt tok/s),
  the following runs reuse the 352,123-token prefix (`cache_n`) and measure decode only. The prompt is private and
  is not in this folder; the script is, for the method.
- **Needle**: `tools/needle_bench.py --lengths 32k,128k,262k,512k --depths 10,50,90`, the repository's own tool and
  prompts, greedy, 40-token answers.

## Results

Synthetic sweep, `sweep-4k-32k-128k.json` (3 runs each; "range" is min-max over the runs):

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 4K synthetic | 4,093 | 0 | 256 | 3 | 1,814 (1,663-1,816) | 68.6 (66.1-70.3) | 2.28 (2.27-2.48) |
| 32K synthetic | 32,708 | 0 | 256 | 3 | 3,299 (3,285-3,306) | 70.2 (68.3-71.2) | 9.96 (9.93-9.98) |
| 128K synthetic | 128,835 | 0 | 256 | 3 | 3,398 (3,366-3,402) | 64.1 (63.5-69.2) | 38.0 (37.96-38.36) |

352K rows (single prompt reads; the warm decode row is 3 runs):

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s | Decode tok/s | TTFT seconds |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 352K synthetic, warm expert cache (`synthetic-352k.json`) | 354,285 | 0 | 256 | 1 | 3,111 | 60.7 | 114.1 |
| 352K real conversation, first request after engine start (`real-thread-352k-fresh-engine.json`) | 352,130 | 0 | 512 | 1 | 2,253 (156.3 s) | 51.3 | 166.6 total |
| 352K real conversation, warm expert cache, full read (`real-thread-352k-warm-expert-cache.json`) | 352,130 | 0 | 512 | 1 | 2,635 (133.6 s) | 51.3 | 143.9 total |
| 352K real conversation, prefix reused, decode only (same file) | 352,130 | 352,123 | 512 | 3 | n/a (7 fresh tokens) | 56.3 (55.3-58.5) | n/a |

Draft acceptance (MTP, `--spec 5`, min-p 0.8): 0.60-0.68 on the real conversation, 0.70-0.82 on the synthetic
prompts; the per-run `draft_n` / `draft_n_accepted` are in the JSON files.

Needle (`needles.json`): **12 of 12 found** at 32K / 128K / 262K / 512K, depths 10 / 50 / 90 %. Actual prompt
lengths 33,275 / 126,551 / 257,725 / 504,901 tokens; the 512K prompts took 243-257 s each. 262K is the model's
native window; the 512K rows run through the YaRN x3 scaling.

Memory: not measured beyond the startup snapshot above (49.6 GiB of experts and 9.3 GiB of K/V pinned in RAM; the
engine's VRAM use is the expert cache's 6.0 GiB plus dense weights, KV window and MTP, `nvidia-smi` showed 15.2 of
16.3 GB in use with the server up and the desktop apps open). No paging, no out-of-memory, no failed request.

## What the synthetic sweep does not show

The repeated-paragraph prompts that most reports use route each chunk to few distinct experts, so they understate
the prompt cost of real text. Same engine, same session, same length:

| 352K prompt | Prompt tok/s | Decode expert cache hit | KV blocks served from VRAM |
| --- | ---: | ---: | ---: |
| synthetic (warm cache) | 3,111 | 68.4 % | 91.6 % |
| real conversation (warm cache) | 2,635 | 56.7 % | 84.3 % |
| real conversation (first request after start) | 2,253 | 56.7 % | 85.3 % |

(The hit rates are the engine's per-request log lines, in `engine.log`.) From 128K to 352K the synthetic prompt
loses 8 % (3,398 to 3,111); the real conversation at 352K reads another 15 % slower than the synthetic one at the
same length with the same cache state, and 28 % slower when it is the first request after the engine starts. One
run per cell, so these are indications of size, not measured distributions. The decode gap goes the same way:
56.3 tok/s median on the real conversation at 352K against 60.7 on the synthetic one, with the expert cache hitting
57-61 % instead of 68 %.

## Correctness and limitations

- The needle check is the only correctness check; 12/12, exact code words, up to 504,901 tokens.
- Greedy decoding was used for every number here. Daily use of this configuration is at temperature 1.0 (the
  model card's recommendation); decode speed depends on the draft acceptance, which differs under sampling.
- Prompt tok/s on the synthetic prompts is optimistic for real text, see above. The real-conversation rows are one
  private prompt: not reproducible by others, included for scale.
- Run-to-run output is reproducible within a session on this configuration (`STRATA_IQ_MT_MIN=1`), but the same
  prompt generated different tokens from a freshly started engine on another day: the engine documents that
  experts computed on the GPU cache round differently from the CPU, and which experts are in the cache depends on
  history.
- 0.1.41 was also compared against 0.1.40.1 on this machine on the same 352K prompt from the command-line
  generator (greedy, 3 runs each, interleaved on the same day): no measurable difference in prompt or decode
  throughput. Not included here, since the CLI generation loop is a different implementation from the server's.
- Single machine, single quantization, one GPU; multi-GPU, IQ3_S and the 0.1.41 CPU prefill share were not tested.
