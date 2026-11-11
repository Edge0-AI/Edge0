# Community benchmark: RTX PRO 4500 Blackwell, Threadripper PRO 3975WX

Measured on 2026-10-01 by [mlfather](https://github.com/mlfather), on a single-GPU Linux workstation. This tests
Strata engine 0.1.31 (source build, same binary and commit for every
quantization below) with three packs on the same GPU: the Unsloth UD-Q4_K_XL
pack, the Unsloth UD-IQ4_XS pack, and ISTA-DASLab's GSQ-RCO IQ3_S pack, each at
three separate context limits (8,192 / 81,920 / 139,264 tokens, one server
process per limit per pack). Greedy decoding, synthetic prompts, no correctness
suite run beyond what is noted in Correctness and limitations.

UD-Q4_K_XL median decode throughput was **61.5 tok/s at a 2,700-token prompt,
72.4 tok/s on a 370-token code-generation prompt, 72.1 tok/s at 64K prompt
tokens, and 70.2 tok/s at 128K**. UD-IQ4_XS reached **82.0 / 98.0 / 95.8 / 91.4
tok/s** on the same four prompts; ISTA IQ3_S reached **106.3 / 123.8 / 120.3 /
116.5 tok/s**. All three packs import and run on pristine, unpatched engine
0.1.31 (see Model and configuration: UD-IQ4_XS previously needed a local,
unpublished patch on engine 0.1.18 and no longer does). These are synthetic
code-review/code-generation requests with a 256-1,024-token output cap. They do
not establish general answer quality or performance on other workloads or
hardware.

## Hardware and software

- NVIDIA RTX PRO 4500 Blackwell; 32,623 MiB reported VRAM (405 MiB reserved by
  the driver); power limit 200 W (min 150 W, max 200 W, fixed at default for
  this test). PCIe: slot/host maximum Gen 4 x16, device capable of Gen 5; the
  link was sampled at Gen 1 x16 while the GPU was idle between runs (NVIDIA's
  power-saving link state), current width 16x. The engine's own PCIe probe
  measured **28.2-28.3 GB/s host-to-device** at each server start. GPU clocks
  were not fixed for this test.
- AMD Ryzen Threadripper PRO 3975WX; 32 cores / 64 threads (no AVX-512; AVX2
  only). The engine selected AVX2 and **31 expert-pool workers plus its host
  thread**.
- 512 GB installed RAM (8-channel DDR4); `free -g` reported 503 GiB total, with
  86 GiB free and 407 GiB in buff/cache at measurement time. Storage for the
  model and pack: a Sabrent SB-RKT4P-4TB NVMe SSD mounted at `/mnt/fast`
  (3.6 TB, 23% used). 32 GiB zram swap (not relevant: no paging under test).
- Omarchy (Arch-based) Linux, kernel `7.2.5-3-omarchy`, NVIDIA driver
  `615.71.09`, CUDA UMD version 13.4 (nvidia-smi reports both `Driver Version`
  and `CUDA Version` as deprecated header fields on this driver; CUDA UMD
  Version is the successor field).
- Source commit `9259cad4cfa3543cd3b8decab5962672b968c649` (tag `v0.1.31`);
  engine 0.1.31. Self-built with CMake + Ninja, `CMAKE_CUDA_ARCHITECTURES=120`
  (sm_120, this GPU's native architecture), `CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc`,
  `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`. See [BUILD.json](BUILD.json).
- NVCC release 13.3, V13.3.73 (`cuda_13.3.r13.3/compiler.38244171_0`); GCC
  (g++-15) 15.3.0; Python 3.13.15 in the project's `.venv`.
- Background workloads: a lightweight hardware/job-monitoring pair of daemons
  using about 2% of one CPU core throughout, and the Hyprland desktop compositor
  running on a second, separate GPU (an RTX PRO 4000) that did not share the
  benchmarked card. Nothing else used the benchmarked GPU during the runs.

## Model and configuration

Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, subfolder `UD-Q4_K_XL`, revision
`38bb39e`:

- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf`
- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf`
- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf`

All four local GGUF SHA-256 hashes matched the values published in the
project's own `docs/UNSLOTH_Q4.md` for this revision (recorded in
[BUILD.json](BUILD.json)). "UD-Q4_K_XL" is a mixed-format quantization: routed
experts are Q4_K (Q5_K in layer 2) for gate/up and Q5_1 (Q8_0 in layers 2, 4,
30, 46, 47) for down; embedding, output head, attention, shared-expert
projections and the PLE key are Q8_0; the PLE table is IQ4_NL.

No vision encoder was used. No custom expert profile or draft vocabulary was
built; the pack used the model's bundled expert profile
(`data/expert-profile.bin`, 24,576 ranked pairs for 24,576 total experts: 48
layers x 512 experts/layer). The pack itself was built with
`tools/iq_pack.py --compat-bf16` (required for this mode: 195 small
hyper-connection/PLE-value tensors are rounded Q8_0-to-BF16, listed with exact
error bounds in the pack's own `conversions.json`). `--experts-bin` was not
used: the four GGUF shards are read in place via `--mmap-experts`, with no
separate 77 GB experts file. The MTP draft layer is the base model's own,
built from the published `mtp.*` tensors the same way the project's other
packs build it.

- Context: three separate server runs at 8,192 / 81,920 / 139,264 tokens
  (`--max-context`), each its own process with its own config file.
- KV: `--kv int8`. `--vram-reserve-mib 1536`. `--expert-cache auto`, no
  eviction policy change. `--prefill auto` (selected 8,192-token chunks).
- `--resident-budget-gib 100`: with this budget, all 24,576 experts were
  resident across GPU VRAM + pinned host RAM at every context length tested,
  with no experts left to the SSD-backed tier:

  | Context | GPU expert-cache slots | GPU VRAM for experts | RAM-resident slots | Total (of 24,576) |
  |---|---:|---:|---:|---:|
  | 8,192 | 7,982 | 23.30 GiB | 16,594 | 24,576 |
  | 81,920 | 7,614 | 22.22 GiB | 16,962 | 24,576 |
  | 139,264 | 7,329 | 21.39 GiB | 17,247 | 24,576 |

- MTP: `--spec 4 --spec-min-p 0.5` (draft depth 4, min-p 0.5); MTP is mandatory
  to start this engine's serve mode.
- Low-RAM mode off (not applicable: `--resident-budget-gib` is this engine's
  own RAM-budgeted mode, distinct from the low-RAM mode used on smaller GPUs
  in the project's own `docs/UNSLOTH_Q4.md`). No GPU vision. No experimental
  speed projection, no control vectors or calibration. Reasoning disabled per
  request (`chat_template_kwargs.enable_thinking=false`), temperature 0,
  top_k 1 (greedy).
- One additional comparison toggled `STRATA_KQ256=1` (multi-token AVX2 kernels
  for the Q4_K/Q5_1/Q8_0 expert formats) against the same 8,192-context config,
  on the short and codegen prompts only. Everything else unchanged.

The three complete measured configurations are
[configs/strata-ud-q4_k_xl.json](configs/strata-ud-q4_k_xl.json) (8,192 ctx),
[configs/strata-ud-q4_k_xl-64k.json](configs/strata-ud-q4_k_xl-64k.json)
(81,920 ctx) and
[configs/strata-ud-q4_k_xl-128k.json](configs/strata-ud-q4_k_xl-128k.json)
(139,264 ctx). Each config's `--native` path and `--pack` directory are local
to the machine that produced them; set them to your own checkout and model
location.

```text
.venv/bin/python serve/server.py --engine strata --config strata-ud-q4_k_xl.json --host 127.0.0.1 --port 8080
```

(`strata-ud-q4_k_xl-64k.json` / `-128k.json` were run the same way as separate
server processes, one context limit per run.)

Engine startup excerpts (AVX selection, worker count, PCIe probe, expert-cache
fill) are in [logs/strata-ud-q4_k_xl.log](logs/strata-ud-q4_k_xl.log),
[logs/strata-ud-q4_k_xl-64k.log](logs/strata-ud-q4_k_xl-64k.log) and
[logs/strata-ud-q4_k_xl-128k.log](logs/strata-ud-q4_k_xl-128k.log). The KQ256
comparison's console transcript is
[logs/strata-ud-q4_k_xl-kq256.log](logs/strata-ud-q4_k_xl-kq256.log).

### Unsloth UD-IQ4_XS (same engine build, 0.1.31)

Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, subfolder `UD-IQ4_XS`, three shards
(`Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf`), already present
locally; this submission did not re-verify the local files' hashes against a
published manifest for this quant (unlike UD-Q4_K_XL above).

**This import no longer needs a patch.** A same-machine 2026-09-28 study
(engine 0.1.18) found that Unsloth UD-IQ4_XS "needed the local patch on branch
`p620-q8_0-down` ... to run at all", because the engine's native PLE-key path
took Q2_0 only and this quant's PLE key is Q8_0. On this 0.1.31 checkout
(commit `9259cad`), `tools/iq_pack.py --gguf <shard 1> --out packs/ud-iq4_xs
--compat-bf16` built the pack directly (one layer's gate/up/down split across
shards, handled by `native_experts.txt` v4's per-role shard column; 460
tensors converted, 195 of them the same "round Q8_0 small projections to
BF16" `--compat-bf16` path UD-Q4_K_XL uses), and the server started and served
all four prompts with no code changes, using the model's own
`data/expert-profile.bin` and the same MTP draft layer (`mtp/rt`) the other
packs use. `docs/UNSLOTH_Q4.md`, current as of this commit, still says "Only
UD-Q4_K_XL ... is targeted" and warns other Unsloth quantizations "may" lack
kernel support; empirically, for UD-IQ4_XS on this engine version and GPU,
that caveat no longer applies. Quality was not compared against the patched
0.1.18 build or against llama.cpp in this submission.

Settings matched the UD-Q4_K_XL runs exactly except `--pack packs/ud-iq4_xs`
and `--native` pointing at the UD-IQ4_XS shard 1:
`--resident-budget-gib 100`, `--kv int8`, `--vram-reserve-mib 1536`,
`--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, one
server per context limit (8,192 / 81,920 / 139,264). All 24,576 experts were
resident (GPU cache + RAM budget, no file reads) at every context length:

| Context | GPU expert-cache slots | GPU VRAM for experts | RAM-resident slots | Total |
|---|---:|---:|---:|---:|
| 8,192 | 10,434 | 23.52 GiB | 14,142 | 24,576 |
| 81,920 | 9,957 | 22.43 GiB | 14,619 | 24,576 |
| 139,264 | 9,585 | 21.60 GiB | 14,991 | 24,576 |

Configs: [configs/strata-ud-iq4_xs.json](configs/strata-ud-iq4_xs.json),
[configs/strata-ud-iq4_xs-64k.json](configs/strata-ud-iq4_xs-64k.json),
[configs/strata-ud-iq4_xs-128k.json](configs/strata-ud-iq4_xs-128k.json).
Logs: [logs/strata-ud-iq4_xs.log](logs/strata-ud-iq4_xs.log),
[logs/strata-ud-iq4_xs-64k.log](logs/strata-ud-iq4_xs-64k.log),
[logs/strata-ud-iq4_xs-128k.log](logs/strata-ud-iq4_xs-128k.log).

### ISTA-DASLab GSQ-RCO IQ3_S (same engine build, 0.1.31)

Model: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, subfolder `IQ3_S`, two
shards, re-downloaded for this submission (the local copy had been deleted):

| File | Bytes | SHA-256 (as downloaded here; not checked against a published manifest) |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Download measured at **104.9 MB/s** in the first two minutes (12.58 GB
transferred); at that rate the full ~83.6 GB download projected to about
13.7 minutes total, well under this submission's 60-minute download budget,
so it ran to completion (actual total: both shards, no incomplete files).

The pack was built with upstream's normal, undocumented-flag path (the same
one `setup.py` uses for this family when a GGUF is already on disk): `tools/
iq_pack.py --gguf <shard 1> --out packs/iq3_s`, no `--compat-bf16` and no
`--experts-bin` (this is not the Q2_0 AVX-512 special case, and `--mmap-experts`
mode needs no `experts.bin`). The pack built in seconds with no tensor
conversions (`conversions.json` is empty: this quant's small tensors are
already a type the engine reads natively, unlike the Unsloth packs' BF16
hyper-connection tensors). Same MTP draft layer (`mtp/rt`) and the model's own
`data/expert-profile.bin`; same per-context settings as the two packs above.

| Context | GPU expert-cache slots | GPU VRAM for experts | RAM-resident slots | Total |
|---|---:|---:|---:|---:|
| 8,192 | 12,812 | 24.31 GiB | 11,764 | 24,576 |
| 81,920 | 12,240 | 23.23 GiB | 12,336 | 24,576 |
| 139,264 | 11,800 | 22.39 GiB | 12,776 | 24,576 |

Configs: [configs/strata-iq3_s.json](configs/strata-iq3_s.json),
[configs/strata-iq3_s-64k.json](configs/strata-iq3_s-64k.json),
[configs/strata-iq3_s-128k.json](configs/strata-iq3_s-128k.json).
Logs: [logs/strata-iq3_s.log](logs/strata-iq3_s.log),
[logs/strata-iq3_s-64k.log](logs/strata-iq3_s-64k.log),
[logs/strata-iq3_s-128k.log](logs/strata-iq3_s-128k.log).

## Method

The benchmark script, prompt builder and their README are not published in
this folder (a private local harness, `bench.py` / `make_prompts.py`). They are
described here so the method is reproducible against an equivalent harness:

- Four fixed prompts, built once by the prompt builder and reused byte-identical
  across every engine/config tested on this machine: **short** (a ~2,700-token
  code-review request over one Python module), **codegen** (a ~370-token
  request to write a self-contained ~400-line Python job-queue module, 1,024-
  token output cap), **long64k** and **long128k** (the same code-review task
  with a large multi-file Python corpus concatenated ahead of it to reach
  roughly 64K and 128K prompt tokens). The corpus concatenated into the long
  prompts is the operator's own local pipeline code, which is why the prompt
  text itself is not included in this submission; the token counts and method
  are otherwise exactly as in the prompt manifest. `short` and `codegen` use a
  512-token output cap; the long prompts use 512; codegen uses 1,024.
- Every prompt is prefixed with a unique per-run tag so no measured run can be
  served from another run's prefix cache; run N's prompt text is otherwise
  identical across configs.
- 1 warm-up request + 3 measured requests for `short` and `codegen`; 1 warm-up
  + 2 measured for `long64k` and `long128k` (a trimmed long-prompt plan, noted
  explicitly here rather than silently reported as 3). Every measured request
  in every configuration hit its output cap (`finish_reason: length`); no
  request failed or was cancelled.
- Requests streamed over the OpenAI-compatible `/v1/chat/completions` endpoint
  with `temperature=0`, `top_k=1` (greedy) and
  `chat_template_kwargs.enable_thinking=false`; `reasoning_chars` was 0 on
  every run, confirming thinking stayed off.
- **TTFT**: wall-clock seconds from request start to the first streamed content
  token (client-side, includes HTTP and local tokenization over loopback).
  **Prefill tok/s**: the server's reported prompt-token count divided by TTFT.
  **Decode tok/s**: (locally re-counted completion tokens - 1) / (time of last
  token - time of first token); completion tokens were re-counted with the
  model's own tokenizer rather than trusting server-reported usage, to keep one
  ruler across configurations.
- The GPU and system RAM were sampled once per second for the duration of each
  request (`nvidia-smi` + `/proc/meminfo`); VRAM and power figures below are the
  per-run **maximum** (VRAM) and **mean** (power) of those samples, not a single
  snapshot.
- The server process was freshly started once per context-limit configuration
  (three server starts total for the primary run, one more for the KQ256
  comparison at the 8,192-context config). Within one server's lifetime, all
  prompts for that config were measured in increasing-length order; model
  loading and expert-cache fill are excluded from every timing (they happen
  once at server start, before the warm-up request).
- `long64k` and `long128k` server starts logged the resident-RAM expert-cache
  fill event-by-event ("copied cache complement through layer N/48...").
  Decode-time per-request logs separately reported a GPU **expert-cache hit
  rate** of 80.7-93.6% across all requests, and a small, request-by-request
  growing count of "blob reads from the file" even though the full 24,576-expert
  pool fit in VRAM + the RAM budget at every context length tested (see
  Correctness and limitations: this was not investigated further).

## Results

Each cell is the median **[minimum-maximum]** of the measured runs (3 for
`short`/`codegen`, 2 for `long64k`/`long128k`). Every request hit its output
cap; no failures.

| Prompt | Prompt tokens (server) | Runs | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---|---|---|
| short | 2,700 | 3 | 1,149.1 [1,129.1-1,149.6] | 61.5 [61.0-64.59] | 2.35 [2.35-2.39] |
| codegen | 370 | 3 | 330.4 [321.2-332.1] | 72.43 [72.33-73.77] | 1.12 [1.11-1.15] |
| long64k | 63,997 | 2 | 2,454.25 [2,451.2-2,457.3] | 72.11 [71.17-73.05] | 26.08 [26.04-26.11] |
| long128k | 127,999 | 2 | 2,351.35 [2,349.3-2,353.4] | 70.15 [69.9-70.39] | 54.44 [54.39-54.48] |

| Prompt | VRAM, MiB (peak) | RAM used, GiB (peak) | GPU power, W (mean, median [min-max]) |
|---|---:|---:|---|
| short | 31,420 | 101.1 | 140.4 [139.0-144.8] |
| codegen | 31,422 | 101.4 | 141.9 [139.7-142.9] |
| long64k | 31,256 | 102.8 | 177.2 [176.5-177.9] |
| long128k | 31,260 | 104.3 | 184.15 [183.7-184.6] |

Raw per-run records: [runs/strata-ud-q4_k_xl/](runs/strata-ud-q4_k_xl/) (each
prompt's `warmup.json`/`run1.json`/`run2.json`/`run3.json`). Aggregates:
[runs/strata-ud-q4_k_xl/summary.json](runs/strata-ud-q4_k_xl/summary.json).
RAM figures are `MemTotal - MemAvailable` (process + page cache + everything
else resident), not a single process's RSS; the model/pack/expert budget
dominates it. Decode expert-cache hit rate per request ranged 80.7% (long128k,
first/coldest request) to 93.6% (short, warmest request).

### Unsloth UD-IQ4_XS (engine 0.1.31, same engine build as UD-Q4_K_XL above)

| Prompt | Prompt tokens (server) | Runs | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---|---|---|
| short | 2,700 | 3 | 1,532.9 [1,520.1-1,538.1] | 82.02 [80.62-83.01] | 1.76 [1.75-1.78] |
| codegen | 370 | 3 | 484.0 [462.0-488.2] | 98.04 [98.01-99.4] | 0.76 [0.76-0.80] |
| long64k | 63,997 | 2 | 3,052.55 [3,049.8-3,055.3] | 95.78 [92.89-98.66] | 20.97 [20.95-20.98] |
| long128k | 127,999 | 2 | 2,893.25 [2,890.7-2,895.8] | 91.36 [88.96-93.76] | 44.24 [44.20-44.28] |

| Prompt | VRAM, MiB (peak) | RAM used, GiB (peak) | GPU power, W (mean, median [min-max]) |
|---|---:|---:|---|
| short | 31,224 | 84.7 | 154.2 [152.5-159.2] |
| codegen | 31,224 | 84.9 | 163.4 [158.8-163.5] |
| long64k | 31,228 | 86.6 | 187.85 [187.0-188.7] |
| long128k | 31,228 | 88.1 | 191.9 [191.5-192.3] |

Decode expert-cache hit rate per request ranged 85.1% (long128k, first/coldest
request) to 94.9% (long64k, warmest request); 91.7-94.6% on `short`/`codegen`.
Raw per-run records: [runs/strata-0131-ud-iq4_xs/](runs/strata-0131-ud-iq4_xs/).
Aggregates: [runs/strata-0131-ud-iq4_xs/summary.json](runs/strata-0131-ud-iq4_xs/summary.json).

### ISTA-DASLab GSQ-RCO IQ3_S (engine 0.1.31, same engine build as the two packs above)

| Prompt | Prompt tokens (server) | Runs | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---|---|---|
| short | 2,700 | 3 | 1,855.1 [1,834.3-1,867.4] | 106.29 [105.78-111.21] | 1.46 [1.45-1.47] |
| codegen | 370 | 3 | 667.4 [666.4-685.5] | 123.81 [121.74-125.45] | 0.55 [0.54-0.56] |
| long64k | 63,997 | 2 | 3,281.15 [3,279.8-3,282.5] | 120.31 [116.14-124.47] | 19.50 [19.50-19.51] |
| long128k | 127,999 | 2 | 3,101.85 [3,100.5-3,103.2] | 116.5 [111.82-121.19] | 41.27 [41.25-41.28] |

| Prompt | VRAM, MiB (peak) | RAM used, GiB (peak) | GPU power, W (mean, median [min-max]) |
|---|---:|---:|---|
| short | 31,192 | 74.4 | 181.4 [171.7-182.1] |
| codegen | 31,192 | 74.8 | 185.4 [183.9-186.3] |
| long64k | 31,182 | 76.7 | 192.9 [191.6-194.2] |
| long128k | 31,190 | 78.3 | 194.6 [194.1-195.1] |

Decode expert-cache hit rate per request ranged 85.5-89.4% (first/coldest
request on `short`/`long64k`/`long128k`) to 97.9% (warmest `short` request).
Raw per-run records: [runs/strata-0131-iq3_s/](runs/strata-0131-iq3_s/).
Aggregates: [runs/strata-0131-iq3_s/summary.json](runs/strata-0131-iq3_s/summary.json).

At every context length and for every pack above (UD-Q4_K_XL, UD-IQ4_XS,
IQ3_S), all 24,576 experts fit resident across the GPU cache and the 100 GiB
RAM budget (see Model and configuration), so none of these numbers reflect
SSD-backed expert reads. On the same engine build, same prompts, same
settings, ISTA IQ3_S decodes 1.19-1.29x faster than UD-IQ4_XS and 1.66-1.73x
faster than UD-Q4_K_XL, consistent with it having the smallest/lowest-bitwidth
expert set of the three and therefore keeping more of it resident in VRAM
rather than on the CPU's AVX2 path.

### STRATA_KQ256 comparison

`STRATA_KQ256=1` turns on multi-token AVX2 kernels for the Q4_K/Q5_1/Q8_0
expert formats, against the same 8,192-context configuration. Tested on
`short` and `codegen` only (3 measured runs each).

| Prompt | Decode tok/s, default | Decode tok/s, KQ256=1 | Prefill tok/s, default | Prefill tok/s, KQ256=1 |
|---|---|---|---|---|
| short | 61.5 [61.0-64.59] | 67.61 [61.58-68.13] | 1,149.1 [1,129.1-1,149.6] | 1,146.3 [1,140.5-1,150.1] |
| codegen | 72.43 [72.33-73.77] | 71.74 [70.73-75.48] | 330.4 [321.2-332.1] | 329.8 [314.4-337.7] |

On `short`, KQ256's median is higher but its range overlaps the default
range almost entirely (61.58 min vs. 61.0 min without it); on `codegen` the two
are statistically indistinguishable (medians within 1%). With only three runs
per configuration, this is not a confident win either way; it is consistent
with the project's own published guidance that this flag is "bit-exact, but
measured no faster" on different hardware. Raw records:
[runs/strata-ud-q4_k_xl-kq256/](runs/strata-ud-q4_k_xl-kq256/).

### Same-machine comparison: other engines/builds on this card

ISTA IQ3_S and Unsloth UD-IQ4_XS were previously measured on this card on
engine 0.1.18 with a local, unpublished Q8_0-down patch (2026-09-28); those
numbers are superseded by the same-engine (0.1.31), same-prompt, no-patch
results in the Results section above and are not repeated here. For reference,
the 0.1.18-patched decode medians were 109.5/129.3/127.6/125.2 tok/s (ISTA
IQ3_S) and 97.2/110.5/107.1/109.3 tok/s (Unsloth UD-IQ4_XS) across
short/codegen/long64k/long128k - both engine versions show the same ranking
(IQ3_S fastest, then UD-IQ4_XS, then UD-Q4_K_XL), but 0.1.31's unpatched
numbers are 10-25% lower than 0.1.18's patched ones on these two packs; this
submission did not investigate why (candidates include the patch's own
changes, prefill-chunk or expert-cache default differences between versions,
or normal run-to-run variance) and that comparison is confounded by version,
patch and day, same caveat as before.

The row below (EXL3, production) **is** still a same-day/same-prompt but
different-engine comparison, not superseded by anything above, and is kept
for context:

**Production baseline: TabbyAPI/exllamav3, EXL3 4.05 bpw, 2026-09-28** (same
card, same prompts, this is the engine normally used in production on this
machine, MTP and n-gram-in-RAM both off):

| Prompt | Decode tok/s (median [min-max]) | Prefill tok/s | TTFT s |
|---|---|---|---|
| short | 34.0 [33.9-34.2] | 533.5 | 5.06 |
| codegen | 34.0 [33.8-34.1] | 248.0 | 1.49 |
| long64k | 33.6 [33.4-33.7] | 559.9 | 114.3 |
| long128k | 33.5 [33.3-33.5] | 545.5 | 234.6 |

A same-day, same-prompt variant of this engine with MTP (4 draft tokens) and
the n-gram table pinned in RAM reached 36.8-50.9 tok/s decode and 1.6-4.4x
faster prefill than the plain baseline, still well under every Strata
configuration above on these prompts. Full detail for both engines on
2026-09-28, including MTP acceptance rates, quality checks and exact
configuration diffs, is kept in the project's internal results file and is
summarized, not reproduced in full, here.

## Correctness and limitations

- **Needle-in-a-haystack recall, UD-Q4_K_XL only, 139,264-ctx server
  (`configs/strata-ud-q4_k_xl-128k.json`)**: `tools/needle_bench.py --lengths
  32k,128k --depths 10,50,90` found the planted code word in all 6
  length/depth combinations (3 depths x 2 lengths), 14-52 s per check. Full
  per-check data (prompt tokens, seconds, the exact word and the model's
  answer) is in [needles.json](needles.json). This is recall only (one short
  exact-match phrase retrieved from a long context), not a general quality or
  reasoning check, and it was run on UD-Q4_K_XL alone; UD-IQ4_XS and IQ3_S
  (above) have no needle check in this submission.
- **Beyond the needle check, no broader correctness/quality suite was run as
  part of this submission** for any of the three packs. The only other quality
  evidence available is from the 2026-09-28 same-machine study, on a different
  engine version (0.1.18, patched) and different quantizations (ISTA IQ3_S and
  Unsloth UD-IQ4_XS): a 10-check coding-quality suite scored equal to or better
  than the production EXL3 baseline under matched sampling, and MTP acceptance
  stayed far higher inside long reasoning on Strata than on the EXL3/TabbyAPI
  stack. That evidence does not cover engine 0.1.31 or the UD-Q4_K_XL pack, so
  answer quality beyond needle recall is otherwise unmeasured here.
- **Single GPU, single machine.** No multi-GPU, no concurrency, no sustained
  thermal run.
- **Greedy decoding only** (temperature 0, top_k 1). No sampled-decoding or
  reasoning/thinking-mode speed was measured in this submission.
- **Long prompts (64K/128K) were measured with only 2 runs each**, not 3, as
  stated in Method; `short` and `codegen` have 3.
- **Every expert format in every pack here runs on this CPU's AVX2 path**
  (the Threadripper PRO 3975WX has no AVX-512); the optional multi-token AVX2
  kernel (`STRATA_KQ256=1`) was evaluated on the UD-Q4_K_XL pack above and made
  no clear difference there. Decode throughput on the UD-Q4_K_XL pack is
  accordingly the lowest of the three same-engine packs measured above (see
  Results): its larger, higher-bitwidth expert set keeps less of it resident
  in VRAM relative to IQ3_S and UD-IQ4_XS.
- **An unexplained small file-read count.** Per-request engine telemetry logged
  a nonzero and request-by-request growing "blobs read from the file" count
  even though startup logs show the full 24,576-expert pool fit entirely in
  GPU VRAM plus the RAM budget (no experts left over for the SSD-backed tier).
  This submission did not investigate whether that counter reflects genuine
  recurring disk I/O, one-time fill accounting carried forward, or something
  else; see the attached engine logs for the raw lines.
- No vision, no tool use, no multi-turn conversation reuse was exercised; every
  measured request started from an empty/unique prefix by design (to defeat
  prefix caching for a clean per-run measurement), so prompt-reuse behavior is
  untested here.

## Two GPUs (layer split)

Added 2026-10-01, same machine, same engine build (0.1.31, commit `9259cad`),
same three packs and the same four prompts as above. Both GPUs are the
identical card: a second **NVIDIA RTX PRO 4500 Blackwell, 32,623 MiB**, on the
same Threadripper PRO 3975WX host (the desktop's RTX PRO 4000 was never
visible to the engine: `CUDA_VISIBLE_DEVICES` was pinned to the two RTX PRO
4500 UUIDs only). `--layer-split auto --split-device 1` was added to each
single-GPU config; everything else (quant, context limit, `--kv int8`,
`--vram-reserve-mib 1536`, `--expert-cache auto`, `--prefill auto`, `--spec 4
--spec-min-p 0.5`, prompts, run counts, sampling) is unchanged from the
single-GPU section above. Configs are
[configs-2gpu/](configs-2gpu/), engine logs [logs-2gpu/](logs-2gpu/), raw runs
[runs-2gpu/](runs-2gpu/).

**`--resident-budget-gib 100` could not be reused.** That flag silently
implies `--resident-cpu-experts` in this engine build, and the engine refuses
to start with `--resident-cpu-experts does not support layer splits or remote
expert caches` (`strata generate`, exit code 2) the moment both are present
(confirmed by a failed server start before the working configs below; the
failed attempt's config was not kept since it never produced a server or
log, only that one-line engine error). The two-GPU configs therefore drop
`--resident-budget-gib` and fall
back to the engine's plain `--expert-cache auto` mode (GPU cache per stage,
the CPU AVX2 pool computes whatever a stage's cache does not hold, same as
`docs/MULTI_GPU.md`'s own worked example) - this is the only configuration
that starts, not a choice made for speed. One consequence: peak host RAM
(`MemTotal - MemAvailable`) reads **higher** on two GPUs than on one for
UD-Q4_K_XL (126-127 GiB vs. 101-104 GiB single-GPU) despite dropping the
RAM-resident expert tier entirely; this is most likely page cache warmed by
the CPU pool reading the mmapped GGUF for whichever half of the model each
stage's cache misses (the file was also still warm in cache from the
single-GPU runs minutes earlier on the same prompts), not a genuine
per-process RSS increase - this was not isolated further (UNCERTAIN; flagged
rather than asserted).

### Split decision and expert-cache occupancy

Startup log line (`layer split auto: K=... predicted ... the caches hold N of
24576 profiled pairs`) is the auto-search's own prediction; the actual final
allocation (`expert cache N slots` for CUDA0, `layer split: CUDA1 runs layers
K-47, expert cache N slots` for CUDA1) comes a few lines later, after VRAM is
also reserved for the prompt path's per-stage borrow and (on CUDA1) the draft
head. The two numbers are close but not identical - actual is consistently
0.5-1.9% below the search's prediction, presumably the reserve taken after the
search picks K:

| Pack | Context | K (CUDA0 layers) | Predicted combined | Actual: CUDA0 slots (VRAM) | Actual: CUDA1 slots (VRAM) | Actual combined |
|---|---:|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | 8,192 | 24 | 16,843 (98.9%) | 8,550 (24.97 GiB) | 8,043 (23.47 GiB) | 16,593 (67.5%) |
| UD-Q4_K_XL | 81,920 | 23 | 16,500 (98.9%) | 8,405 (24.56 GiB) | 7,811 (22.78 GiB) | 16,216 (66.0%) |
| UD-Q4_K_XL | 139,264 | 24 | 16,219 (98.8%) | 8,237 (24.08 GiB) | 7,680 (22.41 GiB) | 15,917 (64.8%) |
| UD-IQ4_XS | 8,192 | 25 | 21,797 (99.7%) | 11,139 (25.00 GiB) | 10,428 (23.65 GiB) | 21,567 (87.8%) |
| UD-IQ4_XS | 81,920 | 25 | 21,349 (99.6%) | 10,914 (24.50 GiB) | 10,164 (23.05 GiB) | 21,078 (85.8%) |
| UD-IQ4_XS | 139,264 | 25 | 21,002 (99.6%) | 10,740 (24.10 GiB) | 9,965 (22.59 GiB) | 20,705 (84.2%) |
| IQ3_S | 8,192 | 24 | **24,576 (100.0%)** | 12,288 (22.14 GiB) | 12,170 (24.45 GiB) | 24,458 (99.5%) |
| IQ3_S | 81,920 | 25 | **24,576 (100.0%)** | 12,800 (23.08 GiB) | 11,776 (23.75 GiB) | **24,576 (100.0%)** |
| IQ3_S | 139,264 | 25 | **24,576 (100.0%)** | 12,800 (23.08 GiB) | 11,605 (23.40 GiB) | 24,405 (99.3%) |

Combined-% is of the model's 24,576 total profiled pairs, not of the routed
mass the predicted-% figures use, so the two percentages in a row are not
directly comparable; both are given as the engine itself reports them. Two
32 GB cards hold roughly **twice** the single-card slot count at matching
context (UD-Q4_K_XL 8K: 7,982 slots on one card in the single-GPU section
above vs. 16,593 actual combined here). IQ3_S's combined cache holds
essentially the entire 24,576-pair profile at every context tested (99.3-
100.0% actual) - the CPU AVX2 pool is barely used for this pack on two cards,
where on one card it carried roughly half the routed experts (11,800-12,812
of 24,576 single-GPU slots, from the Results section above).

**`--split-skip-if-fits` decision** (extra run, UD-Q4_K_XL, 139,264 context,
`configs-2gpu/strata-2gpu-ud-q4_k_xl-128k-skipfits.json`): the engine logged
`--split-skip-if-fits: CUDA0 (NVIDIA RTX PRO 4500 Blackwell) would hold only
22.17 of the profile's 71.73 GiB (26.74 GiB free, 4.57 GiB for the session,
drafter and reserve): the split stays` and ran the split exactly as the
non-skip config did. A single confirmatory measurement (1 warm-up + 1 run,
`runs-2gpu/strata-0131-2gpu-ud-q4_k_xl-skipfits/`) landed within noise of the
non-skip 128K run: 5,130.2 vs. 5,123.85 tok/s prefill, 127.15 vs. 130.15 tok/s
decode. For this pack/context the main card alone is nowhere near large
enough to hold the full 71.73 GiB profile, so the flag is a no-op here; it
would only matter for a pack/context small enough for one RTX PRO 4500 to
hold on its own (none of the three packs at any tested context qualify - even
IQ3_S's 46.84 GiB experts exceed one card's ~27-28 GiB free after reserving
VRAM for context/session/drafter).

### Results: two GPUs vs. one (median of measured runs)

Decode tokens/s, two GPUs vs. the single-GPU figure (Results section above),
and the speedup:

| Prompt | UD-Q4_K_XL 2 GPU | 1 GPU | x | UD-IQ4_XS 2 GPU | 1 GPU | x | IQ3_S 2 GPU | 1 GPU | x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 111.54 | 61.5 | 1.81x | 121.69 | 82.02 | 1.48x | 130.18 | 106.29 | 1.22x |
| codegen | 142.4 | 72.43 | 1.97x | 151.92 | 98.04 | 1.55x | 157.31 | 123.81 | 1.27x |
| long64k | 134.77 | 72.11 | 1.87x | 149.19 | 95.78 | 1.56x | 159.12 | 120.31 | 1.32x |
| long128k | 130.15 | 70.15 | 1.86x | 147.81 | 91.36 | 1.62x | 153.51 | 116.5 | 1.32x |

Prefill tokens/s (TTFT in parentheses), two GPUs vs. one:

| Prompt | UD-Q4_K_XL 2 GPU | 1 GPU | x | UD-IQ4_XS 2 GPU | 1 GPU | x | IQ3_S 2 GPU | 1 GPU | x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1,791.0 (1.51s) | 1,149.1 (2.35s) | 1.56x | 2,158.5 (1.25s) | 1,532.9 (1.76s) | 1.41x | 2,229.4 (1.21s) | 1,855.1 (1.46s) | 1.20x |
| codegen | 607.5 (0.61s) | 330.4 (1.12s) | 1.84x | 952.2 (0.39s) | 484.0 (0.76s) | 1.97x | 996.7 (0.37s) | 667.4 (0.55s) | 1.49x |
| long64k | 4,920.0 (13.01s) | 2,454.25 (26.08s) | 2.00x | 5,649.75 (11.33s) | 3,052.55 (20.97s) | 1.85x | 5,642.85 (11.34s) | 3,281.15 (19.50s) | 1.72x |
| long128k | 5,123.85 (24.98s) | 2,351.35 (54.44s) | 2.18x | 5,769.9 (22.18s) | 2,893.25 (44.24s) | 1.99x | 5,757.25 (22.23s) | 3,101.85 (41.27s) | 1.86x |

**The pattern is consistent across all three packs: the bigger/lower-bitwidth
a pack's expert set relative to one card, the more two cards help.**
UD-Q4_K_XL, the largest pack (7,982-8,550 of 24,576 pairs fit one card, the
rest ride the CPU AVX2 pool), nearly doubles decode (1.81-1.97x) and more than
doubles long-prompt prefill (2.00-2.18x). IQ3_S, which already fit 11,800-
12,812 of 24,576 pairs on one card and now fits all 24,576 on two, gains the
least on decode (1.22-1.32x) because the second card is removing a smaller
CPU-pool fraction, but still gains meaningfully on prefill (1.20-1.86x) from
the pipelined chunk reads docs/MULTI_GPU.md describes. UD-IQ4_XS sits between
the two, as its single-card cache occupancy (9,585-10,434 of 24,576) does too.

VRAM (peak, MiB) and power (mean W) per card, and peak host RAM:

| Prompt | Pack | CUDA0 (main) VRAM | CUDA1 VRAM | CUDA0 power (active, per-request) | CUDA1 power (session mean, incl. idle gaps)\*\* | Host RAM peak |
|---|---|---:|---:|---:|---:|---:|
| short/codegen | UD-Q4_K_XL | 31,130 | 31,324 | 114.1 / 120.2 W | 123.7 W | 125.9-126.0 GiB |
| long64k | UD-Q4_K_XL | 31,136 | 31,330 | 150.6 W | 159.7 W | 126.5 GiB |
| long128k | UD-Q4_K_XL | 31,134 | 31,322 | 174.55 W | 171.4 W | 127.2 GiB |
| short/codegen | IQ3_S | 27,362 | 31,248 | 121.4 / 128.0 W | 137.6 W | 100.6-100.7 GiB |
| long64k | IQ3_S | 28,840 | 31,138 | 166.35 W | 157.2 W | 101.2 GiB |
| long128k | IQ3_S | 29,244 | 31,242 | 175.15 W | 168.5 W | 99.8 GiB |
| short/codegen | UD-IQ4_XS | 31,138 | 31,258 | 122.0 / 129.1 W | 131.4 W | 108.4-108.6 GiB |
| long64k | UD-IQ4_XS | 31,140 | 31,258 | 159.9 W | 156.9 W | 109.3 GiB |
| long128k | UD-IQ4_XS | 31,138 | 31,260 | 175.2 W | 167.7 W | 109.7 GiB |

\*\* CUDA1 was sampled by a second, independent `nvidia-smi` poll at 1 Hz for
the whole server lifetime (not per-request like `bench.py`'s own `--gpu`
sampling of CUDA0), so its power figure is a session mean including the idle
seconds between requests and is a few watts to ~15 W lower than a true
per-request average would read; VRAM peaks are unaffected by this (a peak is
a peak). Raw samples: `logs-2gpu/sec-*.csv`.

### Limitations added by this section

- **`--resident-budget-gib`/`--resident-cpu-experts` and `--layer-split` are
  mutually exclusive on engine 0.1.31** (see above) - this is a hard engine
  check, not a configuration choice, and it means the two-GPU numbers above
  are not on the same expert-residency mode as the single-GPU numbers earlier
  in this report; they use the engine's other, non-experimental caching mode
  instead, the one `docs/MULTI_GPU.md` itself demonstrates.
- Same caveats as the single-GPU section apply here too: greedy decoding
  only, long prompts measured with 2 runs (1 for the skip-if-fits check),
  single machine, no concurrency, no broader quality suite re-run on two
  GPUs.
- Host RAM's higher reading on two GPUs than one (noted above) was not
  isolated from warm page cache left over from the single-GPU runs earlier
  the same day; treat it as UNCERTAIN, not a confirmed two-GPU RAM cost.

### 256K prompt on two GPUs

Added the same evening: UD-Q4_K_XL, the two RTX PRO 4500, `--max-context 266240`
([configs-2gpu/strata-2gpu-ud-q4_k_xl-256k.json](configs-2gpu/strata-2gpu-ud-q4_k_xl-256k.json)),
a 255,988-token prompt (`prompts/long256k.txt` in the bench harness: a second pass over the same
source corpus, since one pass tops out near 201K tokens), 1 warm-up + 2 measured runs, greedy,
512-token output cap. Raw runs: [runs-2gpu/strata-0131-2gpu-ud-q4_k_xl/long256k/](runs-2gpu/strata-0131-2gpu-ud-q4_k_xl/long256k/);
engine log [logs-2gpu/strata-2gpu-ud-q4_k_xl-256k.log](logs-2gpu/strata-2gpu-ud-q4_k_xl-256k.log).

| Prompt | Prompt tokens (server) | Runs | Prefill tok/s | Decode tok/s | TTFT s |
|---|---:|---:|---|---|---|
| long256k | 256,011 | 2 | 4,883.8 [4,880.5-4,887.0] | 114.7 [113.32-116.09] | 52.42 [52.39-52.46] |

Peak VRAM 31,134 MiB on the main card; host RAM in use 124.7 GiB; decode expert-cache hit 98.0-98.5 %.
Against the same pack at 128K on two cards (5,124 tok/s prefill, 130.2 tok/s decode, 25.0 s TTFT):
prefill scales linearly with prompt length and decode gives up about 12 % as the larger KV cache
takes room from the expert cache. The model's full 262K window is usable on two cards.

## Three GPUs (layer split)

Added 2026-10-01, same engine build and packs, with the desktop's **NVIDIA RTX
PRO 4000 Blackwell (24 GB, ~1 GB used by the compositor)** as the third pipeline
stage behind the two RTX PRO 4500 cards: `CUDA_VISIBLE_DEVICES` = 4500 (slot 5),
4500 (slot 3), 4000; `--layer-split auto --split-device 1,2`; everything else as
in the two-GPU section. Configs [configs-3gpu/](configs-3gpu/), engine logs
[logs-3gpu/](logs-3gpu/), raw runs [runs-3gpu/](runs-3gpu/). UD-IQ4_XS at 128K
was not run (time).

Decode tok/s, median, 1 / 2 / 3 GPUs:

| Prompt | UD-Q4_K_XL | UD-IQ4_XS | IQ3_S |
|---|---|---|---|
| short | 61.5 / 111.5 / 112.6 | 82.0 / 121.7 / 112.0 | 106.3 / 130.2 / 131.7 |
| codegen | 72.4 / 142.4 / 137.9 | 98.0 / 151.9 / 145.6 | 123.8 / 157.3 / 151.6 |
| long64k | 72.1 / 134.8 / 135.8 | 95.8 / 149.2 / 145.7 | 120.3 / 159.1 / 150.8 |
| long128k | 70.2 / 130.2 / 131.5 | 91.4 / 147.8 / not run | 116.5 / 153.5 / 144.5 |

Prefill tok/s (TTFT s), three GPUs:

| Prompt | UD-Q4_K_XL | UD-IQ4_XS | IQ3_S |
|---|---|---|---|
| short | 1,650 (1.64) | 1,808 (1.49) | 1,894 (1.43) |
| codegen | 589 (0.63) | 902 (0.41) | 945 (0.39) |
| long64k | 4,265 (15.0) | 3,635 (17.6) | 2,340 (27.4) |
| long128k | 4,093 (31.3) | not run | 2,295 (55.8) |

Split decisions (`layer split auto`): UD-Q4_K_XL K=22,43 / 21,42 / 20,39 with
the caches holding 19,839 / 20,051 / 21,400 of 24,576 profiled pairs at 8K / 64K /
128K; UD-IQ4_XS K=23,46 holding 23,552 / 23,131; IQ3_S K=22,47 / 23,47 / 23,47
holding all 24,576. The third stage lands on the 4000 (its session, the last
2-5 layers and the head; 17 GiB free after the weights).

Reading: a third card adds nothing to decode once two cards already hold about
99.5 % of the routed expert mass (two-GPU section), and it costs long-prompt
prefill, which now crosses two card boundaries with the slowest card last:
IQ3_S at 128K reads at 2,295 tok/s on three cards against 5,757 on two. On this
machine two RTX PRO 4500 cards are the configuration to use.
