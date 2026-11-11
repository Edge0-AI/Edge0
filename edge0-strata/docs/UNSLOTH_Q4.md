# Unsloth UD-Q4_K_XL

This page is for people who want to run Unsloth's 4-bit Qwen3.8-Flash-Next file (UD-Q4_K_XL) with Strata, and for
people who pack it by hand. It also covers UD-IQ4_XS and plain Q4_0 files. For UD-Q5_K_XL, UD-Q6_K_XL and Q8_0, see
[UNSLOTH_Q6.md](UNSLOTH_Q6.md). Quality, speed and images across the Unsloth sizes are in [QUANTS.md](QUANTS.md).

The target is Unsloth's 4-bit quantization of the same model the other packs use:
[unsloth/Qwen3.8-Flash-Next-GGUF, `UD-Q4_K_XL`, revision `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL).
"UD-Q4_K_XL" is a mix of formats, not Q4_K everywhere:

- Routed experts: Q4_K (Q5_K in layer 2) for gate/up, Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) for down. They are
  71.7 GiB, about twice the 3-bit models'.
- Token embedding, output head, attention and shared-expert projections, and the PLE key: Q8_0.
- The 28.8 GB PLE table: IQ4_NL.

## Quick start

Run `strata launch` and choose **Qwen3.8-Flash-Next (Unsloth)**, or:

```sh
strata launch --family unsloth --model UD-Q4_K_XL
```

## What setup does

- **Where the experts live.** Every expert goes into RAM when its 72 GiB plus 16 GiB more fit (about 88 GB of RAM). On a
  PC with less RAM, or on a unified-memory APU, setup keeps a RAM budget: the hottest experts in RAM, the rest read
  from the GGUF on the SSD. It needs 48 GB of RAM or more. With less it asks (default no), and
  `--model UD-Q4_K_XL --yes` installs it anyway.
- **The RAM budget** is the PC's RAM less 24 GB (`--resident-budget-gib 40` on 64 GB), at most all 71 GiB of experts. From
  a 64K context up, where the KV cache moves to RAM, its size comes out of the budget. `--resident-budget-gib N` at
  setup sets another one; a bigger one is kept, with a note on what it risks.
- **Download.** The four shards below from the pinned revision `38bb39e`, resumable. Setup checks each one's size and
  SHA-256 against the table below once and remembers it in the file's finish mark. A file with the wrong hash is
  deleted, so the next run downloads it again.
- **Packing.** `--compat-bf16`, and never an `experts.bin` (`--low-ram` does not apply).
- **Defaults it writes** into the run config's `env`: `STRATA_HC_Q8=1` on CUDA (the GGUF's own Q8_0 hyper-connection
  projections, closer to the file and faster), and `STRATA_ROUTE_TAIL_SKIP=0` where every expert is in RAM. UD-Q4_K_XL
  keeps `STRATA_MMQ_KQ` opt-in.
- **Context.** It recommends 8K on a GPU under 14 GB, because every GB of KV cache is a GB less of cached experts.
- **GPUs.** NVIDIA or AMD. One GPU is the default in the budget mode. With several GPUs, each card caches the experts of
  its own layers and the budget holds the hottest of the rest by the whole expert profile. Setup asks; `--gpus 0,1`
  takes the split, and a start on several GPUs keeps `--resident-budget-gib`.
- **Pictures.** The reader is the CPU one below 24 GB of VRAM and the GPU one from 24 GB, when the package has a reader.
  No experimental speed projection is used with this file.

## The files

Four shards, 111,334,654,784 bytes (111.3 GB) together. Shard 1 holds only the metadata. Layer 11's down projection is
in shard 2 and its gate/up in shard 3, which this engine handles (`native_experts.txt` v4).

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |

Keep the four files together in one folder under these names. The engine finds shards 2 to 4 from shard 1's name, and a
missing shard is an error that names it. By hand, check the files before packing (`sha256sum -c`, or `Get-FileHash` on
Windows). Hugging Face snapshot symlinks work if you pass the snapshot's file name, not the hash-named blob it points to.

## What it needs

| | |
| --- | --- |
| Disk | 111.3 GB for the four files, 1.4 GB for the pack, about 6 GB for the MTP draft layer if you have none yet. There is **no `experts.bin`**: the engine reads the experts from the GGUF files in place. An NVMe SSD matters, because experts that fit neither VRAM nor the RAM budget are read from it for every token. |
| RAM | 48 GB minimum; about 88 GB for every expert in RAM. The RAM budget holds the most-used experts and the rest come from the SSD through the OS file cache. |
| GPU | 12 GB works: after the weights, the draft layer and the buffers, the expert cache held 1,280 of the 24,576 experts (3.7 GiB) at 4K context on an RTX 5070. |

## The pack (by hand)

Build Strata, then from the repository root, with gguf-py from the pinned llama.cpp (setup installs it;
`STRATA_GGUF_PY` can point to its `gguf-py` folder):

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --out packs/ud-q4_k_xl --compat-bf16
```

It takes about 30 seconds and writes `index.txt`, `dense.bin` (1.4 GB), `native_experts.txt`, the tokenizer,
`conversions.json` and `compat-bf16.json`. **`--compat-bf16` is required.** This file stores 195 small projections (the
hyper-connection up/down matrices, `output_hc_*` and the PLE value) as Q8_0, while the engine reads them as BF16.
Without the flag the packer refuses and writes nothing. With it, they are dequantized and rounded to BF16
(nearest-even). Do not add `--experts-bin`: it would write a 77 GB copy of experts that this mode does not need.

## The numbers: what is native and what is converted

The routed experts, the token embedding, the output head, the attention and shared-expert projections, the PLE key and
the PLE table are used as the GGUF stores them. The engine has GPU kernels and CPU (ggml) paths for Q4_K, Q5_K, Q5_1 and
Q8_0.

The small tensors the engine reads as floats are converted. `conversions.json` lists every one with its source and
destination type, method, whether it is exact, the largest absolute error and the source bytes' SHA-256:

- 264 F32 tensors (routers, injections, SSM gates) whose values are already BF16's are stored as BF16 exactly.
- The 195 Q8_0 projections above are rounded to BF16 (largest absolute error 0.0144, in `hc_ffn_up`).
- The PLE convolution (F32) is narrowed to F16 when the engine loads it (largest error 3e-8).

That is the compatibility Unsloth's file needs. It is not the original BF16 checkpoint. The output matches llama.cpp's on
the same file ([Quality](#quality-against-llamacpp-on-the-same-file)).

## The MTP draft layer

The draft layer is the base model's, the same one the other packs use. If setup installed Strata, you have it (`mtp/rt`
in `Strata-data`). Otherwise build it as [ORCA.md](ORCA.md#preparation) shows (`tools/mtp_fetch.py`,
`tools/mtp_pack.py`, `tools/mtp_rt.py`, then copy `data/draft_vocab.bin` to `mtp/rt/`).

## The server

Save as `strata-ud-q4_k_xl.json` at the repository root (or next to setup's other `strata-*.json`), with `/path/to/`
pointing at **shard 1**. `--ple-gguf` is not needed: the engine finds the shard that holds the PLE table
(`per_layer_token_embd.weight`, shard 2) by name. `--resident-budget-gib` turns on the mapped mode with a RAM budget (it
implies `--mmap-experts`).

```json
{
  "exe": "engine/strata.exe",
  "args": [
    "--pack", "packs/ud-q4_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "--resident-budget-gib", "40",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q4_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q4_k_xl",
  "log": "strata-ud-q4_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-ud-q4_k_xl.json --port 8080
```

`engine/strata.exe` is setup's engine on Windows; a self-built one is `build/strata`. The expert profile is the base
model's (`data/expert-profile.bin`). It ranks the same 48 x 512 experts and decides which ones the GPU cache and the RAM
budget take first.

**Choosing the RAM budget N.** Use your RAM minus 20 to 24 GB (the OS, the engine itself, the 126 MB router copy, and
room for the OS file cache that serves the rest). On 64 GB that is 40. The engine checks available RAM and, on Windows,
available commit capacity. If N exceeds the smaller value minus 4 GiB headroom, it reduces N to that amount minus a
256 MiB margin and reports which limit applies. Everything above N comes from the SSD for every token, so N is the
setting that matters most; a bigger budget was faster in every measurement (24 / 32 / 40 GiB).

When the driver page-locks the whole budget, the GPU also computes a share of the misses over PCIe, as in the resident
low-RAM mode. `STRATA_RESIDENT_PIN=0` uses ordinary pageable memory instead, and the CPU then computes every miss. On
Windows the allocation log reports available physical RAM and commit capacity before allocation and after a failure,
because free RAM alone does not guarantee that Windows can commit the requested memory.

**Context.** The KV cache takes VRAM from the expert cache, so a longer context makes decoding slower. 8K is a reasonable
start on 12 GB. `--kv int8` halves the KV cache's VRAM.

## Quality: against llama.cpp on the same file

> **Measured on:** Strata 0.1.32 era engine against llama.cpp pinned commit `3cf0325` (CPU build, GGUF memory-mapped).
> Strata ran with RAM budget 40 GiB, the MTP draft layer on, greedy and `--adapt-every 100000`. Hardware is not recorded
> on this page.

Both were given the same token sequences, and at every position each wrote its 20 most likely next tokens with their
log-probabilities. Strata used its verify windows (`STRATA_LOGPOS` with `STRATA_LOGPOS_TOPK=20`), llama.cpp used
`llama_get_logits_ith` over one batch. Per position: whether the most likely token is the same (argmax agreement), how
many of the 5 and 10 most likely tokens both have, the KL divergence (llama.cpp's distribution against Strata's, over
llama.cpp's top 20 plus one bucket for the rest), and the perplexity of the sequence under each.

| Token set | Positions | Argmax agreement | Top-5 / top-10 overlap | KL | Perplexity Strata / llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code: Strata's greedy answer to a coding prompt (an LRU cache in Python, 56-token prompt) | 400 | **99.0%** | 95.6% / 95.3% | 0.0008 | 1.067 / 1.070 |
| Thinking: Strata's greedy answer to a reasoning puzzle (with thinking, 55-token prompt) | 324 | **97.5%** | 96.2% / 96.4% | 0.0016 | 1.107 / 1.106 |
| After a 16K prompt: Strata's greedy continuation (256 tokens) of the 16,384 real-text tokens below | 256 | 90.2% | 88.1% / 89.3% | 0.039 | 1.619 / 1.623 |
| Real text after a 16K prompt: the last 510 tokens of 16,384 tokens of this repository's docs and code | 510 | 91.2% | 88.3% / 89.9% | 0.035 | 4.27 / 4.29 |
| The same 511 positions with only 512 tokens before them | 511 | 89.2% | 86.0% / 86.3% | 0.061 | 8.99 / 8.87 |

For comparison, [eddoursul/Strata](https://github.com/eddoursul/Strata) reported 98.2 to 99.3% (code), 95.6 to 95.8%
(thinking) and 97.7% (after a 16K prompt) for the same file against llama.cpp on an RTX 3090.

The two differ at near-ties. Where llama.cpp's own gap between its two most likely tokens is 0.2 or more, Strata picks
the same token at every position of the code and thinking answers (695 of 695). After the 16K prompt it does so at every
position where the gap is 0.5 or more (144 of 144 in the continuation, 254 of 254 in the real text). Text after a long
document has many more near-ties than a short answer (58 of the continuation's 256 positions and 135 of the real text's
510 have a gap under 0.2, against 13 of the code answer's 400), which is why its agreement is lower. The perplexities
match within 3% in every set. The 16K prompts were read by Strata's batched prompt path (15,872 tokens), the rest
through the verify windows. These numbers are the FP16 prompt path's. With the MMQ prompt path (`STRATA_MMQ_KQ=1`) the
continuation after the 16K prompt agrees at 92.2% (KL 0.029, three runs, identical).

To repeat it, set `STRATA_LOGPOS=<file>` with `STRATA_LOGPOS_TOPK=20` on a serve engine with `--short-read` covering the
positions to compare. llama.cpp's side was a short program over `llama_decode` and `llama_get_logits_ith` writing the
same top 20 per position.

## Opt-ins and switches (environment variables)

| Variable | Effect |
| --- | --- |
| `STRATA_LOOKAHEAD=0` | Turns off the routing-aware prefetch (on by default in this mode). While the CPU works on a layer, a thread applies the next layer's router to this layer's input and asks the OS to read the predicted experts' pages. Pages only; the answers are the same. `STRATA_LOOKAHEAD_K` sets the experts per token (default 10). |
| `STRATA_MMQ_KQ=1` | Multiplies the Q4_K / Q5_K / Q5_1 experts of a prompt with llama.cpp's MMQ kernels instead of dequantizing them to FP16. It rounds differently from the FP16 path, so it is opt-in for this file. On CUDA the kernels are a library next to the engine (`libstrata_kq.so` / `strata_kq.dll`), opened only then, so they take no VRAM from other models ([QUANTS.md](QUANTS.md)). |
| `STRATA_KQ256=1` | Multi-token AVX2 kernels for the Q4_K / Q5_1 / Q8_0 experts. Bit-exact with ggml's, but measured no faster, so off. |
| `STRATA_PARTIAL_PIN=1` | Registers the hottest part of the RAM budget (up to `STRATA_PARTIAL_PIN_GIB`, default 24) with the GPU driver, so the GPU computes a share of the misses over PCIe (`--pcie-frac`). Measured no faster, and it changes the numerics of those experts, so off. |
| `STRATA_FETCH_THREADS=N` | Threads that read the experts from the GGUF while it answers (default 8). The prompt path has its own: `STRATA_STAGER_THREADS` (default 32 here) and `STRATA_STAGER_RING` (128). |
| `STRATA_RESIDENT_PIN=0` | Ordinary pageable memory for the RAM budget (see above). |

## UD-IQ4_XS

Unsloth's `UD-IQ4_XS` at the same revision is in setup as a regular choice, the Unsloth family's first size:

```sh
strata launch --family unsloth --model UD-IQ4_XS
```

It sits between IQ3_S and UD-Q4_K_XL. Its routed experts are IQ3_S gate/up (IQ4_XS in one layer) with IQ4_NL downs (Q8_0
in five layers), 59.5 GB of them. The dense side (Q8_0 projections, the IQ4_NL PLE table, a Q6_K head) is UD-Q4_K_XL's.
Three shards, 93.7 GB:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | 10,946,624 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | 49,835,229,856 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | 43,836,407,744 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |

Setup treats it like UD-Q4_K_XL: the same RAM budget rule (your RAM less 24 GB, at most all 55 GiB of its experts, so a
PC with about 80 GB of RAM or more holds all of them), the pack with `--compat-bf16`, no `experts.bin`, and one GPU by
default. It uses the original model's image encoder.

It packs and runs on a Strix Halo (AMD gfx1151, 128 GB unified memory), where prompts read at about 820 tokens/s at 8K
and 64K and the answers passed the retrieval checks at 8K and 64K. It has not been measured on NVIDIA; please report
what you see.

One fidelity note applies to every `--compat-bf16` pack, UD-Q4_K_XL included. The pack rounds the file's Q8_0
hyper-connection projections to BF16, and on the Strix Halo that put perplexity 6 to 9% above llama.cpp's on the same
file (teacher-forced, short context). `STRATA_HC_Q8=1`, which setup writes on CUDA for the UD files, reads the file's Q8_0
values instead of the rounded BF16. The 6 to 9% gap was measured on AMD, where setup does not write it.

## Q4_0 files

A plain `llama-quantize` Q4_0 file of this model runs the same way: pack shard 1 with `--compat-bf16` as above and point
`--native` at it. Tested:
[bartowski/Qwen3.8-Flash-Next-GGUF, `Qwen3.8-Flash-Next-Q4_0`, revision `928589f`](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/tree/928589fdb66c6ff07f22ac561e3fbce76553548f/Qwen3.8-Flash-Next-Q4_0)
(3 shards, 100.6 GB). Its routed experts are Q4_0 for gate/up and for down, except down in layers 0 to 5, which is Q4_1.
The PLE table and the token embedding are Q4_0. The pack serves 302 tensors from the GGUF as stored and converts 362
(168 exact, 194 rounded to BF16, largest absolute error 0.0156; 1.23 GiB).

The Q4_0 expert kernels centre the codes (q - 8) before the integer dot product, as ggml-cpu's Q4_0 x Q8_0 dot does,
instead of llama.cpp's CUDA form (uncentred codes, minus 8 x the Q8_1 block sum in float). Q4_1 follows the Q5_1 kernel
without the fifth bit. `native_expert_parity` on this file (layers 0 and 5: Q4_0/Q4_1, layers 20 and 47: Q4_0/Q4_0)
gives a relative error against the float product of 1.1 to 1.3e-2 on the CPU and the GPU alike, and CPU against GPU of
1.1e-7 for Q4_0/Q4_0 and 5e-4 to 1.1e-3 with Q4_1 down.

> **Measured on:** one RTX 3060 12 GB (PCIe 3.0 x16), Xeon E5-2696 v4, 121 GiB DDR4-2400, Linux, CUDA 12.4. Every
> expert in RAM, `--max-context 131072 --kv int8 --kv-resident 20480`, expert cache `auto` (1,960 slots, 5.06 GiB),
> prompt chunk `auto` (6,144 tokens), greedy, `reasoning_effort` medium, one request at a time. Strata version not
> recorded.

| Prompt | Prompt read | Output |
| --- | --- | --- |
| 63 tokens (prose) | - | 25.4 tok/s |
| 111 tokens (code) | - | 28.0 tok/s |
| 24,461 tokens | 787 tok/s | 24.0 tok/s |
| 67,934 tokens (a fact hidden in the middle: found) | 737 tok/s | 26.4 tok/s |

A coding prompt was answered correctly. Not measured: agreement with llama.cpp on this file, AMD, Windows, and
bartowski's Swift-1.5 Q4_0, whose header shows the same formats.

## Scope and validation

- UD-Q4_K_XL and UD-IQ4_XS at revision `38bb39e` are targeted, and plain Q4_0 files. Other quantizations use formats this
  engine may not have kernels for. The engine checks every layer's formats at start and refuses an unsupported one by
  name.
- Tests: the packer's synthetic 4-shard and conversion tests
  (`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`) and the CTests `gguf_split_test`,
  `expert_layout_test`, `native_expert_parity_*` (the three real expert format pairs against ggml-cpu, the Q5_1 min term,
  Q8_0 rows, Q4_0/Q4_0 and Q4_0/Q4_1). With `-DSTRATA_MMQ_KQUANTS=ON`, `prefill_mmq_kquant_test` checks the prompt path's
  MMQ products for Q4_K / Q5_K / Q5_1 / Q8_0 against ggml's dequantized weights.
- Not validated: sampled decoding and long conversations. Please report what you see.

## Credits

The split-artifact loader, the expert formats' GPU kernels and the packing of Unsloth's files follow
[eddoursul/Strata](https://github.com/eddoursul/Strata) (PR #245), which ran these files first. The Q8_0 kernels, the
per-role shard column and the PLE convolution's F32 to F16 fix come from @gopinath87607's PR #255. Per-role shard names
were also proposed by @jagsan-cyber in PR #247. The quantization is [Unsloth](https://huggingface.co/unsloth)'s; the
model is Qwen's ([license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).

## History / older measurements

### RAM budget, speed on a 12 GB card

> **Measured on:** RTX 5070 12 GB, 64 GB RAM, Windows 10, AVX-512 CPU, UD-Q4_K_XL, one short greedy prompt, first
> validation (2026-10-01, engine 0.1.31/0.1.32).

| RAM budget | Greedy output | Notes |
| --- | --- | --- |
| 24 GiB | 5.1 to 5.2 tok/s (2 runs, before the routing prefetch) | 2.2 GB read from the SSD per verify round (about 3.5 tokens) |
| 40 GiB | 7 to 8.5 tok/s (6 runs: 6.7 to 8.6, mean 7.9) | 1.1 GB read from the SSD per verify round, about 0.33 GB per token |

- The first answer comes about 60 s after the engine starts (the RAM budget is copied from the GGUF files at start). A
  30-token prompt then took 5 to 6 s.
- A 16K prompt was read at 160 tokens/s (15,873 tokens in 99 s) on engine 0.1.32, up from 57 tokens/s (273 to 282 s) on
  0.1.31, after the prompt path began reading experts with 32 threads and up to 128 blobs in flight.
- The routing-aware prefetch added a mean of +14% (6.9 to 7.9 tok/s); about half of the SSD reads were predicted.
- With `STRATA_MMQ_KQ=1` the products took 4x less GPU time, but on this PC the prompt waited for the SSD either way (29 s
  per 4K chunk).
- Where the time goes at 40 GiB (`--stats`): about 550 to 750 ms of each verify round (3.5 tokens) reading experts from
  the SSD, about 75 ms the CPU's expert kernels, about 13 ms the GPU.
- For comparison, IQ3_S (all its experts in RAM) wrote about 53 tok/s on the same PC
  ([the speed tables](DETAILS.md#speed-measured)).

### Two GPUs with the RAM budget

> **Measured on:** RTX 3090 (x16) + RTX 3080 Ti (x4), 91 GB RAM, budget 64 GiB, engine 0.1.41.

Decode 45 tok/s against 38 on the 3090 alone (the auto split gives the x4 card two layers); 57.6 GiB of experts in RAM,
no file reads per request.

> **Measured on:** 2x RTX 3090, 165 GiB RAM, an older engine that ran the split without the RAM budget (all 77 GB of
> experts loaded into RAM at start).

Decode 31 tok/s on one card with the budget, 64 to 78 tok/s split (55 tok/s at a 128K prompt), with `MemAvailable` never
under 68 GiB.
