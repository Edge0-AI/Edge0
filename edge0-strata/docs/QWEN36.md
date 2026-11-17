# Qwen3.6-35B-A3B: pictures, several requests at once and speed

This page is for people who run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of
experts, 3B active, 256 experts, top-8) with Strata and want to know how pictures, parallel requests and the prompt
chunk behave. For sizes and memory, see [MODELS.md](MODELS.md).

Strata runs [Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
(UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own engine path (`src/qwen35`; `strata` switches to it from the file's
architecture). It runs on NVIDIA and AMD cards. The Intel Arc and Mac engines do not run it.

## Quick start

Pick it in the app's model list, in `strata launch` or in setup:

```sh
strata launch --family qwen36 --model ud-q4_k_xl
strata launch --family qwen36 --model ud-q4_k_xl --vision on --parallel 2
```

| Option | Effect |
| --- | --- |
| `--vision on\|off\|gpu\|cpu` | Terminal: pictures and which reader. Source setup uses `--vision yes\|no\|gpu\|cpu` and `--vision-tokens N`. |
| `--parallel N` | How many requests are answered at once, 1 to 8. Source setup has the same option. |
| `--prefill 512` | Engine flag in the config's `args`, not a `strata launch` option: the prompt chunk size ([Prompt chunk](#prompt-chunk)). |

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT
with a 2x2 merger, the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width
differs: 2048 here, 2560 there). Strata runs it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the
encoder choices are the Flash-Next ones ([Images](DETAILS.md#images-vision)).

Pictures are on by default, by the same rule as for every model on NVIDIA and AMD cards:

- The CPU reader runs under 24 GB of graphics memory. It takes a few seconds per picture and no VRAM.
- The card's reader runs from 24 GB (GPU encoder, about 1.2 GB of VRAM at up to 1,024 image tokens). On AMD it is
  `strata-vision-hip`, with the CPU reader as its fallback.

Your explicit choice is kept, with a note. A picture becomes up to 1,024 tokens of the context (the CPU default is 300).
The engine puts the encoder's rows where the prompt has `<|image_pad|>` tokens and gives every row its 2-D position,
interleaved M-RoPE as llama.cpp does for this model (rotary sections 11 / 11 / 10).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache. The same conversation
sent again continues from the held state when its pictures are the same (matched by the server's key for each
picture). A different picture starts that conversation again.

> **Measured on:** RTX 3060 12 GB (CUDA) and an R9700 box (32 threads) and a 3060 box (10 threads), UD-Q4_K_XL, Strata
> 0.1.45.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`) or `--parallel N` makes the engine keep N
conversations open (`--batch N`, up to 8) and decode them together. Every window carries one token of each
conversation, so the dense weights, the head and every routed expert that conversations share are read once for all of
them. The server speaks the same engine protocol as for Flash-Next ([BATCHING.md](BATCHING.md)).

A new request waits only for the read of its own prompt, which runs between the decode windows of the others. A
conversation continues in the slot that holds its start, or the engine copies the longer held start from another slot.
A request that runs alone (the solo path, with its MTP drafts) moves into a batch slot when another arrives, and back
when it is alone again, and gives the same text either way.
`STRATA_Q36_READ_WINDOWS=4` lets the decoding slots produce more tokens during a long prompt read, at the cost of a
slower read. The default is one decode window per chunk.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts. With several requests the batch window has
no drafts unless `--batch-mtp` is in the config's `args` (opt-in). Each slot's window then also carries the proposal its
MTP block made, one per slot, up to the rows to spare in the 8-row window.

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10
attention layers and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. Setup writes the context for one
conversation, so N slots take N times that from the room the experts would have. On a card that does not hold every
expert (12 and 16 GB, the hybrid mode) the expert slots lose that room, and a batch reads more distinct experts from RAM.

Setup offers `--parallel` where the sessions do not eat the experts' room, says the price in the same breath and asks
"How many requests at once?" Enter keeps one at a time, and any number you give is kept.

| Card type | Slots setup offers |
| --- | --- |
| Every expert still fits | Up to 3 to 4 (3 on a 32 GB card at 32K) |
| Hybrid mode | The count whose sessions take at most 35% of the room the experts would have. 12 GB: 4 slots at 8K, none at 32K. 16 GB: 2 slots at 32K. |

The engine says what the sessions take before it allocates them and refuses, with the numbers, a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others. The
experts that sit on the GPU or on the CPU change with the routing of all the requests, and the two compute with
different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its solo
run there too.

## Prompt chunk

The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per expert. With every
expert in VRAM the engine takes the largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve
and the sessions' caches.

| Situation | Chunk |
| --- | --- |
| Every expert in VRAM, 8K to 64K context | 4096 |
| Every expert in VRAM, 100K context | 2048 |
| Every expert in VRAM, 131K context | 512 |
| `--batch 2..4` | 1024 at most, so the other slots are not held for a whole chunk |
| `--batch 8` | 512 |
| Hybrid mode | 2048; 1024 or 512 on a card with 7.5 or 5.5 GiB free |

In the hybrid mode a bigger chunk takes expert slots (an RTX 3060 at chunk 4096 read prompts 5% faster and decoded 13%
slower). `--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the 512 size, and then the text is the base engine's, bit
for bit. The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the prompt state. This is
the size of the llama.cpp CPU-against-HIP difference on the same file, not a quality loss. A prompt shorter than the
chunk (every chat turn after the first) is one chunk either way.

## Speed and exactness

Every change on the decode path gives the old kernels' bits. `STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old
kernel beside each new one on the real activations and compares them bit for bit. `STRATA_Q36_OLD_KERNELS=1` switches
the new kernels off. Gate/up for Q4_K experts uses the V2K kernels (`STRATA_EXPERT_V2K`, the default on gfx12).

> **Measured on:** AMD Radeon AI PRO R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, Strata 0.1.45 against the engine before it,
> 5 interleaved pairs, medians; llama.cpp 972b5fc HIP on the same file (`-fa 1`).

| | Before | 0.1.45 | llama.cpp |
| --- | ---: | ---: | ---: |
| Decode tg128 | 83.2 tok/s | 117.8 | 81.9 to 82.1 |
| Prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 3,794 (`-ub 2048`) / 3,042 (`-ub 512`) |

More numbers are in [History](#history--older-measurements).

## Limits

- Eight slots at most (the window holds 8 rows). `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals.
- `logit_bias` is not available on this engine path. The server refuses a request that has it (as it does with
  `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over. The MTP block's
  cache has holes for the tokens a slot decoded in batch windows, so the drafts of a later solo stretch are a little
  worse. The text is not affected.
- Intel Arc (SYCL) and Mac do not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed). Run them under
the box's GPU lock.

| Script | Checks |
| --- | --- |
| `q36_solo_gate.py` + `q36_cmp_gate.py` | The 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's. |
| `q36_batch_gate.py --slots N --waves --long 2500` | Every slot's tokens equal its solo tokens, slots reused, a long prompt read beside decoding slots. |
| `q36_proto_edge.py` | `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests, the context's end. |
| `q36_interleave_probe.py` | What the decoding slots see while a long prompt is read. |
| `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp) | The picture comparison against llama.cpp. |
| `q36_vision_proto.py` | Pictures through `GENI` / `BGENI` (solo, in a batch, the conversation cache). |
| `q36_vision_serve.py`, `q36_serve_par.py`, `q36_serve_handoff.py` | The same through the server. |
| `q36_par_bench.py` | The aggregate-throughput A/B against `llama-server -np N`. |

## History / older measurements

### Pictures checked against llama.cpp

> **Measured on:** R9700 32 GB (HIP), UD-Q4_K_XL, Strata 0.1.45. llama.cpp's `mtmd` and its own graph, the encoder on the
> CPU on both sides, the same prompt ids, llama.cpp's answer fed to both, logits compared step by step over 24 steps.
> The floor is llama.cpp's CPU backend against its own HIP run.

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request | 655 | differs (a near tie; the floor's KL at this step is 0.16) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's. With the picture options on, a request without pictures gives the same
tokens as without them (10 prompts x 128 greedy tokens).

### Aggregate decode speed, several requests at once

> **Measured on:** same prompts, greedy, 128 new tokens each, wall time from first request to last answer, 5
> interleaved pairs, medians, Strata 0.1.45 with `--batch 8 --mtp 0` against `llama-server -np N`.

| Card | Requests | Strata tok/s | llama.cpp | Ratio |
| --- | ---: | ---: | ---: | ---: |
| R9700 32 GB, UD-Q4_K_XL | 1 | 83.9 | 73.8 | 1.14 |
| | 2 | 173.5 | 107.4 | 1.61 |
| | 4 | 252.0 | 165.0 | 1.53 |
| | 8 | 325.4 | 208.1 | 1.56 |
| RTX 3060 12 GB, hybrid (llama.cpp `-ncmoe 28 -t 8`, the best split of a sweep) | 1 | 64.0 | 41.1 | 1.56 |
| | 2 | 85.1 | 52.5 | 1.62 |
| | 4 | 99.3 | 63.0 | 1.58 |

All pairs were faster for Strata. Every slot's tokens equal the tokens of the same request alone.

### Several requests with the base engine, and with `--batch-mtp`

> **Measured on:** R9700 32 GB, 5 interleaved pairs, Strata 0.1.45 (before the speed work below).

| Requests | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 |
| 2 | 173.1 | 207.0 | 1.20 |
| 4 | 251.0 | 275.0 | 1.10 |
| 8 | 324.0 | 320.5 | 0.99 (no row to spare: no proposals) |

> **Measured on:** hybrid cards, UD-Q4_K_XL, medians of 5 interleaved pairs, `--mtp 2`; "together" is the total tokens
> per second of N simultaneous requests against one request at a time with the draft layer.

| Card, context | Slots | One request alone, against 1 slot | N together, against one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26x to 1.64x together (3060 at 8K: 4 slots 1.64x for -5.5%
alone). A single request with drafts is already about 1.6x as fast as one without, so on a small card the main gain is
that a second user starts at once. On a card that holds every expert (R9700 32 GB, 8K prompts), 4 slots run 1.45x
one-at-a-time-with-drafts (1.59x with `--batch-mtp`).

### Prompt read beside decoding slots

> **Measured on:** R9700 32 GB, Strata 0.1.45, three decoding slots.

A 3,008-token prompt was read in 0.95 s (3,200 tok/s, 6 chunks). Each decoding slot paused at most 160 ms (one chunk)
and produced 20 tokens meanwhile. With `STRATA_Q36_READ_WINDOWS=4` they produced 65 tokens during a 1.14 s read.

### Speed work in 0.1.45

> **Measured on:** R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, 5 interleaved pairs, 3 repetitions each, medians; base = the
> engine before 0.1.45.

| | Base | 0.1.45 | Ratio |
| --- | ---: | ---: | ---: |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 tok/s | 183.7 | 1.10 |
| 1 / 2 / 4 / 8 requests (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 |
| Prompt pp4096 at chunk 512 / 1024 / 2048 / 4096 | | 3,453 / 4,786 / 5,766 / 6,223 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, chunk stays 512): decode 80.4 to 106.9 tok/s, pp4096
2,160 to 2,200 (llama.cpp 3,486 at `-ub 2048`, 2,738 at `-ub 512`), `--mtp 2` 154.7 to 167.3. `--chunk 2048` there moves
the experts to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94. That is an option,
not the default.

> **Measured on:** RTX 3060 12 GB (CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer
> in VRAM), Strata 0.1.45.

Decode tg128 58.5 to 59.7 tok/s (+2%), pp2048 1,958 to 2,022 (+3.3%), 10/10 prompts identical, plain and with `--mtp 2`.
llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48 to 52. The GPU part of a token is the dense
Q8_0 at the card's bandwidth and the rest waits for the CPU pool, so only the small kernels' share is gained.

KL against the 512 chunk on six real 1.5K to 3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3,
2048 4.2e-3, 4096 3.6e-3, top-1 agreement 44 to 48 of 48 rows. Greedy 64 tokens after those documents: the default chunk
gives the base engine's text for 2 of 6, `--prefill 512` for 6 of 6.
