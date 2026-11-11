# Community report: Radeon 8060S (gfx1151), Ryzen AI Max+ 395

Measured on 2026-10-04 and 2026-10-05 on the same Radeon 8060S / Ryzen AI Max+ 395 system with 128 GB of unified
memory. The first session compared Strata HIP with llama.cpp Vulkan in interactive Q2_0 and UD-IQ4_XS chats. The
second used matched synthetic prompts to isolate the gfx1151 hipBLASLt tuning table, dense MMQ and shared-expert
decode stream. The controlled run's main result is **250.91 to 528.91 prompt tok/s (2.11x)** from the tuning table;
the shared-expert stream raised matched decode from 45.63 to 46.77 tok/s (2.5%).

The interactive answers sometimes differed in reasoning and length, so their long answers and session aggregates
are observations rather than kernel-only comparisons. The controlled A-E runs use fixed prompts and sampling; raw
request records and the detailed calculation are in [prefill-ab.md](prefill-ab.md).

## Hardware and software

- AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5), 16 cores / 32 threads and AVX-512.
- 128 GB LPDDR5X unified memory; the GPU reported a 116.0 GiB shared GTT pool. This is not independent VRAM.
- CachyOS Linux, kernel 7.2.9; the kernel's amdgpu driver.
- Strata 0.1.39, built locally for `gfx1151` with HIP runtime `7.17.26392-0000000` and hipBLASLt 1.5.0. The
  interactive session used the initial enablement build; [BUILD-interactive.json](BUILD-interactive.json) records
  its generated build metadata.
- The controlled session used source revision `a68d0b7fd88c206d55c2ca1d3c2811300b07bd14`: PR #895's gfx1151
  enablement, PR #820's opt-in dense MMQ, and the gfx1151 tuning table adapted from PR #755. Its manually recorded
  provenance is [BUILD-controlled.json](BUILD-controlled.json).

## Configurations

The exact generated server configurations are [strata-q2_0.json](strata-q2_0.json) and
[strata-unsloth-ud-iq4_xs.json](strata-unsloth-ud-iq4_xs.json). They preserve the measured directory layout as an
audit record, but replace the account name with `$USER`. JSON does not expand that placeholder, so these files are
not portable templates. The Q2_0 file was saved after its vision limit was raised from the measured 300 tokens to
512. That setting does not affect the text-only observations.

| | Strata | llama.cpp |
|---|---|---|
| GPU backend | HIP, `gfx1151` | Vulkan |
| target layers | all | all (`-ngl 99`, `-ngld 99`) |
| parallel requests | 1 | 1 (`-np 1`) |
| MTP | `--spec 4 --spec-min-p 0.5`; prompt lookup can extend the verification window to 6 | Q8_0 MTP, 1-3 drafts, probability floor 0.0 |
| KV/context, Q2_0 | int8, 32K resident of 131K | 65K context; llama.cpp KV setting not recorded |
| KV/context, UD-IQ4_XS | int8, 32K resident of 65K | 65K context; llama.cpp KV setting not recorded |
| Strata version | engine 0.1.39 | - |

Both interactive Strata runs cached all 24,576 routed experts in the shared GPU pool: 31.64 GiB for Q2_0 and 55.43
GiB for UD-IQ4_XS. The latter therefore did not read experts from SSD while answering, despite using the model
shards in place. These interactive runs predate the gfx1151 tuning table and used plain hipBLAS for dense prompt
projections. The controlled Q2_0 run below measures that table separately.

## Q2_0

Model: [Qwen3.8-Flash-Next-GSQ-RCO-Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)

### Long matching prompts

| Prompt | Strata HIP | llama.cpp Vulkan | Strata difference |
|---|---:|---:|---:|
| Long reasoning answer | 1,921 tokens at 47.8 tok/s | 1,545 tokens at 35.23 tok/s | +35.7% |
| Following long answer | 3,401 tokens at 46.3 tok/s | 3,359 tokens at 37.74 tok/s | +22.7% |

The second row is the cleaner comparison because the generated lengths are nearly equal. The first few answers had
different reasoning behavior and were short enough for draft acceptance and startup effects to dominate.

### Whole observed sessions

| Backend | Requests | Output tokens | Decode time | Weighted decode rate | MTP drafts accepted |
|---|---:|---:|---:|---:|---:|
| Strata HIP | 5 | 5,552 | 118.236 s | 46.96 tok/s | 3,169 / 4,684 (67.7%) |
| llama.cpp Vulkan | 6 | 5,540 | 147.197 s | 37.64 tok/s | 3,320 / 6,678 (49.7%) |

The session totals happen to contain almost the same number of output tokens, but llama.cpp has one extra short
request and the generated text is not identical. Treat the resulting 24.8% Strata lead as an observed workload
result, not a kernel-only benchmark.

## Unsloth UD-IQ4_XS

Model: [Qwen3.8-Flash-Next-UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), three shards. Both servers used a 65,536-token context.

| Prompt order | Strata HIP | llama.cpp Vulkan | Strata difference |
|---:|---:|---:|---:|
| 1 | 52 tokens at 44.3 tok/s | 93 tokens at 38.87 tok/s | +14.0% |
| 2 | 266 tokens at 42.2 tok/s | 209 tokens at 34.24 tok/s | +23.2% |
| 3 | 188 tokens at 41.6 tok/s | 378 tokens at 40.87 tok/s | +1.8% |
| 4 | 959 tokens at 34.6 tok/s | 891 tokens at 35.44 tok/s | -2.4% |
| 5 | 2,205 tokens at 36.1 tok/s | 1,438 tokens at 36.02 tok/s | +0.2% |

The two sustained runs are effectively tied at about 35-36 tok/s. Across the complete observed sessions:

| Backend | Output tokens | Decode time | Weighted decode rate | MTP drafts accepted |
|---|---:|---:|---:|---:|
| Strata HIP | 3,670 | 100.784 s | 36.41 tok/s | 2,126 / 3,283 (64.8%) |
| llama.cpp Vulkan | 3,009 | 82.673 s | 36.40 tok/s | 1,912 / 3,300 (57.9%) |

Strata's first prompt read 63 tokens at 75.5 tok/s. Its cached follow-ups read 18-32 new tokens at 63.8-70.7 tok/s.
llama.cpp's first request reported 381 prompt tokens at 215.6 tok/s; its cached follow-ups read 19-30 new tokens at
45.3-67.7 tok/s. The first prompt counts differ too much for a direct prompt-speed comparison.

## Vision

Both Strata models used the CPU vision encoder with 16 threads and a limit of 300 image tokens. The projectors from
the ISTA Q2_0 folder and the Unsloth download have different metadata but byte-identical tensor payloads; encoding
the same image produced byte-identical embeddings. The direct encoder test made 299 tokens on a 23 x 13 grid in
2.43-2.45 seconds.

The same `docs/media/pagoda-preview.webp` image and the prompt `Describe this image concisely.` produced:

| Model | Prompt | Prompt processing | Output | Decode | MTP drafts accepted |
|---|---:|---:|---:|---:|---:|
| Q2_0 | 320 tokens in 2.011 s | 159.2 tok/s | 53 tokens in 1.035 s | 51.2 tok/s | 35 / 38 (92.1%) |
| UD-IQ4_XS | 320 tokens in 2.217 s | 144.3 tok/s | 97 tokens in 3.053 s | 31.8 tok/s | 52 / 104 (50.0%) |

Both answers correctly identified the voxel pagoda, island, cherry blossoms, pond, torii gate and control panel.
UD-IQ4_XS gave more detail; Q2_0 followed "concisely" more literally. This single decode-speed difference is not a
quantization-only result: the answers have different lengths and Q2_0 accepted an unusually high 92.1% of its MTP
drafts, versus 50.0% for UD-IQ4_XS. The API's prompt and decode clocks do not include the separate CPU image-encoder
step.

A second Q2_0 request used a different, larger image and asked `please describe the image and print out any text you
can read`. Reasoning was enabled. It read 355 prompt tokens in 1.903 seconds (186.6 tok/s), generated 577 tokens in
14.062 seconds (41.0 tok/s), and accepted 260 of 424 MTP drafts (61.3%). The server's visible prompt-plus-decode time
was about 16 seconds; the separate CPU image-encoding step is not included. Because the encoder was still capped at
300 image tokens, the larger source resolution did not give the language model a proportionally larger visual grid.

## Where the MTP effect appears

Strata records MTP in three places:

- Startup says `strata mtp: draft layer loaded` and `strata verify: window up to 6 tokens`.
- Every completed request in `strata-<model>.log` says `drafts accepted A of B`.
- The OpenAI response's `timings` object exposes the same counters as `draft_n_accepted` and `draft_n`.

An accepted draft lets the target model commit another output token from the same verification window. Higher
acceptance normally raises throughput. For example, the two long UD-IQ4_XS Strata requests accepted 521/881
(59.1%) and 1,283/1,966 (65.3%) drafts and ran at 34.6 and 36.1 tok/s respectively. Content and context also changed,
so this is consistent with an MTP gain but does not isolate it.

The acceptance counters are utilization figures, not a direct speedup measurement. Quantifying the MTP effect
requires a controlled A/B with identical input tokens and output tokens: one normal speculative run and one plain
decode run without the MTP layer. The interactive Strata server currently requires MTP, so that control should use
the engine's non-server generation/benchmark path rather than comparing unrelated server replies.

## Controlled Q2_0 A/B

`tools/hip/bench_prefill.py` performed one excluded warm-up and four fresh/follow-up pairs per server start. Fresh
prompts contained 140, 280, 140 and 280 generated JavaScript functions. Sampling was fixed at temperature 0, top-k
1, top-p 1, min-p 0 and seed 42, reasoning was disabled, and every measured request generated 128 tokens. Every
fresh request had zero reused KV tokens. The server was restarted between arms.

| Arm | Dense prompt path | Dense MMQ | SH stream | Weighted fresh prompt rate |
|---|---|---|---|---:|
| A | plain hipBLAS | off | default on | 250.91 tok/s |
| B | tuned hipBLASLt | off | default on | 528.91 tok/s |
| C | tuned hipBLASLt | on | default on | 529.96 tok/s |
| D | tuned hipBLASLt | on | explicitly on | 532.43 tok/s |
| E | tuned hipBLASLt | on | off | 532.58 tok/s |

The tuning table improved weighted fresh-prompt throughput by **110.8%**. Its four matched request improvements
were 109-113%. Dense MMQ changed the weighted B result by +0.2%, below this session's observed timing variation.
SH streaming is a decode optimization and did not materially affect prefill.

D and E generated identical responses with the same MTP draft behavior, making them the clean SH-stream pair. D
generated 1,024 tokens in 21.894 seconds (46.77 tok/s); E needed 22.441 seconds (45.63 tok/s). Enabling the stream
improved weighted decode by **2.5%**, with all eight individual requests faster by 1.3-3.5%.

The detailed per-request table, method, result links and limits are in [prefill-ab.md](prefill-ab.md). This is one
ordered session on one machine, not a confidence interval or a result for other GPUs and models. Cached follow-ups
whose generated prefixes differ are retained as evidence but are not used for the hipBLASLt prefill comparison.

## Notes

- Decode rates above use each engine's own decode clock, excluding prompt processing and model loading.
- Short responses vary substantially with MTP acceptance; the sustained runs are more representative.
- Strata's current UD-IQ4_XS compatibility pack converts 195 Q8_0 hyper-connection tensors to BF16. Speed is not a
  quality or fidelity measurement.
