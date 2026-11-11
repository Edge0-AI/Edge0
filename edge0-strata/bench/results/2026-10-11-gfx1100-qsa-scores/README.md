# Exact gfx1100 QSA scorer: measured prefill benefit, opt-in only

Measured on 2026-10-11 on one RX 7900 XTX. This accompanies the HIP port of the
four-lanes-per-query score kernel from [PR #1825](https://github.com/Niko1221/Strata/pull/1825)
by xrip, pinned to `14bf32150490bb9f5fc4d9c7c5dcfdc96dd98797`. That PR's code excludes
HIP; its algorithm and reduction order are the source of this port. This is an
AMD extension, not a claim to invent that method.

The measured benefit is fresh long-prompt processing. The correction-inclusive
coding pilot was 3.3% slower overall. The new path is disabled by default and was
not promoted into the tested machine's production setup.

## Dispatch and implementation

Set `STRATA_SCORES_THREAD=1` **before starting the engine**. Dispatch also requires:

- HIP on the exact `gfx1100` architecture (normal feature suffixes are accepted).
- At least 64 queries and a supplied live bound of at least 16,385 index blocks.
- A live bound no larger than the allocated score capacity. Allocating a 256K
  context alone does not select the new kernel.
- The already supported indexer geometry: four heads, dimension 128, block size 4.

Otherwise the existing scorer runs. Decode, top-k, weights, expert placement,
PLE storage, and sampling are not changed by this patch. CUDA does not compile
the HIP addition. No extra persistent weights or conversation state are allocated.

The kernel uses four wave32 lanes per query and keeps all four heads in registers.
Each key tile in LDS serves 64 queries. The original FP32 per-lane expression and
xor reduction tree are preserved; the existing tail scorer handles the final block.
The retained layout is 64 queries by 32 key blocks, 512 key blocks per workgroup,
256 threads. Four layout variants were screened; none improved stage time by the
predeclared 10% threshold over this one, so the first layout was retained.

`STRATA_SCORES_THREAD_TRACE=1` optionally prints one engagement line per host thread.
No trace output is enabled by default.

## Machine and shared settings

- RX 7900 XTX, 24 GiB nominal VRAM, gfx1100; PCIe 4.0 x16 observed; fixed 272 W cap.
- Ryzen 9 7950X3D; 64 GiB installed RAM, approximately 62 GiB reported by Linux.
- Nobara 44; kernel `7.2.9-200.nobara.fc44.x86_64`; ROCm SDK 10.0.0.
- [OrcaRouter Qwen3.8 Flash Next Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF),
  IQ3_S. Exact shard names, pack/profile hashes, and configuration are in
  [provenance.json](provenance.json). The Hugging Face revision was not recorded.
- One GPU, one server, 15 CPU expert workers; logical context 262,144; int8 KV,
  32,768 resident KV cells; prefill chunk 16,384; spec/MTP 6, floor 0.70, maximum T 6.
- Same balanced expert profile in every arm; 8,144 actual GPU expert slots,
  16,622,003,200 blob bytes; 34 GiB resident budget and 1,024 MiB conversation cache.
- NVMe model storage; PLE lookup shard on a dedicated ext4 SSD partition with
  direct I/O. The lookup was not moved into RAM. GPU vision was enabled but no
  image prompt was used in these measurements.
- Private experiment cgroup: 56 GiB memory cap, swap limit zero; at least 8 GiB
  available host memory required. This is not a claim that system-wide swap was disabled.
- Same local prefix-cache frontend and chat template for both arms. Its hashes
  are recorded. The stock latest Python frontend was not benchmarked here.

The repeated model tests used the reconstructed tuned 0.1.40.3 engine, including
PR #1107 prefix reuse and resident-budget loans in **both** arms. They compare the
installed control against its reconstructed candidate, not two official release
binaries. A separate comparison builds current main at
`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406` plus this patch and uses the **same binary**
with `STRATA_SCORES_THREAD=0` and `1`. Those families are never pooled.

## Before and after

The full per-request values, native prompt/decode timing, token counts, output
hashes, means, medians, and ranges are in [measurements.json](measurements.json).
These are **non-streaming request times with a one-token cap**. Model loading and
queueing were excluded; streaming time to first token and sustained generation
throughput were not measured by this screen.

Each fresh case starts a new private model worker. Actual reused prefix tokens
were zero. The OS file cache was not flushed, so these are fresh model prompts,
not cold-boot storage benchmarks. Inputs are repeated numerical source code,
greedy sampling, seed 6300, reasoning disabled. They are a prefill screen, not a
quality evaluation or representative project-completion benchmark.

| Engine family | Actual fresh tokens | Pairs | Before mean | After mean | Less elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: |
| tuned-v0.1.40.3 | 130,978 | 3 | 66.741 s | 59.769 s | 10.45% |
| tuned-v0.1.40.3 | 239,906 | 3 | 142.718 s | 110.407 s | 22.64% |
| current-main-same-binary | 130,978 | 1 | 68.872 s | 60.020 s | 12.85% |
| current-main-same-binary | 239,906 | 1 | 142.843 s | 111.828 s | 21.71% |

| Tuned build fresh tokens | Before median (range) | After median (range) |
| --- | ---: | ---: |
| 130,978 | 67.514 (64.257–68.451) s | 59.465 (58.863–60.979) s |
| 239,906 | 142.059 (141.660–144.436) s | 110.438 (109.549–111.232) s |

**Cached follow-ups (2,085 new tokens):**

| Engine family | Reused prefix | Pairs | Before mean | After mean | Less elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: |
| tuned-v0.1.40.3 | 130,978 | 2 | 2.829 s | 2.658 s | 6.023% |
| tuned-v0.1.40.3 | 239,906 | 2 | 3.418 s | 3.417 s | 0.035% |
| current-main-same-binary | 130,978 | 1 | 2.896 s | 2.781 s | 3.965% |
| current-main-same-binary | 239,906 | 1 | 3.522 s | 2.958 s | 16.037% |

The repeated near240K cached result is effectively flat (0.035%). The current-main
follow-ups are single pairs and do not establish a repeatable cached-prefix gain.

**Submitted CMake scorer microbenchmark:**

| Scorer-only live context | Queries | Before | After | Stage speedup |
| --- | ---: | ---: | ---: | ---: |
| 131,072 | 256 | 3.065300 ms | 0.742917 ms | 4.126x |
| 240,000 | 256 | 5.264949 ms | 1.389344 ms | 3.790x |

The repeated family used orders off/on, on/off, off/on. All three fresh request
pairs at each length had the same input hash, output hash, and output count.
Cached continuations in trials 2 and 3 added 2,085 actually processed tokens and
reused the preceding 130,978 or 239,906 tokens. Trial 1's much smaller continuation
is retained in the raw data but excluded from the 2K continuation summary.
The follow-up inserts a fixed synthetic assistant message; it does not use the
one generated token as a complete conversational answer. This isolates prefix
reuse and incremental prompt processing.

## Correctness and build checks

The submitted HIP CMake target passed 35 cases comparing against the unchanged
warp scorer: **223,664,777 score values bitwise equal**, and **11,895,800 ID buffer
entries equal**, including padding. Allocation guards passed. Cases cover
query counts 1/6/8/9/63/64/65/127/128/129/256/1024; the 65,535–65,539 boundary,
128K and 256K; capacity versus live extent; absent/negative/oversized bounds;
random and wide-exponent floats, ties, signed zero, NaN/Inf and cancellation.

The same compiled CMake test provides warmed, interleaved ABBA GPU-event timings
with four trials and ten calls per trial. See [correctness.json](correctness.json).
At the real 256-query batch size, this tests the scorer stage rather than making
a claim about whole-request or whole-task speed from a microbenchmark.

On the tuned model, the 4K/131K prefill plus first generated token matched the
control's main state byte for byte. Separate 64-token continuations and
STOP/resend output checks matched. Full internal state after 64 speculative tokens
was not always byte-identical; the original binary also produced differing
speculative states across repeats. This does not establish the cause of the coding
pilot's different repair paths. No claim of universal bitwise model output is made.

- Current-main full HIP engine and the standalone CMake test target built with
  ROCm 10.0.0, Release, gfx1100, native experts and optional prefill MMQ enabled.
- The touched `qsa_select.cu` also compiled with CUDA 13.0 for sm_89. There was no
  CUDA GPU test or full CUDA-engine build. No SYCL source was changed.
- Current-main off/on model requests completed with matching input/output hashes.
- Unsupported GPU architectures were not runtime-tested on hardware.

Build and run the standalone check even from the published tree without `tests/`:

```bash
cmake -S . -B build-qsa -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PARITY_GFX1100_QSA_SCORES=ON
cmake --build build-qsa --target qsa_scores_gfx1100_parity -j 6
ctest --test-dir build-qsa --output-on-failure -R '^qsa_scores_gfx1100_parity$'
```

Supply your normal ROCm compiler/package paths if they are not discoverable.
CTest sets the opt-in and engagement-trace variables and skips non-gfx1100 cards
with return code 77.

## Coding pilot and limitations

Two official Exercism/Aider Polyglot tasks, Go alphametics and Rust xorcism, were
embedded in a 196,608-token source archive, leaving 65,536 tokens within the 256K
window for tool results and repairs. xhigh, 8,192 reasoning tokens, 2,048 answer
reserve, temperature 0.3, top_p 0.95, top_k 20, min_p 0; one matched trial per task,
seeds 7830/7831. Tools and full-suite checks ran until the first pass, with no
post-pass requests. This is a long-context stress pilot, **not an official Aider
leaderboard score**. Its test and correction time is included.

| Task | Original | New scorer | Original/new turns | Result |
| --- | ---: | ---: | ---: | --- |
| go/alphametics | 209.136 s | 400.318 s | 3 / 5 | Both full suites passed |
| rust/xorcism | 473.734 s | 304.906 s | 6 / 6 | Both full suites passed |
| Total | 682.870 s | 705.225 s | | New scorer **3.27% slower** |

The different Go run generated more tokens and took extra repair turns despite
matching normalized inputs through the first three requests. The cause remains
unresolved. The combined pilot did not meet the predeclared promotion gate. More
coding repeats, the held-out extension, and the full 24-task suite were therefore
not run. No Luna, frontier-model, or overall agent-quality superiority is claimed.

An earlier 240K-archive pilot ran out of continuation space: the Rust history
reached 262,714 tokens, beyond 262,144. Its partial attempt is listed separately
in the data and was not reused in the 192K score. It was a benchmark-protocol
failure, not evidence of a scorer crash.

## Reproducing the request screen

[benchmark.py](benchmark.py) generates the exact frozen fresh/continuation request
hashes used here from the included compressed numerical-source fixture. It reads
the model pack's tokenizer and uses only Python plus the existing tokenizer's
`regex` dependency. The supplied [hipBLASLt table](hipblaslt-gfx1100-100401.txt) is
specific to this SDK/device pair; do not assume its solution IDs transfer to another
version. The balanced expert profile is identified by hash, not included; use the
same profile and other settings in your two arms.

Run a loaded private server with the same binary and settings, changing only
`STRATA_SCORES_THREAD=0` or `1` **at startup**. Restart that worker between every
length/arm invocation, preserve every observation, and repeat in both orders:

```bash
python bench/results/2026-10-11-gfx1100-qsa-scores/benchmark.py \
  --url http://127.0.0.1:8080 --tokenizer /path/to/pack/tokenizer \
  --model strata-qwen38-iq3s-uncensored --length 131072 \
  --variant off --output off-128k-trial1.json
```

Then restart with the scorer enabled and use `--variant on` and a new output file.
Repeat with `--length 240000`, then reverse arm order. Use `--prepare-only` to
check request hashes without inference. `STRATA_API_KEY` can supply an API key;
it is never written to results. The script refuses to overwrite any attempt and
records a failure if a fresh prompt reused cached tokens.

The label does not control the server's environment: confirm the startup setting
and, in the enabled arm, the engagement trace. Measure your model's full output
and correction-inclusive workload separately before adopting the feature.
