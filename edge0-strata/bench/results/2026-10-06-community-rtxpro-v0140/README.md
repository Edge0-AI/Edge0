# Community results: Strata 0.1.39 versus 0.1.40 on RTX PRO 6000

Measured on **2026-10-06** with unmodified release engines, Unsloth Flash-Next
**UD-Q4_K_XL / Q8_0**, FP16 KV, **8,192 actual input tokens + 512 output tokens**,
and one request at a time. This contribution adds documentation and measurement
artifacts to the exact v0.1.40 source tree.

## Native 0.1.40: ngram follow-up

![Four decoding paths on native Strata 0.1.40](ngram.png)

The same 8K coding prompt and 512-output cap were run with ngram-only and with
MTP plus ngram. No engine changes were made.

| Model / path | Decode tok/s | Prefill s | Whole request s | Effective tok/s |
|---|---:|---:|---:|---:|
| Q4_K_XL, no speculation | 129.9 | 3.115 | 7.057 | 72.6 |
| Q4_K_XL, MTP T4 | 257.6 | 3.035 | 5.024 | 101.9 |
| Q4_K_XL, ngram only | 133.8 | 2.990 | 6.817 | 75.1 |
| Q4_K_XL, MTP + ngram | 255.4 | 3.017 | 5.023 | 102.0 |
| Q8_0, no speculation | 82.6 | 12.790 | 18.989 | 27.0 |
| Q8_0, MTP T4 | 140.2 | 12.812 | 16.465 | 31.1 |
| Q8_0, ngram only | 81.6 | 12.782 | 19.056 | 26.9 |
| Q8_0, MTP + ngram | 133.8 | 12.761 | 16.589 | 30.9 |

Ngram-only removes `--mtp`, enables `--suffix-draft 3`, and allows the verifier
up to T8 (`--spec 8`). Combined mode keeps MTP capped at T4 and enables
`--suffix-draft 3 --lookup-chain 4 --lookup-chain-min 3`: lookup can replace a
window or append up to four drafts. This tests one recorded configuration,
not a threshold sweep or a claim about every ngram workload.

| Model / path | Suffix drafts accepted / offered | Chained drafts accepted / offered | Exact tokens versus corresponding non-ngram path |
|---|---:|---:|---|
| Q4, ngram only | 33 / 109 | Not enabled | All 512 identical |
| Q4, MTP + ngram | 3 / 14 | 4 / 8 | All 512 identical |
| Q8, ngram only | 6 / 36 | Not enabled | First difference at output token 4 |
| Q8, MTP + ngram | 22 / 27 | 0 / 3 | First difference at output token 4 |

All four follow-ups completed and passed the small function check. Input hashes
matched the corresponding baseline. Counts above are for the timed request,
excluding warm-up. Q4 ngram-only decode was 3.0% faster than plain decode;
combined Q4 decode was 0.8% slower than MTP, with essentially unchanged total
request time. Q8 was 1.2% slower with ngram-only and 4.6% slower with the combined
configuration. Q8 committed output streams differed, so these are different
trajectories at fixed token counts, not exact-work speedup comparisons. The
cause of that difference has not been isolated. This coding-generation prompt
is not a code-edit/copy workload designed to favor prompt lookup.

## Initial release comparison

![Streaming decode and total request time for the two releases](overview.png)

**Q4 MTP decode increased from 244.2 to 257.6 tok/s (+5.5%). Q4 without MTP
increased from 118.5 to 129.9 tok/s (+9.6%).** Both Q4 comparison pairs produced
identical 512-token outputs. These are single observations, not confidence-bounded
speedup estimates.

**Q8 is a support result:** stock v0.1.39 rejected its Q8 PLE table at startup;
v0.1.40 loaded it and completed both modes. There is no stock v0.1.39 Q8 speed
number and no patched-engine substitute in this comparison.

## Results

| Model / mode | Release | Decode tok/s | Prefill s | Whole request s | Effective tok/s |
|---|---|---:|---:|---:|---:|
| Q4_K_XL, MTP T4 | 0.1.39 | 244.2 | 3.023 | 5.121 | 100.0 |
| Q4_K_XL, MTP T4 | 0.1.40 | **257.6** | 3.035 | **5.024** | **101.9** |
| Q4_K_XL, no MTP | 0.1.39 | 118.5 | 3.017 | 7.340 | 69.8 |
| Q4_K_XL, no MTP | 0.1.40 | **129.9** | 3.115 | **7.057** | **72.6** |
| Q8_0, MTP T4 | 0.1.39 | Unsupported Q8 PLE | â€” | â€” | â€” |
| Q8_0, MTP T4 | 0.1.40 | **140.2** | 12.812 | 16.465 | 31.1 |
| Q8_0, no MTP | 0.1.39 | Unsupported Q8 PLE | â€” | â€” | â€” |
| Q8_0, no MTP | 0.1.40 | **82.6** | 12.790 | 18.989 | 27.0 |

Decode is committed output count / engine decode time. Effective throughput is
output count / (engine prefill + decode time). Whole request is client wall time;
it excludes model loading. All six supported cases freshly read 8,192 input
tokens, reused zero prompt tokens, and reached the 512-output cap.

Stock v0.1.39's two Q8 startup checks returned:

```text
per_layer_token_embd.weight is Q8_0, not IQ4_NL, Q5_0 or FP8 (I8)
```

## Hardware and software

- NVIDIA RTX PRO 6000 Blackwell **Workstation Edition**, 96 GB, 400 W power limit.
- AMD Ryzen 9 7950X, 16 cores / 32 threads; 128 GB installed RAM.
- WD_BLACK SN8100 2 TB NVMe. PCIe maximum Gen4 x16; startup H2D probes were about
  28.9 GB/s. Idle link queries downshifted to Gen1 x16.
- Ubuntu 24.04.5 LTS; NVIDIA driver 595.91.07; CUDA 13.2.86.
- v0.1.39: `a1641e9f77aacad4d201b53c8a7ae8fa21059ebb`.
- v0.1.40: `1cbcacbcae2953f3be9edc46369f0c875bc6ab8b`.
- Both source worktrees were clean. Both builds used Release, CUDA architecture
  120, native experts enabled, tests disabled, and GGML
  `3cf03257f219afbe7334045ff7c6a06ac68c627d` as pinned by these releases.

## Placement and request settings

The models are from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
four `Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf` shards or six
`Qwen3.8-Flash-Next-Q8_0-*.gguf` shards. Existing compatibility packs were reused:
small unsupported projections were converted to BF16. Their hashes and the
expert-profile hash are recorded in [measurements.json](measurements.json).
The native model files and each quant's pack were unchanged across versions.
This is not a test of freshly generated installer presets. Model repository
revision and whole-GGUF hashes were not captured for this screen.

| Setting | Q4_K_XL | Q8_0 |
|---|---|---|
| GPU expert slots | All 24,576 | 15,472 |
| GPU expert cache | 71.73 GiB | 75.25 GiB |
| Expert complement | None | About 44.28 GiB in RAM; 56 GiB budget |
| PLE placement | RAM | RAM, about 50.66 GiB |
| Adaptation | Disabled | Every 4 steps, 96 swaps, decay 0.7 |
| New release adaptation flags | Not used | `--adapt-async 1`, `STRATA_EXCHANGE_ROTATE=1` |

Both modes allocated **16,384 tokens**, used FP16 KV, temperature 0, thinking off,
prefill chunk 1,024, a 2,048 MiB VRAM reserve, and the same expert profile. Prompt
and conversation caches were disabled. Ngram/suffix drafting was off in the initial release-comparison cases. MTP used
the same draft runtime, threshold 0.5, and maximum T4; the non-MTP runs used T1
and offered zero draft tokens. EOS was overridden to force the output cap.
This is a whole-release comparison with recorded configuration, not an isolated
test of PLE, adaptation, or any particular kernel.

## Function and token checks

Each supported case first generated a small `triangular(n)` Python function.
It passed checks for `n = 0, 1, 2, 10, 100, 10000`. This also warmed the engine
before the timed request. The timed prompt requested a priority task queue and
tests, padded with ordinary maintenance notes to exactly 8,192 tokens.

- All **six supported cases** passed the function check and exact input/output
  count checks; Q8 on 0.1.39 failed before generation.
- Both Q4 pairs had identical input hashes and all 512 output token IDs matched.
- Q4 MTP accepted 346 / 424 offered drafts in each release. Non-MTP offered zero.
- Q8 on 0.1.40 accepted 358 / 414 drafts in MTP mode; non-MTP offered zero.
- The long generated module was not executed or scored. These checks do not
  establish model intelligence, long-context correctness, or general numerical
  equivalence. An EOS-forced stream is not representative of normal stopping.

## Reproduction and data

[results.csv](results.csv) contains the chart data.
[measurements.json](measurements.json) includes timings, committed token IDs,
binary hashes, launch arguments, relevant environment, and startup errors.
[plot.py](plot.py) regenerates the PNG and SVG with `python plot.py`
(requires matplotlib). [plot-ngram.py](plot-ngram.py) regenerates the ngram chart with
`python plot-ngram.py`. Large historical experiments and model files are not
included in this documentation contribution.

Build each exact tag separately using:

```sh
cmake -S SOURCE -B BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=/path/to/ggml-3cf03257
cmake --build BUILD --parallel 8 --target strata
```

The request harness used one frozen tokenizer/template and native-protocol
Python client for both binaries, from fork commit
[`70ed61f`](https://github.com/CC-David-CC/Strata-a5500/tree/70ed61f158f0ef6364557cc1ba8d1ab03178d2f1).
That client revision is not an engine comparator. HTTP serving was not tested.

To repeat the workload, copy [plan.example.json](plan.example.json) to
`plan.json`, edit the binary, model, pack, client, Python, profile and draft paths,
and run `python replay.py CASE_LABEL`. Labels are in the plan. `$HOME` paths are
expanded. Use a free GPU and sufficient locked-memory limits for `--ple-io ram`.
Each case creates a fresh result directory and refuses to overwrite one.
Q8 cases for the stock 0.1.39 binary are expected to fail at PLE loading.

## Limits of this screen

One observation per case, no repetitions or uncertainty estimate. The release
0.1.40 cases ran first; stock 0.1.39 ran later, in Q8 MTP, Q8 non-MTP, Q4 MTP,
Q4 non-MTP order. Startup page-cache state, clocks and thermals were not matched.
Loading time is excluded and no startup-speedup claim is made. There were no
other compute jobs on the GPU. The GPU was idle after testing. Higher contexts,
concurrency, quality benchmarks and sanitizer coverage are outside this screen.
The four ngram follow-ups ran afterward, in Q4 ngram, Q4 combined, Q8 ngram,
Q8 combined order, each with a fresh process and its own function-check warm-up.
