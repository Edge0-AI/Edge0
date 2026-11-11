# QSA query LDS swizzle qualification

8 October 2026

## Conclusion

The opt-in gfx906 query-LDS swizzle improves the measured INT8 batched-QSA component by **14.646806%** on the final clean upstream patch. It preserves every output byte in the tested component cases. One isolated full-model comparison at 65,536 input tokens and 1,024 generated tokens measures **1.185195% higher prompt throughput** and **1.048564% lower request latency**, with all generated token IDs equal.

These results support a narrowly scoped, default-off upstream proposal. They do not establish a general speedup across workloads, model-level equivalence beyond the tested pairs, or production readiness.

The final hardened combined engine also passes a 200,000-token capacity run and targeted API/cache regression checks. Those qualifications apply to the frozen experimental stack; they do not extend the clean upstream patch's component-only validation to full-model testing.

## Change and dispatch scope

The change permutes only the shared-memory query layout in `attn_chunk_kernel`: `q(d) = d XOR ((d >> 3) & 4)`. The scoring loads invert the permutation. The mapping is bijective and self-inverse over the 256 dimensions, preserves 16-byte load alignment and the 12 KiB query tile, and leaves the logical floating-point operands and operation order unchanged. Score reduction, softmax, value accumulation and merge code are unchanged.

Enable with `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1`; the default is off. The runtime guard requires the calling thread's current device to be exactly gfx906, allowing normal architecture-feature suffixes, with a 64-lane wavefront. It checks the current device on each eligible dispatch, caches properties per thread and last device ordinal, and fails closed on HIP errors.

The opt-in applies to INT8 pools in `qsa_decode_attn_batch`, including PREFILL, VERIFY and MTP calls with `n_q=1`. A main model using k8v4 can still reach this path through its INT8 MTP drafter. `qsa_decode_attn_step`, other pool formats, lane-cell scoring and other GPU/default paths retain their existing dispatch.

## Component evidence

The final parity matrix covers both gfx906 GPUs, both the hardened experimental build and the clean upstream build, both flag states, and these three geometries:

| Queries | Context | Page masking | Output bytes per run |
|---:|---:|---|---:|
| 1 | 4 | None | 24,576 |
| 33 | 4,096 | Masked pages | 811,008 |
| 32 | 65,536 | None | 786,432 |

All **24 runs**, representing 12 on/off comparisons across the four build/device combinations, match raw output bytes for their geometry. The probe uses deterministic inputs, shuffled physical pages, finite/output-write checks and output canaries. These checks cover the tested inputs; they are not a general memory-safety proof. The source-extracted host guard test passes **31 mock-HIP checks**, including architecture rejection, device changes, API failures, recovery and independent thread caches.

The final clean-build timing screen uses GPU 0, `n_q=32`, context 65,536, unmasked pages and separate A/B/B/A processes with 50 repetitions per run. A is flag off; B is flag on. Each value below is a run's measured median for the batched attention component, including chunk and merge.

| Run | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Median ms | 1.188960 | 1.036240 | 1.036560 | 1.187439 |

Throughput gain is `(mean(A medians) / mean(B medians) - 1) × 100 = 14.646806%`. This is an A/B/B/A component screen, not a model benchmark or a confidence interval.

The initial experimental component screen measured 14.791570%, below the predeclared 20% gate. That gate did **not** pass. An explicit cost-benefit exception allowed one bounded 64K/1,024-output model comparison because the layout change was small and the measured component benefit was stable. The final clean-build screen also remains below 20%.

## Compiled kernel audit

The compiled baseline and swizzled kernels retain 24 `ds_read_b128` query reads each. The resource comparison is:

| Property | Baseline | Swizzled |
|---|---:|---:|
| Encoded static instructions | 1,000 | 1,005 |
| VGPRs | 41 | 42 |
| SGPRs | 42 | 42 |
| Total LDS bytes | 15,872 | 15,872 |
| Reported occupancy, waves per SIMD | 4 | 4 |
| Scratch bytes | 0 | 0 |
| SGPR / VGPR spills | 0 / 0 | 0 / 0 |

An earlier instruction-count script included an ellipsis line and reported 1,001/1,006; the encoded-instruction counts above correct that error. The bank-layout analysis motivates the change, but measured speedup is not a direct hardware bank-conflict-counter measurement.

Guard hardening preserves the entire disassembled device text, all 10 function symbols, addresses and machine encodings, and the emitted AMDGPU resource notes relative to the qualified experimental kernel. Only the input-object pathname was removed for comparison. HSACO file hashes differ; the differing bytes outside those inspected views have not been located, so no precise cause is claimed. This audit compares pre-hardening and hardened experimental builds; it does not assert whole-binary identity with rebased upstream main.

## Isolated QSA full-model result

Both arms use the same experimental HC + grouped-MMQ + PR #1525 dequantization stack; only the QSA query-swizzle flag changes. Fresh native processes use INT8 KV, no prompt reuse, a 65,536-token code prompt and 1,024 output tokens. Sampling is temperature 0, top-p 1, top-k 1, seed 12345. The candidate was run before the control.

| Metric | QSA off | QSA on | Change |
|---|---:|---:|---:|
| Prompt throughput, tokens/s | 617.712871 | 625.033976 | +1.185195% |
| Native prompt time, s | 106.0946 | 104.8519 | −1.171313% |
| Request wall time, s | 126.382110 | 125.056913 | −1.048564% |
| Generation throughput, tokens/s | 50.537205 | 50.748591 | +0.418279% |

All 1,024 token IDs match; speculative accepted/offered counts also match at 648/872. This is one matched pair, recorded before final host-guard hardening. The later ISA audit establishes unchanged device code across that hardening; the fresh combined-stack tests below exercise the final hardened engine. The small generation-throughput movement is noise-scale and does not establish a generation improvement.

## Combined stack result against the stable baseline

These fresh comparisons measure the complete HC + grouped-MMQ + PR #1525 dequantization + QSA candidate against the stable engine. They cannot attribute the combined gain to QSA alone. The component and isolated/full-stack percentages must not be added.

| Code prompt | Stable PP, tokens/s | Combined PP, tokens/s | PP change | Request latency change | TG change |
|---|---:|---:|---:|---:|---:|
| 4,096 tokens | 347.622402 | 368.640369 | +6.046206% | −2.343515% | +0.188068% |
| 65,536 tokens | 597.968207 | 624.674728 | +4.466211% | −3.664980% | +0.376379% |

Each context has one matched pair with 1,024 output tokens and all token IDs equal within the pair. These use the final hardened candidate engine. The 4K pair ran control then candidate; the 64K pair ran candidate then control. Generation changes are noise-scale. The stable service was restored after the tests; the candidate was not promoted to production.

For all model tables, PP is the reported full prompt-token count divided by native prompt time; the last prompt token executes in the first decode window. TG is actual output count divided by native decode time. Request wall time is measured separately from model loading. No cross-quantization quality or equal-token claim is made.

## Capacity and service qualification

The final hardened engine (`3d2650fe62fdea24679ac90401eac505b541ccc47886e59885a639c438cb4c17`) completed a fresh **200,000-token prompt plus 256 generated tokens** with a configured capacity of 204,800 and INT8 KV. All 256 output IDs match the archived stable `ce794788…` control. This is a capacity and output-parity qualification against an archived control, **not a fresh speed comparison**. No incremental QSA or combined-stack speedup is inferred from this run.

Targeted API testing verified the actual candidate executable hash and active candidate flags, including the QSA opt-in, then passed:

- Synthetic forced tool call and tool-result roundtrip
- Two synthetic image requests through CPU vision, followed by text
- Unauthenticated-request HTTP 401 and root-route HTTP 404 checks
- Synthetic A/auxiliary/A RAM prompt-cache restoration

| Cache request | Elapsed time, s | Cached prompt tokens |
|---|---:|---:|
| A, cold | 14.3541 | 0 |
| A, warm | 0.4073 | 6,423 |
| A, restored after auxiliary request | 0.3868 | 6,423 |

The three A requests returned the same result. This synthetic cache regression is not a replay of a client session or a general service benchmark. The original configuration and Compose files were restored byte-for-byte, and the stable `ce794788…` API was healthy afterward. No production promotion occurred.

## Provenance and limits

- Hardware scope: two gfx906 wave64 GPUs; the runtime reports AMD Radeon Pro VII. Results are specific to the tested hardware and pinned build stack.
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0`, engine version 0.1.40, INT8 KV. The configured capacity is 204,800 tokens; the final candidate completed the 200,000-token qualification described above.
- Experimental source HEAD: `f152172332693c933b8a3104ce193d8a4315c2b4`, plus campaign changes. It is a separate source scope from the clean PR.
- Pinned llama.cpp revision: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Clean upstream base: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
- Final clean patch SHA-256: `9c81295b8cdf8354fe0e5bc6de36a547199becd1a8fc2a79daabf2b9da9548c6`.
- Final clean probe SHA-256: `a5723bfd34720b690a4031a978ecdfd8c65a35a26cf4ba0471fa4e969570882d`.
- Pre-hardening model-engine SHA-256 used for the isolated QSA pair: `1d9fb9de4e5f5d63fd30257e4077caafa46dde61b5c4d4ac045a3a0f63b78d42`.
- Final hardened model-engine SHA-256: `3d2650fe62fdea24679ac90401eac505b541ccc47886e59885a639c438cb4c17`.
- Stable model-engine SHA-256: `ce794788d64313236bbc24b47a3f8e06c841034afaf4e19a2fc5ffd84e337f14`.
- Build image digest: `sha256:bccb7ee7e7a78274519db9a43ba63c34ddd2e74bb60f8764a8f50aaee1f2c646`. The clean probe uses Release, HIP gfx906, portable/native-expert settings, and grouped-MMQ build options disabled. No floating-point compiler-policy change is part of the QSA patch.
- Compiler verified in that image: AMD Clang `23.0.0git`, ROCm llvm-project revision `46fcb339fb61119b337f973c7ca9e710a319fdd0`, patched revision `440716f8b87be9d8e20ed910e10e5b6d14d57cf6`. The ROCm distribution version is unrecorded because its version file is empty.

The clean rebased patch has component and host-guard validation; the full-model comparisons, 200K capacity run and API/cache qualification belong to the frozen experimental stack. No full-model run on rebased main, upstream CI pass or production promotion is claimed. Single matched speed pairs and targeted synthetic service tests do not establish broader workload or service equivalence.

The overlap review found no direct duplicate, but upstream PR #1402 introduces a third kernel-template argument `G`, conflicting with this patch's third boolean argument. The two changes require coordination before merging. PR #1535 concerns the separate top-k work.

The branch at commit `26f776a` is published. Creating the new PR returned HTTP 403, so no new PR exists; the accompanying PR text remains a draft.

Evidence reviewed: final component/model/combined receipts, build receipts, the final clean patch, source/ISA and hardened-ISA audit records, and the sanitized capacity/API/cache qualification receipt.
