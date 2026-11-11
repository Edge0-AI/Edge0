# Threadripper 3990X / RX 6900 XT: Windows processor-group benchmark

Measured on 2026-10-03 with Coder IQ1_M on native Windows 11 Pro 10.0.26300. The 3990X has 64 physical cores / 128 logical processors, 128 GiB DDR4-3200, and **one NUMA node spanning two processor groups**. The RX 6900 XT has 16 GiB VRAM (gfx1030), driver 32.0.21045.5002.

With the same explicit 63 workers and 3,245 effective expert-cache slots, decode medians changed from **12.69 to 46.09 tok/s** for short inputs and **13.08 to 47.43 tok/s** for 4K inputs after preserving processor groups when pinning threads. The implementation is [PR #626](https://github.com/Niko1221/Strata/pull/626). It leaves automatic worker sizing unchanged. See the [full report](../../../docs/benchmarks/windows-zen2-gfx1030-workers.md) for all rows and limitations, and [PROBE.md](PROBE.md) for the topology/API probe.

## Provenance and conditions

- [build-metadata.json](build-metadata.json) records exact source revisions, binary hashes, ROCm 10.2.0a20260930, Clang 24, pinned ggml, and build flags. The baseline retains upstream affinity behavior. Both paired binaries still contained an earlier automatic-worker proposal, bypassed in every paired run by explicit `--pool-workers 63` or `31`. That proposal is absent from the final code PR.
- Both paired builds used context 65,536, INT8 KV / 32,768 resident cells, auto prefill (8,192), `--expert-cache 2500` (3,245 effective profile-sized slots), the shipped Coder expert profile, and MTP `--spec 4 --spec-min-p 0.5`. Exact server configs are under [configs](configs). Model hashes and hardware snapshots are in the [original release benchmark](../2026-10-03-rx6900xt-coder/README.md).
- Each short/4K row contains three measured requests after a separate 256-token warm-up. Temperature 0, reasoning `none`, no images, 256 output tokens. Every measured request reused zero prefix tokens. Expert caches are warm; these are not cold model-load timings.
- Engine tokens / elapsed engine time give prompt and decode rates; client TTFT and total latency are measured separately. Actual inputs were 94 / 4,022 / 32,685 tokens; the counting endpoint uses a different renderer and estimates 40 more tokens for the long cases.
- Run order: baseline 63, baseline 31, fixed 63, fixed 63 single 32K, coding smoke check, fixed 31, final automatic settings. Only one model was loaded at a time, after compilation/packaging finished. The 32K request has its own excluded warm-up but shares the fixed-63 server session.
- Other desktop applications stayed open. Paired free VRAM varied from 425/423 MiB (baseline 63/31) to 214/145 MiB (fixed 63/31); the latter two were flagged LOW. The fixed-31 regression is retained. Background load and memory conditions were not controlled, so that regression's cause is unresolved.

[comparison.csv](comparison.csv) and [comparison.json](comparison.json) collect median/min/max values from each saved summary. Every session includes requests, generated text, stream timings and usage in `raw.jsonl`, an excluded `warmup.json`, health snapshots, and engine log excerpts. The single 32K observation and final automatic-cache validation are labelled separately from the paired comparison.

## Automatic settings before the review follow-up

The final build at `d828e8e2cea24908f9d24597e987257ed293acc1` removed the earlier hardware-specific cap. Started without `--pool-workers`, it selected **63 workers**. Auto cache selected **2,675 slots**, with 1,143 MiB free after startup. Its three-run short/4K decode medians were **43.13 / 57.91 tok/s**; TTFT medians were **2.20 / 14.29 seconds**. These validate the final default path, but the different cache capacity and generated text/draft acceptance mean they are not the matched before/after affinity comparison. See [final-auto](final-auto), [final-auto-metadata.json](final-auto-metadata.json), and [final-installed-config.json](final-installed-config.json).

## Reproduce

Build each source revision using `tools/hip/build_windows.bat` with the toolchain/flags in `build-metadata.json`. Install and verify the model as described in the original release report. Copy the matching config to a local file and update its absolute executable, model, tokenizer, profile, library and log paths. Start one loopback server at a time from the repository root:

```powershell
python serve/server.py --engine strata --config path/to/local-config.json --port 8081
```

Once it is ready, run the standard-library-only harness, keeping each configuration's output separate:

```powershell
python bench/results/2026-10-03-windows-groups/benchmark.py --url http://127.0.0.1:8081 --cases short,4k --runs 3 --out bench/results/repeat-groups --log path/to/engine.log
```

Restart the model between configurations; use a new output directory and verify `prefix_reused_tokens` is zero. To reproduce the single longer check, use `--cases 32k --runs 1`. For final automatic selection, use the final code revision and omit `--pool-workers`, with `--expert-cache auto`. Do not overwrite the recorded evidence directories.

The generated clamp function passed 2,001 bounded integer cases in [coding-smoke-check.json](coding-smoke-check.json); reproduce with `check_code.py --url http://127.0.0.1:8081`. This is an operational smoke check, not a general coding-quality score. [cpu-tests-final.txt](cpu-tests-final.txt) records the pre-review affinity and stress-test invocations. `cpu-tests.txt` is the earlier candidate's record and includes a worker-default test subsequently removed with that proposal. Both the AVX-512 pool self-test and `pool_stress` skipped their workloads on this CPU. The latter returns zero, so its CTest "Passed" line does not establish a stress-workload pass. Linux compilation was unavailable because the configured WSL image was missing.

## Review follow-up

The later host-restoration correction in PR #626 uses reversible CPU Set selection and is covered by separate [review validation](review-validation.md). It was built and tested, but throughput was not remeasured; all speed figures above remain tied to their original recorded revisions. Local account-name components in captured test paths are redacted as `<user>`; test names, outcomes and timing values are unchanged.
