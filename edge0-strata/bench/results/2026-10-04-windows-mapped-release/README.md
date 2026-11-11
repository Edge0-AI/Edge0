# Windows mapped expert-page release

Measured October 4, 2026, against upstream `6f32ec070f23ced9f50e704d854d775da52591ab` (0.1.39).
The same patched executable was run with `STRATA_FILE_RELEASE=0` and `STRATA_FILE_RELEASE=1`.
Releasing file-backed expert pages after GPU copies substantially increased available physical RAM on this
machine. Prefill was slightly slower. This is a memory-pressure improvement, not a general speed claim.

## What changes

Windows `FileExpertSource::release()` uses `VirtualUnlock` on complete interior pages of its read-only file views.
Windows documents that unlocked pages are removed from the process working set even when the call returns
`FALSE` with `ERROR_NOT_LOCKED`. Other errors are reported. The mapping stays valid and later reads fault in the
original bytes. Shared boundary pages are retained. The code resolves file spans directly, so it does not trim
resident complements, CUDA-pinned buffers or the temporary buffers used to assemble GGUF experts.

Existing cache-upload release calls use this implementation. Two additional calls release pages after prefill's
lent cache slots have finished refilling, in serving and one-shot generation. Other platforms are unchanged.
`STRATA_FILE_RELEASE=1` enables it; unset, `0`, or any other value leaves it disabled. The switch is read once
per engine process, so restart the engine when changing it.

This builds on the existing working-set handling in [#467](https://github.com/Niko1221/Strata/issues/467) and the
mapped-arena work in [#640](https://github.com/Niko1221/Strata/pull/640). Multi-GPU placement already exists upstream;
this patch does not implement it. See Microsoft's [VirtualUnlock documentation](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualunlock).

## Configuration

| Component | Measured configuration |
| --- | --- |
| System | Windows 11 Pro build 26200, Core i9-14900HX, 31.77 GiB usable RAM, 2 x 16 GB DDR5-5600 |
| GPU 0 | RTX 4080 Laptop, 12,282 MiB VRAM; also drives the desktop |
| GPU 1 | RTX 3090, 24,576 MiB VRAM, Thunderbolt eGPU |
| Links | Engine H2D probes approximately 12.3-12.9 GB/s and 2.4 GB/s respectively; no CUDA peer read/write |
| Driver | NVIDIA 616.56 |
| Storage | Model and pack on an internal WD PC SN560 NVMe |
| Build | MSVC 19.29.30157, CUDA 13.1.80, Release, AVX2 portable, SM 86/89, native experts enabled |
| Model | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ2_XS, with MTP |
| Placement | GPUs `0,1`, layer split 12, stage-weight trim, mapped experts |
| Runtime | Context 32768, int8 KV, 15 pool workers, auto expert cache and prefill, spec 4, minimum draft probability 0.5 |

GPU 0 held 6,144 expert pairs (about 8.00 GiB), GPU 1 held 13,381 (about 18.15 GiB); 19,525 of 24,576 pairs in total.
Prefill selected 8192 and borrowed approximately 3.77 GiB of expert cache on each GPU. The native pack used
`experts.bin` (35,454,976,000 bytes), `dense.bin` (1,538,035,200 bytes), and the GGUF PLE table on disk.
Power was AC; the existing power plan, drivers and system-managed pagefile were not changed.

Model revision: `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
MTP checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
llama.cpp revision: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
The two downloaded GGUF shards were fully SHA-256 verified:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| 1 | 39,225,954,592 | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| 2 | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

The measured prototype executable had SHA-256
`9118b02dcc0ad4bd8398ba033c641feb93800e0eeecea59ee42be141e4a186fc`.
It predates the submitted patch's decision to default the switch to OFF. Explicit `0` and `1` have the same
behavior in both versions; the numbers below are from the prototype, not a benchmark of the final rebuild.

## Measurements

Each arm had one excluded short warmup followed by three fresh prompts at each length, with 256 output tokens,
temperature zero and thinking disabled. Prompts ask for explanations of synthetic integer-processing code.
The harness checks that all prompt tokens were processed, no prefix tokens were reused, the response completed,
and exactly one request was recorded by the idle dedicated server. Actual prompt lengths were 1023, 4096 and 16384.

Medians of three requests per cell:

| Prompt | Prefill OFF / ON, tok/s | Decode OFF / ON, tok/s | Client TTFT OFF / ON, s |
| --- | ---: | ---: | ---: |
| 1K | 263.68 / 253.83 | 77.35 / 77.65 | 3.93 / 4.07 |
| 4K | 583.51 / 558.24 | 81.99 / 94.17 | 7.06 / 7.38 |
| 16K | 1045.85 / 1000.16 | 80.88 / 84.93 | 15.73 / 16.45 |

An independent Windows monitor sampled host RAM, process working sets, commit, pagefile use and physical-disk
counters approximately once per second. The following window includes warmup and gaps between requests, from
the first request's start to the last request's finish. GiB means bytes divided by 2^30.

| Metric | Release OFF | Release ON |
| --- | ---: | ---: |
| Available RAM, median | 0.512 GiB | 12.187 GiB |
| Available RAM, minimum | 0.277 GiB | 6.110 GiB |
| Engine working set, median | 23.379 GiB | 11.946 GiB |
| Engine nonprivate working set, median | 20.460 GiB | 8.689 GiB |
| System commit, median | 51.880 GiB | 51.973 GiB |
| Engine private commit, median | 36.807 GiB | 36.804 GiB |
| Pagefile use, median | 1.233 GiB | 1.218 GiB |
| Internal physical-disk reads, total | 16.750 GiB | 16.809 GiB |

The benefit was available physical RAM. Commit was effectively unchanged; this does not solve a commit-limit
failure or eliminate the need for pagefile capacity. Physical reads were also effectively unchanged.
Nonprivate working-set bytes include file mappings and shared images, not only model pages. Physical-disk
counters are system-wide and omit about 1.1-1.3 seconds at the sampled interval edges. The external disk was idle.

These were sequential OFF-then-ON runs on one machine, with warm engines and retained OS file-cache history.
The adaptive expert cache remained enabled. The OS cache was not flushed; this is not a cold-SSD experiment.
Three repetitions provide observed ranges, not confidence intervals. Code-explanation outputs differed between
the two arms despite identical measured requests; speculative decode rates are workload-dependent. Prefill
throughput dropped 3.7-4.4%, and median whole-request time changed by +3.5%, -1.1% and +2.9% at 1K, 4K and 16K.
These observations do not establish a decode-speed improvement, output equivalence, or results on other hardware.

## Reproduction and checks

`benchmark.py` and `monitor_windows.py` are the scripts used for these measurements (the raw per-trial export and the harness
self-tests were left out of the repository). The switch was called `STRATA_ARENA_RELEASE` in the prototype that measured this;
it is `STRATA_FILE_RELEASE` here, because the old name already means the opposite elsewhere (`=0` keeps experts a GPU holds).
The benchmark needs the repository's normal server/tokenizer dependencies; the Windows monitor also needs psutil.

Start a dedicated server on `127.0.0.1:18080` using a configuration with the runtime settings above, the patched
engine, and `"env": {"STRATA_FILE_RELEASE": "0"}`. Wait until the engine is loaded and idle. In another terminal,
run the monitor against the server process ID, and then the benchmark (replace the uppercase placeholders):

```powershell
python bench/results/2026-10-04-windows-mapped-release/monitor_windows.py --server-pid SERVER_PID --out off-host.jsonl --stop-file off-monitor.stop
```

```powershell
python bench/results/2026-10-04-windows-mapped-release/benchmark.py --pack PACK_DIR --config CONFIG_JSON --label release-off --out results-off --targets 1024,4096,16384 --runs 3 --max-tokens 256
```

Stop the monitor by creating its stop file. Unload and stop the server, change only the release value to `1`,
restart, and repeat with new ON output paths. Keep available VRAM, background workloads and model files stable.
For host summaries, select samples with `min(started_epoch_s) <= epoch_s <= max(finished_epoch_s)` from the
benchmark's results, including warmup. Compute sample medians/minima and physical-disk last-minus-first counters.
Sum `windows_memory` counters for `strata.exe` processes only; nonprivate working set is total minus private.

The C++ tests cover canonical and variable native layouts, full-page alignment, preserved boundary pages,
three release/read cycles with byte-for-byte checks, invalid indices, a closed source, and three GGUF roles
across two shards. Build with `STRATA_BUILD_TESTS=ON` and `STRATA_NATIVE_EXPERTS=ON`, then run
`ctest --test-dir BUILD_DIR -R "^(file_expert_source_test|expert_layout_test)$" --output-on-failure` in fresh
processes with the switch unset, `0`, and `1`.

The final submitted source was rebuilt with the toolchain above. Both C++ tests passed with the switch unset,
`0`, and `1`. That rebuilt engine also completed a warmup and one fresh
request each at 1023 and 4096 prompt tokens, generating 256 tokens per measured request on both GPUs with release
enabled. This was an inference smoke test, not a replacement for the prototype's OFF/ON performance comparison.
Its executable SHA-256 is `9f1e96c36b2ddb530422ee37734606491c0f4c3c9b720095c68d63953e89913b`.

Additional prototype checks with release ON completed 30 requests without runtime errors: 21 of 24 objective
checks passed, with three JSON-only checks failing because of Markdown fences. All 24 objective responses and
statuses matched the official engine with the same GPU placement, including those three failures. Six prose
requests were unscored. Needle retrieval passed near 1K/4K/16K tokens at 50% depth. The functional battery was not
run on the release-OFF prototype, and these limited checks do not prove general model quality or token parity.
