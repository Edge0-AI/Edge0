# Page-granular mmap warming

## Change and contract

`SafetensorsMmap.touch(offset, length)` reads one byte from each OS page
intersecting the requested byte range. The mapped file remains read-only.
The method creates short-lived NumPy views and no payload-sized byte copy.

`seq_read()` uses this method inside its existing traversal windows.
`StreamingSwitchGLU.warm_pages()` supplies each selected expert's exact
weight, scale, and bias byte ranges instead of summing all their bytes.

The entry points are explicit expert warming and the Ling startup prewarm
path (`EDGE0_PREWARM=1`). Normal expert loading, cache policy, routing,
quantization, and numerical kernels are unchanged. The benefit applies to
these warming paths; token throughput is outside this benchmark's scope.

## Page-coverage proof

For a nonempty range `[a, a+n)` and OS page size `P`, its page IDs are

`floor(a/P), ..., floor((a+n-1)/P)`.

Read `a` first. Then read every page boundary strictly after `a` and strictly
before `a+n`. There is exactly one selected address in each intersecting
page, and every selected address is inside the requested range. Empty ranges
perform no reads. This also handles the case where a short, unaligned range
crosses a page boundary. A relative stride starting at `a` alone can miss
that final page.

The code uses `mmap.PAGESIZE` rather than a fixed 4 KiB assumption. The tests
instrument the executed address selections for both 4 KiB and 16 KiB pages,
including 1,000 seeded random ranges per page size and explicit boundary
cases. The kernel can read ahead or later evict pages; this method establishes
synchronous reads, not a pinning guarantee or a physical-storage byte count.

## Measurements

Reference: `fbab5f8c08e843e204c0fc6ae18b89a154c652cf`.
Host: Intel Core i5-3470, Linux x86_64, 4 KiB OS pages, Python 3.12.13,
NumPy 2.5.3. MLX-dependent checks used MLX 0.30.4 and MLX-LM 0.31.0 on CPU.
No GPU or phone timing is included.

Each timing comparison randomizes baseline/candidate order within a pair.
The 128 MiB run has 21 pairs and five operations per sample. The 512 MiB
replication has 11 pairs and three operations per resident sample. Cold-cache
samples perform one operation each. Raw timings, seeds, input hashes, and
source hashes are in the linked JSON files.

| Path | Baseline median | Page reads median | Median paired time reduction |
|---|---:|---:|---:|
| Resident 128 MiB whole-shard warmup | 29.731 ms | 0.481 ms | 98.38% |
| Resident four-expert warming, synthetic byte fixture | 5.438 ms | 0.329 ms | 93.98% |
| Resident 512 MiB whole-shard warmup | 75.361 ms | 1.971 ms | 97.40% |
| Explicit warming + materialization of four real experts | 5.593 ms | 1.767 ms | 68.38% |

The last row uses 31 pairs, with three operations per sample. Each operation
warms the selected expert ranges, calls `_build()` for all four experts, and
evaluates the resulting MLX arrays. It bypasses cache-hit shortcuts. Its
paired bootstrap 95% interval for time reduction is **68.20% to 68.53%**.
It measures that explicit host preparation path, not a complete decoder.
A separate-process run of the published probe reproduced the exactness checks
and measured a 73.89% paired reduction; its raw samples are also included.

The real-weight fixture contains nine published Edge0 experts: layers 0,
20, and 39, each with experts 0, 127, and 255. All gate/up/down weight,
scale, and bias payloads are preserved. Source checkpoint:
`Edge0/Edge0-35B-A3B-preview`, revision
`1ff9f4478890faec0368c5463b621d1036d5b518`. The fixture combines those experts
into one test layer; it is not a complete checkpoint or a prompt-derived
activation trace.

### Cold-page-cache control

| Whole-shard path | Baseline median | Page reads median | Paired reduction, 95% interval |
|---|---:|---:|---:|
| 128 MiB, file page cache evicted | 486.236 ms | 495.358 ms | -0.90%, [-2.65%, +0.37%] |
| 512 MiB, file page cache evicted | 2404.585 ms | 2427.666 ms | -0.20%, [-7.05%, +11.44%] |

These runs show no clear cold-cache speed advantage. `mincore` confirmed
zero resident pages before every cold sample. Eviction targets only the
benchmark's temporary file with `madvise(DONTNEED)` and
`posix_fadvise(DONTNEED)`; no global cache flush, root access, or change to
other files is used. Storage-controller caches are outside this control.

### Allocation and residency

The default old chunk-copy path peaks at **33,554,602 traced bytes** per
warming call. The page-read path peaks at **34,352 traced bytes** on this
host. These are `tracemalloc` peaks, not whole-process RSS, device memory,
or model-capacity measurements.

After either whole-file warming path, Linux `mincore` reports all **32,769**
128 MiB-fixture pages or all **131,073** 512 MiB-fixture pages resident.
Input-file SHA-256 remains unchanged.

### Exactness and test execution

- 81 real-weight materialized tensors match byte-for-byte before and after warming.
- 12 real-expert SwiGLU output comparisons match exactly, using synthetic inputs.
- 27 backend-free mmap/hygiene tests pass.
- Full candidate suite on MLX CPU eager execution: 87 passed, 1 skipped,
  2 slow tests deselected. Matched upstream: 60 passed, 1 skipped, 2 deselected.

With CPU JIT compilation enabled, this host's GCC 15 rejects duplicate
`_Float32` / `_Float64` typedefs in MLX 0.30.4's bundled preamble. The same
14 streaming-math failures reproduce on unchanged upstream. The successful
CPU runs disable compilation only in the test harness. Production code and
dependency pins are unchanged; the existing macOS/Metal CI remains the
compiled-backend validation path.

## Reproduce

Backend-free checks and timings require NumPy and pytest:

```sh
python -m pytest tests/test_streaming_mmap.py tests/test_repo_hygiene.py -q
python scripts/bench_page_warmup.py --mib 128 --repeats 21 --loops 5 \
  --cold --json /tmp/edge0-pagewarm-128m.json
python scripts/bench_page_warmup.py --mib 512 --repeats 11 --loops 3 \
  --seed 1092026 --cold --json /tmp/edge0-pagewarm-512m.json
```

Omit `--cold` on macOS. Use `--directory` to select the drive that holds the
temporary fixture. Timing assertions are deliberately excluded from CI tests.

The optional real-weight probe downloads about 16 MB and requires Requests,
Safetensors, and the pinned MLX stack. It never downloads the full checkpoint:

```sh
python -m pip install requests
python scripts/fetch_pagewarm_samples.py --out /tmp/edge0-pagewarm-samples
python scripts/bench_page_warmup_mlx.py \
  --samples /tmp/edge0-pagewarm-samples \
  --json /tmp/edge0-pagewarm-mlx.json --cpu-eager
```

Omit `--cpu-eager` for the installed backend's normal execution mode. The
Linux CPU-eager suite used for these results is reproducible with:

```sh
PYTHONPATH=src python -c 'import mlx.core as mx; import pytest; mx.set_default_device(mx.cpu); mx.disable_compile(); raise SystemExit(pytest.main(["-q"]))'
```

## Raw results

[128 MiB paired samples](page-warmup/benchmark_128m.json) ·
[512 MiB paired samples](page-warmup/benchmark_512m.json) ·
[Real-weight checks and paired samples](page-warmup/real_weights.json) ·
[Real-weight separate-process repeat](page-warmup/real_weights_repeat.json)
