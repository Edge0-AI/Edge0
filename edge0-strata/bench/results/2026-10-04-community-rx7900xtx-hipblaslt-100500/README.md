# Community benchmark on AMD Radeon RX 7900 XTX 24 GB — follow-up: a gfx1100 hipBLASLt 100500 table, the 128K prefill confound resolved

Measured 2026-10-04 by xyzzing. What was tested: the ROCm 10.2 nightly engine from
[the 2026-10-03 comparison](../2026-10-03-community-rx7900xtx-rocm102-nightly/), whose 128K
prefill number was flagged as confounded — its hipBLASLt library reports version 100500, no
gfx1100 table ships for it, and the engine falls back to plain hipBLAS (the documented
version-mismatch mechanism). This follow-up tunes a matching table **on this device, against
that exact nightly library**, and re-measures the same 128K workload with and without it.
Main limitation: results are one-shot `strata generate` cells (not the serve path), and the
table's solution IDs are scoped to this hipBLASLt build — a ROCm update invalidates it, as it
does the shipped 100100/100200 tables.

## Hardware and software

Identical to the parent folder: RX 7900 XTX 24 GB (gfx1100), Ryzen 9 7900X, 96 GB DDR5, NVMe
ext4, PCIe Gen4 x16; Fedora 44, in-tree amdgpu; the ROCm 10.2 nightly SDK
(`libamdhip64.so.7.17.26392`) and the matching nightly hipBLASLt (`libhipblaslt.so.1.5`,
runtime-reported version **100500**). Engine: the same 0.1.38-lineage nightly binary
(`build-nightly/strata`) used for the parent's r10 cells. Desktop (KDE/Wayland + browser)
co-resident; per-cell systemd scope with `MemoryMax=66G MemoryHigh=61G`; each cell waits for
the previous cell's VRAM/RAM to drain before starting.

- The system-packaged ROCm 7.1.1 stack (hipBLASLt 100100) was **not** re-measured; its numbers
  come from the parent comparison and are quoted as context only.

## Model and configuration

Same model and serving-config lineage as the parent (Qwen3.8-Flash-Next GSQ-RCO IQ3_S, 2
shards; expert cache `auto` pre-filled from the shipped routing profile; `--vram-reserve-mib
3072`; KV int8 `--kv-resident 32768`; prefill chunk 8192; MTP loaded; `--spec 3`; greedy):

```text
# per cell (one-shot, not serve): the candidate serve config's exact args + tier context
env LD_LIBRARY_PATH=$SDK/lib \
    STRATA_HIPBLASLT_TUNING=<table|unset>  STRATA_PREFILL_RING=96 STRATA_GR_V3=1 \
    STRATA_SELECT_WMMA=1 STRATA_PREFILL_TIMING=1 \
  strata --tokens-file qsa-profile-131072.tokens --max-new 256 --max-context 139264 \
    --pack packs/iq3_s --native <shard1> --ple-gguf <shard2> --mtp mtp/rt \
    --expert-profile expert-profile.bin --expert-cache auto --spec 3 --kv int8 \
    --kv-resident 32768 --prefill 8192 --vram-reserve-mib 3072
# each cell launched under: systemd-run --user --collect --wait --pipe \
#   -p MemoryMax=66G -p MemoryHigh=61G (see ab-driver.py)
```

The two arms differ **only** in `STRATA_HIPBLASLT_TUNING`: `nightly-100500` points at the new
table (in this folder), `nightly-notable` leaves it unset — the parent folder's exact nightly
condition. The table itself was produced by the tree's own `tune_hipblaslt` run against the
nightly SDK's library (`tune-command.txt`): 16 GEMM shapes × {4096, 8192} token buckets = 32
cases, every candidate correctness-gated (rel-L2 ≤ 1e-4, max-abs ≤ 1e-2, padding canary), and
every table row is its case's measured best. The engine log confirms engagement per cell:
`prefill gemm: hipBLASLt tuning enabled (32 rows, gfx1100, version 100500)`.

## Method

One-shot `strata generate` cells (the tree's published AMD-rates methodology): greedy, 256
generated tokens (none ended early), the 131072-token code-agent prompt workload, one fresh
engine per cell. Arms interleaved-arm-blocked: 3 cells `nightly-100500`, then 3 cells
`nightly-notable`. Published numbers are medians of 3 clean cells with range; a cell would be
flagged and excluded if either phase fell below 40% of the arm's best observed rate
(background-GPU stall rule) — no cell was flagged, 7/7 clean
(`runs.json` has every cell with its raw-log reference).

- **prefill tok/s** = prompt tokens / GPU-timeline seconds (engine's `STRATA_PREFILL_TIMING`
  line) — not derived from wall time.
- **decode tok/s** from the engine's own decode summary line.
- **TTFT**: not available — the one-shot path does not print a TTFT line (serve path only).
- Prompt reuse: none (fresh engine per cell; the whole 131071-token prompt is read each time).
- Memory: engine loads ~46.8 GiB of experts into host memory; VRAM ~22.5 GiB incl. the
  3072 MiB desktop reserve.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prefill tok/s median (range) | Decode tok/s median (range) | TTFT |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| nightly 10.2 + gfx1100-100500 table | 131071 | 0 | 256 | 3 | **1687.0** (1679.6–1694.0) | 62.26 (62.19–62.41) | n/a |
| nightly 10.2, no table (parent's r10 condition) | 131071 | 0 | 256 | 3 | 926.1 (917.4–926.2) | 60.69 (55.24–61.02) | n/a |
| sanity: nightly 10.2 + table, 1k prompt | 1023 | 0 | 256 | 1 | 645.8 (single cell) | 69.38 (single cell) | n/a |

- **The table recovers +82% 128K prefill on the same binary and library** (1687 vs 926
  median). The no-table arm reproduces the parent comparison's confounded r10 cell (921
  tok/s, single run) within noise — the earlier number measured the fallback, not the
  toolchain.
- For context (single cells, different stack, from the parent folder): packaged ROCm 7.1.1
  with its matching 100100 table prefilled at 1535 tok/s. The nightly's 100500 library with a
  matching table prefills faster than that; cross-stack cells are not paired and this is not a
  claim.
- Decode is unchanged by the table, as expected: 62.26 vs 60.69 median with overlapping
  ranges — decode is O(1) in context and table-insensitive at these shapes.

## Correctness and limitations

- Greedy; **no token-identity check between arms** — the two arms use different GEMM
  implementations (hipBLASLt solutions vs plain hipBLAS) whose floating-point rounding can
  differ; the throughput comparison does not depend on token equality. Within the tuner,
  every candidate was byte-compared against the reference before timing (the tuner's own
  gate), and the engine refuses a table that fails its version/arch check.
- Single GPU, single host, one model; the 128K tier is the one the parent comparison flagged.
- Solution IDs are scoped to this hipBLASLt build (100500-era). A ROCm/hipBLASLt update
  invalidates the table exactly like the shipped ones — that version keying is the mechanism
  this folder exists to close, not a defect of it.
- Shipping the table in `tools/hip/` is an engine-tree change and deliberately **not** part
  of this results-only PR; the file, the shape set, and the exact tuning command are in this
  folder for whoever wants to pick it up (upstream tables for gfx1201/gfx1200 at 100500
  already ship; gfx1100 was the gap).
