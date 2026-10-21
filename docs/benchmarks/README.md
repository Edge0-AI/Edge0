# Community Hardware Benchmarks

This directory collects real-world performance numbers for edge0 across different
hardware, contributed by the community. If you run edge0 on your machine, your
numbers are welcome here — **every reporter is credited by handle**.

## How to contribute a result

1. Run the benchmark on your device (macOS / Apple Silicon is supported today):

   ```bash
   cd python
   python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
   export EDGE0_8B_MODEL=$PWD/models/edge0-8b     # or EDGE0_35B_MODEL
   BENCH_LONG=1 .venv/bin/python examples/bench.py edge0-8b
   ```

   To print the discussion's ready-to-paste submission block:

   ```bash
   .venv/bin/python examples/bench.py edge0-8b --submission \
     --reporter @your-handle --storage "internal SSD" --on-battery no
   ```

   `--submission` selects the standard long prompt automatically and retains
   the benchmark's two timed runs, per-run warmup, sampling, mean decode rate,
   and maximum MLX allocator peak. Prefill is the mean of the two per-run
   rates. It rejects `BENCH_PROMPT` rather than silently submitting a custom
   workload. The default invocation and return fields remain unchanged.

   Add `--json` to emit one JSON object on stdout, with Python runtime and
   benchmark progress redirected to stderr:

   ```bash
   .venv/bin/python examples/bench.py edge0-8b --submission --json \
     --reporter @your-handle --notes "quiet system" > submission.json
   ```

   `--json` requires `--submission`. Hardware model, chip, system RAM, macOS
   and installed MLX version are detected when available. Unknown text fields
   remain empty and unknown RAM is JSON `null`; storage and power state are
   supplied by the reporter, not guessed. `--notes` accepts measurement context.
   Non-default token/warmup counts and sampling environment overrides are
   recorded in the notes; use the defaults for community comparisons.

   `peak_active_gib` here is `mlx.core.get_peak_memory()`, **not** process RSS
   or a Windows process working-set peak. JSON uses the existing submission
   block fields; this command does not append to or change `results.csv`.

2. Post your numbers in the pinned
   [Hardware Benchmarks discussion](https://github.com/Edge0-AI/Edge0/discussions/142),
   using the submission block there.

3. Either a maintainer curates your reply into [`results.csv`](./results.csv),
   or you open a PR adding your own row — we'll help review it.

> Other platforms (Windows / iOS / Android) don't have a one-command benchmark
> yet. Help build them! See the good-first-issues linked from the discussion:
> [#139](https://github.com/Edge0-AI/Edge0/issues/139) (Windows),
> [#140](https://github.com/Edge0-AI/Edge0/issues/140) (iOS),
> [#141](https://github.com/Edge0-AI/Edge0/issues/141) (Android).

## Data

Results live in [`results.csv`](./results.csv) (machine-readable; GitHub renders
it as a table). Columns:

| column | meaning |
| --- | --- |
| `date` | submission date (YYYY-MM-DD) |
| `tier` | `edge0-8b` or `edge0-35b` |
| `runtime` | backend used, e.g. `python-mlx` |
| `device` | device / machine name |
| `chip` | SoC, e.g. `Apple M2` |
| `ram_gb` | unified / system memory in GiB |
| `os` | OS version |
| `storage` | `internal SSD` / `external USB-SSD` (matters for expert paging) |
| `mlx_version` | `pip show mlx` version |
| `decode_tok_s` | decode throughput (tokens/sec) |
| `prefill_tok_s` | prefill throughput, prefer `BENCH_LONG` long-context |
| `peak_active_gib` | MLX peak active memory (GiB) |
| `on_battery` | `yes` / `no` |
| `reporter` | GitHub handle, e.g. `@you` |
| `notes` | anything unusual (throttling, short vs long, background load) |

## Why this matters

edge0 streams experts from SSD on demand, so throughput depends heavily on RAM
and storage speed — low-memory machines evict and re-read expert pages more
often. Collecting numbers across real hardware helps us (and users) understand
the practical performance envelope on each device.
