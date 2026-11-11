# Measurement tools

Python scripts (standard library only, run with `uv run --no-project python -I <script>`) that measure the
reference CLI and compare measurements across runs, builds and machines. They only call `edge0-infer`; nothing
here is linked into the engine.

To measure performance on macOS or Android, start with [Benchmarking](../../docs/benchmarking.md): commands,
recommended settings and reference figures. This page is the developer reference for the scripts.

| Script | Purpose |
| --- | --- |
| `hosts.py` | The platform layer: runs the measured command and reads system state. |
| `sweep.py` | Plain measurement: four named configurations, a few processes each, medians and ranges (no pairing, no noise, no stop conditions). |
| `abba.py` | Paired short-process comparison of two commands (A/B/B/A/A/B/B/A); use it to decide whether an optimization stays. |
| `turns.py` | First-token cost of follow-up turns in one conversation, reuse against cold. |
| `report.py` | Report format, `compare` against a baseline, `render` to Markdown. |
| `../validation/model_support.py` | Effect check on six fixed tasks through the official `run` entry. |

## Platform interface

Every script talks to a host object (`hosts.select(platform)`); only `hosts.py` knows about the operating system.
A new platform is one class with the operations below (the first four are the core contract; the background load check is called together with `preflight()`):

| Operation | Contract |
| --- | --- |
| `run(argv)` | Run the measured command (a list of strings, never a shell line) where the engine runs; return exit code, standard output, standard error. A remote platform quotes every argument itself. |
| `preflight()` | Raise `PreflightError` unless the measurement conditions hold (macOS: AC power). Called before every process. |
| `snapshot(model)` | Return one dictionary: `platform`, `time`, `power`, `thermal`, `memory`, `load`, `top_processes` (executable names only) and `residency` of the model asset file in the page cache. |
| `check_background_load()` | Raise `BackgroundLoadError` (a `PreflightError`) when a process outside the system directories uses more than 15 percent CPU; the message and the report carry executable names only. System processes are recorded in the snapshot (`background_load`) but never block. The measuring script and its parent are not counted. Called before every process; a failure makes the cell `failed` and stops the measurement. |
| `device()` | Return model, chip, physical memory in bytes and system version for the report identity. |

Rules for the values: an item that cannot be read is `{"available": false, "reason": "..."}`, never zero and never
an estimate. A platform keeps its own names and sources (for example `vm_stat` text on macOS, a proportional set
size on Android) instead of renaming them to another platform's terms. Unregistered platform names
are refused at argument checking with exit code 2; no measured command runs.

## Report format

`moe-perf-report/1`, a single JSON document:

```text
format, identity{engine, binary_sha256, tools_sha256, workspace, model, device, build_consistency},
protocol{kind, ...}, sides{name: {command, configuration}},
cells[{index, side, status, reason, metrics, memory, snapshot_before, snapshot_after, ...}],
noise, pairs, decision, summary (kind turns only)
```

* `identity.engine` and `identity.model` come from the `bench` summary row (build commit and whether the tree was
  modified, GGML origin and version, registered asset digests); `workspace` is the Git state of the tree the
  script ran from, and `build_consistency` says whether the binary was built from that state. A mismatch is
  reported, not refused.
* `configuration` is the effective load configuration printed by the engine, not inferred from the command line.
* `status` is `ok`, `failed` (with the reason) or `not_run`.
* Writing is incremental: `<out>.partial.jsonl` receives a header and one record per finished process, so cells
  measured before a failure survive. The failed cell is recorded and the measurement stops. Only a complete run
  writes `<out>.json` and `<out>.md` and removes the partial file. Existing outputs are never overwritten.
* Paths are replaced by environment variable names; system snapshots are in the JSON only.
* `noise` uses the pairing names: `prefill_seconds` is the first token time, which also bounds the prefill speed.

## Plain sweep

```text
sweep.py --cli build/mac-metal/bin/edge0-infer --model <assets> --out <prefix> --raw <new directory>
         [--platform android] [--device cpu|metal] [--configs predicted|<names>|all] [--prompt long|medium|short]
         [--runs 3] [--cooldown 30] [--option <bench argument>]...
```

Without options it measures the recommended configuration `predicted` (warm-up 10, 100 timed steps, greedy) on the
platform's device with the default prompt: the 3,084-token prompt for Metal on macOS, the 627-token prompt for
every other platform and device (including `--device cpu` on macOS). Named configurations: `predicted`, `actual` (`routing=actual`), `ctx16384` (`--context 16384`),
`direct` (`expert_read=direct`); `--configs all` selects all four, as in the stored macOS baseline. `--option`
appends one argument to every `bench` command and is recorded in the protocol. The selected configurations rotate
for `--runs` processes each; every process must report the input count of the chosen prompt. A failed process is recorded and the sweep continues; nothing is repeated. Background load is recorded in the
snapshots (`background_load`, top non-system processes by name) and never blocks. `summary` holds, per configuration, medians and ranges, memory maxima and outlier notes (a
process more than 10 percent from the configuration median in decode speed or first token time, with what the
snapshot shows before it).

Prompt tiers keep platform defaults unchanged:

| Tier | Input tokens | Timed output tokens | Purpose |
| --- | ---: | ---: | --- |
| `long` | 3,084 | 100 | Steady prefill speed; default for macOS Metal |
| `medium` | 627 | 100 | Default for every other platform and device |
| `short` | 19 | 512 | Short story request with sustained long-output decode |

All tiers use the same ten-step warm-up. `--prompt short` explicitly selects the
short story fixture; it does not change the defaults for any platform. Bench ignores
end-of-generation for its fixed timing window, so this measures sustained decode,
not necessarily 512 tokens of a normal interactive answer.

## Baselines

`report.py baseline --report <sweep report> --sides predicted actual --threshold 0.05 --out <file>` keeps identity,
protocol, effective configuration and per-cell metrics of the chosen sides plus a fixed threshold. `compare` uses the
larger of the baseline noise (if any), the fixed threshold and 2 percent, requires more than half of the candidate's
processes to be worse than the baseline median, and compares only the baseline's sides (failed cells in other sides
of the candidate are ignored).

## Paired measurement

```text
abba.py --a '["build/mac-metal/bin/edge0-infer","bench","--model","<assets>","--device","metal",
              "--user-file","tests/fixtures/bench/library-report-prompt.txt","--warmup","10","--ntok","100","--runs","1"]'
        --b '<same plus the option under test>' --out <prefix> --raw <new directory> [--noise <noise report>]
        (--noise-only | --record-only | --target decode_tokens_per_second --correct)
        [--cooldown 30] [--platform macos]
```

The workload is fixed (library prompt, warm-up 10, 100 timed steps, one run, Metal) and checked on every
process. Eight processes run in the order A B B A A B B A; adjacent processes form four pairs. `--noise-only`
measures A against identical A and stores the noise (largest pair difference per metric, at least 2 percent).
Other runs need a noise report of the same protocol (cooldown, workload) and either only record
(`--record-only`) or apply the retention rules (target better in at least 3 pairs, mean gain above noise, other
metric not worse than noise, B physical peak at most 4 GiB, correctness flag).

Memory-pressure measurement has been removed: macOS swaps out a helper process that holds physical memory, so the condition never held; reports measured with it are refused as noise reports.

## Comparison

```text
report.py compare --baseline <report.json> --candidate <report.json>
```

Exit 0: no regression. Exit 1: regression. Exit 2: input error (bad format, missing identity, failed cells,
different protocol, device model or physical memory, effective configuration, noise missing, or a report kind
other than `abba`). A metric regresses when the candidate median is worse than the baseline median by more than the
baseline noise (at least 2 percent) and at least 3 of the candidate's 4 processes are worse than the baseline
median. Baselines are per device model and memory size (`baselines/`); they keep identity, protocol, effective
configuration, per-cell metrics and noise, not system snapshots.

`report.py render --out <file.md> <report.json>... [--append <file.md>]...` only lays out numbers already in the
reports; it does not judge.

## Conversation reuse

```text
turns.py --cli build/mac-metal/bin/edge0-infer --model <assets> --out <prefix> --raw <new directory> [--runs 3]
```

Runs `edge0-infer bench --turns <conversation>` (context 16384, greedy) for three combinations in rotation:
predicted routing with thinking off, predicted routing with thinking on, actual routing with thinking off. Each
turn row reports the reuse method, reused and computed token counts, the first token time and, for comparison, a
cold computation of the same input on a fresh session. `computed_tokens` counts every forward of the request (new
input plus decode steps), so `reused_tokens + computed_tokens = input_count + generated_count - 1`. A process
whose turn count, thinking flags, conversation digest or effective configuration do not match is a failed cell.
The report keeps medians and ranges per combination and turn and states whether generated tokens agree across
processes.

## Effect check

```text
model_support.py run    --cli <edge0-infer> --model <assets> --out <new results directory> [--device metal]
model_support.py review --results <directory> --verdicts <json>
model_support.py render --results <directory> [--perf-report <report.json>]... [--out <file.md>]
```

Six fixed tasks run for both routings. Decidable criteria are decided by the script; the three tasks that need
reading stay `pending` unless a certain defect is found. `review` accepts verdicts only for pending items, each with
`verdict` (`pass` or `fail`), a non-empty `reason` and a `reviewer`; one invalid entry rejects the whole file.
`render` exits 1 while any task is pending or not run. A performance report measured with another engine commit or
other asset digests is flagged, not combined.

## Adding a platform

1. Add a host class in `hosts.py` with the operations above and register it in `hosts.select`.
2. Keep platform-specific names and sources in the snapshot; mark unreadable items unavailable with a reason.
3. Record a baseline per device model and memory size with `abba.py` (noise run, then the measured run).
4. Add recorded command output under `tests/fixtures/perf/` and cases to `hosts_test.py`.

## Android CPU measurements

Build and push as described in the [Android guide](../../docs/getting-started.md#android-cpu).
Use `--platform android` with `sweep.py`; pass the cross-compiled binary as `--cli` so
its hash identifies the measured executable. Commands execute through `android-cli`.
On Android the sweep uses the 627-token `library-report-prompt-head.txt` by default (`--prompt long` selects
the 3,084-token prompt, which takes over two minutes to prefill on the device).
The sweep uses CPU and records `direct` as `out_of_scope` with a reason, not a failure.

```sh
uv run --project tools/assets --offline --no-sync python -I tests/perf/sweep.py \
  --platform android --cli build/android-cpu/bin/edge0-infer --model "$MOE_ASSETS_35B" \
  --out build/android-sweep --raw build/android-sweep-raw
```

Keep the device charging over AC or USB and the screen awake for the entire measurement;
preflight refuses other states. Enable performance mode manually and record that choice:
it cannot be reliably queried. Cool between processes and do not overlap other tests.
Snapshots retain device battery, thermal output, per-policy frequency limits, `meminfo`,
load and executable names; unavailable fields include reasons. Process RSS/PSS comes
from the engine's own memory snapshot. PSS peaks and remote `mincore` residency are unavailable.

Report measured throughput, first-token latency and RSS peak without extrapolating results to other devices.

## iOS host

`--platform ios` uses the same App and file transport as the platform suite.
Export `MOE_IOS_DEVICE`, `MOE_IOS_TEAM`, and the asset paths before running.
A wired, unlocked device is required. The validated devicectl version does not
report battery/charging state or physical RAM; those unavailable values carry
reasons. Process snapshots supply physical RAM, available process memory and
thermal state before and after each measured command. Numeric temperature,
CPU frequency and asset page residency are unavailable. Device identity includes
only model and OS version, never the device name or identifier.

The default backend is Metal and the default prompt is medium (627 input tokens).
Use `sweep.py --platform ios --configs all` to include both cached and direct reads.
The default three processes per configuration alternate in rotation with cooldown.
Do not measure concurrently with another device session or immediately after a suite.
The engine reports `phys_footprint` with process lifetime scope; it is not a resettable
per-request peak. A system-terminated process is a failure and must be investigated,
not silently retried or treated as an unsupported configuration.

## BPE merge microbenchmark

The standalone CPU benchmark encodes an 8,192-byte word with a synthetic byte
vocabulary and ranked pair merges. It verifies the exact token ID, warms the
tokenizer once, then prints elapsed milliseconds for three encodings. It isolates
merge scheduling on a long span and does not measure model inference throughput.

```sh
cmake --preset cpu
cmake --build --preset cpu --target moe_app
c++ -O2 -std=c++17 -Isrc -Iinclude tests/perf/bpe_merge_bench.cpp build/cpu/src/app/libmoe_app.a -pthread -o build/bpe_merge_bench
build/bpe_merge_bench
```
