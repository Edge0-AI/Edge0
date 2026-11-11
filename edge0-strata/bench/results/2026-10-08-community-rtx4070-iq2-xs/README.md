# Community benchmark: Flash-Next IQ2_XS on RTX 4070 12 GB / 32 GB DDR4

**Evidence status: the primary E4/S24 result and frozen reference have now been revalidated against the archived per-run command, summary, stdout, stderr, exit, process, and GPU snapshot files on the test desktop.** The raw logs remain on that desktop and are not published here. This is a benchmark report, not an engine patch.

Measured **2026-10-08** by [Hcl192088](https://github.com/Hcl192088), on one consumer GPU with 12 GB installed VRAM and 32 GB DDR4 system RAM. The most interesting result is a **48.23 accepted-decode-tok/s median over five runs** for one adaptive-cache candidate. A separate frozen `CURRENT_BEST` reference is independently pinned at a **47.76 tok/s median over five runs**. Neither number should be interpreted as end-to-end request speed or as a universal hardware ranking.

## Full engine source and reproducibility package

**The benchmark used a modified Adrian-derived Strata engine, not an unmodified official upstream binary.**
The exact source tree used in the archived run is published in this user's fork as
[**benchmark-iq2-source-9ec3806**](https://github.com/Hcl192088/Strata/tree/benchmark-iq2-source-9ec3806),
pinned to Git commit `9ec3806058cf32ab27a55e4377daf7cf0d087dec` (tree
`90346cf6af4fb642ff7847327e0f3456ed90799a`).

- [**REPRODUCE.md**](REPRODUCE.md) — clone/build/model-preparation/verification/run instructions and exact missing-data list.
- [**reproduce.py**](reproduce.py) — standard-library SHA-256 input validator and sequential five-run benchmark harness.
- [**provenance.json**](provenance.json) — immutable source, model/pack/MTP/profile/prompt hashes and original arguments.

**Independent community reproduction:** The original `learned-heart.bin` expert
profile was generated as a Strata expert-cache profile; the historical file's
hash remains recorded, but **its bytes are not required to rerun the method**.
Strata has a built-in, opt-in `--expert-profile-save` feature (in `--serve`
mode with adaptive-cache settings) and `tools/make_profile.py` for generating
new profiles from a tester's own workload. The original 28,912-token prompt
is not redistributed; [`make_local_prompt.py`](make_local_prompt.py)
constructs a length-matched token-ID input from the reviewer's own local text,
and [`reproduce.py`](reproduce.py) accepts `--prompt-file` and `--profile`
overrides.

Different prompt/profile data can change the results. **The published 48.23
tok/s remains the historical observed five-run median, not a guaranteed
result for a new workload**. Source/build/test-method reproduction does not
require publication of a private user's prompt.

## Hardware

| Item | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, **12 GB installed VRAM**, single GPU |
| CPU | Intel Core i5-12600KF |
| RAM | 32 GB DDR4-3200 |
| OS | Windows 11 build 26200 |
| NVIDIA driver / CUDA compatibility reported by `nvidia-smi` | 591.86 / CUDA 13.1 |
| Storage, PCIe, power limits | Not yet verified |
| VRAM telemetry | Runtime reported 5.34 GiB expert cache + 820 MiB MTP allocation + 57.0 MiB verify buffers; the run-boundary `nvidia-smi` snapshots were 662 MiB used / 11,349 MiB free / 12,282 MiB total before and after. No in-run peak `nvidia-smi` sample was saved. |

**Memory terminology:** The runs report 20.00 **GiB of resident host RAM** for expert tiers, with page-locking refused and 20,479 MiB held through working-set minimum plus `VirtualLock`. This is **system memory**, **not VRAM**. The runtime also reports 0.64 GiB of pinned host RAM for KV streaming.

## Engine, model, and configuration

- Model family / quantization: Qwen3.8-Flash-Next, **IQ2_XS**, native/packed execution with MTP. Exact GGUF filenames and SHA-256 values, pack/MTP/tokenizer hashes, profile hash, and prompt hash are recorded in [provenance.json](provenance.json).
- Engine: locally built **Adrian-derived Strata** executable from source commit `9ec3806058cf32ab27a55e4377daf7cf0d087dec`; binary SHA-256 is recorded in [provenance.json](provenance.json). It must not be presented as an unmodified upstream release.
- Benchmark objective: **accepted decode tokens per second**, not speculative proposals/s, prompt throughput, or total request tok/s.
- The contemporaneous test-session report records **1,024 generated/accepted output tokens** and exit code 0 with no residual engine processes for each of the five E4/S24 candidate runs. **1,024 is the output length, not the configured context window.**
- During the primary E4/S24 campaign the verified command used `q4_0` KV, 100,000 context cells, `--kv-resident 20480`, `--spec 6`, `--spec-min-p 0.75`, `--mtp-max-t 3`, `--suffix-draft 3`, `--expert-cache 3796`, 9 pool workers, CPU affinity `all`, host worker, `--adapt-every 4`, `--adapt-swaps 24`, and `STRATA_LOOKAHEAD=0`; the complete public-normalized argument vector and environment are in [provenance.json](provenance.json).
- Adaptive-cache notation **E4/S24** means `--adapt-every 4 --adapt-swaps 24`. The comparison candidate E5/S32 uses `--adapt-every 5 --adapt-swaps 32`.
- The prompt content is not redistributed, but the verified `neuro.tokens` SHA-256, 28,912 prompt-token length, 100,000-cell context limit, sampling state, resident-memory policy, expert slots, and full environment are recorded in [provenance.json](provenance.json). Cache warm/cold comparability remains unresolved.

## Method and observed measurements

The operator's workflow ran separate sequential `strata.exe` processes and the archived primary records contain the command, summary, stdout, stderr, exit, process-before/after, and GPU before/after files for each E4/S24 run. The raw evidence was recovered from the desktop archive roots `strata_current_best_phases_20261008/phase3-adaptive-es/E4-S24` and `rtx4070_exact_benchmark_20261008/final-audit-winner-r1..r5`; [provenance.json](provenance.json) is the sanitized public index of those records, while the large raw logs remain local.

The primary E4/S24 rows below are **validated against archived raw telemetry**; comparison rows remain historical session records unless explicitly marked otherwise. See [transcribed-runs.csv](transcribed-runs.csv) for the per-row origin label.

| Test phase / configuration | Accepted decode tok/s, each run | n | Median | Mean | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Later E/S sweep: E4/S24 | 48.23, 47.98, 49.44, 48.83, 46.92 | 5 | **48.23** | 48.28 | Best observed candidate in this sweep, **not promoted** |
| Later E/S sweep: E5/S32 | 47.32, 48.23, 49.37, 46.57, 46.84 | 5 | 47.32 | 47.67 | Comparison candidate |
| Earlier adaptive sweep: E4/S32 | 46.71, 46.39, 48.97 | 3 | 46.71 | 47.36 | Different optimization phase; do not combine with later sweep |
| Earlier adaptive sweep: E8/S96 | 46.24, 44.97, 46.92 | 3 | 46.24 | 46.04 | Earlier phase only |

A separately maintained `CURRENT_BEST` reference is now verified as **45.94, 47.22, 50.04, 47.76, 49.37 tok/s** with median **47.76 tok/s (n=5)** from its archived final-audit records. Its five-point vector is kept in [provenance.json](provenance.json), not added as extra CSV rows. E4/S24's median advantage is **+0.98%** relative to that reference, below the campaign's **+1% promotion rule**. The control configuration remained E4/S32; it is incorrect to announce E4/S24 as a promoted production winner.

A single candidate run reached **49.53 tok/s** during a later adaptive-decay screen (`decay=0.50`), but **n=1 is not a reproducible performance result** and it is excluded from the summary above.

### Verified provenance

The primary E4/S24 five-run set is now verified from the archived per-run records rather than transcript-only values. The frozen reference set is likewise pinned to its five archived `final-audit-winner` records. See [provenance.json](provenance.json) for the normalized command, environment, artifact hashes, runtime-reported memory, run values, and evidence roots.

The runtime memory figures are allocations reported by Strata's stderr, not a single peak VRAM measurement. The saved `nvidia-smi` files are run-boundary snapshots and are therefore reported separately.

### What these results do and do not show

- They show reported **decode** performance close to 48 accepted tok/s on a PC with **12 GB physical VRAM and 32 GB DDR4**.
- They establish the reported decode result under the exact command and prompt-token identity in [provenance.json](provenance.json); they do **not** establish a universal 48 tok/s speed for other prompt/context sizes. Context capacity and actual processed prompt length must be stated separately.
- They **do not** establish model-answer quality or a general performance advantage versus another GPU or a newer upstream release.
- They are **not** evidence that 48 tok/s is a stable minimum: the five-run E4/S24 range was **46.92–49.44 tok/s**.

## Evidence status and limitations

1. **Completed for the primary result:** the five E4/S24 raw-record sets and five frozen-reference records were recovered and validated; a sanitized provenance index is included in this directory.
2. The raw stdout/stderr files are still desktop-local and are not uploaded. Reviewers cannot independently download the original logs from this PR.
3. The exact prompt bytes are represented by the verified `neuro.tokens` SHA-256 and 28,912 prompt tokens, but the prompt content itself is not redistributed.
4. The archived telemetry does not contain an in-run peak `nvidia-smi` sample; this report therefore does not claim a total peak VRAM number.
5. The comparison rows other than E4/S24 and frozen CURRENT_BEST remain historical benchmark records and are not all covered by the new provenance manifest.
6. This is a results-only submission, separate from any source-code optimization PR. The evidence boundary is explicit so maintainers can assess the measurements without additional benchmarking.

The submitted numbers are a **community benchmark report with explicit evidence limits**: the primary result is source-pinned and raw-record-validated, but the raw logs are not uploaded and the comparison rows are not all covered by the new manifest. Please review the evidence boundary rather than treating the figures as confirmed upstream benchmarks.
