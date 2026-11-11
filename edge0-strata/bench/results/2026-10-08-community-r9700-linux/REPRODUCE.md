# Reproduce the recorded single R9700 measurements

These recipes reconstruct the measured October 8 engine. They do not benchmark the newer engine
in the report PR. Keep the measured baseline and candidate separate, use the same fixed model
assets, and record new binary hashes when rebuilding. The original binary hashes and changed
source hashes are in [source-layout.json](source-layout.json).

## Source and build

Run from a checkout with the public release commit in its git history. Set the report path before
entering the new source tree. The destination must not exist.

```sh
r9700_report="$PWD/bench/results/2026-10-08-community-r9700-linux"
r9700_source="$(mktemp -d /tmp/r9700-repro-parent.XXXXXX)/candidate"
git worktree add --detach "$r9700_source" d5ea7133741e67743c0e886bb426c0ce8d69cf6c
git -C "$r9700_source" apply --unidiff-zero "$r9700_report/patches/measured-baseline.patch"
git -C "$r9700_source" apply --unidiff-zero "$r9700_report/patches/pr1107-candidate.patch"
mkdir -p "$r9700_source/tools/hip/r9700"
cp "$r9700_report"/harness/*.py "$r9700_source/tools/hip/r9700/"
```

For a baseline source tree, repeat with a new destination and apply only `measured-baseline.patch`.
The archived diffs use zero context and require `--unidiff-zero` against the exact recorded base.
The source verification in `verify_report.py --check-sources` applies both patches in a temporary
directory and compares every changed file against its original SHA-256.

Use the system ROCm 7.2.3 compiler/libraries for the measured environment. The source dependency
is llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`; its GitHub source archive SHA-256 is
`c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280`. Set `r9700_ggml` to that
unpacked tree. The original build used CMake 3.31.10 and Python 3.12.

```sh
cmake -S "$r9700_source" -B "$r9700_source/build-r9700" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR="$r9700_ggml" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$r9700_source/build-r9700" --target strata strata-device -j8
```

The Python environment needs the reconstructed tree's root `requirements.txt` dependencies.
The recorded `single_gpu.py` intentionally fixes ROCr UUID `GPU-1d9ca7a7f0a06a6d` and BDF
`0000:63:00.0`; it verifies count=1 and gfx1201 before launch. On another host, explicitly update
its UUID/BDF, config isolation variables and NUMA settings, and record that new environment.
The tests do not change power limits, clocks or global settings.

## Model assets

Use the repository/revisions and all checksums in [model-manifest.json](model-manifest.json) and
the [IQ3_S](evidence/iq3s-assets.json) / [IQ2_XS](evidence/iq2xs-assets.json) inventories.
Set `r9700_assets` to an external directory with the recorded layout:

```text
gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
gguf/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
gguf/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf
packs/iq3_s/                 native pack and tokenizer
packs/iq2_xs/                native pack and tokenizer
mtp-bf16/rt/                 matching MTP runtime and draft_vocab.bin
```

The original preparation used `tools/iq_pack.py` on each native GGUF, then `tools/mtp_fetch.py`,
`tools/mtp_pack.py --experts q2_0`, and `tools/mtp_rt.py`; `data/draft_vocab.bin` was copied into the
runtime directory. The [build guide](../../../docs/AMD_HIP.md) describes the native pack workflow.
The measured config does not enable `--mmap-experts`. Use the assets' recorded hashes rather than
assuming a current model download matches. No weights or generated packs are included in this PR.

## Fresh IQ3_S speed

With the source, build and assets prepared, materialize the archived configuration without
overwriting it. Restore the exact archived token fixture instead of generating a new prompt.

```sh
r9700_run="$(mktemp -d /tmp/r9700-speed.XXXXXX)"
python3 "$r9700_report/configure.py" \
  --template "$r9700_report/current-speed/iq3_s-config.json" \
  --source "$r9700_source" --assets "$r9700_assets" \
  --exe "$r9700_source/build-r9700/strata" --out "$r9700_run/config.json"
gzip -dc "$r9700_report/current-speed/fixtures.json.gz" > "$r9700_run/fixtures.json"
python3 "$r9700_source/tools/hip/r9700/single_gpu.py" \
  --device-binary "$r9700_source/build-r9700/strata-device" \
  --config "$r9700_run/config.json" --record "$r9700_run/launch.json" \
  --cpu-nodes 0,1 --preferred-node 1 run -- \
  python3 "$r9700_source/tools/hip/r9700/bench_engine.py" \
  --config "$r9700_run/config.json" --fixtures "$r9700_run/fixtures.json" \
  --lengths 1024 4096 32768 131072 --shape-warmups 1 --repeats 3 \
  --max-new 256 --out "$r9700_run/session"
```

Only `phase=measurement` enters the result table. Check three records per length, `generated=256`,
`reused=0`, and `prompt_read=input_tokens`. Prompt speed is `prompt_read * 1000 / prompt_ms`;
decode speed is `generated * 1000 / decode_ms`. Use `ttft_s` and `wall_s` directly, and report
all repeats with median and full range. The workload is synthetic; output IDs and draft counts
need not stay identical with the product policies.

## Controlled code comparison

Build both reconstructed sources. Use `configure.py` with the four templates in
`evidence/configs-tensile-repro-fixed/`, substituting the matching baseline/candidate executable.
Use one common `--source` for both configs' working directory and expert-profile path, so the
paired runner can require all controls except executable/log to be identical.
Restore `evidence/fixtures.json.gz` to the new run directory.

```sh
python3 "$r9700_source/tools/hip/r9700/run_pairs.py" \
  --baseline "$r9700_run/iq3_s-baseline.json" \
  --candidate "$r9700_run/iq3_s-candidate.json" \
  --fixtures "$r9700_run/fixtures.json" --out "$r9700_run/incremental" \
  --mode incremental --lengths 32768 --increments 256 512 900 2048 4096 --pairs 5
```

This runner starts independent engines in alternating order, warms each shape and measures it
once per engine. It sets prompt cache to 6 for the incremental path. For the archived fresh
comparison, use `--mode fresh --lengths 4096 32768 131072 --pairs 2` and a new output directory.
Use IQ2_XS templates separately; its recorded incremental contrast was two pairs.

## Archive scope

Paths containing `${MEASURED_REPO}` and `${ASSETS}` are placeholders, not directly executable configs.
The configuration helper replaces them with explicit local paths. The copied result/config hashes
describe the original records; `SHA256SUMS` records the published file hashes separately.
Raw tensor/logit captures, resource timelines, model files and binaries remain outside this compact
report. `verify_report.py` audits the included evidence, not the omitted captures or a fresh GPU run.

All performance results, warmups, launch records and complete engine logs are consolidated in
`data/measurements.json.gz`. Its `json` and `logs` maps use their original report-relative paths.
The three CSVs in `data/` expose per-request timings and the published summaries. To check and unpack
complete records without running inference, choose a destination that does not exist:

```sh
python3 "$r9700_report/verify_report.py" --check-sources --extract /tmp/r9700-records
```

Historical diagnostics and HTTP regression scripts/results are available in the
[original full archive](https://github.com/zihaomu/Strata/tree/17006083063b4443458f6f0b3f5c8325cf9cca2a/bench/results/2026-10-08-community-r9700-linux).
They are outside this report's performance reproduction commands.
