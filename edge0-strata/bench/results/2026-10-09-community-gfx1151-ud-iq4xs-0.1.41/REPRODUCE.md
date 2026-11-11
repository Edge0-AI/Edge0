# Reproduce this report

Run from a checkout of Strata commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (v0.1.41), with this report folder copied into `bench/results/`. The measured engine and upstream task scripts come from that pinned commit; the added report files do not change them. The commands below use separate model, SDK, build and result directories. They do not change the system ROCm installation. Allow roughly 105 GiB for model shards, the small native pack and retained MTP source/output, plus space for the SDK, build and results. Check the destination filesystem and free space before downloading.

## Build and prepare the model

Set these paths to locations on the model filesystem:

```bash
STRATA_REPORT_MODEL=/path/to/models/strata-qwen38-ud-iq4xs
STRATA_REPORT_SDK=/path/to/build/strata-sdk-7.14.1/install
STRATA_REPORT_BUILD=/path/to/build/strata-halo-0141
STRATA_REPORT_GGML=/path/to/build/strata-ggml-pinned
STRATA_REPORT_PY=/path/to/venv/bin/python
STRATA_REPORT_OUT=/path/to/results/measurements
```

The SDK is AMD's [gfx1151 TheRock 7.14.1 archive](https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz), extracted under `STRATA_REPORT_SDK`. Its SHA-256 is `c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00`. The measured build used HIP 7.14.60850 / AMD clang 23, rather than the host's separate HIP installation. Use llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` for `STRATA_REPORT_GGML`. Create a Python environment and install Strata's `requirements.txt` there; generated-code grading also requires Linux `bwrap` and `/usr/bin/python3`.

```bash
export ROCM_PATH="$STRATA_REPORT_SDK" HIP_PATH="$STRATA_REPORT_SDK" HIP_PLATFORM=amd
export LD_LIBRARY_PATH="$STRATA_REPORT_SDK/lib:$STRATA_REPORT_SDK/lib/rocm_sysdeps/lib:$STRATA_REPORT_SDK/lib/llvm/lib"
cmake -S . -B "$STRATA_REPORT_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER="$STRATA_REPORT_SDK/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$STRATA_REPORT_SDK" \
  -DCMAKE_PREFIX_PATH="$STRATA_REPORT_SDK;$STRATA_REPORT_SDK/lib/rocm_sysdeps;$STRATA_REPORT_SDK/lib/llvm" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$STRATA_REPORT_SDK --rocm-device-lib-path=$STRATA_REPORT_SDK/lib/llvm/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR="$STRATA_REPORT_GGML"
cmake --build "$STRATA_REPORT_BUILD" --target strata -j 8

# the `hf` CLI comes from `pip install huggingface_hub`
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include 'UD-IQ4_XS/*' --local-dir "$STRATA_REPORT_MODEL"
export STRATA_GGUF_PY="$STRATA_REPORT_GGML/gguf-py"
"$STRATA_REPORT_PY" tools/iq_pack.py \
  --gguf "$STRATA_REPORT_MODEL/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf" \
  --out "$STRATA_REPORT_MODEL/pack" --compat-bf16
"$STRATA_REPORT_PY" tools/mtp_fetch.py fetch --out "$STRATA_REPORT_MODEL/mtp"
"$STRATA_REPORT_PY" tools/mtp_fetch.py verify --out "$STRATA_REPORT_MODEL/mtp"
"$STRATA_REPORT_PY" tools/mtp_pack.py --src "$STRATA_REPORT_MODEL/mtp" \
  --experts q2_0 --out "$STRATA_REPORT_MODEL/mtp/mtp-q2_0.gguf"
"$STRATA_REPORT_PY" tools/mtp_rt.py \
  --gguf "$STRATA_REPORT_MODEL/mtp/mtp-q2_0.gguf" --out "$STRATA_REPORT_MODEL/mtp/rt"
cp data/draft_vocab.bin "$STRATA_REPORT_MODEL/mtp/rt/draft_vocab.bin"
```

Verify the three GGUF hashes against `provenance/gguf-manifest.json`. Installed Python package versions are recorded in `provenance/python-packages.json`. The MTP fetcher in the pinned Strata checkout pins the Qwen source revision; the measurement's offline verification succeeded. Do not substitute another revision silently.

## Run

Stop other GPU inference services first and wait for their GPU mappings to be released. The runner controls only the Strata processes it creates; service restoration is the caller's responsibility. The measured machine's existing inference service was stopped for the measurements and restored afterwards.

The official needle script builds its corpus from source files, including `third_party/llama.cpp/docs`. Make `third_party/llama.cpp` refer to the pinned llama.cpp checkout before running; `provenance/needle-corpus-manifest.json` records the measured corpus. Do not overwrite an existing checkout or symlink. The exact requests are not included in this folder.

```bash
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1151-hipblaslt-100401.txt"
cmake --build "$STRATA_REPORT_BUILD" --target \
  hip_prefill_hcd_exact_parity hip_prefill_hipblaslt_gemm hip_gdn_rec_head -j 8
ctest --test-dir "$STRATA_REPORT_BUILD" \
  -R '^(hip_prefill_hcd_exact_parity|hip_prefill_hipblaslt_gemm|hip_gdn_rec_head)$' --output-on-failure
"$STRATA_REPORT_PY" bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/scripts/measure.py \
  --model "$STRATA_REPORT_MODEL" --build "$STRATA_REPORT_BUILD" \
  --sdk "$STRATA_REPORT_SDK" --out "$STRATA_REPORT_OUT" --port 18741
"$STRATA_REPORT_PY" bench/results/2026-10-09-community-gfx1151-ud-iq4xs-0.1.41/scripts/aggregate.py \
  --data "$STRATA_REPORT_OUT" --out "$STRATA_REPORT_OUT/analysis"
```

Each measurement starts a fresh Strata process except the repeated requests explicitly sharing a process. The runner records the effective launch arguments and `STRATA_*` environment for every arm. It waits for GPU mappings to be released before loading the next engine. Use a fresh result directory: the runner is not a resume tool and overwrites stage logs.

`quality.py` imports the existing P40 report's task generator, request settings and graders unchanged. It intercepts only execution of generated Python, placing it in a bubblewrap namespace without networking or access to the user's home, with CPU/address-space/file-size limits. `needle.py` adds request recording around the official needle script. `parallel.py` reproduces the approximate prompt length and concurrency conditions discussed in #1356 using public synthetic prompts; those are not the issue reporter's unpublished original prompts.

## Data fields

`raw/` holds only the small files listed below. The bulk raw data (per-request responses and grader output, needle and parallel requests with SSE chunks, engine logs, telemetry) are not in this folder, so `scripts/aggregate.py` needs a fresh run of `measure.py` to regenerate `analysis/`. The measured values are in `analysis/`.

- `analysis/summary.json`: quality counts by repetition/category, repeatability, speed statistics, needle results, parallel results and sampled memory. Repetitions `0` and `1` mean the first and second quality pass.
- `analysis/speed.csv`: one row per speed request. Token counts are tokens; `prefill_tok_s` and `decode_tok_s` are engine rates; `client_ttft_s` and `client_elapsed_s` are seconds. `drafted` and `accepted` are the engine's offered and accepted draft-token counts, not acceptance rounds.
- `raw/*-config.json`, `raw/*-env.json`, `raw/events.jsonl`: effective server configurations, explicit environment and stage commands/return codes. Unset gfx1151 switches retain the engine's architecture defaults.
- `raw/needles.json`, `raw/needles-native.json`, `raw/needle*.log`: the official needle script's results and logs.
- `analysis/quality-tasks.csv`: pass/fail, finish reason and generated tokens for each of the 69 tasks in both passes; `analysis/failures.json` has the full responses of the four first-pass failures.
- `raw/default-long/results.json`, `raw/fast-long/results.json`: the public V100 script's full per-request results, including the separate warm-up.

The commands in `raw/events.jsonl` show the folder's earlier name, `2026-10-09-community-evo-x2`; the folder was renamed before submission and the scripts were not changed.

Textual quality/engine timestamps use the machine's Asia/Tokyo local time. Numeric telemetry/event timestamps are Unix seconds.

Cancelled, failed and skipped runs are counted in `analysis/summary.json`. Throughput summaries do not turn them into successful runs.
