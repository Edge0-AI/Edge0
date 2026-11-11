# Build and reproduce

Measured engine: official `v0.1.40.2`, commit
`e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`, **no local patches**.
Native SYCL Release/SPIR-V JIT, not AOT; oneAPI DPC++ 2026.1.0, oneMKL and
llama.cpp/ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
Compiler options include `-O3 -DNDEBUG -std=c++20 -fsycl`, per-kernel device-code
split, 32-lane subgroups, precise FP and correctly rounded FP32 divide/sqrt.
Exact CMake options are in build.json. Measured binary SHA256:
`8ca822304881a259412656ae7ba63fdb9d8e2c31ce18d9f3149bc8a57448f311`.
A rebuild's hash can depend on paths/toolchain.

With a compatible oneAPI/compiler/MKL installation already available, in an
isolated checkout:

```sh
git checkout e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a
export STRATA_ROOT="$PWD"
export REPORT=/absolute/path/to/this/report
git clone https://github.com/ggml-org/llama.cpp.git /absolute/path/to/llama.cpp
git -C /absolute/path/to/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
export STRATA_GGUF_PY=/absolute/path/to/llama.cpp/gguf-py
source /opt/intel/oneapi/setvars.sh
python3 -m venv .venv-b65
.venv-b65/bin/pip install -r requirements.txt
export PATH="$STRATA_ROOT/.venv-b65/bin:$PATH"
cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j4
```

The portable scripts require Python 3.11+ (`hashlib.file_digest`); the measured
host used Python 3.14.4, with dependency versions listed in system.json. Install neither a driver nor another runtime to reproduce
this report's software comparison. Choose a working compatible driver for your
own system and record its version.

## Retained artifacts

Both GGUF shards are original full 512-expert IQ2_XS from
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09`,
under `IQ2_XS/`. The dense pack/tokenizer were **retained, not repacked for the
new release**. They were prepared by `tools/iq_pack.py` at Strata commit
`6f32ec070f23ced9f50e704d854d775da52591ab` (script SHA256
`0a79ac86f63d9b36371d37643952592770fe5cc2e4e45fff0c63f5c023c6de26`).
Use that preparation checkout when reproducing the exact artifacts. The release
adds two native PLE formats to that tool's format set; do not assume a fresh
release pack has the recorded hashes.

The original Q2_0 MTP draft is from
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
`prepare-mtp.py` uses upstream verification of all 31 tensors and forbids a
floating-main fallback. The MTP pack/runtime preparation tools match the
original preparation versions. No custom draft vocabulary or ranking;
`data/expert-profile.bin` matches this release byte for byte.

```sh
export BENCH_ASSETS=/absolute/path/to/benchmark-assets
export PREPARATION_ROOT=/absolute/path/to/strata-preparation-checkout
git clone https://github.com/Niko1221/Strata.git "$PREPARATION_ROOT"
git -C "$PREPARATION_ROOT" fetch origin 6f32ec070f23ced9f50e704d854d775da52591ab
git -C "$PREPARATION_ROOT" checkout --detach FETCH_HEAD
python "$PREPARATION_ROOT/tools/iq_pack.py" \
  --gguf "$BENCH_ASSETS/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" \
  --out "$BENCH_ASSETS/pack"
python "$REPORT/prepare-mtp.py" --source "$STRATA_ROOT" --assets "$BENCH_ASSETS"
export MODEL_DIR="$BENCH_ASSETS/models/IQ2_XS"
export PACK_DIR="$BENCH_ASSETS/pack"
export MTP_DIR="$BENCH_ASSETS/mtp-rt"
export EXPERT_PROFILE="$STRATA_ROOT/data/expert-profile.bin"
```

Verify sizes/hashes against artifacts.json. On the measured host, large artifacts
were previously hashed and root-sealed; ownership and sealed inode/mtime/size
were rechecked. Engine/profile/environment were freshly SHA256-checked.
No large model or pack is included in this PR.

## Run

Use an independent bounded supervisor with exclusive GPU ownership, enough
pinned-memory allowance and RAM-backed logs. The measured host disabled swap,
core dumps and crash capture during generation, restoring services and capture
after verified cleanup. These portable scripts do not configure those host
policies or stop other model owners.

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0
export SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt
export STRATA_MIRROR_MIB=16384 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1
export STRATA_WARM_GRAPHS=0
export STRATA_STAGER_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset STRATA_DBG_NAN STRATA_VERIFY_EAGER STRATA_QFUSE STRATA_DECODE_TIMING STRATA_PLE_TRACE
sudo install -d -m700 -o "$(id -un)" /run/strata-community
python "$REPORT/benchmark.py" --source "$STRATA_ROOT" \
  --profile "$REPORT/profile-8k-512.json" --ram /run/strata-community \
  --out /absolute/path/to/new-main-512 --runs 3
export STRATA_TEST_RAM=/run/strata-community
export STRATA_TEST_PROFILE="$REPORT/profile-8k-512.json"
export BENCH_OUTPUT=/absolute/path/to/new-public-512
python "$REPORT/public-fixtures.py" matched
```

Repeat with `profile-8k-4096.json` and `profile-262k-512.json`, using distinct
output directories. The input shapes and output caps stay unchanged at 262K;
that suite measures the larger configured memory budget, not a full-window prompt. Profiles expand
the exported path variables. Main fixtures/instructions are embedded in
benchmark.py. Public token-ID fixtures are `benchy-short.ids` and
`benchy-long.ids`; the helper also retains the same task-shaping library in
frozen-community.py. It adds `STRATA_TRACE=1`, matching the measured public suite.
Both suites store counts/timings/hashes, not generated text. Keep native logs in
RAM through clean QUIT and delayed-fault observation, then remove them.

Report fields use seconds for `ttft_s`/`total_s`, milliseconds for native
`prompt_ms`/`decode_ms`, and tokens/s for `prompt_tps`/`decode_tps`. Main suite
`measured=false` rows are warmups or canaries. Public-suite canaries are excluded
from tables. Decode is `generated*1000/decode_ms`; prompt is
`prompt_read*1000/prompt_ms`. Request elapsed time is never used as decode time.
