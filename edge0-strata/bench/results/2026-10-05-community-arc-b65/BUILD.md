# Diagnostic-off source build and reproduction

This repeats the original build/profile with only STRATA_DBG_NAN absent.


The measurements use Strata v0.1.40 source (Intel engine label 0.1.39-sycl) at
`1735d6471df29b42c26170efaac1f1446a58640f`, ggml/llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, and Intel oneAPI DPC++ 2026.1.0.
This is a SPIR-V/JIT Release build, not AOT. Compiler options include `-O3`,
`-DNDEBUG`, `-std=c++20`, `-fsycl`, subgroup 32, per-kernel device code split,
`-fp-model=precise`, sequential MKL, and correctly rounded FP32 divide/sqrt.
The measured binary SHA256 is
`520d1a72a7866956efc0feb4250cafeaf485ef86904e11950a392468e1bb91cb`.
Paths and build environment can affect a rebuilt binary's hash.

Six local patches are attached as reproduction data, with no changes to the
repository engine outside this benchmark folder:

- `0003`: matching free for pinned allocations in a test fixture.
- `0004`: eager commit semantics (eager is off in this benchmark).
- `0005`: refuse startup when NO_HOST lacks complete expert mirror coverage.
- `0006`: drain and explicitly release the USM expert mirror before QUIT exits.
- `0007`: synchronize three newer shared interfaces: fused GR return semantics,
GGUF reader ready-pointer argument, and GEMM input strides.
- `0008`: synchronize remaining linker interfaces, file advice/Linux release
semantics and unavailable elastic-KV stubs. New optional fusion/stride/error-buffer/
deferred-reader paths that are not ported are explicitly refused. This benchmark
uses the existing default kernel paths; it does not claim those new features.

The source is the v0.1.40 release at 1735d6471df29b42c26170efaac1f1446a58640f,
not a full re-migration of the Intel kernels. The Intel CMake project still labels
its engine 0.1.39-sycl. Older thread-affinity and NativeDense build fixes are already
in this tag and are not reapplied. The tagged source otherwise fails compile/link
against its newer shared headers; the repairs are disclosed rather than calling
this an unmodified release. All six patches were applied before a full clean
SYCL rebuild, not linked into old production objects.

With compatible oneAPI/compiler/MKL already installed, in a separate checkout:

```sh
git checkout 1735d6471df29b42c26170efaac1f1446a58640f
export REPORT=/absolute/path/to/this/report
export STRATA_ROOT="$PWD"
git apply "$REPORT"/patches/*.patch
git clone https://github.com/ggml-org/llama.cpp.git /absolute/path/to/llama.cpp
git -C /absolute/path/to/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
source /opt/intel/oneapi/setvars.sh
python3 -m venv .venv-b65
.venv-b65/bin/pip install -r requirements.txt
export PATH="$STRATA_ROOT/.venv-b65/bin:$PATH"
cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_GGML_DIR=/absolute/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j4
```

Model preparation uses the original full 512-expert IQ2_XS, **not** Coder:

- `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
  `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, both files under `IQ2_XS/`.
- Draft tensors: `Qwen/Qwen3.8-Flash-Next` at
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`, upstream 31-tensor hash
  verification, Q2_0 expert packing. `prepare-mtp.py` forbids fallback to main.
- `tools/iq_pack.py` creates the dense/native-expert pack and exports the
  model's tokenizer/template. Full fixed `data/expert-profile.bin` is retained;
  no custom expert ranking or draft vocabulary is used.
- See `artifacts.json` for filenames, byte sizes and recorded hashes. Large
  weights/packs were previously hashed and root-sealed; their size/mtime/inode
  and ownership were rechecked before this campaign. The measured binary,
  profile and launch environment were freshly SHA256-checked.

Place the pinned GGUF shards under `$BENCH_ASSETS/models/IQ2_XS/`, then:

```sh
export BENCH_ASSETS=/absolute/path/to/benchmark-assets
python tools/iq_pack.py \
  --gguf "$BENCH_ASSETS/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf" \
  --out "$BENCH_ASSETS/pack"
python "$REPORT/prepare-mtp.py" --source "$STRATA_ROOT" --assets "$BENCH_ASSETS"
```

Run under your own bounded exclusive-GPU supervisor, with adequate pinned-memory
limits and a RAM-backed directory. The measured machine had swap disabled and
crash capture suspended during generation; independent cleanup restored normal
services and crash capture afterward. This script does not configure those
machine policies or stop other services for you.

```sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0
export SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt
export STRATA_MIRROR_MIB=16384 STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1
export STRATA_WARM_GRAPHS=0
unset STRATA_DBG_NAN
export STRATA_STAGER_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset STRATA_VERIFY_EAGER STRATA_DECODE_TIMING STRATA_PLE_TRACE
sudo install -d -m700 -o "$(id -un)" /run/strata-community
python "$REPORT/benchmark.py" --source "$STRATA_ROOT" \
  --profile "$REPORT/profile.json" --ram /run/strata-community \
  --out /absolute/path/to/new-results --runs 3
```

Native logs are RAM-backed and should be deleted only after clean native exit and independent
delayed-fault checks. The script stores numeric/hash receipts, not output text.
To regenerate summary/CSV/allowlisted timing logs:

```sh
python "$REPORT/summarize.py" /absolute/path/to/new-results /absolute/path/to/summary
```

## Supplemental matrix-stride check

The interface repair's matrix strides were checked separately with
[gemm-stride-check.cpp](gemm-stride-check.cpp): contiguous/padded inputs, beta0/1,
and full/sliced native Q8 matrices. Twelve small synthetic cases passed on the
B65. To rebuild it after the engine build, with oneAPI and ninja on PATH:

```sh
python build-stride-check.py --source "$STRATA_ROOT" --build "$STRATA_ROOT/build-sycl" --out /absolute/path/to/test-build
```

Run the resulting gemm-stride-check binary under the same exclusive-GPU/resource
guards. This is a small matrix check, not another model benchmark.
