# Linux native engine

Build and serve Edge0's 8B and 35B models on Linux x86-64 using the same
[llama.cpp pin](../vendor.llama.pin) and common patch band as Windows.
The engine provides LoRA, advisory prerouter predictions, POSIX expert-page
prefetch and a cooperative process memory budget. This directory contains the
native engine and verification tools; a Linux desktop app is not included.

CPU/AVX2 builds and real-model serving have been verified. CUDA and Vulkan are
build options: Vulkan configuration passed with SDK 1.3.239; CUDA flag selection
was tested, but configuration stopped at the missing CUDA toolkit. Neither GPU
backend was built or run here. The quiet-host speed check remains pending.

## Build

Requirements: Linux x86-64 with AVX2, Bash, Git, `flock` (util-linux), CMake
(3.25 tested) and a C++17 toolchain. Conversion uses Python 3.10+, NumPy,
PyYAML and the pinned vendor's `gguf-py`; the verification scripts use Python
3.11+ (`hashlib.file_digest`). Models and converted artifacts need separate disk
space: allow at least 50 GB free before converting the 35B checkpoint.

Run from the repository root. This disposable Debian bookworm shell supplies
the tools used for the measurements below. Only the checkout and model/work
directory are mounted. Matching the checkout owner's UID/GID avoids Git's
ownership guard and root-owned build output. The container keeps Docker's
default network, so it cannot reach services that listen only on the host's
loopback; a second terminal reaches the server with `docker exec edge0-linux`.

```bash
mkdir -p "$HOME/.edge0"
docker run --rm -it --name edge0-linux \
  -e LOCAL_UID="$(id -u)" -e LOCAL_GID="$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/edge0/vendor/llama.cpp/gguf-py \
  -v "$PWD":/edge0 -v "$HOME/.edge0":/work \
  -w /edge0 debian:bookworm-slim bash -c '
set -e
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake git ca-certificates util-linux procps curl \
  python3 python3-numpy python3-yaml python3-pytest
exec setpriv --reuid="$LOCAL_UID" --regid="$LOCAL_GID" --clear-groups \
  bash --noprofile --norc
'
```

Inside that shell:

```bash
bash linux/scripts/vendor-build.sh --jobs 4
wt/linux/build-cpu/bin/llama-server --version
```

The script materializes pristine `vendor/llama.cpp` at
`7ab4ee7baad2d920464cbacfad4f4b07cf111fd2`, replays `common` then `linux`
patches with `git am --3way` in `wt/linux`, and stages `linux/serve/*` there.
It leaves the vendor checkout unchanged. Each invocation resets the disposable
`wt/linux` worktree; keep source edits in `linux/serve` or the patch band.
It builds Release server, bench and batched-bench targets with AVX2 and no
AVX-512, BLAS or GPU backend by default. Embedded UI downloads are disabled;
the HTTP API is available.

Optional invocations (choose one backend; its SDK must be installed):

```bash
bash linux/scripts/vendor-build.sh --assemble-only
bash linux/scripts/vendor-build.sh --vulkan --configure-only
bash linux/scripts/vendor-build.sh --cuda --configure-only
# Omit --configure-only to compile; GPU outputs use build-vulkan/build-cuda.
```

`--build-dir NAME` selects a directory under `wt/linux`; `--clean` removes only
that selected build directory. `EDGE0_DEPOT` selects a depot containing the pin
and patch bands, and `EDGE0_LLAMA_URL` selects an upstream mirror. Failures
return nonzero and log an error. See the [patch-band guide](../patches/llama.cpp/README.md).

## Convert and serve

Obtain the complete published [8B checkpoint](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
or [35B checkpoint](https://huggingface.co/Edge0/Edge0-35B-A3B-preview), including
config, tokenizer, model shards, LoRA and prerouter safetensors. Place it in
`~/.edge0/models/edge0-8b` or `~/.edge0/models/edge0-35b` on the host. The Docker
shell sees these under `/work/models/`. The following selects 8B; use `35b`
instead for the larger model. Native builds can set the model/output paths to
their own directories.

```bash
export EDGE0_TIER=8b
export EDGE0_MODEL=/work/models/edge0-$EDGE0_TIER
export EDGE0_CONVERTED="${EDGE0_MODEL}-gguf"
python3 windows/tools/convert_mlx_to_gguf.py \
  --dir "$EDGE0_MODEL" --out "$EDGE0_CONVERTED"
```

This existing converter runs the model repacker and
[`lora_mlx_to_gguf.py`](../windows/tools/lora_mlx_to_gguf.py), checks adapter
names/shapes against the base GGUF, and writes checksums plus `manifest.json`.
It reuses matching converted artifacts. It also writes the generated LoRA GGUF
beside the source checkpoint, so both model directories must be writable.
No MLX installation is needed for this conversion path.

Start the server in the container shell; stop it with Ctrl-C:

```bash
OMP_WAIT_POLICY=PASSIVE \
E0_PREROUTER="$EDGE0_MODEL/prerouter_edge0_${EDGE0_TIER}.safetensors" \
E0_PREFETCH=1 \
wt/linux/build-cpu/bin/llama-server \
  -m "$EDGE0_CONVERTED/edge0-$EDGE0_TIER.gguf" \
  --lora "$EDGE0_CONVERTED/lora_edge0_${EDGE0_TIER}-gguf.gguf" \
  --mem-budget-mb 12288 --no-repack \
  --host 127.0.0.1 --port 8080 -ngl 0 -t 8 -tb 8 \
  -c 16384 -b 512 -ub 512 -np 1 --no-warmup --jinja --cache-ram 0 -lv 4
```

Wait for the server's ready message. In a second terminal:

```bash
docker exec edge0-linux curl --fail http://127.0.0.1:8080/health
docker exec edge0-linux curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2 + 2? Reply with only the number."}],"temperature":0,"seed":4242,"max_tokens":64,"cache_prompt":false,"chat_template_kwargs":{"enable_thinking":false}}'
```

The 8B smoke returned HTTP 200 and `4`. LoRA loading and `[pref-trace] INIT ok`
plus positive `steps`, `heads`, `experts` and `advice` counters confirm the
features ran. Prerouter predictions issue page advice; they do not change the
model's selected experts or weights. A malformed sidecar logs rejection and
inference continues with the advisory feature disabled.

| Setting | Behavior |
|---|---|
| `--mem-budget-mb N` / `E0_MEM_BUDGET_MB` | Process RSS budget in MiB; a positive CLI value wins; CLI zero falls back to the environment. A budget that cannot be applied stops the server from starting. Pair it with `--cache-ram 0`: the prompt cache is memory the budget cannot reclaim. |
| `E0_PREROUTER` | Path to the matching model's prerouter safetensors. |
| `E0_PREFETCH=0` / `1` | Predictions/accounting only, or POSIX page advice. Unset defaults to accounting only. |
| `--pool-mb N` / `E0_POOL_MB` | Optional bounded immutable CPU expert cache, with mmap fallback; default zero. |
| `--no-repack` | Avoid permanent repacked expert copies, particularly with a small budget. |
| `OMP_WAIT_POLICY=PASSIVE` | Let idle OpenMP workers sleep; used for all reported comparisons. |

MiB values must be nonnegative decimal integers. With budget, sidecar and pool
settings absent, those optional hooks remain dormant. The optional expert cache
has unit coverage; it was left off in the real-model measurements below.

The budget measures whole-process RSS and cooperatively discards eligible
read-only model pages; protected mutable allocations are preserved. It polls
every 25 ms and checks during inference. When a pass cannot reach its target,
the next pass waits until RSS grows by 64 MiB or 2 s pass. RSS counters are
approximate and transient overruns are possible. Unreclaimable overhead can exceed the budget,
in which case an enforcement failure is logged. This is not a hard RSS ceiling.
Leave RAM for the OS and other processes; avoid `--mlock` with this controller.

## Verification

Run in the toolchain shell after the CPU build:

```bash
cmake -S linux/tests -B wt/linux/build-serve-tests
cmake --build wt/linux/build-serve-tests --parallel 4
ctest --test-dir wt/linux/build-serve-tests --output-on-failure
python3 -m unittest discover -s linux/tests -p 'test_*.py' -v
python3 -m pytest -q -o addopts= python/tests/test_repo_hygiene.py

# Optional ASan/UBSan suite:
cmake -S linux/tests -B wt/linux/build-serve-sanitize -DEDGE0_SANITIZE=ON
cmake --build wt/linux/build-serve-sanitize --parallel 4
ctest --test-dir wt/linux/build-serve-sanitize --output-on-failure

# Real 8B conversion, adapter, chat and deterministic-output smoke:
python3 linux/tests/e2e_server.py --model-dir /work/models/edge0-8b \
  --output-dir /work/bench/linux-smoke
```

The native suites last passed 38/38 each, including budget reclamation,
invalid inputs, protected pages, 8B/35B sidecar shapes, bounded expert ranges,
advice mode gates/failures, cache concurrency, worker teardown and RSS polling
cost. The real 8B smoke passed dormant/accounting/advice/corrupt-sidecar modes
with identical greedy tokens and unchanged GGUF bytes. Python framework tests
requiring MLX still need macOS; the hygiene tests run on Linux.

For a matched vanilla baseline and the pending quiet-host comparison:

```bash
mkdir -p /work/bench
# Create this pristine baseline worktree once.
git -C vendor/llama.cpp worktree add --detach /work/bench/llama-vanilla \
  "$(cat vendor.llama.pin)"
python3 linux/tests/build_baseline.py --source /work/bench/llama-vanilla \
  --build /work/bench/build-vanilla --jobs 4
python3 linux/tests/compare_server.py --model-dir /work/models/edge0-8b \
  --vanilla-source /work/bench/llama-vanilla \
  --vanilla-build /work/bench/build-vanilla \
  --output-dir /work/bench/linux-compare --rounds 3 \
  --require-quiet --diagnostic-controls
```

The harness matches the pin and CMake options, alternates three repetitions,
and compares dormant and active Linux decode against vanilla. The gate requires
at least 95% of vanilla with every sampled one-minute load below 4.
`--require-quiet` saves `UNDER_LOAD` and exits 2 when that condition fails.
LoRA-only/accounting-only controls help diagnose overhead; they cannot pass
the vanilla gate. The comparison is for the 8B model.

To reproduce the 35B conversion, 1/4/8-request runs and 4 GiB observation, use
`large_model_server.py` with the same baseline arguments:

```bash
python3 linux/tests/large_model_server.py --model-dir /work/models/edge0-35b \
  --vanilla-source /work/bench/llama-vanilla \
  --vanilla-build /work/bench/build-vanilla \
  --output-dir /work/bench/linux-35b --convert
```

This uses 12,288 MiB and 4,096 MiB budgets and stops its subprocesses if host
available RAM falls below 4 GiB. The small-budget smoke requests 64 book tokens
and one 128-token completion. Its peak uses sampled RSS and kernel `VmHWM`;
passing that workload does not establish a universal memory bound.

## Measurements on this CPU

Test system: Intel i9-9900K (8 cores / 16 threads, AVX2), 62 GiB RAM,
Ubuntu 24.04 host; Debian bookworm / GCC 12.2 in Docker. **All comparisons below
are provisional (under load).** They establish functional serving, not the
quiet-host 95% performance gate or a 30–100 tok/s single-request claim.

8B, 2026-11-04: medians of three alternating runs, 256 prompt tokens and 128
generated tokens per stream, context 16,384, batch/microbatch 2048/512, passive
OpenMP. Active means LoRA + prerouter + page advice, without a memory budget.
One-minute host load was 10.05–13.19.

| Threads / streams | Vanilla decode tok/s | Linux dormant | Linux active |
|---|---:|---:|---:|
| 8 / 1 | 16.96 | 16.88 | 14.52 |
| 8 / 4 | 33.33 | 36.47 | 31.03 |
| 8 / 8 | 37.84 | 40.05 | 37.25 |
| 16 / 1 | 12.51 | 12.59 | 9.61 |

35B, 2026-11-06: one observation per workload, 8 threads, 330 prompt / 128
generated tokens, eight slots, context 16,384 and batch/microbatch 512/512.
Both builds disable repacking. Linux active adds LoRA, prerouter, page advice
and a 12,288 MiB budget; vanilla has none of those features. Before/after load
was 7.04–15.34 across these engine runs, with further contention during requests.

| Concurrent requests | Vanilla decode tok/s | Linux active, 12 GiB |
|---|---:|---:|
| 1 | 8.47 | 5.58 |
| 4 | 14.25 | 11.63 |
| 8 | 9.39 | 8.77 |

Decode rates aggregate `predicted_n - 1` over the longest server decode time;
the first token belongs to prefill. These are separate workloads from the
mixed examples, whose wall throughput also includes prefill and HTTP.
Loaded timings cannot isolate feature cost from host contention.

The 35B GGUF is 21,749,830,528 bytes (20.26 GiB); conversion took 872.439 s
including checksums. Observed process peak RSS was 11,894.1 MiB with the
12,288 MiB budget and 3,890.4 MiB with 4,096 MiB. The latter decoded its
128-token request at 4.80 tok/s. The 12 GiB 400-token book reply had 11.952 s
TTFT and 6.11 decode tok/s, and reached its token cap.

The benchmark scripts save raw evidence (requests, timings, memory samples and
server logs) in their output directories.
The measured 35B runs kept at least 25.49 GiB host RAM available. Results on
other CPUs, contexts and storage require fresh measurements.

See the [root NOTICE](../NOTICE) for third-party attribution and the
[contribution guide](../CONTRIBUTING.md) for review requirements.
