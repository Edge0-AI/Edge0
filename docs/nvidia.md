# NVIDIA / CUDA support — investigation status (does not work yet)

This is not a how-to. `edge0` does **not** run end-to-end on NVIDIA
hardware today, on any tested MLX version. The project's own README
already says so plainly: *"the MLX backend runs on macOS with Apple
Silicon (M1/M2/M3/M4). The CUDA backend is on the roadmap — no other
platforms are supported yet."* `backends/cuda/` now holds a torch
reference backend (`EDGE0_BACKEND=cuda`) whose ops are checked against
real MLX in `tests/test_cuda_backend.py`, but it is not wired end-to-end
yet — see "Bottom line".

What follows is what we found trying anyway, kept here because it's
exactly the investigation the next person attempting this would
otherwise have to repeat from scratch.

## Environment

- Hardware: DGX Spark, GB10 (`sm_121`, Grace-Blackwell, 128GB unified memory)
- Toolkit: CUDA 13.0 — use `mlx[cuda13]`, not `mlx[cuda12]`; the latter
  ships NVRTC 12.9, which does not compile against the CUDA 13 headers
  (`cuda_fp6.hpp` / `cuda_fp4.hpp`) on this platform.
- Tier tested: edge0-8b (Ling / Bailing hybrid)

## The failure chain (three MLX versions, three distinct errors)

Everything up to the model finishing construction works cleanly at
every version tested: install, checkpoint download, LoRA/prerouter
attach (`lora applied=153 not_found=0`, `prerouter installed: 16 heads`,
built in 0.7s). The failure is always at the first forward pass, and it
moves as the MLX version moves — meaning this is not one bug, it's the
edge of MLX's own CUDA-quantized-op support arriving in stages, with
`edge0`'s code (written and pinned against `0.30.4`) landing in a
different gap each time:

| `mlx` version | Result |
|---|---|
| `0.30.4` (this repo's current pin) | `RuntimeError: QMM NYI` — quantized matmul has no CUDA implementation at all |
| `0.31.1` | `GatherQMM has no CUDA implementation` — the gather-variant specifically still missing |
| `0.32.0` / `0.32.2` | Both ops now present — clears `GatherQMM`, advances from `ling.py:181` to `ling.py:190`, then `IndexError: SmallVector out of range` inside `core.eval(logits)` |

No CPU fallback was viable either: `mlx-cpu==0.30.4` fails to JIT on
g++13/aarch64, and `mlx-cpu==0.32.2` raises `There is no Stream(cpu, 3)`.

The `0.32.x` failure is the most interesting one and the most worth a
second look by someone who wants to pursue this further: both required
CUDA kernels are confirmed present, the crash is downstream inside
`core.eval`, and it reproduced identically on two separate versions —
consistent with a real, narrow incompatibility between `edge0`'s
`ling.py` code path (written for `0.30.4`'s API) and something that
changed by `0.32.x`, not with a fundamentally unsupported operation.

## A reliability caveat, independent of all of the above

`mx.default_device()` reporting `Device(gpu, 0)` is **not** evidence the
GPU is usable — MLX is lazy and this call never touches the driver. It
reported `gpu` in every run above, including the ones where the GPU was
later confirmed dead. The real signal is whether `cuInit()` succeeds.
On this hardware specifically we also hit a driver-level issue
unrelated to `edge0` or MLX: a CUDA process that aborts can leave
`cuInit()` failing (error 999) for every subsequent process, with no
root-level recovery (`nvidia_uvm`'s refcount stays stuck; `rmmod` fails
even as root) — only a reboot clears it. This did not happen after
every abort in our runs, so it is state-dependent, not a strict rule;
flagging it because `nvidia-smi` does not surface it (it goes through
NVML, not the CUDA runtime).

## Bottom line

Running `edge0` today means Apple Silicon + `mlx-metal`, per the
project's own stated support matrix. MLX's own CUDA backend does not
close the gap at any version currently available, so the path forward is
the torch backend in `backends/cuda/` (`EDGE0_BACKEND=cuda`).

## Torch backend: what exists and how it is checked

MLX is the ground truth throughout: the checks run on Apple Silicon, where
both backends are available, and compare the torch side against MLX
(`tests/test_cuda_backend.py`, `tests/test_backend_parity.py`; the latter
runs each case under both backends in subprocesses).

| Piece | Checked against |
|---|---|
| `quant.gather_qmm` (2/4/8-bit affine, all broadcast shapes the streaming layer uses) | `mx.gather_qmm` |
| `core` ops at their real call sites (routing, sampling), `nn.RMSNorm` / `gelu` | the same code on MLX: identical expert choices, identical sampler masks |
| `StreamingSwitchGLU`, every path (exact, whole-layer, hot, staged), on layer 1 of the real edge0-8b checkpoint (`EDGE0_8B_MODEL`) | MLX on the same inputs: 1.2-1.5% of output scale, about two bf16 ulps |
| `io.load_model` + `install_streaming_experts` on a small checkpoint in the exact published edge0-35b format | the source model on the same weights: 4e-7 relative, same argmax |
| `backends/cuda/_impl/bailing_hybrid.py` (torch port of the edge0-8b backbone) on the real checkpoint, every layer, chunked prefill + decode | the vendored MLX model on the MLX CPU device, float32: <= 1.5e-6 per layer, <= 1.7e-6 on the logits |
| the whole edge0-8b engine (`engine/ling.py` unchanged: LoRA, prerouter-staged decode, streaming, sampling) | the same engine on MLX: identical greedy tokens (`pytest -m slow`) |

Why the MLX *CPU* device: on some Apple GPUs MLX runs float32 matmul and
SDPA at reduced precision (an M5 Max measured 7.5e-4 from float64; MLX on
the CPU and torch both 2e-7). Against MLX on the GPU the port looks up to
1000x worse in the attention layers, all of it on the MLX side.

`load_model` handles what the published checkpoints actually contain: MLX
quantization of nearly every linear and embedding (kept 4-bit resident via
`QuantizedLinear` / `QuantizedEmbedding`), 8-bit routers, and, for
edge0-35b, MLX's sanitize (conv1d stored `[C, k, 1]`, five RMSNorm kinds
stored as `w + 1`), which it undoes.

## What is left

* **edge0-35b engine.** `engine/qwen.py` drives the vendored MLX model
  (per-layer callbacks, mlx-lm caches, class-level prerouter patch) and
  refuses other backends. The edge0-8b route applies: port the vendored
  `_impl/qwen3_5_moe.py` / `qwen3_next.py` to torch with the same API and
  check it layer by layer against MLX, so the engine runs unchanged. (The
  transformers `Qwen3_5MoeForCausalLM` also loads and streams, see above,
  but would need its own engine glue.)
* **A real-weight run of edge0-35b** (23 GB) through the torch path; the
  format test above uses a small model written in the same format.
* **Real NVIDIA hardware.** Everything above was checked on Apple Silicon
  with torch on the CPU (the only machine that has both backends); on a
  CUDA device the same code runs with `DEVICE = cuda`, untested there yet.
* **Performance.** `gather_qmm` and the quantized linears dequantize on
  every call and `core.compile` is eager: this is a correctness reference,
  not a fast path.

## A stale assumption this also corrects

`tests/test_repo_hygiene.py`'s module docstring says these hygiene tests
"can run on any platform — including the Linux CI job where MLX has no
wheels." That was true when written; it no longer is — MLX ships both a
CPU-only Linux wheel and CUDA-backed Linux wheels
(`manylinux_2_35_x86_64`/`aarch64`) as of at least `mlx==0.30.4`, the
exact version this repo already pins. Worth knowing regardless of
whether GPU runners are in reach for CI: the assumption that MLX-dependent
tests categorically cannot run in this repo's own CI is no longer correct.
