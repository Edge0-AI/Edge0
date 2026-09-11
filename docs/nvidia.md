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
the torch backend in `backends/cuda/`: its array/nn/quant ops and model
loading exist and are tested, while the streaming layer and engines
still reach into MLX directly and are the remaining work.

## A stale assumption this also corrects

`tests/test_repo_hygiene.py`'s module docstring says these hygiene tests
"can run on any platform — including the Linux CI job where MLX has no
wheels." That was true when written; it no longer is — MLX ships both a
CPU-only Linux wheel and CUDA-backed Linux wheels
(`manylinux_2_35_x86_64`/`aarch64`) as of at least `mlx==0.30.4`, the
exact version this repo already pins. Worth knowing regardless of
whether GPU runners are in reach for CI: the assumption that MLX-dependent
tests categorically cannot run in this repo's own CI is no longer correct.
