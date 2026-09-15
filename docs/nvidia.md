# NVIDIA / CUDA support — status

`edge0` does **not** run on NVIDIA through MLX, at any version tested:
the first forward pass fails, differently at each version (the table
below). It does run on NVIDIA through the torch backend in
`backends/cuda/` (`EDGE0_BACKEND=cuda`): both edge0-8b and edge0-35b
generate text on a GB10, layer for layer within 5.3e-7 (8b) and 3.7e-6
(35b, real weights) of MLX. The rest of this file is
that investigation, kept because it is what the next person attempting
this would otherwise repeat from scratch.

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
On our machine `cuInit()` started returning 999 a few minutes after
each boot while `nvidia-smi` still looked healthy. We first read that
as the driver dying after an aborted CUDA process. It was not: a
host-level cgroup device policy on that machine (unrelated to `edge0`
or MLX) denies `/dev/nvidia-uvm` and `/dev/nvidia-caps/*`, and the CUDA
runtime needs both. `nvidia-smi` goes through NVML on `/dev/nvidiactl`,
so it never notices. If `cuInit()` gives 999, first try opening
`/dev/nvidia-uvm` from the same shell. `EPERM` there means a device
policy, not a broken GPU.

## Bottom line

MLX's own CUDA backend does not close the gap at any version currently
available. The torch backend does: `EDGE0_BACKEND=cuda` runs both shipped
tiers on a GB10 and on CPUs, checked against MLX throughout (next
section). On the real weights, edge0-35b on the GB10 generates the same
32 greedy tokens as MLX run on its CPU device.

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
| the same engine on Linux aarch64 (DGX Spark, torch on the CPU), 32 greedy tokens of a chat prompt | MLX on Apple Silicon, same checkpoint (same file hashes): identical 32 tokens; 1.2 GB peak anonymous memory |
| everything above with torch on an accelerator: `EDGE0_TORCH_DEVICE=mps` (Apple GPU), whole suite including the slow engine test, plus the 32-token run | the same references: all pass, identical 32 tokens. On a device a tensor left on the host fails loudly, as it would on CUDA; this is what found `load_model` leaving init-time buffers (the rotary `inv_freq`) on the host |
| **the edge0-8b backbone on a real GPU** — a GB10 (`sm_121`, DGX Spark), torch 2.14+cu130, float32, every layer fed MLX's input for that layer | MLX on the Apple CPU device: **5.3e-7 max per layer** (median 2.6e-7), 4.3e-7 on the logits, same argmax. TF32 off, `float32_matmul_precision=highest` |
| the whole edge0-8b engine on that GPU, 32 greedy tokens | MLX on Apple Silicon: 31 of 32 tokens identical, diverging at step 31. Not a GPU artifact: torch on the CPU differs from MLX by the same 4.5% median per-step logit distance (the staged prerouter path), and that step's top-2 margin is smaller than that noise |
| `backends/cuda/_impl/qwen3_5_moe.py` (torch port of the edge0-35b backbone), every layer, chunked prefill + decode, on a small model MLX wrote in the published format (bf16, 4-bit, 8-bit router and shared gate) | the vendored MLX model on the MLX CPU device, float32: <= 2.6e-7 per layer, <= 4.1e-7 on the logits |
| the whole edge0-35b engine (`engine/qwen.py` unchanged: streaming, staged decode, the class-level prerouter patch) on that small checkpoint | the same engine on MLX: identical greedy tokens, per-step logits within bf16 noise and tracking MLX *with* the prerouter (the prerouter moves them 5-11%) |
| **the edge0-35b backbone on the real 23 GB checkpoint**, every layer, chunked prefill + decode (`test_qwen35_port_matches_mlx_on_real_weights`, needs `EDGE0_35B_MODEL`) | the vendored MLX model on the MLX CPU device, float32: <= 2.4e-6 per GatedDeltaNet layer, <= 3.7e-6 per gated full-attention layer, <= 2.4e-6 on the logits, same argmax; 40 layers, no missing or unexpected tensor |
| the whole edge0-35b engine on the real weights (LoRA 310 targets, prerouter 33 heads, streamed experts), 32 greedy tokens | MLX on the MLX CPU device: **identical 32 tokens**. Against MLX on Metal, 29 of 32 -- and MLX-Metal disagrees with MLX-CPU at exactly those three positions, so the flip is its GPU precision |
| **the edge0-35b engine on a real GPU** — the same GB10, torch 2.14+cu130, as a Slurm job | **identical 32 tokens** to both MLX-CPU and torch on the Mac's CPU (18.4 s) |
| **the edge0-35b backbone on that GPU**, real weights, float32, every layer fed MLX's own input for that layer | MLX on the Apple CPU device: **1.6e-6 max per layer** (median 2.7e-8) — 5.4e-7 across the 30 GatedDeltaNet layers, 1.6e-6 across the 10 gated full-attention ones — 5.1e-7 on the logits, same argmax; 0 missing / 0 unexpected tensors, 40 streamed expert layers |

Why the MLX *CPU* device: on some Apple GPUs MLX runs float32 matmul and
SDPA at reduced precision (an M5 Max measured 7.5e-4 from float64; MLX on
the CPU and torch both 2e-7). Against MLX on the GPU the port looks up to
1000x worse in the attention layers, all of it on the MLX side.

`load_model` handles what the published checkpoints actually contain: MLX
quantization of nearly every linear and embedding (kept 4-bit resident via
`QuantizedLinear` / `QuantizedEmbedding`), 8-bit routers, and, for
edge0-35b, MLX's sanitize (conv1d stored `[C, k, 1]`, five RMSNorm kinds
stored as `w + 1`), which it undoes.

## Speed, and where it goes (GB10, edge0-8b decode)

The backend was written as a correctness reference, and it shows: the
default profile decodes at **316 ms a token (3.2 tok/s)** on a GB10. A
profile of three decode steps says the GPU is barely working -- 63 ms of
CUDA against 643 ms of CPU -- so this is launch and copy overhead, not
arithmetic:

* **1681 host-to-device copies per token.** The streaming layer rebuilds
  expert bundles from the mmap every step, which is the whole point on a
  Mac with 36 GB and pointless on a 128 GB box. The shared LRU already
  fixes it: `cache_slots=3072` (from the tier's 64) keeps every expert
  resident after warm-up -> **206 ms a token (4.9 tok/s)**, 1.9 GB. A
  config change, no code.
* **235 quantized-linear calls per token**, dequantizing the same
  attention and lm_head weights every time (103 ms a token).
  `EDGE0_TORCH_WEIGHT_CACHE=1` keeps the dequantized weight instead ->
  **124 ms a token (8.1 tok/s)**, 6.0 GB resident.

Together that is **2.5x** over the default profile, with the same greedy
tokens. The two paths are bit-for-bit identical on the CPU
(`test_quantized_linear_weight_cache_is_exact`); on CUDA, keeping the
whole weight changes cuBLAS's split against the chunked path, so they
differ within the engine's own noise -- teacher-forced they pick the same
token at all 32 steps and sit the same distance from MLX (4.8% vs 5.3%
median per-step), while free-running they can diverge a step earlier.

### The syncs cost more than the arithmetic (Metal, edge0-8b decode)

The same CPU-bound shape shows up on this Mac's GPU (`EDGE0_TORCH_DEVICE=mps`),
which makes the overhead measurable without a GB10. There the default
profile decoded at **688 ms a token**, and a profile blamed three torch
calls inside `gather_qmm`: `nonzero` (2.4 s), `tolist` (1.9 s) and
`unique` (1.4 s) of an 8.2 s run -- against **0.07 s** for the matmuls
they were arranging. None of them is arithmetic: each is a device->host
sync, and the loop paid one per expert to ask *which* experts a call
needs and *where* each one's rows are.

Decode never needs to ask. One token routes to K experts, so the whole
call is a handful of rows: gather those rows' experts as one batch and
run a single `bmm`. No sync, and a repeated expert only costs a duplicate
dequantize. The per-expert loop still runs above that size (prefill),
where a dequantized copy per row is what blew memory up in the first
place -- now behind one host transfer of the index list instead of a
`nonzero` per expert. The cap is both a row count and a byte budget
(`EDGE0_TORCH_GATHER_BATCH`, `EDGE0_TORCH_GATHER_BYTES`), because one row
of a 35b expert is far bigger than an 8b one.

| edge0-8b decode, Metal (M5 Max), greedy | ms/token |
| --- | --- |
| tier default (`cache_slots=64`) | 688 |
| + batched decode gather (this change) | 401 |
| + `cache_slots=3072` | 384 |
| + `EDGE0_TORCH_WEIGHT_CACHE=1` | 291 |

**2.4x**, with the same 12 greedy tokens at every step. The GB10 numbers
above predate this and still need re-measuring: the Spark's GPU has been
fenced off since the 13-Sep reboot (`/dev/nvidia-uvm` EPERM, inside the
Slurm job as well as on the host).

What is still on the table: a fused dequantize+matmul kernel (the
elementwise shift/mask/mul/add chain is most of the remaining CUDA time),
captured graphs or `torch.compile` for the per-step launch storm, and
bundles built directly on the device rather than copied per step.

## What is left

* **Performance.** Decode no longer pays a host sync per expert (see
  above), but `gather_qmm` and the quantized linears still dequantize on
  every call and `core.compile` is eager: this is a correctness
  reference, not a fast path.

## A stale assumption this also corrects

`tests/test_repo_hygiene.py`'s module docstring says these hygiene tests
"can run on any platform — including the Linux CI job where MLX has no
wheels." That was true when written; it no longer is — MLX ships both a
CPU-only Linux wheel and CUDA-backed Linux wheels
(`manylinux_2_35_x86_64`/`aarch64`) as of at least `mlx==0.30.4`, the
exact version this repo already pins. Worth knowing regardless of
whether GPU runners are in reach for CI: the assumption that MLX-dependent
tests categorically cannot run in this repo's own CI is no longer correct.
