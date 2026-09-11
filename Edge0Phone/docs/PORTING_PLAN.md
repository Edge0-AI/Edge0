# Edge0 8B -> iPhone port

Target: Edge0/Edge0-8B-A1B-preview (Ling 3.0 tiny / bailing hybrid) using MLX Swift.

## Architecture we must preserve

- 24 layers in groups of `[KDA, KDA, KDA, MLA]`.
- Layer 0 uses a dense SwiGLU MLP; layers 1-23 use sparse MoE.
- 128 routed experts, top-8, plus one resident shared expert.
- Router: sigmoid scores + expert bias for selection; 8 groups, keep 4 groups by top-2 group sum; top-8 experts; weights come from *unbiased* sigmoid scores, normalized and scaled by 2.5.
- Routed experts are int4 affine, group size 64.
- KDA uses causal depthwise conv (kernel 4) and Ling's safe-gate delta recurrence.
- MLA uses q-LoRA 256, kv-LoRA 512, 128 NoPE + 64 RoPE dimensions, interleaved RoPE, theta 6e6, and head-wise output gating.
- Recover-LoRA and prerouter are separate safetensor adapters and should remain unmerged.

## Milestones

### M0 — storage + router probe (validated)

1. Parse safetensors header only.
2. mmap `model.safetensors` read-only.
3. Address a single expert's weight/scales/biases by byte range.
4. Reproduce grouped routing on CPU and MLX.
5. Confirm process resident memory does not jump by ~4.5 GB.

Exit criterion: `edge0-probe` successfully addresses layer 1 / expert 0 and reports tensor shapes.

### M1a — expert/MoE numerical baseline (this delivery)

The continuation request prioritizes expert execution before the resident backbone.
Implemented affine INT4 linear, SwiGLU, true grouped gate, sequential selected-expert
execution, and the shared expert. Tests include an independent scalar oracle and
real checkpoint parity with Edge0 Python in Float32. See VALIDATION.md for evidence
and remaining gates. This changes the order of work, not the eventual architecture.

### M1b — resident backbone (remaining)

Port the non-streamed pieces into `BailingHybrid.swift`:

1. configuration + embedding + RMSNorm + lm_head;
2. dense SwiGLU;
3. MLA;
4. KDA using the existing mlx-swift-lm gated-delta implementation as the reference;
5. KV/KDA cache creation and layer schedule.

For M1, temporarily use a tiny synthetic checkpoint or selected layers. Do **not** load all routed experts.

Exit criterion: Swift and Python produce closely matching hidden states for a tiny deterministic fixture.

### M2 — exact streaming MoE

Integrate the M1a `StreamingMoE` path with the full decoder:

1. run the true gate;
2. transfer only selected expert IDs to CPU;
3. map selected expert slices;
4. run int4 quantized matmul for up/gate/down;
5. weighted sum + shared expert;
6. LRU cache expert bundles (remaining).

Performance is not the goal yet.

Exit criterion: one-token logits agree with Edge0 Python within a defined tolerance.

### M3 — Recover-LoRA

Port parallel LoRA loading from `lora_edge0_8b.safetensors` without merging the base weights.

Exit criterion: logits and a short greedy generation match the Python Edge0 pipeline closely.

### M4 — prerouter + staged double buffer

Port the trained prerouter:

- fc1 -> exact erf GELU -> fc2 + linear_init;
- feature vector includes hidden + current top-k one-hot + previous-token top-k one-hot;
- one-token / one-layer shift;
- preload predicted top-8 experts into fixed slots asynchronously.

Exit criterion: no expert drops, better decode throughput than M2.

### M5 — iOS shell and device profiling

- Minimal SwiftUI chat app.
- Model directory imported/stored in app Documents/Application Support.
- Run on physical iPhone, not simulator.
- Track peak resident memory, jetsam risk, TTFT, decode tok/s, storage reads, and thermal state.
- Start with context <= 512 tokens; increase only after memory measurements.

## First technical risk

MLX Swift's standard safetensors loader exposes whole-file tensor loading. Edge0 requires byte-range expert access. The scaffold therefore parses the safetensors index itself and mmaps the checkpoint. M0 deliberately copies only a selected expert slice into MLX, which preserves bounded active memory while avoiding assumptions about Metal compatibility of a plain file-backed pointer. After correctness parity, benchmark a Metal-compatible zero-copy/staging path; if ordinary mmap cannot feed Metal efficiently, use an explicit staging buffer or a small C/MLX IO bridge rather than abandoning the architecture.
