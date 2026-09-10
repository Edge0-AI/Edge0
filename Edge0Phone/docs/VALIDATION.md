# Validation record — M0 + M1a

Verified locally 2026-09-09, macOS 27.0 (26A5425a), Apple Silicon arm64,
Xcode 26.6 (17F113), Apple Swift 6.3.3, installed Metal Toolchain.

## Result

| Check | Result |
|---|---|
| Swift/C++ build, all targets and tests | PASS |
| Xcode macOS validator + Metal shaders | PASS |
| CPU unit tests | 7/7 PASS |
| Metal unit tests | 7/7 PASS |
| Real model M0 probe | PASS, 4,508,697,328 bytes, 1,196 tensors |
| Swift Metal vs Python, layer 1 / expert 0 | PASS, maximum absolute error 0 |
| Swift Metal vs Python, layer 12 / expert 127 | PASS, maximum absolute error 0 |
| Swift Metal vs Python, layer 23 / expert 55 | PASS, maximum absolute error 0 |
| Swift CPU vs Python Metal, layer 1 / expert 0 | PASS, maximum absolute error 1.1324883e-6 |
| Deliberately corrupted expected MoE output | Rejected, exit 1, one failing element |
| Generic iOS arm64 library + shaders, signing disabled | BUILD SUCCEEDED |
| Physical iPhone / signed application installation | PASS, iPhone 16 on iOS 27.0 |
| Physical iPhone / checkpoint transfer | PASS, real checkpoint copied to app Documents |
| Physical iPhone / M0 storage addressing | PASS, config + mmap + nine layer 1/expert 0 tensors |
| Physical iPhone / M1 Metal MoE | PASS, layer 1, eight routed INT4 experts + shared expert, finite 1536-wide output |

These are **real checkpoint weights with deterministic synthetic hidden vectors**.
They are not captured activations from a complete model. Comparison arithmetic is
Float32, including promotion of checkpoint BF16 coefficients. No adapters were applied.
Exact equality on these Metal cases is observed evidence, not a universal guarantee.

## Numerical boundary

Each fixture checks all 512 up-projection values, all 512 gate-projection values,
all 1536 expert SwiGLU/down outputs, eight selected IDs and weights, and all 1536
complete MoE output values (routed + shared). The ID comparison is order-independent,
and compares weights by expert ID. CPU and GPU argpartition order can differ.
Exact-score tie selection across implementations is not guaranteed.

Projection/expert/block acceptance is elementwise
`abs(actual-expected) <= 2e-4 + 2e-4*abs(expected)`.
Router weights use absolute and relative tolerances of `1e-6`.
Nonfinite values fail, even if present in the expected fixture.
The negative control changes one expected MoE value by 1 and correctly exits 1.

CPU case maximum absolute errors:

- up: 1.1324883e-6
- gate: 9.536743e-7
- expert: 7.4505806e-7
- router: 5.9604645e-8
- complete MoE: 7.748604e-7

Unit coverage includes hand-packed nibbles, two quantization groups with distinct
scales/offsets, BF16 coefficient decoding/promotion, a non-square linear with two
inputs, SiLU gating, top-1 normalization, selection-only expert bias, group dropping
and no-drop routing, malformed dimensions, invalid expert/layer/batch rejection,
shared-expert addition, and tensor-copy ownership after the source mmap closes.
The synthetic MoE fills unselected experts' scales with NaNs to detect unintended
expert execution. It does not measure I/O volume or prove memory residency bounds.

## Reproduce

See README and `scripts/test.sh`. The latter stages the Xcode-produced Metal
library next to SwiftPM's executable **and inside its XCTest bundle**; placing it
only in the build directory is insufficient for the XCTest process.
Run `EDGE0_TEST_GPU=1 scripts/test.sh` for Metal unit coverage.

Real-weight reference fixtures and compact logs are included under `fixtures/`
and `evidence/`. After building the validator:

```sh
.build-xcode/Build/Products/Debug/edge0-m1-validate /path/to/model docs/fixtures/reference-layer1.json
.build-xcode/Build/Products/Debug/edge0-m1-validate /path/to/model docs/fixtures/reference-layer12.json
.build-xcode/Build/Products/Debug/edge0-m1-validate /path/to/model docs/fixtures/reference-layer23.json
.build-xcode/Build/Products/Debug/edge0-m1-validate /path/to/model docs/fixtures/reference-layer1.json --cpu
xcodebuild build -scheme Edge0MLX -destination 'generic/platform=iOS' \
  -derivedDataPath .build-ios -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
```

The reference was generated with Python 3.14, Edge0's pinned MLX 0.30.4 and mlx-lm
0.31.0. Swift uses MLX Swift 0.31.6 (revision in Package.resolved). The implementations
have different version numbers; the recorded parity runs verify these combinations.
The generator imports the actual Edge0 BailingGate and BailingMLP, and upstream
mlx-lm SwitchGLU with only the selected experts instantiated. It does not duplicate
the Swift sequential expert accumulation implementation.

Checkpoint safetensors header SHA-256:
`7ef369fc2aeffe2daaa7d7f690317bb5ccf8121aadccc58090b1ef45e177f1f4`.
This identifies the header, not a full weight-file checksum. No weights are included.

## Build issues resolved during this run

The initial plain SwiftPM build compiled M0 and passed its two Core tests. MLX
runtime tests initially failed because SwiftPM does not produce Metal shaders.
Xcode then reported a missing Metal Toolchain; installing its component and building
with Xcode resolved that. Generated package test schemes were unreliable in this
Xcode version, so the reproducible path builds shaders with Xcode and runs the
SwiftPM test bundle with the colocated library. Initial failures are not counted as passes.

## Sources and remaining gates

- [Edge0 Bailing implementation at verified revision](https://github.com/Edge0-AI/edge0/blob/0700e6532f45e0d0d99e9c588d7d8cd240538ea0/src/edge0/backends/mlx/_impl/bailing_hybrid.py)
- [MLX Swift 0.31.6 quantized operations](https://github.com/ml-explore/mlx-swift/blob/0.31.6/Source/MLX/Ops.swift)
- [MLX Swift build guidance](https://github.com/ml-explore/mlx-swift/blob/0.31.6/README.md)
- [Checkpoint](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)

M0's prior user-reported real-checkpoint validation was independently repeated here.
M1a is complete at the requested expert/MoE boundary and has physical execution
evidence on an iPhone 16 running iOS 27.0. The resident backbone, attention/KDA/MLA
state, full-model logits, tokenizer/generation, Recover-LoRA, expert cache,
prerouter, BF16 performance path, phone-vs-Python numerical parity, and physical
memory/thermal/latency measurements remain open.

One original expert occupies 1,327,104 bytes. Sequential evaluation bounds live
routed-expert graphs, but allocator caching, promoted coefficients, shared weights,
and file-backed resident pages mean peak memory must still be measured separately.
