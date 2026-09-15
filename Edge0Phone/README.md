# Edge0Phone — M1 expert/MoE slice

Experimental Swift/MLX port of Edge0's 8B-A1B model for eventual iPhone inference.
**M0 storage addressing and M1's single-token MoE block are validated on a physical
iPhone. This is not yet a text generator.**

## Implemented

- Header-only safetensors indexing, read-only mmap, selected expert copies.
- Affine INT4/group-64 linear operation through MLX `quantizedMM`, with shape/dtype validation.
- Quantized SwiGLU: `down(silu(gate(x)) * up(x))`.
- Grouped sigmoid routing, selection-only expert bias, top-K normalization/scaling.
- One-token `StreamingMoE`: routed experts plus the unconditional shared expert.
- Sequential expert evaluation to avoid retaining all routed weight graphs.
- Scalar-oracle, routing, shared-expert, mmap-copy lifetime, and invalid-input tests.
- Python reference generator and Swift parity validator using real checkpoint slices.
- A signed iPhone probe app that validates M0 storage access and executes M1 on
  the phone GPU with the real checkpoint.

M1 arithmetic is explicitly **Float32**, including promotion of BF16 scales, offsets,
router weights, and input. INT4 weights stay packed. This isolates layout/math correctness;
BF16 production execution, its tolerances, performance, and memory remain future work.

## Requirements and build

Apple Silicon Mac, Xcode with the Metal Toolchain component, Swift 6.1 or later.
The package declares macOS 14 / iOS 17. `Package.resolved` records tested dependencies.
The iOS declaration alone does not establish device compatibility.

```sh
# From the extracted Edge0Phone directory:
swift run edge0-probe /path/to/Edge0-8B-A1B-preview

# Install only if Xcode reports that the Metal compiler is missing:
xcodebuild -downloadComponent MetalToolchain

xcodebuild build -scheme edge0-m1-validate -destination 'platform=macOS' \
  -derivedDataPath .build-xcode -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
```

MLX's bundled build plugin must be trusted (the command above accepts it). Plain
`swift build` compiles the Swift/C++ code but **does not compile the Metal shaders**.
See `scripts/test.sh` for building the shaders and running the unit suite on CPU or GPU.

## Real checkpoint parity

Use a local copy of `Edge0/Edge0-8B-A1B-preview`; no checkpoint is included in this archive.
Use Python 3.10 or newer (the macOS system Python 3.9 is insufficient).
Create an isolated Python environment and install the reference at the recorded revision:

```sh
git clone https://github.com/Edge0-AI/edge0 reference-edge0
git -C reference-edge0 checkout 0700e6532f45e0d0d99e9c588d7d8cd240538ea0
python3 -m venv .venv
.venv/bin/pip install ./reference-edge0
.venv/bin/python scripts/generate_reference.py /path/to/model reference.json
.build-xcode/Build/Products/Debug/edge0-m1-validate /path/to/model reference.json
```

The reference uses Edge0's actual `BailingGate` and `BailingMLP` and mlx-lm's
`SwitchGLU` gathered-expert execution. It copies only selected weight slices and
never instantiates the full model. Defaults use a seeded **synthetic hidden vector
with real weights**, not a hidden state captured from a running language model.
Pass `--input hidden.json` (a flat array of 1536 floats) to test a captured vector.
Use `--layer`, `--expert`, and `--seed` to exercise other cases.

The validator exits nonzero on shape, routing, nonfinite value, or tolerance failures.
It compares up/gate linear outputs, complete expert output, expert IDs, routing
weights, and the complete routed-plus-shared output. Near-zero outputs use an
absolute tolerance rather than unstable relative error alone. `--cpu` selects
CPU arithmetic; the Metal library is still required by MLX initialization.

## Use the block

```swift
import Edge0Core
import Edge0MLX
import MLX

let configuration = try BailingConfiguration.load(from: modelURL.appendingPathComponent("config.json"))
let store = try ExpertTensorStore(modelURL: modelURL.appendingPathComponent("model.safetensors"),
                                  expertCount: configuration.numExperts)
let block = try StreamingMoE(configuration: configuration, store: store, layer: 1)
let result = try block(hidden) // floating [1, 1536]; output is Float32
```

This API is for the supplied preview config: one shared expert, affine INT4/group-64,
unclipped SiLU, no extra projection biases or expert normalization. It accepts one
token, without a residual addition or input RMSNorm; those belong to the decoder.
It excludes Recover-LoRA and prerouter adapters. Do not compare it directly with
an adapter-enabled Edge0 generation run.

One expert occupies 1,327,104 checkpoint bytes. Float32 coefficient copies,
intermediate arrays, the shared expert, router, allocator cache, and mmap resident
pages add memory. The code's sequential lifetime bounds are **not a measured
peak-RAM or phone-performance claim**. The shared expert remains resident per block.

See [validation](docs/VALIDATION.md) and [remaining port plan](docs/PORTING_PLAN.md).
For the physical iPhone probe and current provisioning state, see
[DEVICE_PROBE.md](docs/DEVICE_PROBE.md).

```sh
scripts/test.sh                  # scalar/CPU checks
EDGE0_TEST_GPU=1 scripts/test.sh # same suite on Metal
```
