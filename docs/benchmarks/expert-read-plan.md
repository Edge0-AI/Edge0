# Prepared expert reads

This change prepares nine typed mmap row readers when a streaming layer is constructed. On a cache miss, `_build` reuses the resolved shard, offset, row stride, count, and dtype. Each reader creates a view of the requested expert directly.

Both implementations keep the source weights memory-mapped. The change removes repeated name lookup, metadata arithmetic, and intermediate NumPy views. Quantization, expert selection, tensor shapes, MLX array construction, and the matrix kernels keep their existing contracts. The hot-stack path is unchanged.

## Exactness

Let O be a tensor's byte offset, B its expert stride, D the dtype item size, and i a valid expert index. The old byte slice and the prepared typed read both cover:

`[O + i*B, O + (i+1)*B)`

The prepared read has `B/D` elements. The same little-endian dtype and final tensor shape are used. Fused gate/up reads keep gate rows before up rows.

Readers normalize integer indices before offset arithmetic. This prevents narrow NumPy integer types from overflowing. Construction stores metadata only and creates no exported buffer. A live returned view keeps the mmap export active until the view is released. Closing a shard with descriptors alone succeeds; later calls fail on the closed mapping.

## Tests

The unmodified baseline passed 60 tests on the Linux host. The candidate passes 97, with one existing skip and two slow tests deselected. The 37 added cases cover exact payload bits, all signed and unsigned NumPy integer widths, bounds, invalid dtypes, empty rows, buffer lifetime, 1,024 concurrent reads, and separate/fused layouts across multiple shards. Multi-shard forward outputs also match exactly.

The benchmark checks every expert's tensor shape, dtype, and raw bits before timing. Its complete baseline `_build` method comes from commit `fbab5f8c08e843e204c0fc6ae18b89a154c652cf`. The executable AST was checked against that revision. Both functions are bound methods on the same layer and mapping.

## Measured scope

These are warm-file, expert-cache-miss materialization measurements on an Intel Core i5-3470 running Linux. Each timed call includes MLX array creation and `eval`. The four-worker case also includes pool scheduling and joint evaluation. Four runs contain 328 paired timing blocks and 65,600 timed expert builds across the baseline and candidate.

The experiment uses MLX CPU 0.30.4, mlx-lm 0.31.0, Python 3.12, and NumPy 2.5.3. MLX compilation was disabled equally for both implementations to avoid a compiler compatibility problem in the pinned Linux CPU JIT. The patch does not alter compilation settings.

Two serial runs and one four-worker run use nine public Edge0 expert payloads. Layers 0, 20, and 39 each contribute experts 0, 127, and 255. The source is `Edge0/Edge0-35B-A3B-preview`, revision `1ff9f4478890faec0368c5463b621d1036d5b518`. These payloads are repacked into a test shard for byte transport only. A fourth run uses seeded synthetic packed weights with the same 2,048/512 matrix dimensions.

| Public-weight case | Baseline median, us/expert | Candidate median, us/expert | Paired median time reduction | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Serial A, separate | 431.84 | 395.35 | 7.18% | 3.86% to 9.36% |
| Serial B, separate | 469.35 | 420.37 | 11.74% | 7.50% to 14.34% |
| Serial A, fused | 526.90 | 488.98 | 7.13% | 4.06% to 9.36% |
| Serial B, fused | 514.71 | 486.89 | 5.75% | 3.16% to 7.96% |
| Four workers, separate | 894.75 | 826.20 | 5.32% | 2.63% to 8.28% |
| Four workers, fused | 583.28 | 549.88 | 5.25% | 2.27% to 6.71% |

Each case has 41 randomly ordered baseline/candidate pairs with 100 expert builds per side. Both sides use the same expert sequence. The reported reduction is the median of paired ratios, so it can differ from the ratio of the two time medians. The host remained a live workstation. Intervals describe repeatability on this host.

This record establishes exact expert payloads and reduced host materialization time under the measured conditions. End-to-end token rate, cold-storage latency, GPU/iPhone speed, and model-capacity changes are unmeasured. Nine small descriptors are added per layer; their construction adds no weight copies.

## Reproduce

From an installed checkout, run:

```sh
PYTHONPATH=src python scripts/bench_expert_reads.py --out serial.json
PYTHONPATH=src python scripts/bench_expert_reads.py \
  --workers 4 --batch-size 4 --out threaded.json
pytest -m 'not slow'
```

The default fixture is generated locally and needs no model download. `--shard` accepts an existing safetensors test shard; use `--prefix` to match its tensor names. Linux CPU reproduction also needs the matching `mlx-cpu` package. Set `MLX_DISABLE_COMPILE=1` to reproduce the recorded CPU configuration. `--cpu` optionally pins the serial process on Linux.

Full settings and summaries are in [expert-read-plan-results.json](expert-read-plan-results.json). Every paired observation is in [expert-read-plan-trials.csv](expert-read-plan-trials.csv).
