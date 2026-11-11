# Resident expert exchange buffer rotation

When the resident RAM complement exchanges an expert with the GPU cache, the
evicted expert lands in a temporary host buffer. The existing path then copies
those bytes into the promoted expert's old RAM slot. Rotation removes that last
host copy: the temporary buffer becomes resident, and the old RAM slot becomes
the next temporary buffer.

## Enable

Rotation is **off by default**. Set `STRATA_EXCHANGE_ROTATE=1` before starting the
engine with your existing arguments:

```bash
export STRATA_EXCHANGE_ROTATE=1
```

```powershell
$env:STRATA_EXCHANGE_ROTATE = '1'
```

Only the exact value `1` opts in. Set it to `0` or remove it to restore the copy
path. This setting does not select resident RAM mode or change any preset.
It applies only when the resident complement is fully pinned and GPU-mapped
and the exchange buffers can also be pinned and mapped. Other layouts keep the
copy path and log why rotation could not be enabled. Configurations without a
resident complement do not use it. Layers may differ in expert block size (a
GGUF with mixed quantizations per layer): see
[Layers of different block sizes](#layers-of-different-block-sizes).

The startup diagnostic confirms activation:

```text
FileExpertSource: exchange buffer rotation enabled: ...; no host commit memcpy
```

Serving diagnostics report the cumulative rotated block count and avoided
`memcpy` payload bytes. This counts the copied payload once, not read plus write
traffic, and does not include the GPU transfers, which still occur.

## Ownership and synchronization

The original allocations own their memory until `FileExpertSource::close()`.
Rotation updates slot ownership and keeps each host pointer paired with its GPU
alias. Atomic slot IDs let background residency queries observe those immutable
pairs. Compute readers and GPU transfers must still finish before the existing
serialized commit; rotation adds no overlapping compute or new synchronization
policy. Reserve the maximum exchange capacity before rotation starts: growing a
live exchange arena is rejected because it can now hold resident experts.

## Layers of different block sizes

A swap exchanges two experts of the same layer, so the RAM slot the promoted
expert leaves has exactly the size of the block the evicted one needs. Rotation
therefore keeps its spare buffers per block size: an exchange of a layer takes
a spare of that layer's size and hands back one of the same size, and every
size keeps its number of spares for good. The first version required one size
for all layers and fell back to the copy path on any other pack; Swift 1.5
IQ3_S has seven sizes (1,510,400 to 2,662,400 bytes) and IQ3_XXS has nine
too, so neither could rotate.

The exchange allocation holds one spare of every size per swap of a round:
`--adapt-swaps` (at most 96) times the sum of the distinct sizes. With
`--adapt-async 1` the second half of the buffers only stages copies and is
kept in the largest size alone. On IQ3_S with `--adapt-swaps 80` that is
1.06 GiB of page-locked RAM (1.26 GiB with `--adapt-async 1`), against 0.20
and 0.40 GiB for the copy path's buffers. The startup line says how many sizes
and how much:

```text
FileExpertSource: exchange buffer rotation enabled: 80 buffers in 7 block sizes (largest 2662400 bytes, 1.06 GiB page-locked); no host commit memcpy
```

With `--adapt-async 1` a rotated round also ends one verify window sooner:
there are no bytes for the background job to move, so the ownership flip runs
in the tick that sees the copies in.

`tests/core/exchange_storage_test.cpp` covers it without CUDA (three sizes,
6,144 exchanges: bytes, aliases, a spare of another size or an expert of
another layer refused, overlapping resident blocks refused).

### Measured: Swift 1.5 IQ3_S on an RTX 5090 with 64 GB (Windows 11)

Core i7-13700K, RTX 5090 32 GB, 64 GB DDR5, Windows 11, driver 617.42, CUDA
13.0.88, engine 0.1.42 built from source with this change. Swift 1.5 IQ3_S,
128K context, `--resident-experts` (31.4 GiB of experts page-locked in RAM,
about 10,800 of 24,576 in VRAM), `--adapt-every 1 --adapt-swaps 80
--adapt-decay 0.97`, `--spec 4`, `--pool-affinity p-cores`. In the first and third
rows one binary ran both arms and only `STRATA_EXCHANGE_ROTATE` differs; in the
second the off arm is the build before this change (the third row's binary also
carried extra timers).

| Run | Rotation off | Rotation on |
|---|---|---|
| One 8K prompt, the same 1,500 forced tokens (`--spec-follow`), blocking tier, decode tok/s | 163.4, 165.1, 163.1, 161.6, 162.6 | 187.4, 188.7, 185.8, 184.9 |
| `--serve`, 23 coding-agent turns (`tools/agent_turns_bench.py`'s turns, greedy, up to 400 new tokens), blocking tier, tok/s over all turns, three interleaved rounds | 147.8, 159.2, 158.4 | 175.2, 184.9, 172.8 |
| The same turns with `--adapt-async 1`, one binary, two interleaved rounds | 166.0, 170.1 | 185.6, 182.5 |

In the first row, two runs of each arm that sized the expert cache alike (10,681
slots) wrote the same `--window-hashes` file for all 434 windows: the same
arithmetic, window by window. (`--expert-cache auto` sizes the cache from the
free VRAM at start, and a run with another slot count places experts
differently, so its hashes differ with or without rotation.) The tier swapped
about 14,000 experts in such a run, 10,100 to 10,600 of them exchanges. In the serve rows the
rotated arm was faster on 23 of 23 turns (blocking tier) and 22 of 23
(`--adapt-async 1`). On this box the tier swaps about 30 experts per verify
window all through a run (the cache holds 44% of the experts), so the copy
path moved about 50 MiB of RAM per window on the decode thread; that is the
time rotation removes. With `--adapt-async 1` the same copy ran on the
background job, and the decode thread still lost about 1 ms per window (0.4 to 2.5 by request)
while it ran; rotated, the tier's part of a window is 0.06 ms.

Not measured: Linux, AMD cards, a layer split, a pack with one block size on
this machine (the code path for it is the one above with a single size).

## Tests

The ownership test can run without CUDA:

```bash
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -pthread -Iinclude tests/core/exchange_storage_test.cpp -o /tmp/exchange_storage_test
/tmp/exchange_storage_test
```

With an existing CUDA build configured:

```bash
cmake --build build --target strata exchange_storage_test file_expert_source_test
ctest --test-dir build -R '^(exchange_storage_test|file_expert_source_test)$' --output-on-failure
./build/file_expert_source_test --rotation-gpu
compute-sanitizer --tool memcheck --error-exitcode 71 ./build/file_expert_source_test --rotation-gpu
```

The CPU test checks byte preservation, pointer/alias ownership, guards, invalid
operations and concurrent residency queries across 12,304 exchanges. The CUDA
fixture performs 64 exchanges in each of three modes: copy with pinned RAM,
rotation with pinned RAM, and requested rotation with pageable RAM (fallback).
It checks exact bytes, GPU alias reads, file fallback, capacity, and close/reopen.
The GPU fixture is explicit; the ordinary CTest invocation does not run it.

On 2026-10-04, the clean patch on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab` built on Linux with GCC 15.2 and CUDA
13.3 for an RTX 4090. Both CTest tests, the ASan/UBSan ownership test, and the
CUDA fixture passed. Compute Sanitizer reported **0 errors**. AMD and Windows
GPU execution were not tested for this patch.

## One native token-identical A/B

Measured on 2026-10-04 using the clean rotation build at
`18a30ad775bce86e99a54e6dc552ccb72c66d16e`, on upstream
`6f32ec070f23ced9f50e704d854d775da52591ab`. The model is the standard
GSQ-RCO **Q2_0**, with its matching IQ4_NL PLE shard. Both shards were verified
against the publisher's SHA-256 hashes at the pinned revision in the receipt.
The native pack uses the existing `tools/iq_pack.py`; no compatibility conversion
or Q8 PLE reader extension is needed.

Both arms used the same Release binary (GCC 13.3, CUDA 13.2), RTX PRO 6000
Blackwell 96GB, Ryzen 9 7950X and 128GB installed RAM. A fresh engine per arm
generated exactly 1,024 tokens after the same 1,024-token counting prompt.
The context allocation was 40,960, with 12,000 GPU expert slots, a 16.19 GiB
pinned/mapped resident complement, FP16 KV, greedy target-only decoding and
zero offered drafts, reused prompt tokens or file-tier blob reads during
generation. Only `STRATA_EXCHANGE_ROTATE` changed. Transfer completion is
awaited before admission.

The GPU cache was deliberately capped at 12,000 slots (15.45 GiB) to exercise
resident RAM exchanges. Q2 can fit entirely on this 96GB GPU; this is a storage
path comparison, not the fastest full-GPU configuration. It does not change
the project's presets or enable resident mode for existing users.

| Rotation | Decode tokens/s | Resident exchanges | Avoided host-copy payload | Output IDs |
|---|---:|---:|---:|---|
| `0` | 104.90 | 3,802 | 0 bytes | 1,024, identical |
| `1` | 113.44 | 3,802 | 5,255,884,800 bytes | 1,024, identical |

Decode rates exclude startup and prefill. The observed throughput change was
+8.1% in this one pair; repeated timing or a general speedup is not established.
The out-of-vocabulary EOS sentinel `2147483647` forces the exact output length,
so this is not a model-quality evaluation. Speculative-path token parity,
AMD and Windows GPU execution remain untested.

The same Blackwell binary passed both CTest checks and the three-mode CUDA
fixture; Compute Sanitizer reported 0 errors. The
[receipt](../bench/results/exchange-rotation-ab.json) contains source, binary,
model, pack and profile hashes, arguments, timings and diagnostics. The
[raw IDs](../bench/results/exchange-rotation-token-ids.json) retain the input
and both outputs. To check the token comparison without a model:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('bench/results')
r = json.loads((p / 'exchange-rotation-ab.json').read_text())
raw = (p / r['token_ids_file']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == r['token_ids_file_sha256']
d = json.loads(raw)
assert len(d['input']) == len(d['off']) == len(d['on']) == 1024
assert d['off'] == d['on']
for name, arm in zip(('off', 'on'), r['arms']):
    sha = hashlib.sha256(json.dumps(d[name], separators=(',', ':')).encode()).hexdigest()
    assert sha == arm['output_token_ids_sha256']
print('1,024 identical output token IDs')
PY
```
