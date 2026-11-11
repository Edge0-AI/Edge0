# `--adapt-async 1` in the arena mode — community report, RTX 3090 + Ryzen 5 5600X (Windows 11)

Branch `pr/async-adapt-arena` on v0.1.42 (`61b3fb5d`). Results and method only; the change is in the pull request.

**Machine.** RTX 3090 24 GB (sm_86, driver 616.64, 350 W, also drives a 2560 px desktop), PCIe 4.0 x16, Ryzen 5 5600X
(6 cores, AVX2), 2 x 32 GB DDR4-3600, Windows 11 Pro 22631. Qwen3.8-Flash-Next GSQ-RCO IQ3_S, the default arena (46.84 GiB of
experts in RAM, 30 GiB of it page-locked), 8,289 expert slots, `--expert-cache auto --prefill auto --spec 4 --mtp
--max-context 262144 --kv int8 --vram-reserve-mib 1200 --kv-grow --spec-min-p 0.5 --adapt-every 1 --adapt-swaps 80
--adapt-decay 0.97`, `draft_vocab fr`, a learned expert profile. Both arms are local builds of the same tree (VS 2022 17.14,
CUDA 12.8, sm_86); `v0142` is the tag, `pr1_on` this branch with `--adapt-async 1`.

**Method** (`strata_ab2.py`). `serve/server.py` on port 8080, a fresh server per arm and round, 2 warm-up requests not
counted. Cold: 12 distinct prompts (FR/EN, code, prose), each sent once per server with a fixed seed, temperature 0.7, top_p
0.8, 256 tokens, thinking off; 2 rounds, order reversed in round 2; per-prompt mean over the rounds, paired ratio. Decode tok/s
is the engine's `strata serve: prompt ... generated in ...` line. A prompt sent twice in a row runs ~25 % faster the second
time (the tier has adapted to it), so no prompt is repeated within a server.

## Cold short prompts

| Arm | Median tok/s | Ratio vs `v0142` | Prompts faster |
|---|---:|---:|---:|
| `v0142` | 97.2 | 1.000 | — |
| `pr1_on` | 107.7 | **1.071** | **12/12** |

An earlier run the same day, same protocol, when the learned profile was younger (fewer cache hits, more swaps): 92.8 →
98.7 tok/s, x1.11, 12/12 (`cold_earlier.jsonl`). The gain follows how much the tier swaps.

`STRATA_DECODE_TIMING=1`, one cold prompt: blocking tier 25.9 ms/window of which ~3.2 ms outside verify + commit/emit + draft
(#1863); with `--adapt-async 1` 21.6 ms/window, `adapt wait 0.41 join 0.00, 40.6 swaps/window`.

## After a long prompt (prompt read once, decode = mean of 3 seeded answers on the cached prefix)

| Prompt tokens | `v0142` prefill / decode | `pr1_on` prefill / decode |
|---:|---|---|
| 55,630 | 2,173 / 95.3 | 2,167 / 97.3 |
| 116,833 | 2,092 / 97.0 | 2,088 / 96.1 |

Level within the noise: reading a long prompt already moves its experts into VRAM (hit rate 97-99 %), the tier has little
left to swap. The prompt read is not touched.

## Default path

Bit-exact mode (`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`), greedy, 6 prompts, `--expert-cache auto` and again with
a fixed cache size: on the 5 prompts that v0.1.42 reproduces between two of its own fresh engines, this branch without
`--adapt-async` gives the same answers byte for byte (sha256). The 6th prompt differs between two runs of v0.1.42 itself in
that mode on this machine (`bitexact*.jsonl`).

## Robustness

63K-token prompt then 4 follow-up turns in the same conversation (K/V grown to 65,536 cells), no `ERR` (measured on the full test branch, this change plus other opt-in switches); the `VRAM` command and
K/V grows land the round in flight first.

Raw: `cold.jsonl` (this table, also holds a third arm from the other PR), `cold_earlier.jsonl`, `context.jsonl`,
`bitexact.jsonl`, `bitexact_fixed_cache.jsonl`; harness `strata_ab2.py`.
