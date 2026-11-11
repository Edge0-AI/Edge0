# Pin by need (`STRATA_PIN_BY_NEED=1`) — community report, RTX 3090 + Ryzen 5 5600X (Windows 11)

Branch `pr/pin-by-need` on v0.1.42 (`61b3fb5d`). Results and method only; the change is in the pull request.

**Machine and config.** As in the `--adapt-async` report: RTX 3090 24 GB (also drives the desktop), Ryzen 5 5600X, 2 x 32 GB
DDR4-3600, Windows 11; Qwen3.8-Flash-Next GSQ-RCO IQ3_S, the default arena, 8,289 expert slots, `--spec 4 --mtp --max-context
262144 --kv int8 --vram-reserve-mib 1200 --kv-grow --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`. Local builds (VS 2022
17.14, CUDA 12.8, sm_86): `v0142` the tag, `pr2_on` this branch with `STRATA_PIN_BY_NEED=1 STRATA_ARENA_PIN_GIB=29
STRATA_PCIE_MIN1=1 --pcie-frac 0.6`.

**What it registers.** v0.1.42: `cudaHostRegister of the whole arena FAILED (out of memory); 32 slices pinned (30 GiB)`, layers
0-31, layers 32-47 without a PCIe share. This branch: `pin by need: 4404 ranges, 28.96 GiB pinned, 92.2 % of the non-resident
experts (48 of 48 layers have a PCIe share)`, 8,124 of the profile's 24,576 pairs taken as GPU-resident.

## Prompt read and decode after it (prompt read once, decode = mean of 3 seeded answers)

| Prompt tokens | `v0142` prefill tok/s (2 servers) | `pr2_on` prefill tok/s | Decode `v0142` / `pr2_on` |
|---:|---|---:|---|
| 55,630 | 2,180 / 2,165 | **2,313 (+6.5 %)** | 95.3 / 96.5 |
| 116,833 | 2,095 / 2,088 | **2,208 (+5.6 %)** | 97.0 / 98.3 |

Same day, on top of `--adapt-async 1` and other opt-in switches: 63K-token prompt read 2,180 → 2,310 tok/s.

## Cold short prompts (12 prompts x 2 rounds, fresh server per arm and round, paired)

| Arm | Median tok/s | Ratio vs reference | Prompts faster |
|---|---:|---:|---:|
| `pr2_on` vs `v0142` (blocking tier) | 99.3 vs 97.2 | 0.999 | 6/12 |
| on top of `--adapt-async 1` + other opt-in switches: pin by need + `--pcie-frac 0.6` | — | 1.023 | 10/12 |
| same + `STRATA_PCIE_MIN1=1` (1 round) | — | 1.026 | 10/12 |
| pin by need alone on top of the same (no PCIe settings) | — | 0.995 | 6/12 |

Decode gains only together with `--adapt-async 1` and a larger PCIe share; the prompt read gains alone.

## Default path

Bit-exact mode (`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`), greedy, 6 prompts: on the 5 prompts that v0.1.42
reproduces between two of its own fresh engines, this branch without the variables gives the same answers byte for byte;
the 6th differs between two runs of v0.1.42 itself in that mode (`bitexact*.jsonl`). Startup load unchanged (6.7 GiB/s; the
padded layout reads through the buffered loader).

Raw: `context.jsonl`, `cold.jsonl`, `cold_on_async*.jsonl`, `bitexact*.jsonl`; harness `strata_ab2.py`.
