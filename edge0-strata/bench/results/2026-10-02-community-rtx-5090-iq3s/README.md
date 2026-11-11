# Community-format benchmark: RTX 5090, Ryzen 9 9950X3D, IQ3_S at 262K

Measured on 2026-10-02 on the production box (the server that serves this Claude Code session).
Reproduces the community method (`docs/COMMUNITY_BENCHMARKS.md`, as used in PRs #389/#417/#418/#433)
against the live `:8080` without restarting it. Main limitation: the GPU also served the live agent
session during the runs, and TTFT is measured at the client on a FIFO server, so it includes queue time.

## Hardware and software

- NVIDIA GeForce RTX 5090, 32,607 MiB; driver 617.14; Windows 11 Home (WDDM). Second GPU (RTX 4090)
  idle, not used.
- AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 96 GB RAM. Engine picked 15 expert-pool workers + host thread.
- Engine 0.1.33 release binary (`BUILD.json`: CUDA 13.0, sm_75/86/89/120+PTX); serve wrapper 0.1.30.

## Model and configuration

Qwen3.8-Flash-Next GSQ-RCO **IQ3_S** (2 GGUF shards + PLE shard), pack `Strata-data/packs/iq3_s`,
expert-profile `Strata/data/expert-profile.bin`, MTP `Strata-data/mtp/rt`, vision on CPU (16 threads).

```text
--pack packs/iq3_s --native ...-00001-of-00002.gguf --ple-gguf ...-00002-of-00002.gguf
--expert-profile expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp mtp/rt
--max-context 262144 --kv int8 --kv-resident 65536 --vision --vram-reserve-mib 100
--conversation-cache-mib 32768 --conversation-cache-slots 8 --pcie-frac 0.35 --spec-min-p 0.70
```

Startup: expert cache **12,331 slots (23.41 GiB)**; the prompt path borrows 2,465 slots (4.69 GiB);
free VRAM after load **489 MiB** (min during the whole sweep: 489 MiB).

## Method

[`bench.py`](bench.py): prompts built from this repository's text (as `tools/needle_bench.py` does),
a fresh slice per run — **0 runs with prefix reuse** (`cache_n = 0` in every measured run). Greedy,
thinking off, 256 output tokens, 1 warm-up + 3 measured per cell. Throughput from the engine's own
clock (the response's `timings`, llama.cpp names). TTFT from one streaming run per cell, client-side.
Recall: [`tools/needle_bench.py`](../../../tools/needle_bench.py) `--lengths 32k,128k,256k --depths 10,50,90`.
Per-run data: [`runs.json`](runs.json), [`needles.json`](needles.json); raw stdout: `bench.out`, `needles.out`.

## Results

| Prompt | prompt tok/s (median, range) | decode tok/s (median, range) | TTFT s (client, incl. queue) | draft accept | expert cache hit | KV in VRAM |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1,024 | 961 (876–973) | 136.2 (127.0–150.6) | 4.19 | 76% | 92–97% | 99.6% |
| 4,096 | 2,473 (2,319–2,480) | 141.8 (139.1–145.6) | 1.6 | 76% | 95–97% | 99.3% |
| 32,768 | 4,168 (4,051–4,286) | 142.2 (137.5–146.7) | 12.67 | 77% | 98% | 97.0% |
| 131,072 | 4,450 (4,360–4,593) | 127.2 (117.9–129.0) | 42.31 | 75% | 94–98% | 94–95% |

Needle recall: **9 of 9 found** — 32k (30,839–30,840 tokens), 128k (124,019–124,021),
256k (**261,669–261,670 tokens**, the edge of the 262,144 window) at depths 10/50/90.

## Against the published rows

- Official IQ3_S row (RTX 5070 12 GB, engine 0.1.26): prompt 427/913/1,624/1,640/1,443 at 1K–128K,
  decode 52.4→45.5. This box: **2.7–3.1× prompt, ~2.8× decode**.
- Community RTX 5090 IQ2_XS (Core Ultra 9 285K, Linux, 0.1.29): decode 179.4/175.7/165.0 at 4K/32K/128K.
  This box with IQ3_S (≈2× the expert bytes, Windows/WDDM): 141.8/142.2/127.2.
- PR #389 (2× TITAN RTX, IQ3_S at 262K): prompt 841/1,620/1,786, decode 58–79. This box: ~2.5× prompt, ~2× decode.
- PR #417 (one RTX PRO 4500, IQ3_S): decode 106–124, prefill 3,102–3,281 at 64–128K. This box is ahead of one
  4500 and approaches their two-card figures (decode 153–159, prefill 5.6–5.8k).
- PR #433 (RTX 5090 + Ryzen 9 5950X, AVX2-only, UD-Q4_K_XL): prefill 2.0–2.2k at 14.7K. Not directly
  comparable (quantization, CPU); our 9950X3D has AVX-512 and shows no such prefill ceiling.

## Observations

- **Decode spread at short prompts: 127.0–150.6 (18%)** with only 489 MiB VRAM free — the WDDM stall
  signature PR #279 describes (stalls appear run-to-run when the desktop competes for the last VRAM).
  At 128K the spread narrows to 9%.
- Draft acceptance on this synthetic text: 75–77% (the live agent session runs 83–85%).
- The conversation cache kept the live session through the whole sweep: 109–111k-token restores in
  118–277 ms while 19 bench conversations evicted each other (parked=8, evictions 26→28).
- KV streaming hit 94–99.6% of block reads in VRAM at `--kv-resident 65536`.

## A/B after `--prefill auto:32768` (same day, after a restart)

The config changed `--prefill auto` → `auto:32768`; the engine log then shows
`prompt chunk auto: 32768 tokens` and the prompt path borrows 7,736 cache slots (14.64 GiB) instead
of 2,465 (4.69 GiB). Same script, same machine, same session; per-run data: [`runs-prefill32k.json`](runs-prefill32k.json),
stdout `bench-prefill32k.out` (the baseline per-run JSON was overwritten by this run; its per-run
lines are in `bench.out`).

| Prompt | prompt tok/s before → after | decode tok/s before → after |
| ---: | --- | --- |
| 1,024 | 961 → 999 | 136.2 → 144.4 |
| 4,096 | 2,473 → 2,439 | 141.8 → 148.6 |
| 32,768 | 4,168 → **5,045 (+21%)** | 142.2 → 142.2 |
| 131,072 | 4,450 → **6,004 (+35%)** | 127.2 → 125.4 |

A real session re-prefill of 138,132 tokens ran at **5,719 tok/s** right after the restart. Decode
is unchanged (within noise); no WDDM stalls appeared (min free VRAM during the sweep: 485 MiB).
The engine printed a startup warning — "0 MiB of VRAM free with everything loaded … add
`--vram-reserve-mib 612`" — which reflects the bigger prompt-path loan peaking during prompt
processing, not the idle state; it stays a watch-item, not an applied change.

The gain is larger than PR #282's +15% (measured at 32K with IQ2_XS) because IQ3_S experts are
bigger, so crossing PCIe four times per prompt costs more — consistent with their NVFP4 note
(+47% with 2× expert size).

## The conversation cache: the flag most installs leave off

The engine's default is `--conversation-cache-mib 0` — **off** (`src/program/generate.cpp:349-351`,
with 4 slots and a 2,560 MiB RAM floor). With the default, every follow-up in an agent-style session
re-reads the whole conversation at prompt speed. Our production config runs
`--conversation-cache-mib 32768 --conversation-cache-slots 8`; [`conversation_cache_demo.py`](conversation_cache_demo.py)
measures what that buys — two ~65k-token document conversations interleaved (A1, B1, A2, B2, A3),
greedy, 128 output tokens, fresh text window per run:

| Request | Prompt tokens | Reused | Fresh read | Wall time |
| --- | ---: | ---: | ---: | ---: |
| A1 doc A, first turn | 66,209 | 0 | 66,209 | 12.63 s |
| B1 doc B, first turn | 64,946 | 0 | 64,946 | 11.69 s |
| A2 follow-up on A | 66,258 | 66,202 | 56 | **1.17 s** |
| B2 follow-up on B | 64,994 | 64,939 | 55 | **1.03 s** |
| A3 follow-up on A again | 66,310 | 66,251 | 59 | **0.61 s** |

The engine's own lines: `restored 66,251 tokens (checkpoint) in 82.4 ms`; the live Claude Code
session (160,978 tokens at the time) restored in **185.9 ms**; and after an LRU eviction the partial
path still worked — `prompt 56,060 tokens = 30,560 reused + 25,500 read in 5,098 ms (5,002 tok/s)`.
A follow-up costs 0.6–1.2 s instead of 11–13 s: **10–20×**, and the cache persists across client
sessions (a rerun minutes later restored the conversations the first run had parked).

Capacity on this box: parked bytes run ~30 KB/token (a 162,150-token conversation snapshot is
3.85 GB); with 32 GiB the cache held 8 conversations totalling ~20.8 GB, including the 160k-token
agent session, with the 2,560 MiB RAM floor never hit (96 GB installed). On the default 4 GiB budget
these conversations would evict each other every turn — which is what makes the flag the difference
between "the agent re-reads the world each turn" and "the agent remembers".

## Correctness and limitations

Needle recall is the only quality check. One machine, single stream, greedy, prompts are repo text
(not code-agent prompts). TTFT includes FIFO queue behind the live session's requests (the 128K TTFT
42.3 s vs 29.4 s of pure prefill). The engine was not restarted; all flags are the production ones.
