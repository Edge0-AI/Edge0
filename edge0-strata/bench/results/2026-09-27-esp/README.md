# 2026-09-27: the experimental-speed-projection control vector

The package's `Qwen3.8-Flash-Next-experimental-speed-projection.gguf` (47 unit directions of 2,560, one per layer),
applied as `--control-vector-scaled FILE:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir
per-layer`: after layers 4-44, every hyper-connection stream `h -= (h . v) v`. The package describes it as a
refusal-direction projection. Q2_0, RTX 5070, fixed experts (`--expert-cache 4000 --pcie-frac 0 --adapt-swaps 0`).

## Kernel (`cvec_parity`)

Project and add against a double-precision reference (max error 5e-7); with the switch off, the kernel's pending
residual write is bitwise the fused hyper-connection read's own, so a loaded-but-off vector is the stock engine.

## Teacher-forced, stock vs projection (`esp_kl.py`, per-token path)

| Text | Tokens | Same top-1 | Mean KL | p99 KL | Perplexity stock | With the vector |
|---|---:|---:|---:|---:|---:|---:|
| Code | 511 | 92.8% | 0.098 | 1.69 | 2.594 | 2.982 |
| Document | 1,023 | 89.8% | 0.058 | 0.84 | 30.299 | 30.984 |
| Chat | 1,023 | 89.6% | 0.050 | 0.33 | 18.409 | 18.485 |
| **All** | 2,557 | | **0.063** (max 4.10) | 1.00 | | |

The package reports a mean KL of 0.186 on its own neutral text with UD-Q2_K_XL in llama.cpp.

Cost on the same tokens: 44.34 -> 44.26, 46.74 -> 46.56, 47.02 -> 46.83 tok/s (0.2-0.4%).

## Serve (`esp_serve.py`)

- Loaded and switched off (`cvec=0`) vs an engine without it: 65 / 65 first tokens after prefixes (verify windows
  and the batched prompt path) and three 256-token generations **token-identical**.
- On, off, on again: identical (the switch drops the conversation cache, so nothing computed one way is reused the
  other way).
- The prompt paths steer like the per-token path: served-with-vector first tokens agree with the per-token path's
  teacher-forced argmax with the vector 62 / 65, and with the stock one 57 / 65 (stock served: 64 / 65 and 57 / 65).
- Decode tok/s of 3 x 256-token generations: stock 66.4, on 76.7 / 81.2. **Not a like-for-like comparison:** the
  model writes different text with it on (on the code prompt it copied a long stretch of the source, which prompt
  lookup drafts at 91% acceptance, where stock stopped after 78 tokens). On the same tokens it is 0.2-0.4% slower.
