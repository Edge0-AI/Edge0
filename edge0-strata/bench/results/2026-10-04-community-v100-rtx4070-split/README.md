# Community benchmark: Tesla V100 32 GB + RTX 4070 12 GB layer split, 16 GB of system RAM

Measured on 2026-10-04 by christopherrobertbrooks-tech. The full Qwen3.8-Flash-Next at IQ2_XS split across two NVIDIA cards of
different architectures (Volta sm_70 + Ada sm_89), in both card orders, on a PC with 16 GB of RAM. Main limitation: one
machine, one size, three runs per configuration; the V100 sits on a PCIe 3.0 x4 link.

## Hardware and software

- GPU 1: Tesla V100-PCIE-32GB (sm_70), PCIe 3.0 **x4**; GPU 0: RTX 4070 12 GB (sm_89), PCIe 4.0 x16, which also drives the
  display (~0.9 GB in use). Default power limits (250 W / 200 W). CPU: i7-13700KF (24 threads, AVX2). RAM: **16 GB DDR5**
  (15 GiB visible). Model files on a SATA SSD (Samsung 870 EVO 500 GB).
- Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic, NVIDIA driver 580.173.02, CUDA 12.9 toolkit.
- Strata main at 6f32ec0, engine 0.1.39, built by setup.sh from source with `--cuda 12` for both cards:
  `engine-cuda12/BUILD.json` `"archs": [70, 89]`. One local change in `serve/frontend.py` (PR #819), not used by these runs.
- Strata's PCIe probe at start: 18.6 GB/s to the 4070, 3.3 GB/s to the V100.
- No other GPU work; the server was started directly (`run-iq2_xs.sh`, port 8080).

## Model and configuration

- Qwen3.8-Flash-Next GSQ-RCO **IQ2_XS** (`--family qwen --model IQ2_XS`): `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf`,
  `-00002-of-00002.gguf`. No vision encoder for these runs. Expert profile `data/expert-profile.bin` (shipped).
- `"layer_split": "auto"`; low-RAM mode as setup chose it (`--mmap-experts`, `--remote-expert-opt`); context 131,072 (raised
  from setup's 32,768); KV int8; expert cache auto; prefill auto; MTP `--spec 4 --spec-min-p 0.5`; reasoning `none`;
  temperature 0; no calibration, no speed projection.
- Auto split, V100 first (`"gpu": [1, 0]`): **K=39** -- V100 layers 0-38, 4070 layers 39-47 and the head; the caches hold
  23,518 of 24,576 profiled pairs (~99.9% of the routed mass); "95% of the experts resident"; 510 MiB VRAM free.
- Auto split, 4070 first (`"gpu": [0, 1]`): **K=11** -- 4070 layers 0-10, V100 layers 11-47; 23,459 profiled pairs; 455 MiB free.

```text
./setup.sh --setup --yes --family qwen --model IQ2_XS --gpus 1,0 --cuda 12 --vision no --data-dir /mnt/steam/Strata-data --no-start
engine args: --pack .../packs/iq2_xs --native ...-00001-of-00002.gguf --ple-gguf ...-00002-of-00002.gguf
  --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --mtp .../mtp/rt --max-context 131072 --kv int8 --mmap-experts --remote-expert-opt
```

## Method

The 2x MI50 report's `benchmark.py`, unchanged (the same script as our single-V100 report, #823): serial fresh prompts (0
reused tokens), one warm-up, three measured runs at 4,096, 32,768 and 128,000 prompt tokens, 256-token output cap. Engine
timings for prompt and decode; client TTFT with the model loaded. PCIe sampled with `nvidia-smi dmon -s t` (1 s) during the
32K runs. Then `tools/needle_bench.py --lengths 32k,128k --depths 10,50,90`.

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| V100 first, auto (K=39) | 4,096 | 0 | 256 | 3 | 996 (502–1,000) | 84.9 (76.8–89.6) | 4.14 (4.13–8.24) |
| V100 first, auto (K=39) | 32,768 | 0 | 256 | 3 | 1,602 (1,502–1,605) | 81.5 (78.6–83.0) | 20.53 (20.48–21.90) |
| V100 first, auto (K=39) | 128,000 | 0 | 256 | 3 | 1,552 (1,547–1,554) | 75.1 (73.9–79.0) | 82.61 (82.52–82.89) |
| 4070 first, auto (K=11) | 4,096 | 0 | 256 | 3 | 985 (370–996) | 79.1 (78.9–82.5) | 4.19 (4.14–11.16) |
| 4070 first, auto (K=11) | 32,768 | 0 | 256 | 3 | 1,535 (1,287–1,542) | 83.8 (75.3–86.0) | 21.45 (21.31–25.56) |
| 4070 first, auto (K=11) | 128,000 | 0 | 256 | 3 | 1,533 (1,532–1,535) | 77.1 (76.9–77.7) | 83.64 (83.58–83.71) |

- The slow first run at 4K (502 / 370 tok/s prompt, 8.2 / 11.2 s TTFT) is the first request after the warm-up; the other two
  runs agree with each other.
- Expert-cache hit rate 0.993–1.000. Read from the model files per request: 113–794 MB with the V100 first, 143–1,482 MB with
  the 4070 first (the experts the cards don't hold come through the OS file cache).
- MTP draft acceptance 0.63–0.81.
- PCIe into the V100 during long prompts: peaks of **3.5–3.7 GB/s -- essentially its whole x4 link** -- median ~0.3–0.4 GB/s;
  into the 4070, peaks of 13.4–13.7 GB/s.
- A fixed `"layer_split": "39"` gave the same numbers and the same memory use as auto (4K / 32K / 128K decode 84.9 / 82.0 / 75.2):
  in 0.1.39 the auto split already loads only each card's own layers.
- Load to first answer: ~100 s.

## Correctness and limitations

- Needles: **6 of 6 found** in both orders (32K and 128K, depths 10/50/90).
- Long prompts on the second card: every 32K and 128K prompt (past the 8,192-token chunk) went through in both orders, no
  `no kernel image` or other errors (#690 reports that failure on a mixed AMD pair).
- HumanEval (164, greedy, thinking off, our own harness, vision on, V100 first): 154/164 in 7.2 min -- the Coder IQ1_M on the
  V100 alone scored 158/164 in 7.1 min (#823).
- For reference, the RTX 5090 community report (same IQ2_XS; 32 GB card + 64 GB RAM): decode 179 / 176 / 165 tok/s, prompt
  4,270 / 5,543 / 5,779 tok/s -- this pair reaches ~45% of its decode and ~29% of its prompt speed.
- Not tested: other sizes, vision on for the speed runs, more than three runs, batching.

The report was drafted with an AI assistant (Claude) from our measurements, and checked by me.
