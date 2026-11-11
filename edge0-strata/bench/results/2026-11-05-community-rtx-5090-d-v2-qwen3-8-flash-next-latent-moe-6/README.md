# Community benchmark on NVIDIA GeForce RTX 5090 D v2

Measured on 2026-11-05 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 5090 D v2, 24 GiB VRAM
- CPU: Intel(R) Core(TM) i9-14900K (24 cores, 32 threads); installed RAM: 125.6 GiB; storage: nvme0n1 ZHITAI TiPlus7100s 2TB nvme
- OS: Linux 6.17.0-14-generic; GPU driver: 580.178.04; CUDA (driver supports up to): 13.0
- Strata: project 0.1.55, commit 84c6f585; engine 0.1.55
- GPU detail (after the run, near idle): power_limit_w=450.00, power_draw_w=228.62, pcie=gen5 x16, pstate=P1, temp_c=57
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-latent-moe
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata --pack latent_fp4_pack --native model.gguf --native-dense-gguf model.gguf --ple-gguf ple-fp8.gguf --ple-io ram --expert-profile expert-profile-bench2.bin --expert-cache auto --prefill auto --pcie-frac 0.25 --spec 4 --spec-min-p 0.5 --mtp rt --max-context 131072 --kv int8 --kv-resident 32768 --vram-reserve-mib 2000 --pool-workers 24 --pcie-mode dma
```

Environment of the config: `STRATA_LATENT_DENSE=latent_dense_q8.bin`, `STRATA_LATENT_A=experts.bin`, `STRATA_LATENT_LABELS=labels.bin`, `STRATA_EMBED_VRAM=1`, `STRATA_LATENT_FUSED=1`, `STRATA_LATENT_FUSED_DU=1`, `STRATA_LATENT_U_GPU=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 69.9 (68.5-72.7) | 0.53 (0.48-0.56) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1766.0 (1763.9-1772.8) | not measured | 2.30 (2.29-2.30) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2598.4 (2566.8-2658.9) | not measured | 6.20 (6.07-6.28) |
| agent session, follow-up turns | 12922 (12907-12941) at the last turn | see below | up to 64 | 3 | see below | 85.4 (65.4-113.1) | 2.12 (1.95-2.37) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.3 % (69.2-69.4); earlier-turn tokens read again per session: 73 (3-74) (earlier turns that the cache did not hold; the 12063 (12063-12065) new tokens of the files are read once); first turn (nothing cached) first token: 1.64 s (1.52-1.66).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 406.8 | 1.516 | 42.4 |
| 1 | 3348 | 620 | 2728 | 1301.1 | 2.121 | 65.4 |
| 2 | 4881 | 3412 | 1469 | 763.0 | 1.947 | 68.6 |
| 3 | 8125 | 4944 | 3181 | 1467.8 | 2.187 | 81.7 |
| 4 | 10194 | 8188 | 2006 | 1000.3 | 2.025 | 93.6 |
| 5 | 12941 | 10187 | 2754 | 1173.5 | 2.369 | 84.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 96 s.
