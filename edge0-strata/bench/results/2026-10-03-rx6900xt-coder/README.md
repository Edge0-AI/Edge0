# Coder IQ1_M on RX 6900 XT and Threadripper 3990X

Measured on 2026-10-03 (Asia/Tokyo), using the real RX 6900 XT through HIP. Coder IQ1_M generated **45.3–48.0 tokens/s across the three workload medians** after selecting 31 CPU pool workers. Setup's default 63 workers produced medians of 6.1–12.5 tokens/s. [Japanese results and usage](RESULTS-JA.md).

**Historical release-binary sweep:** the 31-worker selection below describes the initial installation. The later [processor-group correction and final automatic 63-worker check](../2026-10-03-windows-groups/README.md) supersede that local selection; this report does not propose a 31-worker default. Saved configs below preserve the original measurements.

## Hardware and software

- AMD Radeon RX 6900 XT, 16 GiB VRAM, gfx1030.
- AMD Ryzen Threadripper 3990X, 64 physical cores / 128 logical processors.
- 128 GiB installed RAM (127.9 GiB visible), eight 16 GiB DDR4 modules configured at 3200 MT/s.
- Windows 11 Pro, build 10.0.26300; AMD display driver 32.0.21045.5002.
- Model storage: C: NVMe SSD, CSSD-M2B2TPG3VNF.
- Strata v0.1.38, upstream commit `99f3dbd0b21d1401b3769e0c0d963913607f380b`.
- Official Windows HIP release binary; bundled ROCm `10.2.0a20260930`, hipBLASLt `100500`.
- Normal interactive desktop. Unrelated applications were not stopped. GPU power limits and driver settings were not changed.

Hardware snapshots, binary hashes, and the passing device self-test are stored alongside this report. GPU runtime detection/self-test passed before downloading the model.

## Installation

Source: <https://github.com/Niko1221/Strata/releases/tag/v0.1.38>

```powershell
.\.venv\Scripts\python.exe setup.py --yes --family coder --model IQ1_M --backend hip --context 65536 --vision no --host 127.0.0.1 --no-start
```

Application: `C:\Dev\Strata`; model data: `C:\Dev\Strata-data`. Both GGUF files' complete SHA-256 hashes and byte counts matched Hugging Face's pinned LFS metadata; see `model-integrity.json`. The second shard is shared with the original model.

At the time of this sweep, the selected configuration was saved in `C:\Dev\Strata\strata-coder-iq1_m.json` and starts with `run-coder-iq1_m.bat`. Its local-only endpoint is **http://127.0.0.1:8081**, with OpenAI base URL **http://127.0.0.1:8081/v1**. `selected-config.json` and `selected-launcher.txt` preserve the installed settings. The original setup configuration is preserved as `setup-default-config.json`.

Coder source: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, pinned revision `5348543e0147355ac9cbcb031184a3546350988e`. Coder retains 256 experts per layer; its IQ1_M label does not mean every tensor is uniformly one bit. Setup also obtains and prepares the original checkpoint's MTP draft tensors.

## Measurement method

`benchmark.py` sends streaming OpenAI-compatible requests to the real local engine. Each configuration uses one separate warm-up and three measured requests per selected workload. Requests use temperature 0, reasoning effort `none`, images off, and a 256-token output cap. Actual generated tokens and early stops are retained.

The short workload requests a Python task queue implementation. The longer workloads use synthetic Python modules, sized close to 4,096 and 32,768 tokens through the server's token-count endpoint. Full request bodies, output text, usage, engine timings, and stream events are retained in each run's JSONL.

Each trial changes an early nonce to avoid reusing a long prompt. All 21 measured fresh-prompt requests actually reused **zero tokens** and generated exactly **256 tokens**. The engine's actual fresh/reused token counts are authoritative. The expert cache remains adaptive and warm within a server session; this is different from prompt-prefix reuse. Model loading is excluded from request timings. Each of the four sessions has a separate 256-token warm-up saved in `warmup.json`.

The token-count endpoint uses the Anthropic renderer, whose count here was 40 tokens greater than the OpenAI request's actual usage. The table uses the actual OpenAI/engine counts (4,022 and 32,685), not the estimated target counts.

- Prompt throughput: fresh prompt tokens divided by the engine's prompt time.
- Decode throughput: engine-generated tokens divided by the engine's decode time.
- TTFT: client request start to the first nonempty generated text or reasoning delta; empty stream messages are ignored.
- Total latency: client request start to stream completion.

Example, with the local server already ready:

```powershell
.\.venv\Scripts\python.exe bench/results/2026-10-03-rx6900xt-coder/benchmark.py --cases short,4k,32k --runs 3 --out bench/results/2026-10-03-rx6900xt-coder/default --log strata-coder-iq1_m.log
```

## Results

Median of three runs; parentheses show min–max. All speeds are tokens/s. TTFT and total latency are seconds. Every row uses the same 65,536 context capacity, INT8 KV with 32,768 resident cells, auto prefill (8,192), auto expert cache, original supplied Coder expert profile, MTP `--spec 4 --spec-min-p 0.5`, and no experimental speed projection.

| CPU workers | Actual input tokens | Prompt tokens/s | Decode tokens/s | TTFT median | Total median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 63 (setup default) | 94 | 41.6 (23.4–42.2) | 12.5 (11.3–12.9) | 2.45 | 22.70 |
| 63 (setup default) | 4,022 | 313.2 (282.4–313.9) | 11.6 (9.7–12.6) | 13.04 | 34.96 |
| 63 (setup default) | 32,685 | 349.8 (348.7–353.7) | 6.1 (5.9–7.2) | 94.07 | 135.65 |
| **31 (selected)** | **94** | **46.8 (31.8–47.0)** | **48.0 (44.9–49.5)** | **2.04** | **7.34** |
| **31 (selected)** | **4,022** | **312.5 (295.8–315.1)** | **45.3 (43.6–45.4)** | **12.93** | **18.54** |
| **31 (selected)** | **32,685** | **352.1 (351.7–353.5)** | **46.0 (41.5–46.8)** | **93.02** | **98.55** |
| 15 (comparison) | 94 | 45.6 (31.4–46.2) | 44.4 (40.9–44.4) | 2.10 | 7.83 |

31 workers were fastest among the tested counts (63, 31, 15), not an exhaustive optimum. The short-prompt decode median improved **3.83x** over setup's default. Input processing changed little; CPU worker selection chiefly improved generation in these runs.

Run order: default all three cases, restart with 31 workers for the short case, restart with 15 for the short case, then reload 31 for the two long cases. Only one model was loaded at a time. Separate loopback ports were used for the comparison servers. All raw request bodies, outputs, engine timing lines, per-run data, and summary statistics are under `default/`, `workers-31/`, `workers-15/`, and `workers-31-long/`. `comparison.csv` / `comparison.json` provide the combined statistics.

The worker-count setting was the only intentional inference configuration change. Auto cache capacity varied slightly with the desktop VRAM budget: see `effective-settings.json` (3,267 default slots, 3,297 in the first 31-worker session, and 3,288 in its long-prompt session). This was not a controlled benchmark with all other applications closed, nor a proof of the exact slowdown cause. The Windows affinity code's handling of processor groups is a plausible contributor on this 128-thread CPU, but was not patched or causally isolated.

The actual engine used `pcie_frac=0.55`, a 23,981 MiB expert arena, MTP depth limit 4, suffix lookup 3, and verification window 6. No gfx1030 hipBLASLt tuning table exists in this installation; dense prefill used plain hipBLAS. See the saved engine logs for the complete decisions.

### Cached conversation follow-up

One additional continuation immediately after the last 32K run had **32,962 input tokens, of which 32,940 were reused**. It generated 256 tokens at **47.3 tokens/s**, with **0.669 s TTFT** and **6.032 s total latency**. This is a single observation, excluded from the three-run fresh-prompt medians. Reproduce with `check_followup.py` immediately after `32k-run-3`; results are in `workers-31-long/followup.json`.

### Memory and correctness

During default short inference, a snapshot showed 25.52 GiB engine working set, 71.90 GiB total system RAM in use, and 15.61 GiB dedicated GPU memory in use across the desktop. These are snapshots, not peaks. The 25.48 GiB of shared GPU memory includes host-mapped model memory and does not establish paging by itself. Windows HIP telemetry did not expose temperature or power through Strata, so neither is claimed.

A separate generated `clamp_score` function passed all **2,001 integer inputs from -1,000 through 1,000** after its syntax was checked against a restricted allowlist. The full request, code, and result are in `coding-smoke-check.json`, reproduced by `check_code.py --url http://127.0.0.1:8081`. This is a simple operational check, not a general coding-quality score. No measured request failed.

### Reproduce the selected configuration

To repeat this historical sweep, use the official release binary with the saved `selected-config.json` (31 workers), updating local paths as needed. The current local launcher uses the later group fix with automatic workers; see the follow-up report. With the historical configuration loaded, run:

```powershell
.\.venv\Scripts\python.exe bench/results/2026-10-03-rx6900xt-coder/benchmark.py --url http://127.0.0.1:8081 --cases short,4k,32k --runs 3 --out bench/results/2026-10-03-rx6900xt-coder/repeat --log strata-coder-iq1_m.log
```

This runs all cases in one new session. The original measured 31-worker short/long sessions are kept separately as described above.

## Scope

These measurements concern this Windows installation and these synthetic coding workloads. They do not establish overall coding quality, full 64K-context correctness, or performance on another operating system. The generated 256-token excerpts can be incomplete by design.
