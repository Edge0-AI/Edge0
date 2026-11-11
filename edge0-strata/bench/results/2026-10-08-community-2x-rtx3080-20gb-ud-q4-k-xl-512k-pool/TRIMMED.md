# Trimmed files

Raw dumps were left out to keep the repository small; `README.md`, the summaries, the configuration and the per-run results
(`data/results.json`, `data/summary.json`) are unchanged. The files below came out of the harness run and are not in this folder.

25 files, 3.0 MB:

| file | bytes | why |
|---|---:|---|
| `tokens-104000-run-1-raw.json` | 91,161 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-104000-run-1-request.json` | 198,599 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-104000-run-2-raw.json` | 90,183 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-104000-run-2-request.json` | 198,599 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-104000-run-3-raw.json` | 90,044 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-104000-run-3-request.json` | 198,599 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-127000-run-1-raw.json` | 91,167 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-127000-run-1-request.json` | 242,235 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-127000-run-2-raw.json` | 90,293 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-127000-run-2-request.json` | 242,235 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-127000-run-3-raw.json` | 90,299 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-127000-run-3-request.json` | 242,235 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-25000-run-1-raw.json` | 90,734 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-25000-run-2-raw.json` | 88,877 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-25000-run-3-raw.json` | 91,013 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-4096-run-1-raw.json` | 91,038 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-4096-run-2-raw.json` | 91,374 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-4096-run-3-raw.json` | 91,572 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-51000-run-1-raw.json` | 90,779 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-51000-run-1-request.json` | 98,068 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-51000-run-2-raw.json` | 90,386 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-51000-run-2-request.json` | 98,068 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `tokens-51000-run-3-raw.json` | 90,365 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |
| `tokens-51000-run-3-request.json` | 98,068 | request body (the prompt itself); `benchmark.py` rebuilds it deterministically from the target length, and its sha256 is `request_sha256` in `data/results.json` |
| `warmup-raw.json` | 2,682 | SSE chunk dump of the whole response; the same row (usage, engine timings, text) is in `data/results.json` |

Kept from the harness output: `results.json`, `summary.json`, `initial-status.json`, `gpu.csv` (telemetry sampled beside it), the harness stdout as
`data/harness-stdout.log`, and the request bodies of the warm-up, the 4,096-token and the 25,000-token runs (small).

Other trimming: `engine-start.log` holds the first 200 lines after "engine started"; `engine-requests.log` holds only the
`prompt ... tokens`, `conversation cache: parked/restored`, `decode expert cache hit rate` and `KV streaming` lines of the log from that start on
(the engine's full log of that day is 12.9 MB and covers earlier runs too).
