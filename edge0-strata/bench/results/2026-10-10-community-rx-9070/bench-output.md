# Strata benchmark output

started 2026-10-10 07:54:42, finished 2026-10-10 08:00:00
server http://127.0.0.1:8080/v1/chat/completions, model id `qwen3.8-flash-next-iq3_xxs`, output cap 256 tokens, 3 reps per length

## Speed

| target | actual prompt tok | reused | new read | prefill tok/s | read s | gen tok | decode tok/s | thinking off | answer len | finish | TTFT s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 4195 | 0 | 4195 | 590.7 | 7.1 | 256 | 58.7 | True | 1410 | length | 7.127 | 11.47 |
| 4096 | 4198 | 0 | 4198 | 584.0 | 7.2 | 256 | 57.9 | True | 1208 | length | 7.225 | 11.618 |
| 4096 | 4199 | 0 | 4199 | 586.2 | 7.2 | 256 | 60.5 | True | 1265 | length | 7.208 | 11.41 |
| 32768 | 33467 | 0 | 33467 | 804.8 | 41.6 | 256 | 57.9 | True | 1196 | length | 41.64 | 46.031 |
| 32768 | 33467 | 0 | 33467 | 803.4 | 41.7 | 256 | 59.8 | True | 1310 | length | 41.716 | 45.964 |
| 32768 | 33467 | 0 | 33467 | 804.5 | 41.6 | 256 | 55.8 | True | 1286 | length | 41.669 | 46.231 |
| 128000 | FAILED: HTTP 400: {"error": {"type": "invalid_request_error", "messa |  |  |  |  |  |  |  |  |  |  |  |
| 128000 | FAILED: HTTP 400: {"error": {"type": "invalid_request_error", "messa |  |  |  |  |  |  |  |  |  |  |  |
| 128000 | FAILED: HTTP 400: {"error": {"type": "invalid_request_error", "messa |  |  |  |  |  |  |  |  |  |  |  |

* 4096 prompt tokens: prefill median 586.2 tok/s (range 584.0-590.7, n=3); decode median 58.7 tok/s (range 57.9-60.5)
* 32768 prompt tokens: prefill median 804.5 tok/s (range 803.4-804.8, n=3); decode median 57.9 tok/s (range 55.8-59.8)

Needle checks (exact string recall):
* 32768 tokens, depth 10%: FOUND (prompt 33512 tok, prefill 795.4 tok/s)
* 32768 tokens, depth 50%: FOUND (prompt 33512 tok, prefill 797.6 tok/s)
* 32768 tokens, depth 90%: FOUND (prompt 33512 tok, prefill 788.0 tok/s)
* 128000 tokens, depth 10%: MISSED (prompt None tok, prefill None tok/s)
* 128000 tokens, depth 50%: MISSED (prompt None tok, prefill None tok/s)
* 128000 tokens, depth 90%: MISSED (prompt None tok, prefill None tok/s)
