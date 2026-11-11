# Strata benchmark output

started 2026-10-10 08:02:22, finished 2026-10-10 08:18:27
server http://127.0.0.1:8080/v1/chat/completions, model id `qwen3.8-flash-next-iq3_xxs`, output cap 256 tokens, 3 reps per length

## Speed

| target | actual prompt tok | reused | new read | prefill tok/s | read s | gen tok | decode tok/s | thinking off | answer len | finish | TTFT s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 126640 | 129970 | 0 | 129970 | 844.1 | 154.0 | 256 | 54.6 | True | 1272 | length | 154.094 | 158.761 |
| 126640 | 129972 | 0 | 129972 | 831.8 | 156.3 | 256 | 61.0 | True | 1208 | length | 156.386 | 160.555 |
| 126640 | 129973 | 0 | 129973 | 832.9 | 156.1 | 256 | 57.9 | True | 1234 | length | 156.18 | 160.571 |

* 126640 prompt tokens: prefill median 832.9 tok/s (range 831.8-844.1, n=3); decode median 57.9 tok/s (range 54.6-61.0)

Needle checks (exact string recall):
* 126640 tokens, depth 10%: FOUND (prompt 130018 tok, prefill 832.8 tok/s)
* 126640 tokens, depth 50%: FOUND (prompt 130018 tok, prefill 835.7 tok/s)
* 126640 tokens, depth 90%: FOUND (prompt 130018 tok, prefill 831.3 tok/s)
