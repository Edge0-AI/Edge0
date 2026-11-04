# Single-turn answers

Regression answers exported by this repository's engine with `tests/answers/export.py`.
`tests/answers/judge.py` compares engine output against them; `tests/answers/inventory.py`
lists the suites in use, and this directory must contain exactly those suites.

`cases.json` defines the fixed cases (`chat-intro`, `plain-mixed`, `long-mixed`).

Each suite directory `<model>-<base|lora>-<cpu|metal>-flash[-split]` holds:

| File | Content |
| --- | --- |
| `identity.json` | Asset digests (from the asset catalog), LoRA, backend, chunking (`whole` or `split-last`), model options, greedy settings, engine commit and whether tracked files were clean, exporter source digests, `top`, probe seed and positions, case list and `cases.json` digest, vocabulary and routing dimensions |
| `SHA256SUMS` | Digests of every file except `run.json` and itself |
| `run.json` | Export time and per-case durations; not hashed |
| `<case>/tokens.json` | `input`, `generated`, `stop_reason`, `chunking`, `logits_shape`, `route_steps` (`[step, positions]`, step -1 is the prefill) |
| `<case>/logits.bin` | Per logits row, little endian: the top 32 token IDs (i32), their values (f32, value descending, then token ascending), then the values at the 32 probe positions (f32) |
| `<case>/routes.bin` | Selected experts (u8) for each recorded step in `route_steps` order, then each layer, then each position, then each rank |

Answers are compared only with output of the same backend. Replacing an answer
(an intended numerical change or new assets) means re-exporting the affected
suites, comparing new and old answers with the previous judgement in the same
change, and updating the in-use list.
