# One long document, many questions (0.1.40.2, opt-in)

Read a long document once, then ask many different questions without reading it again. All numbers: Tesla P100 16 GB,
IQ2_XS, `--kv int8 --kv-resident 32768`, MTP on, greedy, the 0.1.40 engine unless a row says "pin".

## What the engine already did (0.1.40)

The conversation cache keeps small checkpoints (the GDN and indexer state, about 118 MiB each); the K/V of the document
stays in the session, so a question that starts from a checkpoint reads only what lies past it.

| 262K document (258,655 tokens) | time to first token |
|---|---|
| first question (cold read) | 904 s (286 tok/s) |
| questions 2-20, engine with a turn boundary at the document's end | 0.53 s median (0.44-0.72) |
| the same through the server's chat API (no turn boundary at the document's end) | 55.5 s each (15 measured): the engine re-reads up to `--prompt-cache-every` (16,384) tokens from the last periodic checkpoint |
| `parallel` 2 through the server | the first two questions 918 s and 979 s; the next pair read the document again from token 0 |

Checkpoint RAM: +804 MiB resident after the first read (6 checkpoints); each later question adds at most 10 MiB. The
K/V of 262K takes 3.06 GiB of pinned RAM, once per session (`--batch` slots each take their own).
Identity (greedy, 64 tokens): questions 1 and 12 on a fresh engine give the same tokens as in the 20-question run.
Caveat: IQ models are not byte-reproducible at temperature 0 between runs unless `STRATA_IQ_MT_MIN=1` (DETAILS.md);
some answers differ between two runs of the same prompt for that reason, with or without the field below.

## `strata_prefix` (and the engine's `pin=N`)

`"strata_prefix": {"messages": 1}`, `{"message": 0, "chars": N}` or `{"tokens": N}` marks the start of the prompt as the
shared document. The checkpoint at its end is pinned: never evicted, kept in the parked-conversation budget, kept by
`SAVE`, and a question that resumes from it does not park the branch it leaves.

| 64K document (P100) | no field | with the field |
|---|---|---|
| engine, raw prompts, questions 2-5 | 48.6 s | 0.35-0.53 s |
| a 100K-token branch between two questions, then a question | 149 s | 0.55 s |
| another 20K prompt between (`--conversation-cache-mib 8192`) | 49 s | 0.97 s (restored), then 0.44 s |
| server, 18K-token document, split messages / one message | 8.8 s | 0.6-0.8 s |
| server `parallel` 2, six questions | 730 s (every pair re-reads) | 220 s (questions 3-6: 2.5-3.8 s) |

Copy-on-write K/V pages across `--batch` slots are NOT built: each slot keeps its own K/V (3.06 GiB at 262K, about 12 GiB
at 1M), and admitting a branch copies the prefix K/V (3 s at 262K). The pinned prefix removes the re-read, not the copy.

`tools/research_run.py` runs a document and a question list against a server with and without the field.

## R3: expert slots against K/V blocks (P100, IQ2_XS)

64K document, 5 questions. Prefill, then median first token of a short question, then decode:

| expert slots (GiB) | prefill | first token | decode |
|---|---|---|---|
| 2,083 (2.81) | 230 s | 0.69 s | 27.5 tok/s |
| 3,500 | 200.7 s | 0.61 s | 30.9 |
| 5,000 | 200.7 s | 0.55 s | 34.2 |
| 7,000 | 200.7 s | 0.51 s | 36.7 |

Marginal value of VRAM: decode 0.31-0.41 s per GiB per 200-token answer between 2,000 and 5,000 slots, 0.15 above;
prefill gains only below the prompt loan's floor. K/V cells in VRAM (`--kv-resident`): 4096 / 16384 / 32768 / 65536 give
34.05 / 34.06 / 34.21 / 34.92 tok/s at 64K; at 262K (5,000 slots) 8192 / 32768 / 131072 give 29.54 / 29.46 / 29.49 tok/s
(block hit rate 92.6 / 94.5 / 95.7 %). So the K/V block cache is worth nothing per GiB, experts only help decode, and the
read is already served by the loan. A split planner would gain almost nothing, so none was built. The one real lever in
the read is the chunk: `--prefill 16384` with 7,000 slots reads 64K in 186.4 s against 200.7 s (-7%), and 4096 takes 228 s;
it needs the larger loan and changes the chunking, hence the rounding, so it is not a placement-only change.
