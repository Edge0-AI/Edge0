# What two opt-ins cost on a single-GPU RTX 4090: `STRATA_KV_PREFETCH=1`, and `"parallel": 2` with `--batch-mtp`

Measured on 2026-10-06 by [Dmitry-B](https://github.com/Dmitry-B), on the machine and with the working configuration
described in
[2026-10-03-community-rtx4090-iq3xxs-200k](../2026-10-03-community-rtx4090-iq3xxs-200k/README.md) (RTX 4090,
`qwen3.8-flash-next-iq3_xxs`, context 204,800, INT8 KV, **no** `--kv-resident`, `--mmap-experts`,
`--spec 4 --mtp`, `draft_vocab=cyrillic`). Engine 0.1.40. Not a version comparison: two checks of two opt-ins, each
with both arms measured on the same day on the same running server.

Short answer for this PC: **both stay off.** `STRATA_KV_PREFETCH=1` is a 2–4% loss here, because with no
`--kv-resident` there is nothing to overlap. `"parallel": 2` costs 28% of single-client decode and only pays when a
third or fourth client appears. `--batch-mtp` cannot admit at all with this model's MTP pack.

## Files in this folder

| File | What it is |
| --- | --- |
| `benchmark.py` | the speed script (the same file as in the paired reports) |
| `config-kvpref.json` | the server config copy of the KV-prefetch arm (the working config plus one environment variable) |
| `runs-kvpref.json` | the KV-prefetch runs |
| `conc_bench.py` | the concurrency script: N identical-shaped requests sent at the same time |
| `runs-parallel-off.json`, `runs-parallel-2.json`, `runs-parallel-2-batch-mtp.json` | the three concurrency arms |

## 1. `STRATA_KV_PREFETCH=1` (#732)

Measured with the machine's working configuration plus that one environment variable, three runs at 32,768 and
131,072 prompt tokens, code prompts, greedy, 256-token cap. The arm it is compared against is the 0.1.40 code arm of
the paired report — same day, same arguments, no such variable.

| Configuration | Prompt tok/s without → with `STRATA_KV_PREFETCH=1` |
| --- | --- |
| 32768-prompt | 3136.3 → 3001.8 (-4.3%) |
| 131072-prompt | 3118.5 → 3052.3 (-2.1%) |

With the option on: 32768-prompt 3001.8 tok/s (range 2936.9–3063.2, n=3), decode 113.8 (110.1–128.5), TTFT 10.51 s
(10.31–10.75);
131072-prompt 3052.3 tok/s (3030.3–3059.4, n=3), decode 106.5 (87.0–113.4), TTFT 41.26 s (one repeat waited behind
an interactive client's request and reached 85.6 s). Peak VRAM 22040 MiB, peak RAM 6.59 GiB. Recall was not checked
in this arm. Raw runs: [runs-kvpref.json](runs-kvpref.json), config: [config-kvpref.json](config-kvpref.json).

The option overlaps streamed KV uploads with prefill (docs/KV_PREFETCH.md), so it can only help when part of the KV
cache lives in system RAM — that is `--kv-resident`. This configuration has no `--kv-resident`: its KV cache stays
in VRAM and there is nothing to overlap, so the measured result is a small loss. That matches the -4…-5% the
option's author reported on an RTX 5070 with `--kv-resident 32768` and the reason it stays off by default: on a card
whose KV fits in VRAM the extra streams and events cost more than they save. Conclusion for this machine: leave it
off. It would have to be re-measured on this PC with `--kv-resident` before any conclusion about the option itself
could be drawn.

## 2. `"parallel": 2` (batch slots, #465) and `--batch-mtp` (#846)

[conc_bench.py](conc_bench.py) sends N requests **at the same time** (N = 1, 2, 3, 4), each a distinct prompt of
about 4,000 tokens carrying a random marker, 256-token output cap, `temperature=0`, `reasoning_effort=none`; two
repeats per level; a 2,000-token warm-up before the levels. The two arms differ from the working config by two keys
only: `"parallel": 2`, and `--batch-mtp` in `args`.

- TTFT: time to the first non-empty streaming delta.
- Per-client decode: generated tokens / (whole request time − TTFT), measured at the client. A batched request does
  not report per-slot timings in the stream, so this column is client-side, not the engine's
  `predicted_per_second`.
- Total: all tokens generated in the level / the level's wall time. This is the number that says whether the server
  served more per second, which is what batch slots are for.

### What the engine said

`"parallel": 2` (no `--batch-mtp`):

```
strata generate: --batch: 2 slot sessions on CUDA0 (2.88 GiB each); 7.98 GiB free
strata generate: --batch 2: the slot sessions take 5.76 GiB of VRAM on CUDA0 that the expert cache would otherwise hold
strata generate: expert cache 3329 slots, 5.41 GiB of VRAM; policy is PROFILE
strata verify: batch windows of up to 2 sequences (layers [0, 48))
```

Without it the same start reports `expert cache 6946 slots, 11.17 GiB of VRAM`. So on this model at this context a
slot costs 2.88 GiB and the expert cache loses more than half of it. The exact resident count depends on the expert
profile state at the moment of the start — the paired speed reports for this configuration show 6,915–6,926 slots;
the numbers above are the two starts of this experiment, taken from the same server log.

`--batch-mtp` on top of it — the draft layer cannot be used for an admission, and the request never really starts:

```
strata batch: MTP admission for slot 0 failed: mtp: unsupported native MMVQ GGML type
```

After that the engine died and the server started it again; the two clients that were in flight finished about 300 s
later having received 1 and 5 tokens. The MTP layer here is the pack setup downloaded for this model
(`Strata-data/mtp/rt`), and its quant is MMVQ. This is the same family as #1012 (a `--batch` request whose engine
died during its prompt read ends only after 300 s) and #1063 (`--batch-mtp` exits on the first admission with a
draft-vocab problem), so it is reported as a data point, not as a new bug claim. The fixed behaviour on 0.1.40.1 is
measured in [2026-10-06-community-rtx4090-toolcall-hotfix](../2026-10-06-community-rtx4090-toolcall-hotfix/README.md).

### Results

| Level | `"parallel"` off | `"parallel": 2` |
| --- | --- | --- |
| 1 client | wall 4.7-5.2 s, TTFT 2.3-3.0 s, decode 93-95 tok/s | wall 6.6-7.1 s, TTFT 2.7-3.5 s, decode **65-72** tok/s |
| 2 clients | wall 8.2-8.6 s, TTFT med 4.2-4.3 s (max 6.6), decode 94-100 tok/s, total 46-52 tok/s | wall 9.5-10.2 s, TTFT med 3.8-4.1 s (max 5.5), decode 43-46 tok/s, total 45-48 tok/s |
| 3 clients | wall 53.9-58.9 s, TTFT med 47.9-52.3 s, total **12.1** tok/s | wall **13.6-14.6 s**, TTFT med 4.9 s (max 11.4), total 45-46 tok/s |
| 4 clients | wall 50.7-53.5 s, TTFT med 41.9-44.6 s, total **16.1-18.0** tok/s | wall **17.2-17.7 s**, TTFT med 7.7-7.9 s (max 13.7), total 48-50 tok/s |

- One client: `"parallel": 2` costs about **28%** of decode speed (93–95 → 65–72 tok/s). This is the expert cache
  shrinking, exactly the trade the docs describe for a card whose experts mostly fit in VRAM.
- Two clients: nothing is gained here (46–52 → 45–48 tok/s total), the waiting is only moved a little.
- Three and four clients: the wall time is 3–4 times lower and the total throughput is 3–4 times higher. This is
  where the option pays for itself on one GPU.
- `--batch-mtp`: unusable with this MTP pack, see above.

Raw runs: [runs-parallel-off.json](runs-parallel-off.json), [runs-parallel-2.json](runs-parallel-2.json),
[runs-parallel-2-batch-mtp.json](runs-parallel-2-batch-mtp.json).

### What this PC does with it

The machine is used by one interactive client most of the time, so the 28% single-stream cost is the number that
matters here and the option stays off. It is kept as a measurement for the case where a second and a third client
appear regularly.

## Limitations

- **Levels 3 and 4 with `"parallel"` off are contaminated by this machine's own interactive session**: a request of
  about 100,000 tokens from another client (33–38 s of engine time) was served between the measured runs, and the
  server answers one sequence at a time. That is why their TTFT medians reach 52 s. With two slots the same agent
  request was served alongside the measured ones. So levels 3 and 4 compare in direction, not in numbers; levels 1
  and 2 are clean.
- Per-client decode in the batched arm is client-side (see above), so it is comparable between the arms only as a
  direction. The engine-side numbers of this PC are in the paired reports linked at the top.
- Two repeats per level, one machine, one model, one context. No answer-quality check was run here; the paired
  reports cover recall (6/6 at 32K and 128K).
- The KV-prefetch arm is one configuration without `--kv-resident`. It says what the option does to this
  configuration, not what it would do to a configuration that can use it.
