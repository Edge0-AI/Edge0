# First real `strata bench run` on the released 0.2.1 (terminal 0.2.1, engine 0.1.46 cuda12 Linux packages)

Model Qwen3.8-Flash-Next IQ3_XXS (hand-written run config that mirrors setup: int8 KV, MTP spec 4; z440 layer split auto, P2003 resident experts),
`strata bench run --profile` (the runner starts and stops the engine), protocol `strata-qwen38-thinking-v1-draft.1`, track `custom`. Dated 2026-10-20.

| | P2003 (RTX 3060 12 GB, Core Ultra 7 265) | z440 (2x RTX A4000, Xeon E5-2643 v3) |
|---|---|---|
| `short` decode tok/s (median of 3) | 64.4 | 90.3 |
| `prompt-4k` prompt / decode tok/s, TTFT | 962 / 55.4, 4.28 s | 1158 / 74.0, 3.56 s |
| `prompt-16k` prompt / decode tok/s, TTFT | 1103 / 56.1, 14.9 s | 1852 / 75.8, 8.9 s |
| sanity checks | 12 of 12 | 12 of 12 |
| required suite after loading | 3 min 12 s | 2 min 24 s |
| optional concurrency (4 at once, server capacity 1), total tok/s | 54.5, 56.4, 56.5 | 72.3, 69.8, 75.4 |

Numbers are from `run3-with-fixes` on each box (run1 and run2 agree within a few percent). Each box has three runs:

- `run1-release-archive-as-shipped`: the release terminal archive untouched. Status `nonconforming`: the archive has no `COMMIT` file.
- `run2-with-commit-and-optional-concurrency`: the same plus a `COMMIT` file (the release commit, added by hand) and `--optional-concurrency`. Status `complete`.
- `run3-with-fixes`: the bench fixes of this branch overlaid on the release terminal. Status `complete`; storage, runtime, engine log and fresh-load line are filled.
  (The track note says "box: BOX" because the run script was not given the box name.)

## What the protocol and the runner got wrong (fixed on this branch unless marked)

1. The release archive carried no COMMIT, so every release run was nonconforming. `package_terminal.py` takes `--commit` / `STRATA_COMMIT` and refuses to build a distributable archive without one. The release must be rebuilt (or a COMMIT file added) for a conforming result.
2. The drive of the model files was not found on an LVM volume (`lsblk -s` nests two levels): "Storage: not measured" on z440.
3. No CUDA runtime on P2003: driver 615 prints `CUDA UMD Version`. Also `nvidia-smi` reports the newest CUDA the driver supports (13.0 / 13.4), not the library the engine ran on (12.9). Now: runtime = the engine's build runtime from `engine-component.json`, plus `cuda-driver-supports`.
4. `engine.log` in the result was empty (`serve --quiet`). Now the run config's log from the start of the run.
5. A freshly started engine was reported as "could not tell whether the server was freshly loaded".
6. The protocol said 20 to 30 minutes after loading, unmeasured. Measured: 2 to 4 minutes (plus 1.5 to 2 minutes for the optional stage and 1.5 to 5 minutes of model hashing before the clock starts). Text corrected in the protocol, COMMUNITY_BENCHMARKS.md, BENCH.bat and bench.sh.
7. 35 of 36 4K/16K performance requests (and 5 of 18 `short`) stop at the 512-token cap while still reasoning, so "Total s" is the time to the cap and "first answer" is mostly missing. The readouts now carry a `Capped` column and say so. The cap itself is part of the frozen protocol (a change needs a new protocol ID); a larger cap or a non-reasoning performance prompt is the proposal if the owner wants answer times.
8. The `short` prompt is about 80 tokens, so its prompt rate (130 to 190 tok/s) is the fixed cost of a request, not a speed. The readouts leave it out (results.json keeps it).
9. Not fixed: the runner hashes the model files while the engine is already loaded (RAM is tight on a 46 GB box with a 44 GB model; it worked) and the hash time is outside `required_duration_s`; run-to-run decode varies up to 9 % between runs with the same engine (adaptive expert cache), more than the protocol's min-max spread within one run shows.
