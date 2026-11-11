# Rope scaling: needle recall past the trained 262K

The feature (PR #84, experimental): `--rope-scaling none|linear|yarn` extends the context past the model's
trained 262,144 positions by rescaling the rotary angles - llama.cpp's types, ggml's `rope_yarn`
arithmetic, one process config applied at the cos/sin table (default path) and as kernel constants
(`--native-rope`, the native indexer, prefill). The three tables here are the first end-to-end check
that a scaled run actually recalls past the trained end.

**Hardware:** RTX 4070 Ti SUPER 16 GB, Ryzen 9 5900X (no AVX-512 - the IQ3_XXS native pack runs its
expert kernels on AVX-2), 125 GB RAM, Linux. **Engine:** this branch built from source (the log records
`rope scaling ... --max-context ... against a trained context of 262144` at startup). The runs below
were taken on the rope tree built over the 0.1.17-era base; the branch then rebased onto main (engine
0.1.20) - the drift touches none of the rope files - and the rebase was re-validated: `rope_parity`
green, a 64-token greedy run token-identical to the plain 0.1.20 build, and the yarn 262k probe
repeated green (see the last table row). Baseline arm A also re-checked the default is the stock
model: a 64-token greedy run produces the same token ids as the ready-made 0.1.17 engine, and
`rope_parity` holds the `none` table bit-identical to the pre-scaling builder.
**Method:** `tools/needle_bench.py` against a running server (launch recipes in `config-*.json`),
`temperature 0`, thinking off, depth 50%, one probe per length.

| Arm | Scaling | Context | Probe | Prompt tokens | Result | Probe time |
| --- | --- | --- | --- | --- | --- | --- |
| A | none | 262,144 | 1k @ 50% | 1,442 | FOUND | 4 s |
| A | none | 262,144 | 32k @ 50% | 31,763 | FOUND | 27 s |
| A | none | 262,144 | 128k @ 50% | 128,328 | FOUND | 86 s |
| B | linear, factor 2 | 560,000 | 262k @ 50% | 274,222 | FOUND | 232 s |
| B | linear, factor 2 | 560,000 | 512k @ 50% | 527,051 | FOUND | 619 s |
| C | yarn, factor 2 | 560,000 | 262k @ 50% | 274,215 | FOUND | 236 s |
| C | yarn, factor 2 | 560,000 | 512k @ 50% | 526,988 | FOUND | 615 s |
| C | yarn, factor 2 | 560,000 | 262k @ 50% (the rebase re-check) | 273,708 | FOUND | 235 s |

All 8 of 8 probes found the needle; JSON per arm: `needle-A-none.json`, `needle-B-linear2.json`,
`needle-B-linear2-512k.json`, `needle-C-yarn2.json` (+ the rebase re-check in
`needle-C-yarn2-rebased.json`); server and engine logs alongside.

What this establishes, and what it does not:

- **Past the trained end the model still reads.** 512k probes are 2.01x the trained 262,144 - position
  interpolation (B) and YaRN (C) both hold a mid-context lookup there.
- **Scaled runs within the trained range are a slightly different model.** `yarn`'s mscale applies at
  every position (a 64-token greedy prompt diverges from the stock run after ~19 tokens); arm A is the
  stock model, arms B/C are not, by design.
- **One depth, one needle, one machine.** A single probe per (length, arm) at depth 50% says "recall
  works there", not "no degradation anywhere" - a full depth sweep and a longer-length ladder (768K,
  1M) on the bench machine (64 GB RAM there vs 125 GB here, so 262K IQ3_XXS fits either way) is the
  natural follow-up. 512K prompts prefill at ~850-950 tokens/s on this card; the probe time is almost
  entirely prefill.
- **The `560,000` context** in arms B/C is headroom for the needle harness's tokenizer overshoot
  (a "512k" probe tokenizes to ~527k); 524,288 exactly would refuse the probe.
