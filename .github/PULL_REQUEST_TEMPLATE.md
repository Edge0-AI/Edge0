<!-- Thanks for contributing to edge0! Keep this short but complete. -->

## What & why

<!-- What does this PR change, and what problem does it solve? Link issues: "Fixes #NN". -->

Fixes #

## Area

- [ ] Engine, CLI or server (`edge0-infer/`)
- [ ] Model asset tools (`edge0-infer/tools/assets/`)
- [ ] Chat app examples (`edge0-infer/examples/`)
- [ ] Model docs (`docs/`)
- [ ] Other

## How was it tested?

<!-- Commands you ran, platform and backend, hardware/chip. Paste the `edge0-infer bench` summary for perf changes. For edge0-145b / edge0-120b, which run through edge0-strata, paste the community benchmark report (BENCH.bat / ./bench.sh) instead: `edge0-infer bench` uses the native loader and cannot measure a delegated model. -->

## Checklist

- [ ] `ctest --preset cpu -L "L0|L1"` passes locally in `edge0-infer/` (engine changes)
- [ ] Ran the higher test levels this change affects (see `edge0-infer/docs/testing.md`)
- [ ] Added/updated tests for new behavior
- [ ] Updated relevant docs / READMEs (incl. translations if user-facing)
