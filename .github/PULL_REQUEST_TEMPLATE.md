<!--
The PR hygiene check fails unless all of these hold:

- the title matches type(scope): summary, with an optional [Area] prefix
- the body links an issue (`Fixes #123` or `Related to #123`), except a docs title
- "How was it tested?" describes commands and results you actually ran
- a perf title includes a bench summary

Related fixes in one area belong in this pull request.
Keep at most two of your pull requests open.
See CONTRIBUTING.md.
-->

## What & why

<!-- What does this PR change, and what problem does it solve? Link the agreed issue. docs-only PRs may omit it. -->

Fixes #

## Area

- [ ] Engine, CLI or server (`edge0-infer/`)
- [ ] Model asset tools (`edge0-infer/tools/assets/`)
- [ ] Chat app examples (`edge0-infer/examples/`)
- [ ] Model docs (`docs/`)
- [ ] `edge0-strata/`
- [ ] Other

## How was it tested?

<!-- Commands you ran, platform and backend, hardware/chip, and the result. Paste the `edge0-infer bench` summary for perf changes. For edge0-145b / edge0-120b, which run through edge0-strata, paste the community benchmark report (BENCH.bat / ./bench.sh) instead: `edge0-infer bench` uses the native loader and cannot measure a delegated model. -->

## Checklist

- [ ] Linked an agreed issue, or this PR is documentation-only
- [ ] Related fixes in this area are in this pull request, one commit per fix
- [ ] This is one of at most two open pull requests from me
- [ ] `ctest --preset cpu -L "L0|L1"` passes locally in `edge0-infer/` (engine changes); CI also runs these and the `L2` tests not labeled `local-only`
- [ ] Changes to model computation: ran the `local-only` answer suites (`ctest --preset cpu -L local-only`) on Apple silicon with i8mm (M2 or later), which CI cannot judge
- [ ] Ran the higher test levels this change affects (see `edge0-infer/docs/testing.md`)
- [ ] Added/updated tests for new behavior
- [ ] Updated relevant docs / READMEs (incl. translations if user-facing)
