# Contributing to edge0

Thanks for your interest in contributing! edge0 is an open-source streaming MoE
inference framework, and contributions of all kinds are welcome — bug reports,
performance measurements, documentation fixes, new model support, and new
platform/backend work.

## Ways to contribute

- **Report a bug / performance issue / request a feature or model** — open an
  issue using one of the [issue forms](.github/ISSUE_TEMPLATE/). The forms ask
  for the platform, backend, chip and memory — these fields resolve the large
  majority of reports, so please fill them in.
- **Ask a question or share results** — please use
  [GitHub Discussions](https://github.com/Edge0-AI/Edge0/discussions) rather
  than an issue.
- **Send a pull request** — see the workflow below.

## Repository layout

| Path | What |
|---|---|
| [`edge0-infer/`](edge0-infer/) | The program you run: the native inference engine (C++, GGML), its CLI and local OpenAI- and Anthropic-compatible server, the model asset tools, and the chat app examples for desktop, Android and iOS |
| [`edge0-strata/`](edge0-strata/EDGE0.md) | The engine for `edge0-145b` and `edge0-120b` (NVIDIA GPU on Linux or Windows, Metal on macOS), a fork of Strata that `edge0-infer` starts for these models |
| [`docs/`](docs/) | Model design docs: MoE routing, the prerouter, and the model tiers |
| [`paper/`](paper/) | The technical report |

## Development setup

Build the engine and get the model assets as described in
[Getting started](edge0-infer/docs/getting-started.md). It covers macOS, iOS,
Android, Linux and Windows. For `edge0-145b` and `edge0-120b`, also build
`edge0-strata/` as [`edge0-strata/EDGE0.md`](edge0-strata/EDGE0.md) describes.

## Running the tests

The tests are CTest tests with a level label from `L0` (static checks) to `L4`
(acceptance on a device). From `edge0-infer/`:

```bash
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"     # no model assets needed
```

Continuous integration runs `L0`, `L1` and most of `L2` on every pull request.
[Testing](edge0-infer/docs/testing.md) describes every level, what each needs
(the tiny test fixtures, the real assets, a device) and the full regression
commands.

## Adding a new model

A model has two parts in `edge0-infer/`: its asset tools in
`edge0-infer/tools/assets/<model>/` (see
[Adding a model](edge0-infer/tools/assets/README.md#adding-a-model)) and its
computation in `edge0-infer/src/models/<model>/` (see
[Architecture](edge0-infer/docs/architecture.md)). Please include a short note
on the architecture (layers / experts / routing width / attention type) in your
PR.

## Pull request workflow

Fork the repo, branch off `main` (for example `fix/expert-slot-oob` or
`feat/tool-calling`), and open the PR against `Edge0-AI/Edge0:main`. Fill in
the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

`edge0-strata/CONTRIBUTING.md` describes the evidence expected for changes in
that directory. The rules on this page decide whether a pull request is
reviewed.

The **PR hygiene** check enforces the items marked below. A failing check is
labeled `needs-info` and is not reviewed. A `needs-info` pull request that
stays inactive is closed. Pull requests opened before 2026-11-19 08:00 UTC
keep the previous terms: the check does not apply to them, and they do not
count toward the two-open limit.

1. **Agree on behavior changes before writing code.** Open an issue and wait
   for a maintainer to confirm the change is wanted. A documentation-only PR
   may skip this. A one-line fix for an obvious bug may go straight to a PR,
   and should still link the issue when one exists.
   **Enforced:** the body contains `Fixes #123`, `Closes #123`, or
   `Related to #123`, unless the title type is `docs`.
2. **Batch related fixes.** Put related changes in the same area (`serve`,
   `text`, `desktop`, `assets`, `strata`, and so on) in one pull request, with
   one commit per fix. Do not open a separate pull request for each finding.
3. **Keep at most two pull requests open.** Wait for review on those before
   opening another. Maintainers are not counted.
   **Enforced** for everyone else.
4. **Add or update tests** for new behavior, and run the levels the change
   touches (at least `L0` and `L1`) locally.
5. **Show what you actually ran.** Fill in "How was it tested?" with commands,
   platform, and results. A performance change also needs an `edge0-infer bench`
   summary, or the community benchmark report for a model that runs through
   `edge0-strata`.
   **Enforced:** the section is not empty, and a `perf` title mentions a bench
   result.
6. **Use a conventional title.** Optional `[Area]` prefix, then
   `type(scope): summary`. `type` is one of `fix`, `feat`, `perf`, `docs`,
   `chore`, `test`, `tests`, `ci`, `build`, `refactor`, `revert`. The scope
   may be omitted (`ci: run L0 on pull requests`).
   **Enforced.**

### Closed without a full review

- A behavior change with no agreed issue.
- Micro-optimizations or refactors that do not report a measured problem.
- Style-only sweeps, drive-by cleanups, and a second pull request for a
  change that is already open.
- More than two open pull requests from the same contributor. The oldest two
  stay in the queue; the rest stay `needs-info` until some are closed.

You must be able to explain the diff and answer review questions about it.
Model-assisted edits are fine on those terms. A tree-wide scan split into many
pull requests is closed.

### Review order

Correctness bugs with a reproduction come first, then work on an issue a
maintainer already accepted, then performance changes that include numbers.
Everything else waits. Reviews run on the existing queue; new pull requests
do not jump it.

### Commit style

Use the same prefixes as the title, for example `fix(experts): ...`,
`feat(server): ...`, `docs(readme): ...`, `perf(decode): ...`.

## Reporting security issues

Do not open a public issue for security problems. Email **samuel@edge0.ai**
instead.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache-2.0](LICENSE) license. Contributions to `edge0-strata/` are
licensed under that directory's [MIT](edge0-strata/LICENSE) license.
