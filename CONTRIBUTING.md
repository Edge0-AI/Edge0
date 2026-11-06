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
| [`edge0-infer/`](edge0-infer/) | The inference engine (C++, GGML), its CLI and OpenAI-compatible server, the model asset tools, and the chat app examples for desktop, Android and iOS |
| [`docs/`](docs/) | Model design docs: MoE routing, the prerouter, and the model tiers |
| [`paper/`](paper/) | The technical report |

`edge0-infer/` is developed in a separate repository and synced here. Open
issues and pull requests for it in this repository as usual: an accepted change
is applied there first and arrives here with the next sync.

## Development setup

Build the engine and get the model assets as described in
[Getting started](edge0-infer/docs/getting-started.md). It covers macOS, iOS,
Android and Windows.

## Running the tests

The tests are CTest tests with a level label from `L0` (static checks) to `L4`
(acceptance on a device). From `edge0-infer/`:

```bash
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"     # no model assets needed
```

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

1. **Fork** the repo and create a feature branch off `main`
   (e.g. `fix/streaming-oob`, `docs/clarify-memory`, `feat/tool-calling`).
2. Make your change; **add or update tests** for new behavior.
3. Run the tests for the levels your change touches (at least `L0` and `L1`)
   locally.
4. Open a PR against `Edge0-AI/Edge0:main` and fill in the
   [PR template](.github/PULL_REQUEST_TEMPLATE.md). Link the issue it closes
   (`Fixes #NN`).
5. Keep PRs focused — one logical change per PR is easier to review and merge.

### Commit / PR style

This repo uses conventional-commit-style prefixes, e.g.
`fix(streaming): ...`, `feat(server): ...`, `docs(readme): ...`,
`perf(streaming): ...`, `chore: ...`. Please follow the same convention.

## Reporting security issues

Do not open a public issue for security problems. Email **samuel@edge0.ai**
instead.

## License

By contributing, you agree that your contributions will be licensed under the
project's [Apache-2.0](LICENSE) license.
