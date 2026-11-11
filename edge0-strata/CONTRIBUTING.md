# Contributing to Strata

**AI assistance is welcome. The contributor is responsible for understanding the change and supplying real evidence.**

These rules apply to issues, pull requests, benchmark reports, and the agents helping prepare them.
You should be able to explain every changed line and answer review questions about it.

## Before starting

1. Search [issues](https://github.com/Niko1221/Strata/issues) and
   [pull requests](https://github.com/Niko1221/Strata/pulls), including closed discussions. Add new evidence to an
   existing thread when it covers your problem.
2. Read [AGENTS.md](AGENTS.md) and the documentation for the component you want to change.
3. Get maintainer agreement on scope before substantial features, refactors, architecture changes, or changes to
   defaults. Describe the user problem, proposed behavior, and tradeoffs in an issue first. Agreement on scope
   does not guarantee a merge. Focused fixes and benchmark reports can go straight to a PR.
4. Keep one problem per PR. Update that PR after feedback; do not open competing attempts for the same change.

Do not resubmit a declined proposal under a different title or through another agent. If new evidence changes
the situation, bring it to the original discussion and let a maintainer decide whether to reopen it.

## Report a problem people can investigate

Use the matching [issue form](https://github.com/Niko1221/Strata/issues/new/choose). A bug report should include:

- The smallest steps or command that reproduce the problem, and the expected and actual behavior.
- Strata version or commit, OS, GPU and VRAM, and relevant model, quantization, and configuration.
- The exact error and relevant logs, with credentials and private content removed.
- Whether you reproduced it yourself. Label a finding based only on reading code or a scanner as
  **unconfirmed static analysis**, and explain the suspected path without presenting it as an observed failure.

Do not invent reproduction steps, errors, affected hardware, or severity. If you cannot reproduce something,
say what you inspected and what would verify the hypothesis. Feature requests should explain a concrete user
need and the current workaround. Questions should say what you tried and where you got stuck.

Example structure, with placeholders to replace with actual observations:

```text
Problem: <one observed behavior>
Environment: <commit, OS, GPU/VRAM, model/quant, relevant settings>
Reproduce: <exact command or short sequence>
Expected: <documented or intended behavior>
Actual: <observed behavior and exact error>
Evidence: <sanitized log attachment>
Reproduction status: <reproduced locally OR unconfirmed static analysis>
```

## Titles

Every issue and pull request title starts with its area in square brackets, then a short description: `[Engine]`, `[GUI]`, `[TUI]`,
`[Benchmark]`, `[Docs]` or `[Other]`. For example `[Engine] fix decode stop on long prompts` or `[GUI] keep the model list after a tab
switch`. The issue forms ask for the area and a check on the title fails when the prefix is missing.

## Prepare a focused pull request

Lead with the problem and resulting behavior. Explain why the approach is needed, the important limits, and
any compatibility effects. Link the issue and scope agreement when applicable. A file-by-file narration or
an AI-generated summary of the diff is not a substitute for this explanation.

- Keep unrelated formatting, cleanup, dependency changes, and speculative abstractions out of the PR.
- Add documentation where it changes what a user or developer needs to know. Avoid repeated summaries,
  speculative roadmaps, and generated documentation with no actionable information.
- Preserve existing behavior outside the intended change. New features are opt-in; follow the default-path
  and performance requirements in [AGENTS.md](AGENTS.md).
- Read the complete diff, including generated files, before submitting. Remove debugging output, secrets,
  private paths, and accidental files.
- Prefer a small relevant regression check over tests that only restate the implementation. Documentation-only
  changes need link, example, and consistency checks; they do not require GPU builds.

## Show the validation you actually performed

List exact commands, the revision and environment tested, and observed outcomes. Keep **passed**, **failed**,
and **not run** separate. Include relevant output or a log attachment. A command you suggest running is not
evidence that it passed; a CI run on a different revision is not validation of the submitted code.

Build every backend a code change touches, including CUDA, HIP, and SYCL where applicable, before requesting
review. For changes that need hardware execution, report that execution separately from compilation. A mock,
static check, or successful build does not establish hardware correctness. If you lack a required environment,
keep the PR as a draft, identify the missing checks, and ask for targeted help in the same thread. Do not claim
compatibility you have not tested.

Performance claims need before/after runs on the same hardware and workload, with every changed setting
identified. Preserve the individual runs, failures, and variability. Follow the
[community measurement guide](docs/COMMUNITY_BENCHMARKS.md); standard submissions will use the
[benchmark protocol](docs/BENCHMARK_PROTOCOL.md) once it is activated. One fast run does not justify changing a
default or claiming a speedup for other machines.

Example PR body; placeholders are not test results:

```text
Problem and result: <trigger; behavior before and after the change>
Related discussion: <issue URL and scope agreement, when required>
Approach: <why this fix; relevant compatibility effects>
Validation:
  Passed: <exact command, revision/environment, observed result, log>
  Failed: <exact command and failure, or none observed>
  Not run: <check, reason, and what remains unverified>
Limits: <known limitations or remaining work>
```

## Rules for AI-assisted work

Use agents to help investigate, implement, and explain work that you can review. Before submitting, verify
their claims against the source, actual command output, or a cited primary reference.

Never submit fabricated measurements, invented test results, fake logs, or copied claims you have not checked.
Do not turn a repository scan into a batch of speculative issues or PRs. Consolidate related findings, establish
their effect, and submit the useful evidence. Do not present an agent's confidence or another model's approval
as validation. Keep review replies concise and address the requested change or evidence directly.

Maintainers may request a smaller change, return a PR to draft, or close a submission that lacks evidence,
duplicates existing work, or ignores agreed scope. A smaller, understandable contribution with honest limits
is easier to review and maintain.

## Share a benchmark

A results-only PR should contain the human report, raw measurements, sanitized configuration, and supporting
logs. Keep it separate from engine changes and include poor results and failed requests.

The [current community benchmark guide](docs/COMMUNITY_BENCHMARKS.md) remains available for measurements made
with today's tools. The [new protocol](docs/BENCHMARK_PROTOCOL.md) defines the draft standard and future approval
workflow. Its draft status must remain visible: existing reports are not automatically standard or approved.
