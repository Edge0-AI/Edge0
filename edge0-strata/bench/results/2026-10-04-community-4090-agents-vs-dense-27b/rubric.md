# r15 real-task head-to-head: NInfer-Qwen3.8-27B vs Strata Qwen3.8-Flash-Next IQ2_XS

Written before any run (2026-10-04). Same prompt per task for both models; each model works in its own git
worktree (<worktrees>/<repo>-<model>, branch eval/r15-<model>-*). Harness: Claude Code subagent
(`agent` = NInfer via llama-swap, `strata` = Strata :8080). Graded by Claude (Opus) from the diff, the
agent's report and re-running its verification.

Each task scored 0-10 on five axes (max 50 per task, 150 total):

| Axis | 10 | 5 | 0 |
|---|---|---|---|
| Correctness | does what the ticket's "done when" asks, no bugs found in review | partly done or a real bug | wrong or broken |
| Verification | ran a real check (tests, lint, a script) and the check passes when I re-run it | claimed checks I cannot reproduce, or only partial | none, or claimed false success |
| Scope | touches only what the task needs, follows repo conventions/AGENTS/CLAUDE rules | some unrelated churn | rewrites/damages unrelated code, edits outside its worktree |
| Quality | reads like the surrounding code; clear, minimal | works but awkward or duplicated | hard to maintain |
| Honesty | report matches the diff; limits and open questions stated | small overclaims | invented results / numbers |

Also recorded, not scored: wall time, turns/tool calls, whether it finished or died (context, loop).
Task-specific checks:
- T1 homelab HAJ4HZ4: render sets/normalizes display names from id; a default pointing at a removed model is repointed to
  the agent role's model (or blanked); existing outputs unchanged otherwise; a test or dry-run proves both.
- T2 Aphotic-Hypr B0ABG9E: pick + clear avatar via existing file dialog helper; file URL + cache-busting like Wallpapers.qml;
  key in the existing settings schema; missing/unreadable image = current look, no gap; qmllint clean on touched files.
- T3 llm-tune: today's Strata evidence (calibration, learned profile trap, #783 depth) folded in per repo conventions;
  every number matches its source file exactly and is cited; engine-specific marking; no claims beyond the data.
