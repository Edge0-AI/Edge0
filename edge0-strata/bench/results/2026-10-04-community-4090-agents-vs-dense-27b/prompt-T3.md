You are working in llm-tune, a Claude skill that teaches quality-aware tuning of local LLMs from measured
evidence. Work ONLY inside the git worktree WT (cd there first). Read-only access to the evidence below; do not
edit anything outside WT, do not push. Commit your finished work on the current branch (no attribution trailers).
Read README.md and skill/SKILL.md first and follow the repo's conventions (tables in skill/references/tables/,
findings.md, evidence.md index, CORRECTIONS.md, engine-specific marking, "never state a number you did not read").

Task: fold today's new Strata measurements into the skill:
- <lab-repo>/state/evals/2026-10-04/strata-calib/RESULTS.md (+ calib.log, results.jsonl, bench.txt): calibration of
  --pcie-frac / --spec-min-p, and a learned expert profile that hurt decode at depth.
- <lab-repo>/state/evals/2026-10-04/strata-pr783-depth/RESULTS.md (+ results.jsonl): engine PR #783 and #796 at depth.
- <lab-repo>/state/evals/2026-10-04/strata-pr783/RESULTS.md: #783 on short prompts.
Put each new measurement where the skill keeps that kind of data, update the decision steps / traps / evidence index
where the new evidence changes or sharpens advice, and mark what is engine-specific or single-run. Every number you
write must match its source file exactly and cite it. Do not generalise beyond what was measured.
End with a short report: files changed, each new claim with its source, anything you chose not to include and why.
