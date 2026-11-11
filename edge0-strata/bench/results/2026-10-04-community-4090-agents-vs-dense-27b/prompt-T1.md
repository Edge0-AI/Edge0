You are working on a ticket in the homelab repo. Work ONLY inside the git worktree WT (cd there first; it is a
full checkout on its own branch). Do not edit <lab-repo> or any other path, do not SSH anywhere, do not push.
Commit your finished work on the current branch (no attribution trailers). Read CLAUDE.md/AGENTS.md in the worktree.

Ticket HAJ4HZ4 - "lsw_consumers should own model display names and default models":
Recurring drift found 2026-10-01: after Coder-Pruned was removed, Pi/Empryo model entries kept name
'agent (Qwen3.6-Coder-Pruned-262K)' and DSH/Empryo defaults still pointed at the removed model, because
lsw_consumers renders only id/input/context. Extend the render to set/normalize the model display name from the
id, and to repoint a default model that no longer exists to the agent role's model (or blank it). Then a roster
change cannot leave a harness pointing at a ghost.

The code is scripts/llama-swap/lsw_consumers.py (and whatever it calls). Prove it works without touching live
harness configs: tests or a dry run against copies/fixtures. Existing behaviour for valid entries must not change.
End with a short report: files changed, how you verified it (exact commands and results), open questions.
