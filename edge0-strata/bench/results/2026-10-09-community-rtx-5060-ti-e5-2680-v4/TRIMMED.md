# Trimmed artifacts

The benchmark script was run with its normal raw capture enabled. To keep the
community-results folder small, the following local artifacts are not included:

- warmup-request.json and warmup-raw.json
- tokens-*-request.json
- tokens-*-raw.json
- server engine logs and launcher stdout/stderr
- local model shards, generated packs, MTP runtime and expert-profile binaries

The compact results.json, summary.json, initial server status, sanitized measured
configs, unchanged benchmark script and engine BUILD metadata are included for both arms.
No measured row was removed from the compact results files.
