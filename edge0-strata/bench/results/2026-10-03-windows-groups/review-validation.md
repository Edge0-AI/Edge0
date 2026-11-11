# Host-restoration review follow-up (2026-10-03)

Source: [d72bd06](https://github.com/Yasei-no-otoko/Strata/commit/d72bd0668550bca29e9e69d20b970ce7ffa7e6fd), [PR #626 review](https://github.com/Niko1221/Strata/pull/626#discussion_r4173337650).

The host now saves and restores its explicit Windows CPU Set selection. An originally empty selection is cleared on restoration, returning to process defaults without changing the caller's hard affinity or implicit Windows 11 all-group eligibility. Pool-owned worker threads retain hard processor-group affinity.

Validation on the same Windows 11 / 3990X machine:

- Full Release HIP engine build completed with the existing ROCm 10.2.0a20260930 / gfx1030 configuration.
- `pool_affinity_test` passed. It verifies empty and nested/multi-CPU selections, inherited process defaults, worker group pins, and preservation of existing hard affinity. After host restoration, actual execution was observed on group 0 / processor 0 and group 1 / processor 0. [Verbose test output](cpu-tests-review.txt).
- A temporary negative control explicitly narrowed the test-owned host thread to a single hard group after restoration. The same test failed with `lost CPU/group eligibility after host restore`. The injection was removed, the exact committed source was rebuilt, and the test passed again. [Negative-control output](cpu-tests-review-negative-control.txt).
- `pool_stress` returns success after reporting `SKIPPED` on this CPU because it lacks the required AVX-512 features. This does not establish that the stress workload passed. The earlier CTest-only records conceal this distinction; the report text now makes it explicit. The AVX-512 pool self-test is also skipped.
- Linux compilation remained unverified because the configured WSL image was unavailable.

These are build and affinity-regression checks. The inference measurements elsewhere in this directory predate this host-restoration change and were not rerun for this revision. No new throughput or coding-quality result is claimed here. Local account names in test paths are redacted.
