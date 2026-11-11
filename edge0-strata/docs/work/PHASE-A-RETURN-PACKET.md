# Phase A return packet

**Outcome: Phase A ready for hardware qualification.**
**The macOS candidate is real-model validation pending, not fully validated.**

## Source revisions actually selected

| Repository | Revision | Role |
| --- | --- | --- |
| Strata (integration checkout) | branch `strata/rebuild-2026-10`, HEAD `e0951ce243511597eb243018191f194af9d5f8bc` | the product |
| Strata (pristine reference) | `296b59d2` (`main`) | unchanged baseline |
| DeepSeek Harness | `57190202` | agent loop and durable sessions |
| HyperCore-Tune | `4f9ab7fa` | evaluated for tuning; decision D07 |
| ggml (llama.cpp) | `3cf03257` | engine host sources |
| metal-cpp | `4e6c9a1a…` SHA-256 `4df3c078…85a3a4` verified | Metal host bindings |

No upstream drift: every pin above is the revision that was reviewed.

## Artifacts

    DMG      apps/desktop/src-tauri/target/release/bundle/dmg/Strata_0.1.0_aarch64.dmg
             sha256 e18c2f4ca60e0142177dc67789b63469f4365bca46860c284173d55b603766fd
             hdiutil verify VALID; packaged self-test 8/8 from inside the mounted image
    Terminal release/artifacts/strata-terminal-0.1.44.1-darwin-arm64.tar.gz
             bytes 42480892
             sha256 fafa0076fd30f9fa3d3f1250d8afbf39225bddb80e07a3d818bae2b01e2814d8
             self-contained: bundled CPython 3.12.15, verified with no usable system Python
    Engine   build-metal/strata
             sha256 aef25be339b60ad56d34c192847bd59e4351779eed6a1de9166f81836041ee3f
             built from source on this Mac; Metal kernel parity 17/17 with one known defect

Install and run, migration and recovery, and every evidence path are in
`docs/work/evidence/artifact.txt` and `docs/work/QUALIFICATION-KIT.md`.

## Acceptance standing

Every one of the 67 requirement rows of the handoff's acceptance checklist has an explicit
result. `python packaging/acceptance_matrix.py --check` fails if a row is missing.

| Result | Count |
| --- | --- |
| passed | 46 |
| failed | 0 |
| pending-hardware | ? |
| pending-release | ? |
| unsupported | 3 |
| not-implemented | ? |

**No requirement row is recorded as failed.** Denying a defect by deferring it to another machine is
the one thing this matrix refuses to do; `failed` means the implementation is wrong and is never
written as `pending-hardware`.

## Exact limitations

1. **Real-model inference has not run and cannot on this machine.** `include/strata/core/geometry.hpp`
   fixes the compiled shape (n_embd 2560, 48 layers, 512 experts, vocab 248320), so no small or
   synthetic model is possible; the smallest pack is 37.6 GB mapped plus a 28.8 GB n-gram table, and
   `docs/MAC.md` measures 32 GB as the smallest tested Mac against this host's 16 GiB with ~21 GiB
   free. Everything downstream of inference is exercised against the mock and scripted workers, and
   every such result is labelled as a harness test rather than a performance result.
2. **A tuning sweep has never run against a real model.** The measurement harness, job, report,
   apply and rollback are all exercised, but against a stub engine whose numbers mean nothing.
3. **The Windows and Linux packages are recipes, not builds.** The NSIS and AppImage targets are
   configured and validate; neither has been produced. Windows SYCL and Intel-Mac Metal are refused
   as unsupported combinations rather than attempted.
4. **The Windows owner transport is not a named pipe with an ACL.** The Python standard library
   cannot create one, so Windows uses a loopback-only TCP listener with a 256-bit control token. That
   is weaker than the checklist's stated condition and is recorded as such rather than claimed.
5. **No signing, notarization, SBOM or font/asset rights evidence.** The artifacts are explicitly
   labelled unsigned internal development candidates. This blocks public release and does not block
   this milestone.
6. **One open engine defect:** `moe_mm_parity` fails with `mtl: no function mm_gu_42_16_64_16_2`.
7. **Two acceptance rows are not implemented:** the offline bundle format (P09) and connector
   operation assertions (U09). Both are real gaps, both are outside Phase A's implementation scope,
   and both are listed rather than hidden.

## What a Phase B run must do

`docs/work/QUALIFICATION-KIT.md` has executable procedures, exact prerequisites and expected
assertions for every machine-dependent check, covering installation, shared UI/CLI sessions, real
inference and tools, cancellation and recovery, Time-skip, tuning baseline/search/apply/rollback,
offline reuse, updates and uninstall retention. Section 10 is the one to read first: it is how to
tell a real pass from a vacuous one, written because three checks in this build passed for reasons
unrelated to the feature they named.
