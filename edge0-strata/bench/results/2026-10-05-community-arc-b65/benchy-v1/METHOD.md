# Public benchy-v1 fixtures on the qualified B65 runtime

Uses the exact 20/2185 raw token IDs from maxfridbe/Strata_B70 at79ad9d5aff716292602895b19860c7661bacc5ed, sycl/bench/v1. This is our qualified native serving protocol benchmark using those fixtures, not an unmodified sycl/benchy.sh run. There are three fresh engine processes. Within each process:2185/256 first,20/256 second, exact canary, graceful QUIT, native exit0 and independent five-second delayed-fault observation. Prefix/adaptation caches off; fixed expert ranking/residency. The first long request includes native graph capture; no CLI graph pre-capture. OS page cache is retained, loading excluded. Native prompt/decode timings come from DONE, output IDs and counts checked, all actual generated counts published. Three repeats per measured shape, medians/ranges. Raw logs/content RAM-only; only numeric lines/hash/config receipts persist.

The portable benchmark.py preserves the measured generation/timing/lifecycle logic while accepting source/profile/output/RAM paths. It is a reproduction adapter, not a second measured run. reference-public-hashes.json records the expected output IDs for our pinned artifacts/runtime. Use a bounded exclusive-GPU/resource supervisor; this script does not stop your other services or install machine guards. Native source/model/MTP/build/worker settings are the diagnostic-off profile in../BUILD.md.

After setting STRATA_ROOT/BENCH_ASSETS and the diagnostic-off environment from that BUILD.md, run:

```sh
python benchmark.py --source "$STRATA_ROOT" --profile profile.json --ram /run/your-benchmark --out /absolute/path/to/new-results
```

The fixture source hash and usage do not certify identical weights, compiler, driver, generated continuation or timing setup on another machine. Max's published numbers have one run per row and different context/prefill/host settings.
