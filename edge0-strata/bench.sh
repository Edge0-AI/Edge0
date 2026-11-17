#!/bin/sh
# Benchmark this Strata install the way it is used and write a report you can share (docs/COMMUNITY_BENCHMARKS.md).
# It starts the model with your settings (or uses the one already running), takes about 10 minutes and changes nothing.
# Options: --clients 4 (several requests at once), --long (a 32K prompt), --compare <older results folder>.
# This is the quick speed check. The standard benchmark (docs/BENCHMARK_PROTOCOL.md), whose results other people can compare with, is `strata bench run`.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  echo "Quick speed check. For results other people can compare with (the standard benchmark, a few minutes), run: .venv/bin/python strata.py bench run"
  echo
  exec .venv/bin/python tools/strata_bench.py "$@"
fi
echo "Strata is not set up here yet: run ./setup.sh first."
exit 1
