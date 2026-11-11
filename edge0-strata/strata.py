"""Strata's small, dependency-free terminal entry point.

Run ``python strata.py --help`` from a source checkout. Optional terminal UI
dependencies are imported only by ``monitor``.
"""

import importlib.util
import sys

#: What the API server (serve, bench) imports. Everything else in the CLI is standard library.
SERVER_PACKAGES = ("jinja2", "yaml", "regex", "psutil", "requests", "numpy", "PIL")


def _preflight(argv):
    """A missing package is a typed, actionable error, never an install: starting does not run pip."""
    command = next((a for a in argv if not a.startswith("-")), "")
    if command not in ("serve", "bench"):
        return 0
    missing = [m for m in SERVER_PACKAGES if importlib.util.find_spec(m) is None]
    if not missing:
        return 0
    sys.stderr.write(
        "strata: the interpreter %s is missing packages the API server needs: %s.\n"
        "        Starting and serving never install anything. Use the Strata terminal archive (it carries them),\n"
        "        or install them yourself: python -m pip install -r requirements.txt\n"
        "        (status, monitor, timeskip and tune plan do not need them)\n" % (sys.executable, ", ".join(missing)))
    return 5


if __name__ == "__main__":
    # a newer Control installed by `strata update` takes over before anything heavy is imported (stdlib only)
    from strata_cli.handover import maybe_hand_over, refuse_old_updater
    refused = refuse_old_updater(sys.argv[1:])
    if refused is not None:
        raise SystemExit(refused)
    handed = maybe_hand_over(sys.argv[1:])
    if handed is not None:
        raise SystemExit(handed)
    code = _preflight(sys.argv[1:])
    if code:
        raise SystemExit(code)
    from strata_cli.cli import main
    raise SystemExit(main())
