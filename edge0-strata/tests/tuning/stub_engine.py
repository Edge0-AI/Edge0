#!/usr/bin/env python3
"""A stub engine for exercising `strata tune run` end to end, on every OS.

Run it as ``[sys.executable, "stub_engine.py", "--serve", ...]`` (``strata_control.tuning.engine_command``
does that for any ``.py`` engine). It speaks the real serve protocol (READY / PP / T / DONE) and
makes its decode speed depend on one flag, so a sweep has something real to find. It is NOT a model
and its numbers mean nothing about any hardware -- it exists so the *sweep machinery* can be tested
without weights.

    --kv int8   decodes at 200 tok/s
    anything else (including --kv fp16) decodes at 100 tok/s

The prompt's length comes from the GEN line's token list, so the workload is genuinely the one the
caller asked for.
"""
import sys


def main(argv):
    kv = "unknown"
    for index, arg in enumerate(argv):
        if arg == "--kv" and index + 1 < len(argv):
            kv = argv[index + 1]
    if "--help" in argv:
        sys.stderr.write("usage:\n  --kv KV\n  --ple-io MODE\n")
        return 0
    out = sys.stdout
    out.write("READY 32768 stop\n")
    out.flush()
    for raw in sys.stdin:
        line = raw.strip()
        if line == "QUIT" or not line:
            return 0
        if not line.startswith("GEN"):
            continue
        fields = line.split()
        count = len(fields[-1].split(","))
        try:
            max_new = int(fields[1])
        except (IndexError, ValueError):
            max_new = 8
        decode_ms = 50.0 if kv == "int8" else 100.0
        out.write(f"PP {count} {count} 200.0 2560.0\n")
        for i in range(max_new):          # one T per token: the harness cross-checks the stream against DONE
            out.write(f"T {i}\n")
        out.write(f"DONE {max_new} {count} 200.0 {decode_ms} stop 0 0\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
