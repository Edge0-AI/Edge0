"""A short tune for DeepSeek-V4.1-Flash (strata-dsv41): the few settings the engine has, measured on this computer.

    python tools/dsv41/tune.py --config <launch config .json> [--mode quick|thorough] [--runs N]

What it does (a few short decode runs through the engine's own command line; each loads the model, so about 1-3 minutes each):
  1. the launch's own settings (the baseline);
  2. the same with the DSpark drafter forced on (`--dspark on --dspark-conf 0.65`) against `--dspark auto`, which leaves it unloaded on a
     small card: kept only when it is clearly faster; with it on, a draft block of 3 against the pack's own (5);
  3. the CPU threads: the number of physical cores against every logical thread (SMT), when they differ: kept only when clearly faster.
The RAM store (`--ssd-cache`) sizes itself from the free memory at start, so it is left as the engine's own; the line says so.
A setting is changed only for a gain of more than 5% (run-to-run noise on this engine is about 5%).  Whatever is kept is written into
the config's args; a run that fails or a computer that is too busy leaves the config as it was, with one plain line.  Always exits 0.
The context stays what the person chose.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GAIN = 1.05
DECODE = re.compile(r"decode:\s*\d+ tokens in [\d.]+ ms \(([\d.]+) tok/s\)")


def say(msg: str) -> None:
    print(msg, flush=True)


def parse_speed(stderr: str) -> float | None:
    m = DECODE.findall(stderr or "")
    return float(m[-1]) if m else None


def drop(args: list[str], flag: str) -> list[str]:
    """`args` without `flag` and its value."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def with_flag(args: list[str], flag: str, *values: str) -> list[str]:
    return [*drop(args, flag), flag, *values]


def physical_cores() -> int | None:
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:                                    # noqa: BLE001
        return None


def measure(exe: str, args: list[str], ids_file: Path, n: int, timeout: int, env: dict) -> float | None:
    """Decode tokens a second of one CLI run, or None when it did not run to the end."""
    cmd = [exe, *[a for a in args if a not in ("--serve",)], "--ids", "@" + str(ids_file), "--n", str(n)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return parse_speed(r.stderr)


def prompt_ids(pack: Path) -> list[int]:
    import smoke
    tok = smoke.tokenizer(pack)
    return tok.encode(smoke.PROMPT, parse_special=True)


def lib_env(cfg: dict) -> dict:
    env = dict(os.environ)
    dirs = [str(d) for d in cfg.get("lib_dirs") or [] if d]
    key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    if dirs:
        env[key] = os.pathsep.join(dirs + [env.get(key, "")])
    return env


def tune(config: Path, mode: str = "quick", n: int = 48, timeout: int = 1500, measure_fn=measure) -> dict:
    """Returns {"changed": [...], "speeds": {...}, "lines": [plain sentences]}; writes the config when something was kept."""
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    exe, args = cfg.get("exe"), [str(a) for a in cfg.get("args") or []]
    res = {"changed": [], "speeds": {}, "lines": []}
    if not exe or "--pack" not in args:
        res["lines"].append("This launch has no DeepSeek engine settings to tune, so it starts as it is.")
        return res
    pack = Path(args[args.index("--pack") + 1])
    env = lib_env(cfg)
    try:
        ids = prompt_ids(pack)
    except Exception as e:                                # noqa: BLE001
        res["lines"].append(f"The tuner could not prepare its test question ({e}), so the standard settings run.")
        return res
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ids.txt"
        f.write_text(",".join(str(i) for i in ids), encoding="ascii")
        base_args = [a for a in args if a != "--serve"]
        say("Tuning DeepSeek: measuring the standard settings first ...")
        base = measure_fn(exe, base_args, f, n, timeout, env)
        if base is None:
            res["lines"].append("The standard settings did not finish a test run here, so nothing was tuned and they stay as they are.")
            return res
        res["speeds"]["standard"] = base
        best, best_args = base, base_args
        say(f"  standard settings: {base:.2f} tokens a second")
        trials = [("the drafter on (DSpark)", lambda a: with_flag(with_flag(a, "--dspark", "on"), "--dspark-conf", "0.65"))]
        trials.append(("a shorter draft block (3)", lambda a: with_flag(a, "--dspark-block", "3")))
        cores, logical = physical_cores(), os.cpu_count() or 0
        if cores and logical and logical > cores and "--cpu-threads" not in base_args:
            trials.append((f"{cores} CPU threads (one for each physical core)", lambda a, c=cores: with_flag(a, "--cpu-threads", str(c))))
        if mode == "thorough" and cores:
            trials.append(("every logical CPU thread", lambda a, c=logical: with_flag(a, "--cpu-threads", str(c))))
        queue = list(trials)
        while queue:
            label, make = queue.pop(0)
            if label == "a shorter draft block (3)":
                if "on" != (best_args[best_args.index("--dspark") + 1] if "--dspark" in best_args else ""):
                    continue                                       # only with the drafter on: with `auto` it may not even be loaded
            cand = make(best_args)
            say(f"  trying {label} ...")
            v = measure_fn(exe, cand, f, n, timeout, env)
            if v is None:
                res["lines"].append(f"The test with {label} did not finish; that setting stays as it was.")
                continue
            res["speeds"][label] = v
            say(f"    {v:.2f} tokens a second")
            if v > best * GAIN:
                best, best_args = v, cand
                res["changed"].append(label)
    tried = ", ".join(f"{k} {v:.1f}" for k, v in res["speeds"].items())
    if res["changed"]:
        new = [a for a in best_args]
        cfg["args"] = new
        cfg.setdefault("tuned", {})["dsv41"] = {"speeds": res["speeds"], "changed": res["changed"]}
        config.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        res["lines"].append(f"Done: {', '.join(res['changed'])} - answers about {100 * (best / base - 1):.0f}% faster "
                            f"({base:.1f} -> {best:.1f} tokens a second). The RAM store sizes itself when the model starts.")
    else:
        res["lines"].append(f"Your computer already runs DeepSeek best with the standard settings ({base:.1f} tokens a second); nothing changed.")
    if len(res["speeds"]) > 1:
        res["lines"].append(f"Tried (tokens a second): {tried}.")
    else:
        res["lines"].append("Only the standard settings could be timed here, so there was nothing to compare.")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="quick", choices=["quick", "thorough"])
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--output", default="human")
    a = ap.parse_args()
    try:
        res = tune(Path(a.config), a.mode, a.tokens, a.timeout)
    except Exception as e:                                # noqa: BLE001 - a tune never stops a launch
        res = {"changed": [], "speeds": {}, "lines": [f"The tuner stopped ({type(e).__name__}), so the standard settings run."]}
    for line in res["lines"]:
        say(line)
    if a.output == "json":
        say(json.dumps({"result": res}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
