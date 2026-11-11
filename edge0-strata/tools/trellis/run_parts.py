"""run_parts.py - tools/quant_quality.py run in parts that survive an engine failure.

The long runs of this experiment met two kinds of "verify: timed out" (the engine's 20 s wait for the GPU) while the box
was shared; one failed segment must not cost the whole run.  The ids file is cut into --parts parts, each part is one engine
run (retried up to --retries times, a failed try is discarded), and the part files are joined at the end into OUT.tsv +
OUT.tsv.seg.json, which `quant_quality.py compare` and swap_compare.py read as usual.  With --dump-dir every part writes its
STRATA_ACT_DUMP files to <dump-dir>/p<i>/ (only the dumps of finished parts are kept).

  python3 tools/trellis/run_parts.py --runner /data/dsv41-enc/run_qq.sh --cfg CFG --ids IDS --out OUT.tsv --parts 6 \
      [--env K=V ...] [--dump-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time


def kill_engines():
    out = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True).stdout.split()
    for p in out:
        subprocess.run(["kill", "-9", p])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    work = out.parent / (out.name + ".parts")
    work.mkdir(parents=True, exist_ok=True)
    d = json.load(open(a.ids))
    S = d["segments"]
    k = -(-len(S) // a.parts)
    done = []
    for i in range(a.parts):
        seg = S[i * k:(i + 1) * k]
        if not seg:
            continue
        ids = work / ("p%d.ids.json" % i)
        json.dump(dict(d, segments=seg), open(ids, "w"))
        tsv = work / ("p%d.tsv" % i)
        ok = pathlib.Path(str(tsv) + ".seg.json").exists()
        for attempt in range(a.retries):
            if ok:
                break
            env = list(a.env)
            if a.dump_dir:
                dd = pathlib.Path(a.dump_dir) / ("p%d" % i)
                shutil.rmtree(dd, ignore_errors=True)
                dd.mkdir(parents=True)
                env += ["STRATA_ACT_DUMP=%s/act" % dd]
            t0 = time.time()
            r = subprocess.run([a.runner, a.cfg, str(ids), str(tsv)] + env, capture_output=True, text=True)
            ok = pathlib.Path(str(tsv) + ".seg.json").exists()
            print("part %d try %d: %s (%.0f s)%s" % (i, attempt + 1, "ok" if ok else "FAILED", time.time() - t0,
                                                       "" if ok else " " + (r.stdout or "")[-200:].replace("\n", " ")), flush=True)
            if not ok:
                kill_engines()
                if a.dump_dir:
                    shutil.rmtree(pathlib.Path(a.dump_dir) / ("p%d" % i), ignore_errors=True)
        if not ok:
            print("part %d gave up" % i, flush=True)
            continue
        done.append(i)
    # join
    lines, segs = [], []
    for i in done:
        tsv = work / ("p%d.tsv" % i)
        part = open(tsv).read().splitlines()
        sj = json.load(open(str(tsv) + ".seg.json"))
        base = len(lines)
        lines += part
        for s in sj["segments"]:
            segs.append({"name": s["name"], "first_line": base + s["first_line"], "end_line": base + s["end_line"]})
    out.write_text("\n".join(lines) + "\n")
    pathlib.Path(str(out) + ".seg.json").write_text(json.dumps({"segments": segs, "parts_done": done, "parts": a.parts}, indent=1))
    print("joined %d parts, %d segments -> %s" % (len(done), len(segs), out))
    return 0 if len(done) == a.parts else 1


if __name__ == "__main__":
    sys.exit(main())
