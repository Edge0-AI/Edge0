"""Runs the engine side of the DFlash2 experiment for one prompt on strix (foreground; one engine at a time)."""
import json, re, subprocess, sys, time
from pathlib import Path

W = Path(r"C:\dflash2-work")
RW = "C:\\strata-test\\dflash2"


def ssh(cmd, timeout=1500):
    r = subprocess.run(["ssh", "-o", "ServerAliveInterval=30", "strix", cmd], capture_output=True, text=True, timeout=timeout)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if not l.startswith("**"))
    return out


def scp_up(local, remote):
    subprocess.run(["scp", "-q", str(local), "strix:" + remote.replace("\\", "/")], check=True, capture_output=True)


def scp_down(remote, local):
    subprocess.run(["scp", "-q", "strix:" + remote.replace("\\", "/"), str(local)], check=True, capture_output=True)


def run_mode(mode, name, extra=""):
    t = time.time()
    out = ssh('powershell -NoProfile -ExecutionPolicy Bypass -File %s\\run_one.ps1 -Mode %s -Name %s %s' % (RW, mode, name, extra))
    print("[%s %s] %.0f s" % (mode, name, time.time() - t))
    print(out)
    (W / "logs").mkdir(exist_ok=True)
    (W / "logs" / ("%s_%s.log" % (mode, name))).write_text(out)
    scp_down("%s\\out\\%s_%s.txt" % (RW, mode, name), W / "logs" / ("%s_%s.txt" % (mode, name)))
    scp_down("%s\\out\\%s_%s.err" % (RW, mode, name), W / "logs" / ("%s_%s.err" % (mode, name)))
    return out


def produced_ids(path):
    txt = Path(path).read_text()
    m = re.search(r"^output  :(.*)$", txt, re.M)
    return [int(x) for x in m.group(1).split()] if m else []


def main(name, max_new, steps):
    prompts = json.load(open(W / "prompts.json"))
    pids = prompts[name]["ids"]
    if "gen" in steps:
        run_mode("gen", name, "-MaxNew %d" % max_new)
    out = produced_ids(W / "logs" / ("gen_%s.txt" % name))
    print(name, "prompt", len(pids), "generated", len(out), "tail", out[-3:])
    seq = pids + out
    (W / "p" / ("%s.out.ids" % name)).write_text(",".join(map(str, out)))
    (W / "p" / ("%s.first.ids" % name)).write_text(",".join(map(str, seq[:2])))
    (W / "p" / ("%s.rest.ids" % name)).write_text(",".join(map(str, seq[2:])))
    json.dump({"prompt": pids, "out": out}, open(W / "p" / ("%s.seq.json" % name), "w"))
    for k in ("out", "first", "rest"):
        scp_up(W / "p" / ("%s.%s.ids" % (name, k)), "%s\\p\\%s.%s.ids" % (RW, name, k))
    for mode in ("genf", "genfm", "tap"):
        if mode in steps:
            run_mode(mode, name)
    if "tap" in steps:
        (W / "taps").mkdir(exist_ok=True)
        scp_down("%s\\out\\tap_%s.bin" % (RW, name), W / "taps" / ("%s.bin" % name))


if __name__ == "__main__":
    name = sys.argv[1]
    max_new = int(sys.argv[2])
    steps = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gen", "genf", "genfm", "tap"]
    main(name, max_new, steps)
