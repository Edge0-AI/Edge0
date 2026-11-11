#!/usr/bin/env python3
"""par_run.py - start serve/server.py with one configuration variant on a test box, run par_load.py workloads, stop.

Research tool for the parallelism program.  One call = one server start = one arm/variant.  Writes next to --out:
  <tag>-cfg.json, <tag>-engine.log, <tag>-server.out, <tag>-<workload>.json (par_load ledgers), <tag>-run.json
(effective engine lines, PSS/RSS of the engine tree, GPU memory, ready time, the cmd/env).

    python3 tools/par_run.py --box boston --tag base1g --gpus 0 --work "w1;w2:4,8" --out /path/dir
    python3 tools/par_run.py --box boston --tag s8g4 --gpus 0,1 --parallel 8 --groups 4 --env "STRATA_PIPED_CONCURRENT=1" \
        --extra "--foo bar" --work "w2:4,8;w4:4" --out dir

--work items are workload[:conc list] separated by ';', optionally with par_load flags after '@' (e.g. "w4:4@--joiner-tokens 16384").
--default-config uses what the box's 0.2.8 launcher would write (no parallel key, layer_split auto when 2 GPUs).
"""
import argparse, json, os, shlex, signal, subprocess, sys, time, urllib.request

BOXES = {
    "boston": {
        "py": "/data/external/niko/t027/venv/bin/python", "libs": ["/usr/local/cuda-13.1/lib64"],
        "pack": "/data/external/niko/pm/pack/iq3_s", "mtp": "/data/external/niko/pm/mtp/rt",
        "gguf1": "/data3/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        "gguf2": "/data3/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
        "profile": "{repo}/data/expert-profile.bin",
    },
    "z440": {
        "py": "/data/Strata/.venv/bin/python", "libs": ["/usr/local/cuda-12.9/lib64"],
        "pack": "/data/strata-models/IQ3_S/pack", "mtp": "/data/Strata-data/mtp/rt",
        "gguf1": "/data/strata-models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        "gguf2": "/data/strata-models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
        "profile": "{repo}/data/expert-profile.bin",
    },
}


def pss_tree(pid, exe=None):
    """Sum of PSS (MB) and RSS of a process and all its descendants (Linux, via /proc ppid)."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read()
                ppid = int(st[st.rindex(")") + 2:].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError):
                pass
    seen, todo = set(), [pid]
    for d in os.listdir("/proc"):          # engines are started detached: also every process running our --exe
        if d.isdigit() and exe:
            try:
                if exe in open(f"/proc/{d}/cmdline").read().replace("\0", " "):
                    todo.append(int(d))
            except OSError:
                pass
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo += kids.get(p, [])
    pss = rss = 0
    for p in seen:
        try:
            for line in open(f"/proc/{p}/smaps_rollup"):
                if line.startswith("Pss:"):
                    pss += int(line.split()[1])
                elif line.startswith("Rss:"):
                    rss += int(line.split()[1])
        except OSError:
            pass
    return {"pss_mb": pss // 1024, "rss_mb": rss // 1024, "procs": len(seen)}


def gpu_mem():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        return [[int(x) for x in l.split(",")] for l in o.strip().splitlines()]
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", required=True, choices=list(BOXES))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--exe", required=True)
    ap.add_argument("--repo", required=True, help="source tree (serve/server.py, data/expert-profile.bin)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--replicas", type=int, default=None)
    ap.add_argument("--split", default="auto")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--prefill", default="auto")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--mtp-depth", type=int, default=4, help="--spec value (the default in setup configs is 4)")
    ap.add_argument("--env", default="", help="K=V;K=V for the engine/server environment")
    ap.add_argument("--extra", default="", help="extra engine args (one string)")
    ap.add_argument("--server-extra", default="", help="extra server.py args")
    ap.add_argument("--port", type=int, default=18930)
    ap.add_argument("--work", default="w2:4,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cap", type=int, default=1500, help="engine ready timeout (s)")
    ap.add_argument("--min-free-gb", type=int, default=14)
    ap.add_argument("--post-cmd", default=None, help="a shell command run after the work items; {url} is the server URL; its output goes to <tag>-post.txt")
    a = ap.parse_args()
    B = BOXES[a.box]
    os.makedirs(a.out, exist_ok=True)
    # memory guard: never start with too little RAM (the replicas and the 2-GPU engine each pin tens of GB)
    avail = 0
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1048576
    if avail < a.min_free_gb:
        print(f"ABORT: only {avail} GB RAM available (< {a.min_free_gb})")
        return 3
    log = os.path.join(a.out, f"{a.tag}-engine.log")
    args = ["--pack", B["pack"], "--native", B["gguf1"], "--ple-gguf", B["gguf2"],
            "--expert-profile", B["profile"].format(repo=a.repo), "--expert-cache", "auto", "--prefill", a.prefill,
            "--spec", str(a.mtp_depth), "--spec-min-p", "0.5", "--layer-split", a.split, "--max-context", str(a.ctx), "--kv", "int8"]
    if not a.no_mtp:
        args += ["--mtp", B["mtp"]]
    args += shlex.split(a.extra)
    gpus = [int(x) for x in a.gpus.split(",")]
    cfg = {"exe": a.exe, "args": args, "cwd": a.repo, "tokenizer": B["pack"] + "/tokenizer", "model_name": "qwen3.8-flash-next-iq3_s",
           "log": log, "lib_dirs": B["libs"], "port": a.port, "gpu": gpus, "gpus_asked": True, "engine_ready_s": a.cap}
    if len(gpus) == 1:
        cfg["args"] = [x for i, x in enumerate(args) if x != "--layer-split" and (i == 0 or args[i - 1] != "--layer-split")]
    if a.replicas:
        cfg["replicas"] = a.replicas
    cpath = os.path.join(a.out, f"{a.tag}-cfg.json")
    json.dump(cfg, open(cpath, "w"), indent=1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env["LD_LIBRARY_PATH"] = ":".join(B["libs"] + [env.get("LD_LIBRARY_PATH", "")])
    for kv in a.env.split(";"):
        if kv.strip():
            k, v = kv.split("=", 1)
            env[k.strip()] = v.strip()
    cmd = [B["py"], f"{a.repo}/serve/server.py", "--engine", "strata", "--config", cpath, "--port", str(a.port)]
    if a.parallel is not None:
        cmd += ["--parallel", str(a.parallel)]
    if a.groups:
        cmd += ["--batch-groups", str(a.groups)]
    cmd += shlex.split(a.server_extra)
    srvout = open(os.path.join(a.out, f"{a.tag}-server.out"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, env=env, stdout=srvout, stderr=subprocess.STDOUT, start_new_session=True, cwd=a.repo)
    ready = None
    while time.time() - t0 < a.cap:
        if srv.poll() is not None:
            print("SERVER DIED", flush=True)
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5).read()
            st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/status", timeout=5).read())
            if (st.get("state") or "ready") in ("ready", "idle", "serving") or st.get("ready"):
                ready = time.time() - t0
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    print(f"[{a.tag}] ready after {ready}s", flush=True)
    rec = {"tag": a.tag, "box": a.box, "cmd": cmd, "env": a.env, "args": args, "ready_s": ready, "work": {}, "mem": {}}
    url = f"http://127.0.0.1:{a.port}"
    if ready is not None:
        time.sleep(2)
        rec["mem"]["after_load"] = pss_tree(srv.pid, a.exe)
        rec["gpu_after_load"] = gpu_mem()
        here = os.path.dirname(os.path.abspath(__file__))
        # a short unmeasured warmup so that lazy code objects/JIT and the first expert-cache fills are out of the way
        subprocess.run([sys.executable, f"{here}/par_load.py", url, "--workload", "w2", "--conc", "2", "--rounds", "1", "--max-tokens", "48",
                        "--nonce", "warm", "--out", os.path.join(a.out, f"{a.tag}-warm.json")], capture_output=True)
        for item in a.work.split(";"):
            if not item.strip():
                continue
            main_part, _, extra = item.partition("@")
            wl, _, conc = main_part.partition(":")
            wl = wl.strip()
            outj = os.path.join(a.out, f"{a.tag}-{wl}{conc.replace(',', '_')}.json")
            c2 = [sys.executable, f"{here}/par_load.py", url, "--workload", wl, "--rounds", str(a.rounds), "--arm", a.tag, "--pair", "0",
                  "--label", a.tag, "--settings", log, "--out", outj]
            if conc:
                c2 += ["--conc", conc]
            c2 += shlex.split(extra)
            r = subprocess.run(c2, capture_output=True, text=True)
            print(r.stdout[-1500:], flush=True)
            if r.returncode:
                print("par_load rc", r.returncode, r.stderr[-300:], flush=True)
            rec["work"][item] = outj
            rec["mem"][item] = pss_tree(srv.pid, a.exe)
        if a.post_cmd:
            r = subprocess.run(a.post_cmd.format(url=url), shell=True, capture_output=True, text=True)
            open(os.path.join(a.out, f"{a.tag}-post.txt"), "w").write(r.stdout + r.stderr)
            print(r.stdout[-1500:], flush=True)
        rec["gpu_after_work"] = gpu_mem()
    try:
        os.killpg(srv.pid, signal.SIGTERM)
        srv.wait(120)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(srv.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    keys = ("batch_slots", "batch_groups", "expert cache", "slot sessions", "groups of", "pipelined", "VRAM free", "layer split", "layer-split",
            "strata p4", "strata par", "mtp", "adapt")
    rec["engine_lines"] = [l[:220] for l in text.splitlines() if any(k in l for k in keys)][:60]
    rec["errors_in_log"] = len([1 for l in text.splitlines() if any(k in l for k in ("illegal instruction", "Traceback", " ERR ", "segfault", "Segmentation", "out of memory"))])
    json.dump(rec, open(os.path.join(a.out, f"{a.tag}-run.json"), "w"), indent=1)
    print("DONE", a.tag, "errors_in_log", rec["errors_in_log"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
