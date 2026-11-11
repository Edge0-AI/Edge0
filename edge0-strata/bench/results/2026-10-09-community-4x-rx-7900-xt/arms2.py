"""Two arms on the same box, same prompts and image: the five STRATA_* opt-ins off, then on (the live runner).

Arm "optins-off": the live runner container is stopped (not removed) and a clone is started from its own
`docker inspect` config with only those five variables dropped. Arm "optins-on": the live container itself,
restarted. Each arm: fresh engine process, cold first request, two warm-ups, 3x 4,096 / 32,768 / 128,000
(bench.py), then the needle checks. VRAM per card is sampled every second. Whatever happens, the clone is
removed and the live container is left running.
"""

import json
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

C = "local-llm-runner-cluster-7900s-strata-bca809cd"
CLONE = C + "-optins-off"
D = Path.home() / "scratch/strata_bench2"
OPT_INS = ("STRATA_SPLIT_OWN", "STRATA_STAGE_TRIM", "STRATA_PF_FUSED", "STRATA_PF_GEMM", "STRATA_PF_SWITCH_MIN_T")


def sh(*args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def status():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8086/v1/status", timeout=5) as r:
            return json.load(r)
    except OSError:
        return None


def wait_loaded(t0, label):
    deadline = time.time() + 900
    while True:
        s = status()
        if s and s.get("loaded") and s.get("uptime_s", 1e9) <= time.time() - t0 + 5:
            say(label, "loaded after", round(time.time() - t0), "s")
            return s
        if time.time() > deadline:
            raise SystemExit(f"{label}: not loaded after 15 min")
        time.sleep(2)


stop = threading.Event()


def sampler():
    with (D / "vram.jsonl").open("a") as f:
        while not stop.is_set():
            out = sh("rocm-smi", "--showmeminfo", "vram", "--json", check=False).stdout
            try:
                used = {k: int(v["VRAM Total Used Memory (B)"]) for k, v in json.loads(out).items()}
            except (ValueError, KeyError):
                used = None
            f.write(json.dumps({"t": round(time.time(), 2), "used_b": used}) + "\n")
            f.flush()
            stop.wait(1)


def bench(arm, container, t0):
    s = wait_loaded(t0, arm)
    (D / f"status_after_load-{arm}.json").write_text(json.dumps(s, indent=1))
    env = json.loads(sh("docker", "inspect", container).stdout)[0]["Config"]["Env"]
    (D / f"container_env-{arm}.txt").write_text("".join(e + "\n" for e in env if e.startswith(("STRATA_", "HIP_"))))
    time.sleep(5)
    subprocess.run(["python3", "bench.py", "prompts.jsonl", f"matrix-{arm}.jsonl"], cwd=D, check=True)
    say(arm, "matrix done")
    with (D / f"needles-{arm}.log").open("w") as log:
        subprocess.run(["python3", str(D / "src/tools/needle_bench.py"), "--url", "http://127.0.0.1:8086",
                        "--lengths", "32k,128k", "--depths", "10,50,90", "--out", str(D / f"needles-{arm}.json")],
                       cwd=D, stdout=log, stderr=subprocess.STDOUT, check=True)
    say(arm, "needles done")
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0 - 1))
    logs = sh("docker", "logs", "--since", since, container, check=False)
    (D / f"engine-{arm}.log").write_text(logs.stdout + logs.stderr)


def handler(signum, frame):
    raise SystemExit(f"signal {signum}")


signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

orig = json.loads(sh("docker", "inspect", C).stdout)[0]
cfg, host = orig["Config"], orig["HostConfig"]
assert not any(k in " ".join(cfg["Cmd"]) for k in OPT_INS), "opt-ins in the command line"
assert all(any(e.startswith(k + "=") for e in cfg["Env"]) for k in OPT_INS), "live env lacks the opt-ins"
run = ["docker", "run", "-d", "--name", CLONE, "--network", host["NetworkMode"], "--ipc", host["IpcMode"],
       "--shm-size", str(host["ShmSize"])]
for g in host.get("GroupAdd") or []:
    run += ["--group-add", g]
for d in host.get("Devices") or []:
    run += ["--device", f"{d['PathOnHost']}:{d['PathInContainer']}"]
for b in host.get("Binds") or []:
    run += ["-v", b]
for e in cfg["Env"]:
    if e.split("=", 1)[0] not in OPT_INS:
        run += ["-e", e]
run += [cfg["Image"], *cfg["Cmd"]]

(D / "vram.jsonl").unlink(missing_ok=True)
threading.Thread(target=sampler, daemon=True).start()
try:
    t0 = time.time()
    say("optins-off: stop live container, start clone without", ",".join(OPT_INS))
    sh("docker", "stop", C)
    sh(*run)
    bench("optins-off", CLONE, t0)
    sh("docker", "rm", "-f", CLONE)
    say("optins-off: clone removed")

    t0 = time.time()
    say("optins-on: start live container (fresh process)")
    sh("docker", "start", C)
    bench("optins-on", C, t0)
finally:
    stop.set()
    sh("docker", "rm", "-f", CLONE, check=False)
    if json.loads(sh("docker", "inspect", C).stdout)[0]["State"]["Running"] is not True:
        sh("docker", "start", C, check=False)
        say("live container started again")
    say("live container running:", json.loads(sh("docker", "inspect", C).stdout)[0]["State"]["Running"])
    print("ALL_DONE", flush=True)
