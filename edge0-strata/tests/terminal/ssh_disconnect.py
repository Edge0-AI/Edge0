#!/usr/bin/env python3
"""A real ssh disconnect ends the server and its engine on a Linux box.

    python tests/terminal/ssh_disconnect.py --host p2003ssh.claryel.space --src /data/split-life/src --python /data/venv/bin/python

Runs from any machine with ssh: it starts `strata serve` on the box in an `ssh -tt` session (a pty, like an interactive
login), kills the LOCAL ssh client (the connection drops, sshd hangs up the pty), and then checks on the box that the
server and the fake engine are gone, the port is free and the lease is free. The engine is the fake package of
lifecycle_matrix.py (no GPU, no model). It prints one line, `PASS ssh-disconnect ...` or `FAIL ...`, and exits 0/1.
"""
import argparse
import json
import subprocess
import sys
import time

REMOTE_RIG = r'''
import json, sys
sys.path.insert(0, "%(src)s")
from pathlib import Path
from tests.terminal import lifecycle_matrix as m
root = Path("/data/split-life/sshrig"); import shutil; shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
rig = m.Rig(root)
print(json.dumps({"argv": rig.serve_argv(), "env": {k: v for k, v in rig.env.items() if k.startswith(("STRATA_", "FAKE_", "XDG_", "LOCALAPPDATA", "PYTHONPATH"))}, "port": rig.port}))
'''

REMOTE_STATE = r'''
import json, os, socket, sys
sys.path.insert(0, "%(src)s")
os.environ.update(json.loads(%(env)r))
from strata_control import procs, runtime
port = %(port)d
left = [r for r in procs.records() if procs.start_token(r["pid"]) == r["token"]]
import psutil
fam = [p.pid for p in psutil.process_iter(["cmdline"]) if any("sshrig" in c or "serve.server" in c or "strata.py" in c for c in (p.info["cmdline"] or [])) and p.pid != os.getpid()]
with socket.socket() as s:
    s.settimeout(0.5); open_ = s.connect_ex(("127.0.0.1", port)) == 0
try:
    fd = runtime.acquire_activity_lease(shared=False); runtime._unlock_file(fd); os.close(fd); lease = True
except Exception:
    lease = False
print(json.dumps({"records_alive": [r["pid"] for r in left], "processes": fam, "port_open": open_, "lease_free": lease}))
'''


def ssh(host, *args, stdin=None, timeout=120):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, *args], input=stdin, capture_output=True, text=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--python", default="python3")
    a = ap.parse_args()
    rig = json.loads(ssh(a.host, a.python, "-", stdin=REMOTE_RIG % {"src": a.src}).stdout.strip().splitlines()[-1])
    env = " ".join(f"{k}='{v}'" for k, v in rig["env"].items())
    remote = f"cd {a.src} && env {env} {a.python} " + " ".join(rig["argv"][1:])
    client = subprocess.Popen(["ssh", "-tt", "-o", "ConnectTimeout=30", a.host, remote], stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ready = False
    for _ in range(120):
        probe = ssh(a.host, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{rig['port']}/health")
        if probe.stdout.strip() == "200":
            ready = True
            break
        time.sleep(1)
    if not ready:
        client.kill()
        print("FAIL ssh-disconnect: the server did not come up")
        return 1
    client.kill()                                       # the connection drops; sshd hangs up the pty
    client.wait(timeout=10)
    state = {}
    for _ in range(40):
        time.sleep(1)
        state = json.loads(ssh(a.host, a.python, "-", stdin=REMOTE_STATE % {"src": a.src, "env": json.dumps(rig["env"]), "port": rig["port"]}).stdout.strip().splitlines()[-1])
        if not state["records_alive"] and not state["processes"] and not state["port_open"] and state["lease_free"]:
            print("PASS ssh-disconnect", json.dumps(state))
            return 0
    print("FAIL ssh-disconnect", json.dumps(state))
    return 1


if __name__ == "__main__":
    sys.exit(main())
