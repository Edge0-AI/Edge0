"""Real-weight server checks; callers run inside an isolated Docker container."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def request(port, endpoint, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    query = Request(f"http://127.0.0.1:{port}{endpoint}", data=data,
                    headers={"Content-Type": "application/json"})
    with urlopen(query, timeout=timeout) as response:
        require(response.status == 200, f"HTTP {response.status}: {endpoint}")
        return json.load(response)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(process, port, log):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        require(process.poll() is None, f"server exited; see {log}")
        try:
            if request(port, "/health", timeout=1).get("status") == "ok":
                return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"server readiness timed out; see {log}")


def server_args(binary, model, port, threads, lora, batch, parallel=8):
    args = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "0", "-t", str(threads), "-tb", str(threads), "-c", "16384",
            "-b", str(batch), "-ub", "512", "-np", str(parallel), "--no-warmup", "--jinja",
            "--cache-ram", "0", "-lv", "4"]
    return args + (["--lora", str(lora)] if lora else [])


def stop_server(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    require(process.returncode == 0, f"server shutdown returned {process.returncode}")


@contextmanager
def running_server(binary, model, log, *, threads=8, lora=None, features=None, batch=512, parallel=8):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E0_", "LLAMA_", "GGML_"))}
    env.update(OMP_WAIT_POLICY="PASSIVE", **(features or {}))
    port = free_port()
    with Path(log).open("w") as output:
        process = subprocess.Popen(server_args(binary, model, port, threads, lora, batch, parallel),
                                   env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, port, log)
            yield port
        finally:
            stop_server(process)


def completion(port, prompt, count=48):
    return request(port, "/completion", {"prompt": prompt, "n_predict": count,
                   "temperature": 0, "seed": 4242, "ignore_eos": True,
                   "cache_prompt": False, "return_tokens": True})


def snapshot():
    """Time and load averages only: saved results must not list other processes or users of the host."""
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load": list(os.getloadavg())}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
