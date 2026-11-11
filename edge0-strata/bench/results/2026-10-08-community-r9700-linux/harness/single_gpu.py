#!/usr/bin/env python3
"""Run R9700 experiments with one UUID, verified HIP BDF and recorded configuration."""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
UUID = "GPU-1d9ca7a7f0a06a6d"
BDF = "0000:63:00.0"
ISOLATION = {"ROCR_VISIBLE_DEVICES": UUID, "HIP_VISIBLE_DEVICES": "0"}
DEVICE_ENV = {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPU_DEVICE_ORDINAL",
              "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"}
FORBIDDEN = {"--gpus", "--layer-split", "--peer-device", "--helper-device", "--mtp-device",
             "--split-skip-if-fits", "--expert-cache-device1", "--expert-cache-device2",
             "--expert-cache-device3", "--expert-cache-remote-placement", "--remote-expert-opt"}


def validate_args(args):
    for index, value in enumerate(args):
        key, _, inline = value.partition("=")
        if key in FORBIDDEN or key in DEVICE_ENV:
            raise ValueError(f"single-GPU experiment forbids {key}")
        if key in {"--gpu", "--device", "--hip-ordinal"}:
            ordinal = inline if "=" in value else (args[index + 1] if index + 1 < len(args) else "")
            if ordinal != "0":
                raise ValueError(f"{key} must name filtered logical device 0")


def validate_config(cfg):
    if cfg.get("backend") != "hip" or cfg.get("gpu") not in (0, "0"):
        raise ValueError('config requires backend="hip", gpu=0')
    if cfg.get("hip_ordinal", 0) not in (0, "0"):
        raise ValueError("hip_ordinal must be 0")
    for key in ("gpus", "layer_split", "peer_device", "helper_device", "mtp_device", "split_skip_if_fits"):
        if key in cfg:
            raise ValueError(f"single-GPU experiment forbids config key {key}")
    if cfg.get("vision"):
        raise ValueError("this text benchmark config must omit the vision worker")
    validate_args(cfg.get("args", []))
    for key, value in cfg.get("env", {}).items():
        if key in DEVICE_ENV and (key not in ISOLATION or str(value) != ISOLATION[key]):
            raise ValueError(f"config overrides GPU isolation: {key}")


def probe(device_binary):
    # Initialize HIP only after the final environment has been installed.
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    code = hip.hipGetDeviceCount(ctypes.byref(count))
    if code or count.value != 1:
        raise RuntimeError(f"HIP must see exactly one device; code={code}, count={count.value}")
    bus = ctypes.create_string_buffer(32)
    code = hip.hipDeviceGetPCIBusId(bus, len(bus), 0)
    if code or bus.value.decode().lower() != BDF:
        raise RuntimeError(f"unexpected logical device 0 BDF: {bus.value!r}, code={code}")
    result = subprocess.run([str(device_binary), "--list-devices"], check=True, capture_output=True, text=True)
    if (re.findall(r"^device (\d+):", result.stdout, re.M) != ["0"]
            or "gfx1201" not in result.stdout or "R9700" not in result.stdout
            or "cannot run" in result.stdout):
        raise RuntimeError(f"strata-device rejected the target: {result.stdout}")
    return {"uuid": UUID, "bdf": BDF, "hip_device_count": count.value,
            "strata_device": result.stdout,
            "numa_node": (Path("/sys/bus/pci/devices") / BDF / "numa_node").read_text().strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-binary", type=Path, default=ROOT / "build-r9700/strata-device")
    parser.add_argument("--config", type=Path, help="validate this config; for serve it is required")
    parser.add_argument("--record", type=Path, required=True, help="new JSON record, never overwritten")
    parser.add_argument("--cpu-nodes", help="numactl CPU nodes; omit for unbound product baseline")
    parser.add_argument("--preferred-node", type=int, help="NUMA preferred memory node (allows fallback)")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("action", choices=["check", "run", "serve"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    cfg = None
    if args.config:
        cfg = json.loads(args.config.read_text())
        validate_config(cfg)
    if args.action == "serve":
        if cfg is None or command:
            parser.error("serve requires --config and takes no command")
        command = [sys.executable, "-m", "serve.server", "--engine", "strata",
                   "--config", str(args.config.resolve()), "--host", "127.0.0.1", "--port", str(args.port)]
    elif args.action == "run" and not command:
        parser.error("run requires a command after --")
    elif args.action == "check" and command:
        parser.error("check takes no command")
    validate_args(command)
    # A config passed through `run` needs the same guard as `serve`.
    for i, value in enumerate(command):
        if value == "--config" or value.startswith("--config="):
            path = value.split("=", 1)[1] if "=" in value else command[i + 1]
            validate_config(json.loads(Path(path).read_text()))
    env = {k: v for k, v in os.environ.items() if k not in DEVICE_ENV}
    env.update(ISOLATION)
    os.environ.clear()
    os.environ.update(env)
    # This lock coordinates our experiments; it does not reserve the GPU against other users.
    lock = open(f"/tmp/strata-r9700-{os.getuid()}-{UUID}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = probe(args.device_binary.resolve())
    prefix = []
    if args.cpu_nodes or args.preferred_node is not None:
        prefix = ["numactl"]
        if args.cpu_nodes:
            prefix += ["--cpunodebind=" + args.cpu_nodes]
        if args.preferred_node is not None:
            prefix += ["--preferred=" + str(args.preferred_node)]
    record = {"utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "command": prefix + command, "cwd": str(ROOT), "config": cfg,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "environment": {k: v for k, v in env.items()
                              if k.startswith(("STRATA_", "HIP_", "ROCR_", "ROCBLAS_", "HIPBLASLT_", "HSA_", "OMP_"))
                              or k in {"PATH", "LD_LIBRARY_PATH"}},
              "device_binary_sha256": hashlib.sha256(args.device_binary.read_bytes()).hexdigest()}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(device), file=sys.stderr, flush=True)
    if args.action != "check":
        os.chdir(ROOT)
        os.set_inheritable(lock.fileno(), True)
        os.execvpe((prefix + command)[0], prefix + command, env)


if __name__ == "__main__":
    main()
