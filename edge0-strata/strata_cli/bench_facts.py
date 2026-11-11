"""What a standard benchmark report says about the machine, the build, the model and the settings (docs/BENCHMARK_PROTOCOL.md).

Every probe is best effort and local: it reads the inference host's own tools (nvidia-smi, rocm-smi, amd-smi, PowerShell, /proc, sysctl) and the
server's answers. A fact that cannot be read becomes `None` with a reason in `Why`, never a guess. Nothing here changes the machine.
"""

from __future__ import annotations

import getpass
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MODEL_SUFFIXES = (".gguf", ".safetensors", ".pack", ".bin")
SECRET_KEY = re.compile(r"(?i)key|token|secret|password|auth")


class Why(dict):
    """JSON Pointer -> the reason a value is null. Built while the report is assembled and written as `missing`."""

    def add(self, pointer: str, reason: str) -> None:
        self[pointer] = reason


def pointer(*parts) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


# ------------------------------------------------------------------------------------------------------------ redaction
def redactor(key: str = ""):
    """text -> text with this machine's home folders, user and host names, the Strata folder, private IP addresses and the API key replaced."""
    subs: list[tuple[re.Pattern, str]] = []

    def add(value, label, word=False):
        if value and len(str(value)) >= 4:
            text = str(value)
            for form in sorted({text, text.replace("\\", "/"), text.replace("/", "\\")}, key=len, reverse=True):
                pattern = re.escape(form)
                subs.append((re.compile(r"(?<![\w-])" + pattern + r"(?![\w-])" if word else pattern, re.I), label))

    add(ROOT, "<strata>")
    add(Path.home(), "<home>")
    add(tempfile.gettempdir(), "<tmp>")
    try:
        add(getpass.getuser(), "<user>", True)
    except Exception:                                   # noqa: BLE001 - no user name is not an error
        pass
    add(socket.gethostname(), "<host>", True)
    add(platform.node(), "<host>", True)
    add(key, "<key>")
    subs.sort(key=lambda item: -len(item[0].pattern))
    path_user = re.compile(r"(?i)([A-Za-z]:)?([\\/])(Users|home)([\\/])(?!<)[^\\/\s\"':;,]+")
    private_ip = re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])"
                            r"|169\.254)\.\d{1,3}\.\d{1,3}\b")
    bearer = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")

    def clean(text):
        if not isinstance(text, str):
            return text
        for pattern, label in subs:
            text = pattern.sub(label, text)
        text = path_user.sub(lambda m: f"{m.group(1) or ''}{m.group(2)}{m.group(3)}{m.group(4)}<user>", text)
        text = bearer.sub(r"\1<key>", text)
        return private_ip.sub("<ip>", text)
    return clean


def redact_object(value, clean):
    if isinstance(value, dict):
        return {clean(k) if isinstance(k, str) else k: redact_object(v, clean) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_object(v, clean) for v in value]
    return clean(value) if isinstance(value, str) else value


# ------------------------------------------------------------------------------------------------------------ commands
def run_cmd(cmd: list[str], timeout: float = 15) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return done.stdout if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def powershell(script: str, timeout: float = 20) -> str:
    return run_cmd(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------------------ hardware
def ram_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "darwin":
        out = run_cmd(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) if out.isdigit() else None
    return None


def cpu_name() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        out = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if out:
            return out
    if os.name == "nt":
        out = powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name").strip()
        if out:
            return out
    return platform.processor() or None


def nvidia_gpus() -> tuple[list[dict], str | None, str | None]:
    """-> (cards, driver version, CUDA version) from nvidia-smi."""
    out = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,power.limit,driver_version",
                   "--format=csv,noheader,nounits"])
    cards = []
    for line in out.splitlines():
        part = [x.strip() for x in line.split(",")]
        if len(part) < 5 or not part[0].isdigit():
            continue

        def number(text):
            try:
                return float(text)
            except ValueError:
                return None
        mib, watts = number(part[2]), number(part[3])
        cards.append({"device_index": int(part[0]), "name": part[1],
                      "memory_bytes": int(mib * 1024 * 1024) if mib else None,
                      "memory_kind": "dedicated", "power_limit_w": watts, "driver": part[4]})
    cuda = None
    if cards:
        # driver 615 prints "CUDA UMD Version: 13.4" where older drivers print "CUDA Version: 13.0"
        match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", run_cmd(["nvidia-smi"]))
        cuda = match.group(1) if match else None
    return cards, (cards[0]["driver"] if cards else None), cuda


def other_gpus() -> tuple[list[dict], str | None]:
    """AMD (rocm-smi, or the Windows registry), Apple silicon: -> (cards, driver version)."""
    cards, driver = [], None
    rocm = run_cmd(["rocm-smi", "--showproductname"])
    names = re.findall(r"Card [Ss]eries:\s*(.+)", rocm) or re.findall(r"Card [Mm]odel:\s*(.+)", rocm)
    if names:
        match = re.search(r"Driver version:\s*(\S+)", run_cmd(["rocm-smi", "--showdriverversion"]))
        driver = match.group(1) if match else None
        cards = [{"device_index": i, "name": n.strip(), "memory_bytes": None, "memory_kind": "dedicated",
                  "power_limit_w": None} for i, n in enumerate(names)]
    elif os.name == "nt":
        try:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from serve.telemetry import windows_amd_cards
            found = windows_amd_cards()
            cards = [{"device_index": i, "name": c["name"], "memory_bytes": c.get("vram_bytes") or None,
                      "memory_kind": "dedicated", "power_limit_w": None} for i, c in enumerate(found)]
            driver = next((c.get("driver") for c in found if c.get("driver")), None)
        except Exception:                               # noqa: BLE001 - a report without the card's name is still a report
            pass
    elif os.name != "nt" and sys.platform != "darwin":      # (a Mac has no AMD card to ask for; its chip is read below)
        # ROCm 7 and newer ship amd-smi and no rocm-smi (verified on ROCm 10.1). Different grammar: `static --asic`
        # gives the marketing name, `static --driver` the kernel driver's version, and the PCI address only comes
        # from `amd-smi list`; the VRAM size is the amdgpu driver's sysfs file for that address.
        def amd_json(args):
            try:
                return json.loads(run_cmd(["amd-smi", "static"] + args + ["--json"]).strip() or "{}").get("gpu_data") or []
            except (ValueError, OSError):
                return []
        try:
            bdf = {e.get("gpu"): (e.get("bdf") or "") for e in json.loads(run_cmd(["amd-smi", "list", "--json"]).strip() or "[]")}
        except (ValueError, OSError):
            bdf = {}
        drv = {e.get("gpu"): ((e.get("driver") or {}).get("version") or "") for e in amd_json(["--driver"])}
        for i, e in enumerate(amd_json(["--asic"])):
            name = ((e.get("asic") or {}).get("market_name") or "").strip()
            if not name:
                continue
            d = drv.get(e.get("gpu"), "")
            driver = driver or (d if d.upper() not in ("", "N/A") else None)
            vram = None
            for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
                try:
                    if bdf.get(e.get("gpu")) and os.path.basename(os.path.realpath(dev)).lower() == bdf[e["gpu"]].lower():
                        with open(os.path.join(dev, "mem_info_vram_total"), encoding="utf-8") as f:
                            total = f.read().strip()
                        vram = int(total) if total.isdigit() else None
                except OSError:
                    pass
            cards.append({"device_index": i, "name": name, "memory_bytes": vram, "memory_kind": "dedicated",
                          "power_limit_w": None})
    elif sys.platform == "darwin":
        chip = cpu_name()
        if chip:
            cards = [{"device_index": 0, "name": chip, "memory_bytes": ram_bytes(), "memory_kind": "shared",
                      "power_limit_w": None}]
    return cards, driver


def rocm_version() -> str | None:
    """ROCm's own version, which `rocm-smi --showdriverversion` used to report with the driver. ROCm 10 has no
    `rocm-smi` and no `/opt/rocm/.info/`: its version file sits under `<install>/core/.info/version`, and
    `amd-smi -h` prints the same version."""
    for base in [os.environ.get("ROCM_PATH", ""), "/opt/rocm"]:
        if not base:
            continue
        for sub in (".info/version", "core/.info/version"):
            try:
                with open(os.path.join(base, sub), encoding="utf-8") as f:
                    found = f.read().strip()
                if found:
                    return found
            except OSError:
                pass
    match = re.search(r"ROCm version:\s*([\d.]+)", run_cmd(["amd-smi", "-h"]))
    return match.group(1) if match else None


def lsblk_disk(data: dict) -> list[dict] | None:
    """The first physical disk in `lsblk -J -s` output. `-s` lists the parents of a device as nested children: an LVM volume
    is lvm -> partition -> disk, two levels down, so the walk goes all the way down."""
    def disks(item):
        if item.get("type") == "disk":
            yield item
        for child in item.get("children") or []:
            yield from disks(child)

    for device in data.get("blockdevices") or []:
        for item in disks(device):
            rotational = str(item.get("rota")).lower() in ("1", "true")
            bus = str(item.get("tran") or "unknown").lower()
            return [{"kind": "hdd" if rotational else ("nvme" if bus == "nvme" else "ssd"),
                     "model": str(item.get("model") or "unknown").strip(), "bus": bus}]
    return None


def storage_of(path: Path | None) -> list[dict] | None:
    """The drive that holds `path`: [{kind, model, bus}], or None when it cannot be told."""
    if path is None:
        return None
    try:
        path = path.resolve()
    except OSError:
        return None
    if os.name == "nt":
        letter = path.drive[:1]
        if not letter.isalpha():
            return None
        out = powershell(f"Get-Partition -DriveLetter {letter} | Get-Disk | Get-PhysicalDisk | "
                         "Select-Object MediaType,FriendlyName,BusType | ConvertTo-Json -Compress")
        try:
            data = json.loads(out)
        except ValueError:
            return None
        rows = data if isinstance(data, list) else [data]
        return [{"kind": str(r.get("MediaType") or "unknown").lower(), "model": str(r.get("FriendlyName") or "unknown"),
                 "bus": str(r.get("BusType") or "unknown").lower()} for r in rows if isinstance(r, dict)] or None
    if sys.platform.startswith("linux"):
        source = run_cmd(["findmnt", "-n", "-o", "SOURCE", "--target", str(path)]).strip()
        if not source.startswith("/dev/"):
            return None
        try:
            data = json.loads(run_cmd(["lsblk", "-J", "-s", "-o", "NAME,TYPE,ROTA,MODEL,TRAN", source]))
        except ValueError:
            return None
        return lsblk_disk(data)
    return None


def gpu_selection(config: dict | None, argv: list[str] | None) -> list[int]:
    """The card numbers the engine is told to use, from the run config's "gpu" or a `--gpu` in its arguments."""
    def parse(value):
        if value is None or value == "":
            return []
        items = value if isinstance(value, (list, tuple)) else str(value).split(",")
        return [int(x) for x in (str(i).strip() for i in items) if x.isdigit()]
    chosen = parse((config or {}).get("gpu"))
    if not chosen and argv and "--gpu" in argv and argv.index("--gpu") + 1 < len(argv):
        chosen = parse(argv[argv.index("--gpu") + 1])
    return chosen


def hardware(why: Why, *, selected: list[int], served_gpu: str | None, model_dir: Path | None,
             declared_background: list[str]) -> tuple[dict, dict]:
    """-> (the report's `hardware`, the driver and runtime facts the `software` block uses)."""
    cards, driver, cuda = nvidia_gpus()
    if not cards:
        cards, driver = other_gpus()
    if selected:
        cards = [c for c in cards if c["device_index"] in selected] or cards
    elif len(cards) > 1 and served_gpu:
        named = [c for c in cards if c["name"] == served_gpu]
        cards = named[:1] or cards[:1]
    served = [n for n in (served_gpu or "").split(" + ") if n and n != "?"]
    # the amdgpu driver's product_name is empty on some cards (gfx1201 among them), so the server falls back to a bare
    # "AMD Radeon": that names the vendor, not the card, and the local probe's marketing name is the one to keep
    stub = re.compile(r"(?i)^(amd radeon|advanced micro devices(,? inc\.?)?|nvidia( corporation)?|intel"
                      r"( corporation)?)( graphics)?$")
    if served and cards and not all(stub.match(n) for n in served) \
            and not all(n in {c["name"] for c in cards} for n in served):
        # the server reads cards the local tools do not list (another vendor's, or a different numbering)
        cards = [{"device_index": i, "name": n, "memory_bytes": None, "memory_kind": "dedicated", "power_limit_w": None}
                 for i, n in enumerate(served)]
        driver = cuda = None
    if not cards and served:
        cards = [{"device_index": i, "name": n, "memory_bytes": None, "memory_kind": "dedicated", "power_limit_w": None}
                 for i, n in enumerate(served)]
    gpus = []
    for i, card in enumerate(cards):
        gpus.append({k: card.get(k) for k in ("device_index", "name", "memory_bytes", "memory_kind", "power_limit_w")})
        if gpus[-1]["memory_bytes"] is None:
            why.add(pointer("hardware", "gpus", i, "memory_bytes"), "the local tools did not report this card's memory")
        if gpus[-1]["power_limit_w"] is None:
            why.add(pointer("hardware", "gpus", i, "power_limit_w"), "no power limit available for this card from the local tools")
    ram = ram_bytes()
    cpu = cpu_name()
    drive = storage_of(model_dir)
    block = {"gpus": gpus, "cpu": cpu, "ram_bytes": ram, "storage": drive,
             "background_workloads": declared_background or None}
    if not gpus:
        why.add(pointer("hardware", "gpus"), "no GPU was found by the local tools or reported by the server")
    if cpu is None:
        why.add(pointer("hardware", "cpu"), "the CPU model could not be read")
    if ram is None:
        why.add(pointer("hardware", "ram_bytes"), "installed memory could not be read")
    if drive is None:
        why.add(pointer("hardware", "storage"), "the drive holding the model files could not be identified")
    if not declared_background:
        why.add(pointer("hardware", "background_workloads"),
                "the runner does not inspect other programs; none were declared with --background")
    # an AMD card on Linux runs on ROCm, which the report names as the runtime; on ROCm 7 and newer that version no
    # longer comes from rocm-smi, but from the install itself
    rocm = rocm_version() if (sys.platform.startswith("linux") and cards and not cuda) else None
    return block, {"driver": driver, "cuda": cuda, "rocm": rocm}


# ------------------------------------------------------------------------------------------------------------ software
def os_info() -> dict:
    if os.name == "nt":
        return {"name": "Windows", "version": platform.version()}
    if sys.platform == "darwin":
        return {"name": "macOS", "version": platform.mac_ver()[0] or platform.release()}
    name = None
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                name = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return {"name": name or platform.system(), "version": platform.release()}


def git_commit(folder: Path = ROOT) -> str | None:
    """The full commit of this checkout, or the COMMIT file a release archive carries; None for neither."""
    if (folder / ".git").exists():
        out = run_cmd(["git", "-C", str(folder), "rev-parse", "HEAD"]).strip().lower()
        if HEX40.match(out):
            return out
    try:
        text = (folder / "COMMIT").read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return text if HEX40.match(text) else None


def engine_build_runtime(exe: str | None) -> dict | None:
    """The GPU runtime a release engine was built with, from its engine-component.json (build_id "a7bdbfdabc9a+cuda12.9"):
    {"name": "cuda", "version": "12.9"}. None for a source build or an engine without that file."""
    if not exe:
        return None
    try:
        record = json.loads((Path(exe).parent / "engine-component.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    match = re.search(r"\+(cuda|rocm|hip|sycl)([\d.]+)\Z", str(record.get("build_id") or "")) if isinstance(record, dict) else None
    return {"name": "rocm" if match.group(1) == "hip" else match.group(1), "version": match.group(2)} if match else None


def software(why: Why, machine: dict, *, engine_version: str | None, build_options: list[str] | None,
             engine_build: dict | None = None) -> dict:
    commit = git_commit()
    runtime = []
    if engine_build:
        # the library the engine runs on; nvidia-smi's "CUDA Version" is only the newest one the driver supports
        runtime.append(engine_build)
        if machine.get("cuda") and engine_build["name"] == "cuda":
            runtime.append({"name": "cuda-driver-supports", "version": machine["cuda"]})
    elif machine.get("cuda"):
        runtime.append({"name": "cuda", "version": machine["cuda"]})
    if machine.get("rocm") and not any(r.get("name") == "rocm" for r in runtime):
        runtime.append({"name": "rocm", "version": machine["rocm"]})
    if sys.platform == "darwin":
        runtime.append({"name": "metal", "version": platform.mac_ver()[0] or platform.release()})
    drivers = [{"name": "gpu-driver", "version": machine["driver"]}] if machine.get("driver") else []
    block = {"os": os_info(), "drivers": drivers or None, "runtime": runtime or None,
             "strata_commit": commit, "engine_version": engine_version,
             "build_kind": "source" if (ROOT / ".git").exists() else "release",
             "runner_commit": commit, "build_options": build_options}
    if not drivers:
        why.add("/software/drivers", "no GPU driver version was reported by the local tools")
    if not runtime:
        why.add("/software/runtime", "no CUDA, ROCm or Metal runtime version was reported by the local tools")
    if commit is None:
        for field in ("strata_commit", "runner_commit"):
            why.add(f"/software/{field}", "not a git checkout and no COMMIT file in this install; a release must carry its commit")
    if engine_version is None:
        why.add("/software/engine_version", "the server reported no engine version (/props build_info, /v1/status engine)")
    if build_options is None:
        why.add("/software/build_options", "the server does not report build options; pass --build-options for a source build")
    return block


# ------------------------------------------------------------------------------------------------------------ model
QUANT = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_[A-Z0-9_]+|Q\d_K(?:_[A-Z]+)?|IQ\d_?[A-Z]*|BF16|F16|NVFP4|Q\d))(?=[-_.]|$)")
SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$")


def model_files_from(paths: list[str]) -> list[Path]:
    """The model files behind these paths: each existing file, and the other shards of a split GGUF."""
    found: dict[str, Path] = {}
    for raw in paths:
        try:
            path = Path(raw)
            if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
        except (OSError, ValueError):
            continue
        found[str(path.resolve())] = path
        match = SHARD.match(path.name)
        if match:
            total = int(match.group(3))
            for index in range(1, total + 1):
                sibling = path.with_name(f"{match.group(1)}-{index:05d}-of-{match.group(3)}{match.group(4)}")
                if sibling.is_file():
                    found[str(sibling.resolve())] = sibling
    return sorted(found.values(), key=lambda p: p.name)


def detect_quantization(names: list[str]) -> str | None:
    for name in names:
        match = QUANT.search(name)
        if match:
            return match.group(1).upper()
    return None


def model(why: Why, *, props: dict, status: dict, files: list[Path], declared: dict, log) -> dict:
    served = props.get("model_alias") or status.get("model")
    rows = []
    for index, path in enumerate(files):
        log(f"hashing model file {index + 1}/{len(files)}: {path.name}")
        rows.append({"path": path.name, "sha256": sha256_file(path)})
    quant = declared.get("quantization") or detect_quantization([p.name for p in files] + [str(served or "")])
    block = {"id": str(served) if served else None,
             "revision": declared.get("model_revision"), "source_revision": declared.get("source_revision"),
             "quantization": quant, "files": rows or None, "tokenizer_revision": declared.get("tokenizer_revision"),
             "custom_pack": declared.get("custom_pack")}
    reasons = {"id": "the server reported no model name", "quantization": "no quantization name found in the model file names; pass --quantization",
               "files": "no local model file could be found to hash (the server reports no model_path or it is not on this host)",
               "revision": "not given: the quantization repository's immutable revision cannot be read from the engine; pass --model-revision",
               "source_revision": "not given: pass --source-revision",
               "tokenizer_revision": "not given: pass --tokenizer-revision",
               "custom_pack": "no custom pack or expert profile was declared by the operator (--custom-pack)"}
    for field, reason in reasons.items():
        if block[field] is None:
            why.add(f"/model/{field}", reason)
    return block


# ------------------------------------------------------------------------------------------------------------ configuration
GROUPS = (
    ("kv_cache", ("--kv",)),
    ("expert_cache", ("--expert",)),
    ("offloading", ("--gpu", "--layer-split", "--low-ram", "--offload", "--split", "--vram", "--pcie", "--host-core",
                    "--pool", "--numa", "--unmap", "--cpu-expert", "--tail-skip")),
    ("prefill", ("--prefill",)),
    ("mtp", ("--mtp", "--spec", "--draft", "--batch-mtp")),
    ("calibration", ("--calib",)),
    ("experimental_options", ("--experimental", "--exp-")),
)


def base_if_path(value):
    if isinstance(value, str) and (("/" in value or "\\" in value) or re.search(r"\.(gguf|bin|json|pack|txt|jinja|so|dll|exe)$", value, re.I)):
        return re.split(r"[\\/]", value.rstrip("\\/"))[-1]
    return value


def parse_options(argv: list[str]) -> dict:
    """['--kv', 'int8', '--low-ram'] -> {'--kv': 'int8', '--low-ram': True}; a repeated option becomes a list."""
    options: dict = {}
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if token.startswith("--"):
            name, value = (token.split("=", 1) + [None])[:2] if "=" in token else (token, None)
            if value is None:
                nxt = argv[i + 1] if i + 1 < len(argv) else None
                if nxt is not None and not (str(nxt).startswith("--") and not re.fullmatch(r"-?\d+(\.\d+)?", str(nxt))):
                    value, i = str(nxt), i + 1
                else:
                    value = True
            value = base_if_path(value)
            if name in options:
                options[name] = [*(options[name] if isinstance(options[name], list) else [options[name]]), value]
            else:
                options[name] = value
        i += 1
    return options


def configuration(why: Why, *, options: dict | None, env: dict | None, props: dict, declared: dict,
                  sampling: dict, template_ref: dict | None) -> tuple[dict, dict]:
    """-> (the report's `configuration`, the engine options no group takes: they go to extensions)."""
    grouped = {name: {} for name, _ in GROUPS}
    other = {}
    for name, value in (options or {}).items():
        for group, prefixes in GROUPS:
            if name.startswith(prefixes):
                grouped[group][name] = value
                break
        else:
            other[name] = value
    for name, value in sorted((env or {}).items()):
        if name.startswith("STRATA_") and not SECRET_KEY.search(name):
            grouped["experimental_options"][name] = base_if_path(str(value))
    n_ctx = ((props.get("default_generation_settings") or {}).get("n_ctx"))
    slots = props.get("total_slots")
    block = {"track": declared["track"], "track_notes": declared["track_notes"],
             "context_tokens": n_ctx if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) else None,
             "server_capacity": slots if isinstance(slots, int) and not isinstance(slots, bool) else None,
             **({name: grouped[name] for name, _ in GROUPS} if options is not None else {name: None for name, _ in GROUPS}),
             "template": template_ref, "sampling": dict(sampling["sampling"]),
             "chat_template_kwargs": dict(sampling["chat_template_kwargs"])}
    if block["context_tokens"] is None:
        why.add("/configuration/context_tokens", "the server's /props reported no context size")
    if block["server_capacity"] is None:
        why.add("/configuration/server_capacity", "the server's /props reported no slot count")
    if options is None:
        for name, _ in GROUPS:
            why.add(f"/configuration/{name}", "engine options are not exposed by a running server; start it from a profile or pass --config")
    if template_ref is None:
        why.add("/configuration/template", "the server's /props returned no chat template")
    return block, other
