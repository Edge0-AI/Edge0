"""setup.py on a Mac: Apple Silicon, the Metal engine (metal/), built from source on the Mac.

    ./setup.sh [options]          (setup.py hands a macOS run to this file; the PC paths are not used)

What it does, each step skipped when it is already done:

  1. checks the Mac: Apple Silicon, the chip, unified memory, macOS version, the Command Line Tools, free disk
  2. recommends a model by memory (recommend, never force: an explicit --model is kept, with what it risks)
  3. installs the Python packages into .venv (cmake and ninja come from pip, just before step 4: no Homebrew, no sudo)
  4. builds the Metal engine from source: metal-cpp (Apple's headers) and llama.cpp's ggml at the pinned commit
  5. downloads the model (setup.py's downloader: resumable, the pinned revision)
  6. prepares the model: the native pack (tools/iq_pack.py) and the MTP draft layer (tools/mtp_fetch.py, mtp_pack.py)
  7. writes strata-<model>.json and run-<model>.sh with the Metal defaults (--spec 2, the Q8_0 hyper-connection
     read) and starts the server (serve/server.py), the same as on a PC

Everything shared - the model table, the downloader, the config and run-script writers, the server - is setup.py's.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from strata_control import mac_fit as _fit  # noqa: E402
from strata_control.mac_fit import (  # noqa: E402,F401  (re-exported: the tests and callers use these names here)
    BIG_MAC_GB, CONTEXTS, GIB_GB, GPU_SHARE, MAC_ABOUT, MAC_MODELS, MIN_MACOS, OVERHEAD_GB, RECOMMENDED,
    RESIDENT_LEAVE_SHARE, STREAM_COPY_GAIN, STREAM_MEASURED, STREAM_MTP_FROM_GB, STREAM_SPEC, TESTED_CONTEXT,
    _sysctl, gpu_budget_gb, mac_info, mac_problems, macos_tuple, model_need_gb, model_note, recommend_context,
    recommend_model, stream_mtp, streams)

# Apple's metal-cpp headers for the macOS 26 SDK (https://developer.apple.com/metal/cpp/), pinned by SHA-256
METAL_CPP_URL = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
METAL_CPP_SHA256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
BUILD_DIR = ROOT / "build-metal"
EXE = BUILD_DIR / "strata"

# the Unsloth files at setup.py's pinned revision: name -> (bytes, SHA-256), the Hub's LFS pointers (UD-Q4_K_XL's
# are setup.py's UNSLOTH_SHARDS)
UD_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf": (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf": (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf": (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf": (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf": (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf": (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf": (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
}


def model_shard_names(model: str) -> list[str]:
    """The model's files in order (shard 1 first)."""
    n = MAC_MODELS[model].get("shards", 2)
    if MAC_MODELS[model].get("family") == "gyro":          # agentionai's one file (the n-gram table is inside it)
        return [S.model_file(S.FAMILIES["gyro"], model, 1)]
    if MAC_MODELS[model].get("family") == "unsloth":
        return [f"Qwen3.8-Flash-Next-{model}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    return [S.model_file(S.FAMILIES["qwen"], model, i) for i in range(1, n + 1)]


def shard_sha(name: str):
    return UD_SHARDS.get(name) or getattr(S, "UNSLOTH_SHARDS", {}).get(name)


# ------------------------------------------------------------------------------------------------ the Mac
# ------------------------------------------------------------------------------------------------ the engine
def metal_cpp() -> Path:
    """Apple's metal-cpp headers in third_party/metal-cpp (one zip, pinned)."""
    dst = ROOT / "third_party" / "metal-cpp"
    if (dst / "Metal" / "Metal.hpp").exists():
        return dst
    z = ROOT / "third_party" / "metal-cpp_26.zip"
    S.download(METAL_CPP_URL, z, "metal-cpp (Apple's Metal headers)")
    S.verify_sha256(z, z.stat().st_size, METAL_CPP_SHA256)
    tmp = ROOT / "third_party" / "_metal_cpp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(tmp / "metal-cpp"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


def tool(name: str) -> str:
    """cmake / ninja from the venv's pip packages (no Homebrew)."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def engine_current() -> bool:
    try:
        meta = json.loads((BUILD_DIR / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return EXE.exists() and meta.get("version") == S.source_version() and meta.get("source") == source_digest()


def source_digest() -> str:
    """What the engine is built from: every source file's path, size and modification time (src/, include/, metal/,
    third_party/ggml).  A git pull, an unzipped update or a local edit changes it; nothing else does."""
    h = hashlib.sha256()
    for d in ("src", "include", "metal", "third_party/ggml"):
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and f.suffix in (".cpp", ".cu", ".h", ".hpp", ".mm", ".metal", ".inc", ".txt", ".cuh"):
                st = f.stat()
                h.update(f"{f.relative_to(ROOT).as_posix()}|{st.st_size}|{int(st.st_mtime)}\n".encode())
    return h.hexdigest()


def build_engine(llama: Path, jobs: int | None = None) -> Path:
    """The Metal engine from source: cmake + ninja (pip's), Apple's clang from the Command Line Tools."""
    if engine_current():
        S.ok(f"Metal engine: {EXE} (built for this source)")
        return EXE
    mcpp = metal_cpp()
    S.ensure_build_tools()                                # cmake and ninja are in requirements-build.txt, not requirements.txt (#3403)
    S.say("  Building the Metal engine from source (5-15 minutes the first time) ...")
    BUILD_DIR.mkdir(exist_ok=True)
    log = BUILD_DIR / "build.log"
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    cfg = [tool("cmake"), "-S", str(ROOT / "metal"), "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DMETAL_CPP_DIR={mcpp}", f"-DSTRATA_GGML_DIR={llama}"]
    build = [tool("ninja"), "-C", str(BUILD_DIR), "strata"] + (["-j", str(jobs)] if jobs else [])
    with open(log, "w", encoding="utf-8") as f:
        for cmd in (cfg, build):
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if rc != 0:
                S.fail(f"the Metal engine did not build (see {log})",
                       "the last lines of the log name the problem; the Command Line Tools must be up to date "
                       "(softwareupdate --list)")
    (BUILD_DIR / "BUILD.json").write_text(json.dumps({"version": S.source_version(), "source": source_digest(),
                                                      "backend": "metal"}, indent=1), encoding="utf-8")
    S.ok(f"Metal engine built: {EXE}")
    return EXE


# ------------------------------------------------------------------------------------------------ the config
def engine_args(pack: Path, shards: list, rt: Path, ctx: int, stream: bool = False, ram_gb: float = 0.0) -> list[str]:
    """The Metal defaults (strata_control/mac_fit.py), with this checkout's expert profile."""
    return _fit.engine_args(pack, shards, rt, ctx, stream, ram_gb, ROOT / "data" / "expert-profile.bin")


# the image encoder's projector (setup.py's pinned revision): name -> (bytes, SHA-256)
VISION_SHA256 = {"mmproj-Qwen3.8-Flash-Next-BF16.gguf":
                 (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")}
VISION_EXE = ROOT / "build-vision" / "bin" / "strata-vision"


def build_vision(llama: Path) -> Path:
    """strata-vision (tools/vision: llama.cpp's mtmd at the pinned commit), with ggml's Metal backend: the encoder
    runs on the GPU."""
    stamp = ROOT / "build-vision" / "BUILD.json"
    src = source_digest()
    try:
        if VISION_EXE.exists() and json.loads(stamp.read_text(encoding="utf-8")).get("source") == src:
            S.ok(f"image encoder: {VISION_EXE}")
            return VISION_EXE
    except (OSError, ValueError):
        pass
    S.ensure_build_tools()
    S.say("  Building the image encoder (1-3 minutes) ...")
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    log = ROOT / "build-vision.log"
    with open(log, "w", encoding="utf-8") as f:
        for cmd in ([tool("cmake"), "-S", str(ROOT / "tools" / "vision"), "-B", str(ROOT / "build-vision"), "-G", "Ninja",
                     "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DLLAMA_DIR={llama}",
                     "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_VISION_METAL=ON", "-DSTRATA_PORTABLE=OFF"],
                    [tool("ninja"), "-C", str(ROOT / "build-vision"), "strata-vision"]):
            if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env) != 0:
                S.fail(f"the image encoder did not build (see {log})")
    stamp.write_text(json.dumps({"source": src}), encoding="utf-8")
    S.ok(f"image encoder built: {VISION_EXE}")
    return VISION_EXE


def ple_shard(files: list) -> Path:
    """The shard that holds the n-gram table (per_layer_token_embd)."""
    sys.path.insert(0, str(ROOT / "metal" / "tools"))
    import metal_repack as R
    for f in files:
        rd = R.Reader(f)
        rd.f.close()
        if any(t["name"] == R.PLE for t in rd.tensors):
            return f
    S.fail(f"no shard of {files[0].name} holds the n-gram table (per_layer_token_embd)")


# the experts' streaming copy (metal/tools/stream_experts.py): what stays free on the disk after it is written
EXPERTS_MARGIN_GB = 10.0


def experts_file_fits(need_gb: float, free_gb: float) -> bool:
    return free_gb >= need_gb + EXPERTS_MARGIN_GB


def experts_file(native: Path) -> None:
    """A streaming Mac: <model>.experts beside the GPU's GGUF - every expert's rows in one piece, read with one SSD
    request per expert instead of three (the engine uses it when it is there and matches; else the GGUF).  Written
    only when the disk has room for it and EXPERTS_MARGIN_GB more; otherwise setup says so and goes on."""
    out = native.with_name(native.name + ".experts")
    if out.exists() and S.done(out):
        S.ok(f"the experts' streaming copy: {out.name}")
        return
    sys.path.insert(0, str(ROOT / "metal" / "tools"))
    import stream_experts as X
    need = X.size_bytes(native) / 1e9
    free = S.free_gb(native.parent)
    if not os.access(native.parent, os.W_OK):
        S.warn(f"{native.parent} cannot be written: the experts' streaming copy ({need:.0f} GB) is not written; the "
               f"engine streams from the model file ({STREAM_COPY_GAIN} slower)")
        return
    if not experts_file_fits(need, free):
        S.warn(f"the experts' streaming copy needs {need:.0f} GB and {EXPERTS_MARGIN_GB:.0f} GB to spare on the disk "
               f"({free:.0f} GB free): not written; the engine streams from the model file ({STREAM_COPY_GAIN} "
               f"slower). Free some space and run ./setup.sh --setup again to add it")
        return
    S.say(f"  Writing the experts in SSD-read order (one time, {need:.0f} GB; each expert is read with one request) ...")
    S.run([sys.executable, str(ROOT / "metal" / "tools" / "stream_experts.py"), "--gguf", str(native), "--verify"])
    S.mark(out)
    S.ok(f"the experts' streaming copy: {out.name}")


def mac_config(exe: Path, args: list, pack: Path, model_name: str, tag: str, port: int) -> dict:
    return {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
            "model_name": model_name, "log": str(ROOT / f"strata-{tag.lower()}.log"), "lib_dirs": [], "port": port,
            "backend": "metal"}


def mac_configs() -> list[Path]:
    out = []
    for p in S.installed_configs():
        try:
            if json.loads(p.read_text(encoding="utf-8-sig")).get("backend") == "metal":
                out.append(p)
        except (OSError, ValueError):
            pass
    return out


REPACK_LOCK = ".repack.lock"       # metal_repack.py: <out>.repack.lock holds the repack's pid while it runs
PREPARE_LOCK = ".preparing.lock"   # setup: beside the repacked file, holds setup's pid from the repack to the end of the pack


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True            # it exists, it is somebody else's
    except OSError:
        return False
    return True


def repack_running(folder: Path) -> list[int]:
    """Pids of repacks (metal_repack.py) or setup's own model preparation (repack + pack) still running for the files in
    `folder`.  A stale lock (its process gone) is removed.  The repack and the pack move 100+ GB through the disk and one
    holds up to 20 GB of memory: an engine with a 70-90 GB footprint beside them is what froze the 128 GB M5 Max twice
    (watchdog panics, 2026-10-13)."""
    pids = []
    if not folder.is_dir():
        return pids
    for lock in [*folder.glob("*" + REPACK_LOCK), folder / PREPARE_LOCK]:
        try:
            pid = int(lock.read_text().strip())
        except (OSError, ValueError):
            continue
        if pid != os.getpid() and _pid_alive(pid):
            pids.append(pid)
        elif pid != os.getpid():
            try:
                lock.unlink()
            except OSError:
                pass
    return pids


def engine_pids(run=subprocess.run) -> list[int]:
    """Pids of the running Metal engines (`strata`, the executable setup builds)."""
    try:
        out = run(["pgrep", "-x", "strata"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split() if x.isdigit() and int(x) != os.getpid()]


def start(cfg_path: Path, port: int | None, open_browser: bool = True) -> int:
    """serve/server.py with this config (setup.py's start without the NVIDIA / AMD card checks)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run ./setup.sh --setup to repair it")
    busy = [p for g in cfg["args"] if g.endswith(".gguf") for p in repack_running(Path(g).parent)]
    if busy:       # never start an engine while a repack (or the pack after it) is writing: wait would hide a stuck one
        S.fail(f"a model repack is still running here (pid {busy[0]}): the engine is not started beside it",
               "wait for it to finish (or stop it), then start again")
    cfg_path.touch()
    if "--mtp" in cfg["args"][:-1]:
        S.refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser and cfg.get("open_browser") is not False:
        cmd.append("--open")
    S.say()
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on Metal: the engine maps the model file and is ready in about "
          "half a minute (longer the first time after a restart, while macOS reads the file).")
    S.say("  Closing this window stops the model.")
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=["qwen", "unsloth", "gyro"], default=None,
                    help="qwen = Qwen3.8-Flash-Next (GSQ-RCO), unsloth = Unsloth's UD-*_K_XL, gyro = agentionai's Gyro-S / Gyro-M "
                         "(follows --model)")
    ap.add_argument("--model", choices=list(MAC_MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--api-key")
    ap.add_argument("--draft-vocab", choices=list(S.DRAFT_VOCABS))
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--data-dir")
    ap.add_argument("--move-data", action="store_true", help="with --data-dir: move the model files already here into the new folder")
    ap.add_argument("--set-default-data-dir", action="store_true", help="with --data-dir: remember the new folder without moving anything")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", choices=S.SOURCES, default=None,
                    help="where model files come from: auto (huggingface), huggingface, modelscope; "
                         "same as STRATA_SOURCE")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--build", action="store_true", help="build the engine again even when it is current")
    ap.add_argument("--vision", choices=["yes", "no", "gpu", "cpu"],
                    help="let the model read images (the image encoder, built here; yes = gpu: it reads on the Mac's GPU, cpu is ~100x "
                         "slower; default: asked, yes)")
    a, rest = ap.parse_known_args(argv)
    if rest:
        S.warn("not used on a Mac: " + " ".join(rest))
    # setup.py hands a Mac over before its own parser runs, so --source has to be honoured here too:
    # source_hint() recommends it on every platform, and a Mac that dropped it failed the download
    # with the same unreachable host (#2363).
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    S.say("Strata on a Mac (Apple Silicon, Metal)")
    data, elsewhere = S.data_folder_for(a)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"

    have = mac_configs()
    if have and not (a.setup or a.model or a.check or a.no_start or a.update or a.build):
        return start(have[0], a.port, a.browser is not False)

    # ---- 1. the Mac
    S.step(1, "checking your Mac")
    info = mac_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = mac_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    # ---- 2. the choices
    S.step(2, "your choices")
    rec, notes = recommend_model(info["ram_gb"])
    for n in notes:
        S.warn(n)
    if a.model:
        model = a.model
    else:
        names = [m for m in MAC_MODELS if not MAC_MODELS[m].get("missing")]   # --model offers the others
        for i, m in enumerate(names, 1):
            S.say(f"  {i}) {m:8} {MAC_ABOUT[m]}  (maps {MAC_MODELS[m]['shard1_gb']:.0f} GB"
                  + ("" if MAC_MODELS[m]["tested"] else ", not measured on a Mac") + ")"
                  + ("  (recommended)" if m in RECOMMENDED else "") + ("   <- default" if m == rec else ""))
        model = names[int(S.ask("Which size?", [str(i) for i in range(1, len(names) + 1)],
                                str(names.index(rec) + 1), a.yes)) - 1]
    for n in model_note(model, info["ram_gb"]):
        S.warn(n)
    vision = a.vision or ("yes" if S.ask("Should the model also read images? (the image encoder, ~1 GB more)",
                                          ["y", "n"], "y", a.yes) == "y" else "no")   # on by default (strata_control/vision_reader.py)
    if vision == "cpu":
        from strata_control import vision_reader as VR
        S.warn(VR.MAC_CPU_NOTE)
    vision_gpu = vision != "cpu"
    vision = "no" if vision == "no" else "yes"
    ctx = a.context or recommend_context(info["ram_gb"])
    if ctx > TESTED_CONTEXT:
        S.say(f"  context {ctx}: measured on the Mac up to {TESTED_CONTEXT} (needles found at 8K-32K); longer is kept")
    S.ok(f"{model}, context {ctx}")
    family = MAC_MODELS[model].get("family", "qwen")
    if a.family and a.family != family:
        S.warn(f"--family {a.family}: {model} is a {family} file; using {family}")
    fam = S.FAMILIES[family]
    tag = fam["tag"] + model
    need = (0 if a.gguf_dir else S.MODELS[model]["download_gb"]) + 6   # --gguf-dir: the files are there already
    if free < need:
        S.warn(f"about {need:.0f} GB are needed on the disk ({free:.0f} GB free): kept as you chose; it may run out")

    # ---- 3. Python packages (cmake and ninja come with the build in step 4, from requirements-build.txt)
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES, "numpy, jinja2, ...")

    # ---- 4. the engine
    S.step(4, "the Strata engine (Metal)")
    llama = S.get_llama_cpp()
    S.ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    if a.build:
        (BUILD_DIR / "BUILD.json").unlink(missing_ok=True)
    exe = build_engine(llama)
    if a.update:
        # update.sh's own text: "The model files are not touched".  The questions are asked anyway, so when the
        # answers name a model (or an image encoder) that is not here, say it: a silent exit reads like a broken
        # download (an M1 Max chose IQ3_S in update.sh and got no model and no message).
        files = [Path(a.gguf_dir) / n if a.gguf_dir else models_root / tag / n for n in model_shard_names(model)]
        pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
        pack_done = (pack / "native_experts.txt").exists() and (pack / "tokenizer" / "vocab.json").exists()
        if not (all(f.exists() and S.done(f) for f in files) and pack_done):
            S.warn(f"{model} is not installed here, and an update does not download model files: nothing was installed")
            S.say(f"  to download and prepare {model}: ./setup.sh --setup")
        elif vision == "yes" and not (models_root / fam["mmproj"]).exists():
            S.warn("the image encoder is not downloaded here, and an update does not download model files")
            S.say("  to add image support: ./setup.sh --setup")
        return 0

    # ---- 5. the model
    S.step(5, f"downloading {fam['title']} {model}")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    models_dir.mkdir(parents=True, exist_ok=True)
    files = [models_dir / n for n in model_shard_names(model)]
    for s in files:
        if s.exists() and S.done(s):
            S.ok(f"{s.name} already downloaded")
            continue
        S.download(fam["hf"].format(q=model) + s.name, s)
    S.check_shards(files)
    for s in files:                                    # the Unsloth files: pinned sizes and SHA-256
        if shard_sha(s.name):
            S.verify_sha256(s, *shard_sha(s.name))
    S.ok("model files present")

    mmproj = vexe = None
    if vision == "yes":
        mmproj = models_root / fam["mmproj"]
        if not (mmproj.exists() and S.done(mmproj)):
            S.download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        sha = VISION_SHA256.get(fam["mmproj"])
        if sha:
            S.verify_sha256(mmproj, *sha)
        vexe = build_vision(llama)

    # ---- 6. the pack and the MTP draft layer
    S.step(6, "preparing the model for Strata")
    pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    # The repack and the pack move 100+ GB through the disk and the pack's python holds up to 20 GB: no engine may run beside
    # them (an engine of 70-90 GB with a repack froze the 128 GB M5 Max twice).  A lock beside the GPU's file tells the engine
    # and start() to wait; a running engine stops setup here.
    lock_dir = data / "metal" if len(files) > 2 else files[0].parent
    native0 = data / "metal" / f"{tag.lower()}-metal.gguf"
    work = (len(files) > 2 and not (native0.exists() and S.done(native0))) or \
        not ((pack / "native_experts.txt").exists() and (pack / "tokenizer" / "vocab.json").exists())
    prep_lock = lock_dir / PREPARE_LOCK
    if work:
        running = engine_pids()
        if running:
            S.fail(f"a Strata engine is running (pid {running[0]}): preparing a model beside it can freeze this Mac",
                   "stop the server (close its window), then run ./setup.sh --setup again")
        lock_dir.mkdir(parents=True, exist_ok=True)
        prep_lock.write_text(str(os.getpid()))
        atexit.register(lambda: prep_lock.unlink() if prep_lock.exists() else None)
    if len(files) > 2:
        # many shards: the experts and dense tensors into one GGUF the GPU maps (or streams), the n-gram table read
        # from its own shard (every tensor checked byte for byte: --verify)
        native = data / "metal" / f"{tag.lower()}-metal.gguf"
        native.parent.mkdir(parents=True, exist_ok=True)
        if not (native.exists() and S.done(native)):
            S.say(f"  Writing the model's experts and weights into one file for the GPU (one time, "
                  f"{MAC_MODELS[model]['shard1_gb']:.0f} GB) ...")
            S.run([sys.executable, str(ROOT / "metal" / "tools" / "metal_repack.py"), "--shard1", str(files[0]),
                   "--out", str(native), "--verify"])
            S.mark(native)
        ple = ple_shard(files)
        shards = [native, ple]
    elif family == "gyro":
        shards = [files[0], files[0]]                  # one file: the model and its n-gram table (--native, --ple-gguf)
    else:
        shards = files
    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
               *fam.get("pack_args", [])], env=env)   # Unsloth: --compat-bf16 (the host's copies; the GPU reads the file)
    S.ok(f"model prepared: {pack}")
    prep_lock.unlink(missing_ok=True)
    mtp = (S.find_in(roots, "mtp/mtp-q2_0.gguf") or data / "mtp" / "mtp-q2_0.gguf").parent
    rt = mtp / "rt"
    stream = streams(model, info["ram_gb"])
    use_mtp = stream_mtp(stream, info["ram_gb"])
    if stream and not use_mtp:
        S.ok("the experts will stream from the SSD: no MTP draft layer (one token per step is faster there); "
             "./setup.sh --setup on a Mac with more memory adds it")
    elif stream:
        S.ok(f"the experts will stream from the SSD; the MTP draft layer stays on, with --spec {STREAM_SPEC} "
             "(a longer window pays when every window has a fixed cost per layer)")
    if stream and info["ram_gb"] < STREAM_MTP_FROM_GB:
        experts_file(shards[0])           # a small Mac: its misses are many; a big Mac's arena holds most experts
    if use_mtp and not (mtp / "mtp-q2_0.gguf").exists():
        S.say("  The MTP draft layer (speculative decoding) comes from the original Qwen checkpoint: only its ~5 GB of")
        S.say("  MTP tensors are downloaded.")
        S.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)],
              env={**env, "STRATA_SOURCE": S.model_source()})
        S.run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
               "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
    if use_mtp:
        rt.mkdir(parents=True, exist_ok=True)          # the Metal engine reads mtp-q2_0.gguf beside it (no rt pack)
        S.refresh_draft_vocab(rt, a.draft_vocab or "cjk")
        S.ok(f"MTP draft layer: {mtp / 'mtp-q2_0.gguf'}")

    # ---- 7. the config, the start script
    S.step(7, "writing the start script")
    port = a.port or 8080
    cfg = mac_config(exe, engine_args(pack, shards, rt, ctx, stream, info["ram_gb"]), pack, f"{fam['name']}-{model.lower()}", tag, port)
    cfg["env"] = {"STRATA_METAL_STREAM": "1" if stream else "0"}   # the engine follows setup's measured choice
    if vision == "yes":
        cfg["args"] += ["--vision"]
        cfg["vision"] = {"exe": str(vexe), "mmproj": str(mmproj), "model": str(files[0]), "gpu": vision_gpu,
                         "max_tokens": 1024}
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.draft_vocab:
        cfg["draft_vocab"] = a.draft_vocab
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    S.write_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    S.ok(f"start script: {script.name}")
    S.say()
    S.say("All set.")
    S.say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1")
    S.say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    S.say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port, a.browser is not False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
