"""The Mac's model rules: which packs a Mac can run, whether one stays resident or streams its experts from the SSD, the
Metal defaults of a run config. ONE definition, read by metal/setup_mac.py (a Mac that builds the engine itself), by setup.py's
Control mode (the app and the Terminal on a Mac: model files only, the engine is a component) and by tools/strata_mcp.py (the
model list, the fit words and the recommendation both show). Pure: nothing here downloads, builds or starts anything.

Unified memory: a Mac has one pool. Metal keeps a share of it resident (GPU_SHARE; on a big Mac the memory less what macOS and
its file cache keep), a pack that does not fit streams its experts from the SSD (the engine decides by itself).
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

MIN_MACOS = (15, 0)        # MTLResidencySet (the engine keeps every buffer resident through one)

# The Mac's packs.  shard1_gb: the file the GPU maps (resident in unified memory while the model runs); the n-gram
# table (shard 2, 28.8 GB) is read from the SSD a few rows at a time.  tested: measured on the M5 Max (docs/MAC.md).
MAC_MODELS = {
    "Q2_0": {"shard1_gb": 37.6, "tested": True},
    "IQ3_XXS": {"shard1_gb": 47.0, "tested": True},
    "IQ3_S": {"shard1_gb": 54.8, "tested": True},
    # Unsloth's ~4-6-bit files (family "unsloth", 4-6 shards cutting through layers): setup repacks the experts and
    # dense tensors into one GGUF for the GPU (metal/tools/metal_repack.py); shard1_gb is that file, the n-gram
    # table stays in its shard on the SSD
    "UD-Q4_K_XL": {"shard1_gb": 82.6, "tested": True, "family": "unsloth", "shards": 4},
    "UD-Q5_K_XL": {"shard1_gb": 103.9, "tested": True, "family": "unsloth", "shards": 6},
    "UD-Q6_K_XL": {"shard1_gb": 114.8, "tested": True, "family": "unsloth", "shards": 6},
    # agentionai's Gyro files (family "gyro", docs/GYRO.md): ONE file with the n-gram table inside (58.5 / 92.0 GB to download);
    # shard1_gb is what the GPU needs of it, without that table (the engine reads the table from the SSD a few rows at a time).
    # The trellis experts run on the Metal engine from the release that added metal/kernels' TQ types (docs/MAC.md).
    "Gyro-S": {"shard1_gb": 29.7, "tested": True, "family": "gyro", "shards": 1, "file_gb": 58.5},
    "Gyro-M": {"shard1_gb": 63.2, "tested": False, "family": "gyro", "shards": 1, "file_gb": 92.0},
}

# The recommended sizes (owner's decision): IQ3_S is the default where it stays resident, UD-Q4_K_XL the higher-quality one
RECOMMENDED = ("IQ3_S", "UD-Q4_K_XL")   # the owner's recommended sizes (UD-IQ4_XS is not offered on Mac yet)
# a pack that does not fit Metal's share streams its experts from the SSD (the engine decides by itself, f-mac-ssd):
# measured on a 32 GB M4 Mac mini, one token per step (MTP is slower there: its layer's memory is worth more as cache)
STREAM_MEASURED = {"Q2_0": "~14 tok/s", "IQ3_XXS": "~10 tok/s"}
STREAM_COPY_GAIN = "decoding about 15%, prompts about 25%"   # without the experts' copy, on the 32 GB M4 (IQ3_XXS)
# A streaming Mac with this much memory keeps most of the experts in its arena (the 128 GB M5 Max: 80% of UD-Q5_K_XL's):
# there the draft layer pays, and a longer window pays more because every window costs a fixed hand-shake per layer
# (UD-Q5_K_XL, story: 28 tok/s without MTP, 41.8 at --spec 2, 50.6 at --spec 3). Smaller Macs stream without it.
STREAM_MTP_FROM_GB = 96
STREAM_SPEC = 3
# beside the mapped file: the prompt kernels' buffers (chunks of 2,048 rows), the KV cache, the draft layer, the
# Q8_0 copies of the hyper-connection matrices and the macOS desktop
OVERHEAD_GB = 8.0
# the share of the unified memory Metal lets one process keep resident (recommendedMaxWorkingSetSize): 107.5 of 128 GB
# measured on the M5 Max; 3/4 is the conservative rule for the other sizes
GPU_SHARE = 0.75
# what a Mac of BIG_MAC_GB or more leaves to macOS, its file cache and the user's other programs when the pack stays RESIDENT
# (the engine's resident_cap in arena_budget.h, the same rule).  A resident pack is mapped file pages: clean, so macOS can
# drop them, never compress them.  Measured on the 128 GB M5 Max, UD-Q4_K_XL resident (wired 91.7 GB, 67%) with a 40 GB file
# write and a 16 GB program beside it: no swap-out, nothing compressed.  (The streaming arena is anonymous memory and gets a
# tighter cap in the engine: 40% left, from the same measurements.)
RESIDENT_LEAVE_SHARE = 0.30
BIG_MAC_GB = 64.0
GIB_GB = 1.073741824    # a GiB in GB: ram_gb is binary, the model sizes are decimal
CONTEXTS = [8192, 32768, 65536, 131072]
MAC_ABOUT = {"Q2_0": "2-bit, the fastest", "IQ3_XXS": "3-bit i-quant, better quality",
             "IQ3_S": "3.5-bit i-quant, the best quality (matches the full model), the slowest",
             "UD-Q4_K_XL": "Unsloth ~4.5-bit (111 GB download)", "UD-Q5_K_XL": "Unsloth ~5.5-bit (158 GB download)",
             "UD-Q6_K_XL": "Unsloth ~6.5-bit (169 GB download)",
             "Gyro-S": "agentionai's trellis experts, the smallest (58 GB download, the n-gram table is inside it)",
             "Gyro-M": "agentionai's trellis experts, about IQ3_XXS quality in less memory (92 GB download)"}
TESTED_CONTEXT = 32768     # needles and the KL harness ran to 32K on the Mac
# Below this much unified memory the app says a model does not fit (a Mac this small was never run: 32 GB is the smallest that was).
# The Terminal keeps its rule of recommending and never forcing: the person can still pick one there.
MIN_STREAM_RAM_GB = 16.0


def _sysctl(name: str, run=subprocess.run) -> str:
    try:
        return run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def mac_info(run=subprocess.run) -> dict:
    """The chip, its unified memory, macOS, the architecture and the Command Line Tools."""
    chip = _sysctl("machdep.cpu.brand_string", run) or "unknown chip"
    try:
        ram = int(_sysctl("hw.memsize", run) or 0) / 2 ** 30   # GB as Apple sells them (128 GB = 2^37 bytes)
    except ValueError:
        ram = 0.0
    try:
        ver = run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        ver = ""
    try:
        clt = run(["xcode-select", "-p"], capture_output=True, text=True, timeout=10)
        clt_path = clt.stdout.strip() if clt.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        clt_path = ""
    # platform.machine() is the PYTHON's build: an Intel Python running through Rosetta says x86_64 on an M1 Max (#3894). The chip name
    # (Rosetta passes "Apple M1 Max" through) and hw.optional.arm64 say what the Mac is.
    python_arm = platform.machine() == "arm64"
    mac_arm = python_arm or _sysctl("hw.optional.arm64", run) == "1" or chip.startswith("Apple")
    return {"chip": chip, "ram_gb": ram, "macos": ver, "arm64": mac_arm, "rosetta_python": mac_arm and not python_arm,
            "clt": clt_path, "perf_cores": int(_sysctl("hw.perflevel0.physicalcpu", run) or 0)}


def macos_tuple(ver: str) -> tuple:
    try:
        return tuple(int(x) for x in ver.split(".")[:2])
    except ValueError:
        return (0, 0)


def mac_problems(info: dict, need_compiler: bool = True) -> list[str]:
    """What stops the setup on this Mac (each with what to do). The app brings a ready-made engine and compiles nothing:
    need_compiler=False leaves the Command Line Tools out."""
    p = []
    if not info["arm64"]:
        p.append(f"this Mac has an Intel CPU ({info['chip']}): the Metal engine needs Apple Silicon (M1 or newer)")
    if macos_tuple(info["macos"]) < MIN_MACOS:
        p.append(f"macOS {info['macos'] or '?'}: the Metal engine needs macOS {MIN_MACOS[0]} or newer (System Settings "
                 "> General > Software Update)")
    if need_compiler and not info["clt"]:
        p.append("the Command Line Tools are missing (the compiler the engine is built with): run "
                 "`xcode-select --install`, then this setup again")
    return p


def gpu_budget_gb(ram_gb: float) -> float:
    """The most a resident pack may hold, in GB: Metal's working set, and on a big Mac the RAM less what macOS and the file
    cache keep (the engine's resident_cap: min of the two)."""
    ws = ram_gb * GPU_SHARE * GIB_GB
    if ram_gb >= BIG_MAC_GB:
        return min(ws, ram_gb * GIB_GB * (1.0 - RESIDENT_LEAVE_SHARE))
    return ws


def model_need_gb(model: str) -> float:
    return MAC_MODELS[model]["shard1_gb"] + OVERHEAD_GB


def streams(model: str, ram_gb: float) -> bool:
    """The pack does not fit Metal's share of the memory: the engine streams its experts from the SSD."""
    return model_need_gb(model) > gpu_budget_gb(ram_gb)


def recommend_model(ram_gb: float) -> tuple[str, list[str]]:
    """The default pack and the notes that go with it. IQ3_S and UD-Q4_K_XL are the recommended sizes (RECOMMENDED):
    the default is IQ3_S when it stays resident, with UD-Q4_K_XL named as the higher-quality choice where it fits too;
    below that the largest pack that stays resident (IQ3_XXS, Q2_0); nothing fits: Q2_0 streamed from the SSD."""
    if not streams("IQ3_S", ram_gb):
        notes = []
        if not streams("UD-Q4_K_XL", ram_gb):
            notes.append("UD-Q4_K_XL (recommended too) also fits this Mac: higher quality than IQ3_S (92.8% top-1 against "
                         "Q8_0 where IQ3_S has 87.9%), about 20% slower on a story; choose it with --model UD-Q4_K_XL")
        return "IQ3_S", notes
    for m in ("IQ3_XXS", "Q2_0"):
        if not streams(m, ram_gb):
            return m, []
    return "Q2_0", [f"{ram_gb:.0f} GB of memory: no pack fits whole, so the engine streams the experts from the SSD "
                    f"(automatic; measured on a 32 GB M4: Q2_0 {STREAM_MEASURED['Q2_0']}, IQ3_XXS "
                    f"{STREAM_MEASURED['IQ3_XXS']}; smaller Macs are untested). Recommended: Q2_0, without the MTP "
                    "draft layer (slower when streaming)"]


def model_note(model: str, ram_gb: float) -> list[str]:
    """What a chosen pack risks on this Mac (said, never refused)."""
    notes = []
    if streams(model, ram_gb):
        notes.append(f"{model} needs ~{model_need_gb(model):.0f} GB of the {gpu_budget_gb(ram_gb):.0f} GB Metal keeps "
                     "resident on this Mac: its experts stream from the SSD (automatic)"
                     + (f"; measured on a 32 GB M4: {STREAM_MEASURED[model]}" if model in STREAM_MEASURED else
                        "; not measured streaming"))
    if MAC_MODELS[model].get("missing"):
        notes.append(f"{model} does not run on this Metal engine yet ({MAC_MODELS[model]['missing']} are missing): the "
                     "engine will stop when it loads; kept as you chose (for testing a newer engine)")
    elif not MAC_MODELS[model]["tested"]:
        notes.append(f"{model} has not been measured on a Mac yet")
    return notes


def recommend_context(ram_gb: float) -> int:
    return 32768 if ram_gb < 96 else 65536


def stream_mtp(stream: bool, ram_gb: float) -> bool:
    """Whether the config carries the MTP draft layer: always when the model is resident; when it streams, only on a
    Mac with STREAM_MTP_FROM_GB or more (a 32 GB M4 is faster without it)."""
    return (not stream) or ram_gb >= STREAM_MTP_FROM_GB


def engine_args(pack: Path, shards: list, rt: Path, ctx: int, stream: bool = False, ram_gb: float = 0.0,
                profile: Path | str = "data/expert-profile.bin") -> list[str]:
    """The Metal defaults: --spec 2 with the MTP draft layer (measured best on the M5 Max, docs/MAC.md) and the Q8_0
    hyper-connection read (the engine's default on Metal; STRATA_HC_Q8=0 in the config's env gives the exact BF16
    read).  Streaming from the SSD: one token per step, no draft layer (measured faster on the 32 GB M4); a streaming
    Mac with STREAM_MTP_FROM_GB or more keeps the draft layer and uses --spec STREAM_SPEC."""
    mtp = stream_mtp(stream, ram_gb)
    args = ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[1])]
    if shards[0].name.endswith("-metal.gguf"):         # the repacked file: it is its own dense GGUF
        args += ["--native-dense-gguf", str(shards[0])]
    args += [
            "--expert-profile", str(profile), "--expert-cache", "64",
            "--prefill", "auto", "--spec", str(STREAM_SPEC if stream and mtp else 2)]
    args += ["--suffix-draft", "0"] if stream else []
    args += ["--mtp", str(rt)] if mtp else []
    return args + ["--max-context", str(ctx),
                   "--mmap-experts"]   # what the Metal engine does anyway (the GGUF mapped in place); the server says so
