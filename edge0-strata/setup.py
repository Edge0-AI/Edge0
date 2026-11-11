#!/usr/bin/env python3
"""Strata one-click setup and start (Windows and Linux, NVIDIA or AMD GPUs).

    START-HERE.bat  (Windows)   /   ./setup.sh  (Linux)      - they install Python if needed and run this file

The first time it asks four questions - which model (the original Qwen3.8-Flash-Next or the Swift 1.5 fine-tune),
which size, how much context, and whether the model should also read images - then installs everything and starts the model on http://127.0.0.1:8080 (OpenAI- and Anthropic-compatible
API; a small page there shows that it runs). Every later start skips straight to running the model: nothing that
is already downloaded, installed or prepared is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA or AMD GPU and driver, RAM, CPU, free disk space
  2. asks the questions
  3. installs the Python packages it needs into .venv (numpy, jinja2, ..., and NVIDIA's CUDA libraries)
  4. gets the Strata engine: a ready-made build for RTX 20/30/40/50 cards (no compiler needed); if none fits your PC,
     it installs the missing build tools (asks first) and compiles the engine for your GPU.  AMD (--backend hip, chosen by
     itself on a PC with no usable NVIDIA card): the ready-made HIP engine on Windows, compiled here on Linux
  5. downloads the model from Hugging Face (resumable), and the vision encoder if you want images
  6. prepares the model for Strata and fetches the MTP draft layer (~5 GB, from the original Qwen checkpoint)
  7. writes run-<model>.bat / run-<model>.sh and starts the model

Options: --family qwen|swift, --model Q2_0|IQ2_XS|IQ3_XXS|IQ3_S, --context 32768, --rope-scaling none|linear|yarn
(--rope-scale F; past the trained 262144 the setup adds yarn and the factor is the final context over 262144,
at least 1 - an explicit --rope-scaling none is refused for such a context), --vision yes|no|gpu|cpu, --port
8080, --yes (recommended
answers, no questions), --setup (install another model / change settings instead of starting), --no-start,
--host 0.0.0.0 --api-key KEY (reach it over IPv4; use --host :: for IPv6), --experimental-speed-projection on|off
(EXPERIMENTAL, off by default),
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --build (compile instead of the ready-made
engine), --check (only check this PC), --resident-budget-gib N (UD-Q4_K_XL's or UD-IQ4_XS's experts in RAM),
--kv-streaming on|off|auto.

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - a longer context, more GPUs, a bigger RAM budget, a size it thinks will not fit - is kept, with
what it risks.  With --yes, an explicit flag (--model, --gpus, ...) is the consent to a risk setup would otherwise
stop at; --yes alone is not.
"""
from __future__ import annotations

import argparse
import calendar
import ctypes
import glob
import hashlib
import http.client
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from strata_control import acquisition
from strata_control import tls

tls.install()      # #3722: HTTPS checks the server against the operating system's certificates (what Chrome and curl.exe do)
from strata_control import atomic
from strata_control import draft_layer
from strata_control import dsv41_context
from strata_control import engine_component
from strata_control.discovery import in_trash
from strata_control import user_args

ROOT = Path(__file__).resolve().parent
# The desktop app runs this installer under Strata Control (its bundled Python, STRATA_CONTROL_INSTALL=1). The source
# (tools/, data/, serve/) stays in ROOT, inside the app; everything the install writes (run configs, logs, third_party,
# the data folder beside it) goes to home(), a per-user folder (STRATA_INSTALL_DIR). In that mode it installs no Python
# package and no engine and compiles nothing: it downloads the model files, prepares them with the bundled tools and
# writes a run config whose engine is "component:strata-engine". Outside that mode home() is ROOT and nothing changes.
CONTROL = os.environ.get("STRATA_CONTROL_INSTALL") == "1"
_CONTROL_HOME = Path(os.environ["STRATA_INSTALL_DIR"]).resolve() if CONTROL and os.environ.get("STRATA_INSTALL_DIR") else None


def home() -> Path:
    """Where an install writes (see above). A function, not a constant, so ROOT stays the one thing a caller can point elsewhere."""
    return _CONTROL_HOME or ROOT


WIN = os.name == "nt"
# #214: every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when this was pinned), so a checkout installs the same files on any
# day.  Every file has a fixed size and SHA-256 too (REPO_PINS); a revision the repository no longer has stops the
# download (download()), unless a tester sets STRATA_ALLOW_UNPINNED=1.
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d",   # 2026-10-07 (adds IQ3_S)
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",                  # 2026-10-12
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": "1ada08422e404aca51ab9df665f089f34c419c6b",           # 2026-10-13
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF": "8a6ec9cf118e785a362ea94ee123c61c70553be6",             # 2026-10-30 (adds UD-IQ1_S / UD-IQ1_M; every file pinned from the 2026-10-20 commit has the same size and SHA-256 here)
    "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF": "cb02422c900df489cea5833e8a92171224059b3c",    # 2026-10-24 (ISTA-DASLab's final 1.75 / 2.00 / 2.25 bpw sets and Engram)
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


def hf_unpinned(url: str) -> str:
    """The same file at the repository's current revision (main)."""
    return re.sub(r"^(https?://[^/]+/.+?/resolve/)[0-9a-f]{40}/", r"\1main/", url, count=1)


_unpinned_said = []


def allow_unpinned() -> bool:
    """STRATA_ALLOW_UNPINNED=1: a tester's opt-out.  A file whose pinned version is gone from Hugging Face is then taken
    from the repository's current files, without the SHA-256 check (nothing pins them).  Off by default, announced when set."""
    on = (os.environ.get("STRATA_ALLOW_UNPINNED") or "").strip() == "1"
    if on and not _unpinned_said:
        _unpinned_said.append(1)
        warn("STRATA_ALLOW_UNPINNED=1: a model file whose pinned version is gone from Hugging Face is taken from its "
             "current files instead, and is NOT checked against a SHA-256. Unset it to get the checked files only")
    return on


def file_pin(url: str):
    """(bytes, sha256) pinned for the Hugging Face file `url` names (REPO_PINS), else None."""
    m = HF_FILE.match(url)
    if m is None:
        return None
    return REPO_PINS.get(m.group("repo"), {}).get(m.group("path").rsplit("/", 1)[-1])


# ModelScope (www.modelscope.cn) hosts every repository above under the same name, with the same file paths and
# sizes, and is reachable from mainland China where huggingface.co often is not.  It serves a repository's current
# files (no pinned revision), so a file from it is checked against the SHA-256 ModelScope publishes for it, and the
# MTP tensors against the pinned checkpoint's own hashes (tools/mtp_fetch.py).
# --source / STRATA_SOURCE: huggingface (what auto means), or modelscope: only when asked for (a different host
# serving current files, not our pinned revision: setup never switches to it by itself, it recommends it, #908).
MS_DEFAULT = "https://www.modelscope.cn"
SOURCES = ("auto", "modelscope", "huggingface")
HF_FILE = re.compile(r"^https?://[^/]+/(?P<repo>[^/]+/[^/]+)/resolve/[^/]+/(?P<path>.+)$")
_sources = {}                                      # model_source()'s answer per (STRATA_SOURCE, HF_ENDPOINT, host)
_ms_files = {}


def ms_endpoint() -> str:
    return (os.environ.get("MODELSCOPE_ENDPOINT") or "").strip().rstrip("/") or MS_DEFAULT


def reachable(url: str, timeout: float = 5.0) -> bool:
    """Whether `url` answers a HEAD request with success (2xx, after redirects) within `timeout` seconds."""
    try:
        _urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"}),
                               timeout=timeout).close()
        return True
    except OSError:                                    # HTTPError too: an error page is not the file
        return False


def model_source() -> str:
    """"modelscope" or "huggingface".  ModelScope only when --source modelscope / STRATA_SOURCE=modelscope asks for it;
    everything else (auto, a HF_ENDPOINT mirror chosen on purpose, #495) is Hugging Face.  Setup does not switch hosts
    by itself (#908): ModelScope serves a repository's current files, not our pinned revision."""
    want = (os.environ.get("STRATA_SOURCE") or "auto").strip().lower()
    key = (want, os.environ.get("HF_ENDPOINT") or "", ms_endpoint())
    if key not in _sources:
        _sources[key] = "modelscope" if want in ("ms", "modelscope") else "huggingface"
    return _sources[key]


def source_hint(url: str) -> str:
    """The recommendation that goes with a failed download from Hugging Face: ModelScope, by an explicit choice."""
    if "huggingface" in url or "hf-mirror" in url:
        return ("; if huggingface.co does not reach you (mainland China), the same files are on ModelScope: run setup "
                "with --source modelscope (docs/INSTALL.md)")
    return ""


def ms_file(url: str):
    """(repo, path) when `url` is a Hugging Face file of a repository setup knows (HF_REVISIONS), else None."""
    m = HF_FILE.match(url)
    if m is None or m.group("repo") not in HF_REVISIONS:
        return None
    return m.group("repo"), m.group("path")


def ms_url(repo: str, path: str) -> str:
    return f"{ms_endpoint()}/models/{repo}/resolve/master/{path}"


def ms_meta(repo: str, path: str):
    """(size, sha256) ModelScope publishes for a file, or None when it cannot be asked."""
    if repo not in _ms_files:
        try:
            api = f"{ms_endpoint()}/api/v1/models/{repo}/repo/files?Recursive=true"
            with _urlopen(urllib.request.Request(api, headers={"User-Agent": "strata-setup"}),
                                        timeout=60) as r:
                files = json.loads(r.read())["Data"]["Files"]
            _ms_files[repo] = {f["Path"]: (int(f.get("Size") or 0), (f.get("Sha256") or "").lower()) for f in files}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return _ms_files[repo].get(path)


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"

# The ready-made engine: <PREBUILT_URL><asset>, a zip with strata(.exe), strata-vision(.exe) and BUILD.json, built
# by tools/make_release.py.  Set this to the GitHub release download folder when publishing, e.g.
# "https://github.com/<you>/Strata/releases/latest/download/" (or pass --prebuilt / set STRATA_PREBUILT_URL).
# With the default, the release of this checkout's own version (PREBUILT_TAG_URL, CMakeLists.txt's version) is
# tried first and the latest release is the fallback (#214): an older checkout keeps the engine it shipped with.
PREBUILT_URL = "https://github.com/Niko1221/Strata/releases/latest/download/"
# The repository the release assets and their SHA-256 come from; `engine_digest` reads the API here even
# when --prebuilt points the download somewhere else, because the hash is only worth having if it comes
# from somewhere the download does not.
REPO = "Niko1221/Strata"
PREBUILT_TAG_URL = "https://github.com/Niko1221/Strata/releases/download/v{version}/"
PREBUILT_ASSET = "strata-windows-x64.zip" if WIN else "strata-linux-x64.zip"
# the CUDA libraries the ready-made engine loads (the same CUDA 13.0 it is built with), from NVIDIA's pip packages
CUDA_WHEELS = ["nvidia-cublas==13.0.2.14", "nvidia-cuda-runtime==13.0.96"]
MIN_DRIVER = 580                       # CUDA 13.0
# Older NVIDIA GPUs (experimental): CUDA 13 dropped Pascal (sm_60/61) and Volta (sm_70), so a model whose GPUs include
# one runs a second engine, built with CUDA 12.9 (-DSTRATA_EXPERIMENTAL_SM60=ON) and kept in its own folder: the
# ready-made one is CUDA12_ASSET (Windows; on Linux it is compiled here with a CUDA 12.x toolkit).  One engine runs per
# model, so the choice is per model config, by its oldest GPU; --cuda 12|13 overrides it (docs/OLDER_GPUS.md).
CUDA13_MIN_ARCH = 75                   # the oldest compute capability CUDA 13 compiles for (sm_75, RTX 20)
CUDA12_ASSET = "strata-windows-x64-cuda12.zip" if WIN else "strata-linux-x64-cuda12.zip"
CUDA12_WHEELS = ["nvidia-cublas-cu12==12.9.1.4", "nvidia-cuda-runtime-cu12==12.9.79"]
# The wheels match the 12.9.1 toolkit the CUDA 12 engine is built with (cuBLAS 12.9.1.4, runtime 12.9.79).
# CUDA12_COMPAT_DRIVER is NVIDIA's minor-version-compatibility line (Linux 525.60.13, Windows 527.41): under it no CUDA 12 program starts, so
# setup stops there.  It is NOT what setup promises: the ready-made packages are built with CUDA 12.9 and carry their own `min_driver`
# (575.51 / 576.02), which `strata update` enforces.  cuda12_min_driver() reads that from the package or the feed (#4247 #4188).
CUDA12_COMPAT_DRIVER = 528 if WIN else 525
ENGINE12_DIR = "engine-cuda12"
MIN_ENGINE = (0, 1, 46)                # versions compare all numbers; v0.1.46: the 0.2 release engine (UPDATE.bat and setup fetch it from the signed feed, #3469 #3437); v0.1.45: Qwen3.6 parallel requests and pictures, fixes #2346 #2372 #2396; v0.1.44.1: CPUs without AVX2 start again (#2324: an illegal instruction before main in the NVFP4 CPU rows), the CUDA 12.0/12.1 toolkits build (#2322), a crashed engine start is named instead of retried as a GPU runtime start, Mac M2 (#2323); v0.1.44: UD-Q4/Q5/Q6_K_XL and Q8_0 Unsloth files in setup (K-quant experts, MMQ prompt library strata_kq), NVFP4 packs, Qwen3.6-35B-A3B, Intel Arc A-series kernels, the pipelined-decode and latency work; v0.1.43.1: the adaptive overlap no longer collapses decode on pinned-host desktops (#1964 #1975), Windows PLE mmap stall (#1984), pinned prompt uploads (#1996), the I/O watchdog allowance; v0.1.43: adaptive-tier copies overlap on CUDA, P100 (sm_60) kernel tables, RDNA4 (gfx1200/gfx1201) defaults (route tail skip, measured PCIe share, 16K prompt chunks, fused IQ3 prompt experts); v0.1.42.1: the GGUF-in-place file tier on Windows unmaps every shard, dual-socket starts judge each NUMA node and STRATA_DROP_CACHE_ON_EXIT, tools/strata_bench.py; v0.1.42: STRATA_ROUTE_TAIL_SKIP=7 default on CUDA (+13-16% decode), the AMD 2-GPU checkpoint fix (H13), Phase 2 fixes and opt-ins; v0.1.41.1 (folded into 0.1.42): the #1576 card order, #1616 stage size, #1595 CPU share, #1630/#1607 Windows HIP chunk cap and commit-limit warning, #1603 parallel-request bookkeeping; v0.1.41: CPU prefill share on by default (NVIDIA, chunks < 1024), --batch-groups auto, the file tier read in batches on Windows, Pascal decode fix, per-arch tables; v0.1.40.3: the #1357 MTP router guard, the #1376 Windows HIP cache floor, #461 runtime DLLs, Intel A750 first-request fix; v0.1.40.2: F4 verify windows, the Linux file tier (#1194), the stager wait (#1057), #1264/#1201/#1139 fixes, opt-in CPU share (#1282), Intel Arc; v0.1.40: --resident-experts on a layer split with the split+resident variant (#848), --kv k8v4 with KV streaming (#711); v0.1.39: the #577 file-tier regression fixed, the OpenAI Responses API (#451, Codex), a reply stuck on one token ended (#606), the head before the arena (#620), effort_position (#458), --vram-reserve hot resize opt-in (#533), PR batch; v0.1.38: prompts faster (one gather per expert group #372, the first chunk's PLE rows beside layer 0 #374, DeltaNet three heads per thread #413), --kv q4_0 prompts on tensor cores (#452), Q5_0 experts on the GPU (#473), IQ4_XS on AVX-2 (#415), unbuffered expert loading on Windows (#357 #362), --peer-device (#531), a 6 GB card starts (#496), PR batch; v0.1.37: a silent engine is restarted (#481), Windows AMD counts the desktop's VRAM (#380 #377 #497), a steadier PCIe probe (#485), fixes #496 #495 #498 #505 #493; v0.1.36: a cancelled prompt logged as read so far (#471), the draft-head hint (#474), UPDATE.bat (#475), --expert-profile-save (#477); v0.1.35: Windows AMD uses its bundled HIP runtime (#468 #461), the low-RAM resident mode on Windows 32 GB (#467), fixes #460 #459 #446 #447 #457 #448 #444; v0.1.34: AMD on Windows (a ready-made HIP engine), an MCP server for AI assistants (tools/strata_mcp.py), a shorter README; v0.1.33: a portable image encoder again (#411 #412), setup recommends instead of forcing (#406 #403 #364 #384), fixes #352 #365 #369 #371 #375 #393 #408 #414; v0.1.32: split prompts faster (#340), AMD router +12%, Unsloth Q4 in setup, faster Q4 prompts, #326/#327/#342/#344 fixes, PR batch; v0.1.31: Unsloth UD-Q4_K_XL (experimental), GGUF-in-place low-RAM mode, Windows GGUF load 2x, server race + tokenizer fixes, AMD intrinsics; v0.1.30: short prompts faster (streaming from 1024 tokens), resident low-RAM variant, multi-GPU session carve, RDNA4; v0.1.29: sampled answers faster (split top-k), #154 correctness fixes; v0.1.28: the expert cache reserves the draft head, a cancelled request no longer fails the next; v0.1.27: RTX 20 (sm_75) in the ready-made engine, the HIP build without CUDA headers; v0.1.26: the draft layer's prompt pass in batches; v0.1.25: faster prompts (grouping off the copy engine, fused hyper-connection kernels), AMD HIP backend, --kv k8v4; v0.1.24: long prompts faster (QSA select on tensor cores); v0.1.23: image requests honor sampling, 8 GB cards start, batched verify window; v0.1.22: faster prompts (tensor-core attention), multi-GPU across images/steering/KV streaming; v0.1.21: multi-GPU layer split (--gpus); v0.1.20: system-prompt checkpoint, PCIe probe, hit rate; v0.1.19: penalties
# KV bytes per context token and attention layer: 8-bit 1056, rotated 4-bit 576, hybrid K8V4 (8-bit K, 4-bit V) 816
KV_CELL_BYTES = {"q4_0": 576, "k8v4": 816}
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (#214)
BUILD_PACKAGES = ["cmake", "ninja"]
BUILD_REQUIREMENTS = ROOT / "requirements-build.txt"

# NVFP4 experts (MODEL-UPDATE Phase 4; the engine path is ported from sergqwer/strata-nvfp4, MIT).  Ready-made Strata
# packs on Hugging Face by the fork's author: nothing is converted here.  Every file is pinned by its upload's commit,
# size and SHA-256 (the fork's setup.py NVFP4_REPOS, checked against the Hub's LFS pointers 2026-10-12).  The FP8 n-gram
# table and the BF16 token embedding are the same files in both repositories (a second install links them).  The
# MTP draft layer is setup's own, as for every other model, so the repositories' mtp/ folder is not downloaded.
# role: pack = the experts pack folder (index.txt, experts.bin, ...), dense = the dense GGUF (--native and
# --native-dense-gguf), ple = the FP8 n-gram table (--ple-gguf), embd = the BF16 token embedding (--embd-gguf)
NVFP4_GPTQ_REPO = "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata"     # Qwen's BF16 experts by GPTQ (the fork)
NVFP4_NVIDIA_REPO = "Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata"  # nvidia/Qwen3.8-Flash-Next-NVFP4, converted
_NVFP4_TOKENIZER = {
    "pack/tokenizer/chat_template.jinja": ("pack", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "pack/tokenizer/merges.txt": ("pack", 3600844, "080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9"),
    "pack/tokenizer/token_type.json": ("pack", 744960, "5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b"),
    "pack/tokenizer/tokenizer.json": ("pack", 554, "87be2ac47d8bc7393b7435df6ccd276740019a393941a2585d5edce59f450c54"),
    "pack/tokenizer/vocab.json": ("pack", 5737005, "4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4"),
}
_NVFP4_SHARED = {
    "ple-fp8.gguf": ("ple", 51200246176, "37f89b8ce5a3425932763257c40229ec02723ad2ab14b4bbfc26a486f4725c9c"),
    "token-embd-bf16.gguf": ("embd", 1271398656, "386203af9070c7b43acb24ded8922876721d97bbad1a709b25cb49d694a7f51d"),
}
NVFP4_REPOS = {
    NVFP4_GPTQ_REPO: {"revision": "640a1db97446ddd5e89d2b70ecd0297987192c93", "files": {
        "pack/dense.bin": ("pack", 1538035200, "974df6dcefcae85e3fd2c19fcc12eeaacaadde8261ab732d5af5af6ce2e25a7a"),
        "pack/experts.bin": ("pack", 67948118016, "325e5e0d85c262a8d8278daf73b20696ec877d9a3ff584e57391fa69ad10bc64"),
        "pack/index.txt": ("pack", 93663, "f62dc71e1c74a72794db54d3917bc71131cc6eae6f3dcdc3fdada48ff70f93ab"),
        "pack/native_experts.txt": ("pack", 1599, "615f655fb28bf60b243664860b135710bf4177b111c6be89073e780a35de1018"),
        **_NVFP4_TOKENIZER,
        "qwen-nvfp4-gptq-dense.gguf": ("dense", 5991569472, "abdf74336b53d8f7ca3d99947f1eac8f19b8a544966750127cb61e561b127e3b"),
        **_NVFP4_SHARED}},
    NVFP4_NVIDIA_REPO: {"revision": "cca8f7fee5ce0af932097340a75e06df25c4e57d", "files": {
        "pack/dense.bin": ("pack", 1538625024, "1f1c2422069de292673ae6cee7d69f7090732dec69f3f125612a0de456279714"),
        "pack/experts.bin": ("pack", 67948118016, "bb42080bb7585505b64d091d5a734c05376bbd6171406fa9688a46fbc261b957"),
        "pack/index.txt": ("pack", 117490, "17793d1c5b187a73d3c1dc1677af4d4c9b23845394dd7e88bfa53396b41b77e3"),
        "pack/native_experts.txt": ("pack", 1590, "d7f8b4372bb122f3d711030e5d522bff1c8746ac7d133a59a2cc7277283e59b6"),
        **_NVFP4_TOKENIZER,
        "qwen-nvidia-nvfp4-dense.gguf": ("dense", 5992176896, "0f795ca0cbb908e0cf2b894b30103e21455b5a05346a7ef79da208540cfd2be3"),
        **_NVFP4_SHARED}},
}
NVFP4_FP4LOCAL_REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"   # NVIDIA's QAD checkpoint (ModelOpt safetensors), prepared here
# Not a ready-made pack: the safetensors (95 GiB) are downloaded as NVIDIA ships them and tools/nvfp4_local_pack.py prepare
# turns them into the pack, the dense GGUF, the n-gram table and the draft layer on this PC (nothing of NVIDIA's weights is
# redistributed).  Role "src" = a file of the checkpoint; the prepared files are in FP4LOCAL_PREPARED.
NVFP4_REPOS[NVFP4_FP4LOCAL_REPO] = {"revision": "b9ce18a6887e18c089fdea8509a33e63429a3300", "files": {
        "config.json": ("src", 75217, "b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28"),
        "generation_config.json": ("src", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
        "chat_template.jinja": ("src", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
        "merges.txt": ("src", 3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
        "tokenizer.json": ("src", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
        "tokenizer_config.json": ("src", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
        "vocab.json": ("src", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
        "model.safetensors.index.json": ("src", 47280974, "5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1"),
        "tensors/model-00001-of-00021.safetensors": ("src", 5000643504, "ebf3648c5f4cd1f24e79789e7803498dc4e74cb52515cc515b2d22cf8161953b"),
        "tensors/model-00002-of-00021.safetensors": ("src", 5002792096, "79a643fc5c024089cecd32ca27dc43e671cbec70ff3628b728909b7656ac43ec"),
        "tensors/model-00003-of-00021.safetensors": ("src", 5002927400, "c45f942203db4c7f5a1071700b93b908495cc790271acddfff39091c2b953923"),
        "tensors/model-00004-of-00021.safetensors": ("src", 5003194672, "071ab9cf963ca0f1a6a194b4cec485f5de18ef9804da915b2e6abd0eb0c707af"),
        "tensors/model-00005-of-00021.safetensors": ("src", 5003151448, "f80d17870eb1d36917257494770984953476583c57737f61fdbd341c957b9715"),
        "tensors/model-00006-of-00021.safetensors": ("src", 5002892148, "3e9183d4ebb267fa92429184d5132d302cb1a98fd049149e5d01dccd4ca65795"),
        "tensors/model-00007-of-00021.safetensors": ("src", 5002907708, "aa1ad959c4d62e0bc4600fd0b57cc2939b463e9fb53dc01020e188552b52ce5a"),
        "tensors/model-00008-of-00021.safetensors": ("src", 5003103756, "0b18844027be74b84b240f23bf3a744bc29b35de0bd384f41c40cd85e6339326"),
        "tensors/model-00009-of-00021.safetensors": ("src", 5002976316, "2d0b6c66790c17f8b1fdcf442f541778c8f0a381aca26c11b9d7cfcf7c96f2f0"),
        "tensors/model-00010-of-00021.safetensors": ("src", 5002834136, "3ed9bd73745aebf6dc136f437b67d6b9fc3d38d25ad84300b399164bc87a72b7"),
        "tensors/model-00011-of-00021.safetensors": ("src", 5002685828, "4936ff28c6cc8b66653ddb9cae28bc284569a1a2f7116819a017dc1a0e517ed8"),
        "tensors/model-00012-of-00021.safetensors": ("src", 5002912920, "dd7df5a6ff73e9960412243f744eab9ec35f4be40da5bb158dc4ba8796b33742"),
        "tensors/model-00013-of-00021.safetensors": ("src", 5003237060, "28a455c8ea4ebaa847fb4cbed2bc1de9eff0b501dd3f848a3a3cfc38778e8500"),
        "tensors/model-00014-of-00021.safetensors": ("src", 4845006312, "6941676e685abf9817d9c48a17fdeda20b2b72d16217555252aa4c602b7f98b8"),
        "tensors/model-00015-of-00021.safetensors": ("src", 4860574920, "2a1129b5acfc24a45bf402aa63dcef08b531aedfdaf5233c65fd3d12dc303fdd"),
        "tensors/model-00016-of-00021.safetensors": ("src", 4950031028, "635e97fbad78fa988647d407e2b58e55d551ef94d798c9b6fb5550de25e86429"),
        "tensors/model-00017-of-00021.safetensors": ("src", 4950030896, "bf60d564d89a805b67f8be5f2847a1b1567a70dbc23d19754395c127568474ad"),
        "tensors/model-00018-of-00021.safetensors": ("src", 4950030896, "c5bde22f9364e82eb25973456795592fdcb4b158bc2095898979a2566bf4c562"),
        "tensors/model-00019-of-00021.safetensors": ("src", 4950030904, "da3dc3b4f428f5b9dd53362691043d0242f48d03adca9b7b553bd9f3a1e96350"),
        "tensors/model-00020-of-00021.safetensors": ("src", 4950030904, "4978bb6b8480cfc02d3c8bc84ba86202889f5f10d090bb17c21fee663cea7683"),
        "tensors/model-00021-of-00021.safetensors": ("src", 2700016864, "48203838f2c76921bddbf1cc51d20a23858c128821a21add55fa1778e5e25283"),
}}
FP4LOCAL_PREPARED = "prepared"                       # <models>/nvidia--Qwen3.8-Flash-Next-FP4-Local/prepared/

# the first release engine with NVFP4 experts (a source build of this checkout has them whatever its number says)
NVFP4_ENGINE = (0, 1, 44)
# the first engine that reads NVIDIA's NVFP4 n-gram table (strata.ple.format = nvfp4x16); a source build of this checkout has it
FP4LOCAL_ENGINE = (0, 1, 45)

# agentionai's Gyro files (Qwen3.8-Flash-Next with trellis-coded, Hadamard-rotated experts, TQ2_T / TQK6 / TQK7): one
# GGUF per size, the n-gram table inside it (hence the 58 / 92 GB downloads for 27.6 / 35 GB of memory).  The engine
# reads them from 0.1.45.1 on: the experts are native_experts.txt v5 (Hadamard-folded, hadamard.txt), which the engine reads since the
# APR port (commit f47b0385f, 2026-10-02, first in the 0.2.0 line).  The released 0.1.45 engine reads only up to v4 and refuses a
# Gyro pack, so 0.1.45 is NOT enough: 0.1.45.1 is the first engine package built after the reader was merged (checked: the
# "native_experts.txt v5 declares Hadamard-folded experts" message is in every 0.1.45.1+ strata binary and in no 0.1.45 release one).
# (ported from agentionai/Strata rc2, MIT); the sizes and SHA-256 are the Hub's LFS pointers
# of the revision pinned in HF_REVISIONS.  The model's license is Qwen's.
GYRO_ENGINE = (0, 1, 45, 1)
GYRO_FILES = {
    "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf":
        (58487015808, "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87"),
    "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf":
        (91965774272, "7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f"),
}

# ISTA-DASLab's Qwen3.8-Flash-Next shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and the picture reader at the revision pinned in HF_REVISIONS: name -> (bytes, sha256),
# the Hub's LFS pointers (tools/hf_pin_fetch.py).  The n-gram table (shard 2) is the same file for all four sizes.
ISTA_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
# Swift 1.5's shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and its picture reader, same source and form.
SWIFT_FILES = {
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39788473344, "ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28363693824, "e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (44900816736, "46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (38842920384, "70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (39785790560, "3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (36180282560, "b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (39799117984, "79e2a3873f1b05df835a1a191b20f2e74b9e12df949cd89861d17d98b6ef9678"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (26750834816, "ef3bb04fb4e14f04624abd8acf25faf1603594a034050b048df85ec152ad5ae0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf":
        (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
}
# The Coder's IQ1_M shards and its picture reader (the original's file), same source and form.
CODER_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf":
        (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}

MODELS = {
    # the original model only for now: Swift 1.5's Q2_0 files split one layer's experts across the two shards, which
    # the pack tool (tools/iq_pack.py) cannot prepare yet (#171)
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # matches the full BF16 model on the published benchmarks; Swift 1.5 got an IQ3_S tier of its own (#1651)
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # #621: Unsloth's UD-IQ4_XS - IQ3_S gate/up experts with IQ4_NL (43 layers) or Q8_0 (5) downs, the dense side as
    # UD-Q4_K_XL's; three shards.  A regular choice from 0.1.39 (no longer experimental).  Its 59.5 GB of experts: a
    # RAM budget of them, like UD-Q4_K_XL, but far fewer read from the SSD on a 64 GB PC and none from ~80 GB of RAM.
    # Images: the vision path has no restriction for this pack (the same base model and image encoder), so setup asks
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "engine": (0, 1, 38),
                  "vision": True},
    # Unsloth's 4-bit file (docs/UNSLOTH_Q4.md); experimental until MODEL-UPDATE Phase 2.  Its 77 GB of experts stay in
    # RAM from ~94 GB of RAM (unsloth_budget); on a 64 GB PC the engine keeps a RAM budget of them
    # (--resident-budget-gib, chosen below) and reads the rest from the GGUF on the SSD.  It runs on AMD too (R9700:
    # the quality yardstick and the decode/prompt numbers in docs/QUANTS.md), so it is no longer NVIDIA-only (#429).
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic): K-quant experts, between UD-IQ4_XS and UD-Q5_K_XL in quality",
                   "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",), "budget": True,
                   # #967 / MODEL-UPDATE: images run end to end with this file (CUDA: RTX A4000, GPU encoder; HIP:
                   # R9700, CPU encoder; the test picture's text and shapes as the IQ3_XXS pack answers them)
                   "vision": True},
    # MODEL-UPDATE: Unsloth's larger files (docs/QUANTS.md).  The same GGUF-in-place mode: every expert in RAM where it
    # fits with UNSLOTH_RESIDENT_BESIDE_GB more (unsloth_budget), else a RAM budget of them and the rest read from the
    # SSD while it answers - so "ram_gb" is the all-in-RAM need, and below it setup asks first (default no).  Their
    # shards 2 (the head) and 3 (the 54 GB Q8_0 PLE table) are the same bytes in all three files.  UD-Q5_K_XL and
    # UD-Q6_K_XL need the Q6_K expert kernels every engine has from MODEL-UPDATE Phase 1 on (Q5: layer 2's gate/up).
    "UD-Q5_K_XL": {"about": "5-bit (Unsloth Dynamic): Q5_K experts with Q8_0 downs, close to Q8_0 in quality",
                   "download_gb": 158.3, "ram_gb": 108, "arena_gb": 98.4, "families": ("unsloth",), "budget": True,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 44),
                   "vision": True},
    "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic): Q6_K experts with Q8_0 downs",
                   "download_gb": 169.2, "ram_gb": 118, "arena_gb": 109.2, "families": ("unsloth",), "budget": True,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 44),
                   "vision": True},
    "Q8_0": {"about": "8-bit, the reference quality (every expert Q8_0); for a 192 GB PC",
             "download_gb": 188.2, "ram_gb": 136, "arena_gb": 128.3, "families": ("unsloth",), "budget": True,
             "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 38),
             "vision": True},
    # MODEL-UPDATE Phase 4: NVFP4 experts, ready-made packs (NVFP4_REPOS).  63.3 GiB (68.0 GB) of experts in the
    # pack's experts.bin; the low-RAM mode reads them from there on a PC whose RAM does not hold them.  NVIDIA only
    # (the HIP and SYCL engines have no NVFP4 kernels).  Images: both packs keep Qwen's vision tower unquantized, so the
    # original's image encoder (ISTA's mmproj, the same file as for every Flash-Next model) reads pictures for them;
    # checked end to end on an RTX 5070 and an RTX 3060 against IQ3_XXS's answers on the same picture.
    # Gyro (agentionai): "ram_gb" is the memory the weights need in total, VRAM and RAM together ("mem_total"): Gyro-S
    # 27.6 GB, Gyro-M 35 GB (the rest of the download is the n-gram table, read from the SSD).  Neither is anyone's
    # default: setup and the MCP server mention them where the other sizes do not fit.  Images: the same mmproj as every
    # Flash-Next model; strata-vision reads only the vocabulary and header of the Gyro GGUF (tools/vision/gguf_meta.h),
    # whose TQK6 / TQK7 tensors llama.cpp does not know.
    "Gyro-S": {"about": "the smallest option, for PCs with little memory (27.6 GB of VRAM + RAM together); lower quality "
                        "than the 2-3-bit sizes, and the 2-bit Q2_0 decodes faster",
               "download_gb": 58.5, "ram_gb": 27.6, "arena_gb": 25.8, "families": ("gyro",), "mem_total": True,
               "file": "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", "shards": 1, "engine": GYRO_ENGINE, "vision": True},
    "Gyro-M": {"about": "about IQ3_XXS quality with about 20% less GPU memory (35 GB of VRAM + RAM together); decodes "
                        "about 16-21% slower than Q2_0 on 12 GB cards",
               "download_gb": 92.0, "ram_gb": 35.0, "arena_gb": 32.1, "families": ("gyro",), "mem_total": True,
               "file": "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", "shards": 1, "engine": GYRO_ENGINE, "vision": True},
    "NVFP4": {"about": "best quality of the NVFP4 versions (GPTQ, sergqwer's strata-nvfp4 pack); RTX 50 has FP4 tensor cores, "
                       "other NVIDIA cards run it at 8-bit precision", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
              "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_GPTQ_REPO,
              "engine": NVFP4_ENGINE,
              "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's): https://huggingface.co/Qwen/Qwen3.8-Flash-Next"},
    "NVFP4-NVIDIA": {"about": "NVIDIA's own NVFP4 checkpoint (plain rounding), converted by sergqwer; the same size "
                              "and speed as NVFP4", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
                     "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_NVIDIA_REPO,
                     "engine": NVFP4_ENGINE,
                     "license": "NVIDIA Open Model License (Licensed by NVIDIA Corporation under the NVIDIA Open Model "
                                "License) with Qwen3.8-Flash-Next's Qwen Community License 1.0: "
                                "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4"},
    "NVFP4-QAD": {"about": "NVIDIA's FP4, trained for FP4 (NVIDIA's Qwen3.8-Flash-Next-FP4-Local) - slightly faster on short prompts; "
                           "a 102 GB download that this PC turns into a Strata pack once (about 20-40 minutes, 110 GB more disk); "
                           "RTX 50 cards have FP4 tensor cores, other NVIDIA cards run it at 8-bit precision. It reads pictures (the shared Qwen3.8-Flash-Next encoder)",
                  "download_gb": 102.2, "prepare_gb": 110.0, "ram_gb": 80, "arena_gb": 68.0,
                  "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_FP4LOCAL_REPO,
                  "prepare": "fp4-local", "engine": FP4LOCAL_ENGINE,
                  "license": "NVIDIA Open Model Agreement (Licensed by NVIDIA Corporation under the NVIDIA Open Model "
                             "Agreement) with Qwen3.8-Flash-Next's Qwen Community License 1.0: "
                             "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local"},
}
# The experimental Unsloth file's four shards at the pinned revision: name -> (bytes, sha256), checked after the
# download (setup trusts no other model file by name and size alone either: check_shards reads their directories).
UNSLOTH_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
}
# #621: UD-IQ4_XS's three shards at the same revision (sizes and SHA-256: the Hub's LFS pointers)
UNSLOTH_IQ4_XS_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
}
# MODEL-UPDATE: UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 at the same revision (sizes and SHA-256: the Hub's LFS pointers; shards
# 2 and 3 are byte-identical across the three)
UNSLOTH_BIG_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf":
        (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf":
        (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf":
        (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf":
        (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf":
        (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf":
        (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf":
        (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf":
        (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf":
        (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf":
        (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf":
        (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf":
        (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
}
UNSLOTH_ENGINE = (0, 1, 32)     # the first engine setup configures for UD-Q4_K_XL (0.1.31 ran it by hand)
UNSLOTH_RAM_LEFT_GB = 24        # RAM beside the budget: the OS, the engine, and the file cache the rest is read through
UNSLOTH_MIN_RAM_GB = 48         # the RAM-budget mode below which setup says an Unsloth file does not fit (UD-Q4_K_XL's floor)
# Contexts past 262144 (the model's trained length) extend it by rope scaling: for the context it will
# serve the setup resolves the method (yarn, or one question when interactive) and derives the factor
# from the final context (final / 262144, at least 1) itself (below), keeps an explicit
# --rope-scaling/--rope-scale, and refuses an explicit --rope-scaling none there - the stock angles past
# the trained range are out of spec.
# 204800 (200K) sits between 128K and 256K: it is inside the trained 262144, so it needs no rope scaling and
# costs ~2.8 GB of 8-bit KV with IQ3_S (vs ~3.6 GB at 256K) - a middle step for PCs that cannot hold 256K.
CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's UD-IQ4_XS (three shards, #621), UD-Q4_K_XL (four), UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 (six) of the original
    # model (docs/QUANTS.md): regular choices since f-quants; the shard count and "vision" are per model (MODELS)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's 4- to 8-bit quantizations",
                "about": "UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL or Q8_0 (94 to 188 GB downloads): every expert in "
                         "RAM from ~80 GB (UD-IQ4_XS) to ~136 GB (Q8_0) of RAM, else part of them read from the SSD",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "vision": True, "pack_args": ["--compat-bf16"],
                "sha256": {**UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS, **UNSLOTH_BIG_SHARDS}},
    # MODEL-UPDATE Phase 4: the original model with NVFP4 experts, as ready-made packs (NVFP4_REPOS): the files come
    # as the engine reads them (no GGUF to pack), each size has its own repository ("repo" in MODELS)
    # agentionai's Gyro quantizations (see GYRO_FILES): after the regular families, never the default
    "gyro": {"title": "Qwen3.8-Flash-Next (Gyro)", "by": "Gyro quantizations by agentionai",
             "about": "Gyro-S (27.6 GB of memory) and Gyro-M (35 GB): the smallest sizes, for PCs with little memory; "
                      "58 and 92 GB downloads (the files include the n-gram table)",
             "hf": hf("agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"), "file": "Qwen3.8-Flash-Next-{q}.gguf", "shards": 1,
             "tag": "gyro-", "name": "qwen3.8-flash-next-gyro", "vision": True, "pack_args": ["--compat-bf16"],
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
             "sha256": GYRO_FILES,
             "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; the Gyro files are agentionai's quantization): "
                        "https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"},
    "nvfp4": {"title": "Qwen3.8-Flash-Next (NVFP4)", "by": "NVFP4 packs by sergqwer (strata-nvfp4)",
              "about": "4-bit NVFP4 experts, NVIDIA cards; a 128 GB download, ~80 GB of RAM to keep every expert in RAM",
              "tag": "nvfp4-", "name": "qwen3.8-flash-next-nvfp4", "ready": True,
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
              "mmproj_sha256": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# Qwen3.6-35B-A3B (GGUF architecture qwen35moe), Unsloth's GGUFs with the MTP block: a family with an engine path of its
# own (src/qwen35; `strata` switches to it from the file's architecture).  No pack, no expert profile, no separate MTP
# draft layer: the engine reads the GGUF as it is, keeps every expert it can in VRAM and computes the rest on the CPU
# (the hybrid mode), and drafts with the file's own MTP block.  Not in FAMILIES (whose entries are Flash-Next files with
# packs): OWN_PATH_FAMILIES, offered in the same menu and installed by install_own_path.  NVIDIA (CUDA) and AMD (HIP)
# only: the Intel Arc (SYCL) engine refuses it.
QWEN36_FILES = {
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": (22853663008, "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"),
    "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf": (32611711264, "35fce994cd36104a7dc1bd8a4bdf13778145664c00fdef6773aebc9246e5019c"),
    "Qwen3.6-35B-A3B-Q8_0.gguf": (37801097504, "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131"),
    "mmproj-F16.gguf": (899283584, "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160"),
}
OWN_PATH_FAMILIES = {
    "qwen36": {"title": "Qwen3.6-35B-A3B", "by": "Qwen; GGUFs with the MTP layer by Unsloth", "arch": "qwen35moe",
               "about": "35B mixture of experts with 3B active (256 experts, top-8): a smaller cousin of "
                        "Qwen3.8-Flash-Next with the same tokenizer",
               "hf": hf("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"), "file": "Qwen3.6-35B-A3B-{q}.gguf", "tag": "qwen36-",
               "mmproj_hf": hf("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"), "mmproj": "mmproj-F16.gguf",
               "name": "qwen3.6-35b-a3b", "license": "Apache 2.0: https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
               # RAM: the engine keeps a copy of every expert in RAM when they do not all fit in VRAM (the hybrid mode),
               # so the file's size plus room for the OS and the rest (measured: UD-Q4_K_XL on a 46 GB PC with a 12 GB
               # card, 18.75 GiB of experts pinned)
               "models": {"UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), the smallest", "ram_gb": 32},
                          "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic), closer to the full model", "ram_gb": 44},
                          "Q8_0": {"about": "8-bit, the closest to the full model", "ram_gb": 52}},
               "sha256": QWEN36_FILES,
               # the engine's KV cache per context token: 10 attention layers + the MTP block, 2 KV heads x 256, K and
               # V in fp32 (strata-q36)
               "kv_bytes_per_token": 11 * 2 * 256 * 2 * 4,
               "mtp": 2,                     # drafts per step (greedy requests): measured best on the R9700
               # pictures: Unsloth's mmproj (the same projector family as Flash-Next's: qwen3vl_merger) through
               # strata-vision; a parallel slot's session: its KV cache (kv_bytes_per_token) and the recurrent state of
               # the 30 GDN layers (state_gb)
               "vision": True, "state_gb": 0.07}
}
PREVIEW_FAMILIES = OWN_PATH_FAMILIES             # the earlier name (tools and tests that import it)

# DeepSeek-V4.1-Flash (GGUF architecture deepseek41), Unsloth's UD-Q8_K_XL at the revision the port was checked against: a
# family with an engine of its own (strata-dsv41, src/dsv41, ported from Project Maya's maya-ds4, MIT) that is not a
# Flash-Next pack and not a single GGUF: nine shards (508.4 GB, the routed experts as MXFP4, the dense weights in BF16, and
# in shard 8 the 202.8 GB n-gram table that stays on the SSD), the engine's pack folder (tools/dsv41/pack_gguf.py: hash
# tables, tensor map, the tokenizer and the GGUF's own chat template, a few MB) beside them.  MULTI_FILE_FAMILIES, offered
# in the same menu and installed by install_dsv41.  Where it fits: strata_control/dsv41_fit.py (real numbers).  The DSpark
# drafter (dspark-...-Q8_0.gguf, 8.0 GB) and the picture reader (mmproj-BF16.gguf, 1.0 GB) are pinned here but not fetched: the
# engine does not use them yet.
DSV41_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
DSV41_SHARD = "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-{i:05d}-of-00009.gguf"
DSV41_FILES = {
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf": (5780589, "cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00002-of-00009.gguf": (50536121984, "9cb2047676bd20e448be2377b480080c97fb38fae117f4846e07de2c65bb8522"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00003-of-00009.gguf": (50536121984, "17d398dd327e7ff1991b382677814db75d970b074661757564b6f2ac3d0c8d95"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00004-of-00009.gguf": (50536122016, "7ea8da1143b9a6bf1424a361eb1b0d6f66c7ce79eac9711162f212cc4f39c319"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00005-of-00009.gguf": (50536122016, "1952da19a841176b46f9681c99ea57c674f8a8861e16d76fb35cec4befe039b2"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00006-of-00009.gguf": (50536122016, "d6711781d8effab79b410af9c40920bffcc0c28976e509bc1227ff0c6669c33a"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00007-of-00009.gguf": (36097230048, "2ebc8d198d03a14865e8a85df637285ceb5dd79b1ffc78424bd3c3e06df57823"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009.gguf": (202758032592, "12df293355efbd104b088a77e13ee22410fb1ec3f7ab3863947f09896e64bec7"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf": (16835109952, "df33722d1dcf5ffe0347ed8406f7acb3b18061f4ded2c13c2b0a184b8d41c9ea"),
    "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf": (7976914240, "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"),
    "mmproj-BF16.gguf": (972208288, "9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9"),
}
# The two OFFICIAL IST-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF at the revision above, a flat repository; Hub LFS sha256 = what the
# files hash to): IQ2_XXS is the 2.00 bpw set, IQ2_XS the 2.25 bpw set and IQ1_M the 1.75 bpw set, 4 shards each, plus ONE Engram table (110.6 GB, the same file for both sets:
# downloaded once into the shared folder) and the DSpark drafter and the 5.8 MB metadata shard of Unsloth's set that pack_gguf --engram-meta reads
# (both pinned in DSV41_FILES above).  This is the set a normal PC runs (64 GB RAM + a 12 GB card; docs/DEEPSEEK41.md); the UD-Q8_K_XL entry stays
# for the big machines.
DSV41_ISTA_REPO = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
DSV41_ISTA_ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
DSV41_ISTA_SHARD = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quant}-{i:05d}-of-00004.gguf"
DSV41_DSPARK = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
DSV41_ISTA_FILES = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf": (39358830464, "6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00004.gguf": (39176113248, "34d68c99fdae9948f9f0af0dafe308e805ce0a78d2c48641b24cc57242fdc0af"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00004.gguf": (39997334720, "a08468c83955e0c4d3a9eb3f7b86c45b7bbac55d99c4640039a296a8a4b327ea"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00004.gguf": (2188841824, "b05f26a92325aa6845cebcb3ec1f93d2e65ebf89bdc3f805695960d2f0681832"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf": (39535776992, "4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf": (39494617536, "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf": (39423838496, "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf": (36771591232, "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf": (39075714624, "d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf": (39388449504, "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf": (39246891616, "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf": (20527837312, "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059"),
    "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf": (110595290816, "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"),
}
# ISTA re-published the first shard of each set on 2026-10-26 with a larger chat template (images in tool results; their PATCHING.md, and what their
# patch_chat_template.py makes of a shard 1 already downloaded).  setup downloads the pinned revision (HF_REVISIONS), so DSV41_ISTA_FILES stays the older
# file; a newer shard 1 "already here" (--gguf-dir) is the same weights with the newer template and is accepted by its own size and SHA-256, never fetched.
DSV41_NEW_SHARD1 = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf": (39358841536, "a7be0d5a48e4d1c13c5535f9123d9131ab8964e5acf8a0c01d14970e4b7f4df9"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf": (39535788064, "d9b949d55e27242b8afc7c2928749b888372d4c39682f9a131670d1c2555155e"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf": (39075725728, "2dfe9711eac1f7814aa59154d11e8f27739f8b96875dc1e667eac9f8c756e1fe"),
}
# The files the first release of these sets was downloaded as (helcig/tmp-oct26-2, 2026-10-23: model-2.00bpw-0000N-of-00004.gguf, model-2.25bpw-...,
# engram-IQ4_NL.gguf, usually in a 2.00bpw/ or 2.25bpw/ folder): testers and our boxes have them.  They are the same weights: shards 2-4 are the
# same bytes as today's, and shard 1 and the Engram differ only in their GGUF header (metadata).  setup accepts them from "files already here"
# (size and SHA-256 checked against these pins) and never downloads them.  Not in REPO_PINS: nothing is fetched under these names.
DSV41_LEGACY_FILES = {
    "model-2.00bpw-00001-of-00004.gguf": (39075713728, "78ab9efc46a26647c3e94b322c99cd7e2af261dd28b38f2712e128562db1ef47"),
    "model-2.00bpw-00002-of-00004.gguf": (39388449504, "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70"),
    "model-2.00bpw-00003-of-00004.gguf": (39246891616, "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3"),
    "model-2.00bpw-00004-of-00004.gguf": (20527837312, "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059"),
    "model-2.25bpw-00001-of-00004.gguf": (39535776064, "ddd13aca16faa23b08f5196b5e0daddf90f04688a271f5889eb22fa7d7d65526"),
    "model-2.25bpw-00002-of-00004.gguf": (39494617536, "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e"),
    "model-2.25bpw-00003-of-00004.gguf": (39423838496, "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240"),
    "model-2.25bpw-00004-of-00004.gguf": (36771591232, "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf"),
    "engram-IQ4_NL.gguf": (110595290720, "1f17d8ae37e9df3bb45f7ef1b4fcd1706d7b759af0413c117bf8e92aed459281"),
}
# today's name -> the name it had in the first release
DSV41_LEGACY_NAMES = {
    **{DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=i): "model-2.00bpw-%05d-of-00004.gguf" % i for i in range(1, 5)},
    **{DSV41_ISTA_SHARD.format(quant="IQ2_XS", i=i): "model-2.25bpw-%05d-of-00004.gguf" % i for i in range(1, 5)},
    DSV41_ISTA_ENGRAM: "engram-IQ4_NL.gguf",
}
# The SIX-shard layout (0.2.8.1).  On 2026-10-27 (commit a419d74c, llama.cpp b11870 and newer) ISTA-DASLab re-split the three sets: five shards per set
# (shard 1 is 5.8 MB of metadata and the tokenizer, no tensors) and the Engram as the sixth shard of the split.  The files are not the pinned ones, so they
# are a second layout beside the four-shard one: setup downloads the pinned four-shard files as before (HF_REVISIONS stays at cb02422c, the owner decides
# when the default moves), and takes this layout from "files already here" (--gguf-dir / --model-path, the app's "Use a local GGUF") by size and SHA-256.
# Same model: the experts and dense weights have the same types, shapes and sizes in both layouts; the converter folded the Engram gate (q_weight *
# k_weight -> engram_gate), merged the two Engram tables into one tensor (per_layer_token_embd, the same 110,595,290,400 bytes) and added the image tokens'
# routing bias (exp_probs_b_vl, used only for picture tokens).  The engine reads both (engram.gate.weight or q_weight/k_weight; ds4_model.cu).
DSV41_ISTA6_SHARD = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quant}-{i:05d}-of-00006.gguf"
DSV41_ISTA6_SHARDS = 5            # the files of a set itself; the sixth shard of its split is the Engram
DSV41_ISTA6_FILES = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf": (5780704, "3e8ddd11441edd4bc19bacaef5dbe38818af2cbdf0820fb70f1d2d851fcc192a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00006.gguf": (39340012896, "be4ceb1caf1bed28e886860cf6a52adc8f5ca9eb0d2098ade39bb4179d48b323"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00006.gguf": (38698805888, "758c66ba65236e8ce0820c21a7b6f9b2947681df8cc509f672a77722e8191b9a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00006.gguf": (39707055008, "04a5bea2b38eb576f49efc19c0a677768d6e2a8ea5772d72fb6cb79e1733a212"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00005-of-00006.gguf": (2969539840, "7c8dbee99890f05dbc420eba6f44393f5228d06248fe61e1ae8ed002449c7549"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf": (5780704, "0182a2a1f42d319a0dbc6e34aa82a063880c069fe3aafbbea86018f386e64e5f"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00006.gguf": (39581533600, "5c88039fa640460a9fc3465a6dfcd162a4083dd738d55e5b52e025be14555a08"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00006.gguf": (39417972640, "0fddebfd05f0a36cfd107ba5dfd2547e64df61adbab39560a5a5f5521ca099a5"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00006.gguf": (39079187648, "c577a43c6e6e615838a1655a0b162541edeac6f36542f08c4a9593a70dcaf500"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00005-of-00006.gguf": (37141423776, "851d8605bf504962054d17ea8a0d9e2e0a228d085d396bd865a4de9f16d90fbd"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf": (5780704, "f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf": (39535781184, "25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf": (39736712000, "8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf": (39036908384, "9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf": (19923784832, "7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1"),
}
# The Engram of that commit: the same name and size as the pinned one, converted in place (one tensor, split 6 of 6), so a new SHA-256.  Either file is the
# Engram of either layout (the pack reads the tables from both); the pin stays the old one, this hash is accepted beside it.
DSV41_ENGRAM_CONVERTED = (110595290816, "3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3")
DSV41_SHA_ALSO = {DSV41_ISTA_ENGRAM: (DSV41_ENGRAM_CONVERTED[1],)}
# the files every official set shares (name -> where it comes from); they live in models/deepseek-v4.1-flash-shared, once
DSV41_MMPROJ = "mmproj-BF16.gguf"     # the vision tower's weights (972 MB, pinned): packed into the model's pack when pictures are on (0.2.7)
DSV41_SHARED = {
    DSV41_MMPROJ: hf(DSV41_REPO),
    DSV41_ISTA_ENGRAM: hf(DSV41_ISTA_REPO),
    DSV41_SHARD.format(i=1): hf(DSV41_REPO) + "UD-Q8_K_XL/",      # only its metadata is read (pack_gguf --engram-meta)
    DSV41_DSPARK: hf(DSV41_REPO),
}
# Unsloth's own low-bit sets (2026-10-30, unsloth/DeepSeek-V4.1-Flash-1017-GGUF, UD-IQ1_S and UD-IQ1_M): shard 1 = metadata, shards 2-4 = the model, shard 5 = the
# Engram table (IQ4_NL, 110.6 GB; its data is byte-identical to ISTA's Engram file, so an ISTA Engram already here is used instead of fetching shard 5).
# Same tensor names as the IST-DAS six-shard layout; dense weights in Q6_K / Q8_0 / Q5_K, router F32.  Opt-in model names, not the download default.
DSV41_UD_SHARD = "DeepSeek-V4.1-Flash-1017-{quant}-{i:05d}-of-00005.gguf"
DSV41_UD_FILES = {
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf": (5780868, "5197c309479c7be06e3612dc696d7f18e13e5942cfe52e70c3c586b7e585a5ac"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf": (49707332128, "fbfb546557a33f41332ffbed7f15fbed249b9f7796975d9db571c8cf3870f335"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00003-of-00005.gguf": (49292777280, "394b7a06e04d4b1b96b6028ec3d4dec49e4716900c26a493e8c2aa38a357ea1a"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00004-of-00005.gguf": (40803165856, "323a7f59a2938879eaac82b37c443c85151e518071ad44c79b1e4580c91d0ed7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf": (110595290592, "771805301e695d72d268151dfc43043581ba904298574aa879d4349cbe67b020"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf": (5780868, "5197c309479c7be06e3612dc696d7f18e13e5942cfe52e70c3c586b7e585a5ac"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00002-of-00005.gguf": (49120285248, "e83af39db628614d252d49a316af8cd5911d78ab96a9e7a5997b392ece26eaf7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00003-of-00005.gguf": (48899707008, "6caa1da6302b532374d55da6510fc92a9031a0ee1bd2bb5bf7defca2477f3d98"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00004-of-00005.gguf": (24358544704, "71c6c54dd1f20008e2130188cc655742d3738a365de4571d21a56de503b206b7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00005-of-00005.gguf": (110595290592, "771805301e695d72d268151dfc43043581ba904298574aa879d4349cbe67b020"),
}
DSV41_FILES.update(DSV41_UD_FILES)      # (the repository's pins: REPO_PINS)
DSV41_ALL_FILES = {**DSV41_FILES, **DSV41_ISTA_FILES, **DSV41_ISTA6_FILES, **DSV41_LEGACY_FILES, **DSV41_UD_FILES}
DSV41_PROFILES = ROOT / "data" / "dsv41"           # the shipped hot-expert profiles (profile-2.00.bin, profile-2.25.bin, + .sha256)
DSV41_CONTEXT = 8192          # the context the engine was checked at (4096 and 8192; 64K is not tried yet)
DSV41_MAC_VISION = False      # an Apple Silicon Mac: the Metal engine has not run DeepSeek's picture tower (0.2.7.1), so pictures stay off there
MULTI_FILE_FAMILIES = {
    "deepseek41": {"title": "DeepSeek-V4.1-Flash", "by": "DeepSeek; the 2-bit GGUFs by IST-DAS, UD-Q8_K_XL by Unsloth", "arch": "deepseek41",
                   "about": "a big model that runs on a PC with a 12 GB card and 64 GB of RAM (the two official 2-bit versions: 257 or "
                            "274 GB of disk, about 5-7 tokens a second) or, in Unsloth's 508 GB version, on several graphics cards",
                   "hf": hf(DSV41_REPO) + "UD-Q8_K_XL/", "tag": "deepseek-v4.1-flash-", "name": "deepseek-v4.1-flash",
                   "license": "DeepSeek's model license: https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
                   "models": {"UD-Q8_K_XL": {"about": "Unsloth Dynamic 8-bit: the experts in MXFP4 as DeepSeek made them, the rest "
                                                       "in BF16 (the checkpoint's own precision)",
                              "ram_gb": 306, "shards": [DSV41_SHARD.format(i=i) for i in range(1, 10)]}},
                   "sha256": DSV41_ALL_FILES, "vision": True}     # the engine's own tower (strata-dsv41 --vision on, the default), 0.2.7
}
# The official sets first (a normal PC), Unsloth's 508 GB set last.  Each model: its shards, the sub-folder they are in at the repository
# (and in a --gguf-dir folder), the files shared with the other set ("shared"), the pack recipe and the shipped expert profile.
MULTI_FILE_FAMILIES["deepseek41"]["models"] = {
    "IST-DAS-2.00bpw": {
        "label": "IST-DAS 2.00 bpw",
        "about": "the smaller official version: 138 GB of model + 111 GB n-gram table + 8 GB drafter (257 GB of disk). Runs on a PC with "
                 "a 12 GB card and 40-64 GB of RAM at about 5-7 tokens a second; quality about 15.5% above the original model's "
                 "perplexity (code and agent work 5-9%)",
        "ram_gb": 40, "recipe": "ista", "subdir": None, "legacy_subdir": "2.00bpw", "profile": "profile-2.00.bin", "hf": hf(DSV41_ISTA_REPO),
        "shards": [DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=i) for i in range(1, 5)],
        "shards6": [DSV41_ISTA6_SHARD.format(quant="IQ2_XXS", i=i) for i in range(1, DSV41_ISTA6_SHARDS + 1)],
        "shared": [DSV41_ISTA_ENGRAM, DSV41_SHARD.format(i=1), DSV41_DSPARK]},
    "IST-DAS-2.25bpw": {
        "label": "IST-DAS 2.25 bpw",
        "about": "the better official version: 155 GB of model + 111 GB n-gram table + 8 GB drafter (274 GB of disk, 17 GB more than "
                 "2.00). About the same speed; quality about 7.3% above the original model's perplexity (code and agent work 2-3%)",
        "ram_gb": 44, "recipe": "ista", "subdir": None, "legacy_subdir": "2.25bpw", "profile": "profile-2.25.bin", "hf": hf(DSV41_ISTA_REPO),
        "shards": [DSV41_ISTA_SHARD.format(quant="IQ2_XS", i=i) for i in range(1, 5)],
        "shards6": [DSV41_ISTA6_SHARD.format(quant="IQ2_XS", i=i) for i in range(1, DSV41_ISTA6_SHARDS + 1)],
        "shared": [DSV41_ISTA_ENGRAM, DSV41_SHARD.format(i=1), DSV41_DSPARK]},
    "IST-DAS-1.75bpw": {
        "label": "IST-DAS 1.75 bpw",
        "about": "the smallest official version: 121 GB of model + 111 GB n-gram table + 8 GB drafter (239 GB of disk, 18 GB less than 2.00). Runs "
                 "where 2.00 is tight (a 12 GB card and 32-40 GB of RAM). Smallest: fine for code and agent work, noticeably weaker on plain prose; pick 2.00 if it fits",
        "ram_gb": 34, "recipe": "ista", "subdir": None, "legacy_subdir": None, "profile": "profile-2.00.bin", "hf": hf(DSV41_ISTA_REPO),
        "shards": [DSV41_ISTA_SHARD.format(quant="IQ1_M", i=i) for i in range(1, 5)],
        "shards6": [DSV41_ISTA6_SHARD.format(quant="IQ1_M", i=i) for i in range(1, DSV41_ISTA6_SHARDS + 1)],
        "shared": [DSV41_ISTA_ENGRAM, DSV41_SHARD.format(i=1), DSV41_DSPARK]},
    "UD-IQ1_S": {
        "label": "Unsloth UD-IQ1_S",
        "about": "Unsloth's smallest dynamic version: 122 GB of model (experts in IQ1_S / IQ2_XXS, attention in Q6_K) + the 111 GB n-gram table + 8 GB drafter. Same size class as IST-DAS 1.75 bpw. A 1-bit version: expect weaker answers than the 2.00 and 2.25 versions on long, detailed or careful work (short answers, code and tool calls looked fine in our check; its quality is not measured against the original yet). Pick 2.00 or 2.25 if they fit",
        "ram_gb": 36, "recipe": "udlow", "subdir": "UD-IQ1_S", "legacy_subdir": None, "profile": "profile-2.00.bin", "hf": hf(DSV41_REPO) + "UD-IQ1_S/",
        "shards": [DSV41_UD_SHARD.format(quant="UD-IQ1_S", i=i) for i in range(1, 5)],
        "engram_shard": DSV41_UD_SHARD.format(quant="UD-IQ1_S", i=5),
        "shared": [DSV41_ISTA_ENGRAM, DSV41_DSPARK]},
    "UD-IQ1_M": {
        "label": "Unsloth UD-IQ1_M",
        "about": "Unsloth's next dynamic version: 140 GB of model (experts in IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S, attention in Q6_K) + the 111 GB n-gram table + 8 GB drafter. Same size class as IST-DAS 2.00 bpw. Part of its experts are 1-bit: expect somewhat weaker answers than the 2.25 version on long, detailed or careful work (short answers, code and tool calls looked fine in our check; its quality is not measured against the original yet)",
        "ram_gb": 40, "recipe": "udlow", "subdir": "UD-IQ1_M", "legacy_subdir": None, "profile": "profile-2.00.bin", "hf": hf(DSV41_REPO) + "UD-IQ1_M/",
        "shards": [DSV41_UD_SHARD.format(quant="UD-IQ1_M", i=i) for i in range(1, 5)],
        "engram_shard": DSV41_UD_SHARD.format(quant="UD-IQ1_M", i=5),
        "shared": [DSV41_ISTA_ENGRAM, DSV41_DSPARK]},
    "UD-Q8_K_XL": {
        "label": "Unsloth UD-Q8_K_XL",
        "about": "Unsloth Dynamic 8-bit: the experts in MXFP4 as DeepSeek made them, the rest in BF16 (the checkpoint's own precision); "
                 "508 GB, for machines with hundreds of GB of RAM or several cards",
        "ram_gb": 306, "recipe": "ud", "subdir": "UD-Q8_K_XL", "shards": [DSV41_SHARD.format(i=i) for i in range(1, 10)],
        "shared": []},
}

# Every model file Strata downloads has a fixed size and SHA-256: (repository, file name) -> (bytes, sha256).  download()
# checks the file against it as soon as it is whole (a wrong file is deleted and setup stops), and stops before the
# download when the server's size is not the pinned one.  The files are fetched from the commits in HF_REVISIONS.
REPO_PINS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": ISTA_FILES,
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": SWIFT_FILES,
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": CODER_FILES,
    "unsloth/Qwen3.8-Flash-Next-GGUF": {**UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS, **UNSLOTH_BIG_SHARDS},
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": QWEN36_FILES,
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": GYRO_FILES,
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF": DSV41_FILES,
    "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF": DSV41_ISTA_FILES,
}
# EXPERIMENTAL, off by default (setup asks): a control vector shipped with the repository, see its README
ESP_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
# the image encoder on the GPU (~1.2 GB at 1024 image tokens) warms up before the engine starts, so the engine
# sizes its expert slots around it and the default reserve (700 MiB) is enough; engines before 0.1.2 need more
VISION_GPU_SMALL_RESERVE_MIB = 1000    # the tip for images on a <= 12 GB card (the engine's LOW line asked ~1003)
VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700},
          "cpu": {"max_tokens": 300, "reserve_mib": 0}}      # the CPU reader takes no VRAM: no reserve (serve/server.py adds 700 only for a GPU reader)
EXE = "strata.exe" if WIN else "strata"
VEXE = "strata-vision.exe" if WIN else "strata-vision"
HVEXE = "strata-vision-hip.exe" if WIN else "strata-vision-hip"     # AMD: the picture reader on the GPU (tools/vision, STRATA_VISION_HIP)
# MODEL-UPDATE Phase 1: the K-quant prompt kernels (Unsloth's UD-Q4/Q5/Q6_K_XL experts) are a library next to the CUDA
# engine, opened only for those models; an engine without it runs them on the FP16 prompt path
KQ_LIB = "strata_kq.dll" if WIN else "libstrata_kq.so"


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def flush_typed_ahead() -> None:
    """#841: keys pressed while setup was busy (a download, a build) are not answers to the next question: Enter pressed
    to "wake" a slow step answered "Go on anyway?" with its default.  Dropped before a question is asked, on a terminal
    only (a pipe or a test keeps its input)."""
    try:
        if not sys.stdin.isatty():
            return
        if WIN:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def ask(question, choices, default, yes):
    if yes:
        return default
    flush_typed_ahead()
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            fail("input ended before a setup answer was received",
                 "run setup in a terminal, or pass --yes to accept the recommended answers")
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def _urlopen(req, timeout=None):
    """Every network request of this file goes through here: starting and serving never reach the network
    (strata_control/acquisition.py), only an explicit setup, update or install does."""
    acquisition.require("a download")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        if tls.is_certificate_error(e):               # not the raw urlopen text, and never a way to skip the check
            raise tls.CertificateProblem(tls.host_of(req.full_url if hasattr(req, "full_url") else str(req))) from None
        raise


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    """An install, build or preparation step (pip, cmake, a package manager, tools/*.py): never part of a start."""
    acquisition.require("an install or build step (" + Path(str(cmd[0])).name + ")")
    if CONTROL:                                        # the app: only this Python running a bundled tool
        argv = [str(c) for c in cmd]
        if len(argv) < 2 or os.path.normcase(argv[0]) != os.path.normcase(sys.executable) or argv[1] == "-m" or \
                Path(argv[1]).resolve().parent not in ((ROOT / "tools").resolve(), (ROOT / "tools" / "dsv41").resolve()):
            raise acquisition.AcquisitionRefused("the desktop app's installer only runs its bundled preparation tools; "
                                                 + " ".join(argv[:2]))
    say("  > " + " ".join(str(c) for c in cmd))
    env = dict(os.environ if env is None else env)
    env.setdefault("STRATA_EXIT_WITH_PARENT", "1")     # tools/mtp_fetch.py ends when setup ends (#3257: a cancel left it writing)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True, errors="replace")
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def fetch_mtp(mtp: Path, env: dict, tries: int = 3):
    """The ~5 GB of MTP tensors (tools/mtp_fetch.py).  A timeout or a dropped connection stops one run, not the install (#3214 #3193):
    it is run again, and each run continues where the last one stopped.  The progress lines it prints are what the app shows."""
    cmd = [sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)]
    env = {**env, "STRATA_SOURCE": model_source()}
    for n in range(1, tries + 1):
        r = run(cmd, env=env, check=False)
        rc = 0 if r is None else r.returncode              # (a test double answers None)
        if rc == 0:
            return
        if n < tries:
            warn(f"the draft layer's download stopped (exit {rc}); trying again, it continues where it stopped "
                 f"({n + 1} of {tries})")
            time.sleep(5)
    fail(f"the draft layer's download keeps stopping (exit {rc}): the connection to Hugging Face is slow or drops",
         "run setup again - what is downloaded is kept. If it still stops, set STRATA_SOURCE=modelscope (setup --source modelscope) "
         "to read the same files from ModelScope")


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60).stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    """Write the finish mark.  A folder that cannot be written (--gguf-dir on a read-only share, #570) only costs the
    mark: setup says so and goes on (the step is repeated on the next run), it does not stop."""
    try:
        path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    except OSError as e:
        warn(f"the finish mark of {path.name} cannot be written ({e}): the next run does this step again")


# ------------------------------------------------------------------------------------------------ the PC
def _memory_status():
    """Windows' GlobalMemoryStatusEx: RAM, and the commit limit (ullTotalPageFile = RAM + page file)."""
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


def ram_gb():
    if WIN:
        return _memory_status().ullTotalPhys / 2**30
    if sys.platform == "darwin":
        try:
            value = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True, errors="replace", timeout=3, check=True).stdout.strip()
            return int(value) / 2**30
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0.0
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    return 0.0


_MM_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"


def _mm_multi_sz(name):
    """A REG_MULTI_SZ value of Memory Management, None when it cannot be read.  PagingFiles is what the Virtual memory
    dialog set; ExistingPageFiles is where the page files are now."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MM_KEY) as k:
            v, _t = winreg.QueryValueEx(k, name)
    except (ImportError, OSError):
        return None
    return [x for x in v if x] if isinstance(v, list) else None


def parse_paging_file(line: str):
    """One PagingFiles entry ("C:\\pagefile.sys 16000 64000", "C:\\pagefile.sys 0 0", "?:\\pagefile.sys"): (drive,
    initial MB, maximum MB), both -1 when Windows manages the size and the drive '?' when it manages every drive;
    None for anything else."""
    s = line.strip()
    if len(s) < 2 or s[1] != ":":
        return None
    parts = s.split()
    try:
        initial, maximum = (int(parts[1]), int(parts[2])) if len(parts) >= 3 else (-1, -1)
    except ValueError:
        initial, maximum = -1, -1
    if maximum <= 0:
        initial, maximum = -1, -1
    return s[0].upper(), initial, maximum


PAGE_FILE_FIXED_MB = 65536   # the advice: a fixed size (initial = maximum) of 64 GB, the size #60's PC started with


def _drive_mb(drive: str):
    """(free MB, the page file's size now in MB) of a drive, None when it cannot be read.  The page file is locked:
    its size comes from the directory entry."""
    root = f"{drive}:\\"
    try:
        du = shutil.disk_usage(root)
    except OSError:
        return None
    current = 0
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.name.lower() == "pagefile.sys":
                    current = e.stat(follow_symlinks=False).st_size
                    break
    except OSError:
        pass
    return du.free / 2**20, current / 2**20


def page_file_setting():
    """The page files Windows is set to use (PagingFiles, what the Virtual memory dialog writes): [(drive, initial MB,
    maximum MB, MB now, the drive's free MB)], the sizes -1 for a system-managed file; "?:" (every drive automatic) is
    the drive ExistingPageFiles names.  None off Windows or when the setting cannot be read."""
    if not WIN:
        return None
    lines = _mm_multi_sz("PagingFiles")
    if lines is None:
        return None
    auto = []                                          # "?:\pagefile.sys": where Windows keeps it now, else C:
    for s in _mm_multi_sz("ExistingPageFiles") or []:
        i = s.find(":")
        if i > 0:
            auto.append(s[i - 1].upper())
    auto = auto or [(os.environ.get("SystemDrive") or "C:")[0].upper()]
    files = []
    for line in lines:
        e = parse_paging_file(line)
        if e is None:
            continue
        drive, initial, maximum = e
        for d in (auto if drive == "?" else [drive]):
            free, now = _drive_mb(d) or (0.0, 0.0)
            files.append((d, initial, maximum, now, free))
    return files


def page_file_grows(files) -> bool:
    """Whether Windows grows one of these page files on demand: system-managed, or an initial size below the maximum."""
    return any(maximum <= 0 or initial < maximum for _d, initial, maximum, _now, _free in files or [])


def page_file_gb():
    """The page files' size (GB) that is there for sure on Windows, None elsewhere.  The graphics card's memory needs
    room there too: under Windows' driver model every allocation on the card is also charged to the commit (RAM + page
    file), so with the page file off or tiny the engine cannot use the free VRAM (issue #60).  What counts is the size
    the files have now (the commit limit minus RAM), or a file's configured initial size when that is larger (within
    its drive's free space) - never what a file may grow to: a page file Windows grows on demand may not grow in time
    while the card's memory is charged (#60: "System managed" and 4096-32768 MB still failed, a fixed 64 GB worked)."""
    if not WIN:
        return None
    m = _memory_status()
    mb = max(0.0, (m.ullTotalPageFile - m.ullTotalPhys) / 2**20)
    files = page_file_setting()
    if files:
        mb = max(mb, sum(max(now, min(max(initial, 0), now + free)) for _d, initial, _m, now, free in files))
    return mb / 1024


def page_file_advice(pf: float, files) -> str:
    """Step 1's warning for a page file below 4 GB (issue #60): a fixed size, and why when the setting (`files`,
    page_file_setting) has Windows grow it on demand."""
    grows = ("It is set to grow on demand (\"System managed\" or an initial size below the maximum), and a growing "
             "page file may not grow in time for the card's memory (#60: \"System managed\" and 4096-32768 MB still "
             "failed, a fixed 64 GB worked). ") if page_file_grows(files) else ""
    return (f"Windows' page file is {pf:.1f} GB: the graphics card's memory needs room there too (issue #60), so the "
            f"model may not start or may use less VRAM. {grows}Set a fixed size: System > About > Advanced system "
            "settings > Performance > Advanced > Virtual memory > Change, untick \"Automatically manage\", Custom size "
            f"with initial and maximum size {PAGE_FILE_FIXED_MB} MB, Set, then restart Windows")


def cpu_cores():
    """#642: (performance cores, efficiency cores) of a hybrid CPU (Intel 12th gen+, AMD Zen 5 + Zen 5c), counted
    as the engine's pool counts them (detect_cpu_topology: physical cores, by Windows' EfficiencyClass or Linux's
    cpu_capacity); None on a CPU whose cores are all alike, or when the OS does not say."""
    classes = []                                       # one entry per physical core: its efficiency/capacity class
    try:
        if WIN:
            k32 = ctypes.windll.kernel32
            n = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))   # RelationProcessorCore: the size
            if not n.value:
                return None
            buf = ctypes.create_string_buffer(n.value)
            if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
                return None
            raw, at = buf.raw[:n.value], 0
            while at + 10 <= len(raw):   # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, then
                rel, size = struct.unpack_from("<II", raw, at)               # PROCESSOR_RELATIONSHIP (Flags,
                if size <= 0:                                                # EfficiencyClass, ...)
                    break
                if rel == 0:
                    classes.append(raw[at + 9])
                at += size
        else:
            classes = linux_core_classes()
            if classes is None:
                return None
    except (OSError, ValueError, AttributeError):
        return None
    if not classes or max(classes) == min(classes):
        return None
    p = sum(1 for c in classes if c == max(classes))
    return p, len(classes) - p


def linux_core_classes(sys_root: str = "/sys"):
    """One class per physical core on Linux: 1 for a P-core, 0 for an E-core; None when the kernel does not say.
    #798: Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus) when they exist; else cpu_capacity, where an E-core is
    one under 90% of the largest - Turbo Boost Max 3.0 gives the "favored" P-cores a slightly higher capacity (1024
    against 1012), so "below the maximum" counted 2 of a Core Ultra 7 270K Plus's 8 P-cores."""
    pmu = linux_hybrid_pmu(sys_root + "/devices")
    seen, top = {}, 0
    for cpu in sorted((Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cap = cpu / "cpu_capacity"
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        if not cap.exists() and pmu is None:
            return None
        key = (pkg.read_text(encoding="utf-8").strip(), core.read_text(encoding="utf-8").strip()) \
            if pkg.exists() and core.exists() else cpu.name
        num = int(cpu.name[3:])
        capacity = int(cap.read_text(encoding="utf-8").strip()) if cap.exists() else 0
        top = max(top, capacity)
        if pmu is not None and num in pmu[1]:
            seen.setdefault(key, None)           # a listed E-core (the atom list and, #3262, the LP-E list)
        elif pmu is not None and num in pmu[0]:
            seen.setdefault(key, False)          # a listed P-core
        else:                                    # in no list (or no lists at all): its capacity decides
            seen.setdefault(key, capacity)
    # an E-core is under 90% of the largest capacity; the PMU lists (None: E, False: P) decide where they say
    return [0 if c is None else 1 if c is False else (1 if c * 10 >= top * 9 else 0) for c in seen.values()]


def cpulist(text: str) -> set:
    """A kernel cpulist ("0-7,16") as a set of CPU numbers."""
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def linux_hybrid_pmu(root: str = "/sys/devices"):
    """#798: (P-core CPUs, E-core CPUs) from Intel's hybrid PMU lists (cpu_core/cpus, cpu_atom/cpus), or None on a CPU
    that has no such lists."""
    try:
        p = cpulist((Path(root) / "cpu_core" / "cpus").read_text(encoding="utf-8"))
        e = cpulist((Path(root) / "cpu_atom" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        return None
    try:   # #3262: Meteor Lake / Arrow Lake-H list their LP-E cores in a third set, in neither of the two above
        e |= cpulist((Path(root) / "cpu_lowpower" / "cpus").read_text(encoding="utf-8"))
    except OSError:
        pass
    return (p, e) if p and e else None


def hybrid_pool_workers(cores) -> int | None:
    """#642 (Hardin22's measurement): on a hybrid CPU the expert pool runs best on the P-cores but the host loop's one
    plus HALF of the E-cores - an E-core runs the expert kernels ~2.2x slower and each layer waits for its slowest
    part (i9-14900KF, 8P + 16E: 15 workers decoded 165 / 116 tok/s against 106 / 84 with all 23).  Only on a CPU with
    more E-cores than P-cores: on an i7-13700KF (8P + 8E, docs/AMD_HIP.md's gfx1030 report) all 15 workers decoded
    38-42 tok/s against 36 with 8, so there the engine's own count stays.  None: the engine's own default (one worker
    per physical core but the host's) stays."""
    if not cores:
        return None
    p, e = cores
    if e <= p:
        return None
    return max(1, p - 1 + e // 2)


def recommend_pool_workers(args: list) -> list:
    """`args` with setup's recommended `--pool-workers` for a hybrid CPU, unless they set one already (a calibration's
    measured count, or the user's own).  A recommendation: the config line can be edited or removed."""
    n = hybrid_pool_workers(cpu_cores())
    if n is None or "--pool-workers" in args:
        return args
    n = min(n, POOL_WORKERS_CAP)   # setup never writes a big worker count (see plan_pool_workers)
    p, e = cpu_cores()
    ok(f"hybrid CPU ({p} performance + {e} efficiency cores): {n} CPU expert workers - the performance cores and half "
       "of the efficiency cores (--pool-workers in the config; START-HERE --calibrate measures it on this PC)")
    return [*args, "--pool-workers", str(n)]


def linux_sockets(sys_root: str = "/sys"):
    """(sockets, physical cores per socket) from the kernel's topology files, or None when they do not say."""
    cores = {}
    for cpu in (Path(sys_root) / "devices" / "system" / "cpu").glob("cpu[0-9]*"):
        pkg, core = cpu / "topology" / "physical_package_id", cpu / "topology" / "core_id"
        try:
            cores.setdefault(pkg.read_text(encoding="utf-8").strip(), set()).add(core.read_text(encoding="utf-8").strip())
        except OSError:
            return None
    return (len(cores), min(len(c) for c in cores.values())) if cores else None


def cpu_sockets():
    """(sockets, physical cores per socket); None when unknown.  Windows: the packages and cores that
    GetLogicalProcessorInformationEx lists (RelationProcessorPackage 3, RelationProcessorCore 0)."""
    try:
        if not WIN:
            return linux_sockets()
        k32 = ctypes.windll.kernel32
        counts = {}
        for rel in (0, 3):
            n = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(rel, None, ctypes.byref(n))
            if not n.value:
                return None
            buf = ctypes.create_string_buffer(n.value)
            if not k32.GetLogicalProcessorInformationEx(rel, buf, ctypes.byref(n)):
                return None
            raw, at, c = buf.raw[:n.value], 0, 0
            while at + 8 <= len(raw):
                size = struct.unpack_from("<II", raw, at)[1]
                if size <= 0:
                    break
                c += 1
                at += size
            counts[rel] = c
        return (counts[3], counts[0] // counts[3]) if counts[3] else None
    except Exception:
        return None


def two_socket_note(sockets) -> list[str]:
    """Bench #674 #707 (2-socket Xeons: 17-18 workers beat 35 by ~20%): a tip to keep the expert pool on one socket's
    cores, one fewer for the host loop.  Nothing is written to the config."""
    if not sockets or sockets[0] < 2 or sockets[1] < 3:
        return []
    n = sockets[1] - 1
    return [f"tip: this PC has {sockets[0]} CPU sockets of {sockets[1]} cores. Expert workers on the other socket have "
            f"been measured slower than fewer, local ones (35 vs 17-18 on 2-socket Xeons): try --pool-workers {n} in "
            "the config's args (START-HERE --calibrate measures it on this PC)"]


# The engine's default count of CPU expert workers (include/strata/kernels/cpu/pool_plan.hpp, which tools/test_pool_plan.py
# keeps in step with these numbers).  Setup writes no worker count on an ordinary CPU: the engine picks it, from the
# physical cores (not the SMT siblings), minus the host thread's core and a reserve on a big CPU, on the host thread's
# socket, under a cap.  What setup does write (a hybrid CPU's count, the vision encoder's threads) is capped the same way.
POOL_WORKERS_CAP = 32
POOL_RESERVE_FROM = 16
POOL_RESERVE = 1
VISION_THREADS_CAP = 16


def plan_pool_workers(logical: int, cores_per_package=None, quota: float = 0.0, hybrid=None) -> int:
    """The engine's default pool worker count for a CPU of `logical` threads whose physical cores per socket are
    `cores_per_package` (the host thread's socket first; None/empty: the layout is not known) and a container CPU
    quota (0: none)."""
    cores = list(cores_per_package or [])
    if cores and sum(cores) > 0:
        local = cores[0]
    else:                                              # no layout: half the logical processors on a many-thread CPU
        local = logical // 2 if logical >= 16 else logical
    n = max(1, local - 1)                              # the host thread has the first core
    if n >= POOL_RESERVE_FROM:
        n -= POOL_RESERVE
    if hybrid and hybrid[0] > 0 and hybrid[1] > hybrid[0]:   # (P-cores, E-cores): the P-cores but the host's + half the E-cores
        n = min(n, max(1, hybrid[0] - 1 + hybrid[1] // 2))
    if quota > 0:
        n = min(n, max(1, int(quota) - 1))
    return max(1, min(n, POOL_WORKERS_CAP))


def cpu_core_layout():
    """(logical threads, [physical cores per socket]); the cores list is empty when the OS does not say."""
    logical = os.cpu_count() or 1
    sockets = cpu_sockets()
    return logical, ([sockets[1]] * sockets[0] if sockets else [])


def pool_workers_note(args, layout=None) -> list[str]:
    """A plain warning when the config's `--pool-workers` is above what this CPU uses well.  Recommend, never force: the
    count stays as it is and the engine runs with it."""
    if "--pool-workers" not in args:
        return []
    try:
        asked = int(args[args.index("--pool-workers") + 1])
    except (IndexError, ValueError):
        return []
    hybrid = cpu_cores() if layout is None else None
    logical, cores = layout or cpu_core_layout()
    rec = plan_pool_workers(logical, cores, hybrid=hybrid)
    if asked <= rec:
        return []
    why = ("more than this PC has CPU threads" if asked + 1 > logical else
           "more than its physical cores" if cores and asked + 1 > sum(cores) else
           "more than the cores of one CPU socket" if len(cores) > 1 and asked + 1 > cores[0] else
           "more than the memory bandwidth can feed")
    return [f"the config asks for {asked} CPU expert workers, {why}: extra workers can make Strata slower. Recommended: "
            f"--pool-workers {rec} (or remove the setting: the engine picks that by itself). It stays as you set it."]


def vision_threads(logical: int | None = None) -> int:
    """The CPU vision encoder's threads: one per physical core as before (half the logical threads), at most
    VISION_THREADS_CAP - a 256-thread server does not need 128 of them."""
    return max(1, min((logical if logical is not None else (os.cpu_count() or 8)) // 2, VISION_THREADS_CAP))


def cpu_info():
    """(name, avx2, avx512): avx512 means everything Strata's fast AVX-512 kernels use (F, BW, VL, VNNI, VBMI),
    the same test the engine makes (cpu_avx512_ok), not just AVX-512F."""
    name, avx2, avx512 = platform.processor() or "unknown CPU", False, False
    if WIN:
        pf = ctypes.windll.kernel32.IsProcessorFeaturePresent
        avx2 = bool(pf(40)) or _cpuid_avx2()      # PF_AVX2_INSTRUCTIONS_AVAILABLE, else the CPU itself (#159)
        n = out(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
        name = n or name
        avx512 = bool(pf(41)) and _cpuid_avx512_full()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            mflags = re.search(r"^flags\s*:\s*(.*)$", txt, re.M)
            if mflags is None:
                # aarch64 lists Features, not the x86 flags line, and no "model name": lscpu names the cores (a DGX Spark:
                # Cortex-X925 + Cortex-A725). No AVX of any kind; the arm64 engine needs none.
                cores = re.findall(r"^\s*Model name:\s*(.+)$", out(["lscpu"]), re.M)
                name = " + ".join(dict.fromkeys(c.strip() for c in cores)) or (platform.processor() or "arm64 CPU")
                return name, False, False
            flags = set(mflags.group(1).split())
            avx2 = "avx2" in flags
            avx512 = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"} <= flags
            m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
            name = m.group(1) if m else name
        except OSError:
            pass
    return name, avx2, avx512


def _cpuid_floor() -> str:
    """Below AVX2 (Windows): "avx" when the CPU has AVX and the OS saves the YMM registers, "sse4.2" with SSE4.2 and
    POPCNT, else ""."""
    try:
        regs = (ctypes.c_uint32 * 4)()
        _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                         0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                  ctypes.addressof(regs), 1)
        ecx1 = regs[2]
        if (ecx1 >> 27) & 1 and (ecx1 >> 28) & 1:                     # OSXSAVE, AVX
            xcr0 = (ctypes.c_uint32 * 2)()
            _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                    # r8=rcx; ecx=0; xgetbv
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
            if xcr0[0] & 6 == 6:
                return "avx"
        return "sse4.2" if (ecx1 >> 20) & 1 and (ecx1 >> 23) & 1 else ""
    except Exception:
        return ""


def is_arm64() -> bool:
    """An ARM64 CPU (DGX Spark / GB10, Grace): there is no AVX here, and no ready-made engine for it."""
    return platform.machine().lower() in ("aarch64", "arm64")


def is_gb10(g) -> bool:
    """A DGX Spark's GPU: the GB10 (compute capability 12.1), on arm64 Linux, with the CPU's memory as its own. The part the Spark settings
    below were measured on (docs/DGX_SPARK_LINUX.md); any other unified-memory NVIDIA GPU gets the ordinary settings."""
    return isinstance(g, dict) and bool(g.get("uma")) and str(g.get("arch")) == "121" and is_arm64() and sys.platform.startswith("linux")


def cpu_floor(avx2: bool) -> str:
    """The experimental older-CPU build this PC needs (#394 #595 #623): "" with AVX2 (the normal engine), "avx" (Sandy /
    Ivy Bridge, AMD Bulldozer), "none" (SSE4.2 + POPCNT: Nehalem, Westmere), or "unsupported".  STRATA_ISA_FLOOR=avx|
    none asks for that build on any PC (testing it on a newer one)."""
    forced = os.environ.get("STRATA_ISA_FLOOR", "").strip().lower()
    if forced in ("avx", "none"):
        return forced
    if is_arm64():
        return ""
    if avx2:
        return ""
    if WIN:
        f = _cpuid_floor()
    else:
        try:
            txt = open("/proc/cpuinfo").read()
            flags = set(re.search(r"^flags\s*:\s*(.*)$", txt, re.M).group(1).split())
        except (OSError, AttributeError):
            flags = set()
        f = "avx" if "avx" in flags else "sse4.2" if {"sse4_2", "popcnt"} <= flags else ""
    return {"avx": "avx", "sse4.2": "none"}.get(f, "unsupported")


def _cpuid_avx512_full() -> bool:
    """Windows has no feature bit for VNNI / VBMI: ask the CPU (CPUID leaf 7) through a tiny machine-code stub."""
    try:
        code = bytes([0x53, 0x49, 0x89, 0xC8, 0xB8, 0x07, 0x00, 0x00, 0x00, 0x31, 0xC9, 0x0F, 0xA2,   # push rbx; r8=rcx; cpuid(7,0)
                      0x41, 0x89, 0x18, 0x41, 0x89, 0x48, 0x04, 0x5B, 0xC3])                   # [r8]=ebx,[r8+4]=ecx; pop rbx
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
        if not buf:
            return False
        ctypes.memmove(buf, code, len(code))
        regs = (ctypes.c_uint32 * 2)()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(buf)(ctypes.addressof(regs))
        ebx, ecx = regs[0], regs[1]
        need_ebx = (1 << 16) | (1 << 30) | (1 << 31)                   # F, BW, VL
        need_ecx = (1 << 1) | (1 << 11)                                # VBMI, VNNI
        return (ebx & need_ebx) == need_ebx and (ecx & need_ecx) == need_ecx
    except Exception:
        return False


def _run_stub(code: bytes, *args) -> None:
    """Runs a few bytes of x64 machine code (Windows calling convention: the arguments in rcx, rdx)."""
    k32 = ctypes.windll.kernel32
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
    buf = k32.VirtualAlloc(None, len(code), 0x3000, 0x40)
    if not buf:
        raise OSError("VirtualAlloc failed")
    try:
        ctypes.memmove(buf, code, len(code))
        ctypes.CFUNCTYPE(None, *[ctypes.c_void_p] * len(args))(buf)(*args)
    finally:
        k32.VirtualFree(buf, 0, 0x8000)


def _cpuid_avx2() -> bool:
    """AVX2 asked from the CPU (CPUID leaf 7 EBX bit 5), with the OS saving the YMM registers (OSXSAVE + XCR0):
    Windows' IsProcessorFeaturePresent(PF_AVX2) says no on some PCs whose CPU has it (a Ryzen 9 3950X, #159)."""
    try:
        def cpuid(leaf):
            regs = (ctypes.c_uint32 * 4)()
            _run_stub(bytes([0x53, 0x49, 0x89, 0xC8, 0x89, 0xD0, 0x31, 0xC9, 0x0F, 0xA2,      # push rbx; r8=rcx; eax=edx; ecx=0; cpuid
                             0x41, 0x89, 0x00, 0x41, 0x89, 0x58, 0x04, 0x41, 0x89, 0x48, 0x08,  # [r8]=eax, [r8+4]=ebx, [r8+8]=ecx
                             0x41, 0x89, 0x50, 0x0C, 0x5B, 0xC3]),                             # [r8+12]=edx; pop rbx
                      ctypes.addressof(regs), leaf)
            return list(regs)
        if cpuid(0)[0] < 7:
            return False
        ecx1 = cpuid(1)[2]
        if not (ecx1 >> 27) & 1 or not (ecx1 >> 28) & 1:             # OSXSAVE, AVX
            return False
        xcr0 = (ctypes.c_uint32 * 2)()
        _run_stub(bytes([0x49, 0x89, 0xC8, 0x31, 0xC9, 0x0F, 0x01, 0xD0,                        # r8=rcx; ecx=0; xgetbv
                         0x41, 0x89, 0x00, 0x41, 0x89, 0x50, 0x04, 0xC3]), ctypes.addressof(xcr0))
        if xcr0[0] & 6 != 6:                                           # the OS saves XMM and YMM
            return False
        return bool((cpuid(7)[1] >> 5) & 1)
    except Exception:
        return False


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use)."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc, drv = [x.strip() for x in line.split(",")]
            try:
                mem_gb = float(mem) / 1024.0   # nvidia-smi nounits: MiB
            except ValueError:
                if not is_arm64():
                    raise   # as before on x86: a card whose memory nvidia-smi does not report is not listed
                # a DGX Spark (GB10): one memory for the CPU and the GPU, and nvidia-smi gives memory.total as [N/A]. Counted as
                # Strix Halo's is (amd_apply_uma): the GPU may use all of the RAM but the OS's share, and none of it is extra room
                # beside the RAM (dedicated_gb 0), so the low-RAM rules do not count it twice
                shared = max(0.0, ram_gb() - UMA_OS_LEFT_GB)
                found.append({"index": int(idx), "name": name, "vram_gb": shared, "arch": cc.replace(".", ""),
                              "driver": drv, "uma": True, "dedicated_gb": 0.0, "shared_gb": shared})
                continue
            found.append({"index": int(idx), "name": name, "vram_gb": mem_gb, "arch": cc.replace(".", ""),
                          "driver": drv})
        except ValueError:
            continue
    return found


def gpu_drives_display(g) -> bool:
    """#779: True when nvidia-smi says this card has a display attached (display_active Enabled): its desktop needs
    VRAM too, and a full expert cache beside it has crashed laptops."""
    text = out(["nvidia-smi", "-i", str(g.get("index", 0)), "--query-gpu=display_active", "--format=csv,noheader"])
    return text.strip().lower() == "enabled"


def gpu_compute_mode(index: int) -> str:
    """#1445: nvidia-smi's compute mode for this card ("Default", "Exclusive_Process", "Prohibited", ...); "" when it
    does not say."""
    return out(["nvidia-smi", "-i", str(index), "--query-gpu=compute_mode", "--format=csv,noheader"]).strip()


def compute_mode_warning(index: int, mode: str) -> str | None:
    """#1445: the warning for a card that is not in the Default compute mode, or None.  In Exclusive_Process only one
    process may hold a CUDA context, so the engine, the vision encoder and the tuning run cannot share the card
    ("CUDA-capable device(s) is/are busy or unavailable"); Prohibited allows none.  A warning only: it is the
    administrator's setting, Strata never changes it."""
    if not mode or mode.lower() == "default" or mode.lower().startswith("n/a") or mode.lower().startswith("[n/a"):
        return None
    return (f"GPU {index} is in the compute mode {mode}, not Default: a second process cannot use the card while "
            "the first holds it, so the vision encoder or the tuning run can fail with \"CUDA-capable device(s) is/are "
            f"busy or unavailable\". If it does, set it back with: sudo nvidia-smi -i {index} -c DEFAULT")


def pcie_link(index: int) -> dict | None:
    """The NVIDIA card's PCIe link: {"gen": the generation card and board both run (an idle card drops to a lower
    one, so the current generation is not asked), "gpu_gen", "host_gen", "width", "max_width"}; None when nvidia-smi
    does not say."""
    s = out(["nvidia-smi", "-i", str(index), "--query-gpu=pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax,"
             "pcie.link.width.current,pcie.link.width.max", "--format=csv,noheader,nounits"])
    try:
        gen, gpu_gen, host_gen, width, max_width = (int(x.strip()) for x in s.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"gen": gen, "gpu_gen": gpu_gen, "host_gen": host_gen, "width": width, "max_width": max_width}


PCIE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.97, 5: 3.94}    # GB/s per lane, each direction


def pcie_lines(link: dict) -> tuple[str, str | None]:
    """(the ok line, a warning or None) for step 1.  The copy rate the engine measures is about 75% of the link's."""
    lim = []
    if link["gpu_gen"] > link["gen"]:
        lim.append(f"the card supports {link['gpu_gen']}.0, the board {link['host_gen']}.0")
    line = f"PCIe: {link['gen']}.0 x{link['width']}" + (f" ({'; '.join(lim)})" if lim else "")
    rate = PCIE_GBPS.get(link["gen"], 0) * link["width"]
    if 0 < rate < 12:
        line += (f" - up to ~{rate:.0f} GB/s to the GPU: long prompts are read slower than on a PCIe 4.0 x16 PC (their "
                 "experts are copied over it); the engine measures the link at start for the output speed")
    warn_line = None
    if link["width"] < link["max_width"]:
        # (#912) the width is read now, and some cards and laptops narrow the link while idle: a hint, not a verdict
        warn_line = (f"the GPU's PCIe link reads {link['width']} of its {link['max_width']} lanes right now (some cards "
                     "narrow it when idle); if it stays narrow under load: is it in the right slot (the one wired "
                     "x16), fully seated, and not sharing lanes with an M.2 drive? (the BIOS can say)")
    return line, warn_line


GPU_PICK = None                                         # --gpu N (issue #51); None: the card with the most VRAM
SPLIT_MIN_VRAM_GB = 8                                   # a card sharing a model holds the dense weights and its own
                                                        # prompt buffers too (docs/MULTI_GPU.md)
SPLIT_PROMPT_VRAM_GB = 12                               # #448: a split stage lends a prompt chunk's buffers from its
                                                        # own cache; below this one cannot fund a 4096-token chunk
                                                        # (a 10 GB RTX 3080 beside a 32 GB card: 512 tokens, prompts
                                                        # 6.2x slower), while the big card alone could


def cc(g) -> str:
    return f"{g['arch'][:-1]}.{g['arch'][-1]}"


OLD_GPUS = None       # why Pascal / Volta cards are admitted in this run (old_gpus_opt_in), None: they are not


def experimental_sm60() -> bool:
    """#295: STRATA_EXPERIMENTAL_SM60=1 admits Pascal (6.x) and Volta (7.0) cards, run by the experimental CUDA 12
    engine (-DSTRATA_EXPERIMENTAL_SM60=ON).  So does naming such a card (--gpu N / --gpus), --cuda 12, or a PC that
    has no newer card (old_gpus_opt_in)."""
    return os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1" or OLD_GPUS is not None


def sm60_card(arch) -> bool:
    return 60 <= int(arch) <= 70


def maxwell_card(arch) -> bool:
    """#3218: Maxwell (5.x).  No ready-made engine holds code for it; it is a source build (docs/OLDER_GPUS.md)."""
    return 50 <= int(arch) < 60


def maxwell_note(found) -> str:
    """#3218: what to tell a PC whose only NVIDIA cards are Maxwell: setup cannot install an engine for it, and a
    source build exists.  Empty when there is no Maxwell card.  A warning with the way forward, never a switch: setup
    does not route a Maxwell card to an engine that holds no code for it."""
    cards = [g for g in found if maxwell_card(g["arch"])]
    if not cards:
        return ""
    names = ", ".join(f"GPU {g['index']} ({g['name']})" for g in cards)
    return (f"{names}: Maxwell (compute capability {cc(cards[0])}). No ready-made Strata engine holds code for it, so "
            "setup cannot install one. One reporter built the engine from source for two Quadro M6000 with a CUDA 12.x "
            "toolkit (cmake -B build -G Ninja -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=52) and saw about "
            "22 tokens a second with a 1-bit Qwen3.8-Flash-Next; nobody has checked the answer text yet. Steps and "
            "caveats: docs/OLDER_GPUS.md. A card of the RTX 20 series or newer runs the ready-made engine")


def old_gpus_opt_in(found, named=(), cuda=None, other=False):
    """Why this run may use Pascal / Volta cards (the experimental CUDA 12 engine), or None.  The cards are an opt-in:
    the user named one (`named`: --gpu / --gpus), asked for --cuda 12, set STRATA_EXPERIMENTAL_SM60=1, or the PC has
    no card the ready-made engine runs on and no supported AMD card (`other`; it used to stop there).  A PC with a
    newer card keeps recommending it."""
    old = [g for g in found if sm60_card(g["arch"])]
    if not old:
        return None
    if os.environ.get("STRATA_EXPERIMENTAL_SM60", "").strip() == "1":
        return "STRATA_EXPERIMENTAL_SM60=1"
    if str(cuda) == "12":
        return "--cuda 12"
    picked = [g for g in old if g["index"] in set(named)]
    if picked:
        return "you chose " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in picked)
    if not other and not any(int(g["arch"]) >= CUDA13_MIN_ARCH for g in found):
        return "it is the only kind of NVIDIA GPU in this PC"
    return None


def named_gpus(gpu, gpus) -> list:
    """The card numbers --gpu / --gpus name (an unreadable value: none; parse_gpus says what is wrong later)."""
    try:
        if gpus and str(gpus).strip().lower() != "all":
            return [int(x) for x in str(gpus).split(",") if x.strip()]
        return [int(gpu)] if gpu is not None else []
    except ValueError:
        return []


_CUDA12_FLOORS: dict = {}


def cuda12_min_driver(feed: bool = False) -> tuple[int, str, str]:
    """(driver major, the package's figure such as "576.02", where it came from): the oldest NVIDIA driver the CUDA 12 engine runs on, read
    from the engine package (installed, else - with `feed` - the signed feed, else the figure the packages carry) and not from a constant
    of setup's own.  Remembered for the run.  Only a setup run asks the feed (a few seconds at most, once); a plain call reads the disk."""
    got = _CUDA12_FLOORS.get(True) or _CUDA12_FLOORS.get(feed)
    if got:
        return got
    from strata_control import driver_floor
    roots = [engine_component.components_dir(), home() / "engine-feed"]
    source = None
    if feed and not acquisition.is_forbidden():
        try:
            from strata_cli import update_cmd
            source = update_cmd._feed_source(argparse.Namespace(feed=None), home())
        except Exception:                                  # noqa: BLE001 - no feed address: the built-in figure
            source = None
    try:
        from strata_control import targets
        platform_name = targets.host_platform()
    except Exception:                                      # noqa: BLE001
        platform_name = ""
    figure, where = driver_floor.cuda12_floor(platform_name, roots=roots, feed_source=source)
    _CUDA12_FLOORS[feed] = got = (driver_floor.major(figure), figure, where)
    return got


def cuda_choice(archs, cuda=None, driver=None):
    """The CUDA toolkit of one model's engine: (12 or 13, why).  13 (the ready-made engine) unless a card is older
    than CUDA 13 supports (Pascal / Volta: CUDA 13 cannot compile for them) - one engine runs per model, so its oldest
    card decides.  `cuda` (--cuda 12|13) overrides it; setup recommends, it does not refuse (the caller warns)."""
    archs = sorted({int(x) for x in archs})
    old = [a for a in archs if a < CUDA13_MIN_ARCH]
    if str(cuda) == "13":
        return 13, ("--cuda 13 (as you chose)" + (f"; CUDA 13 has no code for sm_{old[0]}: the engine will not run "
                                                   "on that card" if old else ""))
    if str(cuda) == "12":
        return 12, "--cuda 12 (as you chose" + ("; RTX 50 (sm_120) engines built with CUDA 12.8 crashed on long "
                                                  "prompts, #220" if archs and archs[-1] >= 120 else "") + ")"
    if old:
        return 12, (f"sm_{old[0]} is older than CUDA 13 supports (it dropped Pascal and Volta): this model runs the "
                    "experimental CUDA 12 engine")
    if driver and CUDA12_COMPAT_DRIVER <= int(driver) < MIN_DRIVER:      # #3843: a driver too old for CUDA 13 still runs the CUDA 12 engine
        return 12, (f"the NVIDIA driver {driver} is older than CUDA 13 needs ({MIN_DRIVER}): this model runs the CUDA 12 engine "
                    f"(a newer driver from https://www.nvidia.com/drivers gives the CUDA 13 engine)")
    return 13, None


def cuda12_driver_warning(driver) -> str | None:
    """A plain warning when the NVIDIA driver is older than the ready-made CUDA 12 engine package needs, or None.  The number is the package's
    own (cuda12_min_driver), the same one `strata update` enforces, so setup no longer says "works" where the update says "no engine" (#4247 #4188).
    Recommend, never force: setup goes on (an engine compiled here, or one the person brings, may still run)."""
    floor, figure, where = cuda12_min_driver()
    if not driver or int(driver) >= floor:
        return None
    return (f"the NVIDIA driver {driver} is older than the ready-made CUDA 12 engine package needs ({figure}, {where}): `strata update` will not "
            "install that package on this PC. Update the driver from https://www.nvidia.com/drivers"
            + ("; setup compiles the engine here instead, which can run on an older driver" if not WIN else ""))


def engine_dir(toolkit=13) -> Path:
    """The folder of the engine a model runs: engine/ (CUDA 13, or HIP), engine-cuda12/ (the experimental one)."""
    return ROOT / (ENGINE12_DIR if int(toolkit) == 12 else "engine")


def config_toolkit(cfg: dict) -> int:
    """12 when a model config runs the experimental CUDA 12 engine (its exe is in engine-cuda12/), else 13."""
    return 12 if cfg.get("cuda") == 12 or Path(str(cfg.get("exe", ""))).parent.name == ENGINE12_DIR else 13


def gpu_problem(g, together=False):
    """Why Strata cannot use this card, in plain words (None: it can)."""
    if int(g["arch"]) < 75 and not (sm60_card(g["arch"]) and experimental_sm60()):
        return (f"not supported - older than the RTX 20 series (compute capability {cc(g)}; Strata needs 7.5 or "
                "newer" + ("; experimental: choose it with --gpu " + str(g["index"]) + " (the CUDA 12 engine, "
                          "docs/OLDER_GPUS.md)" if sm60_card(g["arch"]) else
                          "; Maxwell only runs from a source build, docs/OLDER_GPUS.md" if maxwell_card(g["arch"]) else "") + ")")
    if together and g["vram_gb"] < SPLIT_MIN_VRAM_GB - 0.5:
        return (f"not supported together with other GPUs - {g['vram_gb']:.0f} GB of VRAM (a card sharing the model "
                f"needs {SPLIT_MIN_VRAM_GB} GB or more)")
    return None


def gpu_rank(g):
    """The order cards share a model in: the newest generation first (it gets the first layers and most of the
    work), then the most VRAM."""
    return (-int(g["arch"]), -round(g["vram_gb"]), g["index"])


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your NVIDIA GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM - " + ("can be used" if p is None else p))


def together_ok(found) -> list:
    """The cards that can share one model, in the order they would (empty if fewer than two)."""
    ok_ = sorted([g for g in found if gpu_problem(g, together=True) is None], key=gpu_rank)
    return ok_ if len(ok_) >= 2 else []


def split_short(cards) -> list:
    """#448: the later cards of a split that would cap its prompt chunk below what the first card alone reads (a card
    under SPLIT_PROMPT_VRAM_GB beside one that has it).  Empty: the split is recommended as before."""
    if len(cards) < 2 or cards[0]["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5:
        return []
    return [g for g in cards[1:] if g["vram_gb"] < SPLIT_PROMPT_VRAM_GB - 0.5]


def split_short_note(g) -> str:
    return (f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB) is too small to lend a split its prompt buffers: "
            "it would cap prompt reading at 512-2048-token chunks, several times slower than the first card alone "
            "(#448). It can serve as a helper expert cache instead (docs/SECOND_GPU.md)")


def parse_gpus(text, found) -> list:
    """--gpus / --gpu with several: "0,2" or "all" (every card that can share the model)."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in together_ok(found)]
        if not sel:
            gpu_table(found)
            fail("--gpus all: this PC does not have two GPUs Strata can use together")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 0,2 (one GPU: --gpu 0)")
    return sel


def check_gpus(sel, found, what="", yes=False, named=False) -> None:
    """Stops with a plain message when a chosen card is missing or cannot be used, and says what can.  named: the user
    named these cards (--gpus 0,1, or a config that has them): a card that is only short of VRAM for sharing the model
    is then a risk to confirm, not a stop (the owner's rule; --yes with the named cards is the consent)."""
    together = len(sel) > 1
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g, together)
        if p is None:
            continue
        if named and g is not None and gpu_problem(g) is None:     # it runs Strata; only its VRAM is small
            confirm_risk(f"GPU {i} ({g['name']}) has {g['vram_gb']:.0f} GB of VRAM: a card sharing the model needs "
                         f"{SPLIT_MIN_VRAM_GB} GB or more (it holds the dense weights of its layers and its own prompt "
                         "buffers), so the model may not start, or run slower than without it", True, yes,
                         f"GPU {i} ({g['name']}) {what}is not used together with other GPUs: {p}",
                         "leave it out of --gpus, or answer y to use it anyway", "  Use it anyway?")
            warn(f"GPU {i} ({g['name']}) is used together with the others, as you chose")
            continue
        say()
        gpu_table(found)
        can = together_ok(found)
        single = [x for x in found if gpu_problem(x) is None]
        ones = " or ".join(f"--gpu {x['index']}" for x in single)
        both = "--gpus " + ",".join(str(x["index"]) for x in can) if can else ""
        hint = ((f"use these together: {both}" + (f" (or one card: {ones})" if not together else "")) if can else
                f"use one card: {ones}" if single else "Strata needs an NVIDIA RTX 20 series or newer card")
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} {what}cannot be used: {p}", hint)


def arch_numbers(value) -> list:
    """The CUDA architectures of a BUILD.json `archs` / `vision_archs` value, as numbers.  The packaging recipes write the picture
    reader's as ONE string ("75;86;89;120", packaging/package_engine.py stamp_vision_archs, build_cuda_linux.sh), the engine's own as a
    list of numbers ([75, 86, 89, 120]); [60, "sm_61", "70-real", "compute_89"] and "60,61 70" are read too (#3810: a string was
    iterated character by character and setup stopped with int(';'))."""
    if value is None or value == "":
        return []
    parts = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[;,\s]+", str(value))
    out: list = []
    for part in parts:
        m = re.search(r"\d+", str(part))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def arch_names(value) -> list:
    """The AMD architectures ("gfx1100") of a BUILD.json `archs` / `vision_archs` value, as strings: a list or one ";" / "," separated string."""
    if value is None or value == "":
        return []
    parts = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[;,\s]+", str(value))
    return [str(p) for p in parts if str(p)]


def engine_archs(toolkit=13):
    """The GPU generations the installed engine has code for: (archs, ptx), or None when there is none."""
    info = engine_dir(toolkit) / "BUILD.json"
    try:
        meta = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return arch_numbers(meta.get("archs")), bool(meta.get("ptx"))


def engine_archs_hip():
    """The AMD architectures the installed HIP engine was compiled for ("gfx1201", ...), or None."""
    try:
        meta = json.loads((ROOT / "engine" / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return arch_names(meta.get("archs")) if meta.get("backend") == "hip" else None


def engine_runs_on(g, toolkit=13) -> bool:
    ea = engine_archs(toolkit)
    if ea is None or not ea[0]:
        return True
    archs, ptx = ea
    return int(g["arch"]) in archs or (ptx and int(g["arch"]) > max(archs))


def start_gpus(text):
    """--gpus when starting an installed model: NVIDIA cards as nvidia-smi numbers them, or on a PC whose AMD cards
    are the ones Strata can use, AMD cards as setup lists them ("all": every supported AMD card)."""
    if not text:
        return None
    found = gpus()
    if str(text).strip().lower() == "all" and not WIN and not together_ok(found):
        amd = amd_gpus()
        if len([g for g in amd if amd_problem(g) is None]) >= 2:
            return [g["index"] for g in amd_parse_gpus("all", amd)]
    if not found and not WIN:                          # #1594: no NVIDIA card, AMD ones: "1,0" names those, as setup lists them
        amd = amd_gpus()
        if amd:
            return [g["index"] for g in amd_parse_gpus(text, amd)]
    return parse_gpus(text, found)


def choose_gpus(a, found) -> list:
    """Which cards this install uses: --gpus / --gpu, or asked when two or more can share the model (the two best
    together recommended), else the supported card with the most VRAM.  Returns their numbers, in the engine order, the Main card last when roles are chosen."""
    if a.gpus:
        sel = parse_gpus(a.gpus, found)
        check_gpus(sel, found, yes=a.yes, named=str(a.gpus).strip().lower() != "all")
        return sel
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu]
    single = sorted([g for g in found if gpu_problem(g) is None], key=lambda x: (-round(x["vram_gb"]), x["index"]))
    if not single:
        gpu_table(found)
        fail("none of your GPUs can run Strata", "it needs an NVIDIA RTX 20 series or newer (compute capability 7.5+)")
    can = together_ok(found)
    if not can:
        for g in found:                                # said, not asked: a card that works alone but cannot share a model beside a bigger one
            if g["index"] != single[0]["index"] and gpu_problem(g) is None:
                say(f"  GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB) is not used: a card that shares a model needs {SPLIT_MIN_VRAM_GB} GB or more, "
                    f"so the model runs on GPU {single[0]['index']} alone.")
        return [single[0]["index"]]
    return choose_roles(a, found)


def fallback_card(a, gpu, chosen):
    """A model that cannot share its memory across cards keeps ONE card: the one setup always chose (the most VRAM, then the lower number),
    not the Main card of the roles - unless the person named the roles (--gpu-roles), then their Main card."""
    if getattr(a, "_roles_explicit", False) or not getattr(a, "_roles", None):
        return gpu
    return min(chosen, key=lambda g: (-round(g["vram_gb"]), g["index"]))


def card_speeds() -> dict:
    """{card number: SMs x max clock} of the NVIDIA cards (strata_control/gpu_roles.py); {} when it cannot say."""
    from strata_control import gpu_roles
    return gpu_roles.nvidia_speeds()


def roles_cards(found) -> list:
    sp = card_speeds()
    return [{"index": g["index"], "name": g["name"], "vram_gb": g["vram_gb"], "usable": gpu_problem(g) is None, "vendor": "nvidia",
             **({"speed": sp[g["index"]]} if g["index"] in sp else {})} for g in found]


def choose_roles(a, found) -> list:
    """Two or more cards can share the model: the recommended role of each (the best card Main, the others Helper, a card with 8 GB or less
    Off) and ONE question, "Use these roles?".  n takes the person's own: 1=main,0=helper (cards left out are Off).  Returns the card
    numbers in the engine's order (Main last) and keeps the roles in a._roles for the config (docs/MULTI_GPU.md)."""
    from strata_control import gpu_roles
    cards = roles_cards(found)
    roles = dict(gpu_roles.recommend(cards)["roles"])
    say()
    say("  Strata recommends a role for each graphics card. Main does the last steps of every answer; a Helper holds part of the model;")
    say("  Off leaves the card alone (a card with 8 GB or less starts Off beside a bigger card). Details: docs/MULTI_GPU.md.")
    by = {c["index"]: c for c in cards}
    for i in sorted(roles):
        say(f"    GPU {i + 1}  {gpu_roles.short_name(by[i]):<22} {by[i]['vram_gb']:>4.0f} GB   {roles[i].capitalize()}")
    for g in found:
        if gpu_problem(g, together=True) is not None and roles.get(g["index"]) == "off":
            say(f"     (GPU {g['index'] + 1}, {g['name']}: {gpu_problem(g, together=True)})")
    if ask("Use these roles?", ["y", "n"], "y", a.yes or a.check) == "n":
        while True:
            try:
                text = input("Roles, e.g. " + ",".join(f"{i}={r}" for i, r in sorted(roles.items())) + " (cards you leave out are Off): ").strip()
            except EOFError:
                fail("input ended before a setup answer was received", "run setup in a terminal, or pass --yes to accept the recommended roles")
            if not text:
                break
            try:
                roles = gpu_roles.roles_from_text(text, cards)
                break
            except ValueError as e:
                say(f"  {e}")
    roles = gpu_roles.normalize(roles, cards)
    lst = gpu_roles.order(roles, cards)
    if not lst:
        fail("every graphics card is Off", "setup needs one card for Main (the app and `strata launch` can start with all cards Off)")
    a._roles = roles
    a._roles_cards = cards
    ok(gpu_roles.describe(roles, cards))
    return lst


def split_mmap(cfg: dict) -> bool:
    """#364 #384: the low-RAM mode's resident variant (--resident-experts) has no layer split yet.  A config with it
    that runs on several GPUs reads the experts the GPUs do not hold through the OS file cache instead
    (--mmap-experts: the placement those reports measured 1.3-1.6x faster than one GPU), said plainly - the engine
    used to refuse the pair.  #642: the engines from RESIDENT_SPLIT_ENGINE run it on a split, so it is kept.  True
    when the config changed."""
    a = cfg.get("args", [])
    if "--resident-experts" not in a or resident_split():
        return False
    a[a.index("--resident-experts")] = "--mmap-experts"
    warn("the low-RAM mode's resident variant (--resident-experts) has no layer split yet: on several GPUs the experts "
         "the GPUs do not hold are read through the OS file cache (--mmap-experts) instead, and RAM can fill up to 0 "
         "free during long prompts. One GPU keeps them in RAM (steady RAM use): START-HERE --setup, or --gpu N for a "
         "start")
    return True


def model_file(fam: dict, model: str, i: int) -> str:
    """Shard i's file name: the family's pattern, or the model's own (#621: UD-IQ4_XS has three shards, not four)."""
    return MODELS.get(model, {}).get("file", fam["file"]).format(q=model, i=i)


def model_shards(fam: dict, model: str) -> int:
    return MODELS.get(model, {}).get("shards", fam.get("shards", 2))


def budget_model(cfg: dict) -> str:
    """The Unsloth model a config with a RAM budget runs, from its --native shard's name (UD-Q4_K_XL by default)."""
    a = cfg.get("args", [])
    native = Path(a[a.index("--native") + 1]).name.upper() if "--native" in a and a.index("--native") + 1 < len(a) \
        else ""
    return next((m for m, d in MODELS.items() if d.get("budget") and f"-{m}-" in native), "UD-Q4_K_XL")


def unsloth_split_need_gb(model="UD-Q4_K_XL") -> float:
    """#498: the RAM UD-Q4_K_XL needs on several GPUs, where it has no RAM budget (the engine refuses
    --resident-budget-gib with a layer split): its GGUF files and UNSLOTH_RAM_LEFT_GB more (~135 GB).  Measured safe
    at 165 GiB (2x RTX 3090: MemAvailable never under 68 GiB); the 0-free case of #384 was 47 GB with a 70 GB model."""
    return MODELS[model]["download_gb"] + UNSLOTH_RAM_LEFT_GB


def split_budget(cfg: dict, yes: bool = False, explicit: bool = False) -> bool:
    """#498: a UD-Q4_K_XL config (its RAM budget, --resident-budget-gib) started on several GPUs.  The engine refused
    the budget with a layer split (it exited with code 2), so the split runs without it - all the experts loaded into
    RAM at start - where the RAM holds the GGUFs and 24 GB more; else setup says so and asks (#737: a recommendation,
    not a wall - 128 GB ran it fine): a "no" stops, before the config is saved; `explicit` (--gpus) with --yes goes on.
    #642: the engines from RESIDENT_SPLIT_ENGINE keep the resident RAM copy on a split - every card's cache left out
    of it, the rest ranked by the whole expert profile - and the budget is that copy's size, so it is kept, as
    split_mmap keeps --resident-experts.  True when the config changed."""
    a = cfg.get("args", [])
    if "--resident-budget-gib" not in a or resident_split():
        return False
    model = budget_model(cfg)
    need, ram = unsloth_split_need_gb(model), ram_gb()
    if ram < need:
        confirm_risk(f"{model} cannot share its RAM budget across GPUs (the engine has no layer split with it), and "
                     f"without the budget it needs ~{need:.0f} GB of RAM (its GGUF files and {UNSLOTH_RAM_LEFT_GB} GB "
                     f"more); this PC has {ram:.0f} GB. The estimate is a worst case: a 128 GB PC ran it (#737), but "
                     "it may page or stall here.", explicit, yes,
                     f"{model} on several GPUs needs ~{need:.0f} GB of RAM; this PC has {ram:.0f} GB",
                     "start it on one GPU: START-HERE.bat --gpu N (Linux: ./setup.sh --gpu N)")
    i = a.index("--resident-budget-gib")
    del a[i:i + 2]
    ok(f"{model} on several GPUs: no RAM budget (the engine has none with a layer split) - all its experts are "
       "loaded into RAM from the model files at start, and the files pass through the OS file cache (#498)")
    return True


REMOTE_EXPERT_OPT = "--remote-expert-opt"
HELPER_CACHE_FLAGS = ("--expert-cache-device1", "--expert-cache-device2", "--expert-cache-device3")


def recommend_remote_expert_opt(cfg: dict, off: bool = False) -> None:
    """0.1.39b (#578): a config on two or more GPUs with a helper cache gets --remote-expert-opt - the helper expert caches
    (--expert-cache-device1..3) then stay complementary to the main GPU's, return their rows already weighted and skip
    the CPU's activation quantization where no expert is left to it (dual RTX 4090: +63% mixed, +132% code over the
    plain helper path).  The engine uses it only with a helper cache; a layer split runs as before.  A recommendation:
    `off` (setup's --no-remote-expert-opt) or "remote_expert_opt": false in the config keeps it out, and a single-GPU
    config is not touched."""
    if len(gpu_sel(cfg.get("gpu"))) < 2:
        return
    args = cfg.setdefault("args", [])
    if off or cfg.get("remote_expert_opt") is False:
        if REMOTE_EXPERT_OPT in args:
            args.remove(REMOTE_EXPERT_OPT)
        return
    # #1352: --pipeline-windows (the first card starts the next window) is switched off by --remote-expert-opt (the
    # helper caches), so a config that asks for it keeps its pipeline: one of the two, not both (docs/MULTI_GPU.md)
    pw = args.index("--pipeline-windows") if "--pipeline-windows" in args else -1
    if pw >= 0 and pw + 1 < len(args) and args[pw + 1] != "0":
        if REMOTE_EXPERT_OPT in args:
            warn("--pipeline-windows and --remote-expert-opt are both in this config: the engine turns the pipeline off "
                 "beside the helper caches. Keep one - remove --remote-expert-opt for the pipeline "
                 "(docs/MULTI_GPU.md, #1352)")
        else:
            ok("multi-GPU: --pipeline-windows is set, so --remote-expert-opt is not added (the helper caches turn the "
               "pipeline off; docs/MULTI_GPU.md)")
        return
    # #1447: the flag acts only on the helper caches (--expert-cache-device1..3); without one the engine builds nothing
    # from it, and a plain layer split gained a flag that says "helpers" for no reason. Setup adds it only beside a
    # helper cache; a flag already in the config stays (the user's, or an earlier setup's - harmless).
    if not any(a.split("=", 1)[0] in HELPER_CACHE_FLAGS for a in args):
        return
    if REMOTE_EXPERT_OPT not in args:
        args.append(REMOTE_EXPERT_OPT)
        ok("multi-GPU: --remote-expert-opt (helper expert caches complementary to the main GPU's, #578; "
           "--no-remote-expert-opt leaves it out)")


def offer_together(cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """Starting a model set up for one card on a PC with two or more that can share it: asked once (the answer is
    saved in its config)."""
    if isinstance(cfg.get("gpu"), list) or cfg.get("gpus_asked"):
        return cfg
    found = gpus()
    can = together_ok(found)
    if not can:
        return cfg
    # #498: UD-Q4_K_XL's RAM budget had no layer split; without it the RAM must hold the GGUFs and 24 GB more.  #642:
    # the engines from RESIDENT_SPLIT_ENGINE keep the budget on both cards, so any RAM is offered them (one GPU stays
    # the recommendation, the tested setup)
    budget = "--resident-budget-gib" in cfg.get("args", [])
    if budget and not resident_split() and ram_gb() < unsloth_split_need_gb(budget_model(cfg)):
        return cfg
    pair = can[:2]
    cfg["gpus_asked"] = True
    # #364 #384: the resident low-RAM variant stays on one card unless the user says otherwise (its RAM use is steady);
    # #642: the engines from RESIDENT_SPLIT_ENGINE keep it on both cards, so they are offered as for any config
    resident = "--resident-experts" in cfg.get("args", []) and not resident_split()
    say()
    say("  This PC has " + " and ".join(gpu_name(g) for g in pair) + ": Strata can share the model across both.")
    say("  Together they hold about twice the model's experts and read prompts about 20% faster (docs/MULTI_GPU.md).")
    if resident:
        say("  This model runs in the low-RAM mode with its experts kept in RAM, on one GPU (recommended: steady RAM")
        say("  use). On both, the experts the GPUs do not hold are read through the OS file cache instead: faster in")
        say("  two reports (#364, #384), but RAM can fill up to 0 free during long prompts.")
    if budget and resident_split():
        say(f"  This model ({budget_model(cfg)}) runs on one GPU with a RAM budget of its experts (recommended: the tested")
        say("  setup). On both it keeps the budget: each card caches the experts of its own layers, and the RAM holds")
        say("  the hottest of the rest (#642).")
    elif budget:
        say(f"  This model ({budget_model(cfg)}) runs on one GPU with a RAM budget of its experts (recommended: the tested")
        say("  setup). On both it has no budget: all its experts are loaded into RAM at start, which this PC's RAM")
        say("  holds - about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s).")
    short = split_short(pair)             # #448: one card recommended (asked "n" by default), as for --resident
    for g in short:
        say(f"  {split_short_note(g)}.")
    tk = config_toolkit(cfg)
    missing = [g for g in pair if not (engine_runs_on(g) if tk == 13 else engine_runs_on(g, tk))]
    if missing:
        say("  The installed engine has no code for " + ", ".join(g["name"] for g in missing) + ": to use them "
            "together, run START-HERE.bat --setup --gpus " + ",".join(str(g["index"]) for g in pair))
    elif ask("  Use both from now on? (you can change it later: START-HERE.bat --gpu N for one card)",
             ["y", "n"], "n" if resident or short or budget else "y", yes) == "y":
        cfg["gpu"] = [g["index"] for g in pair]
        cfg["layer_split"] = cfg.get("layer_split") or "auto"
        split_mmap(cfg)
        split_budget(cfg)
        recommend_remote_expert_opt(cfg)
        ok("from now on this model runs on " + " + ".join(gpu_name(g) for g in pair))
    else:
        ok("staying on one GPU (START-HERE.bat --gpus " + ",".join(str(g["index"]) for g in pair) + " switches)")
    write_config(cfg_path, cfg)
    return cfg


def gpu_info(pick=None):
    """The GPU Strata runs on: `pick` (nvidia-smi's number) if given, else the one with the most VRAM (ties: the
    lower number).  None when there is no NVIDIA GPU.  The dict also says how many there are ("count")."""
    found = gpus()
    if not found:
        return None
    pick = GPU_PICK if pick is None else pick
    if pick is not None:
        g = next((x for x in found if x["index"] == pick), None)
        if g is None:
            fail(f"there is no GPU {pick}: " + ", ".join(f"{x['index']} = {x['name']}" for x in found))
    else:
        g = max(found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def find_nvcc(below=None):
    """The newest CUDA toolkit's nvcc and its (major, minor); with `below`, the newest older than that version.
    #601: STRATA_NVCC=<path to nvcc> is the only one considered (a newer toolkit beside it that cannot build on this
    PC - CUDA 12.9 with glibc 2.43 - is not taken instead)."""
    pick = os.environ.get("STRATA_NVCC")
    if pick:
        if not Path(pick).exists():
            warn(f"STRATA_NVCC={pick}: no such file; looking for a CUDA toolkit as usual")
        else:
            v = re.search(r"release (\d+)\.(\d+)", out([pick, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and below is not None and ver >= below:
                warn(f"STRATA_NVCC={pick} is CUDA {ver[0]}.{ver[1]}; this build needs one older than "
                     f"{below[0]}.{below[1]}")
                return (None, None)
            return (pick, ver) if ver else (None, None)
    cands = [shutil.which("nvcc")]
    for var in ("CUDA_PATH", "CUDA_HOME"):           # CUDA_HOME: Linux's usual name (#601)
        if os.environ.get(var):
            cands.append(str(Path(os.environ[var]) / "bin" / ("nvcc.exe" if WIN else "nvcc")))
    if WIN:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            cands += [str(p / "bin" / "nvcc.exe") for p in sorted(base.iterdir(), reverse=True)]
    else:
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
        cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"), reverse=True)]   # Arch (#46)
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and (below is None or ver < below) and (best[1] is None or ver > best[1]):
                best = (c, ver)
    return best


def find_vcvars(cuda_v=None):
    """Visual Studio's vcvars64.bat.  #985: CUDA 13.0-13.2 accept Visual Studio 2019 and 2022 only, so a newer one
    (2026 = version 18) is taken only with CUDA 13.3 or newer (`cuda_v`, the toolkit's (major, minor)).

    The version range is CUDA's requirement alone, so a caller with no CUDA (`cuda_v is None`) asks vswhere for no
    range at all: the CPU image encoder is built by MSVC, which is happy with any version (tools/hip/build_windows.bat
    already calls vswhere that way).  With the range applied to every caller, a machine whose only C++ tools are
    Visual Studio 2026 gets None back and nothing can be built there (#881).  STRATA_VCVARS=<vcvars64.bat> picks one
    by hand, for an install vswhere cannot use."""
    override = os.environ.get("STRATA_VCVARS")
    if override:
        v = Path(override)
        return v if v.exists() else None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        return None
    # CUDA 13 accepts Visual Studio 2019 and 2022 only: a newer one (2026 = version 18) installed next to them
    # must not be picked ("unsupported Microsoft Visual Studio version"); with only a newer one there is none
    args = [str(vswhere), "-latest", "-products", "*"]
    if cuda_v is not None:
        upper = "19.0" if tuple(cuda_v) >= (13, 3) else "18.0"
        args += ["-version", f"[17.0,{upper})"]      # 17 = Visual Studio 2022: the sources are C++20, and nvcc silently drops -std=c++20 under 2019 (#3566)
    args += ["-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]
    p = out(args).strip()
    v = Path(p) / "VC/Auxiliary/Build/vcvars64.bat" if p else None
    return v if v and v.exists() else None


CMAKE_MIN = (3, 24)                    # CMakeLists.txt: cmake_minimum_required(VERSION 3.24)


def _tool_version(path, name):
    """(ok, version tuple) of `path --version`: ok False when it does not run (a pip launcher whose Python is gone)."""
    try:
        done = subprocess.run([path, "--version"], capture_output=True, text=True, errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False, ()
    if done.returncode != 0:
        return False, ()
    found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", done.stdout or "")
    return True, tuple(int(x) for x in found.groups() if x) if found else ()


def find_tool(name):
    """The tool the build uses (cmake, ninja): the one pip installed next to THIS Python first (the virtualenv's own, #3822), then PATH,
    then ~/.local/bin.  One that does not run (a pip launcher whose Python is gone) is skipped, and so is a cmake older than the
    one CMakeLists.txt needs while another is found; with only an older one it is returned and the build says so."""
    exe = name + (".exe" if WIN else "")
    candidates = [str(d / exe) for d in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent) if (d / exe).exists()]
    on_path = shutil.which(name)
    if on_path and on_path not in candidates:
        candidates.append(on_path)
    local = Path.home() / ".local" / "bin" / exe
    if local.exists() and str(local) not in candidates:
        candidates.append(str(local))
    older = None
    for c in candidates:
        runs, version = _tool_version(c, name)
        if not runs:
            continue
        if name == "cmake" and version and version[:2] < CMAKE_MIN:
            older = older or c
            continue
        return c
    return older


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ downloads
def drop_archive(z: Path) -> None:
    """An unpacked or refused engine archive and its .done mark go: a refused one kept them, and every later run
    reused it ("already downloaded") instead of the published one (PR #324)."""
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)


# Parallel ranged downloads (0.2.0).  One TCP stream often cannot fill a fast line (a long or lossy route, a CDN that limits
# each connection): download() takes the file in DL_CHUNK pieces over several connections, writes each piece at its own offset,
# and keeps the finished pieces in <name>.part.chunks so a stopped download continues where it was.  Anything that is not a
# ranged file of known length (no Range support, no length, a small file) is one stream, as before.
DL_CONNECTIONS = 16             # STRATA_DL_CONNECTIONS=1 restores one stream (for testers)
DL_CHUNK = 32 << 20             # bytes per piece: what the piece list records and what a failed request loses at most
DL_RUN = 4                      # pieces one request asks for (4 x 32 MiB): a new connection per piece spends its first seconds
                                # in TCP slow start (measured: 8 connections 66 MB/s with a request per piece, 73 with 128 MiB)
DL_MIN_PARALLEL = 256 << 20     # smaller files are one stream
DL_CHUNK_TRIES = 5              # tries per request (1, 2, 4, 8 s apart) before the round is given up and run again
DL_ROUNDS = 30                  # rounds (10 s apart) before the download is given up, like the 30 tries of one stream
DL_BACKOFF = 1.0                # seconds before the second try of a request, doubled each time (tests set 0)
DL_SAVE_EVERY = 5.0             # seconds between writes of the progress file


def dl_connections() -> int:
    """How many connections a big download uses: STRATA_DL_CONNECTIONS, else DL_CONNECTIONS (1 = one stream)."""
    raw = (os.environ.get("STRATA_DL_CONNECTIONS") or "").strip()
    try:
        n = int(raw) if raw else DL_CONNECTIONS
    except ValueError:
        n = DL_CONNECTIONS
    return max(1, min(n, 32))


class _NoRange(Exception):
    """The server does not answer a byte range with that range (no 206, or another range): one stream instead."""


class _DiskError(Exception):
    """The .part file cannot be written (disk full, no permission): retrying would not help."""


class _PieceFailed(Exception):
    """A piece failed DL_CHUNK_TRIES times."""


class _Changed(Exception):
    """The server's file is not the one the first pieces came from (another ETag): what was written cannot be trusted."""


_CONTENT_RANGE = re.compile(r"\s*bytes (\d+)-(\d+)/(\d+)\s*$")


def _etag(headers) -> str | None:
    """The object identity a response names: ETag without the weak mark and quotes; None when it sends none."""
    raw = (headers.get("ETag") or "").strip()
    if raw[:2] in ("W/", "w/"):
        raw = raw[2:]
    return raw.strip('"') or None


def _check_ranged(r, pos: int, end: int, total: int, etag: str | None):
    """A ranged answer is trusted only when it is exactly the bytes asked for: status 206, Content-Range `pos-end/total`
    (a 200 with the whole file, another range or another total is the server not honouring the request), a Content-Length
    that fits, and the same object (ETag) as the first answer.  _NoRange: this server cannot be used with ranges;
    OSError: a bad answer worth asking again (another edge)."""
    if r.status != 206:
        raise _NoRange(f"HTTP {r.status}")
    m = _CONTENT_RANGE.match(r.headers.get("Content-Range") or "")
    if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (pos, end, total):
        raise _NoRange(f"Content-Range {(r.headers.get('Content-Range') or '')!r} for bytes {pos}-{end}/{total}")
    cl = (r.headers.get("Content-Length") or "").strip()
    if cl and cl.isdigit() and int(cl) != end - pos + 1:
        raise OSError(f"Content-Length {cl} for {end - pos + 1} bytes asked")
    got = _etag(r.headers)
    if etag and got and got != etag:
        raise OSError("the server answered with another version of the file")


def _write_all(f, pos: int, data):
    """Write `data` at `pos`; a raw unbuffered file may take fewer bytes than it was given, so go on until all are down."""
    view = memoryview(data)
    f.seek(pos)
    done_n = 0
    while done_n < len(view):
        n = f.write(view[done_n:])
        if not n:
            raise OSError("the disk took no more bytes")
        done_n += n


def _read_exact(f, pos: int, n: int) -> bytes:
    f.seek(pos)
    parts = []
    while n > 0:
        b = f.read(n)
        if not b:
            break
        parts.append(b)
        n -= len(b)
    return b"".join(parts)


def _url_ttl(url: str):
    """Seconds a signed URL (the CDN's, after the redirect) is still valid by its own query, else None."""
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    try:
        if "Expires" in q:
            ttl = float(q["Expires"][0]) - time.time()
        elif "X-Amz-Expires" in q and "X-Amz-Date" in q:
            t0 = calendar.timegm(time.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ"))
            ttl = t0 + float(q["X-Amz-Expires"][0]) - time.time()
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return ttl if 0 < ttl < 30 * 86400 else None       # a PC clock that is far off says nothing: a 403 re-resolves


class _Source:
    """The URL the pieces are read from: the redirect (Hugging Face answers with a signed, time-limited CDN address) is
    resolved once and reused; it is resolved again a little before it expires and after a 401/403/410."""

    def __init__(self, url: str, total: int):
        self.url, self.total = url, total
        self.final, self.deadline = None, None
        self.etag = None                                 # the object the first answer named; every later answer must match
        self.lock = threading.Lock()

    def resolve(self, stale=None) -> str:
        with self.lock:
            if stale is not None and self.final is not None and self.final != stale:
                return self.final                        # another connection already got a fresh address
            req = urllib.request.Request(self.url, headers={"User-Agent": "strata-setup", "Range": "bytes=0-0"})
            with _urlopen(req, timeout=60) as r:
                cr = (r.headers.get("Content-Range") or "").strip()
                if r.status != 206 or cr != f"bytes 0-0/{self.total}":
                    raise _NoRange(f"HTTP {r.status}, Content-Range {cr!r}")
                tag = _etag(r.headers)
                if self.etag and tag and tag != self.etag:
                    raise _Changed(f"the file on the server changed while it was downloaded ({self.etag[:12]} -> {tag[:12]})")
                self.etag = self.etag or tag
                r.read(1)
                final = r.geturl() or self.url
            ttl = _url_ttl(final)
            self.final = final
            self.deadline = time.monotonic() + max(ttl - 120, 0) if ttl else None
            return final

    def get(self) -> str:
        with self.lock:
            cur, expired = self.final, self.deadline is not None and time.monotonic() >= self.deadline
        return self.resolve(stale=cur) if cur is None or expired else cur


class _Pieces:
    """The shared state of one parallel download: finished pieces, bytes of the pieces in flight, the stop flags."""

    def __init__(self, total: int, done: set, conns: int = 1):
        self.total, self.n, self.conns = total, -(-total // DL_CHUNK), conns
        self.done, self.flight = set(done), {}
        self.lock = threading.Lock()
        self.stop = threading.Event()                    # the whole download stops (Ctrl-C, a fatal error)
        self.abort_round = threading.Event()             # a piece failed for good: end this round, run the next
        self.fatal, self.errors = None, []

    def size(self, i: int) -> int:
        return min((i + 1) * DL_CHUNK, self.total) - i * DL_CHUNK

    def add(self, i: int, n: int):
        with self.lock:
            self.flight[i] = self.flight.get(i, 0) + n

    def finish(self, i: int):
        with self.lock:
            self.flight.pop(i, None)
            self.done.add(i)

    def have(self) -> int:
        with self.lock:
            return sum(self.size(i) for i in self.done) + sum(self.flight.values())

    def snapshot(self) -> set:
        with self.lock:
            return set(self.done)


def _ranges(done) -> list:
    out = []
    for i in sorted(done):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def _prefix_bytes(done: set, total: int) -> int:
    """Bytes of the unbroken run of finished pieces from the start of the file."""
    i = 0
    while i in done:
        i += 1
    return min(i * DL_CHUNK, total)


def _load_pieces(part: Path, total: int) -> set:
    """The finished pieces of an earlier run.  <name>.part.chunks lists them; a .part without it comes from one stream
    (an older setup, or STRATA_DL_CONNECTIONS=1): its bytes are a run from the start, so those pieces are finished."""
    side = part.with_name(part.name + ".chunks")
    n = -(-total // DL_CHUNK)
    try:
        size = part.stat().st_size if part.is_file() else None
    except OSError:
        size = None
    if side.is_file():
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            if data.get("v") == 1 and data.get("total") == total and data.get("chunk") == DL_CHUNK and size is not None:
                done = {i for a, b in data["done"] for i in range(int(a), min(int(b), n))}
                return {i for i in done if min((i + 1) * DL_CHUNK, total) <= size}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        side.unlink(missing_ok=True)                    # not this file's list: start the file over
        part.unlink(missing_ok=True)
        return set()
    if size is None:
        return set()
    if size > total:
        part.unlink(missing_ok=True)
        return set()
    return set(range(n)) if size >= total else set(range(size // DL_CHUNK))


def _save_pieces(part: Path, total: int, done: set) -> bool:
    """Flush the data first, then record the pieces: a list never names bytes the disk does not have.  False when the
    list could not be written (an older list stays, which only costs re-downloading a few pieces after a stop)."""
    try:
        with open(part, "r+b") as g:
            os.fsync(g.fileno())
        side = part.with_name(part.name + ".chunks")
        tmp = side.with_name(side.name + ".tmp")
        tmp.write_text(json.dumps({"v": 1, "total": total, "chunk": DL_CHUNK, "done": _ranges(done)}), encoding="utf-8")
        with open(tmp, "r+b") as g:
            os.fsync(g.fileno())
        atomic.replace(tmp, side)
        return True
    except OSError:
        return False


def _claim_part(part: Path, url: str, etag: str | None = None):
    """Record where a .part's bytes come from (<name>.part.src: the address asked, and the file's ETag when the server
    names one).  A .part written from another address (the other mirror, another revision) or from another version of
    the file is dropped with its piece list: two sources are not mixed inside one file, even when they should hold the
    same bytes.  A .part from an older setup has no record and is kept as before."""
    src = part.with_name(part.name + ".src")
    try:
        old = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else None
    except (OSError, ValueError):
        old = {}
    if old is not None:
        same = isinstance(old, dict) and old.get("url") == url and not (old.get("etag") and etag and old.get("etag") != etag)
        if not same:
            part.unlink(missing_ok=True)
            part.with_name(part.name + ".chunks").unlink(missing_ok=True)
            src.unlink(missing_ok=True)
            old = None
    if old is None or (etag and not old.get("etag")):
        try:
            src.write_text(json.dumps({"url": url, "etag": etag}), encoding="utf-8")
        except OSError:
            pass


def _single_ready(part: Path):
    """Before one stream continues a .part: keep only the unbroken run from the start of the file (what a parallel
    download left may have gaps behind it) and drop the list."""
    side = part.with_name(part.name + ".chunks")
    if not side.is_file():
        return
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        keep = _prefix_bytes({i for a, b in data["done"] for i in range(int(a), int(b))}, int(data["total"]))
        if data.get("chunk") != DL_CHUNK:
            keep = 0
    except (OSError, ValueError, TypeError, KeyError):
        keep = 0
    try:
        if part.is_file():
            with open(part, "r+b") as g:
                g.truncate(min(keep, part.stat().st_size))
    except OSError:
        part.unlink(missing_ok=True)
    side.unlink(missing_ok=True)


def _take_run(st: _Pieces, todo: list) -> list:
    """The next request's pieces: up to DL_RUN neighbours from the front of `todo`; fewer near the end, so the last pieces
    are spread over every connection and no single slow one holds the finish."""
    with st.lock:
        if not todo:
            return []
        n = max(1, min(DL_RUN, len(todo) // (2 * st.conns)))
        run = [todo.pop(0)]
        while len(run) < n and todo and todo[0] == run[-1] + 1:
            run.append(todo.pop(0))
        return run


def _fetch_run(src: _Source, st: _Pieces, f, run: list):
    """One ranged request for the neighbouring pieces in `run`, written at their offsets; each piece counts as finished
    when its last byte is down.  Every answer is checked to be exactly the range asked for (status, Content-Range, length,
    ETag); a connection that breaks continues from the byte it reached."""
    start = run[0] * DL_CHUNK
    end = min((run[-1] + 1) * DL_CHUNK, st.total) - 1
    pos, tries = start, 0
    with st.lock:
        for i in run:
            st.flight[i] = 0
    while True:
        if st.stop.is_set():
            return
        url, err, wait, ra = None, None, None, 0.0
        attempt_pos = pos
        try:
            url = src.get()
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={pos}-{end}"})
            with _urlopen(req, timeout=60) as r:
                _check_ranged(r, pos, end, st.total, src.etag)
                while pos <= end:
                    if st.stop.is_set():
                        return
                    cur = pos // DL_CHUNK
                    cur_end = min((cur + 1) * DL_CHUNK, st.total)             # one past the piece's last byte
                    b = r.read(min(1 << 20, cur_end - pos))
                    if not b:
                        raise IOError("the connection closed early")
                    try:
                        _write_all(f, pos, b)
                    except OSError as e:
                        raise _DiskError(str(e)) from e
                    pos += len(b)
                    st.add(cur, len(b))
                    if pos == cur_end:
                        st.finish(cur)
                        if pos <= end and st.abort_round.is_set():            # a piece failed for good: stop at a piece edge
                            with st.lock:
                                for i in run:
                                    if i not in st.done and st.flight.get(i) == 0:
                                        st.flight.pop(i, None)
                            return
            return
        except (_NoRange, _DiskError, _Changed):
            raise
        except urllib.error.HTTPError as e:
            err = e
            if e.code in (401, 403, 410) and url is not None:       # the signed address ran out: ask for a new one
                try:
                    src.resolve(stale=url)
                    wait = 0
                except (_NoRange, _Changed):
                    raise
                except (OSError, http.client.HTTPException):
                    pass
            elif e.code in (429, 503):
                try:
                    ra = min(float(e.headers.get("Retry-After") or 0), 60)
                except (TypeError, ValueError):
                    pass
        except (OSError, http.client.HTTPException) as e:
            err = e
        # A try that moved forward is not a failed one.  The request goes on from the byte it reached: what came before the
        # break is exactly the range that was asked for (the answer's status, Content-Range, length and ETag were checked and
        # every byte was written in full), and writing a piece again from its start would make a route that resets more often
        # than every DL_CHUNK bytes unable to finish any piece.  The whole-file SHA-256 and the repair are the net behind it.
        tries = 1 if pos > attempt_pos else tries + 1
        if tries >= DL_CHUNK_TRIES:
            raise _PieceFailed(str(err))
        st.stop.wait(max(DL_BACKOFF * 2 ** (tries - 1), ra) if wait is None else wait)


def _piece_worker(src: _Source, st: _Pieces, todo: list, part: Path):
    try:
        with open(part, "r+b", buffering=0) as f:
            while not (st.stop.is_set() or st.abort_round.is_set()):
                run = _take_run(st, todo)
                if not run:
                    return
                _fetch_run(src, st, f, run)
    except _PieceFailed as e:
        st.errors.append(e)
        st.abort_round.set()
    except (_NoRange, _DiskError, _Changed) as e:
        st.fatal = st.fatal or e
        st.stop.set()
    except OSError as e:                                # the .part itself could not be opened
        st.fatal = st.fatal or _DiskError(str(e))
        st.stop.set()


class StallNote:
    """A download that gets no byte for a while says so (#3567): "stalled at 69%" instead of a silent line that never moves. `check` is called
    often with the bytes so far; it answers a sentence at most once a minute while nothing arrives, and None otherwise."""

    AFTER = 120.0

    def __init__(self, now: float = 0.0):
        self.have, self.since, self.said = -1, now, 0.0

    def check(self, have: int, total: int, now: float) -> str | None:
        if have != self.have:
            self.have, self.since, self.said = have, now, 0.0
            return None
        idle = now - self.since
        if idle < self.AFTER or now - self.said < 60:
            return None
        self.said = now
        return (f"no data for {int(idle // 60)} min at {100 * have // max(1, total)}%: the connection is stuck. It keeps trying; to stop, press Ctrl+C, "
                "and run the same command again: the download continues where it stopped")


def parallel_download(url: str, part: Path, total: int, label: str, conns: int, name: str) -> bool:
    """Fill `part` (total bytes) over `conns` ranged connections.  True when every byte is there; False when the server
    does not do ranges (the caller reads the file in one stream)."""
    src = _Source(url, total)
    try:
        src.resolve()
    except _NoRange:
        return False
    except (OSError, http.client.HTTPException):
        return False                                    # the one-stream path has its own retries and messages
    part.parent.mkdir(parents=True, exist_ok=True)
    _claim_part(part, url, src.etag)                    # a .part from another address or version is dropped here
    done = _load_pieces(part, total)
    st = _Pieces(total, done, conns)
    if not part.exists():
        part.touch()
    if not part.with_name(part.name + ".chunks").is_file() and not _save_pieces(part, total, done):
        return False                                    # no piece list can be kept here: the pieces would not be told from holes
    last_print = last_save = 0.0
    stall = StallNote(time.time())

    def report(force=False):
        nonlocal last_print, last_save
        now = time.time()
        if now - last_print > 2 or force:
            last_print = now
            have = st.have()
            print(f"\r  {label}: {have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)   ", end="", flush=True)
            note = stall.check(have, total, now)
            if note:
                print()
                warn(note)
        if now - last_save > DL_SAVE_EVERY:
            last_save = now
            _save_pieces(part, total, st.snapshot())

    try:
        for rnd in range(DL_ROUNDS):
            todo = [i for i in range(st.n) if i not in st.done]
            if not todo:
                break
            st.abort_round.clear()
            st.errors.clear()
            threads = [threading.Thread(target=_piece_worker, args=(src, st, todo, part), daemon=True)
                       for _ in range(min(conns, len(todo)))]
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(0.25)
                report()
            if st.fatal is not None:
                break
            if st.errors:
                print()
                warn(f"download interrupted ({st.errors[0]}); retrying in 10 s ...")
                _save_pieces(part, total, st.snapshot())
                time.sleep(10)
    except BaseException:
        st.stop.set()                                   # Ctrl-C: let the connections stop, keep what is finished
        _save_pieces(part, total, st.snapshot())
        raise
    report(force=True)
    print()
    if isinstance(st.fatal, _Changed):
        part.unlink(missing_ok=True)
        part.with_name(part.name + ".chunks").unlink(missing_ok=True)
        part.with_name(part.name + ".src").unlink(missing_ok=True)
        fail(f"{name} changed on the server while it was downloading: the part that was there was deleted",
             "run it again (the download starts over with the file that is there now)")
    if isinstance(st.fatal, _DiskError):
        _save_pieces(part, total, st.snapshot())
        fail(f"cannot write {part.name}: {st.fatal}", "free some disk space (or pick another --models-dir) and run it again")
    if isinstance(st.fatal, _NoRange):
        _save_pieces(part, total, st.snapshot())
        return False                                    # the server stopped doing ranges: one stream from the unbroken start
    if len(st.done) < st.n:
        _save_pieces(part, total, st.snapshot())
        have = st.have()
        fail(f"could not finish downloading {name}: {have:,} of {total:,} bytes",
             "check your internet connection and run it again (the download resumes where it stopped)")
    try:
        with open(part, "r+b") as g:
            g.truncate(total)
            os.fsync(g.fileno())
    except OSError:
        pass
    part.with_name(part.name + ".chunks").unlink(missing_ok=True)
    return True


def _repair_piece(src: _Source, st: _Pieces, f, i: int, stats: dict):
    """Read piece `i` from the server in 1 MiB blocks and compare each with what is on the disk; a block that differs is
    written again.  A broken connection asks again from the block it reached (every block is final once compared)."""
    pos, end = i * DL_CHUNK, min((i + 1) * DL_CHUNK, st.total) - 1
    tries = 0
    while pos <= end:
        if st.stop.is_set():
            return
        url, wait, ra = None, None, 0.0
        before = pos
        try:
            url = src.get()
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={pos}-{end}"})
            with _urlopen(req, timeout=60) as r:
                _check_ranged(r, pos, end, st.total, src.etag)
                while pos <= end:
                    if st.stop.is_set():
                        return
                    b = r.read(min(1 << 20, end + 1 - pos))
                    if not b:
                        raise IOError("the connection closed early")
                    try:
                        if _read_exact(f, pos, len(b)) != b:
                            _write_all(f, pos, b)
                            with st.lock:
                                stats["bytes"] += len(b)
                                stats["pieces"].add(i)
                    except OSError as e:
                        raise _DiskError(str(e)) from e
                    pos += len(b)
                    st.add(i, len(b))
            return
        except (_NoRange, _DiskError, _Changed):
            raise
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 410) and url is not None:
                try:
                    src.resolve(stale=url)
                    wait = 0
                except (_NoRange, _Changed):
                    raise
                except (OSError, http.client.HTTPException):
                    pass
            elif e.code in (429, 503):
                try:
                    ra = min(float(e.headers.get("Retry-After") or 0), 60)
                except (TypeError, ValueError):
                    pass
            err = e
        except (OSError, http.client.HTTPException) as e:
            err = e
        tries = 1 if pos > before else tries + 1
        if tries >= DL_CHUNK_TRIES:
            raise _PieceFailed(str(err))
        st.stop.wait(max(DL_BACKOFF * 2 ** (tries - 1), ra) if wait is None else wait)


def repair_download(url: str, path: Path, total: int, label: str, conns: int):
    """A finished file whose SHA-256 is wrong is not thrown away: read it again from the server piece by piece, compare
    each block with the disk and write only the blocks that differ.  Returns (damaged pieces, pieces), or None when the
    server cannot do it (no ranges, or the file there is another one); the caller checks the SHA-256 again."""
    if not total or path.stat().st_size != total:
        return None
    src = _Source(url, total)
    try:
        src.resolve()
    except (_NoRange, _Changed, OSError, http.client.HTTPException):
        return None
    st = _Pieces(total, set(), conns)
    stats = {"bytes": 0, "pieces": set()}
    todo = list(range(st.n))
    errors = []

    def worker():
        try:
            with open(path, "r+b", buffering=0) as f:
                while not st.stop.is_set():
                    with st.lock:
                        if not todo:
                            return
                        i = todo.pop(0)
                    _repair_piece(src, st, f, i, stats)
                    st.finish(i)
        except (_PieceFailed, _NoRange, _DiskError, _Changed, OSError) as e:
            errors.append(e)
            st.stop.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(conns, st.n))]
    for t in threads:
        t.start()
    last = 0.0
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.25)
            if time.time() - last > 2:
                last = time.time()
                have = st.have()
                print(f"\r  {label}: {have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)   ", end="", flush=True)
    except BaseException:
        st.stop.set()
        raise
    print()
    if errors:
        warn(f"could not compare {path.name} with the server again ({errors[0]})")
        return None
    try:
        with open(path, "r+b") as g:
            os.fsync(g.fileno())
    except OSError:
        pass
    return len(stats["pieces"]), st.n


def download_single(url: str, part: Path, total: int, label: str):
    """One stream with a Range resume from the size of `part` (the way every download worked before parallel pieces).
    A resumed answer is used only when it is a 206 that starts at the byte asked for; anything else starts the file over."""
    _single_ready(part)
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:                    # stopped after the last byte, before the rename: nothing to
            break                                      # ask for (a range past the end is a 416, retried 30 times)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with _urlopen(req, timeout=60) as r:
                if have:
                    m = _CONTENT_RANGE.match(r.headers.get("Content-Range") or "")
                    if r.status != 206 or not m or int(m.group(1)) != have or (total and int(m.group(3)) != total):
                        if r.status != 200:              # another range: its bytes are not the file's; ask again from the start
                            part.unlink(missing_ok=True)
                            raise OSError(f"the server answered another range than bytes {have}- (HTTP {r.status})")
                        have = 0                         # the server ignored the range and sent the whole file: start over
                elif r.status not in (200, 206):
                    raise OSError(f"HTTP {r.status}")
                with open(part, "ab" if have else "wb") as f:
                    last = 0.0
                    while True:
                        b = r.read(8 << 20)
                        if not b:
                            break
                        if total and have + len(b) > total:
                            raise OSError("the server sent more bytes than the file has")
                        f.write(b)
                        have += len(b)
                        if time.time() - last > 2:
                            last = time.time()
                            size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total                                 else f"{have / 1e6:7.1f} MB"
                            print(f"\r  {label}: {size}   ", end="", flush=True)
                    f.flush()
                    os.fsync(f.fileno())
            print()
            if not total or have >= total:
                break
        except (OSError, http.client.HTTPException) as e:
            print()
            if part.exists() and total and part.stat().st_size > total:
                part.unlink(missing_ok=True)             # more than the file has can only be wrong bytes
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)


def range_total(url: str) -> int:
    """The file's length from a one-byte ranged GET (Content-Range: bytes 0-0/N), for a server whose HEAD gives none
    (ModelScope's: #1915); 0 when it does not say.  With the length known the file comes over several connections."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": "bytes=0-0"})
        with _urlopen(req, timeout=60) as r:
            m = re.match(r"\s*bytes 0-0/(\d+)\s*$", r.headers.get("Content-Range") or "")
            return int(m.group(1)) if r.status == 206 and m else 0
    except (OSError, http.client.HTTPException):
        return 0


def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    acquisition.require(what or dst.name)
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    ms = ms_file(url) if model_source() == "modelscope" else None
    if ms is not None:
        if reachable(ms_url(*ms), timeout=30):
            url = ms_url(*ms)
        else:
            warn(f"{what or dst.name}: ModelScope does not answer; downloading it from {hf_endpoint()}")
            ms = None
    part = dst.with_name(dst.name + ".part")
    total = 0
    # the file's pinned size and SHA-256, when it has them (ModelScope's copy is checked against the same pin)
    pin = REPO_PINS.get(ms[0], {}).get(ms[1].rsplit("/", 1)[-1]) if ms is not None else file_pin(url)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(_urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404 and hf_unpinned(url) != url:  # the pinned revision is gone from the repository
                if not allow_unpinned():
                    fail(f"the pinned version of {what or dst.name} is no longer on Hugging Face",
                         "update Strata: a newer version pins the files that are there now")
                warn(f"{what or dst.name}: not at the pinned revision any more; downloading the repository's "
                     "current file (not checked against a SHA-256)")
                url = hf_unpinned(url)
                pin = None
                continue
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
        except tls.CertificateProblem as e:           # the same answer five times: no retry
            fail(f"cannot download from {url.split('/')[2]}: its security certificate was not accepted", str(e))
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})",
                     "check your internet connection and run it again" + source_hint(url))
            time.sleep(5)
    if not total:
        total = range_total(url)
    meta = ms_meta(*ms) if ms is not None else None
    if not total:                                  # published size also governs partial-file completion
        total = pin[0] if pin else (meta[0] if meta else 0)
    if pin and total and total != pin[0]:              # not the pinned file: stop before the download, not after
        fail(f"{what or dst.name} is {total:,} bytes on {url.split('/')[2]}, not the pinned {pin[0]:,}",
             "update Strata (the file there has changed since this version pinned it)")
    label = what or dst.name
    conns = dl_connections()
    fixed = pin if pin else (meta if meta is not None and meta[1] else None)
    again = (lambda: repair_download(url, dst, total, label, conns)) if total >= DL_MIN_PARALLEL and conns > 1 else None
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an older setup (no mark yet), or stopped before the check
        if fixed:                                      # a file that has a fixed hash is checked, not just marked
            verify_sha256(dst, *fixed, repair=again)
        else:
            mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    if not (total >= DL_MIN_PARALLEL and conns > 1 and parallel_download(url, part, total, label, conns, dst.name)):
        _claim_part(part, url)
        download_single(url, part, total, label)
    if total and part.stat().st_size != total:
        fail(f"could not finish downloading {dst.name}: {part.stat().st_size:,} bytes on disk, the server says {total:,}",
             "check your internet connection and run it again (the download resumes where it stopped)")
    try:
        atomic.replace(part, dst)                      # waits a moment for a virus scanner or indexer that holds the file
    except OSError as e:
        fail(f"cannot finish {dst.name}: {e}", "close any program that has the file open and run setup again")
    for side in (".chunks", ".src"):
        part.with_name(part.name + side).unlink(missing_ok=True)
    if pin:
        verify_sha256(dst, *pin, repair=again)         # the fixed hash: a wrong file is repaired, else deleted and setup stops
    elif meta is not None and meta[1]:
        verify_sha256(dst, meta[0], meta[1], repair=again)   # ModelScope's published hash; kept in the finish mark
    else:
        if ms is not None:
            warn(f"{what or dst.name}: ModelScope publishes no SHA-256 for it: the file is not verified")
        mark(dst)
    ok(f"{what or dst.name} downloaded")


def whole_shard(s: Path) -> bool:
    """A shard as long as its own tensor directory says (check_shards' test, without stopping setup)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(s)
        return s.stat().st_size >= g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors),
                                                     default=0)
    except (OSError, ValueError, struct.error):
        return False


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_dir_shards(folder: Path, fam: dict, model: str) -> list[Path]:
    """--gguf-dir's shards (#305): every -0000i-of-0000N file of the model, N read from the first shard's name as the
    engine and tools/iq_pack.py do.  The published name first; else the one first shard in the folder whose name has
    the size in it (an upload split or named differently: -00001-of-00003, Unsloth's ...-00001-of-00004.gguf).  A
    missing shard is check_shards' error later, as before."""
    first = folder / model_file(fam, model, 1)
    if not first.exists():
        found = sorted(p for p in folder.glob("*-00001-of-*.gguf") if SHARD_NAME.search(p.name))
        mine = [p for p in found if model.lower() in p.name.lower()]
        pick = mine if mine else found
        if len(pick) == 1:
            first = pick[0]
    m = SHARD_NAME.search(first.name)
    total = int(m.group(2)) if m else 1
    if not m or total < 1:
        return [first]
    stem = first.name[:m.start()]
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, total)) for i in range(1, total + 1)]


# #444: the quantization in a GGUF's name (Unsloth's UD-IQ3_XXS, a K-quant's Q2_K_XL, a GSQ-RCO IQ3_S, ...)
GGUF_QUANT = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))"
                        r"(?=-\d{5}-of-\d{5}\.gguf$|\.gguf$)", re.I)
SUPPORTED_GGUFS = ("Strata runs ISTA-DASLab's GSQ-RCO files (Qwen3.8-Flash-Next Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift "
                   "1.5's; the Coder's IQ1_M) and Unsloth's UD-IQ4_XS, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 "
                   "only: other GGUFs (Unsloth's UD-IQ3_XXS or UD-Q2_K_XL, other K-quant files) cannot be used")


def gguf_unsupported(name: str) -> str | None:
    """#444: the quantization a GGUF's name says, when it is one Strata cannot run (not a setup size); else None."""
    m = GGUF_QUANT.search(name)
    return m.group(1) if m and m.group(1).upper() not in MODELS and not name.lower().startswith("mmproj") else None


def gguf_choice(name: str) -> tuple | None:
    """#444: (--family, --model) whose published first shard this file is, or None (the Coder's IQ1_M is named like
    the original's sizes: the size tells them apart)."""
    for f, d in FAMILIES.items():
        if d.get("ready"):                             # ready-made packs (NVFP4): no GGUF shards to match
            continue
        for m in MODELS:
            if f in MODELS[m].get("families", ("qwen", "swift")) and name == model_file(d, m, 1):
                return f, m
    return None


def gguf_dir_problem(folder: Path, first: Path, fam: dict, model: str) -> tuple | None:
    """#444: (message, hint) when --gguf-dir has no file setup can use for this choice: the chosen shard is a GGUF
    Strata cannot run (Unsloth's UD-IQ3_XXS taken for IQ3_XXS by its name), or it is missing and the folder holds
    other GGUFs - then the hint names the --family/--model of the usable ones.  None otherwise (a missing shard in a
    folder without GGUFs stays check_shards' "missing")."""
    bad = gguf_unsupported(first.name) if first.exists() else None
    if bad:
        return f"{first.name} is {bad}, a GGUF Strata cannot run", SUPPORTED_GGUFS
    if first.exists():
        return None
    firsts = sorted(p.name for p in folder.glob("*.gguf")
                    if not p.name.lower().startswith("mmproj") and (not SHARD_NAME.search(p.name)
                                                                     or SHARD_NAME.search(p.name).group(1) == "00001"))
    usable = list(dict.fromkeys(c for c in map(gguf_choice, firsts) if c))
    unusable = [n for n in firsts if gguf_unsupported(n)]
    if not usable and not unusable:
        return None
    hint = SUPPORTED_GGUFS
    if unusable:
        hint += ".\n       Not usable here: " + ", ".join(unusable)
    if usable:
        hint += ".\n       Usable here: " + ", ".join(f"--family {f} --model {m}" for f, m in usable)
    return f"{folder} has no {fam['title']} {model} file", hint


def _hash_file(s: Path, size: int) -> str:
    h = hashlib.sha256()
    seen, last = 0, 0.0
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            seen += len(b)
            if time.time() - last > 2 and size >= (1 << 30):    # a figure for the app and the Terminal: hashing 60 GB takes minutes
                last = time.time()
                print(f"\r  checking {s.name}: {seen / 1e9:6.2f} / {size / 1e9:.2f} GB ({100 * seen // size}%)   ", end="", flush=True)
    if seen >= (1 << 30):
        print()
    return h.hexdigest()


def own_gguf_is_whole(path: Path) -> bool:
    """The person's own GGUF reads as a whole file: its header parses and every tensor lies inside the file (a cut-short copy fails this)."""
    try:
        from strata_control import gguf_preflight
        gguf_preflight.inspect_metadata(Path(path))
        return True
    except Exception:                                  # noqa: BLE001 - anything unreadable is "not whole"
        return False


def verify_sha256(s: Path, size: int, sha: str, own: bool = False, repair=None, also: tuple = ()) -> None:
    """A shard's size and SHA-256 against the pinned values (the Unsloth file); the result is kept in its finish mark,
    so the ~5 minutes of hashing 111 GB happen once.  A wrong file is deleted, so the next run downloads it again.
    `repair` (a downloaded file's own re-reader, see repair_download): the wrong file is first compared with the server
    piece by piece and only the damaged pieces are written again, then it is checked a second time.
    `also`: other SHA-256 values that are the same file for Strata's purposes (the Engram ISTA-DASLab converted in place on 2026-10-27, DSV41_SHA_ALSO)."""
    m = s.with_name(s.name + ".done")
    if m.exists() and any(f"sha256 {x}" in m.read_text(encoding="utf-8", errors="replace") for x in (sha, *also)):
        return
    have = s.stat().st_size if s.exists() else -1
    if own and have > 0 and have != size and own_gguf_is_whole(s):
        # a fine-tune or another build with the layout of the published file (#3777: 96 bytes apart): recommend, never refuse
        warn(f"{s.name} is {have:,} bytes; the published file is {size:,}. It is a complete GGUF file, so Strata uses it as it is; "
             "it may be another version or a fine-tune, so check a few answers before you rely on it")
        return
    if have != size:
        if own:                                        # --gguf-dir: the person's own file is never deleted by setup
            fail(f"{s.name} is {have:,} bytes; the published file is {size:,}",
                 "it may be cut short or another version; your file was left as it is. Copy the whole published file into the folder")
        fail(f"{s.name} is {have:,} bytes, not {size:,}", "delete it and run setup again (the download restarts)")
    say(f"  checking {s.name} (SHA-256, {size / 1e9:.1f} GB) ...")
    digest = _hash_file(s, size)
    if digest != sha and repair is not None and not own:
        say(f"  Some parts of {s.name} were damaged; downloading only those parts again.")
        res = repair()
        if res is not None:
            say(f"  {res[0]} of {res[1]} parts were different and have been written again; checking the whole file again ...")
            digest = _hash_file(s, size)
        else:
            say("  The damaged parts could not be found this way.")
    if digest != sha and digest in also:
        say(f"  {s.name} is the converted version ISTA-DASLab published on 2026-10-27 (the same tables in one tensor); it is used as it is")
        mark(s, f"sha256 {digest}")
        return
    if digest != sha:
        if own and own_gguf_is_whole(s):
            warn(f"{s.name} is not the published file (SHA-256 {digest[:12]}..., expected {sha[:12]}...) but it is a complete GGUF file, so "
                 "Strata uses it as it is; it may be another version or a fine-tune, so check a few answers before you rely on it")
            return
        if own:
            fail(f"{s.name} does not match the published file (SHA-256 {digest}, expected {sha})",
                 "it may be damaged or another version; your file was left as it is. Copy the whole published file into the folder")
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({digest}, expected {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def ready_files(model: str) -> list:
    """A ready-made pack's files (NVFP4_REPOS): (role, repo, revision, path, bytes, sha256) each."""
    repo = MODELS[model]["repo"]
    t = NVFP4_REPOS[repo]
    return [(role, repo, t["revision"], path, size, sha) for path, (role, size, sha) in t["files"].items()]


def ready_needed(model: str, models_dir: Path) -> list:
    """ready_files without the checkpoint's tensors once a prepared model (MODELS[..]["prepare"]) is prepared: the
    engine reads the prepared files only, so the ~95 GB of safetensors need not stay (or come back) on the disk."""
    files = ready_files(model)
    if MODELS[model].get("prepare") and prepared_done(model, models_dir):
        files = [f for f in files if not f[3].startswith("tensors/")]
    return files


def ready_where(models_dir: Path, f) -> Path:
    """Where one of them goes: <models>/<owner>--<repo>/<its path in the repository> (the repository's layout, so the
    pack folder is whole as the engine reads it)."""
    return Path(models_dir) / f[1].replace("/", "--") / f[3]


def prepared_dir(model: str, models_dir: Path) -> Path:
    """Where a model that is prepared on this PC (MODELS[..]["prepare"]) keeps what it prepared."""
    return Path(models_dir) / MODELS[model]["repo"].replace("/", "--") / FP4LOCAL_PREPARED


def prepared_done(model: str, models_dir: Path) -> bool:
    """Every step of tools/nvfp4_local_pack.py prepare is marked done (its marks, written when each step is whole)."""
    d = prepared_dir(model, models_dir)
    return all((d / m).is_file() for m in ("ple-nvfp4.gguf.done", "dense.gguf.done", "pack.done", "draft.done"))


def ready_paths(model: str, models_dir: Path) -> dict:
    """What the engine is given: the pack folder and the dense, PLE and embedding GGUFs (a prepared model also has its
    own draft layer, "mtp", and no separate embedding: the token embedding is in its dense GGUF)."""
    if MODELS[model].get("prepare"):
        d = prepared_dir(model, models_dir)
        return {"pack": d / "pack", "dense": d / "dense.gguf", "ple": d / "ple-nvfp4.gguf", "mtp": d / "mtp"}
    out = {}
    for f in ready_files(model):
        p = ready_where(models_dir, f)
        if f[0] == "pack" and p.name == "index.txt":
            out["pack"] = p.parent
        elif f[0] in ("dense", "ple", "embd"):
            out[f[0]] = p
    return out


def ready_verified(p: Path, sha: str) -> bool:
    """Downloaded whole and checked: its finish mark holds the pinned SHA-256 (verify_sha256)."""
    m = p.with_name(p.name + ".done")
    try:
        return p.is_file() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def ready_fetch(model: str, models_dir: Path) -> None:
    """Every file of a ready-made pack: already here and checked; the same file (by SHA-256) checked for another
    ready-made pack, linked instead of downloaded (the PLE table and the embedding); or downloaded from its pinned
    revision and checked against its size and SHA-256 (a wrong file is deleted and setup stops)."""
    others = [(ready_where(models_dir, g), g[5]) for m in MODELS if MODELS[m].get("repo") for g in ready_files(m)]
    for f in ready_needed(model, models_dir):
        dst, size, sha = ready_where(models_dir, f), f[4], f[5]
        if ready_verified(dst, sha):
            ok(f"{f[3]} already downloaded")
            continue
        src = next((p for p, s in others if s == sha and p != dst and ready_verified(p, s)), None)
        if src is not None and not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
                mark(dst, f"sha256 {sha}")
                ok(f"{f[3]}: the same file as {src} (linked, not downloaded again)")
                continue
            except OSError:
                pass                                   # another drive: downloaded
        download(f"{hf_endpoint()}/{f[1]}/resolve/{f[2]}/{f[3]}", dst, f"{f[1].split('/')[-1]}: {f[3]}")
        verify_sha256(dst, size, sha)


def resident_budget_gib(model, ram, kv_ram_gb=0.0) -> int:
    """UD-Q4_K_XL: the GiB of experts the engine keeps in RAM (--resident-budget-gib): the RAM (GiB, ram_gb()) less
    24 for the OS, the engine and the file cache the other experts are read through, less a KV cache streamed to
    RAM; at most all of them, at least 8.  64 GB: 40, the measured setting (docs/UNSLOTH_Q4.md)."""
    gib = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(kv_ram_gb)
    return max(8, min(gib, int(MODELS[model]["arena_gb"] / 1.073741824)))


# MODEL-UPDATE Phase 2: Unsloth's files keep every expert in RAM (a pinned arena, as ISTA's files do) where the experts
# and this much more fit the PC's RAM; the RAM budget (--resident-budget-gib: the hottest experts in RAM, the rest read
# from the GGUF in place) only below that.  The rule and the 16 GiB are sergqwer's fork (sergqwer/strata-nvfp4,
# 0.1.42-nvfp4.2, MIT): there, on 128 GB with an RTX 5090, UD-Q4_K_XL decoded at 169 tokens/s and read a 32K prompt in
# 4.9 s with every expert in RAM, against 118 tokens/s and 30 s in the budget mode.
UNSLOTH_RESIDENT_BESIDE_GB = 16


def unsloth_budget(model, ram, gpu=None, asked=None) -> bool:
    """Does setup write the RAM-budget mode (--resident-budget-gib) for this Unsloth file: asked for (--resident-budget-gib
    N), a unified-memory APU (#1715: there the budget and the GPU's cache are the same RAM, and the all-in-RAM mode was
    measured on discrete cards only), or its experts with UNSLOTH_RESIDENT_BESIDE_GB beside them do not fit this PC's RAM
    (GiB, ram_gb()).  Else every expert is kept in RAM, as for ISTA's files."""
    return asked is not None or bool((gpu or {}).get("uma")) or \
        ram < MODELS[model]["arena_gb"] / 1.073741824 + UNSLOTH_RESIDENT_BESIDE_GB


def images_default(model, family, ram, gpu=None) -> bool:
    """Is "images on" setup's default answer (the question's Enter, and --yes) for this choice?  Owner rule 2026-10-18: pictures are on
    wherever the model can read them - the original Flash-Next, Swift and Coder files, Unsloth's, NVFP4 - on every make of card; which
    reader runs is strata_control/vision_reader.py's rule (the CPU under 24 GB of VRAM, the GPU from 24 GB).  The CPU reader takes no
    VRAM (no --vram-reserve-mib), so a small card is no exception."""
    return MODELS.get(model, {}).get("vision", FAMILIES.get(family, {}).get("vision")) is not False


unsloth_images_default = images_default     # the earlier name


IMAGES_GPU_ENCODER_MIN_VRAM_GB = 23.5   # = strata_control/vision_reader.py GPU_MIN_VRAM_GB, the one place for the rule; #2406: below this the images answer "yes" gets the CPU encoder (a card of 12-16 GB: the GPU
                                        # encoder's ~1.1-1.4 GB is ~850 expert slots lost, 3-5% of decode and prompt on P100 / A4000
                                        # 16 GB (Linux, IQ3_S / IQ3_XXS), and 35-60% on an RTX 4080 SUPER under Windows as reported)


def flash_vision_encoder(asked: str, gpu: dict) -> str:
    """#2406: which encoder a Flash-Next images answer gets on NVIDIA.  "yes" (the question's y, or --vision yes) means the
    GPU encoder on a card of 24 GB and more and the CPU encoder below that: a picture then takes a few seconds, the text
    keeps the ~1 GB of VRAM for experts.  --vision gpu is always kept (with a note on a small card); --vision cpu is kept."""
    vram = float((gpu or {}).get("vram_gb") or 0.0)
    small = 0 < vram < IMAGES_GPU_ENCODER_MIN_VRAM_GB
    if asked == "gpu" and small:
        warn(f"the GPU image encoder keeps ~1.4 GB of VRAM that a {vram:.0f} GB card would use for experts (measured 3-5% "
             "of decode on 16 GB cards, more on some Windows PCs, #2406): kept as you chose; --vision cpu reads pictures "
             "on the CPU instead (a few seconds each)")
    if asked == "yes":
        return "cpu" if small else "gpu"
    return asked


UMA_EXPERTS = 24576            # the Flash-Next model's experts (48 layers x 512): the most an expert cache can hold


def uma_expert_cache(model, ram, budget_gib, kv_ram_gb=0.0):
    """#1715: on a unified-memory APU (Strix Halo) the GPU's expert cache and the RAM budget of experts are both RAM.
    `--expert-cache auto` sizes the cache from the memory the OS can give back at start, which counts all the RAM the
    budget has not taken yet: a 128 GB box with a 55 GiB budget and the desktop open was killed by the OOM killer.  So
    there setup writes a number, not `auto`: the RAM less UNSLOTH_RAM_LEFT_GB less the budget (and a KV cache streamed
    to RAM), in slots (an expert is priced at 1.1x the average one: the Q8_0 downs are bigger).  None when no room is left
    (the config keeps `auto`).  A recommendation: the number is in the config's args, and --expert-cache N there sets it."""
    slot_gib = MODELS[model]["arena_gb"] * 1e9 / UMA_EXPERTS * 1.1 / 2**30
    room = ram - UNSLOTH_RAM_LEFT_GB - budget_gib - math.ceil(kv_ram_gb)
    slots = min(UMA_EXPERTS, int(room / slot_gib))
    return slots if slots >= 256 else None


def budget_choice(model, ram, asked) -> float:
    """S4: UD-Q4_K_XL's RAM budget: --resident-budget-gib N as given, else the recommendation (resident_budget_gib).
    More than the recommendation is kept, with what it risks (the owner's rule: setup recommends, it never forces)."""
    rec = resident_budget_gib(model, ram)
    if asked is None:
        return rec
    if asked > rec:
        warn(f"a {asked:g} GiB RAM budget is more than setup recommends for this PC ({rec} GiB: the RAM less "
             f"{UNSLOTH_RAM_LEFT_GB} GB for the OS, the engine and the file cache that reads the other experts). Kept "
             "as you chose: the engine clamps it to the RAM it finds free at start (less 4 GB), and the file cache "
             "gets less room - it may be slower, or run the PC out of RAM under load")
    return int(asked) if asked == int(asked) else asked


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand, a download finished by an older setup) otherwise passes as a model file and the engine
    fails much later, at the first tensor that runs past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if not g.tensors:
            # #2346: a metadata-only shard (Unsloth's UD-Q5_K_XL 00001, 10,946,618 bytes) ends at its header: the
            # alignment padding before an empty data section is not in the file, so data_start can be past its end
            need = min(need, have)
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")
        if len(g.tensors) == 1 and have - need >= 64:       # (a few bytes may be the last tensor's alignment padding)
            # #657: a shard that holds one tensor (the PLE table) is exactly as long as that tensor; extra bytes are a
            # damaged or wrong file that the engine would refuse at start ("PLE table size mismatch")
            fail(f"{s.name} is longer than its tensor: {have:,} bytes, {need:,} expected ({have - need:,} extra)",
                 f"delete {s.name} and its .done mark and run setup again")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build, gguf-py for the tools, mtmd for images), as a zip: no git."""
    llama = home() / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = home() / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = home() / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(z) as f:
            # llama.cpp's own web UI (tools/ui) is not used, and its deep paths passed Windows' 260-character limit in a
            # folder like Downloads\Strata-main\Strata-main (#206)
            f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
    except zipfile.BadZipFile:
        # #1797: do not let the .done mark make every later run reuse a damaged source archive.
        try:
            drop_archive(z)
        except OSError as e:
            warn(f"cannot remove damaged {z.name} ({e}): delete it and its .done mark before running setup again")
        raise
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    # PR #63: on Windows a rename can fail with PermissionError while an antivirus scanner still holds a file of the
    # fresh unpack; shutil.move falls back to copy-and-delete, and a few retries let the scanner finish.  The target
    # is `llama` itself - moving into its parent would keep the zip's `llama.cpp-<sha>` folder name.
    for attempt in range(5):
        try:
            shutil.move(str(top), str(llama))
            break
        except PermissionError:
            if attempt == 4:
                raise
            shutil.rmtree(llama, ignore_errors=True)   # a partial copy from the failed attempt
            time.sleep(2)
    shutil.rmtree(tmp, ignore_errors=True)
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)
    return llama


def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """A requirements file's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


MONITOR_REQUIREMENTS = ROOT / "requirements-monitor.txt"


def runtime_packages() -> list[str]:
    """The packages setup installs for a run: the pinned runtime ones, plus the Terminal's screens (textual, which brings rich: `strata launch`
    and `strata monitor` need them; the packaged Terminal bundles them, a source install gets them here)."""
    base = requirement_lines() if REQUIREMENTS.exists() else list(PY_PACKAGES)
    if MONITOR_REQUIREMENTS.exists():
        base += [r for r in requirement_lines(MONITOR_REQUIREMENTS) if r not in base]
    return base


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def control_missing_packages() -> list:
    """The Python packages the preparation tools need that this interpreter does not have (the app bundle has them all)."""
    import importlib.util
    return [m for m in ("numpy", "jinja2", "regex", "yaml", "tqdm", "requests", "PIL", "psutil")
            if importlib.util.find_spec(m) is None]


def pip_stamp_is_ours(stamp: Path) -> bool:
    """The list of installed packages was written for THIS system (#3634). A .venv on a disk that Windows and Linux both start from keeps
    the stamp of the other one, which then says "already installed" about packages this Python does not have. A marker beside the stamp
    names the system that wrote it; an older stamp without a marker is believed only while the packages every run needs can be imported."""
    marker = stamp.with_name(stamp.name + ".os")
    try:
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip() == sys.platform
    except OSError:
        return False
    import importlib.util
    return all(importlib.util.find_spec(m) is not None for m in ("numpy", "jinja2", "regex", "yaml"))


def pip_install(packages, what, extra=()):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements (#214) recorded bare names: those packages are kept as they are (nothing is reinstalled), and the
    pinned dependencies it already has count as installed."""
    if CONTROL:                                        # the app ships these (hash-locked); nothing is installed here
        missing = control_missing_packages()
        if missing:
            fail("this copy of Strata Control lacks " + ", ".join(missing), "reinstall the Strata app")
        ok(f"{what}: bundled with the app")
        return
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() and pip_stamp_is_ours(stamp) else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    acquisition.require(what)
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *extra, *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    stamp.with_name(stamp.name + ".os").write_text(sys.platform, encoding="utf-8")
    ok(f"{what} installed")


NVFP4_LOCAL_REQUIREMENTS = ROOT / "requirements-nvfp4-local.txt"
NVFP4_LOCAL_MODULES = ("torch", "transformers", "safetensors")


def nvfp4_local_packages() -> list:
    """The pinned lines of requirements-nvfp4-local.txt (comments and options left out)."""
    try:
        lines = NVFP4_LOCAL_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [l.split("#")[0].strip() for l in lines if l.strip() and not l.lstrip().startswith(("#", "-"))]


def ensure_nvfp4_local_packages() -> None:
    """#3227: NVIDIA's FP4-Local checkpoint is converted on this PC with llama.cpp's converter, which imports torch and transformers
    (the runtime requirements do not have them: nothing else in setup needs them).  Installed only for a prepare that runs; what is
    already there is kept.  The app does not ship them (hundreds of MB): it says so and points at the source installer."""
    import importlib.util
    missing = [m for m in NVFP4_LOCAL_MODULES if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if CONTROL:
        fail("preparing NVIDIA's FP4-Local checkpoint needs PyTorch and transformers, which the app does not include",
             "install this model from the source download (SETUP.bat on Windows, ./setup.sh on Linux), or pick another version in the app")
    pkgs = nvfp4_local_packages()
    if not pkgs:
        fail("requirements-nvfp4-local.txt is missing", "update Strata, or run: python -m pip install torch transformers safetensors")
    say("  Converting NVIDIA's checkpoint needs PyTorch, transformers and safetensors (a few hundred MB, installed once).")
    pip_install(pkgs, "PyTorch, transformers and safetensors (for the FP4-Local conversion)",
                extra=[] if WIN else ["--extra-index-url", "https://download.pytorch.org/whl/cpu"])   # Linux: the CPU build, not 3 GB of CUDA libraries
    importlib.invalidate_caches()


def cuda_lib_dirs(toolkit=13):
    """Where pip put NVIDIA's CUDA libraries (nvidia/cu13/bin/x86_64 on Windows, nvidia/cu13/lib on Linux).
    toolkit 12: the CUDA 12 wheels' cuBLAS and runtime, in two folders (nvidia/cublas/bin, nvidia/cuda_runtime/bin)."""
    if int(toolkit) == 12:
        patterns = ("cublas64_12.dll", "cudart64_12.dll") if WIN else ("libcublas.so.12*", "libcudart.so.12*")
    else:
        patterns = ("cublas64_13.dll",) if WIN else ("libcublas.so.13*",)
    dirs = []
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        for pattern in patterns:
            for hit in (sp / "nvidia").rglob(pattern) if (sp / "nvidia").is_dir() else []:
                if hit.parent not in dirs:
                    dirs.append(hit.parent)
    return [str(d) for d in dirs]


# ------------------------------------------------------------------------------------------------ AMD
# The RX 7900 XT / XTX (gfx1100) and the RX 9070 series / Radeon AI PRO R9700 (gfx1201) on Linux, through the HIP
# backend (docs/AMD_HIP.md); the RX 7800 XT / 7700 XT (gfx1101, #254) and the RX 9060 XT (gfx1200, #256) were run by
# their owners; the RX 6800 / 6900 series (gfx1030, #311) and the RX 6700 XT (gfx1031, #524) run but are unvalidated.  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;
# a system ROCm 7 in /opt/rocm is used when it has hipcc and hipBLAS) and the engine is compiled here for the cards.
# No images yet.
ROCM_INDEXES = {"gfx1100": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",   # TheRock's wheels per GPU family
                "gfx1101": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1102": "https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/",
                "gfx1200": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1201": "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1031": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",
                "gfx1151": "https://rocm.nightlies.amd.com/v2/gfx1151/",       # Strix Halo (docs/STRIX_HALO.md)
                "gfx1150": "https://rocm.nightlies.amd.com/v2/gfx1150/",       # Strix Point: only with STRATA_EXPERIMENTAL_GFX1150=1
                "gfx1103": "https://rocm.nightlies.amd.com/v2/gfx110X-all/"}   # Radeon 780M: only with STRATA_EXPERIMENTAL_GFX1103=1
# The TheRock nightly indexes are pruned and move on, and each GPU family's index holds its own range (#1103: gfx103X-all
# starts at 7.13.0a20260422 and has no 7.10; #1267: the 7.10 wheel segfaults on a Strix Halo with kernel 7.2.8), so one
# version for every card cannot work.  The wheel is chosen per index (the family): a preferred version, tried first; when
# the index no longer offers it, the newest one of the same 7.x line, else of the same major, with a warning (rocm_pick).
# STRATA_ROCM_VERSION still forces one exact version (no lookup).
ROCM_VERSION_DEFAULT = "7.10.0a20251120"                 # what Strata's HIP build was tested with (gfx120X, gfx110X)
ROCM_FAMILY_PINS = {"gfx103X-all": "7.13.0a20260515",    # #1103: 7.13.0a20260515 runs; the 7.14 nightlies time out
                    "gfx110X-all": "7.10.0a20251121",    # the experimental gfx1103: this index has no ...20251120
                    "gfx1151": "7.14.0a20260608"}        # #1267: 7.14.0a20260529 to 20260608 run on kernel 7.2.8; the 7.10 wheel segfaults
ROCM_VERSION_OVERRIDE = os.environ.get("STRATA_ROCM_VERSION") or None
ROCM_VERSION = ROCM_VERSION_OVERRIDE or ROCM_VERSION_DEFAULT   # kept for callers that read one version
ROCM_SYSTEM_MIN = (7, 0)       # an older system ROCm is passed over for the wheels (gfx1201 needs ROCm 6.4 or newer)
# STRATA_EXPERIMENTAL_GFX1103=1 (opt-in, unsupported): the Radeon 780M / 760M / 740M iGPU (Ryzen 7040 / 8040, gfx1103) is taken
# as a unified-memory AMD card like Strix Halo, with the native dot4 and gfx11 WMMA kernels, the gfx1103 exact speed switches
# and a hipBLASLt table when one matches (measured on one machine: Ryzen 7 8745HS).  Unset: unchanged.
GFX1103_OPT_IN = os.environ.get("STRATA_EXPERIMENTAL_GFX1103") == "1"
# STRATA_EXPERIMENTAL_GFX1150=1 (opt-in, unvalidated): the Radeon 890M / 880M iGPU (Ryzen AI 300, Strix Point, gfx1150)
# is taken as a unified-memory AMD card like Strix Halo.  The engine builds and runs for gfx1150 from 0.1.40.2 (#1217;
# tools/hip/gfx1150-hipblaslt-100401.txt is calibrated on it), but no release ships a ready-made Windows engine for it:
# build one with `STRATA_HIP_ARCHS=gfx1150 tools\hip\build_windows.bat` and start setup with `--prebuilt dist\`.
# Unset: unchanged.
GFX1150_OPT_IN = os.environ.get("STRATA_EXPERIMENTAL_GFX1150") == "1"
AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx1030", "gfx1031", "gfx1151") \
            + (("gfx1103",) if GFX1103_OPT_IN else ()) + (("gfx1150",) if GFX1150_OPT_IN else ())
# The exact (bit-identical) speed switches the engine turns on by default on gfx1151 (src/core/arch_defaults.cpp), less
# STRATA_HCD_EXACT: that one copies the gfx1151 hipBLASLt table's solutions 1176 / 1177, and the gfx1103 table
# (tools/hip/gfx1103-hipblaslt-100401.txt) picks others, so it would only report that it cannot run.  A gfx1103 install
# takes them from here (setup writes them into the config's "env"), the same way the table is written: one 780M measured
# the same greedy answers with all 17 on and with all 17 off, +13% to +15% prompt and decode +3.2% (the decode half
# alone).  The engine keeps gfx1103 out of its own arch defaults (src/core/arch_defaults_test.cpp asserts it stays
# empty), so this keeps the default path byte-identical.
GFX1103_SWITCHES = {
    # prompt
    "STRATA_GDN_HEAD": "1", "STRATA_GDN_PP": "2", "STRATA_GDN_CONVL2": "1", "STRATA_GDN_NOY": "1",
    "STRATA_CVEC_FUSE": "1", "STRATA_PF_PAD": "1",
    # decode
    "STRATA_Q8_PACKED": "1", "STRATA_Q6_PACKED": "1", "STRATA_MMVF_ROWS": "1", "STRATA_ATTN_LANECELL": "1",
    "STRATA_EXPERT_V2": "1", "STRATA_TSUM": "1", "STRATA_LFUSE": "1", "STRATA_GDN_SPLIT": "1",
    "STRATA_QFUSE": "1", "STRATA_PLE_BATCH": "1", "STRATA_SH_STREAM": "1",
}
# The same set for a gfx1150 install (Radeon 890M / 880M): less STRATA_HCD_EXACT as well, because a gfx1150 table picks other
# solutions than the gfx1151 one.  The Radeon 890M report (docs/AMD_HIP.md; bench/results/2026-10-07-community-gfx1150) measured the
# same greedy answers with all 17 on and with all 17 off, decode +2% and prompts +3-6%.  The engine's arch defaults are not used
# for gfx1150 (src/core/arch_defaults_test.cpp asserts gfx1150 stays empty), so this keeps the default path byte-identical.
GFX1150_SWITCHES = dict(GFX1103_SWITCHES)
AMD_NAMES = {"gfx1100": "AMD Radeon RX 7900 series (gfx1100)",   # when sysfs has no product name
             "gfx1101": "AMD Radeon RX 7800 XT / 7700 XT (gfx1101)",
             "gfx1102": "AMD Radeon RX 7600 / 7600 XT (gfx1102)",
             "gfx1200": "AMD Radeon RX 9060 series (gfx1200)",
             "gfx1201": "AMD Radeon RX 9070 series / AI PRO R9700 (gfx1201)",
             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)",
             "gfx1031": "AMD Radeon RX 6700 XT series (gfx1031)",
             "gfx1151": "AMD Radeon 8060S / 8050S / 8040S (Ryzen AI Max, Strix Halo, gfx1151)",
             # integrated Radeons of other Ryzen families: named so they are not taken for Strix Halo (not supported)
             "gfx1150": "AMD Radeon 890M / 880M (Ryzen AI 300, Strix Point, gfx1150)",
             "gfx1152": "AMD Radeon 860M / 840M (Ryzen AI 300, Krackan Point, gfx1152)",
             "gfx1103": "AMD Radeon 780M / 760M / 740M (Ryzen 7040 / 8040, Phoenix / Hawk Point, gfx1103)"}
AMD_CARDS = ("the RX 7900 XT / XTX (gfx1100), RX 7800 XT / 7700 XT (gfx1101), RX 9060 XT (gfx1200) and "
             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030) and RX 6700 XT "
             "(gfx1031, #524), and the RX 7600 / 7600 XT (gfx1102, one run reported, #942), all unvalidated, and the Ryzen AI Max \"Strix Halo\" APU (Radeon 8060S / 8050S / 8040S, "
             "gfx1151: experimental, docs/STRIX_HALO.md)")


def rocm_index(arch):
    return os.environ.get("STRATA_ROCM_INDEX") or ROCM_INDEXES[arch]


def rocm_family(index):
    """The family name of a TheRock index url (.../v2/gfx103X-all/ -> gfx103X-all)."""
    return index.rstrip("/").rsplit("/", 1)[-1]


def rocm_vkey(version):
    """Sort key of a TheRock version: 7.13.0a20260515 -> (7, 13, 0, 20260515); None when it is not one."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[a-z]+(\d+))?", version)
    return tuple(int(g or 0) for g in m.groups()) if m else None


def rocm_index_versions(html):
    """The versions of the `rocm` package that a TheRock index page (<index>/rocm/) lists, oldest first."""
    return sorted({m for m in re.findall(r"rocm-(\d+\.\d+\.\d+(?:[a-z]+\d+)?)\.(?:tar\.gz|whl)", html or "")
                   if rocm_vkey(m)}, key=rocm_vkey)


def rocm_pick(available, preferred):
    """(version, note) for an index that lists `available`: `preferred` when it is there, else the newest of the same
    7.x line, else the newest of the same major (the note says so); `preferred` unchanged when the list is empty (the
    index could not be read: pip then reports its own error) or nothing of that major exists (the note says so)."""
    if not available or preferred in available:
        return preferred, None
    want = rocm_vkey(preferred)
    for same, what in ((lambda k: k[:2] == want[:2], "line"), (lambda k: k[0] == want[0], "major")):
        hits = [v for v in available if same(rocm_vkey(v))]
        if hits:
            got = max(hits, key=rocm_vkey)
            return got, (f"ROCm {preferred} is not on this index any more: using {got}, the newest of the same {what} "
                         f"(set STRATA_ROCM_VERSION to choose another)")
    return preferred, (f"ROCm {preferred} is not on this index and nothing of its major version is: "
                       f"{available[0]} to {available[-1]} are (set STRATA_ROCM_VERSION to one of them)")


def rocm_wanted(index, listing=None):
    """(ROCm wheel version, warning or None) for a TheRock index (see ROCM_FAMILY_PINS).  `listing` is the index
    page's html (None: fetched from <index>/rocm/; a failed fetch keeps the preferred version)."""
    if ROCM_VERSION_OVERRIDE:
        version, note = ROCM_VERSION_OVERRIDE, None
    else:
        preferred = ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT)
        if listing is None:
            try:
                req = urllib.request.Request(index.rstrip("/") + "/rocm/", headers={"User-Agent": "strata-setup"})
                with _urlopen(req, timeout=30) as r:
                    listing = r.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                listing = ""
        version, note = rocm_pick(rocm_index_versions(listing), preferred)
    if rocm_family(index) == "gfx1151" and (rocm_vkey(version) or (9, 0))[:2] < (7, 11):
        note = ((note + "; ") if note else "") + (
            "the ROCm 7.10 wheels segfault in the HSA runtime on a Strix Halo with a new kernel (7.2.8, #1267): "
            "the 7.14 line (docs/STRIX_HALO.md) is the one that works")
    return version, note


# Strix Halo (Ryzen AI Max 380 / 385 / 390 / 395 + PRO: Radeon 8040S / 8050S / 8060S, RDNA 3.5, gfx1151) and the other
# integrated Radeons it must never be taken for.  pci.ids lists ONE device id for the whole Strix Halo family
# ("1586  Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics]"; the 8040S shares it: the
# three differ by their CU count, 40 / 32 / 16, not by id).  Aurora's real answers (tools/test_setup_strix_halo.py):
# KFD gfx_target_version 110501, vendor_id 4098, device_id 5510 (0x1586), simd_count 80.
STRIX_HALO_ARCH = "gfx1151"
STRIX_HALO_PCI_IDS = frozenset({0x1586})
AMD_IGPU_PCI = {0x1586: "gfx1151",                                    # Strix Halo
                0x150E: "gfx1150",                                    # Strix Point: Radeon 880M / 890M
                0x1114: "gfx1152", 0x1902: "gfx1152",                 # Krackan Point (Krackan2): Radeon 840M / 860M
                0x15BF: "gfx1103", 0x15C8: "gfx1103", 0x164F: "gfx1103",   # Phoenix: Radeon 740M / 760M / 780M
                0x1900: "gfx1103", 0x1901: "gfx1103"}                 # Hawk Point
STRIX_HALO_BY_SIMDS = {80: "8060S", 64: "8050S", 32: "8040S"}   # KFD simd_count (2 per CU): 40 / 32 / 16 CUs
UMA_OS_LEFT_GB = 6             # the share of the unified memory kept for the OS: what the engine's device_free_bytes() leaves


def gfx_arch_is(arch, name) -> bool:
    """include/strata/kernels/gfx_arch.hpp's gfx_arch_is: `arch` (gcnArchName, "gfx1151" or "gfx1151:sramecc-:xnack-")
    is exactly `name`, never a prefix: gfx1150 and gfx1152 are not gfx1151 (and gfx1151 is not gfx115)."""
    arch = str(arch or "")
    return arch == name or arch.startswith(name + ":")


def is_strix_halo(g) -> bool:
    return isinstance(g, dict) and gfx_arch_is(g.get("arch"), STRIX_HALO_ARCH)


RDNA2_ARCHS = ("gfx1030", "gfx1031")        # RX 6800 / 6900 / 6950 series and RX 6700 XT: the cards of the ready-made RX 6000 engine package


def is_rdna2(g) -> bool:
    """A discrete RX 6000 card the ready-made RX 6000 engine package covers (gfx1030 / gfx1031); never an RX 7000 / 9000 or an APU."""
    return isinstance(g, dict) and any(gfx_arch_is(g.get("arch"), a) for a in RDNA2_ARCHS)


RDNA2_PACKAGE_NOTE = ("RX 6000 series (RDNA2, {arch}): Strata has a ready-made engine package for these cards, built, not tested on a real "
                      "RX 6000 card. It runs plain kernels (this chip has no matrix cores, so the RX 7000 / RX 9000 prompt paths stay off) "
                      "and plain hipBLAS (no hipBLASLt tuning table for it). If something goes wrong, open an issue with the end of "
                      "strata-<model>.log: docs/AMD_HIP.md, \"RDNA2\".")


def rdna2_notes(gpu) -> list[str]:
    """What setup tells the owner of an RX 6000 card (gfx1030 / gfx1031); nothing for any other card."""
    if not is_rdna2(gpu):
        return []
    return ["  " + RDNA2_PACKAGE_NOTE.format(arch=str(gpu.get("arch")).split(":")[0])]


def is_uma_apu(g) -> bool:
    """A unified-memory AMD APU whose "VRAM" is a BIOS carve-out plus shared system memory: Strix Halo (gfx1151), and
    the opt-in gfx1150 (Strix Point) / gfx1103 (Phoenix / Hawk Point) iGPUs.  Everything else is a discrete card."""
    arch = g.get("arch") if isinstance(g, dict) else None
    return (is_strix_halo(g)
            or (GFX1150_OPT_IN and gfx_arch_is(arch, "gfx1150"))
            or (GFX1103_OPT_IN and gfx_arch_is(arch, "gfx1103")))


def amd_apply_uma(g: dict, gtt_gb: float = 0.0, ram: float | None = None) -> dict:
    """Strix Halo is an APU with unified memory.  Its "dedicated VRAM" is a BIOS carve-out (512 MB to 96 GB; 2 GB on the
    maintainers' box) and the GPU also reaches shared system memory (Linux: the GTT pool, `mem_info_gtt_total`, half the
    RAM unless the kernel was told otherwise; Windows: "shared GPU memory", half the RAM).  The shared part IS the RAM
    the model's experts live in, so it is not extra room beside the RAM.  g gets: uma, dedicated_gb (the carve-out: the
    only part that adds to the RAM, which the low-RAM mode counts) and vram_gb = the memory the GPU can use in all
    (the carve-out plus the shared pool less what the OS keeps), which sizes the context and the parallel slots."""
    if g.get("uma") or not is_uma_apu(g):
        return g
    ram = ram_gb() if ram is None else ram
    carve = max(0.0, float(g.get("vram_gb") or 0.0))
    shared = gtt_gb if gtt_gb > 0 else ram / 2
    shared = max(0.0, min(shared, ram - UMA_OS_LEFT_GB))
    g.update({"uma": True, "dedicated_gb": carve, "shared_gb": shared, "vram_gb": carve + shared})
    return g


def low_ram_vram(g) -> float:
    """The GPU memory the low-RAM mode may count as room beside the RAM: a discrete card's VRAM; an APU's BIOS
    carve-out only (its shared memory is the RAM itself)."""
    return g.get("dedicated_gb", 0.0) if g.get("uma") else g["vram_gb"]


def amd_mem_text(g) -> str:
    """How setup says a card's memory: "24 GB VRAM", or for an APU its unified memory."""
    if g.get("uma"):
        return (f"unified memory: {g['dedicated_gb']:.1f} GB BIOS carve-out + {g['shared_gb']:.0f} GB shared with the "
                f"RAM, {g['vram_gb']:.0f} GB usable by the GPU")
    return f"{g['vram_gb']:.0f} GB VRAM"


STRIX_HALO_FAMILY, STRIX_HALO_MODEL = "unsloth", "UD-IQ4_XS"    # the model docs/STRIX_HALO.md measures
STRIX_HALO_MIN_GB = 80         # UD-IQ4_XS keeps all of its experts in memory from ~80 GB (its menu line says so)


RECOMMENDED_SIZES = {"qwen": ("IQ3_S",), "swift": ("IQ3_S",), "unsloth": ("UD-Q4_K_XL", "UD-IQ4_XS")}   # owner's pick, 0.1.44


def mem_have(model: str, ram: float, vram: float = 0.0) -> float:
    """The memory this PC offers a model, in the unit of its MODELS "ram_gb": RAM, or for a "mem_total" model (Gyro) the
    VRAM and the RAM together."""
    return ram + vram if MODELS[model].get("mem_total") else ram


# 0.2.0, the owner's memory tiers (README "The other sizes"): RAM + VRAM together pick the size.  The floors sit 2 GB
# below the tiers' names (40 / 48 / 64 GB) because the OS reports a little less than the installed memory.  64 GB and
# up: IQ3_S (the original; Swift 1.5's IQ3_S is an equal choice), 48 GB: IQ3_XXS, 40 GB: Swift 1.5's IQ2_XS, with Gyro-M
# (35 GB) as the other choice.  Below 40 GB the older rules stand (Gyro-S, or the Coder for code).  The Mac has its own
# rules (strata_control/mac_fit.py).  tools/strata_mcp.py's recommend() holds the same floors (the app and `strata launch`).
MEMORY_TIERS = ((62.0, "qwen", "IQ3_S"), (46.0, "qwen", "IQ3_XXS"), (38.0, "swift", "IQ2_XS"))
# #3425 (owner's rule): the tiers count RAM + VRAM less this much, which Windows and the system keep for themselves.  A PC
# with a 16 GB card and 32 GB of RAM used to land in the 48 GB tier and then read ~11 GB of it from the SSD on every answer.
# tools/strata_mcp.py has the same constant; the Mac (strata_control/mac_fit.py) keeps its own macOS share and does not use it.
TIER_SYSTEM_RESERVE_GB = 5.0


def tier_pick(ram: float, vram: float = 0.0):
    """(family, size) for this PC's RAM + VRAM by MEMORY_TIERS, or None below the lowest tier (then the older rules pick:
    Gyro-S, or the Coder).  With less RAM than the size's own figure the low-RAM mode starts by itself and the model is
    slower.  A recommendation: every size stays a choice."""
    for floor, fam, m in MEMORY_TIERS:
        if ram + vram - TIER_SYSTEM_RESERVE_GB >= floor:
            return fam, m
    return None


def tier_note(ram: float, vram: float = 0.0, gyro: bool = True) -> str:
    """The sentence under the model menu for the lowest tier: Gyro-M is the other choice there (NVIDIA and AMD only)."""
    pick = tier_pick(ram, vram)
    if not gyro or pick != ("swift", "IQ2_XS"):
        return ""
    return ("this PC's RAM + VRAM is in the 40 GB tier: Swift 1.5's IQ2_XS is the pick; Gyro-M (35 GB) is the other choice, "
            "close to IQ3_XXS in quality but 16-21% slower at writing answers than Q2_0")


def recommended_size(family: str, names: list, ram: float, vram: float = 0.0) -> str:
    """The size the menu pre-selects (and --yes takes): the memory tier's size (RAM + VRAM, MEMORY_TIERS) when this
    family has it, else the family's recommended size when this PC's RAM fits it (UD-Q4_K_XL before UD-IQ4_XS), else
    the largest size that fits, else the first.  Recommend, never force: every other size stays a choice."""
    tier = tier_pick(ram, vram)
    if tier and tier[1] in names and family in ("qwen", "swift"):
        return tier[1]
    for m in RECOMMENDED_SIZES.get(family, ()):
        if m in names and mem_have(m, ram, vram) >= MODELS[m]["ram_gb"]:
            return m
    fitting = [m for m in names if mem_have(m, ram, vram) >= MODELS[m]["ram_gb"]]
    if fitting:
        return max(fitting, key=lambda m: MODELS[m]["download_gb"])
    return "UD-IQ4_XS" if family == "unsloth" and "UD-IQ4_XS" in names else names[0]


def size_tag(family: str, m: str, installed: str | None = None) -> str:
    rec, have = m in RECOMMENDED_SIZES.get(family, ()), m == installed
    return " (recommended, installed)" if rec and have else " (recommended)" if rec else " (installed)" if have else ""


def installed_size(family: str, names: list, have: list) -> str | None:
    """0.1.44.1: the size of this family that is already installed (the most recently used config first), or None.  Setup
    run again (SETUP.bat, --setup) pre-selects it: the 0.1.44 recommendation (IQ3_S, about 15% slower than IQ3_XXS and
    7 GB more RAM) is for a first install, and Enter must not move an installed model to another size."""
    for c in have:
        try:
            ch = choices_from_config(c)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        if ch.get("family") == family and ch.get("model") in names:
            return ch["model"]
    return None


def unsloth_default_env(family: str, model: str, hip: bool, in_budget: bool) -> dict:
    """0.1.44 (owner, review-quants.md section 5): the run-config "env" setup writes for Unsloth's files.
    STRATA_MMQ_KQ=1 (the K-quant MMQ prompt kernels) for UD-Q5_K_XL and UD-Q6_K_XL, where its prompt-path KL is equal
    or better than the FP16 path's; UD-Q4_K_XL keeps it opt-in.  STRATA_HC_Q8=1 on CUDA for the UD-* files (the
    GGUF's own Q8_0 hyper-connection projections, closer to the file and faster; the GSQ-RCO packs are untouched).
    STRATA_ROUTE_TAIL_SKIP=0 where every expert is in RAM (the skip costs KL there); the RAM-budget mode keeps it."""
    env = {}
    if family != "unsloth":
        return env
    if model in ("UD-Q5_K_XL", "UD-Q6_K_XL"):
        env["STRATA_MMQ_KQ"] = "1"
    if not hip and model.startswith("UD-"):
        env["STRATA_HC_Q8"] = "1"
    if not in_budget:
        env["STRATA_ROUTE_TAIL_SKIP"] = "0"
    return env


def menu_families(hip: bool) -> list:
    """The families the model menu lists, in menu order.  The NVFP4 packs run on NVIDIA cards only (the AMD and Intel
    engines have no NVFP4 kernels), so on those the menu does not offer them; `--family nvfp4` still says why not."""
    return [f for f in FAMILIES if not (hip and FAMILIES[f].get("ready"))]


def nvfp4_card_note(gpu) -> str:
    """What the NVFP4 menu line says about this NVIDIA card: RTX 50 (Blackwell, sm_120) has FP4 tensor cores; on older
    RTX cards the experts run at 8-bit precision, which measured slower than UD-IQ4_XS (docs/NVFP4.md)."""
    try:
        arch = int((gpu or {}).get("arch") or 0)
    except (TypeError, ValueError):
        arch = 0
    if arch >= 120:
        return "native FP4 on this card"
    return "runs, but UD-IQ4_XS is faster on this card"


def nvfp4_cpu_note(avx2: bool, avx512: bool) -> str:
    """What the NVFP4 menu line and the install step say about this CPU (#2347).  Its experts that stay on the CPU run
    on AVX-512 rows when the CPU has them, else on AVX2 rows (bit-equal to ggml's AVX2 dot).  Measured (docs/NVFP4.md):
    per core the AVX2 rows are ~10% slower, and on a whole pool that reads DRAM the two are the same speed up to two
    tokens a window (Ryzen 7 255: 58.9 vs 58.7 GB/s) and ~12% apart at four.  A note only: nothing is refused."""
    if avx512 or not avx2:
        return ""                                      # no AVX2 at all: the older-CPU build's own warning says it
    return ("this CPU has no AVX-512, so NVFP4's CPU rows run on AVX2: about as fast when memory-bound, up to ~12% "
            "slower on 4-token windows")


def family_menu_note(family: str, hip: bool, gpu, avx2: bool = True, avx512: bool = True) -> str:
    """The card-, CPU- and RAM-dependent tail of a family's line in the model menu."""
    if family == "nvfp4":
        d = MODELS["NVFP4"]
        cpu = nvfp4_cpu_note(avx2, avx512)
        return (f"   [{nvfp4_card_note(gpu)}; {d['download_gb']:.0f} GB download, ~{d['ram_gb']} GB of RAM keeps every "
                "expert in RAM" + (f"; {cpu}" if cpu else "") + "]")
    if family == "unsloth" and hip:
        return "   [UD-IQ4_XS is the 4-bit choice on AMD and Intel cards; NVFP4 runs on NVIDIA only]"
    if family == "gyro":
        return ("   [for PCs with little memory: VRAM + RAM together 27.6 GB (Gyro-S) or 35 GB (Gyro-M)"
                + ("; AMD is experimental, not for Intel Arc" if hip else "") + "]")
    return ""


def strix_halo_recommends(gpu, ram) -> bool:
    """Is UD-IQ4_XS the recommended model here: a Strix Halo whose unified memory (the OS's RAM plus the BIOS carve-out)
    holds it - the model docs/STRIX_HALO.md measures.  A recommendation only: the menus still list every size."""
    return is_strix_halo(gpu) and bool(gpu.get("uma")) and ram + gpu.get("dedicated_gb", 0.0) >= STRIX_HALO_MIN_GB


def amd_device_access_problem(dev="/dev", access=os.access, listing=glob.glob) -> str | None:
    """Linux AMD: /dev/kfd and the render nodes must be readable and writable by this user (the render / video groups).
    None when they are, else a sentence for the user.  A missing /dev/kfd is not this problem (no driver / no ROCm)."""
    kfd = os.path.join(dev, "kfd")
    if not os.path.exists(kfd):
        return None
    bad = [kfd] if not access(kfd, os.R_OK | os.W_OK) else []
    bad += [n for n in sorted(listing(os.path.join(dev, "dri", "renderD*"))) if not access(n, os.R_OK | os.W_OK)]
    if not bad:
        return None
    return (f"this user cannot open {', '.join(bad)}: ROCm will find no GPU and the engine will fail with 'no ROCm-capable "
            "device' although nothing else holds the GPU. Add your user to the render and video groups: "
            "sudo usermod -aG render,video $USER, then log out and in again")


def gfx1103_notes(gpu, ram) -> list[str]:
    """What setup tells the owner of a Radeon 780M / 760M / 740M (Phoenix / Hawk Point, gfx1103, unified memory): never
    the Strix Halo text, which is about another chip (gfx1151).  Opt-in only (STRATA_EXPERIMENTAL_GFX1103=1)."""
    notes = [f"  Radeon 780M / 760M class (gfx1103, Ryzen 7040 / 8040): the CPU and the GPU share one memory pool ({ram:.0f} GB "
             f"seen by the OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in "
             "that pool and the expert cache is sized from the memory the OS can give back."]
    notes.append("  " + ("The engine is compiled here for gfx1103" if not WIN else "This GPU has no ready-made Windows engine")
                 + " with the native dot4 and WMMA kernels and the 780M's exact speed switches"
                 + " (experimental opt-in, STRATA_EXPERIMENTAL_GFX1103=1; measured on one machine, not validated on a real "
                 "card): see docs/AMD_HIP.md and docs/RADEON_780M.md. This is not a Strix Halo.")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel option ttm.pages_limit raises it; setup changes no "
                     "host setting - a bigger model needs the room, a smaller one runs as it is")
    return notes


def gfx1150_notes(gpu, ram) -> list[str]:
    """What setup tells the owner of a Radeon 890M / 880M (Ryzen AI 300, Strix Point, gfx1150, unified memory): never
    the Strix Halo text, which is about another chip (gfx1151).  Opt-in only (STRATA_EXPERIMENTAL_GFX1150=1)."""
    notes = [f"  Radeon 890M / 880M class (gfx1150, Strix Point): the CPU and the GPU share one memory pool ({ram:.0f} GB "
             f"seen by the OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in "
             "that pool and the expert cache is sized from the memory the OS can give back."]
    notes.append("  " + ("The engine is compiled here for gfx1150" if not WIN else
                         "Windows has no ready-made engine for gfx1150: build one with STRATA_HIP_ARCHS=gfx1150 "
                         "tools\\hip\\build_windows.bat and start setup with --prebuilt dist\\")
                 + " (experimental opt-in, STRATA_EXPERIMENTAL_GFX1150=1, unvalidated: docs/AMD_HIP.md).")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     "default is about half of the RAM). The kernel option ttm.pages_limit raises it; setup changes no "
                     "host setting - a bigger model needs the room, a smaller one runs as it is")
    return notes


def igpu_notes(gpu, ram) -> list[str]:
    """The unified-memory notes for exactly this chip: Strix Halo (gfx1151) or the gfx1150 / gfx1103 opt-in, nothing else."""
    if is_strix_halo(gpu):
        return strix_halo_notes(gpu, ram)
    if GFX1150_OPT_IN and gfx_arch_is(gpu.get("arch"), "gfx1150"):
        return gfx1150_notes(gpu, ram)
    if gfx_arch_is(gpu.get("arch"), "gfx1103"):
        return gfx1103_notes(gpu, ram)
    return []


def unused_apu_notes(chosen, amd, ram) -> list[str]:
    """A Strix Halo / APU that is NOT in the run but sits beside the card that is (a Radeon AI PRO R9700 and the 8060S): its BIOS carve-out is memory
    the OS does not have, and it is the RAM the experts live in.  Nothing is changed: a "!" line says what is lost and what to set."""
    out = []
    used = {g["index"] for g in chosen}
    for g in amd:
        if g.get("uma") and g["index"] not in used and g.get("dedicated_gb", 0.0) > 16:
            out.append(f"!the BIOS gives {g['dedicated_gb']:.0f} GB of your memory to {g['name']}, which this run does not use: the OS has "
                       f"{ram:.0f} GB left for the experts and the desktop, and the engine can run out of RAM. A small carve-out (512 MB - 2 GB, "
                       "'UMA Frame Buffer Size' in the BIOS) gives it back; the GPU reaches the shared memory anyway")
    return out


def strix_halo_notes(gpu, ram) -> list[str]:
    """What setup tells a Strix Halo owner (a leading "!" = a warning): the unified memory, the build, the guide, and the
    memory settings that give the GPU room.  Nothing is changed or refused: recommend, never force."""
    notes = [f"  Strix Halo (gfx1151, Ryzen AI Max): the CPU and the GPU share one memory pool ({ram:.0f} GB seen by the "
             f"OS + a {gpu.get('dedicated_gb', 0.0):.1f} GB BIOS carve-out), so the model's experts live in that pool "
             "and the expert cache is sized from the memory the OS can give back, not from the carve-out."]
    notes.append("  " + ("The engine is compiled here for gfx1151" if not WIN else "Windows uses the ready-made AMD engine, "
                                                                                     "which must be built for gfx1151")
                 + " (experimental, measured on one machine): see docs/STRIX_HALO.md.")
    if not WIN and gpu.get("shared_gb", 0) < 0.75 * ram - 1:
        notes.append(f"!the GPU can reach {gpu.get('shared_gb', 0):.0f} GB of shared memory (the GTT pool; the kernel's "
                     f"default is about half of the RAM). docs/STRIX_HALO.md boots with ttm.pages_limit and "
                     "ttm.page_pool_size to give it most of the RAM; setup changes no host setting - a bigger model "
                     "needs the room, a smaller one runs as it is")
    if gpu.get("dedicated_gb", 0.0) > 16:
        notes.append(f"!the BIOS carve-out is {gpu['dedicated_gb']:.0f} GB: that memory is taken from the OS, while the GPU "
                     "reaches the shared memory anyway. A small carve-out (the BIOS's default or 512 MB - 2 GB) leaves "
                     "the RAM to the model")
    if strix_halo_recommends(gpu, ram):
        notes.append(f"  Recommended model: {STRIX_HALO_MODEL} (the one docs/STRIX_HALO.md measures); the menus list the "
                     "others.")
    return notes


def amd_rank(g):
    """The order setup picks a card in, best first: a discrete card before an APU (its own GDDR is faster than the
    shared LPDDR5X), then the most memory.  A Strix Halo beside a Radeon RX / PRO card is therefore left to --gpu N."""
    return (bool(g.get("uma")), -round(g["vram_gb"]), g["index"])


def amd_pci_devices(sysfs="/sys") -> list[dict]:
    """The AMD (vendor 0x1002) display devices the kernel lists in /sys/class/drm/card*/device, as [{"pci_id",
    "vram_gb", "gtt_gb", "arch"}] with the arch AMD_IGPU_PCI knows ("" when it does not).  The KFD topology answers the
    same without ROCm; this one only names a card the topology does not list (a kernel without /dev/kfd)."""
    found = []
    base = Path(sysfs) / "class/drm"
    if not base.is_dir():
        return found
    for card in sorted(base.glob("card[0-9]*")):
        if not card.name[4:].isdigit():
            continue
        try:
            dev = card / "device"
            if int((dev / "vendor").read_text().strip(), 16) != 0x1002:
                continue
            did = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        row = {"pci_id": did, "arch": AMD_IGPU_PCI.get(did, ""), "vram_gb": 0.0, "gtt_gb": 0.0}
        for key, f in (("vram_gb", "mem_info_vram_total"), ("gtt_gb", "mem_info_gtt_total")):
            try:
                row[key] = int((dev / f).read_text()) / 2 ** 30
            except (OSError, ValueError):
                pass
        found.append(row)
    return found


def amd_gpus(sysfs="/sys"):
    """AMD GPUs from the kernel's KFD topology (the amdgpu driver; no ROCm needed), numbered as HIP numbers them:
    the GPU nodes in order, the CPU nodes skipped.  Integrated GPUs are listed too (not supported).
    sysfs: the tree to read (tools/test_setup_amd.py passes a mocked one).  Windows: amd_gpus_win."""
    if WIN:
        return amd_gpus_win()
    base = Path(sysfs) / "class/kfd/kfd/topology/nodes"
    found = []
    if not base.is_dir():
        return found
    for node in sorted((p for p in base.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            props = {}
            for line in (node / "properties").read_text().splitlines():
                k, _, v = line.partition(" ")
                props[k] = v.strip()
            ver = int(props.get("gfx_target_version") or 0)
            if ver == 0 or int(props.get("simd_count") or 0) == 0:
                continue
        except (OSError, ValueError):
            continue
        arch = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
        dev = Path(sysfs) / f"class/drm/renderD{props.get('drm_render_minor', '')}/device"
        try:
            vram = int((dev / "mem_info_vram_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            vram = 0.0
        try:
            name = (dev / "product_name").read_text().strip() or f"AMD Radeon ({arch})"
        except OSError:
            name = f"AMD Radeon ({arch})"
        if name == f"AMD Radeon ({arch})" and arch in AMD_NAMES:
            name = AMD_NAMES[arch]
            if gfx_arch_is(arch, STRIX_HALO_ARCH) and int(props.get("simd_count") or 0) in STRIX_HALO_BY_SIMDS:
                name = (f"AMD Radeon {STRIX_HALO_BY_SIMDS[int(props['simd_count'])]} "
                        "(Ryzen AI Max, Strix Halo, gfx1151)")   # the CU count tells the 8060S / 8050S / 8040S apart
        try:
            gtt = int((dev / "mem_info_gtt_total").read_text()) / 2 ** 30
        except (OSError, ValueError):
            gtt = 0.0
        g = {"index": len(found), "name": name, "vram_gb": vram, "arch": arch, "driver": "amdgpu", "vendor": "amd"}
        try:
            g["pci_id"] = int((dev / "device").read_text().strip(), 16)
        except (OSError, ValueError):
            pass
        found.append(amd_apply_uma(g, gtt))            # Strix Halo: unified memory (carve-out + shared)
    return found


def amd_problem(g):
    if g["arch"] not in AMD_ARCHS:
        return (f"not supported - Strata's AMD backend runs on {AMD_CARDS} only, this is {g['arch']}"
                + (" (an integrated Radeon, not Strix Halo)" if g["arch"] in ("gfx1150", "gfx1152", "gfx1103") else ""))
    if g.get("cannot_run"):                            # Windows: the installed engine's own check (--list-devices)
        return g["cannot_run"]
    return None


def amd_gpus_win() -> list[dict]:
    """Windows: the AMD GPUs as the HIP runtime numbers them once the HIP engine is installed (hip_devices), else in
    the display-adapter order (amd_gpus_windows)."""
    hip = hip_devices()
    if not hip:
        return amd_gpus_windows()
    halo = [g for g in hip if is_uma_apu(g)]
    if halo:                                           # an APU: the HIP runtime's figure is not the BIOS carve-out
        reg = [r for r in amd_gpus_windows() if is_uma_apu(r)]
        for k, g in enumerate(halo):
            reported = g["vram_gb"]
            g["vram_gb"] = reg[k]["dedicated_gb"] if k < len(reg) else 0.0
            amd_apply_uma(g)
            g["vram_gb"] = max(g["vram_gb"], reported)
    return hip


def amd_parse_gpus(text, amd) -> list:
    """--gpus with AMD cards, numbered as HIP numbers them (setup's list): "1,0", or "all" (every supported card, the
    most VRAM first).  Every chosen card must be one Strata supports (AMD_ARCHS; they may be of different
    architectures: the engine is compiled for each).  Returns the cards, the main one first."""
    usable = [g for g in amd if amd_problem(g) is None]
    if str(text).strip().lower() == "all":
        # "all" means the cards that help: a Strix Halo / APU beside a card with memory of its own adds no memory (its memory is the RAM),
        # draws the desktop, and as a stage it only slows the split down.  Named by number (--gpus 1,0) it is still taken as asked.
        own = [g for g in usable if not g.get("uma")]
        sel = [g["index"] for g in sorted(own if len(own) >= 2 else usable, key=amd_rank)]
        left = [g for g in usable if g["index"] not in sel]
        if left and len(own) >= 2:
            say("  --gpus all leaves out " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in left)
                + ": it shares the system memory and draws the desktop. Name it (--gpus ...) to use it anyway")
    else:
        try:
            sel = [int(x) for x in str(text).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes AMD GPU numbers as setup lists them, e.g. --gpus 1,0 (or --gpus all), not {text!r}")
    if len(sel) < 2 or len(set(sel)) != len(sel):
        fail("--gpus takes two or more different GPUs, e.g. --gpus 1,0 (one GPU: --gpu 1)",
             "this PC has " + (f"{len(usable)} AMD card{'s' if len(usable) != 1 else ''} Strata can use"
                               + (": " + ", ".join(f"GPU {g['index']} ({g['name']})" for g in usable) if usable else "")))
    byid = {g["index"]: g for g in amd}
    for i in sel:
        g = byid.get(i)
        p = "not found on this PC" if g is None else amd_problem(g)
        if p is not None:
            fail(f"AMD GPU {i}{'' if g is None else ' (' + g['name'] + ')'} cannot be used: {p}",
                 ("use these together: --gpus " + ",".join(str(x["index"]) for x in usable)) if len(usable) >= 2 else
                 ("use one card: --gpu " + str(usable[0]["index"])) if usable else f"the AMD backend runs on {AMD_CARDS}")
    return [byid[i] for i in sel]


# ------------------------------------------------------------------------------------------------ AMD on Windows
# Windows has no KFD topology: the cards are found from the display adapters (Win32_VideoController: the ones present,
# with their PCI ids) and the display-class registry (each adapter's 64-bit VRAM size), before any AMD software is
# needed.  The engine is the ready-made HIP one (WIN_HIP_ASSET: strata.exe, strata-device.exe and the ROCm libraries it
# loads, built by tools/hip/build_windows.bat); it needs only the AMD driver.  Once it is installed, the cards are
# numbered as the HIP runtime numbers them (`strata-device --list-devices`): an integrated Radeon takes HIP's device 0
# and pushes the discrete card to 1, which the display-adapter order does not show (#325).
WIN_HIP_ASSET = "strata-windows-x64-hip.zip"   # releases up to 0.1.45; 0.2 releases publish per-card packages in the signed feed (engine_from_feed)
WIN_HIP_MIN_ENGINE = max(MIN_ENGINE, (0, 1, 33))         # the first release with a Windows HIP engine
WIN_AMD_DRIVER = "https://www.amd.com/en/support/download/drivers.html"
# PCI device ids (VEN_1002) of the cards the AMD backend knows; the names below cover a card whose id is not listed
_WIN_AMD_DID = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100",            # RX 7900 XTX/XT/GRE, W7900, W7800
                0x747E: "gfx1101",                                                  # RX 7800 XT / 7700 XT
                0x7480: "gfx1102",                                                  # RX 7600 / 7600 XT
                0x7590: "gfx1200",                                                  # RX 9060 XT
                0x7550: "gfx1201", 0x7551: "gfx1201",                               # RX 9070 / 9070 XT, AI PRO R9700
                0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030",            # RX 6800 / 6800 XT / 6900 XT / 6950 XT
                0x73DF: "gfx1031",                                                  # RX 6700 XT / 6750 XT / 6800M (#881)
                **AMD_IGPU_PCI}                                                     # Ryzen APUs: 0x1586 = Strix Halo (gfx1151)
_WIN_AMD_NAME = ((re.compile(r"\b80[456]0S\b", re.I), "gfx1151"),                   # Radeon 8060S / 8050S / 8040S
                 (re.compile(r"\b8[89]0M\b", re.I), "gfx1150"),                     # Radeon 890M / 880M
                 (re.compile(r"\b8[46]0M\b", re.I), "gfx1152"),                     # Radeon 860M / 840M
                 (re.compile(r"\b7[468]0M\b", re.I), "gfx1103"),                    # Radeon 780M / 760M / 740M
                 (re.compile(r"\b9070\b|R9700", re.I), "gfx1201"),
                 (re.compile(r"\b9060\b", re.I), "gfx1200"),
                 (re.compile(r"RX\s*7900|W7900|W7800", re.I), "gfx1100"),
                 (re.compile(r"RX\s*7800|RX\s*7700(?!\s*S)|W7700", re.I), "gfx1101"),
                 (re.compile(r"RX\s*7600|W7600|W7500", re.I), "gfx1102"),
                 (re.compile(r"RX\s*6800(?!\s*[MS])|RX\s*6900|RX\s*6950|W6800", re.I), "gfx1030"))
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _win_display_adapters() -> list[dict]:
    """The display adapters present ({"name", "pnp"}), from Win32_VideoController."""
    ps = ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "      # the pipe is read as UTF-8 (#4675)
          "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID + '|' + "
          "$_.AdapterRAM }")
    text = out(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    found = []
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[1]:
            ram = parts[2] if len(parts) > 2 else ""
            found.append({"name": parts[0].strip(), "pnp": parts[1].strip(),
                          "ram": int(ram) if ram.strip().isdigit() else 0})
    return found


def _win_display_registry() -> list[dict]:
    """Each display driver instance's description, matching PCI id and VRAM size (the 64-bit value; the WMI one stops
    at 4 GB), from the display-class registry key - readable without admin."""
    found = []
    try:
        import winreg
        cls = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except (ImportError, OSError):
        return found
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(cls, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(cls, sub)
        except OSError:
            continue
        vals = {}
        for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize", "DriverVersion"):
            try:
                vals[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                pass
        found.append(vals)
    return found


def _pci_device_id(text: str) -> int | None:
    m = re.search(r"VEN_1002&DEV_([0-9A-F]{4})", str(text or ""), re.I)
    return int(m.group(1), 16) if m else None


def win_amd_arch(device_id: int | None, name: str) -> str:
    """A Windows AMD adapter's architecture from its PCI device id, else its name; "" when it is none Strata knows
    (an integrated Radeon, an older card)."""
    if device_id in _WIN_AMD_DID:
        return _WIN_AMD_DID[device_id]
    return next((a for rx, a in _WIN_AMD_NAME if rx.search(name or "")), "")


def amd_gpus_windows(adapters=None, registry=None) -> list[dict]:
    """The AMD display adapters present, in display-adapter order (setup's numbering until the HIP engine is
    installed), with the arch Strata would run them as ("" = unknown: listed, not supported).  adapters / registry:
    tools/test_setup_amd.py passes mocked ones."""
    adapters = _win_display_adapters() if adapters is None else adapters
    registry = _win_display_registry() if registry is None else registry
    if not adapters:                                   # no WMI answer: the registry alone (it can list removed cards)
        adapters = [{"name": r.get("DriverDesc", ""), "pnp": r.get("MatchingDeviceId", ""), "ram": 0} for r in registry]
    used = set()
    found = []
    for ad in adapters:
        did = _pci_device_id(ad.get("pnp"))
        if did is None:
            continue                                   # not an AMD (VEN_1002) PCI device
        vram, driver = 0.0, ""
        for k, r in enumerate(registry):               # the same card's driver instance: its 64-bit VRAM size
            if k in used or _pci_device_id(r.get("MatchingDeviceId")) != did:
                continue
            if (r.get("DriverDesc") and ad.get("name") and "\ufffd" not in ad["name"]
                    and r["DriverDesc"].strip() != ad["name"].strip()):
                continue
            used.add(k)
            mem = r.get("HardwareInformation.qwMemorySize")
            if isinstance(mem, bytes):
                mem = int.from_bytes(mem[:8], "little")
            vram = int(mem) / 2 ** 30 if isinstance(mem, int) and mem > 0 else 0.0
            driver = str(r.get("DriverVersion") or "")
            break
        if vram == 0.0 and ad.get("ram", 0) > 0:
            vram = ad["ram"] / 2 ** 30                 # WMI's 32-bit figure (at most 4 GB)
        arch = win_amd_arch(did, ad.get("name", ""))
        name = ad.get("name") or AMD_NAMES.get(arch, f"AMD Radeon (device {did:04X})")
        g = {"index": len(found), "name": name, "vram_gb": vram, "arch": arch or f"unknown (PCI {did:04X})",
             "driver": driver or "amd", "vendor": "amd", "pci_id": did}
        found.append(amd_apply_uma(g))                 # Strix Halo: the registry's figure is the BIOS carve-out only
    return found


def engine_env(eng: Path) -> dict:
    """The environment for running an installed engine's own programs (`strata --families`, `strata-device
    --list-devices`): the engine's ROCm library folders (its BUILD.json "lib_dirs") first on the dynamic-library path -
    on Windows that is PATH, where the ready-made HIP engine keeps hipblas.dll / libhipblaslt.dll in its rocm/bin and the
    loader finds them only through PATH (#2372: a launch without it shows a "not found" dialog per DLL and prints
    nothing); on Linux and elsewhere it is LD_LIBRARY_PATH, where a probe without it exits 127 on libhipblas.so.3 and
    prints nothing."""
    env = dict(os.environ)
    dirs = [str(d) for d in hip_lib_dirs(eng)]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


def hip_probe_path() -> Path:
    """Where the installed engine's `strata-device` is: engine/ beside setup.py (a checkout), else the HIP engine package in the
    component folder (the app and `strata update` keep it there, #3418: with only engine/ looked at, the app numbered the cards in
    display-adapter order and an iGPU on HIP ordinal 0 made "GPU 0" the wrong card)."""
    name = "strata-device.exe" if WIN else "strata-device"
    own = ROOT / "engine" / name
    if own.exists():
        return own
    try:
        from strata_control import engine_component as EC
        found = EC.resolve(backend="hip", full=False)
        return Path(found.root) / name
    except Exception:                                  # noqa: BLE001 - no component, a damaged one: the old answer
        return own


def hip_devices(probe: Path | None = None, text: str | None = None) -> list[dict] | None:
    """The GPUs the HIP runtime enumerates, numbered as HIP_VISIBLE_DEVICES numbers them, from the installed engine's
    `strata-device --list-devices`; None when there is no HIP engine here or it does not answer.  text: its output
    (tests)."""
    if text is None:
        probe = probe or hip_probe_path()
        hip_engine = False
        for record in ("BUILD.json", "engine-component.json"):     # a checkout's build record, or a component's manifest
            try:
                hip_engine = json.loads((probe.parent / record).read_text(encoding="utf-8-sig")).get("backend") == "hip"
            except (OSError, ValueError, AttributeError):
                continue
            if hip_engine:
                break
        if not probe.exists() or not hip_engine:
            return None
        try:
            hip_runtime_beside_exe(probe.parent)       # #468 #461: not the driver's System32 copy
            r = subprocess.run([str(probe), "--list-devices"], capture_output=True, text=True, errors="replace", timeout=120,
                               cwd=str(probe.parent), env=engine_env(probe.parent))
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode != 0:
            return None
        text = r.stdout
    found = []
    for line in text.splitlines():
        m = re.match(r"device\s+(\d+):\s*(.*)$", line.strip())
        if m:
            found.append({"index": int(m.group(1)), "name": m.group(2).strip(), "vram_gb": 0.0, "arch": "",
                          "driver": "hip", "vendor": "amd"})
            continue
        if not found:
            continue
        a = re.match(r"arch\s+(gfx[0-9a-f]+)\s*,\s*([\d.]+)\s*GiB", line.strip())
        if a and not found[-1]["arch"]:
            found[-1]["arch"], found[-1]["vram_gb"] = a.group(1), float(a.group(2))
        elif line.strip().startswith("cannot run:"):
            found[-1]["cannot_run"] = line.strip()[len("cannot run:"):].strip()
    for g in found:
        if not g["arch"]:
            g["arch"] = "unknown"
        if g["arch"] in AMD_NAMES and g["name"] in ("", "AMD Radeon Graphics"):
            g["name"] = AMD_NAMES[g["arch"]]
    return found


def hip_lib_dirs(eng: Path) -> list[Path]:
    """Where the ready-made Windows HIP engine's ROCm DLLs are (its BUILD.json "lib_dirs", relative to engine/)."""
    try:
        rel = json.loads((eng / "BUILD.json").read_text(encoding="utf-8")).get("lib_dirs") or []
    except (OSError, ValueError):
        rel = []
    return [eng / d for d in rel if (eng / d).is_dir()]


# #468 #461: the HIP runtime the ready-made engine was built with, next to strata.exe.  Windows looks for an imported
# DLL in the exe's folder, then System32, and only then on PATH (where rocm/bin is): an AMD driver that installs its own
# amdhip64_7.dll in System32 won, and the bundled rocBLAS/hipBLAS ran on that runtime - an access violation (W7900) or
# hipErrorInvalidDeviceFunction (7900 XTX) on the first prompt.  Only the runtime and the compiler it loads by name:
# rocBLAS/hipBLASLt stay in rocm/bin, where they find their kernel libraries and ../.kpack.
HIP_RUNTIME_DLLS = ("amdhip64_*.dll", "amd_comgr*.dll")


def hip_runtime_closure(d: Path) -> list[str]:
    """The DLLs of `d` (the engine's rocm/bin) that amdhip64_7.dll and amd_comgr.dll need, the two included, read from
    their PE import tables and followed recursively (tools/hip/dll_closure.py).  #461: amdhip64_7.dll imports
    rocm_kpack.dll, which imports the MSVC runtime; with only the two DLLs beside strata.exe the full-path load fails
    (error 126), Windows falls back to System32's copy of the runtime, and the first big prompt dies with
    hipErrorInvalidDeviceFunction.  Falls back to the names above when the helper cannot be read."""
    roots = sorted({p.name for pat in HIP_RUNTIME_DLLS for p in d.glob(pat)})
    try:
        sys.path.insert(0, str(ROOT / "tools" / "hip"))
        from dll_closure import dll_closure
        return dll_closure(d, roots)
    except Exception:                                  # noqa: BLE001 - a missing helper must not stop setup
        return roots
    finally:
        if str(ROOT / "tools" / "hip") in sys.path:
            sys.path.remove(str(ROOT / "tools" / "hip"))


def hip_runtime_beside_exe(eng: Path) -> None:
    """Copy the bundled HIP runtime DLLs and everything they import from rocm/bin next to the engine's exes when
    missing or different (a 0.1.34 install, whose zip had them in rocm/bin only, is fixed on its next start; a
    0.1.40.2 install, which got only the two runtime DLLs, gets rocm_kpack.dll and the C++ runtime now, #461)."""
    for d in hip_lib_dirs(eng):
        for name in hip_runtime_closure(d):
            src = d / name
            dst = eng / src.name
            try:
                if dst.exists() and dst.stat().st_size == src.stat().st_size and \
                        dst.stat().st_mtime >= src.stat().st_mtime:
                    continue
                shutil.copy2(src, dst)
            except OSError as e:                   # e.g. the engine is running and holds the old copy
                warn(f"could not put {src.name} next to the AMD engine ({e}); if the engine stops on its first "
                     "request, close Strata and run START-HERE.bat again")


def hip_match(card: dict, listed: list[dict], hip: list[dict]) -> dict | None:
    """The HIP device that is setup's `card` (from `listed`, the display-adapter order): the k-th device of the same
    architecture, k = the card's rank among the listed cards of that architecture.  None when HIP has no such card."""
    same = [g["index"] for g in listed if g["arch"] == card["arch"]]
    k = same.index(card["index"]) if card["index"] in same else 0
    cand = [g for g in hip if g["arch"] == card["arch"]]
    return cand[k] if k < len(cand) else None


def hip_card(eng: Path, gpu: dict, listed: list[dict]) -> dict:
    """Windows: setup's chosen AMD card as the installed HIP engine numbers it (HIP_VISIBLE_DEVICES), checked by the
    engine itself before the model download: an integrated Radeon is HIP's device 0 (#325), and a PC without a
    working AMD driver stops here with what to install."""
    hip = hip_devices(eng / "strata-device.exe")
    hint = (f"install or update the AMD driver (AMD Software: Adrenalin Edition) from {WIN_AMD_DRIVER}, restart the "
            f"PC and run this again; {eng / 'strata-device.exe'} --list-devices shows what the HIP runtime sees")
    if not hip:
        fail("the AMD HIP runtime finds no GPU (the ready-made engine's device check)", hint)
    m = hip_match(gpu, listed, hip) if gpu.get("driver") != "hip" else \
        next((x for x in hip if x["index"] == gpu["index"]), None)
    if m is None:
        fail(f"the HIP runtime does not list your {gpu['name']} ({gpu['arch']})", hint)
    if amd_problem(m):
        fail(f"HIP device {m['index']} ({m['name']}) cannot be used: {amd_problem(m)}")
    if gpu.get("driver") != "hip" and m["index"] != gpu["index"]:
        ok(f"HIP numbers this card {m['index']} (an integrated GPU comes first): the engine is pointed at it")
    vram = gpu["vram_gb"] if gpu.get("uma") else (m["vram_gb"] or gpu["vram_gb"])   # an APU: not the runtime's figure
    return {**gpu, "index": m["index"], "count": len(hip), "vram_gb": vram, "driver": "hip"}


def engine_from_feed(eng: Path, updating: bool = False) -> Path | None:
    """#3469 #3437: the 0.2 releases no longer carry the flat engine zips (strata-windows-x64*.zip); they publish one package
    per card in the signed update feed.  `strata update --no-control` fetches, verifies and test-starts the package for this PC
    (into a folder of its own under the install), and its files then become the installed engine in `eng`, the same way a
    zip's do.  None when it cannot be had (no internet, nothing for this card): the caller goes on as before."""
    acquisition.require("the ready-made engine")
    from strata_control import engine_component as ec
    store = home() / "engine-feed"
    env = dict(os.environ)
    env.pop("STRATA_NO_ACQUIRE", None)
    env["STRATA_COMPONENTS_DIR"] = str(store)
    say("  Fetching the ready-made engine from the signed update feed ...")
    try:
        run = subprocess.run([sys.executable, str(ROOT / "strata.py"), "update", "--no-control", "--output", "json"],
                             env=env, cwd=str(ROOT), capture_output=True, text=True, errors="replace", timeout=3600)
    except (OSError, subprocess.SubprocessError) as e:
        warn(f"the update feed could not be run ({e})")
        return None
    if run.returncode != 0:
        why = (run.stderr or run.stdout or "").strip().splitlines()[-1:] or [f"exit {run.returncode}"]
        warn(f"no engine from the update feed: {why[0][:200]}")
        return None
    try:
        component = ec.resolve(store, full=True)
    except ec.ComponentProblem as e:
        warn(f"the engine from the update feed is not usable: {e}")
        return None
    eng.mkdir(exist_ok=True)
    tmp = eng / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(component.root, tmp)
    try:
        meta = json.loads((tmp / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if meta.get("source") != "prebuilt":                # the setup's own books: a ready-made engine, not one compiled here
        meta["source"] = "prebuilt"
        (tmp / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (eng / "BUILD.json").replace(eng / "BUILD.json.prev") if (eng / "BUILD.json").exists() else None
    install_unpacked(tmp, eng)
    shutil.rmtree(tmp, ignore_errors=True)
    if not (eng / EXE).exists():
        fail("the engine package from the update feed has no " + EXE)
    if not WIN:
        for x in (EXE, VEXE):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    ok(f"ready-made engine {meta.get('version', '')} from the update feed ({component.root.name})")
    return eng


def get_prebuilt_hip(url_base, gpu, updating=False) -> Path | None:
    """The ready-made Windows HIP engine (WIN_HIP_ASSET) in engine/, kept between runs; None when it cannot be had
    (not published for this version, no internet) or has no code for the card."""
    acquisition.require("the ready-made AMD engine")
    eng = ROOT / "engine"
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists():
        try:
            meta = json.loads(info.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
        if meta.get("backend") == "hip" and meta.get("source") == "prebuilt" and ver >= WIN_HIP_MIN_ENGINE and \
                gpu["arch"] in arch_names(meta.get("archs")) and not updating:
            ok("ready-made AMD engine already installed")
            return eng
    if not url_base:
        return None
    eng.mkdir(exist_ok=True)
    z = eng / WIN_HIP_ASSET
    bases = prebuilt_bases(url_base)
    for i, base in enumerate(bases):
        if not base.startswith(("http://", "https://")):
            break
        try:
            req = urllib.request.Request(base + WIN_HIP_ASSET, method="HEAD", headers={"User-Agent": "strata-setup"})
            _urlopen(req, timeout=60).close()
            break
        except OSError as e:
            if i + 1 < len(bases):
                say(f"  No ready-made AMD engine for v{source_version()} ({e}): the latest release instead")
                continue
            warn(f"no ready-made AMD engine at {base} ({e})")
            return engine_from_feed(eng, updating)
    say("  Downloading the ready-made Strata engine for AMD GPUs (with the ROCm libraries it uses) ...")
    download(base + WIN_HIP_ASSET, z, "Strata AMD engine")
    try:
        verify_engine_archive(z, WIN_HIP_ASSET, base)
    except UnverifiedEngine as e:
        engine_refused(WIN_HIP_ASSET, e, updating)
        return None
    tmp = eng / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    try:
        meta = json.loads((tmp / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
    why = None
    if meta.get("backend") != "hip" or not (tmp / EXE).exists():
        why = "it is not a HIP engine"
    elif ver < WIN_HIP_MIN_ENGINE:
        why = f"it is version {meta.get('version')}; this setup needs {'.'.join(map(str, WIN_HIP_MIN_ENGINE))}"
    elif gpu["arch"] not in arch_names(meta.get("archs")):
        why = (f"it is built for {', '.join(arch_names(meta.get('archs')))}; your GPU is {gpu['arch']}.  A card it has no code "
               "for runs from a self-build of the engine: see docs/AMD_HIP.md (--build)")   # #881
        if is_strix_halo(gpu):                         # never a zip without gfx1151 code: say so, and what to do
            why = (f"it is built for {', '.join(arch_names(meta.get('archs'))) or 'other GPUs'} and has no gfx1151 code: your "
                   f"{gpu['name']} (Strix Halo) needs a ready-made engine from Strata 0.1.40 or newer, whose zip is "
                   "built for gfx1151 too.  Until then: the Linux guide docs/STRIX_HALO.md, or compile the engine "
                   "here with tools\\hip\\build_windows.bat (its default list has gfx1151) and start setup with "
                   "--prebuilt <its dist folder>")
    if why:
        warn(f"the ready-made AMD engine at {base} cannot be used: {why}")
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    install_unpacked(tmp, eng)
    shutil.rmtree(tmp, ignore_errors=True)
    drop_archive(z)
    hip_runtime_beside_exe(eng)                        # #468 #461
    ok(f"ready-made AMD engine {meta.get('version', '')} for {', '.join(arch_names(meta.get('archs')))} "
       f"(ROCm {meta.get('rocm', '?')})")
    return eng


def rocm_version(root):
    """(major, minor) of a ROCm install, from rocm-core's header; None when it has none."""
    try:
        text = (Path(root) / "include" / "rocm-core" / "rocm_version.h").read_text(encoding="utf-8")
        return tuple(int(re.search(rf"#define\s+ROCM_VERSION_{k}\s+(\d+)", text).group(1)) for k in ("MAJOR", "MINOR"))
    except (OSError, AttributeError, ValueError):
        return None


def rocm_dev_missing(sysroot: Path) -> list:
    """#446: the HIP development files the engine's build needs that a system ROCm lacks (a runtime-only install has
    hipcc and libhipblas but not these, and cmake's enable_language(HIP) then fails on the hip-lang package)."""
    lang = "cmake/hip-lang/hip-lang-config.cmake"     # where CMake's HIP support looks for it
    need = {"lib/" + lang: [sysroot / d / lang for d in ("lib", "lib64", "lib/x86_64-unknown-linux-gnu")],
            "include/hip/hip_runtime.h": [sysroot / "include" / "hip" / "hip_runtime.h"]}
    return [name for name, paths in need.items() if not any(p.is_file() for p in paths)]


def rocm_root(archs):
    """ROCm for compiling and running the HIP engine for `archs` (one arch or a list: the cards of a layer split):
    (root, library folders).  A system ROCm 7 with hipcc, hipBLAS and the HIP development files (#446), else AMD's
    TheRock wheels (rocm_wanted: a version per card family, from the family's index) installed into .venv."""
    archs = [archs] if isinstance(archs, str) else list(archs)
    sysroot = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
    if (sysroot / "bin" / "hipcc").exists() and list((sysroot / "lib").glob("libhipblas.so*")):
        ver = rocm_version(sysroot)
        missing = rocm_dev_missing(sysroot)
        if (ver is None or ver >= ROCM_SYSTEM_MIN) and not missing:
            return sysroot, [str(sysroot / "lib")]
        if ver is not None and ver < ROCM_SYSTEM_MIN:
            warn(f"the ROCm in {sysroot} is {ver[0]}.{ver[1]}; Strata needs {ROCM_SYSTEM_MIN[0]}.{ROCM_SYSTEM_MIN[1]} "
                 "or newer: using AMD's wheels in .venv instead")
        else:                                          # #446: a runtime-only ROCm (no -dev packages): cmake would fail
            warn(f"the ROCm in {sysroot} has no HIP development files ({', '.join(missing)}): using AMD's wheels in "
                 ".venv instead (or install them, e.g. AMD's amdrocm-core-dev package for your ROCm and card)")
    indexes = list(dict.fromkeys(rocm_index(a) for a in archs))
    if len(indexes) > 1:                               # TheRock's wheels hold one GPU family's libraries
        fail(f"cards of two GPU families ({', '.join(archs)}) need a system ROCm 7 (in /opt/rocm): AMD's Python "
             "wheels come per family", "install ROCm 7 system-wide, or use cards of one family (--gpu N for one card)")
    index = indexes[0]
    stamp = Path(sys.prefix) / ".strata-rocm.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    wheel = None
    if have.get("index") == index and have.get("version") and not ROCM_VERSION_OVERRIDE \
            and (Path(sys.executable).parent / "rocm-sdk").exists() and have["version"] == ROCM_FAMILY_PINS.get(rocm_family(index), ROCM_VERSION_DEFAULT):
        wheel = have["version"]                         # the family's pin is installed: no lookup
    if wheel is None:
        wheel, note = rocm_wanted(index)
        if note:
            warn(note)
    if have.get("version") != wheel or have.get("index") != index:
        say(f"  Installing ROCm {wheel} for AMD GPUs into .venv (AMD's TheRock wheels, ~10 GB, no sudo) ...")
        pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--index-url", index]
        if have.get("version") == wheel:               # the same version for another GPU family: its own libraries
            run(pip + ["--force-reinstall", "--no-deps", f"rocm=={wheel}"])
        run(pip + [f"rocm[libraries,devel]=={wheel}"])
        stamp.write_text(json.dumps({"version": wheel, "index": index}))
    sdk = Path(sys.executable).parent / "rocm-sdk"
    root = Path(out([str(sdk), "path", "--root"]).strip())
    if not (root / "llvm" / "bin" / "clang++").exists():
        fail(f"ROCm was installed but its compiler is missing ({root})",
             f"remove {stamp} and run this again; or install ROCm 7 system-wide")
    # the card family's libraries only (gfx120X-all -> _rocm_sdk_libraries_gfx120X_all): another family's
    # libhipblaslt.so first on the path would have no kernels for this card
    family = "_rocm_sdk_libraries_" + index.rstrip("/").rsplit("/", 1)[-1].replace("-", "_")
    dirs = [str(root / "lib")]
    for sp in {Path(p) for p in sys.path if p.endswith("site-packages")}:
        libs = sorted(sp.glob("_rocm_sdk_libraries_*"))
        libs = [d for d in libs if d.name.lower() == family.lower()] or libs
        dirs += [str(d / "lib") for d in libs if (d / "lib").is_dir()]
    ok(f"ROCm: {root}")
    return root, list(dict.fromkeys(dirs))


def hipblaslt_version(lib_dirs):
    """The installed hipBLASLt's version as the engine reads it (hipblasLtGetVersion: 1.4.1 -> 100401), from the
    header of the ROCm whose libraries the engine loads; None when not found."""
    for d in lib_dirs:
        try:
            text = (Path(d).parent / "include" / "hipblaslt" / "hipblaslt-version.h").read_text(encoding="utf-8")
            v = [int(re.search(rf"#define\s+HIPBLASLT_VERSION_{k}\s+(\d+)", text).group(1))
                 for k in ("MAJOR", "MINOR", "PATCH")]
        except (OSError, AttributeError, ValueError):
            continue
        return v[0] * 100000 + v[1] * 100 + v[2]
    return None


def hipblaslt_table(arch, lib_dirs, ver=None, extra_dirs=()):
    """<arch>-hipblaslt-<version>.txt for this card AND the installed hipBLASLt, else None: its solution
    ids are valid only for that pair (the engine refuses any other table and uses plain hipBLAS).  ver: the
    hipBLASLt version the ready-made engine ships (its BUILD.json), else read from the installed headers.  extra_dirs: the
    engine folder's own tuning/ first (a ready-made engine carries its tables), then tools/hip of this checkout.
    STRATA_HIPBLASLT_TABLE=off: none (strata_control/hipblaslt_tuning.py)."""
    from strata_control import hipblaslt_tuning as HT
    if HT.switched_off():
        say(f"  hipBLASLt tuning table: off ({HT.SWITCH}=off); the prompt's dense matrix products use plain hipBLAS")
        return None
    ver = ver or hipblaslt_version(lib_dirs)
    for folder in (*extra_dirs, ROOT / "tools" / "hip"):
        table = Path(folder) / HT.table_name(arch, ver or 0)
        if ver is not None and table.exists() and HT.header_ok(table, arch, ver):
            ok(f"hipBLASLt tuning table: {table.name} (faster prompts)")
            return table
    have = sorted(p.name for p in (ROOT / "tools" / "hip").glob(f"{arch}-hipblaslt-*.txt"))
    warn(f"no hipBLASLt tuning table for {arch} with hipBLASLt {ver or '(version unknown)'}"
       + (f" (have: {', '.join(have)})" if have else "")
       + ": the prompt's dense matrix products use plain hipBLAS (tools/hip/tune_hipblaslt makes a table: "
         "docs/AMD_HIP.md, Tuning table)")
    return None


def build_engine_hip(gpu, llama, vision="none") -> Path:
    """Compile the HIP engine for this AMD GPU into engine/ (again only when its source changed: a `git pull`).
    gpu["archs"]: every architecture it needs code for (the cards of a layer split), else gpu["arch"].  vision "cpu"
    (#304): the image encoder too, for the CPU (there is no HIP encoder build yet)."""
    acquisition.require("an engine build")
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    archs = sorted(set(gpu.get("archs") or [gpu["arch"]]))
    has_archs = set(archs) <= set(arch_names(meta.get("archs")))
    floor = cpu_floor(cpu_info()[1])                     # "" on an AVX2 CPU: the normal engine
    engine_ok = meta.get("backend") == "hip" and (eng / EXE).exists() and meta.get("src") == src and has_archs and \
        (meta.get("isa_floor") or "") == floor and not stale_binary(eng / EXE)
    vision_ok = vision == "none" or ((eng / VEXE).exists() and meta.get("vision_src") == vsrc and
                                     (vision != "gpu" or ((eng / HVEXE).exists() and set(archs) <= set(arch_names(meta.get("vision_archs"))))))
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    if engine_ok:
        return build_vision_hip(eng, stamp, meta, llama, vsrc, archs, vision)
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        fail("a C++ compiler and git are needed to compile the AMD engine",
             "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    say("  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
        if meta.get("backend") == "hip" and (eng / EXE).exists() and has_archs
        else f"  Compiling the Strata engine for your AMD GPU{'s' if len(archs) > 1 else ''} ({', '.join(archs)}; "
             "10-20 minutes, once) ...")
    cmake_build(ROOT, ROOT / "build-hip", "strata",
                ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                 "-DSTRATA_PREFILL_MMQ=ON", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON", "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                 f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                 "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                 f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                 f"-DSTRATA_GGML_DIR={llama}", *isa_floor_defs(floor, ROOT / "build-hip", meta)], None, "")
    if stale_binary(ROOT / "build-hip" / EXE):            # #2054: no stamp for a binary of another version
        fail(f"the engine build finished without a new engine ({EXE} is version "
             f"{binary_engine_version(ROOT / 'build-hip' / EXE)}, this source is {source_version()}): "
             "the installed engine is left as it was", "delete the folder build-hip and run this again")
    shutil.copy2(ROOT / "build-hip" / EXE, eng / EXE)
    meta = {"source": "local-hip", "backend": "hip", "version": source_version(), "archs": archs, "vision": "none",
            "lib_dirs": dirs, "src": src, **({"isa_floor": floor} if floor else {})}
    if vision != "none":
        return build_vision_hip(eng, stamp, meta, llama, vsrc, archs, vision)
    stamp.write_text(json.dumps(meta, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


def hip_vision(asked, vram_gb: float = 0.0) -> str:
    """The picture reader wanted with the AMD backend (--vision): when nothing was asked or "yes" (pictures are on by default) the
    default of strata_control/vision_reader.py for this card - the CPU reader under 24 GB, the GPU reader from 24 GB (owner decision
    2026-10-18, every make of card); "gpu" / "cpu" as asked (#304), "none" for no.  "gpu" is a wish: hip_reader() settles it against
    what the engine folder carries (the HIP reader, else the CPU one) once the engine is known."""
    from strata_control import vision_reader as VR
    return VR.wanted(asked, vram_gb)


def reader_archs(eng: Path) -> list:
    """The GPU architectures the engine folder's HIP picture reader was built for: its BUILD.json (vision_archs, else archs),
    else the component manifest's archs; [] when neither says (the reader then checks the card itself when it starts)."""
    for name, keys in (("BUILD.json", ("vision_archs", "archs")), ("engine-component.json", ("archs",))):
        try:
            data = json.loads((eng / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        for key in keys:
            if isinstance(data, dict) and arch_names(data.get(key)):
                return arch_names(data.get(key))
    return []


def hip_reader(eng: Path, vision: str, gpu: dict):
    """AMD: the picture reader this engine folder gets for `vision` ("gpu" or "cpu"; strata_control/vision_reader.py): the HIP
    reader with the CPU reader to fall back on, else the CPU reader.  Returns (vision, Reader); a Reader of mode "none" means
    the folder has no reader.  The one plain note (a card the HIP reader has no code for, no HIP reader in this engine) is said."""
    from strata_control import vision_reader as VR
    r = VR.choose(eng, vision, windows=WIN, gpu_arch=str(gpu.get("arch") or ""), archs=reader_archs(eng))
    if r.note:
        warn(r.note)
    return r.mode, r


def build_vision_hip(eng: Path, stamp: Path, meta: dict, llama, vsrc, archs: list, vision: str) -> Path:
    """The picture readers beside the HIP engine: the CPU reader always (the fallback), and for "gpu" the HIP reader
    (tools/vision, STRATA_VISION_HIP) for the same architectures.  The HIP reader needs the ROCm environment the engine build
    sets; a build that fails leaves the CPU reader and says so (pictures still work, on the CPU)."""
    if vision == "gpu" and not (shutil.which("hipconfig") or os.environ.get("ROCM_PATH")):
        root, dirs = rocm_root(archs)                    # the engine was built earlier: set the same environment again
        os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                           "HIP_PATH": str(root)})
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    build_vision_cpu(eng, stamp, meta, llama, vsrc)
    meta = json.loads(stamp.read_text(encoding="utf-8"))
    if vision != "gpu":
        return eng
    root, dirs = rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    say("  Compiling the picture reader for your AMD GPU (a few minutes, once) ...")
    try:
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision-hip", "strata-vision",
                    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_HIP=ON", "-DSTRATA_VISION_HIP_ARCHS=" + ";".join(archs),
                     "-DSTRATA_PORTABLE=OFF", f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}",
                     f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}", "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                     f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}"], None, "")
        shutil.copy2(ROOT / "build-vision-hip" / "bin" / HVEXE, eng / HVEXE)
    except SystemExit:
        raise
    except Exception as e:                               # noqa: BLE001 - the CPU reader is there; pictures work without the GPU one
        warn(f"the GPU picture reader did not compile ({e}): pictures are read on the CPU")
        return eng
    stamp.write_text(json.dumps({**meta, "vision": "gpu", "vision_archs": archs, "vision_gpu": HVEXE, "vision_cpu": VEXE}, indent=1))
    ok(f"picture reader (GPU): {eng / HVEXE}, (CPU fallback): {eng / VEXE}")
    return eng


def build_vision_cpu(eng: Path, stamp: Path, meta: dict, llama, vsrc) -> Path:
    """#304: the CPU image encoder beside the HIP engine (tools/vision without CUDA), recorded in its BUILD.json."""
    if not ((eng / VEXE).exists() and meta.get("vision_src") == vsrc):
        say("  Compiling the image encoder (for the CPU) ...")
        cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision", "strata-vision",
                    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=OFF"],
                    find_vcvars() if WIN else None,
                    "build-vision-cpu.bat" if WIN else "")   # #881: MSVC's environment on Windows, as the CUDA path has
        shutil.copy2(ROOT / "build-vision" / "bin" / VEXE, eng / VEXE)
    stamp.write_text(json.dumps({**meta, "vision": "cpu", "vision_src": vsrc}, indent=1))
    ok(f"engine: {eng / EXE}, image encoder (CPU): {eng / VEXE}")
    return eng


# ------------------------------------------------------------------------------------------------ image generation
IMAGE_ENGINE = ROOT / "engine-image"           # sd-cli for Qwen-Image-2.1-Turbo (tools/image/qimage.py), --images


def _qimage():
    if str(ROOT / "tools" / "image") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools" / "image"))
    import qimage
    return qimage


def get_sd_cpp() -> Path:
    """stable-diffusion.cpp and its ggml at the pinned commits (tools/image/qimage.py), as zips: no git."""
    QI = _qimage()
    src = ROOT / "third_party" / "stable-diffusion.cpp"
    if QI.source_ok(src):
        return src
    z1 = ROOT / "third_party" / f"stable-diffusion.cpp-{QI.SD_CPP_COMMIT[:7]}.zip"
    z2 = ROOT / "third_party" / f"sd-ggml-{QI.SD_GGML_COMMIT[:7]}.zip"
    download(QI.SD_CPP_ZIP, z1, "stable-diffusion.cpp source")
    download(QI.SD_GGML_ZIP, z2, "stable-diffusion.cpp's ggml source")
    try:
        src = QI.unpack_source(ROOT / "third_party", z1, z2)
    except zipfile.BadZipFile:
        for z in (z1, z2):
            drop_archive(z)
        raise
    for z in (z1, z2):
        z.unlink(missing_ok=True)
        z.with_name(z.name + ".done").unlink(missing_ok=True)
    return src


def vulkan_sdk() -> Path | None:
    """What a Vulkan build of sd-cli needs (the Vulkan headers and glslc): $VULKAN_SDK, the distribution's packages,
    else (Linux) the LunarG SDK unpacked into third_party (no sudo, ~370 MB).  Returns the SDK folder it put on PATH,
    None when the system's files are used."""
    QI = _qimage()
    have = os.environ.get("VULKAN_SDK")
    if have and Path(have).is_dir():
        return None
    if not WIN and shutil.which("glslc") and Path("/usr/include/vulkan/vulkan.h").exists():
        return None
    if WIN:
        fail("compiling the image generator for this GPU (Vulkan) needs the Vulkan SDK",
             "install it from https://vulkan.lunarg.com/sdk/home#windows, open a new window and run setup again "
             "(or --image-backend cpu)")
    base = ROOT / "third_party" / "vulkan-sdk"
    sdk = base / QI.VULKAN_SDK_VERSION / "x86_64"
    if not (sdk / "bin" / "glslc").exists():
        import tarfile
        tar = ROOT / "third_party" / f"vulkansdk-{QI.VULKAN_SDK_VERSION}.tar.xz"
        download(QI.VULKAN_SDK_URL, tar, f"the Vulkan SDK {QI.VULKAN_SDK_VERSION} (LunarG, for compiling only)")
        verify_sha256(tar, QI.VULKAN_SDK_SIZE, QI.VULKAN_SDK_SHA256)
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True)
        with tarfile.open(tar) as t:
            t.extractall(base, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        tar.unlink(missing_ok=True)
        tar.with_name(tar.name + ".done").unlink(missing_ok=True)
    os.environ["VULKAN_SDK"] = str(sdk)
    os.environ["PATH"] = os.pathsep.join([str(sdk / "bin"), os.environ.get("PATH", "")])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(sdk / "lib" / "VulkanLoader" / "lib"),
                                                     os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    ok(f"Vulkan SDK: {sdk}")
    return sdk


def apply_native_runner(src: Path) -> None:
    """I2: tools/image/sdcpp_native's runner into the stable-diffusion.cpp source (idempotent)."""
    import importlib.util
    QI = _qimage()
    spec = importlib.util.spec_from_file_location("strata_sdcpp_native_apply", QI.NATIVE_DIR / "apply.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply(src)


def build_image_engine(backend: str, gpu: dict) -> tuple[Path, list[str]]:
    """Compile sd-cli for `backend` into engine-image/ (again only when the pins or the backend changed) -> (its path,
    the library folders it runs with)."""
    QI = _qimage()
    IMAGE_ENGINE.mkdir(exist_ok=True)
    exe, stamp = IMAGE_ENGINE / QI.SD_EXE, IMAGE_ENGINE / "BUILD.json"
    arch = gpu.get("arch") if backend in ("cuda", "hip") else None
    want = QI.build_stamp(backend, arch)
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    lib_dirs = [d for d in meta.get("lib_dirs") or [] if Path(d).is_dir()]
    if exe.exists() and {k: meta.get(k) for k in want} == want:
        ok(f"image generator already built ({backend})")
        return exe, lib_dirs
    src = get_sd_cpp()
    if backend in QI.NATIVE_BACKENDS:                 # Strata's DiT runner (inert unless the config asks for it)
        apply_native_runner(src)
    vcvars, lib_dirs = None, []
    if backend == "cuda":
        nvcc, cuda_v = find_nvcc()
        if nvcc is None:
            fail("compiling the image generator for NVIDIA needs the CUDA Toolkit (nvcc)",
                 "install it from https://developer.nvidia.com/cuda-downloads and run setup again "
                 "(or --image-backend vulkan / cpu)")
        root = Path(nvcc).resolve().parent.parent
        defs = QI.cmake_defs("cuda", arch=arch, cuda_root=root) + [f"-DCMAKE_CUDA_COMPILER={nvcc}"] + \
            host_compiler_defs(True)
        vcvars = find_vcvars(cuda_v) if WIN else None
        lib_dirs = [str(root / "bin")] if WIN else [str(d) for d in (root / "lib64", root / "lib") if d.is_dir()]
    elif backend == "hip":
        root, dirs = rocm_root([arch])
        os.environ.update({"HIP_PLATFORM": "amd", "ROCM_PATH": str(root), "HIP_PATH": str(root)})
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
        defs = QI.cmake_defs("hip", arch=arch, rocm=root)
        lib_dirs = list(dirs)
    elif backend == "vulkan":
        sdk = vulkan_sdk()
        defs = QI.cmake_defs("vulkan")
        vcvars = find_vcvars() if WIN else None
        if sdk is not None:                          # the SDK's loader (lib/VulkanLoader): CMake does not look there
            loader = sdk / "lib" / "VulkanLoader" / "lib"
            defs += [f"-DVulkan_LIBRARY={loader / 'libvulkan.so'}", f"-DVulkan_INCLUDE_DIR={sdk / 'include'}",
                     f"-DVulkan_GLSLC_EXECUTABLE={sdk / 'bin' / 'glslc'}"]
            import ctypes.util
            if not ctypes.util.find_library("vulkan"):
                lib_dirs = [str(loader)]             # no system Vulkan loader to run with: the SDK's
    else:
        defs = QI.cmake_defs("cpu")
        vcvars = find_vcvars() if WIN else None
    say(f"  Compiling the image generator (stable-diffusion.cpp {QI.SD_CPP_COMMIT[:7]}, {backend}; 5-20 minutes, "
        "once) ...")
    bdir = ROOT / f"build-image-{backend}"
    cmake_build(src, bdir, "sd-cli", defs, vcvars, f"build-image-{backend}.bat" if WIN else "")
    built = bdir / "bin" / QI.SD_EXE
    if not built.exists():
        fail(f"the image generator's build finished without {built}", f"delete the folder {bdir.name} and run again")
    shutil.copy2(built, exe)
    stamp.write_text(json.dumps({**want, "lib_dirs": lib_dirs}, indent=1))
    ok(f"image generator compiled: {exe}")
    return exe, lib_dirs


def image_backend_choice(a, gpu: dict | None, hip: bool) -> tuple[str, str, float]:
    """--images: (backend, quant, the GPU memory in GB the quant was picked for).  An explicit --image-quant above the
    recommendation is kept, with a warning (recommend, never force)."""
    QI = _qimage()
    vendor = "amd" if hip else ((gpu or {}).get("vendor") or "nvidia")
    backend = a.image_backend or QI.pick_backend({**gpu, "vendor": vendor} if gpu else None, WIN)
    gb = float((gpu or {}).get("vram_gb") or 0.0) if backend != "cpu" else 0.0
    rec = QI.pick_quant(gb) if backend != "cpu" else "Q4_K_M"
    quant = a.image_quant or rec
    if QI.QUANTS.index(quant) > QI.QUANTS.index(rec):
        warn(f"image model {quant}: setup recommends {rec} for {gb:.0f} GB of GPU memory; kept as you chose "
             "(stable-diffusion.cpp then keeps part of it in RAM, which is slower)")
    return backend, quant, gb


def setup_images(a, gpu: dict, hip: bool, models_dir: Path) -> dict | None:
    """--images qwen-image: sd-cli for this GPU and Qwen-Image-2.1-Turbo's three files -> the run config's "images"
    section (serve/images.py); None for --images no, or when not wanted after all."""
    if not a.images or a.images == "no":
        return None
    QI = _qimage()
    backend, quant, _ = image_backend_choice(a, gpu, hip)
    say(f"  Image generation: Qwen-Image-2.1-Turbo {quant} on {backend} (8 steps, CFG 1; POST /v1/images/generations)")
    say("  " + QI.LICENSE_NOTE)
    need = sum(QI.FILES[k][2] for k in (quant, "vae", "llm")) / 1e9
    if ask(f"Download the image model ({need:.1f} GB) and set up image generation?", ["y", "n"], "y", a.yes) != "y":
        return None
    models_dir.mkdir(parents=True, exist_ok=True)
    if free_gb(models_dir) < need + 5:
        warn(f"less than {need + 5:.0f} GB free for the image model: the download may run out of disk space")
    exe, lib_dirs = build_image_engine(backend, gpu or {})
    for name in (quant, "vae", "llm"):
        repo, path, size, sha = QI.FILES[name]
        dst = QI.local_path(models_dir, name)
        download(QI.file_url(name, hf_endpoint()), dst, f"image model: {Path(path).name}")
        verify_sha256(dst, size, sha)
    cfg = QI.image_config(backend, quant, exe, models_dir, uma=QI.is_apu(gpu), lib_dirs=lib_dirs,
                          policy=a.image_gpu_policy or "auto", native=getattr(a, "image_native", None) == "on")
    ok(f"image generation: {quant} on {backend}, GPU policy {cfg['gpu_policy']}")
    return cfg


# ------------------------------------------------------------------------------------------------ the engine
def driver_major(gpu):
    try:
        return int(gpu["driver"].split(".")[0])
    except (ValueError, KeyError):
        return 0


class UnverifiedEngine(Exception):
    """The engine archive could not be verified, with the reason.

    Raised rather than `fail()`-ed on purpose.  The engine-UPDATE paths wrap their download in
    `except Exception` and fall back to the engine already installed, so a refusal has to be an Exception:
    `fail()` ends in `sys.exit(1)`, and SystemExit is a BaseException, so it flies past that guard and
    kills the start instead.  Measured on this change before the fix - a wrong hash, a wrong size and a
    missing digest each escaped get_prebuilt(updating=True) as SystemExit(1).
    """


def release_of(base: str) -> tuple[str, str | None] | None:
    """("github", tag) for a GitHub release URL, or None when the URL is not one.

    `base` is one of the bases from `prebuilt_bases`: a `releases/download/v0.1.40/` URL (which names its
    tag), the `releases/latest/download/` URL, or something else entirely - a local folder, a plain mirror.
    That last case matters: it has no published digest, so treating it as "latest" would check the file
    against a release the user did not ask for, over the network, which is wrong twice over.

    The repository comes out of the URL when it names one, so a fork's own releases are checked against
    the fork rather than against Niko1221/Strata (which never has the fork's tags, so the check would
    always fail).
    """
    m = re.search(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/releases/(?:download/v([^/]+)|latest)/",
                  base)
    if not m:
        return None
    return f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}/releases", m.group(3)


def is_local(base: str) -> bool:
    """True for a path on this machine - `C:/mirror`, `/mnt/mirror`, `//share/engine`, `file://...`.

    A local folder is the user's own file on their own disk, the same trust decision as `--gguf-dir`, and
    there is no published digest for it to be checked against.  It is treated differently from a remote
    mirror on purpose: nothing is between the file and setup, whereas a remote mirror has a network in the
    middle and still no release to check it against.
    """
    b = base.strip()
    if b.lower().startswith("file://"):
        return True
    return not re.match(r"^[a-z][a-z0-9+.\-]*://", b, re.I)


def engine_digest(asset: str, base: str) -> tuple[int, str] | None:
    """(size, "sha256:<hex>") for `asset` in the release `base` points at, or None if GitHub will not say.

    The size and hash come from the releases API - a different origin from the download, which is the
    whole point: a mirrored, substituted or TLS-intercepted download does not come with a matching digest,
    while a compromised release does (see `verify_engine_archive` for what that leaves uncovered).

    GitHub populates `digest` for every asset, including ones uploaded before the field existed
    (measured on v0.1.34 through v0.1.40.1).  None means no answer - no network, a rate limit, or a
    release that does not publish one - and the caller warns and installs anyway (never refuses).
    """
    where = release_of(base)
    if where is None:                  # a local folder or a plain mirror: nothing published to check against
        return None
    releases, tag = where
    url = f"{releases}/tags/v{tag}" if tag else f"{releases}/latest"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "strata-setup",
                                                   "Accept": "application/vnd.github+json"})
        with _urlopen(req, timeout=60) as r:
            rel = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None                     # offline, rate-limited, or the release is not there
    for a in rel.get("assets") or []:
        if a.get("name") == asset and a.get("digest") and str(a.get("digest")).startswith("sha256:"):
            return int(a.get("size") or 0), str(a["digest"]).split(":", 1)[1]
    return None


def drop_download(z: Path) -> None:
    """The archive and its finish mark. A refused engine is not left where a later run would reuse it."""
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)


def verify_engine_archive(z: Path, asset: str, base: str) -> None:
    """The downloaded engine archive against the published size and SHA-256, before it is unpacked.

    Without this, a ready-made engine is installed on nothing more than the byte count matching the
    server's Content-Length - and a substituted file of the same length passes that.  It runs before the
    archive is opened, so a wrong engine never reaches `_unpack`, let alone the engine directory.

    Raises UnverifiedEngine on a refusal; the caller decides whether that stops setup (a first install,
    with nothing to fall back to) or keeps the engine that is already there (an update).

    Where the hash comes from, and where it cannot:

    - a GitHub release URL: the tag is read out of the URL, so the exact release the bytes claim to come
      from is the one checked rather than "latest" (setup tries the checkout's own release first, #214),
      and the repository comes out of the URL too, so a fork's releases are checked against the fork;
    - a local folder (`--prebuilt D:/mirror`): nothing published to check against.  This warns and goes
      ahead - it is the user's own file, the same trust decision as `--gguf-dir`, and there is nothing an
      API could say about it;
    - any other remote mirror: nothing published to check against either, but a network is in the middle,
      so it warns and installs anyway (no digest is never a reason to refuse).

    What a digest from the API does NOT cover, stated plainly: it proves the bytes are the ones GitHub
    published for that asset, so it catches a corrupted transfer, a mirror or proxy that substituted the
    file, and a hostile network.  It does not make a malicious RELEASE safe - if whoever can publish a
    release publishes a hostile engine, the digest matches it.  Only a hash pinned in this file closes
    that, at the cost of a commit per release; a dict of tag -> sha next to `engine_digest` is where one
    would go.

    The size and the hashing below are spelled out rather than delegated to `verify_sha256`, for two
    reasons, both measured here.  That function ends in `fail()`, and a refusal has to be an Exception so
    the update paths can fall back.  And it keeps a `.done` mark so 111 GB of Unsloth shards is hashed
    once; the engine archive is 190 MB, which hashes in 0.79 s at 242 MB/s on this machine, against 3.0 s
    to download the same file.  A mark that saves 0.8 s is not worth a class of hole - anything that
    changes the file after it was verified, at any length - so there is no mark here.
    """
    if os.environ.get("STRATA_SKIP_SHA256") == "1":
        warn(f"STRATA_SKIP_SHA256=1: NOT checking {asset} against its SHA-256. If the file is corrupt or "
             f"tampered with, it will be installed anyway. Unset it to get the check back.")
        return
    want = engine_digest(asset, base)          # one API call: this is a rate-limited API
    if not want:
        # no published digest (offline, rate limit, a mirror or folder): install anyway, never refuse
        warn(f"could not get a SHA-256 for {asset} from GitHub (offline, rate limited, or not a release "
             f"URL), so it was NOT verified. Installing it as it is.")
        return
    size, sha = want
    have = z.stat().st_size if z.exists() else -1
    if have != size:
        drop_download(z)
        raise UnverifiedEngine(f"{z.name} is {have:,} bytes, not the published {size:,}")
    h = hashlib.sha256()
    with open(z, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        got = h.hexdigest()
        drop_download(z)
        raise UnverifiedEngine(f"{z.name} has the wrong SHA-256 ({got}, expected {sha})")
    ok(f"{z.name}: SHA-256 checksum verified against GitHub's ({sha[:12]}...)")


def engine_refused(asset: str, e: Exception, updating: bool) -> None:
    """Report a refusal and either stop setup or keep the engine that is installed.

    A first install has nothing to fall back to, so it stops.  An update does: the engine already in
    place is untouched by a refusal (nothing has been unpacked), it works, and stopping the model from
    starting over a hash is worse than keeping what is there - which is the whole reason that call site
    catches what the download can throw.
    """
    if not updating:
        fail(f"the downloaded Strata engine does not match GitHub's checksum: {e}",
             "the bad download has been deleted (corrupt or tampered), so the next run fetches the "
             "published file again. Set STRATA_SKIP_SHA256=1 only if you insist on installing it anyway")
        return
    warn(f"keeping the engine that is installed: {e}")
    say("       Nothing was replaced. The bad download was deleted; run setup again to fetch it afresh.")


def prebuilt_bases(url_base) -> list[str]:
    """Where to look for the ready-made engine, in order (each ending in a slash).  The default: the release of this
    checkout's version first, then the latest (#214); an explicit --prebuilt / STRATA_PREBUILT_URL: only that."""
    base = url_base if url_base.endswith(("/", "\\")) else url_base + "/"
    if base != PREBUILT_URL:
        return [base]
    version = source_version()
    if version == "0":                                  # no source tree to name a release by (the packaged Terminal): the latest release
        return [base]
    return [PREBUILT_TAG_URL.format(version=version), base]


PREVIOUS_ENGINE = ".previous"


def engine_version_of(folder: Path) -> str:
    try:
        return str(json.loads((folder / "BUILD.json").read_text(encoding="utf-8")).get("version", "?"))
    except (OSError, ValueError):
        return "?"


def install_unpacked(tmp: Path, eng: Path) -> None:
    """#670: the unpacked engine (tmp) replaces the installed one in `eng`, which is kept, one generation, in
    eng/.previous (about 210 MiB): a new engine that turns out bad has a way back (`setup.py --rollback-engine`).  The
    swap is all or nothing: a failure part-way puts the old files back instead of leaving a mix of old and new."""
    prev = eng / PREVIOUS_ENGINE
    new = list(tmp.iterdir())
    old = [eng / p.name for p in new if p.name != PREVIOUS_ENGINE and (eng / p.name).exists()]
    if (eng / KQ_LIB).exists() and all(p.name != KQ_LIB for p in new):   # a zip without strata_kq: an older one of
        old.append(eng / KQ_LIB)                       # another build would only be refused (kept in .previous too)
    shutil.rmtree(prev, ignore_errors=True)
    moved, placed = [], []
    try:
        if old:
            prev.mkdir()
        for dst in old:
            shutil.move(str(dst), str(prev / dst.name))
            moved.append(dst)
        for p in new:
            p.replace(eng / p.name)
            placed.append(eng / p.name)
    except OSError:
        for dst in placed:                             # back to the old engine, whole
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink(missing_ok=True)
        for dst in moved:
            shutil.move(str(prev / dst.name), str(dst))
        shutil.rmtree(prev, ignore_errors=True)
        raise
    # #1403 debt: get_prebuilt moved the old BUILD.json aside (BUILD.json.prev) instead of deleting it; it goes with the
    # engine it describes, so engine_version_of(.previous) says the version instead of "?"
    kept = eng / "BUILD.json.prev"
    if kept.exists():
        try:
            if moved and not (prev / "BUILD.json").exists():
                shutil.move(str(kept), str(prev / "BUILD.json"))
            else:
                kept.unlink()
        except OSError:
            pass
    if moved:
        ok(f"the engine it replaced ({engine_version_of(prev)}) is kept in {prev}; "
           "setup.py --rollback-engine puts it back")


def rollback_engine(toolkit=13) -> int:
    """#670: `setup.py --rollback-engine`: the engine kept in engine/.previous becomes the installed one again, and the
    one it replaces is kept in its place (run it again to go forward)."""
    eng = engine_dir(toolkit)
    prev = eng / PREVIOUS_ENGINE
    if not prev.is_dir() or not any(prev.iterdir()):
        warn(f"no earlier engine is kept in {prev} (one is kept when an update replaces the engine)")
        return 1
    swap = eng / ".swap"
    shutil.rmtree(swap, ignore_errors=True)
    swap.mkdir()
    was = engine_version_of(eng)
    for p in list(prev.iterdir()):                     # what the earlier engine has: the current copy goes aside
        cur = eng / p.name
        if cur.exists():
            shutil.move(str(cur), str(swap / p.name))
        shutil.move(str(p), str(cur))
    for p in list(swap.iterdir()):
        shutil.move(str(p), str(prev / p.name))
    shutil.rmtree(swap, ignore_errors=True)
    ok(f"engine {engine_version_of(eng)} is installed again (it replaced {was}, which is kept in {prev})")
    return 0


def get_prebuilt(url_base, gpu, vision, updating=False, toolkit=13) -> Path | None:
    """The ready-made engine in engine/ (kept between runs), or None when there is none for this PC.
    updating: called to replace an installed engine, which starts instead when this fails (no compile).
    toolkit 12: the experimental CUDA 12 engine (CUDA12_ASSET) in engine-cuda12/."""
    acquisition.require("the ready-made engine")
    eng = engine_dir(toolkit)
    asset = CUDA12_ASSET if int(toolkit) == 12 else PREBUILT_ASSET
    info = eng / "BUILD.json"
    if info.exists() and (eng / EXE).exists() and json.loads(info.read_text(encoding="utf-8")).get("backend") != "hip":
        meta = json.loads(info.read_text(encoding="utf-8"))
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
        if meta.get("source") == "local":              # compiled here: build_engine checks its source and cards
            return None
        have = arch_numbers(meta.get("archs"))
        miss = [int(x) for x in gpu.get("archs", [gpu["arch"]])
                if have and int(x) not in have and not (meta.get("ptx") and int(x) > max(have))]
        if miss:                                       # a card it has no code for (#128): compiled here instead
            warn(f"the installed engine is built for {', '.join(str(a) for a in have)}; your GPU is "
                 f"{', '.join(str(x) for x in miss)}: compiling instead")
            return None
        if ver >= MIN_ENGINE:
            ok("ready-made engine already installed")
            return eng
        say(f"  Updating the ready-made engine ({meta.get('version')} -> {'.'.join(map(str, MIN_ENGINE))} or newer) ...")
        info.replace(eng / "BUILD.json.prev")   # kept for the .previous copy
    if is_arm64() and sys.platform.startswith("linux"):
        # a DGX Spark: the release has no flat engine zip for it (and the CUDA 12 engine is x86). The signed feed has the linux-aarch64
        # package when the release carries one; without it the caller compiles (spark/build_engine.sh does the same by hand)
        got = engine_from_feed(eng, updating) if int(toolkit) != 12 else None
        if got is None:
            warn(f"no ready-made engine for {platform.machine()} in the update feed" + ("" if updating else ": compiling it here, once"))
        return got
    if not url_base:
        return None
    eng.mkdir(exist_ok=True)
    z = eng / asset
    bases = prebuilt_bases(url_base)
    for i, base in enumerate(bases):
        if not base.startswith(("http://", "https://")):
            break
        try:                                           # not published (yet), or no internet: compile instead
            req = urllib.request.Request(base + asset, method="HEAD", headers={"User-Agent": "strata-setup"})
            _urlopen(req, timeout=60).close()
            break
        except OSError as e:
            if i + 1 < len(bases):                     # #214: this checkout's release is not published (yet)
                say(f"  No ready-made engine for v{source_version()} ({e}): the latest release instead")
                continue
            if int(toolkit) != 12:                     # 0.2 releases: the signed feed has it (#3469)
                got = engine_from_feed(eng, updating)
                if got is not None:
                    return got
            warn(f"no ready-made engine at {base} ({e})" + ("" if updating else ": compiling instead"))
            return None
    say("  Downloading the ready-made Strata engine" + (" (CUDA 12, experimental)" if int(toolkit) == 12 else "") + " ...")
    download(base + asset, z, "Strata engine")
    try:
        verify_engine_archive(z, asset, base)
    except UnverifiedEngine as e:
        engine_refused(asset, e, updating)
        return None
    tmp = eng / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
    except zipfile.BadZipFile:                         # not a zip, or a damaged one: a kept .done mark would make
        drop_archive(z)                                # every later run fail on it instead of downloading it again
        raise
    meta = json.loads((tmp / "BUILD.json").read_text(encoding="utf-8"))
    if tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit()) < MIN_ENGINE:
        need = ".".join(map(str, MIN_ENGINE))
        if updating:                                   # these files are newer than the published release (#58)
            warn(f"engine {need} is not published yet (the release may still be uploading): run this again "
                 f"in a few minutes to update it")
        else:
            warn(f"the ready-made engine at {base} is version {meta.get('version')}; this setup needs "
                 f"{need}: compiling instead")
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    archs = arch_numbers(meta.get("archs"))
    miss = [int(x) for x in gpu.get("archs", [gpu["arch"]])
            if int(x) not in archs and not (meta.get("ptx") and int(x) > max(archs))]
    if miss:
        warn(f"the ready-made engine is built for {', '.join(str(a) for a in archs)}; your GPU is "
             f"{', '.join(str(x) for x in miss)}" + ("" if updating else ": compiling instead"))
        shutil.rmtree(tmp, ignore_errors=True)
        drop_archive(z)
        return None
    install_unpacked(tmp, eng)
    shutil.rmtree(tmp, ignore_errors=True)
    drop_archive(z)
    if not (eng / EXE).exists():
        fail("the ready-made engine archive has no " + EXE)
    if not WIN:
        for x in (EXE, VEXE, "strata-dsv41"):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    ok(f"ready-made engine {meta.get('version', '')} for {', '.join('sm_' + str(a) for a in archs)} (CUDA "
       f"{meta.get('cuda', '?')})")
    return eng


def update_engine_gpu(toolkit=13):
    """#1485: the card an engine update is for.  Not simply the one with the most VRAM: a Tesla V100 (32 GB) beside two
    RTX 4090s was taken, and CUDA 13 cannot compile for it.  The cards the installed models are set up for come
    first (their "gpu"); for the CUDA 13 engine a card it cannot compile for is left out when another can.  An
    explicit --gpu (GPU_PICK) still wins.  None when there is no NVIDIA GPU."""
    if GPU_PICK is not None:
        return gpu_info()
    found = gpus()
    if not found:
        return None
    named = set()
    for c in installed_configs():
        try:
            g = json.loads(c.read_text(encoding="utf-8-sig")).get("gpu")
        except (OSError, ValueError):
            continue
        named.update(i for i in (g if isinstance(g, list) else [g]) if isinstance(i, int) and not isinstance(i, bool))
    pool = [g for g in found if g["index"] in named] or found
    if int(toolkit) != 12:
        pool = [g for g in pool if int(g["arch"]) >= CUDA13_MIN_ARCH] or pool
    g = max(pool, key=lambda x: (round(x["vram_gb"]), -x["index"]))
    return {**g, "count": len(found)}


def engine_update_note() -> None:
    """A start looks at the installed engine(s) and says when UPDATE would replace one; it changes nothing and reaches
    no network (the engine that is installed starts as it is).  update_installed_engine, below, is what UPDATE.bat /
    update.sh and a setup run do."""
    for toolkit in (13, 12):
        eng = engine_dir(toolkit)
        info = eng / "BUILD.json"
        if not info.exists() or not (eng / EXE).exists():
            continue
        try:
            meta = json.loads(info.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
        need = WIN_HIP_MIN_ENGINE if (meta.get("backend") == "hip" and WIN) else MIN_ENGINE
        if meta.get("source") == "local":
            old = meta.get("src") != source_hash(ENGINE_SOURCES) or stale_binary(eng / EXE)
        elif meta.get("backend") == "hip" and not WIN:
            old = meta.get("src") != source_hash(ENGINE_SOURCES) or stale_binary(eng / EXE)
        else:
            old = ver < need
        if old:
            warn(f"the installed engine ({meta.get('version') or 'unknown version'}) is older than this Strata expects: "
                 "starting it as it is. Starting never updates anything; run "
                 + ("UPDATE.bat" if WIN else "./update.sh") + " (or `strata install --update`) to update it")


def update_installed_engine(url_base, toolkit=None) -> None:
    """An installed ready-made engine older than MIN_ENGINE is replaced before the model starts, so a plain
    START-HERE.bat on an existing install picks up a new release.  If that cannot happen (no internet, the model
    still running, no ready-made engine for this GPU) the installed engine is kept and starts as before.
    toolkit None: engine/, then the experimental CUDA 12 engine in engine-cuda12/ when one is installed."""
    if toolkit is None:
        update_installed_engine(url_base, 13)
        if (engine_dir(12) / "BUILD.json").exists():
            update_installed_engine(url_base, 12)
        return
    eng = engine_dir(toolkit)
    info = eng / "BUILD.json"
    if not info.exists() or not (eng / EXE).exists():
        return
    meta_text = info.read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    if meta.get("backend") == "hip" and WIN:           # AMD on Windows: the ready-made HIP engine, when older
        ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
        if meta.get("source") == "prebuilt" and ver < WIN_HIP_MIN_ENGINE:
            try:
                g = next((x for x in amd_gpus() if amd_problem(x) is None), None)
                if g is None:
                    raise RuntimeError("no supported AMD GPU found")
                say(f"  Updating the ready-made AMD engine ({meta.get('version')} -> "
                    f"{'.'.join(map(str, WIN_HIP_MIN_ENGINE))} or newer) ...")
                if get_prebuilt_hip(url_base, g, updating=True) is None:
                    raise RuntimeError("not published yet")
            except (Exception, SystemExit) as e:
                warn(f"could not update the AMD engine{'' if isinstance(e, SystemExit) else f' ({e})'}: "
                     "starting the installed one")
        return
    if meta.get("backend") == "hip":                   # AMD: compiled here, again when its source changed
        if meta.get("src") != source_hash(ENGINE_SOURCES) or stale_binary(eng / EXE):
            try:
                usable = [x for x in amd_gpus() if amd_problem(x) is None]
                g = next((x for x in usable if x["arch"] in arch_names(meta.get("archs"))), usable[0] if usable else None)
                if g is None:
                    raise RuntimeError("no supported AMD GPU found")
                # every architecture it was built for (a layer split across two families keeps both)
                build_engine_hip({**g, "archs": [x for x in arch_names(meta.get("archs")) if x in AMD_ARCHS] or [g["arch"]]},
                                 get_llama_cpp(), meta.get("vision") or "none")
            except (Exception, SystemExit) as e:
                warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: "
                     "starting the installed one")
        return
    ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
    local = meta.get("source") == "local"
    vision = meta.get("vision") or "none"
    if local:                                          # compiled here: is it older than the source (a git pull)?
        if meta.get("src") == source_hash(ENGINE_SOURCES) and not stale_binary(eng / EXE) and \
                (vision == "none" or meta.get("vision_src") == source_hash(VISION_SOURCES)):
            return
    elif ver >= MIN_ENGINE:
        return
    try:                                               # a running engine cannot be replaced (Windows keeps it locked)
        for x in (EXE, VEXE):
            if (eng / x).exists():
                with open(eng / x, "r+b"):
                    pass
    except OSError:
        warn(f"engine {meta.get('version') or ''} is in use: close the model window and run this again to update it")
        return
    gpu = update_engine_gpu(toolkit)
    if local:
        try:                                           # a failed compile must not stop the model from starting
            if gpu is None:
                raise RuntimeError("no NVIDIA GPU found")
            gpu = {**gpu, "archs": sorted({int(gpu["arch"]), *arch_numbers(meta.get("archs"))})}
            if int(toolkit) == 12:                     # the cards it was compiled for (the main GPU may be newer)
                gpu["archs"] = sorted(arch_numbers(meta.get("archs"))) or gpu["archs"]
            build_engine(gpu, vision, False, get_llama_cpp(), toolkit=toolkit)
        except (Exception, SystemExit) as e:
            warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: starting the installed one")
        return
    new = None
    if gpu is not None:
        try:
            if int(toolkit) == 12:                     # the cards the CUDA 12 engine serves, not the newest one
                gpu = {**gpu, "archs": [x for x in arch_numbers(meta.get("archs")) if x < CUDA13_MIN_ARCH]
                       or [int(gpu["arch"])]}
            new = get_prebuilt(url_base, gpu, "gpu", updating=True, toolkit=toolkit)
        except Exception as e:                         # a failed download must not stop the model from starting
            warn(f"updating the engine failed ({e})")
    if new is None:
        if not info.exists():
            info.write_text(meta_text)                 # get_prebuilt drops it before downloading: put it back
        warn(f"could not update the engine: starting the installed {meta.get('version')}")
        return
    pip_cuda_libs(toolkit)


def pip_cuda_libs(toolkit=13) -> None:
    """NVIDIA's cuBLAS and CUDA runtime for a ready-made engine, from pip: CUDA 13's, or the CUDA 12 engine's."""
    acquisition.require("NVIDIA's CUDA libraries (pip)")
    if int(toolkit) == 12:
        pip_install(CUDA12_WHEELS, "NVIDIA CUDA 12 libraries for the experimental engine (cuBLAS, CUDA runtime; ~0.7 GB)")
    else:
        pip_install(CUDA_WHEELS, "NVIDIA CUDA libraries (cuBLAS, CUDA runtime; ~0.4 GB)")


CUDA_SM120_SUSPECT = (13, 2)   # #892 #968: nvcc 13.2.0 / 13.2.1 (build 13.2.51) for sm_120 made garbage answers (IQ1_S / IQ2_S / IQ3_S) and prompts
                               # (K-quant MMQ) that the same source compiled with 13.0.88 answers correctly (llama.cpp hit it too)
CUDA_SM120_FIXED_BUILD = 86    # CUDA 13.2.2 (nvcc build 13.2.86) fixes it (ggml-org/llama.cpp#28581, confirmed on a 5090 in #968)


def nvcc_build(nvcc):
    """The build number of an nvcc ("Build cuda_13.2.r13.2/compiler...", "V13.2.86" -> 86); None when it cannot be read."""
    m = re.search(r"\bV\d+\.\d+\.(\d+)", out([nvcc, "--version"]) if nvcc else "")
    return int(m.group(1)) if m else None


def sm120_nvcc(nvcc, cuda_v, archs):
    """#892 / #968: CUDA 13.2.0 / 13.2.1's nvcc for an RTX 50 card (sm_120).  Returns (nvcc, cuda_v): an older 13.x toolkit
    when one is installed and the user did not name a compiler (STRATA_NVCC), else the one found, with a warning that says
    what to do.  CUDA 13.2.2 (build 86 or newer) is fine.  Recommends, never forces: a user who picked 13.2 keeps it."""
    if not nvcc or cuda_v != CUDA_SM120_SUSPECT or max(archs) < 120:
        return nvcc, cuda_v
    build = nvcc_build(nvcc)
    if build is not None and build >= CUDA_SM120_FIXED_BUILD:
        return nvcc, cuda_v
    if not os.environ.get("STRATA_NVCC"):
        alt, alt_v = find_nvcc(below=CUDA_SM120_SUSPECT)
        if alt and alt_v and alt_v >= (13, 0):
            ok(f"CUDA {alt_v[0]}.{alt_v[1]} is used for the RTX 50 card (sm_120): CUDA 13.2.0 / 13.2.1's compiler made wrong "
               "answers there (#892, #968); STRATA_NVCC=<nvcc> picks another")
            return alt, alt_v
    warn("CUDA 13.2.0 and 13.2.1's compiler (nvcc) made garbage prompts and answers for RTX 50 cards (sm_120) in two "
         "reports (#892, #968): the same source compiled with CUDA 13.0, 13.1 or 13.2.2 is right. If the engine answers "
         "with nonsense, update to CUDA 13.2.2 or install CUDA 13.0 next to it (https://developer.nvidia.com/cuda-toolkit-archive) "
         "and set STRATA_NVCC to its nvcc; the ready-made engine is built with 13.0")
    return nvcc, cuda_v


def cuda_apt_repo(osr):
    """The NVIDIA apt repo for this distribution ("ubuntu2404", "debian13") from /etc/os-release, or None when it is
    not one setup adds the CUDA keyring for.  Both families publish cuda-keyring and CUDA 13.x."""
    ver = osr.get("VERSION_ID", "").strip('"').replace(".", "")
    repo = {"ubuntu": "ubuntu", "debian": "debian"}.get(osr.get("ID", ""), "") + ver
    return repo if repo in ("ubuntu2204", "ubuntu2404", "debian12", "debian13") else None


def install_build_tools(gpu, yes):
    """The compiler and the CUDA toolkit, installed for the user (asks once).  Returns (nvcc, vcvars)."""
    acquisition.require("the compiler and CUDA toolkit")
    archs = [int(x) for x in gpu.get("archs", [gpu["arch"]])]
    # #295: Pascal / Volta need a CUDA 12.x toolkit - CUDA 13 cannot build sm_60/sm_70; gpu["toolkit"] = 12: the
    # experimental CUDA 12 engine for any cards (--cuda 12, docs/OLDER_GPUS.md)
    old = int(gpu.get("toolkit") or (12 if min(archs) < CUDA13_MIN_ARCH else 13)) == 12
    need12 = (12, 8) if max(archs) >= 120 else (12, 0)     # sm_120 needs CUDA 12.8 or newer
    if old and max(archs) >= 120:
        warn("an RTX 50 card (sm_120) in a CUDA 12 engine: engines built with CUDA 12.8 crashed on long prompts there "
             "(#220, #224); the RTX 50 card alone (--gpu N) runs the ready-made CUDA 13 engine")
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
    if old and (nvcc is None or cuda_v < need12):
        fail("the experimental CUDA 12 engine (Pascal / Volta, or --cuda 12) is compiled here with the NVIDIA CUDA "
             f"Toolkit {need12[0]}.{need12[1]} or a newer 12.x (CUDA 13 cannot compile for these cards)" +
             (f"; found CUDA {cuda_v[0]}.{cuda_v[1]}" if nvcc else ""),
             "install CUDA 12.9 (it can sit next to a newer one) from https://developer.nvidia.com/cuda-toolkit-archive "
             "and run it again (STRATA_NVCC=<its nvcc> picks one toolkit)")
    # RTX 50 (sm_120): CUDA 13.0 - an engine built with 12.8 crashed in the prompt path on Linux (#220)
    need_cuda = need12 if old else (13, 0) if max(archs) >= 120 else (12, 0)
    vcvars = find_vcvars(cuda_v) if WIN else None
    have_cc = vcvars is not None if WIN else shutil.which("g++") is not None
    missing = []
    if not have_cc:
        missing.append("Visual Studio 2022 Build Tools (C++)" if WIN else "the C++ compiler (build-essential)")
    if nvcc is None or cuda_v < need_cuda:
        missing.append("the NVIDIA CUDA Toolkit 13.0")
    if not missing:
        nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
        ok(f"build tools present (CUDA {cuda_v[0]}.{cuda_v[1]})")
        return nvcc, vcvars
    say("  The engine has to be compiled for your PC, which needs: " + " and ".join(missing) + ".")
    say("  They can be installed now (about 8-10 GB, 15-40 minutes" + (", Windows will ask for permission" if WIN else
                                                                        ", sudo will ask for your password") + ").")
    if ask("  Install them now?", ["y", "n"], "y", yes) != "y":
        fail("the build tools are needed", "install them yourself (see README.md) and run it again")
    if WIN:
        if shutil.which("winget") is None:
            fail("winget (Windows package manager) is not available",
                 "install 'App Installer' from the Microsoft Store, or install the tools by hand (README.md)")
        wg = ["winget", "install", "-e", "--source", "winget", "--accept-package-agreements",
              "--accept-source-agreements", "--disable-interactivity"]
        if not have_cc:
            run([*wg, "--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
                 "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
                check=False)
        if nvcc is None or cuda_v < need_cuda:
            run([*wg, "--id", "Nvidia.CUDA", "--version", "13.0"], check=False)
        vcvars = find_vcvars(cuda_v)
    else:
        apt = shutil.which("apt-get")
        if apt is None:
            fail("missing: " + " and ".join(missing) + " (the automatic install is only done on Ubuntu/Debian)",
                 "install them with your distribution's packages (Arch: pacman -S base-devel cuda; nvcc is found on "
                 "PATH, in /usr/local/cuda* and in /opt/cuda*), then run it again")
        if not have_cc:
            run(["sudo", "apt-get", "install", "-y", "build-essential"])
        if nvcc is None or cuda_v < need_cuda:
            osr = dict(line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line)
            repo = cuda_apt_repo(osr)
            if repo is None:
                fail("the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 and Debian 12 / 13 only",
                     "install it from https://developer.nvidia.com/cuda-downloads and run it again")
            deb = Path("/tmp/cuda-keyring.deb")
            # NVIDIA's arm64 server repository is "sbsa" (a DGX Spark; DGX OS normally has the toolkit already)
            download(f"https://developer.download.nvidia.com/compute/cuda/repos/{repo}/"
                     f"{'sbsa' if is_arm64() else 'x86_64'}/cuda-keyring_1.1-1_all.deb", deb, "CUDA repository key")
            run(["sudo", "dpkg", "-i", str(deb)])
            run(["sudo", "apt-get", "update"])
            if repo.startswith("ubuntu"):                  # as before: 13.0, and a failed install stops setup here
                run(["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"])
            else:
                # Debian 13's CUDA repo carries no 13.0 package: take the oldest 13.x minor it has
                for minor in range(10):
                    if run(["sudo", "apt-get", "install", "-y", f"cuda-toolkit-13-{minor}"], check=False).returncode == 0:
                        break
                else:
                    fail(f"no cuda-toolkit-13-x package could be installed from NVIDIA's {repo} repository",
                         "install the CUDA Toolkit 13 from https://developer.nvidia.com/cuda-downloads and run it again")
    nvcc, cuda_v = find_nvcc(below=(13, 0)) if old else find_nvcc()
    if (WIN and find_vcvars(cuda_v) is None) or (not WIN and shutil.which("g++") is None):
        fail("the C++ build tools did not install", "install them by hand (README.md) and run it again")
    if nvcc is None or cuda_v < need_cuda:
        fail("the CUDA Toolkit did not install", "install it from https://developer.nvidia.com/cuda-downloads, then run it again")
    nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
    ok(f"build tools installed (CUDA {cuda_v[0]}.{cuda_v[1]})")
    return nvcc, find_vcvars(cuda_v) if WIN else None


def tool_usable(name):
    """find_tool found one that runs and (cmake) is new enough: an older system cmake counts as missing, so pip puts the right one in the venv."""
    found = find_tool(name)
    if found is None:
        return False
    if name == "cmake":
        runs, version = _tool_version(found, name)
        return runs and (not version or version[:2] >= CMAKE_MIN)
    return True


def ensure_build_tools():
    """cmake and ninja for a local compile, installed from requirements-build.txt when missing (#3403: the Mac setup
    compiles without cmake_build(), so it calls this too).  Returns (cmake, ninja)."""
    # The ready-made engine path never calls this function. Keep build helpers out of
    # the normal serving environment, and use already installed tools offline.
    build_packages = requirement_lines(BUILD_REQUIREMENTS) if BUILD_REQUIREMENTS.exists() else BUILD_PACKAGES
    missing = [package for package in build_packages if not tool_usable(req_name(package))]
    if missing:
        pip_install(missing, "CMake and Ninja build tools")
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv python -m pip install -r requirements-build.txt")
    return cmake, ninja


def configured_generator(bdir: Path) -> str | None:
    """The generator a build folder was configured with (CMakeCache.txt), None when it was never configured."""
    try:
        for line in (Path(bdir) / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_GENERATOR:"):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def fresh_if_other_generator(bdir: Path, generator: str = "Ninja") -> bool:
    """CMake refuses to configure a folder with another generator than the one it was made with ("Does not match the generator used
    previously"). The manual build in docs/INSTALL.md uses CMake's default, Unix Makefiles on Linux, in the same build/ folder (#4038).
    Such a folder is configured afresh: its cache is removed (the compiled objects of the other generator are not reused anyway), with one
    plain line. True when it did."""
    had = configured_generator(bdir)
    if had is None or had == generator:
        return False
    say(f"  {Path(bdir).name}/ was configured with {had}; setup builds with {generator}, so it is configured again (the other build is not used)")
    (Path(bdir) / "CMakeCache.txt").unlink(missing_ok=True)
    shutil.rmtree(Path(bdir) / "CMakeFiles", ignore_errors=True)
    return True


def cmake_build(src, bdir, target, defs, vcvars, bat_name):
    acquisition.require("a compile (cmake)")
    cmake, ninja = ensure_build_tools()
    fresh_if_other_generator(bdir)
    conf = [cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(bdir),
            "-DCMAKE_BUILD_TYPE=Release", *defs]
    build = [cmake, "--build", str(bdir), "--target", target, "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    # A failed build is tried once more: CUDA 13.0's ptxas now and then fails to parse a PTX file it just wrote, and
    # the same command then gets past it (issue #45); a second attempt only compiles what is still missing.
    if WIN:
        if vcvars is None:                             # #881: the build needs MSVC's environment
            fail("the Visual Studio 2022 C++ build tools were not found (Visual Studio 2019 is too old: the sources are C++20)",
                 "install Visual Studio 2022 Build Tools (workload 'Desktop development with C++'), then run setup again; a ready-made engine "
                 "needs no compiler: `strata update` fetches it")
        bat = ROOT / bat_name
        q = lambda c: " ".join(f'"{x}"' if " " in str(x) else str(x) for x in c)  # noqa: E731
        bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{q(conf)} || exit /b 1\r\n{q(build)} && exit /b 0\r\n'
                       f'echo   (the build stopped - trying it once more)\r\n{q(build)} || exit /b 1\r\n',
                       encoding="utf-8")
        run(["cmd", "/c", str(bat)])
    else:
        run(conf)
        if run(build, check=False).returncode != 0:
            say("  (the build stopped - trying it once more)")
            run(build)


ENGINE_SOURCES = ("CMakeLists.txt", "cmake", "src", "include", "third_party/ggml")
VISION_SOURCES = ("tools/vision",)


def source_hash(parts) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in engine/BUILD.json: when a `git pull`
    changes them, the engine is compiled again (issue #31)."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        for f in [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file()):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def isa_floor_defs(floor: str, bdir: Path, meta: dict) -> list:
    """The experimental older-CPU build's CMake definition (STRATA_ISA_FLOOR, CMakeLists.txt), none for the normal
    build.  A build folder configured for another floor is configured afresh: ggml's CPU options are cached there."""
    if (meta.get("isa_floor") or "") != floor and (bdir / "CMakeCache.txt").exists():
        (bdir / "CMakeCache.txt").unlink()
    return [f"-DSTRATA_ISA_FLOOR={floor}"] if floor else []


def host_compiler_defs(cuda: bool) -> list:
    """CC / CXX / CUDAHOSTCXX pick the host compilers of a build - as -D definitions, so an existing build folder
    takes them too (the environment reaches CMake only on a folder's first configure, which is why CXX=g++-14 alone
    could die at the last step, #1645).  CUDAHOSTCXX is nvcc's host compiler (-ccbin); it defaults to CXX, as
    CMake's does.  `cuda`: the project compiles CUDA (the host-compiler definition means nothing to a C++-only
    one)."""
    cc = os.environ.get("CC", "").strip()
    cxx = os.environ.get("CXX", "").strip()
    hostcxx = os.environ.get("CUDAHOSTCXX", "").strip() or cxx
    return ([f"-DCMAKE_C_COMPILER={cc}"] if cc else []) + \
           ([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []) + \
           ([f"-DCMAKE_CUDA_HOST_COMPILER={hostcxx}"] if cuda and hostcxx else [])


def toolkit_root_defs(nvcc) -> list:
    """CUDAToolkit_ROOT for the toolkit whose nvcc builds the engine.  Without it CMake can take cudart and cuBLAS from
    another toolkit: with STRATA_NVCC=/opt/cuda-13.0/bin/nvcc on Ubuntu 24.04 that also has the distribution's CUDA
    12.0 (nvidia-cuda-toolkit), the engine was compiled with the 13.0 headers but linked libcudart.so.12 from
    /usr/lib/x86_64-linux-gnu.  The distribution's own nvcc (/usr/bin) keeps CMake's search as before."""
    root = Path(nvcc).resolve().parent.parent
    return [] if root == Path("/usr") else [f"-DCUDAToolkit_ROOT={root}"]


def engine_defs(archs, toolkit=13) -> list:
    """Extra CMake definitions for the engine: the experimental Pascal/Volta build (#295) for cards below sm_75, and
    for every CUDA 12 engine (the same build as the ready-made CUDA 12 one: it admits the older cards)."""
    return ["-DSTRATA_EXPERIMENTAL_SM60=ON"] if min(int(x) for x in archs) < 75 or int(toolkit) == 12 else []


def prebuilt_vision(meta: dict, gpu: dict, vision: str) -> str:
    """The image encoder to use with a ready-made engine (`meta`: its BUILD.json).  The encoder can cover fewer cards
    than the engine (0.1.30/0.1.31: no RTX 20 code, #331): such a card gets the CPU encoder - the same program - instead
    of compiling one, which fails on most Windows PCs (no Visual Studio / CUDA toolkit); --build compiles it."""
    if vision != "gpu":
        return vision
    va = arch_numbers(meta.get("vision_archs", meta.get("archs")))
    if va and int(gpu["arch"]) not in va and not (meta.get("ptx") and int(gpu["arch"]) > max(va)):
        warn(f"the ready-made image encoder has no code for your GPU (sm_{gpu['arch']}): it runs on the CPU instead "
             "(images take longer; setup --build compiles one for your GPU)")
        return "cpu"
    return vision


def local_build_archs(archs, found) -> list:
    """#4662: the architectures a compiled engine gets code for: the cards this install uses AND every other card Strata can run on in the PC.
    A PC with a TITAN RTX (sm_75) and an RTX 3080 (sm_86) was updated with archs [75] (the pair was not picked as a split by setup), and the
    layer-split run on both cards then launched kernels the sm_75 build holds only a trap for: "unspecified launch failure" on every long prompt.
    One more architecture is a few more minutes of compile, once; a card Strata cannot use (older than sm_75) is left out."""
    try:
        present = {int(g["arch"]) for g in found or [] if gpu_problem(g) is None}
    except (KeyError, TypeError, ValueError):
        present = set()
    return sorted({int(x) for x in archs} | present)


def build_engine(gpu, vision, yes, llama, toolkit=None) -> Path:
    """Compile the engine (and, for images, the encoder) for this GPU; the results go to engine/.  A compiled
    engine whose source files changed since (a `git pull`) is compiled again: only the changed files, a few minutes.
    toolkit 12 (default: 12 for a card older than CUDA 13 supports): the experimental CUDA 12 engine, in
    engine-cuda12/ with its own build folders."""
    acquisition.require("an engine build")
    if toolkit is None:
        toolkit = 12 if min(int(x) for x in gpu.get("archs", [gpu["arch"]])) < CUDA13_MIN_ARCH else 13
    t12 = int(toolkit) == 12
    eng = engine_dir(toolkit)
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD.json"
    meta = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    want_vision = vision != "none"
    local = meta.get("source") == "local"
    src, vsrc = source_hash(ENGINE_SOURCES), source_hash(VISION_SOURCES)
    archs = sorted({int(x) for x in gpu.get("archs", [gpu["arch"]])})    # every card the model runs on
    archs = local_build_archs(archs, gpus())                             # #4662: and every other card in the PC
    built = set(arch_numbers(meta.get("archs")))
    # a card the engine has no code for (a GPU added with --gpus, #128) needs a compile even when the source is the
    # same; the compile keeps the generations it was built for
    new_arch = local and not set(archs) <= built
    floor = cpu_floor(cpu_info()[1])                     # "" on an AVX2 CPU: the normal engine
    engine_ok = local and (eng / EXE).exists() and meta.get("src") == src and not new_arch and \
        (meta.get("isa_floor") or "") == floor and not stale_binary(eng / EXE)
    vision_ok = not want_vision or ((eng / VEXE).exists() and (not local or meta.get("vision_src") == vsrc))
    if engine_ok and vision_ok:
        ok("engine already built for this PC")
        return eng
    if local:
        archs = sorted(built | set(archs))
    nvcc, vcvars = install_build_tools({**gpu, "archs": archs, "toolkit": toolkit}, yes)
    cuda_archs = ";".join(str(x) for x in archs)
    bdir, vdir = (ROOT / "build-cuda12", ROOT / "build-vision-cuda12") if t12 else (ROOT / "build", ROOT / "build-vision")
    if not engine_ok:
        say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) + " (a card it had no code for; "
            "10-20 minutes, once) ..." if new_arch else
            "  The engine's source changed: compiling it again (only what changed, a few minutes) ..."
            if local and (eng / EXE).exists() else "  Compiling the Strata engine for your GPU (10-20 minutes, once) ...")
        cmake_build(ROOT, bdir, "strata",
                    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON",
                     f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                     f"-DCMAKE_CUDA_COMPILER={nvcc}", *toolkit_root_defs(nvcc), f"-DSTRATA_GGML_DIR={llama}",
                     *host_compiler_defs(True), *engine_defs(archs, toolkit),
                     *isa_floor_defs(floor, bdir, meta)],
                    vcvars, "build-strata-cuda12.bat" if t12 else "build-strata.bat")
        if stale_binary(bdir / EXE):                      # #2054: the stamp below is written only for a binary of this source
            fail(f"the engine build finished without a new engine ({EXE} is version {binary_engine_version(bdir / EXE)}, "
                 f"this source is {source_version()}): the installed engine and its stamp are left as they were",
                 "the build folder " + str(bdir) + " may be configured for something else: delete it and run this again")
        shutil.copy2(bdir / EXE, eng / EXE)
        if (bdir / KQ_LIB).exists():                      # built with it (Phase 1): it goes next to the engine
            shutil.copy2(bdir / KQ_LIB, eng / KQ_LIB)
        elif (eng / KQ_LIB).exists():                     # an older one of another build would only be refused
            (eng / KQ_LIB).unlink()
    if not vision_ok:
        say("  Compiling the image encoder" + (" with CUDA (10-20 minutes, once) ..." if vision == "gpu" else " ..."))
        defs = [f"-DLLAMA_DIR={llama}", f"-DSTRATA_VISION_CUDA={'ON' if vision == 'gpu' else 'OFF'}",
                "-DSTRATA_PORTABLE=OFF", *host_compiler_defs(vision == "gpu")]   # native, like the engine
        if vision == "gpu":
            defs += [f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
                     *toolkit_root_defs(nvcc)]
        cmake_build(ROOT / "tools" / "vision", vdir, "strata-vision", defs, vcvars,
                    "build-vision-cuda12.bat" if t12 else "build-vision.bat")
        shutil.copy2(vdir / "bin" / VEXE, eng / VEXE)
    bindir = Path(nvcc).parent                            # the toolkit's own libraries (bin, bin/x64, lib64)
    dirs = [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()]
    stamp.write_text(json.dumps({"source": "local", "version": source_version(), "archs": archs,
                                 "vision": vision, **({"toolkit": 12} if t12 else {}),
                                 "cuda_dirs": dirs, "src": src, "vision_src": vsrc if want_vision else None,
                                 **({"isa_floor": floor} if floor else {})}, indent=1))
    ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the data folder
# The model files - the GGUFs, the prepared packs and the MTP layer, 70-120 GB - live in a data folder NEXT TO the
# Strata folder (`Strata-data`), not inside it: updating Strata by unzipping a new copy used to give a new, empty
# folder and a full download again.  Where it is, and which Strata folders this user ran, is kept in a small
# per-user file, so every Strata folder on the PC finds the same files.
DATA_ITEMS = ("models", "packs", "mtp")


LOW_RAM_HEADROOM_GB = 10   # RAM beside the experts: the OS, the engine's other buffers, the server
RESIDENT_ENGINE = (0, 1, 30)   # the first engine with --resident-experts (the low-RAM mode's resident variant)
RESIDENT_SPLIT_ENGINE = (0, 1, 40)   # #642: the first engine that runs it on a layer split


def resident_split() -> bool:
    """#642: the engine setup requires runs the low-RAM mode's resident variant on a layer split - each card's cache
    holds the most-used experts of its own layers, and the RAM copy leaves out what any card holds.  Before it, a
    resident config on several GPUs read those experts through the OS file cache (#364 #384)."""
    return MIN_ENGINE >= RESIDENT_SPLIT_ENGINE


def low_ram_needed(model, ram) -> bool:
    """The model's experts do not fit this PC's RAM with room left for the rest: they are then mapped from the pack's
    experts.bin instead of copied into RAM (the low-RAM mode)."""
    return ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB


#: The draft layer (MTP) takes about 1 GB of graphics memory: its 786 MiB of files plus the draft head's ~218 MiB
#: (phaseA-tests/notes-021-settings.md).  It comes out of the expert cache, so every fit check counts it while MTP is on.
MTP_VRAM_GB = 1.0


def low_ram_gpu_gb(model, vram_gb, ctx=32768, kv="int8", mtp=True) -> float:
    """About how many GB of the model's experts the GPU's cache holds: its VRAM minus ~5 GB for the dense weights,
    buffers and a 32K context's KV cache, minus the KV cache of a longer context (in VRAM in the low-RAM mode: its RAM
    has no room for KV streaming), minus ~1 GB for the draft layer (MTP) when it is on."""
    kv_tok = 13 * KV_CELL_BYTES.get(kv, 1056)           # bytes per context token: 12 QSA layers + the draft layer
    longer = max(0, ctx - 32768) * kv_tok / 1e9
    return max(0.0, min(MODELS[model]["arena_gb"], vram_gb - 5 - (MTP_VRAM_GB if mtp else 0.0) - longer))


def low_ram_gpu_share(model, vram_gb, ctx=32768, kv="int8", mtp=True) -> float:
    """About how much of the model's experts the GPU holds."""
    return low_ram_gpu_gb(model, vram_gb, ctx, kv, mtp) / MODELS[model]["arena_gb"]


def low_ram_resident(model, ram, vram_gb, ctx=32768, kv="int8", mtp=True) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM with the usual room beside them, so they are
    copied into RAM once (the resident variant, `--resident-experts`) instead of being read through the OS file cache
    (plain `--mmap-experts`, which a PC this short of RAM keeps re-reading from the SSD)."""
    rest = MODELS[model]["arena_gb"] - low_ram_gpu_gb(model, vram_gb, ctx, kv, mtp)
    return ram >= rest + LOW_RAM_HEADROOM_GB


def low_ram_fits(model, ram, vram_gb, mtp=True) -> bool:
    """In the low-RAM mode: the experts the GPU does not hold fit the RAM left beside the rest (as file cache)."""
    arena = MODELS[model]["arena_gb"]
    return ram - 6 + max(0.0, vram_gb - 5 - (MTP_VRAM_GB if mtp else 0.0)) >= arena


def low_ram_wanted(model, ram, choice="auto") -> bool:
    """Does setup put this model in the low-RAM mode on this PC: its experts do not fit the RAM with the usual room
    beside them (`low_ram_needed`), or the user asked for it.  Unsloth's 4-bit file is not the low-RAM mode at all:
    it always reads part of its experts from the files, through its RAM budget.  Step 5 applies exactly this answer,
    and the launcher's preset diff asks it too - so a preset that says "auto" is compared with what auto decides,
    not with the word."""
    if MODELS[model].get("budget"):
        return False
    return choice in ("on", "resident", "mmap") or (choice == "auto" and low_ram_needed(model, ram))


def kv_streaming_ram_gb(ctx, kv) -> float:
    """The RAM a streamed KV cache takes: ~13.7 KB per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with
    K8V4, 7.5 KB with 4-bit - 12 QSA layers + the draft layer."""
    return ctx * (13 * KV_CELL_BYTES.get(kv, 1056)) / 1e9


def kv_streaming_wanted(model, ctx, kv, ram, choice="auto") -> bool:
    """Will setup stream this model's KV cache on this PC: from 64K up, when the RAM holds the cache beside the
    model's experts (+1 GB); `--kv-streaming on|off` overrides the RAM test (the owner's rule), and WSL never
    streams.  Step 7 writes `--kv-resident` on exactly this answer, and the launcher's preset diff
    asks it too: a config's `--kv-resident` and a preset's "auto" are the same setting when this says yes."""
    if is_wsl() or choice == "off":
        return False
    return ctx >= 65536 and (ram >= MODELS[model]["ram_gb"] + kv_streaming_ram_gb(ctx, kv) + 1 or choice == "on")


def low_ram_one_gpu_why(model, ram, choice, sel=None) -> list[str]:
    """#250: why the low-RAM mode recommends one GPU, with the RAM math that turned it on; #364 #384: and how to use
    all of them (sel: the cards, for the --gpus example)."""
    arena, need = MODELS[model]["arena_gb"], MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB
    if choice == "auto":
        why = [f"Why: {model}'s experts are {arena:.0f} GB and must fit in RAM with ~{LOW_RAM_HEADROOM_GB} GB beside "
               f"them for the OS and the rest: {arena:.0f} + {LOW_RAM_HEADROOM_GB} = {need:.0f} GB, and this PC has "
               f"{ram:.0f} GB.",
               "So setup uses the low-RAM mode: the experts come from the model's file (copied into RAM as far as "
               "it fits), the GPU holds the most-used ones."]
    else:
        why = [f"Why: you chose the low-RAM mode (--low-ram {choice}); without it {model} needs {arena:.0f} + "
               f"{LOW_RAM_HEADROOM_GB} = {need:.0f} GB of RAM, this PC has {ram:.0f} GB."]
    return why + ["Its resident variant (the experts the GPU does not hold copied into RAM once: steady RAM use) runs "
                  "on one GPU: the engine has no layer split for it yet.",
                  f"To use all the GPUs: --gpus {','.join(str(i) for i in sel) if sel else '0,1'} - the experts the "
                  "GPUs do not hold are then read through the OS file cache: faster in two reports (1.3-1.6x, #364 "
                  f"#384), but RAM can fill up to 0 free during long prompts. Or {need:.0f} GB of RAM or more, or a "
                  "smaller size."]


def low_ram_together(a, model, ram, gpu, chosen) -> bool:
    """#364 #384: the low-RAM mode with several GPUs chosen.  One GPU is recommended: the resident variant keeps the
    experts the GPU does not hold in RAM (steady RAM use) and has no layer split.  All the GPUs together read those
    experts through the OS file cache instead (--mmap-experts) - 1.3-1.6x faster in those reports, but RAM can fill
    up to 0 free during long prompts.  An explicit --gpus (or an earlier install's cards) is kept; otherwise asked,
    one GPU by default (--yes: one GPU, as before).  True: all of them."""
    sel = [g["index"] for g in chosen]
    names = " + ".join(gpu_name(g) for g in chosen)
    if a.gpus:
        warn(f"the low-RAM mode on {names}, as you chose (--gpus): the experts the GPUs do not hold are read through "
             "the OS file cache (the resident variant has no layer split yet), and RAM can fill up to 0 free during "
             "long prompts")
        say(f"       One GPU keeps them in RAM (steady RAM use, recommended): --gpu {gpu['index']}")
        if a.low_ram == "resident":
            warn("--low-ram resident has no layer split yet: the experts are read through the OS file cache instead")
        return True
    if a.low_ram != "resident" and not a.yes:
        say()
        for line in low_ram_one_gpu_why(model, ram, a.low_ram, sel):
            say("  " + line)
        say(f"  1) {gpu_name(gpu)} only: the experts it does not hold kept in RAM where they fit   (recommended: "
            "steady RAM use)")
        say(f"  2) {names} together: the experts the GPUs do not hold read through the OS file cache - faster")
        say("     in two reports (1.3-1.6x, #364 #384), but RAM can fill up to 0 free during long prompts")
        if ask("Low-RAM mode: which GPUs?", ["1", "2"], "1", a.yes) == "2":
            ok(f"the low-RAM mode on {names}: the experts read through the OS file cache, as you chose")
            return True
        warn("the low-RAM mode: using " + gpu_name(gpu) + " only")
        return False
    warn("the low-RAM mode: using " + gpu_name(gpu) + " only (recommended)")
    for line in low_ram_one_gpu_why(model, ram, a.low_ram, sel):
        say("       " + line)
    return False


def unsloth_together(a, model, ram, gpu, chosen) -> bool:
    """#498: UD-Q4_K_XL with several GPUs chosen.  Its RAM budget (--resident-budget-gib) has no layer split, so a
    split runs without it: all its experts loaded into RAM from the GGUFs at start, as with the 2-3-bit models - only
    where the RAM holds the GGUF files and 24 GB more (unsloth_split_need_gb; 165 GiB, 2x RTX 3090: 31 -> 64-78
    tok/s).  An explicit --gpus is honoured there; otherwise asked, one GPU by default (--yes: one GPU, as before); an
    explicit --resident-budget-gib keeps one GPU.  #642: the engines from RESIDENT_SPLIT_ENGINE keep the budget on a
    split (the experts no card holds, the hottest by the whole profile), so there the cards go together with it, at
    any RAM: --gpus, or "2" when asked; one GPU stays the recommendation (--yes alone: one GPU, as before).  True: all
    of them."""
    names = " + ".join(gpu_name(g) for g in chosen)
    if resident_split():
        note = "with its RAM budget (each card caches its own layers' experts, the RAM the hottest of the rest; #642)"
        if a.gpus:
            ok(f"{model} on {names}, as you chose (--gpus), {note}")
            return True
        if not a.yes:
            say()
            say(f"  {model} runs with a RAM budget of its experts, on one GPU or on {names} together:")
            say(f"  1) {gpu_name(gpu)} only   (recommended: the tested setup)")
            say(f"  2) {names} together: each card caches the experts of its own layers (#642)")
            if ask(f"{model}: which GPUs?", ["1", "2"], "1", a.yes) == "2":
                ok(f"{model} on {names}, {note}")
                return True
        warn(f"{model} runs on one GPU: using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen)
             + f" shares it across {names}, with its RAM budget)")
        return False
    need = unsloth_split_need_gb(model)
    if ram < need:
        note = (f"{model} on several GPUs has no RAM budget and needs ~{need:.0f} GB of RAM (its GGUF files and "
                f"{UNSLOTH_RAM_LEFT_GB} GB more), this PC has {ram:.0f}; the estimate is a worst case (#737: a 128 GB PC "
                "ran it)")
        if a.gpus and a.resident_budget_gib is None:    # #737: asked for by name: a question, not a refusal
            confirm_risk(note + ", but it may page or stall here", True, a.yes,
                         f"{model} on several GPUs needs ~{need:.0f} GB of RAM; this PC has {ram:.0f} GB",
                         f"start it on one GPU: --gpus {gpu['index']}", "  Use them together anyway?")
            ok(f"{model} on {names}, as you chose (--gpus), with {ram:.0f} GB of RAM")
            return True
        warn(f"{note} - using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen) +
             " uses them together anyway)")
        return False
    if a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib has no layer split: {model} runs on one GPU with it - using {gpu_name(gpu)} only "
             f"(leave the budget out to use {names} together)")
        return False
    note = (f"no RAM budget - all of its experts (~{MODELS[model]['arena_gb']:.0f} GB) are loaded into RAM from the "
            f"model files at start, and the files pass through the OS file cache (needs ~{need:.0f} GB of RAM, this "
            f"PC has {ram:.0f})")
    if a.gpus:
        ok(f"{model} on {names}, as you chose (--gpus): {note}")
        return True
    if not a.yes:
        say()
        say(f"  {model} can run on one GPU with a RAM budget of its experts, or on {names} together without one:")
        say(f"  1) {gpu_name(gpu)} only: the most-used experts kept in RAM, the rest read from the SSD   (recommended: "
            "the tested setup)")
        say(f"  2) {names} together: {note};")
        say("     about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s)")
        if ask(f"{model}: which GPUs?", ["1", "2"], "1", a.yes) == "2":
            ok(f"{model} on {names}: {note}")
            return True
    warn(f"{model} runs on one GPU: using {gpu_name(gpu)} only (--gpus " + ",".join(str(g["index"]) for g in chosen) +
         f" shares it across {names} without the RAM budget: this PC's RAM holds it)")
    return False


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """The owner's rule: setup recommends, it never forces.  A choice setup expects to fail or run badly is said
    plainly (msg), then asked (default n; `default` keeps an older question's own default), or with --yes taken as
    consent when it was asked for explicitly (a flag such as --model or --gpus): --yes alone keeps the stop
    (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def confirm_paging(model, ram, choice, yes, explicit_model=False):
    """The model's experts do not fit this PC's RAM and the low-RAM mode is off.  #125: a warning and a question, not
    a stop - the user may accept paging.  Asked "no" by default, so an unattended --yes install stops here, unless
    the low-RAM mode was turned off explicitly (--low-ram off, #250) or the size was (--model): that is the choice
    already made."""
    need_gb, arena = MODELS[model]["ram_gb"], MODELS[model]["arena_gb"]
    off = choice == "off"
    confirm_risk(f"{model} needs about {need_gb} GB of RAM and this PC has {ram:.0f} GB: its experts alone are "
                 f"{arena:.0f} GB and must stay in RAM, so Windows/Linux will page part of them from disk. Expect it "
                 "to be much slower, and it may not start at all.\n       A smaller size (Q2_0 or IQ2_XS) fits; more "
                 "RAM fixes it.", off or explicit_model, yes,
                 f"{model} needs about {need_gb} GB of RAM; this PC has {ram:.0f} GB",
                 "choose Q2_0 or IQ2_XS, or add RAM" + ("" if off else f"; or --model {model} --yes (or --low-ram off "
                                                                        "--yes) to install it anyway"),
                 "  Install it anyway?", "y" if off else "n")
    warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose" + (" (--low-ram off)" if off else
                                                                          " (--model)" if explicit_model else ""))


def ctx_ram_need(model, ctx, low_ram=False):
    """#406: the RAM (GB) setup estimates for a long context with IQ3_XXS / IQ3_S: their experts + the context's
    8-bit KV cache + 24 GB of room for everything else (the 0.1.29 arithmetic, counted).  None where the context does
    not count against RAM by this rule: the other sizes, and the low-RAM mode (its KV cache stays in VRAM)."""
    if model not in ("IQ3_XXS", "IQ3_S") or low_ram:
        return None
    return MODELS[model]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24


def ram_ctx(model, ram, low_ram=False) -> int:
    """#406: the longest context the RAM rule recommends: 128K, or longer where the estimate fits this PC's RAM.  It
    is part of the recommended default (the smaller of it and the GPU's rule); a longer choice is kept, with a note.
    The 200K menu step is never the recommendation (#608)."""
    # 204800 (#608) is a menu step between 128K and 256K, never the recommendation: where 256K does not fit, 128K stays
    return max(c for c in CONTEXTS if c <= 131072 or (c != 204800 and (ctx_ram_need(model, c, low_ram) or 0) <= ram))


def app_settings_file() -> Path | None:
    """The desktop app keeps its own settings file (STRATA_SETTINGS_FILE, in the app's data folder) and never reads or writes the
    one an older command-line Strata uses, so an old install is never adopted or moved."""
    raw = os.environ.get("STRATA_SETTINGS_FILE")
    return Path(raw) if raw else None


def settings_path() -> Path:
    own = app_settings_file()
    if own is not None:
        return own
    if WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Strata" / "settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    if s.get("data_dir") and s.get("data_dir") != load_settings().get("data_dir"):
        s = {**s, "data_dir_at": time.time()}           # when the models folder was chosen: the app and the Terminal keep the newer choice
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def other_installs(settings: dict) -> list:
    """Strata folders besides this one that may hold model files: the ones this user ran before, and Strata* folders
    next to this one (a zip unpacked again lands in e.g. `Strata-main (1)\\Strata-main`)."""
    cands = [Path(p) for p in settings.get("installs", []) if not in_trash(p)]
    for base in dict.fromkeys(() if app_settings_file() else (home().parent, home().parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != home() and d not in found and (d / "setup.py").is_file():
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            atomic.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                nv, no = os.path.normcase(v), os.path.normcase(o)   # Windows: C:\ and c:\ are the same place
                if nv == no or nv.startswith(no + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def library_size_gb(folder: Path) -> float:
    """The GB of model files (models, packs, mtp) in a data folder."""
    total = 0
    for d in DATA_ITEMS:
        for root, _dirs, files in os.walk(folder / d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total / 1e9


def same_folder(a: Path, b: Path) -> bool:
    """The same folder, however it is spelled (a different case, a short 8.3 name, a link)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def data_folder(requested: str | None, move: bool = False, yes: bool = True, set_default: bool = False) -> tuple:
    """(the data folder, folders that still hold model files and are used where they are).

    A --data-dir that names a NEW folder never moves anything by itself.  When model files already exist elsewhere,
    the person is asked: keep them where they are and use the new folder for new downloads (the answer with --yes
    or when nobody can answer), or move them in (`move`, --move-data, or the answer "move").  Only a move, or
    --set-default-data-dir, or a PC with no model files anywhere, changes the remembered data folder.  Without
    --data-dir the model files of this folder and of earlier Strata folders on the same drive are gathered into the
    remembered data folder and their configs point there (that is the data folder's own job)."""
    settings = load_settings()
    remembered = Path(settings["data_dir"]) if settings.get("data_dir") else home().parent / "Strata-data"
    dest = Path(requested).expanduser().resolve() if requested else remembered
    from strata_control import folders
    bad = folders.problem(dest)
    if bad and requested:
        fail(folders.words(bad)[:400], "Nothing was downloaded or changed, and the folder was not saved as your data folder.")
    if bad and settings.get("data_dir"):                # an unusable folder saved earlier: go back to the default and say so; never keep it
        warn(f"The data folder saved before cannot be used ({bad[0]}) Using {home().parent / 'Strata-data'} instead.")
        remembered = dest = home().parent / "Strata-data"
        settings = {k: v for k, v in settings.items() if k != "data_dir"}
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {home()}")
        dest = home()
    elsewhere = []
    # #198: the data folder remembered before (a --data-dir to a new place) is a source too, and so is a Strata-data
    # folder nested in any of them (an install that kept its models one level down)
    sources = [home(), *other_installs(settings)]
    if settings.get("data_dir") and not same_folder(Path(settings["data_dir"]), dest):
        sources.append(Path(settings["data_dir"]))
    sources += [f / "Strata-data" for f in list(sources) if not same_folder(f / "Strata-data", dest)]
    if requested and not same_folder(dest, remembered):
        sources.append(remembered)
        # a different folder than the one in use: nothing is moved unless the person says so
        held = []
        for folder in dict.fromkeys(sources):
            if not same_folder(folder, dest) and has_data(folder) and not any(same_folder(folder, h) for h in held):
                held.append(folder)
        movable = [f for f in held if same_drive(f, dest)]
        if movable and not move and not yes:
            gb = sum(library_size_gb(f) for f in movable)
            say()
            size = f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"
            say(f"  Your models are in {', '.join(str(f) for f in movable)} ({size}).")
            say(f"  keep) leave them there and use {dest} only for new downloads")
            say(f"  move) move your {size} of models to {dest}")
            move = ask(f"Move your {size} of models to the new folder, or use the new folder only for new downloads?",
                       ["keep", "move"], "keep", False) == "move"
        if held and not move:
            for folder in held:
                say(f"  Models you already have in {folder} stay there; new downloads go to {dest}.")
            if set_default:
                save_settings({**settings, "data_dir": str(dest)})
            return dest, held
        if not held:
            move = True                                 # nothing to lose track of: the new folder becomes the default
    seen = set()
    for folder in sources:
        key = os.path.normcase(str(folder))
        if key in seen:
            continue
        seen.add(key)
        if same_folder(folder, dest) or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 70 GB copy)
            continue
        # the downloads merge file by file (the same file wherever it came from); a prepared pack or MTP layer moves
        # whole or not at all, so two copies are never mixed
        if (folder / "models").is_dir():
            move_into(folder / "models", dest / "models")
        for item in [*((folder / "packs").glob("*") if (folder / "packs").is_dir() else []), folder / "mtp"]:
            rel = item.relative_to(folder)
            if item.exists() and not (dest / rel).exists():
                move_into(item, dest / rel)
        for d in ("packs",):
            try:
                (folder / d).rmdir()                    # empty now
            except OSError:
                pass
        for c in model_config_files(folder):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(home())] + [p for p in settings.get("installs", [])
                                if p != str(home()) and Path(p).is_dir() and not in_trash(p)]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


def data_folder_for(a) -> tuple:
    """data_folder() with the answers on the command line (--data-dir, --move-data, --yes, --set-default-data-dir)."""
    return data_folder(getattr(a, "data_dir", None), getattr(a, "move_data", False), bool(getattr(a, "yes", True)),
                       getattr(a, "set_default_data_dir", False))


def data_folder_in_place() -> tuple:
    """(the data folder, []) for a start: where settings.json says the model files are, or the default place.  Reads
    the settings and nothing else: no folder is created, nothing is moved or copied, no setting is saved.  The run
    configs name their model files, so an install made by an earlier Strata starts from where it is."""
    settings = load_settings()
    dest = Path(settings["data_dir"]) if settings.get("data_dir") else home().parent / "Strata-data"
    from strata_control import folders
    if settings.get("data_dir") and folders.problem(dest):      # a folder saved earlier that cannot be used: the default, and it is not kept
        dest = home().parent / "Strata-data"
    return dest, []


def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (#459): to a temporary file first, then moved over the old one, so
    a setup stopped half-way (a closed window, a full disk) never leaves an empty strata-*.json behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    atomic.replace(tmp, path)


# #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the user's - a
# "sampling" or "mcp_servers" block, "allowed_hosts", "cors_origins", "open_browser" - and is kept when setup runs again
# "args_written" (strata_control/user_args.py) is setup's own; "user_args" is not in SETUP_KEYS on purpose: carry_over keeps it,
# then applies the person's options over the generated ones
SETUP_KEYS = frozenset({"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port", "backend", "env",
                        "gpu", "gpus_asked", "gpus_by", "layer_split", "host", "api_key", "draft_vocab", "vision", "args_written"})
SETUP_ENV = frozenset({"STRATA_HIPBLASLT_TUNING", "STRATA_ROCBLAS_TUNING", "STRATA_RESIDENT_PIN", "STRATA_NO_ARENA_THP", "STRATA_MMQ_KQ", "STRATA_HC_Q8",
                       "STRATA_ROUTE_TAIL_SKIP"}) | frozenset(GFX1103_SWITCHES) | frozenset(GFX1150_SWITCHES)   # the "env" entries setup writes (the gfx1103 switches are setup's too: a re-run refreshes them, they do not linger)
# 0.1.44 added these three, written for Unsloth's files only (unsloth_default_env).  In any other model's config they are the
# user's own (STRATA_ROUTE_TAIL_SKIP=0 to turn the tail skip off, STRATA_MMQ_KQ=1 ...): a setup run again keeps them there
# (0.1.44 dropped them with the rest of SETUP_ENV).
SETUP_ENV_UNSLOTH = frozenset({"STRATA_MMQ_KQ", "STRATA_HC_Q8", "STRATA_ROUTE_TAIL_SKIP"})
SETUP_VISION = frozenset({"exe", "cpu_exe", "mmproj", "model", "gpu", "max_tokens", "threads"})


def carry_over(old: dict, cfg: dict, explicit: list[str] | None = None) -> list[str]:
    """#629: setup run again for an installed model keeps what the user added to its run config: every key setup does
    not write (`SETUP_KEYS`), the "env" entries setup does not write, and in "vision" the keys setup does not write
    plus an mmproj of their own (a Q8_0 one, #625) that still exists.  `cfg` (the new config) is updated in place;
    the names of what was kept are returned.  Engine options added by hand to "args" are not merged (setup chooses
    those): `args_dropped` names them.  `explicit`: engine options given on this run's command line (they win and are
    kept from now on, see carry_user_args)."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    mine = SETUP_ENV if "-unsloth-" in str(cfg.get("model_name") or "") else SETUP_ENV - SETUP_ENV_UNSLOTH
    env = {k: v for k, v in (old.get("env") or {}).items() if k not in mine and k not in (cfg.get("env") or {})} \
        if isinstance(old.get("env"), dict) else {}
    if env:
        cfg["env"] = {**(cfg.get("env") or {}), **env}
        kept += [f"env {k}" for k in env]
    ov, nv = old.get("vision"), cfg.get("vision")
    if isinstance(ov, dict) and isinstance(nv, dict):
        for k, v in ov.items():
            if k not in SETUP_VISION and k not in nv:
                nv[k] = v
                kept.append(f"vision {k}")
        mm = ov.get("mmproj")                          # a file of the user's own: not the one setup downloads
        if isinstance(mm, str) and Path(mm).name != Path(str(nv.get("mmproj"))).name and Path(mm).is_file():
            nv["mmproj"] = mm
            kept.append("vision mmproj")
    kept += carry_profile_args(old, cfg)
    kept += carry_user_args(old, cfg, explicit)
    off = draft_layer.apply(cfg)                       # "mtp": false was the user's (a key setup does not write): the new args honour it
    if off:
        kept.append("mtp off (" + " ".join(off) + " left out of args)")
    return kept


# the options setup chooses itself on every run from its answers (context, KV type, pictures, the draft layer, the layer split):
# a hand edit of one of these is carried by the answers (choices_from_config) or by its own key ("mtp", "vision"), not by user_args
SETUP_CHOICE_FLAGS = frozenset({"--max-context", "--kv", "--vision", "--mtp", "--spec-min-p", "--batch-mtp", "--layer-split"})


def setup_flags() -> frozenset:
    """Every engine option this file names (setup.py's own source): the options setup can write.  Used for a run config
    that has no `args_written` (written by an earlier Strata): an option in its args that is not in this set was added by hand."""
    global _SETUP_FLAGS
    if _SETUP_FLAGS is None:
        try:
            text = Path(__file__).read_text(encoding="utf-8")
            _SETUP_FLAGS = frozenset(re.findall(r'"(--[a-z][a-z0-9-]+)"', text))
        except OSError:
            _SETUP_FLAGS = frozenset()
    return _SETUP_FLAGS


_SETUP_FLAGS = None


def explicit_args(a) -> list[str]:
    """The engine options this run's command line sets that the person keeps from now on (user_args): --vram-reserve-mib."""
    n = getattr(a, "vram_reserve_mib", None)
    return ["--vram-reserve-mib", str(n)] if n is not None else []


def carry_user_args(old: dict, cfg: dict, explicit: list[str] | None = None) -> list[str]:
    """#3216 #3215 #3270: the engine options the person set by hand in the earlier run config are put back over the
    generated ones, and recorded in the new one's "user_args" (strata_control/user_args.py).  They are the earlier
    "user_args", the options of "args" that differ from "args_written" (hand edits), for a config with no "args_written"
    the options its args had and the new ones do not (control vectors, --spec, ...) and an explicit --vram-reserve-mib of
    the earlier file.  `explicit`: options given on this run's command line, which win.  The names kept are returned."""
    nargs = cfg.get("args")
    if not isinstance(nargs, list):
        return []
    mine = user_args.capture(old, SETUP_CHOICE_FLAGS)
    oargs = old.get("args")
    if user_args.WRITTEN_KEY not in old and isinstance(oargs, list):
        have = {user_args.flag_of(g) for g in user_args.groups(mine)}
        mine += user_args.flat([g for g in user_args.groups(user_args.dropped_by_setup(oargs, nargs, setup_flags(), SETUP_CHOICE_FLAGS))
                                if user_args.flag_of(g) not in have])
        reserve = flag_value(oargs, "--vram-reserve-mib")   # an old file: the reserve it names, unless it is the vision default or the app's own
        vis = isinstance(old.get("vision"), dict)
        if reserve is not None and reserve.isdigit() and not (vis and int(reserve) == VISION["gpu"]["reserve_mib"])                 and "ui_resource_baseline" not in old and "--vram-reserve-mib" not in mine:
            mine += ["--vram-reserve-mib", reserve]
    for g in user_args.groups(explicit or []):
        mine = user_args.merge(mine, g)
    if not mine:
        user_args.stamp(cfg, [])
        return []
    cfg["args"] = user_args.merge(nargs, mine)
    user_args.stamp(cfg, mine)
    return ["args " + f for f in dict.fromkeys(g[0] for g in user_args.groups(mine) if g[0].startswith("--"))]


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return str(args[i + 1])
    return None


def carry_profile_args(old: dict, cfg: dict) -> list[str]:
    """#775: a learned expert profile the user wired in by hand (--expert-profile <their file>, --expert-profile-save
    <file>) survives a setup run: setup writes the shipped profile, which a measured one of their own beats (2.5-3.1x at
    256K on a 16 GB card in the report).  Their --expert-profile replaces the shipped one only when it is another file
    that still exists; --expert-profile-save is kept when the new config has none.  The names kept are returned."""
    oargs, nargs = old.get("args"), cfg.get("args")
    if not isinstance(oargs, list) or not isinstance(nargs, list):
        return []
    kept = []
    mine, shipped = flag_value(oargs, "--expert-profile"), flag_value(nargs, "--expert-profile")
    if mine and shipped and Path(mine).name != Path(shipped).name and Path(mine).is_file():
        nargs[nargs.index("--expert-profile") + 1] = mine
        kept.append("args --expert-profile")
    save = flag_value(oargs, "--expert-profile-save")
    if save and flag_value(nargs, "--expert-profile-save") is None:
        nargs += ["--expert-profile-save", save]
        kept.append("args --expert-profile-save")
    return kept


def args_dropped(old: dict, cfg: dict) -> list[str]:
    """#629: the engine options of the earlier run config that the new one has no more (by flag name): options added
    by hand, which a setup run does not carry over - the start of the line that names them."""
    def flags(c):
        a = c.get("args") if isinstance(c.get("args"), list) else []
        return [str(x) for x in a if str(x).startswith("--")]
    new = set(flags(cfg))
    return list(dict.fromkeys(f for f in flags(old) if f not in new))


def engine_exe(eng: Path) -> str:
    """The `exe` of a run config: the engine component in the app, else the engine binary's path."""
    return "component:strata-engine" if CONTROL else str(eng / EXE)


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None, explicit: list[str] | None = None) -> None:
    """#629: setup's run config, written over an earlier one for the same model without losing what the user added
    to it: the keys setup does not write are carried over (carry_over), and the earlier file is kept as
    strata-<model>.json.bak when it changes.  `source`: an earlier install's config to carry the keys over from when
    this folder has none yet (a copy set up like the last one).  A line says what was kept, one what was not."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg, explicit) if old is not None else []
    if old is None:
        carry_user_args({}, cfg, explicit)              # a first install: only what this run's command line sets (nothing to report as kept)
    user_args.stamp(cfg)                               # args_written: what setup wrote, for the next hand-edit check
    if "images" in cfg and cfg["images"] is None:      # --images no (or not downloaded): no image section
        cfg.pop("images")
    bak = None
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
            bak = None
    write_config(cfg_path, cfg)
    if kept:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))
    if bak is not None:
        dropped = args_dropped(old, cfg) if old is not None else []
        say(f"  the earlier run config is kept as {bak.name}" + (
            f"; engine options it had that this one has not (setup chooses those): {' '.join(dropped)}"
            if dropped else ""))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def model_config_files(folder: Path) -> list:
    """#346: the strata-*.json files of a folder that can be model configs. The server keeps a config's Chat settings
    next to it as strata-<model>.shared-settings.json: it matches the pattern but is no config (no exe, no args)."""
    return [p for p in folder.glob("strata-*.json") if not p.name.endswith(".shared-settings.json")]


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another Strata folder on this PC, for a folder that has none yet.  One
    that does not parse (an empty or cut-off file, #459) is skipped with a warning: the newest readable one is used,
    and with none this copy is set up as a fresh install."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += model_config_files(folder)
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def own_path_config(cfg: dict) -> bool:
    """A config for a family with its own engine path (Qwen3.6): its --mtp is a draft count, not a draft layer."""
    return cfg.get("arch") in {d["arch"] for d in OWN_PATH_FAMILIES.values()}


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (family, size, context, KV, images, projection, network)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    big = next((f for f, d in MULTI_FILE_FAMILIES.items() if tag.startswith(d["tag"])), None)
    if big is not None:                                # DeepSeek-V4.1-Flash: its own size; no KV / images / projection settings
        d = MULTI_FILE_FAMILIES[big]
        size = next((m for m in d["models"] if m.lower() == tag[len(d["tag"]):].lower()), None)
        a = cfg.get("args", [])
        ctx = a[a.index("--max-context") + 1] if "--max-context" in a and a.index("--max-context") + 1 < len(a) else None
        return {"family": big, "model": size, "context": int(ctx) if ctx and ctx.isdigit() else None, "kv": None,
                "vision": "none", "esp": "off", "host": cfg.get("host"), "api_key": cfg.get("api_key"),
                "port": cfg.get("port"), "gpu": cfg.get("gpu"), "layer_split": None, "cuda": None, "vram_reserve_mib": None}
    own = next((f for f, d in OWN_PATH_FAMILIES.items() if tag.startswith(d["tag"])), None)
    if own is not None:                                # Qwen3.6: its own sizes; no KV / images / projection settings
        d = OWN_PATH_FAMILIES[own]
        size = next((m for m in d["models"] if m.lower() == tag[len(d["tag"]):].lower()), None)
        a = cfg.get("args", [])
        ctx = a[a.index("--max-context") + 1] if "--max-context" in a and a.index("--max-context") + 1 < len(a) else None
        vis = cfg.get("vision")
        return {"family": own, "model": size, "context": int(ctx) if ctx and ctx.isdigit() else None, "kv": None,
                "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none", "esp": "off", "host": cfg.get("host"), "api_key": cfg.get("api_key"),
                "port": cfg.get("port"), "gpu": cfg.get("gpu"), "layer_split": None,
                "cuda": 12 if config_toolkit(cfg) == 12 else None, "vram_reserve_mib": None}
    family = next((f for f, d in FAMILIES.items() if d["tag"] and tag.startswith(d["tag"])), "qwen")
    model = (tag[len(FAMILIES[family]["tag"]):] if tag.startswith(FAMILIES[family]["tag"]) else tag).upper()
    if model not in MODELS:                            # (sizes have no dash except UD-Q4_K_XL: the old rule)
        model = tag.split("-")[-1].upper()
    a = cfg.get("args", [])
    val = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
    vis = cfg.get("vision")
    esp = val("--control-vector-scaled")
    esp_path = esp.rsplit(":", 1)[0] if esp else None
    return {"family": family, "model": model if model in MODELS else None,
            "context": int(val("--max-context")) if val("--max-context") else None,
            "kv": val("--kv") if val("--kv") in ("int8", "q4_0") else None,
            "vision": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "none",
            "esp": ("on" if Path(esp_path).name == ESP_VECTOR.name else esp_path) if esp_path else "off",
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "layer_split": cfg.get("layer_split"), "cuda": 12 if config_toolkit(cfg) == 12 else None,
            # #493: --vram-reserve-mib given at setup (images write the default 700 themselves)
            "vram_reserve_mib": int(val("--vram-reserve-mib")) if (val("--vram-reserve-mib") or "").isdigit() and (
                vis is None or int(val("--vram-reserve-mib")) != VISION["gpu"]["reserve_mib"]) else None}


def calibration_targets(have: list, model: str | None, family: str | None) -> list:
    """#2036: the installed configs `--calibrate --model X` (and/or --family) means; every config with neither."""
    if not (model or family):
        return [c for c in have if choices_from_config(c)["family"] not in (*OWN_PATH_FAMILIES, *MULTI_FILE_FAMILIES)]
    out = []
    for c in have:
        ch = choices_from_config(c)
        if ch["family"] in OWN_PATH_FAMILIES or ch["family"] in MULTI_FILE_FAMILIES:   # the tuning measures Flash-Next's engine settings
            continue
        if (not model or ch["model"] == model) and (not family or ch["family"] == family):
            out.append(c)
    return out


def find_in(roots: list, rel: str):
    """The first of roots/rel that exists."""
    for r in roots:
        if (r / rel).exists():
            return r / rel
    return None


# ------------------------------------------------------------------------------------------------ start
def model_config(path: Path) -> bool:
    """#549: a model's run config (a JSON object with "exe" and "args"). Any other strata-*.json in the folder (a
    file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a Strata model config")
    return False


def installed_configs():
    return [p for p in sorted(model_config_files(home()), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version).  "0" when there is none to read: the packaged
    Terminal carries no CMakeLists.txt (it ships VERSION and BUNDLE.json), and a missing file was a FileNotFoundError out of
    prebuilt_bases() that ended the whole setup (#4188)."""
    try:
        m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return "0"
    return m.group(1) if m else "0"


def binary_engine_version(exe) -> str | None:
    """The version compiled into an engine binary (the `engine=X.Y.Z` line it prints at start; 0.1.13 and newer carry
    it), read from the file without running it.  None: unreadable, or a binary without it."""
    try:
        m = re.search(rb"engine=(\d+\.\d+\.\d+(?:\.\d+)?)\n", Path(exe).read_bytes())
    except OSError:
        return None
    return m.group(1).decode() if m else None


def stale_binary(exe) -> bool:
    """#2054: the engine binary is of another version than this source tree builds.  A BUILD.json that carries the
    source's hash and version cannot be trusted alone: a build that stopped (a host compiler the toolkit rejects) can
    leave the old binary beside a stamp of the new source, and nothing would ever compile it again."""
    v, source = binary_engine_version(exe), source_version()
    return v is not None and source != "0" and v != source      # "0": no source tree to compare with (the packaged Terminal)


def engine_version(exe: Path) -> tuple:
    """The version in the engine folder's BUILD.json, or else the one compiled into the binary.  A locally compiled
    engine is not necessarily the source's version: when compiling a `git pull` fails, the previous engine is kept
    (issue #49)."""
    try:
        meta = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    v = str(meta.get("version") or "")
    if not v:                                          # the version compiled into the binary: 0.1.13 and newer
        try:                                           # carry it, so a binary without it is older
            m = re.search(rb"engine=(\d+\.\d+\.\d+(?:\.\d+)?)\n", Path(exe).read_bytes())
            v = m.group(1).decode() if m else "0.1.12"
        except OSError:
            v = "0"
    return tuple(int(x) for x in v.split(".")[:4] if x.isdigit())


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.uname().release.lower()


def ple_ram_platform_ok() -> bool:
    """147-verify: --ple-io ram for a PC with RAM to spare was measured on Linux only (Windows locks pages differently and
    WSL pins little), so only there."""
    return not WIN and not is_wsl()


def rotational_disk(path) -> str | None:
    """#605 (Linux): the disk's name when `path` is on a rotational disk (sysfs queue/rotational), else None."""
    if WIN:
        return None
    try:
        st = os.stat(path)
        p = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        for q in (p, p.parent):                        # a partition has its disk's queue
            f = q / "queue" / "rotational"
            if f.exists():
                return q.name if f.read_text().strip() == "1" else None
    except (OSError, ValueError, AttributeError):
        pass
    return None


def hip_config_cards(sel) -> list[dict]:
    """#566: the AMD cards a HIP config runs on - its "gpu" (one index or a list) in HIP's numbering, as amd_gpus
    lists them; no "gpu": the supported card with the most VRAM, as setup picks it.  Each card's name carries its
    architecture (an RX 7900 XTX and an RX 7900 XT are both gfx1100; a card without a product name in sysfs is known
    by its arch only).  A card that is not found is {} (the key then says "?")."""
    amd = amd_gpus()
    if sel is None:
        usable = [g for g in amd if amd_problem(g) is None]
        sel = min(usable, key=amd_rank)["index"] if usable else None
    byid = {g["index"]: g for g in amd}
    cards = []
    for i in (sel if isinstance(sel, list) else [sel]):
        g = byid.get(i)
        if g is None:
            cards.append({})
            continue
        arch = g.get("arch") or ""
        name = g.get("name") or "?"
        cards.append({**g, "name": name if not arch or arch in name else f"{name} ({arch})"})
    return cards


def gpu_sel(value) -> list:
    """A config's "gpu" as a list of card numbers: 0, [0, 2] or the string form "0,2" (which serve splits
    the same way).  [] when it names none.  Without this, a string "0,1" reached gpu_info() whole and
    calibrate died with "there is no GPU 0,1" on a dual-GPU config."""
    if value is None or value == "":
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    sel = []
    for x in items:
        x = str(x).strip()
        if x == "":
            continue
        sel.append(int(x) if x.isdigit() else x)
    return sel


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: this GPU, CPU and RAM, and the model with its context and images setting
    (the context's KV cache and the image encoder take VRAM from the expert cache).  #566: a HIP config's cards are
    AMD's (hip_config_cards) - nvidia-smi's list named them "?" (or another card with that number) before."""
    sel = cfg.get("gpu")
    if isinstance(sel, str):
        sel = gpu_sel(sel)
    if cfg.get("backend") == "hip":
        gl = hip_config_cards(sel)
    else:
        gl = [gpu_info(i) or {} for i in sel] if isinstance(sel, list) else [gpu_info(sel) or {}]
    g = {"name": " + ".join(x.get("name", "?") for x in gl), "vram_gb": sum(x.get("vram_gb", 0) for x in gl)}
    a = cfg.get("args", [])
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a else "?"
    return "|".join([g.get("name", "?"), f"{g.get('vram_gb', 0):.0f}GB", cpu_info()[0], f"{ram_gb():.0f}GB",
                     cfg.get("model_name", "?"), ctx, "images" if "--vision" in a else "text"])


def calibrate_config(cfg_path: Path) -> bool:
    """Measure the engine's hardware-dependent settings on this PC (tools/calibrate.py), write them into the run
    config and remember them per PC and model in the settings file, so an update or a reinstall keeps them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import calibrate as CAL
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    say()
    say("  Tuning Strata for this PC: the output speed is measured with a few engine settings (the PCIe share, the")
    say("  draft depth, the CPU threads, the expert cache; on two or more cards also the prompt chunks). It takes")
    say("  roughly 15-30 minutes (3 rounds of each value); the PC is busy meanwhile.")
    try:
        since = os.path.getsize(cfg["log"]) if cfg.get("log") and os.path.isfile(cfg["log"]) else 0
    except OSError:
        since = 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops an install: the defaults stay
        warn(f"the tuning did not finish ({e}): the default settings stay")
        why = CAL.engine_error(cfg.get("log"), since)  # #447: the engine's own reason, not only "see the log"
        if why:
            say(f"       the engine said: {why}")
        return False
    cfg["args"] = CAL.apply(cfg["args"], res["settings"])
    if "env" in res:                                   # HIP: the engine variables it measured (STRATA_SH_STREAM)
        cfg["env"] = CAL.apply_env(cfg.get("env"), res["env"])
    write_config(cfg_path, cfg)
    st = load_settings()
    st.setdefault("calibration", {})[hardware_key(cfg)] = {"settings": res["settings"], "env": res.get("env") or {},
                                                           "tok_s": res["report"].get("tok_s"),
                                                           "date": time.strftime("%Y-%m-%d")}
    save_settings(st)
    chosen = {**res["settings"], **(res.get("env") or {})}
    if chosen:
        ok("tuned for this PC: " + ", ".join(f"{k} {v}" if k.startswith("--") else f"{k}={v}" for k, v in chosen.items())
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    else:
        ok("tuned for this PC: the default settings are already the fastest here"
           + (f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""))
    return True


def saved_calibration(cfg: dict) -> dict | None:
    """The settings an earlier calibration found for this PC and model, if any."""
    return (load_settings().get("calibration") or {}).get(hardware_key(cfg))


def setup_calibration(cfg: dict, hip: bool) -> dict | None:
    """The calibration a (re-)install applies to its new config: the one saved for this PC and model.  #566: on Linux
    HIP too - hardware_key now names the AMD cards, so a `./setup.sh --calibrate` run is matched to its card and
    model.  Windows HIP keeps the defaults for now (not tried there).  Setup still offers the tuning itself on NVIDIA
    only: each control is verified on HIP first."""
    if hip and WIN:
        return None
    return saved_calibration(cfg)


def upgrade_config(cfg_path: Path, cfg: dict) -> dict:
    """Configs written before v0.1.13 read prompts in fixed 2048-token chunks; the engine now picks the chunk
    itself (`--prefill auto`: up to 8192, as the free VRAM allows - about 2x faster on long prompts).  Under WSL,
    KV streaming is dropped: its RAM copy must be pinned, and the driver pins only about 1 GB there."""
    a = cfg.get("args", [])
    changed = False
    ver = engine_version(cfg["exe"]) if "--prefill" in a else (0, 0, 0)
    if "--prefill" in a and a[a.index("--prefill") + 1] == "2048" and ver >= (0, 1, 13):
        a[a.index("--prefill") + 1] = "auto"
        changed = True
        ok("prompt reading: the engine now picks its chunk size (--prefill auto)")
    elif "--prefill" in a and a[a.index("--prefill") + 1] == "auto" and (0, 0, 0) < ver < (0, 1, 13):
        a[a.index("--prefill") + 1] = "2048"           # an older engine kept after a failed update (issue #49)
        changed = True
        warn(f"the installed engine is {'.'.join(map(str, ver))}: prompts are read in 2048-token chunks until it is updated")
    if is_wsl() and "--kv-resident" in a:
        i = a.index("--kv-resident")
        del a[i:i + 2]
        changed = True
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
    if changed:
        write_config(cfg_path, cfg)
    return cfg


def update_install(have: list, a) -> int:
    """#475: `setup.py --update` (UPDATE.bat / update.sh, after their git pull): what a plain START-HERE.bat does to
    an install before it starts the model, without starting it - the Python packages, the ready-made engine when this
    setup needs a newer one (MIN_ENGINE; a compiled engine when its source changed), each installed model's config
    upgrades and its draft subset.  No question is asked and the model files are not touched; a model still running
    keeps its engine (update_installed_engine says to close it and run this again)."""
    have = [p for p in have if model_config(p)]        # #549: a strata-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this Strata folder yet: run START-HERE.bat (Linux: ./setup.sh) to set it up -")
        say("  it finds an earlier install's model files next to it and reuses them.")
        return 0
    pip_install(runtime_packages(),
                "numpy, jinja2, regex, pyyaml, tqdm, requests, pillow, psutil, textual (the screens)")
    if not a.build:
        update_installed_engine(a.prebuilt)
    mtp_done: set = set()
    for cfg_path in have:
        cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
        if not own_path_config(cfg):
            refresh_shared_draft_vocab(cfg, mtp_done)
        if cfg.get("backend") == "hip" and WIN:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461
        ok(f"{cfg.get('model_name', cfg_path.stem)}: up to date")
    ver = engine_version(Path(json.loads(have[0].read_text(encoding="utf-8-sig"))["exe"]))
    say()
    start = "START-HERE.bat" if WIN else "./setup.sh"
    if any(ver) and tuple(ver) < MIN_ENGINE:             # #3653: the engine step failed above; "updated" would be untrue
        warn(f"Strata's files are updated, but the engine was NOT: it is still {'.'.join(map(str, ver))} and this Strata wants "
             f"{'.'.join(map(str, MIN_ENGINE))} or newer. The model still starts with the old engine.")
        say(f"  Next step: check the internet connection (and that the model window is closed), then run {'UPDATE.bat' if WIN else './update.sh'} again; "
            "the line above this one says why the engine was not fetched.")
        return 0
    ok("Strata is updated" + (f" (engine {'.'.join(map(str, ver))})" if any(ver) else "") +
       f". Start the model with {start} when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564: the settings a start uses, in one line: the config's engine options (the model's file paths left out)
    and the server's own fields, so a change made by hand to strata-<model>.json can be checked without the log."""
    a, out, i = [str(x) for x in cfg.get("args") or []], [], 0
    while i < len(a):
        flag = a[i]
        val = a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None
        i += 1 if val is None else 2
        if not flag.startswith("--"):
            continue                                   # a positional: the model file
        if val is not None and ("/" in val or "\\" in val or val.lower().endswith((".gguf", ".bin"))):
            continue                                   # a path: --native, --mtp, --profile ...
        out.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', 8080)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "layer_split", "draft_vocab", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            srv.append(f"{k} {','.join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}")
    return " ".join(out) + ("; " if out else "") + "server " + ", ".join(srv)


def widen_setups_one_card(cfg_path: Path, cfg: dict, amd: list) -> dict:
    """A DeepSeek run config on AMD that names one card and was written by setup 0.2.8 or earlier (the card was setup's own pick, marked "asked"
    by mistake) moves to every card that counts, here at the start, with its pack and files untouched.  A card the person named
    (strata_control/gpu_pick.py) stays.  Linux only: several AMD cards sharing a model is not supported on Windows."""
    if WIN or not isinstance(cfg, dict) or cfg.get("gpus_by") or not cfg.get("gpus_asked"):
        return cfg
    try:
        from strata_control import gpu_pick
        usable = amd_usable_cards(amd)
        if not gpu_pick.looks_like_setups_pick(cfg, usable):
            return cfg
        cards = [c["index"] for c in dsv41_auto_cards(amd, ROOT / "engine")]
        if len(cards) < 2:
            return cfg
        was = gpu_pick.single_card(cfg)
        new = gpu_pick.widened(cfg, cards)
        write_config(cfg_path, new)
        ok(f"DeepSeek now uses all {len(cards)} of your AMD cards (it was set up on GPU {was} alone by an earlier Strata). Nothing is rebuilt. "
           f"To keep it on one card: strata launch --gpu {was} (it is remembered from then on)")
        return new
    except Exception as e:                             # noqa: BLE001 - never stop a start over this
        warn(f"could not check which cards this model should use ({e})")
        return cfg


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
          layer_split=None, keep=None) -> int:
    """Start the model.  Nothing is installed, downloaded or compiled while it starts (strata_control/acquisition.py):
    what is missing is named with the command that installs it (SETUP / UPDATE / strata install)."""
    with acquisition.forbidden():
        return _start(cfg_path, port, gpu, open_browser, yes, layer_split, keep)


def _start(cfg_path: Path, port: int | None, gpu: int | list | None = None, open_browser=True, yes=False,
           layer_split=None, keep=None) -> int:
    """keep: settings given on this start that the model keeps from now on (--host, --api-key, --draft-vocab,
    --vram-reserve-mib)."""
    cfg = upgrade_config(cfg_path, json.loads(cfg_path.read_text(encoding="utf-8-sig")))
    component = engine_component.is_reference(cfg["exe"])
    # The server resolves and verifies component engines; the reference itself is not a file.
    files = ([] if component else [cfg["exe"]]) + [a for a in cfg["args"] if a.endswith(".gguf")]
    missing = [p for p in files if not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup to repair")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    reserve = keep.pop("vram_reserve_mib", None)       # #493: an engine argument, kept in the config's args
    if reserve is not None:
        args = cfg["args"]
        if "--vram-reserve-mib" in args[:-1]:
            args[args.index("--vram-reserve-mib") + 1] = str(reserve)
        else:
            args += ["--vram-reserve-mib", str(reserve)]
        cfg[user_args.USER_KEY] = user_args.set_flag(user_args.capture(cfg), "--vram-reserve-mib", reserve)   # the person's own value: kept by every later setup run and start
        if user_args.WRITTEN_KEY in cfg:
            cfg[user_args.WRITTEN_KEY] = list(cfg["args"])
        write_config(cfg_path, cfg)
        ok(f"saved for this model: {reserve} MiB of VRAM kept free for other programs (--vram-reserve-mib)")
    if keep and any(cfg.get(k) != v for k, v in keep.items()):   # #179: a --host/--api-key on a start was ignored
        cfg.update(keep)
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(
            "api key" if k == "api_key" else ("the browser opens" if v else "no browser") if k == "open_browser"
            else f"{k.replace('_', ' ')} {v}" for k, v in keep.items()))
    cfg_path.touch()                                     # the most recently used model
    if "--mtp" in cfg["args"][:-1] and not own_path_config(cfg):
        refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if cfg.get("backend") == "hip":                    # AMD, numbered as HIP numbers them (setup's KFD order)
        if WIN and not component:
            hip_runtime_beside_exe(Path(cfg["exe"]).parent)   # #468 #461: also fixes a 0.1.34 install
        amd = amd_gpus()
        if isinstance(gpu, list):                      # --gpus: saved, this model runs on these cards from now on
            cards = amd_parse_gpus(",".join(str(i) for i in gpu), amd)
            built = [] if component else (engine_archs_hip() or [])
            miss = [x for x in cards if built and x["arch"] not in built]
            if miss:
                fail("the installed engine has no code for " + ", ".join(f"{x['name']} ({x['arch']})" for x in miss),
                     "set it up for these cards: ./setup.sh --setup --backend hip --gpus " + ",".join(map(str, gpu)))
            cfg["gpu"], cfg["gpus_asked"], cfg["gpus_by"] = gpu, True, "person"
            cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
            split_budget(cfg, yes, True)               # #498: before it is saved (asks when the RAM is short, #737)
            write_config(cfg_path, cfg)
            gpu = None
        elif gpu is not None:
            cmd += ["--gpu", str(gpu)]
            if not isinstance(gpu, list):                  # a DeepSeek model on AMD keeps a card the person names (the others: this start only)
                from strata_control import gpu_pick
                saved = gpu_pick.remember(cfg_path, [gpu])
                if saved is not None:
                    cfg = saved
                    ok(f"saved for this model: it runs on GPU {gpu} from now on")
        else:
            cfg = widen_setups_one_card(cfg_path, cfg, amd)   # a DeepSeek config from setup 0.2.8 and earlier: setup's own one-card pick
        found = []
        use = gpu if gpu is not None else cfg.get("gpu")
        if isinstance(use, list):
            byid = {x["index"]: x for x in amd}
            ok("GPUs: " + " + ".join(gpu_name(byid[i]) if i in byid else f"GPU {i} (not found)" for i in use)
               + f" together, AMD (layers split {cfg.get('layer_split') or 'auto'})")
        else:
            g = next((x for x in amd if x["index"] == (use if use is not None else x["index"])
                      and amd_problem(x) is None), None)
            if g is not None:
                ok(f"GPU: {g['name']} ({g['vram_gb']:.0f} GB, AMD)")
    else:
        found = gpus()
        global OLD_GPUS                                # Pascal / Volta: named on this start, or this model's CUDA 12 engine
        OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, gpu if isinstance(gpu, list) else [gpu] if gpu is not None else
                                               cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [cfg.get("gpu")],
                                               12 if config_toolkit(cfg) == 12 else None)
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(gpu, list):                        # --gpus: saved, this model runs on these cards from now on
        check_gpus(gpu, found, yes=yes, named=True)
        cfg["gpu"], cfg["gpus_asked"] = gpu, True
        cfg["layer_split"] = layer_split or cfg.get("layer_split") or "auto"
        split_budget(cfg, yes, True)                   # #498: before it is saved (asks when the RAM is short, #737)
        recommend_remote_expert_opt(cfg)
        write_config(cfg_path, cfg)
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        check_gpus([gpu], found)
        cmd += ["--gpu", str(gpu)]
    else:
        cfg = offer_together(cfg_path, cfg, yes)
    use = gpu if gpu is not None else cfg.get("gpu")
    # #364 #384: a resident low-RAM config on several GPUs; #498: a UD-Q4_K_XL config with its RAM budget (by hand)
    if isinstance(use, list) and (split_mmap(cfg) | split_budget(cfg)):
        write_config(cfg_path, cfg)
    if cfg.get("backend") == "hip":
        pass
    elif isinstance(use, list):
        check_gpus(use, found, "(chosen for this model) ", yes=True, named=True)
        byid = {g["index"]: g for g in found}
        cfg = ensure_engine_for([byid[i] for i in use], cfg_path, cfg, yes)
        ok("GPUs: " + " + ".join(gpu_name(byid[i]) for i in use) + f" together (layers split {cfg.get('layer_split') or 'auto'})")
    elif found:
        g = next((x for x in found if x["index"] == use), None) if use is not None else max(
            found, key=lambda x: (round(x["vram_gb"]), -x["index"]))
        if g is not None:
            cfg = ensure_engine_for([g], cfg_path, cfg, yes)
            ok("GPU: " + gpu_name(g))
    browser = open_browser and cfg.get("open_browser") is not False   # #609: "open_browser": false, --no-browser
    if browser:
        cmd.append("--open")
    gb = 0.0
    if "--native" in cfg["args"]:
        try:
            gb = Path(cfg["args"][cfg["args"].index("--native") + 1]).stat().st_size / 1e9
        except (OSError, IndexError):
            pass
    say()
    say("  " + "-" * 100)
    size = f'about {gb:.0f} GB' if gb >= 1 else '34-55 GB'
    a_ = cfg["args"]
    if "--mmap-experts" in a_ and "--resident-budget-gib" not in a_ and "--resident-experts" not in a_:
        # #505: the mapped low-RAM mode loads nothing into RAM up front (the server's narrator says the same)
        say(f"  Starting {cfg.get('model_name', 'the model')}: it maps {size} of experts from the model files (the OS "
            "file cache reads them).")
    else:
        say(f"  Starting {cfg.get('model_name', 'the model')}: it loads {size} into RAM and locks part of it for the "
            "GPU.")
    say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES (longer the first time after a")
    say("  restart). That is normal: please wait and don't close this window - " + (
        "the browser opens when it is ready." if browser else "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    if not WIN and os.environ.get("STRATA_EXECV"):
        # Replace this process instead of spawning a child. The Docker image sets STRATA_EXECV=1,
        # so there the server is PID 1 and docker stop's SIGTERM reaches the process that can
        # answer the engine with QUIT. Normal Linux starts keep spawning the server as a child.
        os.execve(cmd[0], cmd, {**os.environ, acquisition.ENV: "1"})
    # the server and everything it starts are told, too: nothing in a running model installs or downloads
    return subprocess.call(cmd, env={**os.environ, acquisition.ENV: "1"})


# the draft subsets setup copied before (sha256): replaced by the current one, a subset made by hand is kept
OLD_DRAFT_VOCABS = {"369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"}   # to 0.1.26: 27 Han tokens


DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}


def saved_draft_vocab(cfg_path: Path) -> str | None:
    """The draft subset a model's config chose earlier (--draft-vocab), or None: a setup run again without the flag
    rewrites the config, and would otherwise put the default subset back."""
    try:
        v = json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("draft_vocab")
    except (OSError, ValueError, AttributeError):
        return None
    return v if v in DRAFT_VOCABS else None


def vision_tokens(asked: int | None, vision: str, earlier: Path | None) -> int:
    """#625: the most image tokens a picture becomes (the config's vision.max_tokens): --vision-tokens N, else what an
    earlier config of this model chose for the same encoder device (a setup run again keeps it), else the device's
    default (VISION).  More is allowed with a note on the time it takes: setup recommends, it does not cap."""
    default = VISION[vision]["max_tokens"]
    if asked is None and earlier is not None:
        try:
            v = json.loads(earlier.read_text(encoding="utf-8-sig")).get("vision")
        except (OSError, ValueError, AttributeError):
            v = None
        mt = v.get("max_tokens") if isinstance(v, dict) and bool(v.get("gpu")) == (vision == "gpu") else None
        if isinstance(mt, int) and mt > 0 and mt != default:
            asked = mt
    if asked is None:
        return default
    note = ""
    if vision == "cpu" and asked > default:
        note = (" - on the CPU a picture takes longer to encode the more tokens it gets (several seconds more at "
                "1,024 than at 300)")
    elif asked > VISION["gpu"]["max_tokens"]:
        note = " - more than the encoder's default needs more VRAM and context per picture"
    ok(f"images: up to {asked} image tokens per picture (--vision-tokens; default {default}){note}")
    return asked


DRAFT_VOCAB_MIB = {"cjk": 348, "cyrillic": 193, "fr": 151, "en": 133}   # the draft head's VRAM per subset (IQ3_S: the largest)
SMALL_DRAFT_VRAM_GB = 14   # #474: below this the default subset's head can be what does not fit


def draft_vocab_note(vram_gb: float, chosen: str | None) -> list[str]:
    """#474: on a card under 14 GB, the default draft subset (cjk, ~348 MiB of VRAM) can be what does not fit at the
    start ("the draft head does not fit"), and the engine's expert cache gets what a smaller one leaves.  Setup
    RECOMMENDS a smaller one here and changes nothing (the owner's rule, #403 #406): a subset chosen with
    --draft-vocab, or kept from an earlier install, gets no note.  [] for every other case."""
    if chosen or not 0 < vram_gb < SMALL_DRAFT_VRAM_GB:
        return []
    start = "START-HERE.bat" if WIN else "./setup.sh"
    return [f"Tip for a {vram_gb:.0f} GB card: the draft layer's default token subset (with Chinese, Japanese and "
            f"Korean) needs up to ~{DRAFT_VOCAB_MIB['cjk']} MiB of VRAM.",
            f"  For English and code answers, {start} --draft-vocab en needs up to ~{DRAFT_VOCAB_MIB['en']} MiB "
            f"(cyrillic: ~{DRAFT_VOCAB_MIB['cyrillic']}) and leaves the rest to the expert cache - and it is the",
            "  fix when the start stops with \"the draft head does not fit\". The model keeps the choice."]


SMALL_CARD_GB = 7.5            # #496: a card under 8 GB gets a tip (an 8 GB card lists 7.99)


def small_card_note(ctx: int, draft_vocab: str | None) -> list[str]:
    """#496: what frees VRAM on a card under 8 GB when the start stops with "no VRAM is left for the expert cache"
    (the engine already lowers its own reserve on such a card) - a recommendation, setup changes none of it.  (The
    draft layer stays: the server needs it.)"""
    start = "START-HERE.bat --setup" if WIN else "./setup.sh"
    tips = []
    if ctx > 8192:
        tips.append("an 8K context (a smaller KV cache)")
    if draft_vocab != "en":
        tips.append(f"--draft-vocab en (a draft head of ~{DRAFT_VOCAB_MIB['en']} MiB instead of "
                    f"~{DRAFT_VOCAB_MIB[draft_vocab or 'cjk']})")
    lines = ["If the start stops with \"no VRAM is left for the expert cache\" (the engine's log says how much is "
             "short):"]
    if tips:
        lines.append(f"  run {start} again with " + " and ".join(tips) + ", or close other programs that use the GPU.")
    else:
        lines.append("  close other programs that use the GPU.")
    return lines


# Unsloth's files on a card of about 8 GB.  The dense side is Q8_0 (~3 GB) and the draft layer binds ~0.8 GB, so little is
# left for the expert cache.  Measured on an 8 GB Arc A750 (the Intel worker, 2026-10-13): setup's 8192 / int8 / reserve 400
# start stopped with "no VRAM is left for the expert cache, short 473 MiB"; --max-context 4608 --kv q4_0 --vram-reserve-mib 0
# started with 3 cache slots (4.4 tok/s).  UD_FIT_FIXED_MIB is that measurement turned into a constant (everything but the
# session's KV, the reserve and the draft layer); an ESTIMATE for CUDA and HIP cards of the same size, whose engine says what
# is short if it is off.  The engine itself lowers its default reserve to 300 MiB on such a card.
UD_FIT_FIXED_MIB = 7356
UD_FIT_MTP_MIB = 802
UD_FIT_MIN_SLACK_MIB = 64         # at least a few cache slots beside the estimate's error
UD_FIT_CARD_GB = 9.5              # the check is for cards up to here
UD_FIT_RUNTIME_MIB = 250          # CUDA / HIP: the runtime's own context (the SYCL measurement above has none of it)
UD_FIT_ENGINE_RESERVE_MIB = 300   # the reserve the engine settles on when none is given and the card is tight
KV_WORDS = {"int8": "8-bit", "q4_0": "4-bit (Hadamard-rotated)", "k8v4": "8-bit keys, 4-bit values", "fp16": "16-bit"}
UD_FIT_KV_BYTES = {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}   # per token per attention layer (13)
UD_FIT_CTX_STEPS = (8192, 6144, 4608)


def unsloth_runtime_mib(gpu: dict) -> int:
    return 0 if (gpu or {}).get("driver") in ("xe", "i915") else UD_FIT_RUNTIME_MIB


def unsloth_vram_slack_mib(vram_gb: float, ctx: int, kv: str, reserve_mib: int, mtp: bool = True,
                           runtime_mib: int = 0) -> float:
    """MiB left for the expert cache by setup's estimate (negative: the engine would refuse to start)."""
    session = ctx * 13 * UD_FIT_KV_BYTES.get(kv, 1056) / 2 ** 20
    return vram_gb * 1024 - UD_FIT_FIXED_MIB - (UD_FIT_MTP_MIB if mtp else 0) - session - max(reserve_mib, 0) - runtime_mib


def unsloth_small_card_plan(vram_gb: float, ctx: int, kv: str, reserve_mib: int | None, ctx_fixed=False, kv_fixed=False,
                            runtime_mib: int = 0):
    """(ctx, kv, mtp, slack_mib) of the least harmful start that the estimate says fits, or None when none does.
    Order of harm: the KV in 4 bits, then a smaller context (down to 4608), then no draft layer on the GPU (decode
    up to 2x slower, but 0.8 GB for the cache); a value the user chose (--context, --kv) is not changed."""
    reserve = UD_FIT_ENGINE_RESERVE_MIB if reserve_mib is None else reserve_mib
    ctxs = [ctx] if ctx_fixed else [ctx] + [c for c in UD_FIT_CTX_STEPS if c < ctx]
    kvs = [kv] if kv_fixed or kv == "q4_0" else [kv, "q4_0"]
    for mtp in (True, False):
        for c in ctxs:
            for k in kvs:
                slack = unsloth_vram_slack_mib(vram_gb, c, k, reserve, mtp, runtime_mib)
                if slack >= UD_FIT_MIN_SLACK_MIB:
                    return c, k, mtp, slack
    return None


def unsloth_small_card_lines(model: str, vram_gb: float, plan, start_ok: bool) -> list[str]:
    """The words for a ~8 GB card: that UD-* is very slow here and what to pick instead (always), and when the start
    would have stopped, that and what setup changed instead.  Recommend, never force."""
    lines = [f"{model} on a {vram_gb:.0f} GB card is very slow: the dense side and the draft layer fill the card, the expert "
             "cache gets a few slots, and nearly every expert comes from RAM or the SSD (3-5 tok/s measured on an 8 GB "
             "Arc A750). IQ3_XXS or IQ3_S of the Qwen3.8-Flash-Next family fit such a card far better; run setup again "
             "and pick one of them."]
    if not start_ok and plan is None:
        lines.append("By setup's estimate even a 4608 context with a 4-bit KV cache and no draft layer leaves no room "
                     "for the expert cache on this card: the start will probably stop with \"no VRAM is left for the "
                     "expert cache\".")
    return lines


PARALLEL_MAX = 8               # #465: the engine's batch window holds at most 8 requests
PARALLEL_SHARE = 0.2           # #465: the slots' sessions may take this share of the VRAM the expert cache would hold
PARALLEL_HELD = 0.5            # #465: ... and only where the cache still holds this share of the experts beside them
PARALLEL_COST_NOTE = ("parallel N reduces waiting for several users but costs about 10-25% speed per request on this "
                      "card")


def parallel_slot_gb(ctx: int, kv: str, streaming: bool) -> float:
    """#465: the VRAM one batch slot's session takes: its KV cache (12 QSA layers; with KV streaming only the 32K
    positions the attention reads stay in VRAM) and the DeltaNet state (~0.17 GB).  Measured: 0.56 GiB at 32K int8."""
    kv_tok = 12 * (576 if kv == "q4_0" else 1056)
    return (min(ctx, 32768) if streaming else ctx) * kv_tok / 1e9 + 0.17


def parallel_recommend(vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> int:
    """#465: how many requests at once ("parallel") to recommend: 0 = none (one at a time).  Only where the experts
    mostly fit in VRAM - the expert cache (each card's VRAM less ~5 GB, every card of a layer split) still holds
    PARALLEL_HELD of the model's experts beside the slots' sessions, which take at most PARALLEL_SHARE of it, up to 4.
    Where the experts mostly run on the CPU a batch reads about as many experts as the requests one by one and every
    slot's VRAM is expert cache lost: measured on a 12 GB RTX 5070 (Q2_0, 32K), a request alone 11-24% slower with 2-4
    slots, 4 requests together 63 tok/s against 71 one after the other (docs/BATCHING.md)."""
    if isinstance(vram_gbs, (int, float)):
        vram_gbs = [vram_gbs]
    cache_gb = sum(max(0.0, v - 5) for v in vram_gbs)
    slot = parallel_slot_gb(ctx, kv, streaming)
    best = 0
    for n in (2, 3, 4):
        if n * slot <= PARALLEL_SHARE * cache_gb and (cache_gb - n * slot) >= PARALLEL_HELD * arena_gb:
            best = n
    return best


def parallel_note(asked: int | None, vram_gbs, arena_gb: float, ctx: int, kv: str, streaming: bool) -> list[str]:
    """#465: what setup says about "parallel": the recommendation (or, where it would cost speed, why it is left at
    one), or how the asked count compares with it (kept as asked: recommend, never force)."""
    rec = parallel_recommend(vram_gbs, arena_gb, ctx, kv, streaming)
    slot = parallel_slot_gb(ctx, kv, streaming)
    if asked is None or asked <= 1:
        if not rec:
            return [f"Several requests at once: left at one at a time - {PARALLEL_COST_NOTE} (docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together instead of one "
                f"after the other (each takes ~{slot:.1f} GB of VRAM from the expert cache; docs/BATCHING.md)."]
    lines = [f"parallel requests: {asked} at once (each takes ~{slot:.1f} GB of VRAM from the expert cache, "
             f"{asked * slot:.1f} GB in all)"]
    if asked > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec:
        lines.append(f"recommended for this card: one at a time - {PARALLEL_COST_NOTE}; kept as you chose")
    elif asked > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


PREFILL_BIG_RAM_GB = 96        # bench #433 #440 #834 #669: --prefill auto:32768 +21-35% at 96 GB, ~3x slower at 32 GB
PREFILL_RISK_RAM_GB = 64       # below this an explicit auto:32768 is warned about
HEADROOM_RAM_GB = 48           # bench #834: STRATA_RESIDENT_HEADROOM_GIB=6 on a PC with this much RAM or less
AGENT_CACHE_FREE_GB = 24       # bench #882 #440: RAM left beside the model for the conversation cache
AGENT_CACHE_MIB = 8192
SMALL_VISION_VRAM_GB = 12.5    # bench #469: the GPU image encoder shrinks the prompt chunks on cards up to 12 GB


def arg_after(args, flag):
    """The value after `flag` in an argument list, or None."""
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def bench_tips(args, env, ram: float, model_ram_gb: float, vram_gb: float, vision: str, win: bool) -> list[str]:
    """Recommendations from the community bench data (plan 0.1.40 item 14).  Text only: nothing here changes a
    default, the config or the engine's arguments (recommend, never force).  `ram` is this PC's RAM, `model_ram_gb`
    what the chosen model keeps in it."""
    tips = []
    cache = arg_after(args, "--expert-cache")
    if win and cache is not None and cache != "auto":
        tips.append(f"warning: --expert-cache {cache} is a fixed size. On Windows, a size that leaves almost no VRAM free "
                    "has run up to 7x slower (#780 #781); a smaller number, or auto, leaves the driver room")
    prefill = arg_after(args, "--prefill")
    if prefill == "auto:32768" and ram < PREFILL_RISK_RAM_GB:
        tips.append(f"warning: --prefill auto:32768 on {ram:.0f} GB of RAM: it ran ~3x slower than --prefill auto with "
                    "32 GB (#834 #669); it paid off (+21-35%) with 96 GB")
    elif prefill == "auto" and ram >= PREFILL_BIG_RAM_GB:
        tips.append(f"tip: with {ram:.0f} GB of RAM, --prefill auto:32768 in the config's args read prompts 21-35% "
                    "faster in community benchmarks (#433 #440 #834); not set, nothing changes")
    resident = any(a in args for a in ("--resident-experts", "--resident-budget-gib"))
    if resident and ram <= HEADROOM_RAM_GB and "STRATA_RESIDENT_HEADROOM_GIB" not in (env or {}):
        tips.append(f"tip: on a PC with {ram:.0f} GB of RAM, \"env\": {{\"STRATA_RESIDENT_HEADROOM_GIB\": \"6\"}} in the "
                    "config kept decode speed in a community benchmark and left the system 2 GiB more (#834); the "
                    "default is 4")
    if vision == "gpu" and 0 < vram_gb <= SMALL_VISION_VRAM_GB:
        tips.append("tip: the image encoder on the GPU makes a prompt's chunks smaller on a card this size (243 vs 750 "
                    "tok/s measured, #469); run setup again with --vision cpu to read prompts ~3x faster, at about "
                    "2-3 s per picture")
    return tips


DISPLAY_RESERVE_MIB = 1500     # #779: the tip for an NVIDIA card that also drives a display
DESKTOP_RESERVE_MIB = 3072     # #560 #516: what kept a KDE/Wayland desktop alive beside a full expert cache


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X)."""
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_reserve_note() -> list[str]:
    """#560 #516: an AMD card that also drives a Linux desktop - with the default 700 MiB reserve the expert cache
    fills it, and when the desktop needs more VRAM amdgpu moves the cache to system RAM, where the OOM killer then ends
    the compositor.  A recommendation, setup changes nothing."""
    return [f"If this AMD card also drives your desktop and the desktop or apps crash once the model is loaded, keep "
            f"more VRAM free: ./setup.sh --vram-reserve-mib {DESKTOP_RESERVE_MIB}",
            "  (remembered for this model; the expert cache gets ~2.3 GB less, a few % of speed)"]


def mtp_corrupt(mtp: Path, env=None) -> bool:
    """#327: True when the MTP tensors an install fetched are not the pinned checkpoint's (tools/mtp_fetch.py verify,
    which hashes only files that changed since they last checked out).  A mirror that ignored range requests left the
    shards' starts there instead, and the draft layer built from them accepted nothing - with no error anywhere."""
    if not (mtp / "tensors").is_dir():
        return False
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "verify", "--out", str(mtp)], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    return r.returncode == 3


def refresh_draft_vocab(rt: Path, choice: str = "cjk") -> None:
    """The draft layer's token subset in the MTP folder: `cjk` (data/draft_vocab.bin, since 0.1.27, #137), `en`
    (data/draft_vocab_en.bin, the English/code subset before it: ~110 MiB less VRAM, English answers 1-2% faster) or
    `cyrillic` (data/draft_vocab_cyrillic.bin: English/code and the whole Cyrillic script, for Ukrainian, Russian,
    Bulgarian, Serbian... answers) or `fr` (data/draft_vocab_fr.bin: English/code and the tokens of French text, #597).
    Copied when missing or when a shipped subset other than the chosen one is there; a subset made by hand is kept."""
    new, dst = ROOT / "data" / DRAFT_VOCABS.get(choice, "draft_vocab.bin"), rt / "draft_vocab.bin"
    if not new.exists() or not rt.is_dir():
        return
    if dst.exists():
        old = hashlib.sha256(dst.read_bytes()).hexdigest()
        shipped = OLD_DRAFT_VOCABS | {hashlib.sha256((ROOT / "data" / f).read_bytes()).hexdigest()
                                      for f in DRAFT_VOCABS.values() if (ROOT / "data" / f).exists()}
        if old not in shipped or old == hashlib.sha256(new.read_bytes()).hexdigest():
            return
        ok("draft layer: the token subset " + {"cjk": "with Chinese, Japanese and Korean",
                                               "cyrillic": "with the Cyrillic script",
                                               "fr": "for French"}.get(choice,
                                                                                          "for English and code (less VRAM)"))
    shutil.copyfile(new, dst)


def refresh_shared_draft_vocab(cfg: dict, done: set) -> None:
    """update_install's draft subset of one config, for configs taken most recently used first (installed_configs'
    order): an MTP folder already in `done` is left alone.  Several configs can name one MTP folder (a second config of
    the same model: another GPU mode, another context), each with its own "draft_vocab".  Refreshed once per config,
    the folder ended with the subset of the config used LONGEST ago: a config without the key (cjk) put the default
    subset over the cyrillic one of the config in use, and its Russian answers were drafted without Cyrillic tokens.
    The folder takes the subset of the most recently used config that names it; a start still installs its own
    config's."""
    args = cfg.get("args") or []
    if "--mtp" not in args[:-1]:
        return
    rt = Path(args[args.index("--mtp") + 1])
    key = os.path.normcase(os.path.abspath(rt))
    if key in done:
        return
    done.add(key)
    refresh_draft_vocab(rt, cfg.get("draft_vocab", "cjk"))


def ensure_engine_for(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """The installed engine must have code for every card the model starts on: a card added later (--gpus with an
    older or newer generation, #128) or a new GPU in the PC otherwise stops the start with 'no kernel image'.  Such a
    card gets the engine compiled for all of them, before the start.  A Pascal / Volta card added to a model on the
    CUDA 13 engine moves the model to the experimental CUDA 12 engine (CUDA 13 has no code for it)."""
    if engine_component.is_reference(cfg.get("exe")):
        return cfg                                    # the server checks the installed component, not ROOT/engine
    tk = config_toolkit(cfg)
    if tk == 13 and any(int(g["arch"]) < CUDA13_MIN_ARCH for g in cards):
        return use_cuda12(cards, cfg_path, cfg, yes)
    missing = [g for g in cards if not (engine_runs_on(g) if tk == 13 else engine_runs_on(g, tk))]
    if not missing:
        return cfg
    acquisition.require("an engine with code for " + ", ".join(f"{g['name']} (sm_{g['arch']})" for g in missing),
                        ("START-HERE.bat --setup" if WIN else "./setup.sh --setup") + " (it compiles the engine for these cards)")
    eng = engine_dir(tk)
    info = eng / "BUILD.json"
    meta = json.loads(info.read_text(encoding="utf-8"))
    say()
    say("  The installed engine has no code for " + ", ".join(f"{g['name']} (sm_{g['arch']})" for g in missing) +
        ": it is compiled for " + ("these cards" if len(cards) > 1 else "it") + " now.")
    main = gpu_info(cards[0]["index"])
    archs = sorted(set(arch_numbers(meta.get("archs"))) | {int(g["arch"]) for g in cards})
    vision = meta.get("vision") or ("gpu" if (eng / VEXE).exists() else "none")
    build_engine({**main, "archs": archs}, vision, yes, get_llama_cpp(), toolkit=tk)
    dirs = json.loads(info.read_text(encoding="utf-8")).get("cuda_dirs") or []
    cfg["lib_dirs"] = dirs + [d for d in cfg.get("lib_dirs") or [] if d not in dirs]
    write_config(cfg_path, cfg)
    return cfg


def get_cuda12_engine(url_base, gpu, vision, yes, build=False) -> Path:
    """The experimental CUDA 12 engine for these cards (gpu["archs"]): the ready-made one (Windows) with NVIDIA's
    CUDA 12 libraries, or compiled here with a CUDA 12.x toolkit (Linux, --build, or no ready-made one)."""
    eng = None if build else get_prebuilt(url_base, gpu, vision, toolkit=12)
    if eng is not None and json.loads((eng / "BUILD.json").read_text(encoding="utf-8")).get("source") != "local":
        pip_cuda_libs(12)
        if vision != "none" and not (eng / VEXE).exists():
            eng = None
    return eng if eng is not None else build_engine(gpu, vision, yes, get_llama_cpp(), toolkit=12)


def engine_lib_dirs(eng: Path, toolkit=13) -> list:
    """The library folders a CUDA engine loads from: its own (a compiled one: the toolkit's), else pip's wheels."""
    meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    return meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(toolkit)


def use_cuda12(cards, cfg_path: Path, cfg: dict, yes: bool) -> dict:
    """A model on the CUDA 13 engine now has a Pascal / Volta card (a --gpus at start): the model's config moves to
    the experimental CUDA 12 engine (one engine per model; its other models keep theirs)."""
    acquisition.require("the experimental CUDA 12 engine")
    old = min(int(g["arch"]) for g in cards)
    acquisition.require("the experimental CUDA 12 engine for sm_%d" % old,
                        ("START-HERE.bat --setup --cuda 12" if WIN else "./setup.sh --setup --cuda 12"))
    say()
    warn(f"sm_{old} is older than CUDA 13 supports (it dropped Pascal and Volta): this model moves to the experimental "
         "CUDA 12 engine (docs/OLDER_GPUS.md; START-HERE.bat --setup --cuda 13 and newer cards only moves it back)")
    main = gpu_info(cards[0]["index"]) or cards[0]
    vision = "gpu" if cfg.get("vision") else "none"
    eng = get_cuda12_engine(os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                            {**main, "archs": sorted({int(g["arch"]) for g in cards})}, vision, yes)
    cfg["exe"] = str(eng / EXE)
    cfg["cuda"] = 12
    cfg["lib_dirs"] = engine_lib_dirs(eng, 12)
    if cfg.get("vision") and (eng / VEXE).exists():
        cfg["vision"]["exe"] = str(eng / VEXE)
    write_config(cfg_path, cfg)
    ok(f"engine: {eng / EXE} (CUDA 12, experimental)")
    return cfg


def write_run_script(model, cfg_path, port, open_browser=True):
    """run-<model>.bat / .sh: the server with this config; `open_browser` False (#609: --no-browser) leaves --open out."""
    serve = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
             "--port", str(port)] + (["--open"] if open_browser else [])
    if WIN:
        script = home() / f"run-{model.lower()}.bat"
        script.write_text("@echo off\r\ntitle Strata " + model + "\r\ncd /d \"" + str(ROOT) + "\"\r\n" +
                          " ".join(f'"{x}"' for x in serve) + "\r\nif errorlevel 1 pause\r\n", encoding="utf-8")
    else:
        script = home() / f"run-{model.lower()}.sh"
        script.write_text("#!/bin/sh\ncd \"" + str(ROOT) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n",
                          encoding="utf-8")
        script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ the rope config
def derived_factor(ctx: int, trained: int = 262144) -> float:
    """The automatic extension factor: the FINAL context over the trained one, at least 1.

    Factor 1 removes the automatic expansion - the trained angles stand as they are - but it is not a
    switch for rope as a whole: an explicitly chosen method's settings keep their defined behavior.
    """
    return max(1.0, float(ctx) / float(trained))


def resolve_rope(ctx: int, scaling, scale, trained: int = 262144):
    """The rope config for the context ACTUALLY SERVED: (scaling, scale); scaling None = no scaling flags.

    An explicit --rope-scaling/--rope-scale always wins - a valid user-supplied factor is kept verbatim even
    when a reduction changed the context.  Past the trained range an omitted method defaults to yarn -
    llama.cpp's extension method: the trained angles survive on the high-frequency pairs and the
    magnitude correction keeps the attention temperature - and an omitted factor is derived from the
    final context (final / trained, at least 1), as is an explicitly chosen method's missing factor
    inside the trained range: factor 1, the trained angles, no expansion.  An explicit none is refused
    past the trained range (the setup will not configure a run it knows is out of spec) rather than
    silently overridden.
    """
    if scale is not None and (not math.isfinite(scale) or scale < 1.0):
        raise ValueError("--rope-scale must be a finite value >= 1, e.g. --rope-scale 4")
    if ctx <= trained:
        if scale is not None and scaling in (None, "none"):
            raise ValueError("--rope-scale needs --rope-scaling linear or yarn (the chosen context fits the "
                             "trained 262144, so there is nothing to scale)")
        if scaling in (None, "none"):
            return None, None          # the stock model, by choice or by default
        return scaling, scale if scale is not None else derived_factor(ctx, trained)
    if scaling == "none":
        raise ValueError(f"a {ctx // 1024}K context is past the model's trained 262144, and --rope-scaling none "
                         "keeps the stock angles there - the model has never seen those positions, so the setup "
                         "refuses the combination instead of quietly overriding it. Pick --rope-scaling yarn or "
                         "linear, or rerun with --context 262144 or lower")
    return scaling or "yarn", scale if scale is not None else derived_factor(ctx, trained)


# ------------------------------------------------------------------------------------------------ main
def mac_setup(argv) -> int:
    """A Mac (Apple Silicon): the Metal engine, built from source on the Mac by metal/setup_mac.py (its own steps:
    the Mac check, the pack by unified memory, the engine build, the Metal defaults). Nothing of the CUDA / HIP paths
    is used or changed."""
    return subprocess.call([sys.executable, str(ROOT / "metal" / "setup_mac.py"), *argv])


def sycl_setup(argv) -> int:
    """--backend sycl: the Intel Arc engine (the SYCL port in sycl/, PR #423), experimental. There is no ready-made
    Intel engine: it is compiled from source on the PC (docs/INTEL_ARC.md), then sycl/setup_intel.py runs this setup
    with the Intel steps swapped in. Nothing of the CUDA / HIP paths is used or changed."""
    say()
    say("  Intel Arc (--backend sycl): supported since 0.1.40.2 on Linux, tested on an Arc Pro B70 (xe) and an Arc A750 "
        "(i915); other Arc cards and driver versions are untested, and reports help (docs/INTEL.md).")
    if WIN:
        fail("the Intel Arc engine has no Windows setup yet (no ready-made Intel engine either)",
             "run it on Linux (Ubuntu 24.04 with Intel's GPU driver and oneAPI): docs/INTEL_ARC.md")
    say("  There is no ready-made Intel engine: it is built from source with Intel oneAPI (icpx + oneMKL),")
    say("  docs/INTEL_ARC.md. Setup continues with sycl/setup_intel.py.")
    rest, skip = [], False
    for x in argv:                                     # setup_intel.py drives this setup through its AMD path
        if skip:
            skip = False
        elif x == "--backend":
            skip = True
        elif not x.startswith("--backend="):
            rest.append(x)
    script = ROOT / "sycl" / "setup_intel.py"
    if not script.exists():
        fail(f"{script} is missing", "use a full Strata checkout (git clone) - docs/INTEL_ARC.md")
    return subprocess.call([sys.executable, str(script), *rest])


def control_engine_step(gpu, vision):
    """Step 4 in the app: the engine is the component folder's package (strata_control/engine_component.py), found if it is
    there and never fetched or built. A model can be downloaded and prepared before the engine is installed; the run config
    names "component:strata-engine" and the engine is looked up when the model is loaded. llama.cpp's source zip is fetched
    for gguf-py only (the preparation tools), as setup always did: no git, no compiler."""
    from strata_control import engine_component as EC
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py for the preparation tools)")
    try:
        found = EC.resolve(full=False)
        eng, meta = Path(found.root), {"version": found.version, "backend": found.backend, "source": "local"}
        ok(f"engine component: {found.version} ({found.platform}-{found.backend}) in {eng}")
        if found.backend == "hip" and is_rdna2(gpu) and any(a in RDNA2_ARCHS for a in found.hip_archs):
            for line in rdna2_notes(gpu):                       # the RX 6000 package: built, not tested on a real card
                say(line)
    except EC.EngineMissing as e:
        plain = getattr(e, "plain_notice", None) or getattr(e, "driver_notice", None)
        if os.environ.get("STRATA_NEED_ENGINE") == "1" and os.environ.get("STRATA_ALLOW_NO_ENGINE") != "1":
            # strata launch needs an engine now: stop before step 5 instead of after the download, unless launch itself asked to go on
            # (STRATA_ALLOW_NO_ENGINE: the person said yes to "download the model anyway").  The app does not stop (owner decision
            # 2026-10-18): it says so in its download card and still downloads the model.
            fail(plain or "no engine is installed for this PC",
                 "get the engine that matches this graphics card (`strata update`, or the app's Settings), then run this again; "
                 "nothing was downloaded and files already downloaded are kept")
        eng = home() / "engine"
        meta = {"version": "99.0.0", "source": "local", "component": "missing"}   # its version is checked when it loads
        warn(f"{plain + ' ' if plain else ''}no engine is installed for this PC yet ({e}). The model is downloaded and prepared now; "
             "install the engine component for your GPU (the app's Settings, or the Terminal) before loading it")
    if vision != "none":
        if (eng / VEXE).is_file():
            ok(f"image encoder: {eng / VEXE} (carried by the engine component)")
            return llama, eng, meta, [], gpu, vision
        warn("this engine component has no image encoder (strata-vision): images off for this model")
    return llama, eng, meta, [], gpu, "none"


def engine_step(a, gpu, hip, amd, cuda_tk, vision):
    """Step 4: llama.cpp's sources and the engine (ready-made, or compiled here).  Returns (llama, engine folder, its
    BUILD.json, its library folders, the GPU (AMD on Windows: as the engine sees it), the image-encoder choice)."""
    if CONTROL:
        return control_engine_step(gpu, vision)
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py, ggml, mtmd)")
    if hip and WIN:                                    # AMD on Windows: the ready-made HIP engine (no compiler)
        eng = None if a.build else get_prebuilt_hip(a.prebuilt, gpu)
        if eng is None:
            fail("no ready-made AMD engine for this Strata version" + (" (--build)" if a.build else ""),
                 "compiling it on Windows: tools\\hip\\build_windows.bat makes strata-windows-x64-hip.zip, then run "
                 "START-HERE.bat --backend hip --prebuilt <its dist folder> (docs/AMD_HIP.md)")
        gpu = hip_card(eng, gpu, amd)
        a.gpu = gpu["index"] if gpu["count"] > 1 else a.gpu
    else:
        eng = None if a.build or hip else get_prebuilt(a.prebuilt, gpu, vision, **({"toolkit": 12} if cuda_tk == 12
                                                                                    else {}))
    if eng is not None and not hip and json.loads((eng / "BUILD.json").read_text(encoding="utf-8")).get("source") != "local":
        pip_cuda_libs(cuda_tk)
        if vision != "none" and not (eng / VEXE).exists():
            warn("the ready-made engine has no image encoder: compiling it")
            eng = None
        else:
            vision = prebuilt_vision(json.loads((eng / "BUILD.json").read_text(encoding="utf-8")), gpu, vision)
    if eng is None:
        eng = build_engine_hip(gpu, llama, vision) if hip else build_engine(gpu, vision, a.yes, llama, toolkit=cuda_tk)
    meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
    if hip and WIN:                                    # the ready-made engine's rocm/bin, first on the engine's PATH
        lib_dirs = [str(d) for d in hip_lib_dirs(eng)]
    else:
        lib_dirs = meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(cuda_tk)
    return llama, eng, meta, lib_dirs, gpu, vision


def engine_families(eng: Path) -> set:
    """The model families an engine runs (`strata --families`; an engine from before it: Flash-Next's only)."""
    try:
        text = subprocess.run([str(eng / EXE), "--families"], capture_output=True, text=True, errors="replace", timeout=60,
                              env=engine_env(eng)).stdout      # #2372: with the engine's rocm/bin on PATH
    except (OSError, subprocess.SubprocessError):
        return {"qwen4exp"}
    names = set(text.split())
    return names if "qwen4exp" in names else {"qwen4exp"}


def own_path_context(fam: dict, vram_gb: float) -> int:
    """The recommended context for an own-path family: its KV cache (VRAM) beside the dense weights and the most-used
    experts - 32K up to 16 GB, 64K up to 24 GB, 128K above."""
    return 32768 if vram_gb <= 16.5 else 65536 if vram_gb <= 24.5 else 131072


def own_path_size(fam: dict, ram: float) -> str:
    """The recommended size: the largest file whose RAM need this PC has (the smallest when none fits)."""
    fits = [m for m, d in fam["models"].items() if ram >= d["ram_gb"]]
    return max(fits, key=lambda m: fam["sha256"][fam["file"].format(q=m)][0]) if fits else next(iter(fam["models"]))


def own_path_vision(a, fam: dict, gpu: dict, hip: bool) -> str:
    """The image encoder for an own-path family (Qwen3.6-35B-A3B): "gpu", "cpu" or "none".  Pictures are on by default (owner rule
    2026-10-18) and the reader follows strata_control/vision_reader.py's one rule on every make of card: the CPU encoder under 24 GB of
    VRAM (the GPU one takes ~1.4 GB the experts would use), the GPU encoder from 24 GB.  --vision yes|no|gpu|cpu, else a question whose
    default answer is yes; what is asked for with --vision gpu / cpu is kept, with a note (recommend, never force)."""
    asked = a.vision
    vram = float(gpu.get("vram_gb", 0.0))
    if asked in ("no", "none"):
        return "none"
    if asked is None:
        from strata_control import vision_reader as VR
        mode = VR.default_mode(vram)
        say()
        say(f"  Images: {fam['title']} can also read pictures (screenshots, photos, scanned pages). This adds a "
            f"{fam['sha256'][fam['mmproj']][0] / 1e9:.1f} GB download" + (
                " and the image encoder runs on the GPU (about 1-1.4 GB of VRAM kept free, a fraction of a second per picture)."
                if mode == "gpu" else
                "; the encoder runs on the CPU (a few seconds per picture), because the GPU encoder would take ~1.4 GB "
                "of VRAM from the experts on this card (--vision gpu to use it anyway)."))
        if ask("Do you want images?", ["y", "n"], "y", a.yes) != "y":
            return "none"
        asked = "yes"
    if hip:
        return hip_vision(asked, vram)               # settled against the engine's readers in hip_reader() (GPU reader, else CPU)
    if asked == "yes":
        asked = "gpu" if (vram <= 0 or vram >= IMAGES_GPU_ENCODER_MIN_VRAM_GB) else "cpu"
    if asked == "gpu" and 0 < vram < IMAGES_GPU_ENCODER_MIN_VRAM_GB:
        warn(f"the GPU image encoder takes ~1.4 GB of VRAM, which on a {vram:.0f} GB card is room the experts would use: "
             "--vision cpu reads pictures on the CPU instead (a few seconds each); kept as you chose")
    return asked


def own_path_parallel_slot_gb(fam: dict, ctx: int) -> float:
    """The VRAM one batch slot's session takes: its KV cache (fp32 K and V of the attention layers and the MTP block) and
    the recurrent state of the GDN layers."""
    return ctx * fam["kv_bytes_per_token"] / 1e9 + fam["state_gb"]


PARALLEL_SMALL_SHARE = 0.35     # a card that does not hold every expert: the slots' sessions may take this share of the room the
                                # experts would have (VRAM less ~4.5 GB for the dense weights and buffers).  Measured (Q4_K_XL):
                                # RTX 3060 12 GB, 8K context, 4 slots (23% of that room): a request alone -4%; A4000 / P100
                                # 16 GB, 32K, 2 slots (27%): -11% / -9%, 4 slots (54%): -27% and no gain together.


def own_path_parallel_fits_all(fam: dict, model: str, vram_gb: float, ctx: int, n: int) -> bool:
    """True where every routed expert still sits in VRAM beside n slots' sessions (one more session than the slots: the
    solo path's own)."""
    size = fam["sha256"][fam["file"].format(q=model)][0] / 1e9
    base = size * 0.93 + 4.5       # the routed experts + the dense weights and buffers (the file's 7% beyond them)
    return base + (n + 1) * own_path_parallel_slot_gb(fam, ctx) <= vram_gb


def own_path_parallel_recommend(fam: dict, model: str, vram_gb: float, ctx: int) -> int:
    """How many requests at once to recommend: 0 = none (the card cannot hold even two slots).  Where every routed expert
    still fits in VRAM with the slots' sessions, up to 4 (a batch reads each expert once for all its rows).  On a card
    that does not hold every expert even for one session (the hybrid mode already) it is recommended too, as an offer with
    its price said: measured on
    an RTX 3060 12 GB, 4 requests together run ~1.5x as fast as one, a request alone ~5% slower, because the slots take
    VRAM from the expert cache; up to the count whose sessions leave PARALLEL_SMALL_ROOM_GB of the card."""
    slot = own_path_parallel_slot_gb(fam, ctx)
    best = max([n for n in (2, 3, 4) if own_path_parallel_fits_all(fam, model, vram_gb, ctx, n)], default=0)
    if best:
        return best
    if own_path_parallel_fits_all(fam, model, vram_gb, ctx, 0):
        return 0       # a card that holds every expert for one session only: more sessions would push experts to the CPU
                       # (measured on a 32 GB R9700 at 128K: 4 slots left 15 of 256 experts per layer, decode 81 tok/s
                       # against ~180): not offered there, a number asked for is kept
    room = max(vram_gb - 4.5, 0.0)
    return max([n for n in (2, 3, 4) if n * slot <= PARALLEL_SMALL_SHARE * room], default=0)


def own_path_parallel_note(a_parallel, fam: dict, model: str, vram_gb: float, ctx: int) -> list:
    """What setup says about "parallel" for an own-path family: the recommendation, or how the asked count compares with
    it (kept as asked: recommend, never force)."""
    rec = own_path_parallel_recommend(fam, model, vram_gb, ctx)
    slot = own_path_parallel_slot_gb(fam, ctx)
    if a_parallel is None or a_parallel <= 1:
        if not rec:
            return ["Several requests at once (opt-in, --parallel N): left at one at a time, because this card cannot spare "
                    f"the room: each session takes ~{slot:.1f} GB of VRAM at this context, and where the card holds every "
                    "expert only for one session, more would move experts to the CPU and slow every request a lot (a "
                    "smaller context leaves room for more; docs/QWEN36.md)."]
        if own_path_parallel_fits_all(fam, model, vram_gb, ctx, rec):
            return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together instead of one after "
                    f"the other (each takes ~{slot:.1f} GB of VRAM at this context; docs/BATCHING.md)."]
        return [f"Several requests at once (opt-in): --parallel {rec} decodes up to {rec} together, so a second user or agent "
                "starts at once instead of waiting for the first. The price on this card, which does not hold every expert: "
                f"each session takes ~{slot:.1f} GB of VRAM from the expert cache, so a request alone runs about 4-11% "
                "slower, and next to one-at-a-time WITH the draft layer the requests together are only 1.0-1.15x faster "
                "(measured, Q4_K_XL: RTX 3060 12 GB at 8K context, 4 requests: 1.15x, alone -4%; A4000 / P100 16 GB at "
                "32K, 2 requests: 1.0x, alone -7 to -11%). Worth it for several users at once, not for one chat "
                "(docs/QWEN36.md)."]
    lines = [f"parallel requests: {a_parallel} at once (each takes ~{slot:.1f} GB of VRAM at this context, "
             f"{a_parallel * slot:.1f} GB in all)"]
    if a_parallel > PARALLEL_MAX:
        lines.append(f"the engine runs at most {PARALLEL_MAX} at once; it will use {PARALLEL_MAX}")
    if not rec:
        lines.append("recommended for this card: one at a time - there is no room for more sessions; kept as you chose")
    elif a_parallel > rec:
        lines.append(f"recommended for this card: {rec} - more slots leave fewer experts in VRAM, which can make every "
                     "request slower; kept as you chose")
    return lines


def install_own_path(a, family, gpu, hip, amd, cuda_tk, multi, port, ram, adopted):
    """Steps 2-7 for a family with its own engine path (OWN_PATH_FAMILIES: Qwen3.6-35B-A3B): the size, the context, the
    engine, the GGUF (pinned revision, size and SHA-256), the tokenizer from the GGUF, the config and the start
    script.  The engine places the experts itself (every one in VRAM when they fit, else the most-used ones there and
    the rest on the CPU), so there is no pack, profile or RAM budget to choose."""
    fam = OWN_PATH_FAMILIES[family]
    say()
    say(f"  {fam['title']} - {fam['by']}: {fam['about']}")
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    if multi:
        warn(f"{fam['title']} runs on one GPU so far: using GPU {gpu.get('index', 0)} ({gpu_name(gpu)}) only")
    names = list(fam["models"])
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names))
    say()
    for i, m in enumerate(names, 1):
        d = fam["models"][m]
        size = fam["sha256"][fam["file"].format(q=m)][0]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        say(f"  {i}) {m:10s} {d['about']}; download {size / 1e9:.1f} GB{fit}")
    rec = own_path_size(fam, ram)
    if not a.model:
        say(f"  Recommended for {ram:.0f} GB of RAM: {rec} (the largest that fits; any size can be chosen)")
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)],
                                     str(names.index(rec) + 1), a.yes)) - 1]
    if ram < fam["models"][model]["ram_gb"]:
        confirm_risk(f"{model} needs about {fam['models'][model]['ram_gb']} GB of RAM; this PC has {ram:.0f} GB: the "
                     "OS may page and it may not start", bool(a.model), a.yes,
                     f"{model} needs about {fam['models'][model]['ram_gb']} GB of RAM",
                     f"choose a smaller size, or --model {model} --yes to install it anyway", "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
    ok(f"size: {model}")
    tag = fam["tag"] + model
    rec_ctx = own_path_context(fam, gpu.get("vram_gb", 0.0))
    if a.context:
        ctx = a.context
    else:
        ctxs = [c for c in CONTEXTS if c <= 262144]   # the model's trained length; no rope scaling on this path yet
        say()
        say("  Context length = how much text the model can see at once. Its cache takes "
            f"{fam['kv_bytes_per_token'] * 32768 / 2**30:.1f} GB of VRAM per 32K tokens, room the experts would use:")
        for i, c in enumerate(ctxs, 1):
            say(f"  {i}) {c // 1024}K tokens" + ("   (recommended for your GPU)" if c == rec_ctx else ""))
        ctx = ctxs[int(ask("Context?", [str(i) for i in range(1, len(ctxs) + 1)], str(ctxs.index(rec_ctx) + 1),
                           a.yes)) - 1]
    if ctx > 262144:
        warn(f"{ctx // 1024}K is past {fam['title']}'s trained 262,144 positions and this engine path has no rope "
             "scaling yet: kept as you chose, but answers past 256K are untested")
    ok(f"context: {ctx} tokens")
    vision = own_path_vision(a, fam, gpu, hip)
    ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    vram_gb = float(gpu.get("vram_gb", 0.0))
    par = getattr(a, "parallel", None)
    par_said = False
    if par is None and not a.yes:                      # offered, with its price; Enter keeps one at a time (never forced)
        say()
        for line in own_path_parallel_note(None, fam, model, vram_gb, ctx):
            say("  " + line)
        par_said = True
        if own_path_parallel_recommend(fam, model, vram_gb, ctx):
            n = int(ask("How many requests at once (1 = one at a time)?", [str(i) for i in range(1, PARALLEL_MAX + 1)],
                        "1", a.yes))
            par = n if n > 1 else None
    if a.rope_scaling or a.rope_scale or a.kv or a.resident_budget_gib is not None or a.low_ram not in (None, "auto"):
        warn(f"--rope-scaling / --kv / --resident-budget-gib / --low-ram are Flash-Next settings: {fam['title']} "
             "does not use them")
    folder = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
    gguf = folder / fam["file"].format(q=model)
    size = fam["sha256"][gguf.name][0]
    part = gguf.with_name(gguf.name + ".part")
    def have(p: Path) -> bool:                         # --gguf-dir: the person's own file has no finish mark; verify_sha256 checks and writes it
        return p.exists() and (done(p) or bool(a.gguf_dir))
    left = 0 if have(gguf) else (size - (part.stat().st_size if part.is_file() else 0)) / 1e9
    if left and free_gb(folder) < left:
        fail(f"not enough free disk space in {folder}: need ~{left:.0f} GB, {free_gb(folder):.0f} GB free", ("use --data-dir on a bigger drive (strata launch --data-dir)" if CONTROL else "use --models-dir on a bigger drive"))
    if left and free_gb(folder) < left + 1:
        warn(f"little free disk space in {folder}: {free_gb(folder):.1f} GB free for about {left:.1f} GB. Setup goes on and should fit")

    step(3, "Python packages")
    pip_install(runtime_packages(),
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil, textual (the screens)")

    step(4, "the Strata engine")
    llama, eng, meta, lib_dirs, gpu, vision = engine_step(a, gpu, hip, amd, cuda_tk, vision)
    reader = None
    if hip and vision != "none":
        vision, reader = hip_reader(eng, vision, gpu)
    elif vision != "none" and not (eng / VEXE).exists():
        warn("this engine has no image encoder: images off (run setup again with --build to compile it)")
        vision = "none"
    if fam["arch"] not in engine_families(eng):
        fail(f"this engine ({meta.get('version')}) does not run {fam['title']} yet",
             "update Strata, or compile the engine with --build, and run setup again")
    ok("engine: the engine component (looked up when the model loads)" if CONTROL else f"engine: {eng / EXE}")

    step(5, f"downloading {fam['title']} {model}")
    if have(gguf):
        ok(f"{gguf.name} already downloaded")
    else:
        if a.gguf_dir:
            fail(f"{gguf} is missing", f"put {gguf.name} in {folder}, or drop --gguf-dir to download it")
        download(fam["hf"] + gguf.name, gguf)
    verify_sha256(gguf, *fam["sha256"][gguf.name], own=bool(a.gguf_dir))
    ok(f"model file: {gguf}")
    mmproj = folder / fam["mmproj"]
    if vision != "none":                               # the image encoder: Unsloth's mmproj, pinned, size + SHA-256 checked
        if have(mmproj):
            ok(f"{mmproj.name} already downloaded")
        else:
            if a.gguf_dir:
                fail(f"{mmproj} is missing", f"put {mmproj.name} in {folder}, or drop --gguf-dir to download it")
            download(fam["mmproj_hf"] + mmproj.name, mmproj, "vision encoder")
        verify_sha256(mmproj, *fam["sha256"][mmproj.name], own=bool(a.gguf_dir))
        ok(f"vision encoder: {mmproj}")

    step(6, "preparing the model for Strata")
    data, _ = data_folder_for(a)
    pack = data / "packs" / tag.lower()            # only the tokenizer: the engine reads the GGUF itself
    if not (pack / "tokenizer" / "vocab.json").exists():
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(pack)],
            env=dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py")))
    ok(f"tokenizer: {pack / 'tokenizer'}")

    step(7, "writing the start script")
    args = ["--native", str(gguf), "--max-context", str(ctx), "--prefill", "auto", "--mtp", str(fam["mtp"])]
    if a.vram_reserve_mib is not None:             # VRAM left free for other programs: the experts take that much less
        vram = gpu.get("vram_gb", 0.0)
        if vram > 0:
            args += ["--expert-vram", f"{max(vram - 3.0 - a.vram_reserve_mib / 1024, 0):.1f}"]
        ok(f"VRAM kept free for other programs: about {a.vram_reserve_mib} MiB (the experts' share is set lower)")
    cfg = {"exe": engine_exe(eng), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "arch": fam["arch"],
           "log": str(home() / f"strata-{tag.lower()}.log"), "lib_dirs": lib_dirs, "port": port}
    if cuda_tk == 12:
        cfg["cuda"] = 12
    if hip:
        cfg["backend"] = "hip"
    if gpu.get("count", 1) > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]
        cfg["gpus_asked"] = True
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg["args"] = recommend_pool_workers(cfg["args"])   # --pool-workers: this engine path's CPU expert threads too
    for line in pool_workers_note(cfg["args"]):
        warn(line)
    cfg_path = home() / f"strata-{tag.lower()}.json"
    if vision != "none":
        old_cfg = cfg_path if cfg_path.is_file() else adopted
        vt = vision_tokens(getattr(a, "vision_tokens", None), vision, old_cfg)
        cfg["args"].append("--vision")
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj), "model": str(gguf), "gpu": vision == "gpu",
                         "max_tokens": vt}          # the text model, opened vocab-only
        if reader is not None:                      # AMD: the HIP reader with the CPU reader behind it, or the CPU reader
            from strata_control import vision_reader as VR
            cfg["vision"] = VR.vision_section(reader, cfg["vision"])
        if vision == "cpu":
            cfg["vision"]["threads"] = vision_threads()
    elif getattr(a, "vision_tokens", None) is not None:
        warn("--vision-tokens: images are off, so it is not used")
    # requests at once ("parallel", opt-in): written only when given; the recommendation is said once
    if par is not None:
        if par >= 2:
            cfg["parallel"] = par
            for i, line in enumerate(own_path_parallel_note(par, fam, model, vram_gb, ctx)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None, explicit_args(a))
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")
    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if par is None and not par_said:                   # the opt-in, said once (nothing changes)
        for line in own_path_parallel_note(None, fam, model, vram_gb, ctx):
            say("  " + line)
    if a.no_start:
        return 0
    return start(cfg_path, port)


# DeepSeek's engine is its own package (feed member "dsv41", strata_control/dsv41_engine.py), downloaded when DeepSeek is chosen and unpacked in
# <data>/engine-dsv41/<version>-<platform>-<backend>/ beside the engine: the Qwen engine's files are never touched.  Fetched, verified and
# resumed like an engine update (the signed feed; --feed / STRATA_UPDATE_FEED name another one, as for `strata update`).
DSV41_ENGINE_DIR = "engine-dsv41"


def dsv41_engine_dir(eng: Path) -> Path:
    """Beside the engine folder; for the app / Terminal (the engine is a component folder Strata does not write into): in Strata's own folder."""
    return (home() if CONTROL else Path(eng).parent) / DSV41_ENGINE_DIR


def local_feed(feed: str | None = None) -> bool:
    """Is the update feed this run reads a local one (a folder, a file or a loopback address, as a test or an offline copy uses)?  Then
    nothing is downloaded from GitHub and its being slow or down must not stop the run.  `feed` is --feed, else STRATA_UPDATE_FEED."""
    src = (feed or os.environ.get("STRATA_UPDATE_FEED") or "").strip()
    if not src:
        return False
    if not re.match(r"^[a-z][a-z0-9+.-]*://", src, re.I) or src.lower().startswith("file://"):
        return True                                     # a path (a Windows C:\... one too)
    host = (urllib.parse.urlparse(src).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def github_reachable(timeout: float = 10.0) -> bool:
    """Does GitHub answer within `timeout` seconds (any HTTP answer counts; no answer, a refused or timed-out connection does not)?  One quick check
    instead of minutes of retries when the connection is down (a SYN that is never answered)."""
    try:
        _urlopen(urllib.request.Request("https://github.com", method="HEAD", headers={"User-Agent": "strata-setup"}), timeout=timeout).close()
        return True
    except urllib.error.HTTPError:
        return True
    except (OSError, urllib.error.URLError):
        return False


def dsv41_engine_pending(eng: Path) -> bool:
    """A DeepSeek engine package file the person put in engine-dsv41/ is there to be installed (no download needed)."""
    from strata_control import dsv41_engine
    return bool(dsv41_engine.pending(dsv41_engine_dir(eng)))


def dsv41_manual_steps(eng: Path) -> str:
    return (f"If you downloaded the DeepSeek engine package from the release page (the file strata-dsv41-...-{'windows' if WIN else 'linux'}-....tar.gz), "
            f"run the same command again with --dsv41-package PATH-OF-THE-FILE (strata launch accepts it too), or put the file in {dsv41_engine_dir(eng)} "
            "and run it again. Strata checks it before it is used.")


def get_dsv41_engine(eng: Path, hip: bool, package: str | None = None, feed: str | None = None, archs=None) -> Path | None:
    """The DeepSeek engine package next to the engine `eng`: one the person gave (--dsv41-package) or put in engine-dsv41/ by hand, else downloaded from
    the signed feed `feed` (--feed; else STRATA_UPDATE_FEED, a remembered https feed, or the release's).  Checked first either way; its
    strata-dsv41, or None when it cannot be had."""
    from strata_control import dsv41_engine
    root = dsv41_engine_dir(eng)
    if package:
        exe = dsv41_engine.install_package(root, package, emit=lambda m: say("  " + m))
        if exe is None:
            fail("the DeepSeek engine package you gave could not be used", "see the line above; nothing else was changed")
        return exe
    for f in dsv41_engine.pending(root):
        say(f"  Found {f.name} in {root}: checking it ...")
        exe = dsv41_engine.install_package(root, f, emit=lambda m: say("  " + m))
        if exe is not None:
            return exe
    acquisition.require("the DeepSeek engine")
    from strata_cli import update_cmd
    source = update_cmd._feed_source(argparse.Namespace(feed=feed), root.parent)
    say("  Downloading the DeepSeek engine (its own package, separate from the Strata engine) ...")
    hw = None
    if hip:
        from strata_control import updater
        hw = {**updater.detect_hardware(), "backend": "hip", "backend_source": "requested"}
        if archs:                                          # the package is chosen for the cards this model runs on, not for every card in the PC
            hw["gfx"] = sorted({str(x) for x in archs})
    exe = dsv41_engine.install(root, source, hardware=hw, emit=lambda m: say("  " + m))
    if exe is None:
        warn("no DeepSeek engine package could be fetched (see above)")
    return exe


def upgrade_dsv41_engine(eng: Path, exe: Path, feed: str | None = None) -> Path:
    """The installed DeepSeek engine package is older than the signed feed's (--feed, else STRATA_UPDATE_FEED, a remembered https feed,
    or the release's): updated next to it, switched after its test start, the old one removed (strata_control.dsv41_engine.upgrade).
    Silent when the feed cannot be reached; a failure is one plain line and `exe` keeps running."""
    from strata_control import dsv41_engine
    try:
        from strata_cli import update_cmd
        root = dsv41_engine_dir(eng)
        source = update_cmd._feed_source(argparse.Namespace(feed=feed), root.parent)
        if not dsv41_engine.source_reachable(source):
            return exe
        new = dsv41_engine.upgrade(root, source, installed=exe, emit=lambda m: say("  " + m))
    except Exception:                                      # noqa: BLE001 - never stop a setup for this
        return exe
    return new or exe


def dsv41_exe(eng: Path) -> Path | None:
    """The DeepSeek-V4.1 engine (strata-dsv41): STRATA_DSV41_EXE (a build of your own), else its own package folder beside the engine (engine-dsv41/),
    else beside the engine in `eng` (an older all-in-one package); None when there is none."""
    own = (os.environ.get("STRATA_DSV41_EXE") or "").strip()
    if own:
        return Path(own) if Path(own).is_file() else None
    name = "strata-dsv41.exe" if WIN else "strata-dsv41"
    from strata_control import dsv41_engine
    exe = dsv41_engine.find(dsv41_engine_dir(eng)) or (Path(eng) / name)
    if exe.is_file() and not WIN and not os.access(exe, os.X_OK):      # a package copied without its modes (scp / tar from Windows): fix it, do not stop
        try:
            exe.chmod(0o755)
        except OSError:
            pass
    return exe if exe.is_file() else None


def dsv41_folder(a, fam: dict, tag: str, model: str = "UD-Q8_K_XL") -> Path:
    """Where the model's shards are: --gguf-dir (a folder with the shards, or the repository's folder with the set's own folder inside:
    UD-Q8_K_XL, 2.00bpw, 2.25bpw), else the models folder's own."""
    if a.gguf_dir:
        folder = Path(a.gguf_dir)
        d = fam["models"][model]
        names = [n for s in d["shards"] for n in (s, DSV41_LEGACY_NAMES.get(s)) if n] + list(d.get("shards6") or [])
        if any((folder / n).exists() for n in names):
            return folder
        for sub in (d.get("subdir"), d.get("legacy_subdir"), model):          # the set's own folder inside (the first release used 2.00bpw/, 2.25bpw/)
            if sub and (folder / sub).is_dir():
                return folder / sub
        return folder
    return Path(a.models_dir) / tag


def dsv41_pick(folder: Path, name: str) -> Path:
    """`name` in `folder`, or the file it was called in the first release (DSV41_LEGACY_NAMES; the same weights) when only that one is there;
    else `name` (where it will be downloaded to)."""
    if (folder / name).exists():
        return folder / name
    old = DSV41_LEGACY_NAMES.get(name)
    if old and (folder / old).exists():
        return folder / old
    return folder / name


def dsv41_layout_of(md: dict, present) -> str:
    """Which layout of a set the files are: "four" (the pinned files, and the first release's names) or "six" (ISTA-DASLab's 2026-10-27 re-split,
    DSV41_ISTA6_FILES).  `present(name)` says whether a file is there.  Six only when its files are there and the four-shard ones are not (or are
    not whole while the six are): a folder holding both whole sets is used as the pinned four-shard one, as before."""
    four = [bool(present(n) or (DSV41_LEGACY_NAMES.get(n) and present(DSV41_LEGACY_NAMES[n]))) for n in md["shards"]]
    six = [bool(present(n)) for n in md.get("shards6") or []]
    return "six" if six and any(six) and (not any(four) or (all(six) and not all(four))) else "four"


def dsv41_parts(md: dict, present) -> tuple:
    """(layout, the names of its files, the ones that are not there) for a set's files seen through `present(name)`."""
    layout = dsv41_layout_of(md, present)
    names = list(md["shards6"] if layout == "six" else md["shards"])
    missing = [n for n in names if not present(n) and not (layout == "four" and DSV41_LEGACY_NAMES.get(n) and present(DSV41_LEGACY_NAMES[n]))]
    return layout, names, missing


def dsv41_pack_layout(pack: Path) -> str | None:
    """"four" or "six" when the pack names the files of that layout (tensors.txt's `file N <name>` lines), else None.  The pack folder is the same for
    both layouts of a set, and a pack points into the files it was made from: one made from the other layout's files must be built again."""
    try:
        with (Path(pack) / "tensors.txt").open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("file "):
                    break
                if "-of-00006" in line:
                    return "six"
                if "-of-00004" in line:
                    return "four"
    except OSError:
        pass
    return None


def dsv41_shared_dir(a, fam: dict) -> Path:
    """Where the files the official sets share (the Engram, the DSpark drafter, the metadata shard) are downloaded to: one folder, so
    installing the second set does not download the 111 GB Engram again."""
    return Path(a.models_dir) / (fam["tag"] + "shared")


def dsv41_find_shared(a, name: str, folder: Path, shared: Path, alt: tuple = ()) -> Path:
    """The file `name` (an Engram, the drafter, ...) the person already has: in the shared folder, beside the shards, or (--gguf-dir) in the
    folder above them or one folder level down from either; else where it will be downloaded to (the shared folder).  The size must be the
    pinned one, so a half-copied file is not taken for the file."""
    names = [n for n in (name, DSV41_LEGACY_NAMES.get(name), *alt) if n]       # today's name, the first release's (the same weights), and `alt` (an Unsloth set's own shard 5: the same Engram data)
    roots = [shared, folder]
    if a.gguf_dir:
        base = Path(a.gguf_dir)
        roots += [base, base.parent, base.parent.parent]          # (strata launch hands over the set's own folder: 2.00bpw/)
    seen = []
    for r in roots:
        if r in seen:
            continue
        seen.append(r)
        try:
            kids = [r] + ([k for k in r.iterdir() if k.is_dir()] if (a.gguf_dir and r != shared) else [])
        except OSError:
            continue
        for k in kids:
            for n in names:
                f = k / n
                try:
                    if f.is_file() and f.stat().st_size == DSV41_ALL_FILES[n][0]:
                        return f
                except OSError:
                    pass
    return shared / name


def dsv41_pack_has_tower(pack: Path) -> bool:
    """Does the pack carry the vision tower (pack_gguf.py writes vision.* and image_token_id into ds4.txt when it found the mmproj)?"""
    try:
        txt = (Path(pack) / "ds4.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "\nimage_token_id " in "\n" + txt and "vision.num_hidden_layers" in txt


def dsv41_pack_cmd(folder: Path, shards: list, engram: Path, meta: Path, dspark: Path, pack: Path, mmproj: Path | None = None) -> list:
    """The pack recipe for an official IST-DAS set - this and nothing else: another one (--skip-tensors / --dense-override + engram_fold)
    packs without an error and gives garbage text (P2003, 2026-10-23).  The drafter's noise token (128799) and markov rank (256) are stated:
    the IST-DAS GGUFs do not carry them, and a pack with noise id 0 drafts deep less often (decode ~10% slower with DSpark).  --shards names the four files, so an Engram or another
    set in the same folder is not read as part of the model."""
    return [sys.executable, str(ROOT / "tools" / "dsv41" / "pack_gguf.py"), "--model", str(folder), "--shards", *[Path(n).name for n in shards],
            "--engram", str(engram), "--engram-meta", str(meta), "--dspark-gguf", str(dspark),
            "--dspark-noise-token", "128799", "--dspark-markov-rank", "256", *(["--mmproj", str(mmproj)] if mmproj else []), "--out", str(pack)]


def dsv41_prefill_chunk_args(cards, ram_gb) -> list:
    """A bigger prompt chunk on a unified-memory APU (Ryzen AI Max, gfx1151), where the engine's 1,024-token rule was calibrated on
    cards that hold most of the experts.  Measured on the 128 GB Strix Halo with the 2.00 bpw set (2026-10-25, engine 0.1.51):
    4K prompt 78.7 -> 87.2 tok/s, 8K 108.9 -> 125.1, 32K 155.2 -> 176.9; decode unchanged (16.53 -> 16.59 tok/s), because the
    chunk buffers take ~5 GB from the GPU expert tier and the RAM store's hit rate rises to compensate.  The prompt chunk's
    summation order changes, not the model: the repo's chunk-size KL check (1024 0.1705 / 2048 0.1692 / 4096 0.1688 / 8192 0.1695)
    shows no quality move.  Only where the shared pool has room; STRATA_DSV41_NO_BIG_CHUNK=1 leaves the engine's own choice.

    A 16,384-token chunk on the 128 GB box (2026-10-26, engine 0.1.51, the same set, 1-2 runs a point): 8K 119.8 -> 119.4 (nothing,
    the set is read once a chunk either way), 16K 152.2 -> 158.6 (+4.2%), 32K 174.0 -> 179.7 (+3.3%), decode 17.21 -> 17.27
    (2+2 interleaved, unchanged) although the bigger chunk moves ~30% more experts to the CPU lane.  So a pool that holds this
    box's 128 GB takes 16,384; a smaller unified PC keeps 8,192 (not measured below this box)."""
    if not cards or not any(c.get("uma") for c in cards):
        return []
    pool = float(ram_gb or 0) + max([float(c.get('dedicated_gb') or 0.0) for c in cards] or [0.0])
    if pool < 64.0:                       # the RAM Windows sees plus the GPU's BIOS carve-out
        return []
    if os.environ.get("STRATA_DSV41_NO_BIG_CHUNK") == "1":
        return []
    return ["--prefill-chunk", "16384" if pool >= 96.0 else "8192"]


def dsv41_config_env(cards) -> dict:
    """Engine env settings for the DeepSeek engine on a unified-memory APU (Ryzen AI Max, gfx1151).

    DS4_ASYNC_T=4: the engine's windowed decode uses the mailbox ("async") path for windows up to this size, 8 by default.  On the
    128 GB Strix Halo the async path is slower than the synchronous one for the 6-row verify window: measured 2026-10-25 (2.00
    bpw, engine 0.1.51, 256-token code set, greedy): 15.87-16.33 tok/s with the default against 20.14-21.28 with 4 (+31%); the
    agent set with the same accepted tokens a round: 15.93 against 18.48 (+16%).  The same bits come out - only the scheduling
    changes - and the drafter's own single-token windows (T=1) stay on the async path.  A discrete card keeps the engine's
    default; STRATA_DSV41_ASYNC_T=n overrides the value, STRATA_DSV41_ASYNC_T=8 restores the default."""
    if not cards or not any(c.get("uma") for c in cards):
        return {}
    value = os.environ.get("STRATA_DSV41_ASYNC_T") or "4"
    return {} if value == "8" else {"DS4_ASYNC_T": value}


def dsv41_engine_env(lib_dirs) -> dict:
    env = dict(os.environ)
    dirs = [str(d) for d in lib_dirs or [] if d]
    if dirs:
        env["PATH" if WIN else "LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("PATH" if WIN else "LD_LIBRARY_PATH", "")])
    return env


def dsv41_pack_info(exe: Path, pack: Path, lib_dirs) -> tuple[bool, str]:
    """`strata-dsv41 --pack-info --pack P`: (True, text) when every file of the pack is there (no MISSING line), else (False, text)."""
    try:
        r = subprocess.run([str(exe), "--pack-info", "--pack", str(pack)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                           env=dsv41_engine_env(lib_dirs))
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"the engine could not read the pack: {e}"
    text = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0 and "MISSING" not in text and "all present" in text, text


def dsv41_smoke(exe: Path, pack: Path, lib_dirs, gpus=None, timeout: int = 5400) -> tuple[bool, str]:
    """16 tokens of a fixed prompt through the engine (tools/dsv41/smoke.py), the text checked: (True, text) or (False, why).  Takes a
    few minutes: the engine reads what it needs from the SSD the first time."""
    cmd = [sys.executable, str(ROOT / "tools" / "dsv41" / "smoke.py"), "--exe", str(exe), "--pack", str(pack)]
    if gpus:
        cmd += ["--gpus", ",".join(str(g) for g in gpus)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           env=dsv41_engine_env(lib_dirs))
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"the test run did not finish: {e}"
    text = ((r.stdout or "") + (r.stderr or "")).strip()
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"            # the reply is printed by say()/fail(): a Windows console code page cannot show DeepSeek's special tokens
    return r.returncode == 0, text[-1500:].encode(enc, "replace").decode(enc, "replace")


def earlier_card_kept(prev: Path) -> bool:
    """A new copy of Strata set up like an earlier install carries over the card that install was set to.  A card setup picked itself on a
    DeepSeek model on AMD (strata_control/gpu_pick.py) is not a choice to carry over; every other config keeps its card exactly as before
    (no AMD listing is read for them, and nothing here can stop a setup)."""
    try:
        earlier = json.loads(Path(prev).read_text(encoding="utf-8-sig"))
        from strata_control import gpu_pick
        if WIN or not gpu_pick.is_deepseek_amd(earlier):
            return True
        return gpu_pick.chosen_by_person(earlier, amd_usable_cards(amd_gpus()))
    except Exception:                                      # noqa: BLE001 - an unreadable file or listing: keep the card, as before
        return True


def cards_named(a, hip: bool = False) -> bool:
    """Did the person name the cards (--gpu / --gpus)?  On AMD setup's own pick is written into a.gpu further on (other steps rely on it), so
    the AMD branch records the answer first (a._cards_named); NVIDIA keeps its way (choose_gpus sets a.gpu to the cards in use)."""
    named = getattr(a, "_cards_named", None)
    return bool(named) if (hip and named is not None) else bool(a.gpu is not None or a.gpus)


def amd_usable_cards(amd) -> list:
    """The AMD cards that count for a model (gpu_fit.usable_cards: no integrated chip beside a card with memory of its own, none under 8 GB)."""
    from strata_control import gpu_fit
    return gpu_fit.usable_cards([{**g, "vendor": "amd", "usable": amd_problem(g) is None} for g in amd])[0]


def dsv41_covered_archs(eng):
    """The AMD architectures the installed DeepSeek engine package holds code for (its manifest); None when it names none (no package yet, or one
    without a list): the caller then uses the release's own rule (dsv41_fit.run_cards) and says nothing about a card it cannot know."""
    try:
        from strata_control import engine_component as ec
        exe = dsv41_exe(Path(eng))
        if exe is not None and (exe.parent / ec.MANIFEST).is_file():
            archs = ec.read_manifest(exe.parent).hip_archs
            if archs:
                return tuple(archs)
    except Exception:                                      # noqa: BLE001 - no list: None
        pass
    return None


def dsv41_auto_cards(amd, eng) -> list:
    """DeepSeek on AMD with no card named: every card that counts and the engine package can run (one card on Windows: several AMD cards sharing
    a model is Linux-only).  The same rule as `strata launch --plan` (strata_control/dsv41_fit.run_cards).  What is left out is said."""
    from strata_control import dsv41_fit
    # only real AMD cards (gfx...): sycl/setup_intel.py runs this same branch with its Intel cards in `amd` (arch "xe" / "i915"), and the DeepSeek
    # engine has no Intel package: those keep the one card setup picked
    pool = [{**g, "vendor": "amd", "usable": amd_problem(g) is None} for g in amd
            if amd_problem(g) is None and str(g.get("arch") or "").startswith("gfx")]
    keep, left = dsv41_fit.run_cards(pool, "amd", windows=WIN, covered=dsv41_covered_archs(eng))
    byid = {g["index"]: g for g in amd}
    for c in left:
        if c.get("why_ignored") and not str(c["why_ignored"]).startswith("it shares the system memory"):
            say(f"  Not used for DeepSeek: GPU {c.get('index')} ({c.get('name')}): {c['why_ignored']}.")
    return [byid[c["index"]] for c in keep]


def dsv41_cards(gpu: dict, all_cards: list | None, asked: bool) -> list:
    """The cards the model runs on: the ones asked for (--gpu / --gpus), else every card setup can use (recommend: the model is
    larger than any one card, so the more graphics memory the more of its experts stay off the CPU)."""
    if asked or not all_cards:
        return list(gpu.get("cards") or [gpu])
    return list(all_cards)


def install_dsv41(a, family, gpu, hip, amd, cuda_tk, multi, port, ram, adopted, all_cards=None):
    """Steps 2-7 for DeepSeek-V4.1-Flash (MULTI_FILE_FAMILIES): where it fits (strata_control/dsv41_fit.py, said plainly: a 508 GB
    download and a model larger than most computers' memory are warnings, never a refusal), the engine (strata-dsv41), the nine
    pinned shards (or your own, --gguf-dir), the engine's pack folder, the tokenizer and chat template from the GGUF, the config.  The
    engine places the experts itself (as many as the cards hold in graphics memory, the rest computed on the CPU from the files)."""
    from strata_control import dsv41_fit
    mac = sys.platform == "darwin"                          # an Apple Silicon Mac (0.2.7.1): the Metal build of the DeepSeek engine, one pool of memory
    fam = MULTI_FILE_FAMILIES[family]
    say()
    say(f"  {fam['title']} - {fam['by']}: {fam['about']}")
    ok(f"model: {fam['title']}")
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    names = list(fam["models"])
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose: " + ", ".join(names))
    if a.model:
        model = a.model
    else:
        try:
            room = free_gb(Path(a.models_dir) if Path(a.models_dir).exists() else Path(a.models_dir).parent)
        except Exception:                                    # noqa: BLE001 - the disk is only a hint here
            room = None
        rec = dsv41_fit.recommend(ram, room)                 # the best version this computer's memory and disk suit; warn-never-refuse
        names = [m for m in names if not dsv41_fit.not_here(m)]   # the Unsloth 1-bit sets: no Metal kernel yet on a Mac, untested on a DGX Spark
        for i, m in enumerate(names, 1):
            say(f"  {i}) {fam['models'][m].get('label', m)}: {fam['models'][m]['about']}" + ("   (recommended)" if m == rec else ""))
        model = names[int(ask("Which version?", [str(i) for i in range(1, len(names) + 1)], str(names.index(rec) + 1), a.yes)) - 1]
    if dsv41_fit.not_here(model):       # Mac: IQ1_S experts have no Metal kernel yet; DGX Spark: never run (strata_control/dsv41_fit.not_here); said before anything is fetched or packed
        fail(dsv41_fit.not_here(model), "choose --model IST-DAS-2.00bpw or --model IST-DAS-2.25bpw (their files work with --gguf-dir too); nothing has been downloaded")
    ok(f"size: {model}")
    md = fam["models"][model]
    say("  Its GGUF files: " + ("IST-DAS / ISTA-DASLab (huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), with the DSpark drafter and a small metadata file from Unsloth"
                              if md.get("recipe") == "ista" else
                              "Unsloth (huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF); the Engram table is the same data as IST-DAS's, so a copy of it already here is used"
                              if md.get("recipe") == "udlow" else "Unsloth (huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)"))
    udlow = md.get("recipe") == "udlow"          # Unsloth's UD-IQ1_S / UD-IQ1_M: packed and checked like the IST-DAS sets; shard 5 is the Engram
    ista = md.get("recipe") in ("ista", "udlow")
    tag = fam["tag"] + model
    named = cards_named(a, hip)
    cards = dsv41_cards(gpu, all_cards, named)
    ctx = a.context or DSV41_CONTEXT
    if dsv41_context.problem(ctx):                 # 8K to 1M: the model reads 1,048,576 positions (Project Maya #442); a size it cannot address stops here, before any download
        fail(dsv41_context.problem(ctx), "--context takes 8192 up to 1048576 (8K, 32K, 64K, 128K, 256K, 512K, 1M); the default is 8192")
    if dsv41_context.note(ctx):
        warn(f"context {dsv41_context.label(ctx)}: {dsv41_context.note(ctx)}")
    folder = dsv41_folder(a, fam, tag, model)
    layout = dsv41_layout_of(md, lambda n: (folder / n).exists()) if (ista and a.gguf_dir) else "four"
    shards = [dsv41_pick(folder, n) for n in (md["shards6"] if layout == "six" else md["shards"])]
    if layout == "six":
        say(f"  {folder} holds ISTA-DASLab's six-shard files of 2026-10-27 ({len(shards)} files and the Engram as the sixth): they are used where they are.")
    shared_dir = dsv41_shared_dir(a, fam)
    # Pictures are on by default (owner rule): the engine reads them with its own vision tower, which the pack carries when the tower's weights
    # (mmproj-BF16.gguf, a shared file) were there when it was made.  --vision no leaves the file out and the pack text-only.
    ds_vision = "off" if a.vision in ("no", "off") or (mac and not DSV41_MAC_VISION) else "on"
    if mac and not DSV41_MAC_VISION and a.vision not in ("no", "off"):
        warn("pictures are off for DeepSeek on a Mac: its picture reader has not been run on the Metal engine yet, so it reads text only")
    shared_names = list(md.get("shared", [])) + ([DSV41_MMPROJ] if ds_vision == "on" else [])
    shared = [dsv41_find_shared(a, n, folder, shared_dir, (md["engram_shard"],) if (udlow and n == DSV41_ISTA_ENGRAM) else ()) for n in shared_names]
    mmproj_file = shared[len(md.get("shared", []))] if ds_vision == "on" else None
    own_files = {f: bool(a.gguf_dir) or f.parent != shared_dir for f in shared}         # a file found outside setup's own folder is the person's

    def partial(f: Path) -> int:
        p = f.with_name(f.name + ".part")
        return p.stat().st_size if p.is_file() else 0

    left = sum(0 if f.exists() and (done(f) or a.gguf_dir or own_files.get(f) or f.name in DSV41_LEGACY_FILES) else fam["sha256"][f.name][0] - partial(f)
               for f in [*shards, *shared]) / 1e9
    probe = folder if folder.exists() else (folder.parent if folder.parent.exists() else Path(a.models_dir))
    free = free_gb(probe)
    plan = dsv41_fit.plan([{**c, "vram_gb": float(c.get("vram_gb") or 0)} for c in cards], ram, free if left else None, ctx, model=model,
                         named=True)               # `cards` are decided above (dsv41_auto_cards / the person's): the plan describes them
    say()
    for line in plan["lines"]:
        say("  " + line)
    if left:
        say(f"  Download: {left:.0f} GB to fetch (checked against pinned sizes and SHA-256; it resumes if it stops). At 50 MB/s that "
            f"is {left * 1e9 / 50e6 / 3600:.1f} hours. Files you already have: --gguf-dir DIR.")
    else:
        say("  Nothing to download: the files are already here.")
    unchecked = [f for f in [*shards, *shared] if f.exists() and not any(
        f"sha256 {x}" in (f.with_name(f.name + ".done").read_text(encoding="utf-8", errors="replace") if done(f) else "")
        for x in (fam['sha256'][f.name][1], *DSV41_SHA_ALSO.get(f.name, ())))]
    if unchecked:
        gb = sum(fam["sha256"][f.name][0] for f in unchecked) / 1e9
        say(f"  The files already here ({gb:.0f} GB) are checked against their SHA-256 once, now: about {max(1, round(gb / 55))} minutes on a fast "
            "drive (4-5 minutes for the 257 GB of 2.00), longer on a slow one. The next runs skip it.")
    for w in plan["warnings"]:
        if "free disk space" in w:
            continue                                                      # the disk is a hard limit, said below
        warn(w)
    if plan["verdict"] == "spills" and mac:
        confirm_risk(f"this Mac's memory holds about {plan['vram_experts_gb']:.0f} GB of {model}'s {plan['experts_gb']:.0f} GB of experts and the "
                     f"other {plan['streamed_gb']:.0f} GB are read from the SSD for every word: much slower than on a Mac with 128 GB (a Mac this "
                     "small was not measured)", bool(a.model), a.yes,
                     f"{model} is slow on a Mac with {ram:.0f} GB", f"a Mac with 64 GB or more suits it; --model {model} --yes "
                     "installs it anyway", "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GB of memory, as you chose")
    elif plan["verdict"] == "spills":
        confirm_risk(f"{model} runs best with about {plan['ram_needed_gb']:.0f} GB of RAM beside this PC's graphics memory and this PC has "
                     f"{ram:.0f} GB: most of its experts are read from the SSD while it answers (slow)", bool(a.model), a.yes,
                     f"{model} needs about {plan['ram_needed_gb']:.0f} GB of RAM", f"more RAM or cards fix it; --model {model} --yes "
                     "installs it anyway", "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
    if left and free < left:
        fail(f"not enough free disk space in {folder}: need ~{left:.0f} GB, {free:.0f} GB free",
             "use --models-dir on a bigger drive (an NVMe drive is best: the n-gram table is read from it while the model "
             "answers), or --gguf-dir for files you already have")
    if left and free < left + dsv41_fit.DISK_HEADROOM_GB:
        warn(f"little free disk space in {folder}: {free:.1f} GB free for about {left:.1f} GB. Setup goes on and should fit")
    if ctx > 65536:
        warn(f"{ctx // 1024}K tokens of context: the engine was checked at 4K and 8K; longer is kept as you chose, but untested")
    # Pictures are on by default (owner rule): the engine reads them with its own vision tower (319 MiB, loaded with the model, no separate
    # reader process).  --vision no turns them off (--vision off for the engine); gpu / cpu name a Flash-Next reader and mean "on" here.
    if a.vision in ("gpu", "cpu"):
        warn(f"{fam['title']} reads pictures with the engine's own tower, so --vision {a.vision} is taken as yes: pictures on")
    if a.rope_scaling or a.rope_scale or a.kv or a.resident_budget_gib is not None or a.low_ram not in (None, "auto"):
        warn(f"--rope-scaling / --kv / --resident-budget-gib / --low-ram are Flash-Next settings: {fam['title']} does not use them")

    step(3, "Python packages")
    pip_install(runtime_packages(),
                "numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil, textual (the screens)")

    # DeepSeek runs on its own engine package: the Strata (Qwen) engine, its llama.cpp source and its CUDA libraries are not needed for it, so none of
    # them is downloaded or test-started here (they were ~700 MB, and a stop on every one of them when GitHub could not be reached).
    step(4, "the DeepSeek engine")
    eng, meta, lib_dirs = ROOT / "engine", {"version": None}, []
    exe = dsv41_exe(eng)
    if (exe is None and not getattr(a, "dsv41_package", None) and not dsv41_engine_pending(eng) and not local_feed(getattr(a, "feed", None))
        and not github_reachable()):
        fail("Strata can't reach GitHub to download the DeepSeek engine.",
             "Check the internet connection and run the same command again. " + dsv41_manual_steps(eng))
    if exe is None and not getattr(a, "build", False):
        get_dsv41_engine(eng, hip, getattr(a, "dsv41_package", None), getattr(a, "feed", None),
                         archs=[c.get("arch") for c in cards if c.get("arch")] if hip else None)
        exe = dsv41_exe(eng)
    if (exe is not None and not getattr(a, "dsv41_package", None) and not getattr(a, "build", False)
            and not (os.environ.get("STRATA_DSV41_EXE") or "").strip() and exe.parent.parent == dsv41_engine_dir(eng)):
        exe = upgrade_dsv41_engine(eng, exe, getattr(a, "feed", None))   # an installed package older than the signed feed's: updated next to it, switched after it starts
    if exe is not None and exe.parent != Path(eng):
        from strata_control import dsv41_engine
        lib_dirs = [*lib_dirs, *dsv41_engine.lib_dirs(exe)]   # its own package, and the rocm/bin an AMD package keeps the ROCm DLLs in
    if hip and exe is not None and named:                  # cards the person named that this package holds no code for: said, never refused
        covered = dsv41_covered_archs(eng)
        for c in cards if covered else []:
            if c.get("arch") and c["arch"] not in covered:
                warn(f"GPU {c['index']} ({c['name']}, {c['arch']}): this DeepSeek engine package holds code for {', '.join(covered)} only, so it may "
                     "not start on that card. Leave it out (--gpus ...) to be sure.")
    if exe is None:
        if os.environ.get("STRATA_VERBOSE") == "1" or "--verbose" in sys.argv:
            say("  Details for developers: build it with cmake -DSTRATA_BUILD_DSV41=ON -DCMAKE_CUDA_ARCHITECTURES=<your card: 86 RTX 30, 89 RTX 40, 120 RTX 50> "
                "and point STRATA_DSV41_EXE at it (docs/DEEPSEEK41.md, \"Building the engine for your card\").")
        fail("DeepSeek's engine is not available for this computer yet",
             ("It runs on an Apple Silicon Mac with the Metal DeepSeek engine package from the Strata update feed. " if mac else
              "Supported graphics cards: NVIDIA RTX 30, 40 and 50 series, the Radeon AI PRO R9700, and a Ryzen AI Max (Strix Halo) on "
              "Windows with the AMD DeepSeek engine package. "
              + ("The DGX Spark (GB10) builds its own: spark/build_engine.sh puts strata-dsv41 beside the engine (docs/DGX_SPARK_LINUX.md). "
                 if is_arm64() else "")) + dsv41_manual_steps(eng)
             + " Or update Strata later (a newer version may add your card), pick another model, or build the engine for your card yourself "
             "(docs/DEEPSEEK41.md, \"Building the engine for your card\"). Nothing has been downloaded.")
    ok(f"engine: {exe}")

    step(5, f"downloading {fam['title']} {model}")
    for f in shards:
        if f.exists() and done(f):
            ok(f"{f.name} already downloaded")
        else:
            if a.gguf_dir and not f.exists():
                fail(f"{f} is missing", f"put {f.name} in {folder}, or drop --gguf-dir to download it")
            if not a.gguf_dir and f.name not in DSV41_LEGACY_FILES:      # (a first-release file is checked, never fetched under its old name)
                download((md.get("hf") or fam["hf"]) + f.name, f)
        if f.name in DSV41_LEGACY_FILES:
            ok(f"{f.name}: a file of the first release of this set (the same weights); it is used as it is")
        pin = fam["sha256"][f.name]
        new = DSV41_NEW_SHARD1.get(f.name)
        if new and a.gguf_dir and f.is_file() and f.stat().st_size == new[0] and f.stat().st_size != pin[0]:      # shard 1 with the 2026-10-26 chat template
            pin = new
            ok(f"{f.name}: the version with ISTA's newer chat template (the same weights); it is used as it is")
        verify_sha256(f, *pin, own=bool(a.gguf_dir) or f.name in DSV41_LEGACY_FILES)
    for f in shared:                                  # the Engram, the drafter, the metadata shard: once for every official set
        if f.exists() and (done(f) or own_files[f]):
            ok(f"{f.name} already here ({f.parent})")
        else:
            download(DSV41_SHARED[f.name] + f.name, f)
        verify_sha256(f, *fam["sha256"][f.name], own=own_files[f], also=DSV41_SHA_ALSO.get(f.name, ()))
    ok(f"model files: {folder} ({len(shards)} shards" + (f"; shared files: {shared_dir if any(not own_files[f] for f in shared) else shared[0].parent}" if shared else "") + ")")

    step(6, "preparing the model for Strata")
    data, _ = data_folder_for(a)
    pack = data / "packs" / tag.lower()            # the engine's map of the shards, the hash tables, the tokenizer: a few MB
    have_pack = (pack / "tokenizer" / "vocab.json").exists() and (pack / "tensors.txt").exists()
    if have_pack and ds_vision == "on" and mmproj_file is not None and not dsv41_pack_has_tower(pack):
        warn("the pack from an earlier run was made without the picture reader: building it again with it (a few minutes)")
        shutil.rmtree(pack, ignore_errors=True)
        have_pack = False
    if have_pack and ista and dsv41_pack_layout(pack) not in (None, layout):
        warn(f"the pack from an earlier run was made from the {'four' if layout == 'six' else 'six'}-shard files of this set; these are the {layout}-shard files: building it again")
        shutil.rmtree(pack, ignore_errors=True)
        have_pack = False
    if have_pack and ista:                         # a pack from an earlier run is kept only when the engine finds all of its files
        good, why = dsv41_pack_info(exe, pack, lib_dirs)
        if not good:
            warn("the pack from an earlier run is incomplete; building it again")
            shutil.rmtree(pack, ignore_errors=True)
            have_pack = False
    if not have_pack:
        if ista:
            if udlow:       # the set's own shard 1 carries the metadata (and the drafter's geometry); shared = [Engram, drafter]
                run(dsv41_pack_cmd(folder, [f.name for f in shards], shared[0], shards[0], shared[1], pack, mmproj_file))
            else:
                run(dsv41_pack_cmd(folder, [f.name for f in shards], shared[0], shared[1], shared[2], pack, mmproj_file))
        else:
            run([sys.executable, str(ROOT / "tools" / "dsv41" / "pack_gguf.py"), "--model", str(folder), "--out", str(pack),
                 *(["--mmproj", str(mmproj_file)] if mmproj_file else [])])
    if ista:
        prof = DSV41_PROFILES / md["profile"]          # which experts are used most: the RAM store starts with them (the engine reads <pack>/expert-profile.bin)
        if prof.is_file() and not (pack / "expert-profile.bin").exists():
            pack.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prof, pack / "expert-profile.bin")
        good, why = dsv41_pack_info(exe, pack, lib_dirs)
        if not good:
            tail = "\n".join(l for l in why.splitlines() if "MISSING" in l)[:600]
            fail(f"the model's pack is not complete ({tail or why[-300:]})",
                 f"delete {pack} and run setup again; if it is still incomplete, the files in {folder} may be damaged (setup checks them)")
        ok(f"pack: {pack} (checked: every file the engine needs is there)")
        if os.environ.get("STRATA_DSV41_NO_SMOKE") == "1":
            warn("the 16-word test run was skipped (STRATA_DSV41_NO_SMOKE=1)")
        else:
            say("  Testing the model with a short question (a few minutes: the engine reads the first experts from the disk) ...")
            good, why = dsv41_smoke(exe, pack, lib_dirs, [c["index"] for c in cards] if (a.gpu is not None or a.gpus or (hip and len(cards) > 1)) else None)
            if not good:
                fail("the model does not answer sensibly after the preparation: " + (why.splitlines()[-1] if why else "no answer"),
                     f"delete {pack} and run setup again so the pack is built fresh; if the test fails again, "
                     f"tell us (the log is in {home()})")
            ok("test question answered sensibly")
    else:
        ok(f"pack: {pack}")
    if dsv41_context.problem(ctx, dsv41_context.read_limit(pack)):       # the pack's own limit (ds4.txt) is the last word; no file, no limit
        fail(dsv41_context.problem(ctx, dsv41_context.read_limit(pack)), f"choose a smaller --context (the pack in {pack} says what it reads)")

    step(7, "writing the start script")
    # --dspark auto, never off: "off" still LOADS the drafter, and its 7 GB of experts take the graphics memory (found 2026-10-23);
    # auto keeps it unloaded when the experts are far larger than the card.  The RAM store sizes itself (--ssd-cache auto is the
    # engine's default), and <pack>/expert-profile.bin is read from the pack folder without a flag.  The layer-major chunked prompt
    # is the engine's own default (1024 on a single card whose cards hold most of the experts; its own choice for several); on a
    # unified-memory APU the chunk is written out instead (dsv41_prefill_chunk_args: 8,192 reads 11-15% faster there).
    # --dspark-conf 0.65: the drafts' confidence floor (H100 replay, DSPARK-DEFAULTS.md: within 0.01-0.02 of the per-class best,
    # never below 0.3).  The engine has no flag for the draft block (3 in the same table): it reads the block size from the pack,
    # so only the floor is written.
    # A Mac (0.2.7.1): the drafter is not loaded at all (--dspark none). It was exact there but not faster (15.1 and 11.0 tokens/s against 15.3
    # without it on the M5 Max), and the engine's "auto" would load it on a 128 GB Mac: its 7 GB of experts take memory the trunk's experts need.
    args = ["--pack", str(pack), "--max-context", str(ctx), "--eos", "1", "--vision", ds_vision, "--dspark", "none" if mac else "auto",
            *(["--dspark-conf", "0.65"] if md.get("recipe") == "ista" and not mac else []), *dsv41_prefill_chunk_args(cards, ram),
            *plan["args"]]
    env = dsv41_config_env(cards)
    cfg = {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": fam["name"], "arch": fam["arch"], "log": str(home() / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port, "engine_ready_s": 3600,      # 300+ GB read from the files: a slow disk needs its time
           **({"env": env} if env else {})}
    if cuda_tk == 12:
        cfg["cuda"] = 12
    if hip:
        cfg["backend"] = "hip"
    if mac:
        cfg["backend"] = "metal"
    idx = [c["index"] for c in cards]
    if len(idx) > 1 or a.gpu is not None or a.gpus:      # (a.gpu is also setup's own pick: one card pinned when several devices are in the PC)
        cfg["gpu"] = idx if len(idx) > 1 else idx[0]
        cfg["gpus_asked"] = True
        if hip:                                          # who chose them (strata_control/gpu_pick.py): the person, or setup itself
            cfg["gpus_by"] = "person" if named else "setup"
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = home() / f"strata-{tag.lower()}.json"
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None, explicit_args(a))
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")
    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    say("  Loading takes a few minutes: the CPU's share of the experts is read from the files first.")
    if a.no_start:
        return 0
    return start(cfg_path, port)


def preview_family(a) -> int:
    """--family qwen36: a family setup knows but the engine does not run yet (PREVIEW_FAMILIES).  Downloads the chosen
    size (pinned revision, size and SHA-256 checked) and, with --vision, the image encoder - and says plainly that
    Strata cannot start it yet.  Nothing is configured: no engine, no pack, no start script."""
    fam = PREVIEW_FAMILIES[a.family]
    names = list(fam["models"])
    say()
    say(f"  {fam['title']} - {fam['by']}: {fam['about']}")
    warn(f"this Strata version does not run {fam['title']} yet (support is in development): setup downloads its "
         "files, and the engine refuses them with a message until a version that runs it")
    if a.model and a.model not in names:
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names))
    ram = ram_gb()
    for i, m in enumerate(names, 1):
        d = fam["models"][m]
        size = fam["sha256"][fam["file"].format(q=m)][0]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        say(f"  {i}) {m:10s} {d['about']}; download {size / 1e9:.1f} GB{fit}")
    if a.check:
        return 0
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], "1", a.yes)) - 1]
    say(f"  Its license: {fam['license']}")
    want = [fam["file"].format(q=model)]
    if a.vision in ("yes", "gpu", "cpu"):
        want.append(fam["mmproj"])
    folder = Path(a.models_dir) / (fam["tag"] + model)
    files = [folder / w for w in want]
    left = sum(fam["sha256"][f.name][0] - (f.with_name(f.name + ".part").stat().st_size
                                           if f.with_name(f.name + ".part").is_file() else 0)
               for f in files if not (f.exists() and done(f))) / 1e9
    if left and free_gb(folder) < left:
        fail(f"not enough free disk space in {folder}: need ~{left:.0f} GB, {free_gb(folder):.0f} GB free", ("use --data-dir on a bigger drive (strata launch --data-dir)" if CONTROL else "use --models-dir on a bigger drive"))
    if left and free_gb(folder) < left + 1:
        warn(f"little free disk space in {folder}: {free_gb(folder):.1f} GB free for about {left:.1f} GB. Setup goes on and should fit")
    step(5, f"downloading {fam['title']} {model}")
    for f in files:
        url = (fam["mmproj_hf"] if f.name == fam["mmproj"] else fam["hf"]) + f.name
        download(url, f, "vision encoder" if f.name == fam["mmproj"] else None)
        verify_sha256(f, *fam["sha256"][f.name])
    ok(f"{fam['title']} {model}: {folder}")
    say()
    say(f"{fam['title']} is downloaded, but this Strata version cannot run it yet (its engine path is in "
        "development); nothing was configured. Run setup again without --family qwen36 to install a model it runs.")
    return 0


def mac_vision(asked, model: str, ram: float) -> str:
    """The picture setting of a Mac's setup: "gpu", "cpu" or "none".  Pictures are on wherever the model can read them (owner rule
    2026-10-18) and the reader follows strata_control/vision_reader.py: on a Mac, the GPU (Metal) reader, because the memory is shared
    (no card to keep free) and the CPU reader is ~100x slower.  --vision cpu is kept, with a note (recommend, never force)."""
    from strata_control import vision_reader as VR
    if asked in ("no", "none", "off"):
        return "none"
    if asked is None and not images_default(model, "qwen", ram):
        return "none"
    if MODELS.get(model, {}).get("vision") is False:
        if asked is not None:
            warn(f"images are not available with {model} yet: off")
        return "none"
    mode = VR.wanted(asked, None, unified=True)
    if mode == "cpu":
        warn(VR.MAC_CPU_NOTE)
    return mode


def mac_reader(eng: Path, vision: str):
    """The picture reader the Mac engine folder carries (strata-vision: Metal built in), for `vision` "gpu" or "cpu".  A folder
    without it gets pictures off, said once."""
    from strata_control import vision_reader as VR
    r = VR.choose(eng, vision, windows=False, mac=True)
    if r.mode == "none":
        warn(r.note or "this engine has no picture reader: pictures off")
    return r


def mac_control_setup(a) -> int:
    """The app (and the Terminal's `strata launch`) on a Mac, STRATA_CONTROL_INSTALL=1: the same job as on Windows and Linux. It
    downloads and prepares the model files and writes a run config whose engine is "component:strata-engine"; it builds nothing
    (the Metal engine is a component: a ready-made package from the feed, or the one inside the installer) and runs no
    metal/setup_mac.py. Which sizes a Mac can run, whether one stays in memory or streams its experts from the SSD, and the
    Metal flags of the config are strata_control/mac_fit.py's, the module setup_mac.py and the model list use too."""
    from strata_control import mac_fit as MF
    say("Strata on a Mac (Apple Silicon, Metal)")
    step(1, "checking your Mac")
    info = MF.mac_info()
    ram = float(info["ram_gb"] or ram_gb())
    ok(f"{info['chip']}, {ram:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    if info.get("rosetta_python"):
        say("  (the Python that runs this is an Intel build running through Rosetta: Strata works, and the Apple Silicon Python "
            "from python.org or Homebrew in /opt/homebrew is a little faster to start)")
    problems = MF.mac_problems(info, need_compiler=False)       # the app compiles nothing: no Command Line Tools needed
    for p in problems:
        warn(p)
    data, elsewhere = data_folder_for(a)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"
    free = free_gb(data if data.exists() else home())
    ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        fail(problems[0])

    step(2, "your choices")
    app_sizes = [m for m, d in MF.MAC_MODELS.items() if d.get("family", "qwen") in ("qwen", "gyro")]   # Gyro: one file, no repack
    if a.family not in (None, "qwen", "gyro"):
        fail(f"the app on a Mac runs the Qwen3.8-Flash-Next sizes ({', '.join(app_sizes)}); {a.family} is not one of them",
             "choose Qwen3.8-Flash-Next")
    rec, notes = MF.recommend_model(ram)
    if rec not in app_sizes:
        rec = "Q2_0"
    model = a.model or rec
    if model not in app_sizes:
        fail(f"{model} is not offered on a Mac in the app", "choose one of: " + ", ".join(app_sizes))
    for n in MF.model_note(model, ram):
        warn(n)
    vision = mac_vision(a.vision, model, ram)
    ctx = a.context or MF.recommend_context(ram)
    if ctx > MF.TESTED_CONTEXT:
        say(f"  context {ctx}: measured on the Mac up to {MF.TESTED_CONTEXT} (needles found at 8K-32K); longer is kept")
    ok(f"{model}, context {ctx}")
    family = MF.MAC_MODELS[model].get("family", "qwen")
    fam = FAMILIES[family]
    tag = fam["tag"] + model
    stream = MF.streams(model, ram)

    step(4, "the Strata engine (Metal)")
    from strata_control import engine_component as EC
    llama = get_llama_cpp()                            # gguf-py for the preparation tools only: no git, no compiler
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (gguf-py for the preparation tools)")
    reader = None
    try:
        found = EC.resolve(full=False)
        ok(f"engine component: {found.version} ({found.platform}-{found.backend})")
        if vision != "none":
            reader = mac_reader(Path(found.root), vision)
            vision = reader.mode
    except EC.EngineMissing as e:
        warn(f"no Metal engine is installed for this Mac yet ({e}). The model is downloaded and prepared now; the engine comes "
             "with the app or its update feed, and is looked up when the model loads")
        if vision != "none":
            warn("pictures are off for now: the picture reader comes with the engine (run setup again once it is installed)")
            vision = "none"
    else:
        if family == "gyro":                           # an engine from before the trellis kernels refuses the file when it loads
            try:
                has_tq = b"inputs are rotated on the GPU" in Path(found.exe).read_bytes()
            except (OSError, AttributeError):
                has_tq = True                          # cannot tell: not a reason to hold the install back
            if not has_tq:
                warn(f"the Metal engine component {found.version} has no kernels for Gyro's trellis experts yet: the model is "
                     "downloaded and prepared, and it runs once the app or its update feed brings a newer engine")
    ok("engine: the engine component (looked up when the model loads)")
    ok("images: " + {"none": "off", "gpu": "on (read on the Mac's GPU)", "cpu": "on (read on the CPU)"}[vision])

    step(5, f"downloading {fam['title']} {model}")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    files = gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else \
        [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)]
    problem = gguf_dir_problem(models_dir, files[0], fam, model) if a.gguf_dir else None
    if problem:                                        # files Strata cannot run, or another choice's files
        fail(*problem)
    for s in files:                                    # a whole file copied in by hand has no finish mark
        if s.exists() and not done(s) and whole_shard(s):
            try:
                mark(s, "whole (checked against its own tensor directory)")
            except OSError as e:                       # a read-only folder: the file is still whole
                warn(f"{s.name} is whole but its finish mark cannot be written ({e})")
    need = (0 if a.gguf_dir or all(s.exists() and done(s) for s in files) else MODELS[model]["download_gb"]) + 6
    if free_gb(models_dir if models_dir.exists() else data) < need:
        warn(f"about {need:.0f} GB are needed on the disk: kept as you chose; it may run out")
    if not a.gguf_dir:
        models_dir.mkdir(parents=True, exist_ok=True)
        missing = [s.name for s in files if not (s.exists() and done(s))]
        if missing:
            say(f"  The model files go in {models_dir}")
        for s in files:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            download(fam["hf"].format(q=model) + s.name, s)
    check_shards(files)
    ok("model files present")
    mmproj = None
    if vision != "none":
        mmproj = Path(models_root) / fam["mmproj"]
        if not mmproj.exists():                        # an earlier install's copy (the same file for every family)
            mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        elif not (mmproj.exists() and done(mmproj)):
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        if fam["mmproj"] in ISTA_FILES:
            verify_sha256(mmproj, *ISTA_FILES[fam["mmproj"]])
        ok(f"vision encoder: {mmproj}")

    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(files[0]), "--out", str(pack),
             *fam.get("pack_args", [])], env=env)       # Gyro: --compat-bf16
    ok(f"model prepared: {pack}")
    use_mtp = MF.stream_mtp(stream, ram)
    mtp = (find_in(roots, "mtp/mtp-q2_0.gguf") or data / "mtp" / "mtp-q2_0.gguf").parent
    rt = mtp / "rt"
    if stream and not use_mtp:
        ok("the experts stream from the SSD: no MTP draft layer (one token per step is faster there)")
    elif stream:
        ok(f"the experts stream from the SSD; the MTP draft layer stays on, with --spec {MF.STREAM_SPEC}")
    if use_mtp and a.mtp == "off" and not (mtp / "mtp-q2_0.gguf").exists():
        ok("the draft layer (MTP) is off: its ~5 GB are not downloaded (`--mtp on` fetches it later)")   # round 8, N3
        use_mtp = False
    if use_mtp and not (mtp / "mtp-q2_0.gguf").exists():
        say("  The MTP draft layer (speculative decoding) comes from the original Qwen checkpoint: only its ~5 GB of MTP tensors are downloaded.")
        fetch_mtp(mtp, env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
    if use_mtp:
        rt.mkdir(parents=True, exist_ok=True)          # the Metal engine reads mtp-q2_0.gguf beside it (no rt pack)
        refresh_draft_vocab(rt, a.draft_vocab or "cjk")
        ok(f"MTP draft layer: {mtp / 'mtp-q2_0.gguf'}")

    step(7, "writing the run settings")
    port = a.port or 8080
    args = MF.engine_args(pack, [files[0], files[0]] if family == "gyro" else files, rt, ctx, stream, ram,
                          ROOT / "data" / "expert-profile.bin")      # Gyro: one file, the model and its n-gram table
    cfg = {"exe": engine_exe(Path("engine")), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(home() / f"strata-{tag.lower()}.log"),
           "lib_dirs": [], "port": port, "backend": "metal", "env": {"STRATA_METAL_STREAM": "1" if stream else "0"}}
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.draft_vocab:
        cfg["draft_vocab"] = a.draft_vocab
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    if vision != "none":                               # the Mac's one program reads on the GPU (--gpu) or the CPU (strata_control/vision_reader.py)
        from strata_control import vision_reader as VR
        cfg["args"].append("--vision")
        cfg["vision"] = VR.vision_section(reader, {"exe": str(reader.exe), "mmproj": str(mmproj), "model": str(files[0]),
                                                   "gpu": vision == "gpu", "max_tokens": vision_tokens(a.vision_tokens, vision, None)})
        if vision == "cpu":
            cfg["vision"]["threads"] = vision_threads()
    cfg_path = home() / f"strata-{tag.lower()}.json"
    write_setup_config(cfg_path, cfg)
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"start script: {script.name}")
    say()
    say("All set.")
    return 0


def mac_dsv41_setup(a) -> int:
    """DeepSeek-V4.1-Flash on an Apple Silicon Mac (the app and `strata launch`, Control mode; 0.2.7.1): the model files and the DeepSeek
    engine package (macOS arm64, Metal) from the signed feed.  The steps are install_dsv41's; this only checks the Mac and hands over its
    one pool of unified memory as the "card" the fit plan reads (strata_control/dsv41_fit.py, mac_plan)."""
    from strata_control import mac_fit as MF
    step(1, "checking your Mac")
    info = MF.mac_info()
    ram = float(info["ram_gb"] or ram_gb())
    ok(f"{info['chip']}, {ram:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = MF.mac_problems(info, need_compiler=False)       # the app compiles nothing: no Command Line Tools needed
    for p in problems:
        warn(p)
    data, _elsewhere = data_folder_for(a)
    if not a.models_dir:
        a.models_dir = str(data / "models")
    ok(f"free disk: {free_gb(data if data.exists() else home()):.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        fail(problems[0])
    gpu = {"index": 0, "name": info["chip"], "vendor": "apple", "vram_gb": 0.0, "unified_gb": ram, "usable": True}
    return install_dsv41(a, a.family, gpu, False, False, 0, False, a.port or 8080, ram, None, [gpu])


def main() -> int:
    try:                                               # files made by a run as administrator still belong to the user
        from strata_control import private_files as _pf
        _pf.adopt_user_defaults()
    except Exception:
        pass
    if sys.platform == "darwin" and not CONTROL:       # a Mac: metal/setup_mac.py (the Metal engine); the app (Control mode) goes on below
        return mac_setup(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES) + list(PREVIEW_FAMILIES) + list(MULTI_FILE_FAMILIES),
                    help="qwen = Qwen3.8-Flash-Next, swift = Swift 1.5, qwen36 = Qwen3.6-35B-A3B (pictures with --vision, "
                         "several requests at once with --parallel; docs/QWEN36.md), deepseek41 = DeepSeek-V4.1-Flash "
                         "(the official 1.75 / 2.00 / 2.25 bpw versions, 239 / 257 / 274 GB; docs/DEEPSEEK41.md)")
    ap.add_argument("--model", choices=list(dict.fromkeys([*MODELS, *(m for f in PREVIEW_FAMILIES.values()
                                                                       for m in f["models"]),
                                                          *(m for f in MULTI_FILE_FAMILIES.values() for m in f["models"])])))
    ap.add_argument("--context", type=int)
    ap.add_argument("--rope-scaling", choices=["none", "linear", "yarn"],
                    help="the RoPE extension for a context past the model's trained 262144: linear (position "
                         "interpolation) or yarn - llama.cpp's types. Omitted with a scaled context, the setup "
                         "picks yarn; 'none' is refused for such a context")
    ap.add_argument("--rope-scale", type=float,
                    help="the extension factor, finite and >= 1 (default: the final context over the trained "
                         "262144, at least 1 - 1.5 for 384K, 2 for 512K, 1 inside the trained range)")
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4", "fp16"],
                    help="KV cache precision above 8K context: int8 (default), q4_0 (half the memory, a little less "
                         "precise) k8v4 (hybrid: INT8 K + 4-bit V, 816 B/cell) or fp16 (16-bit, twice the memory of int8)")
    ap.add_argument("--vision", choices=["yes", "no", "none", "gpu", "cpu"],
                    help="let the model read images (on by default where the model can; yes = the best reader for this card: the CPU "
                         "under 24 GB of VRAM, the GPU from 24 GB, any make; gpu / cpu = that reader; no = text only)")
    ap.add_argument("--vision-tokens", type=int, metavar="N",
                    help="the most image tokens a picture becomes (default 1024 with the encoder on the GPU, 300 on "
                         "the CPU): more reads small text and charts better, and takes longer to encode; remembered "
                         "for this model")
    # image generation (Qwen-Image): in preview, not offered in setup or --help; the options still work (docs/IMAGES.md)
    ap.add_argument("--images", choices=["qwen-image", "no"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--image-quant", choices=["Q4_K_M", "Q6_K", "Q8_0"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--image-backend", choices=["cuda", "hip", "vulkan", "cpu"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--image-native", choices=["on", "off"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--image-gpu-policy", choices=["auto", "keep", "shrink", "unload"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--experimental-speed-projection", metavar="on|off|GGUF",
                    help="EXPERIMENTAL, off by default: the control vector in data/experimental-speed-projection "
                         "(or another GGUF) as a projection on layers 4-44; see docs/DETAILS.md")
    ap.add_argument("--port", type=int, help="the server's port (default: the one the install was set up with, 8080 for a new one)")
    ap.add_argument("--gpu", help="one GPU, numbered as nvidia-smi numbers them (AMD: as setup lists them) (default: asked when several can be "
                                  "used; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="several GPUs sharing one model, as nvidia-smi numbers them (AMD: as setup lists "
                                   "them): \"0,2\", or \"all\" (every card that can); Strata puts the faster card last, "
                                   "and the last card is the Main one. Saved, also when starting (see docs/MULTI_GPU.md)")
    ap.add_argument("--gpu-roles", metavar="LIST", help="a role for each GPU: main, helper or off, e.g. 1=main,0=helper (cards you do not "
                                                        "name are off). Main does the last steps of every answer. Saved (docs/MULTI_GPU.md)")
    ap.add_argument("--layer-split", help="with --gpus: where each later GPU's layers start (\"18\", \"16,32\"), one "
                                          "rising number per GPU after the first - not layers per card; default "
                                          "auto, placed from each GPU's free VRAM")
    ap.add_argument("--no-remote-expert-opt", action="store_true",
                    help="with two or more GPUs: leave out --remote-expert-opt, which setup adds there (#578)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = all IPv4 "
                                   "interfaces, :: = all IPv6 interfaces (issue #26; an --api-key is recommended)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0 or --host ::); "
                                      "several keys: key1,key2")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.bat/.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--data-dir", help="where the model files go (~70-120 GB): default Strata-data next to this folder, "
                                       "remembered for every Strata folder on this PC")
    ap.add_argument("--move-data", action="store_true",
                    help="with --data-dir: move the model files already on this PC into the new folder (without it nothing "
                         "is moved: the old files stay usable and the new folder gets the new downloads)")
    ap.add_argument("--set-default-data-dir", action="store_true",
                    help="with --data-dir: remember the new folder as the data folder without moving anything")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--dsv41-package", metavar="FILE", help="DeepSeek: the engine package you downloaded from the release page (a .tar.gz or .zip, "
                                                           "or its unpacked folder); it is checked and installed instead of being fetched")
    ap.add_argument("--feed", metavar="URL|DIR",
                    help="where the signed update feed (the DeepSeek engine package) is: default STRATA_UPDATE_FEED, a remembered https feed, "
                         "or the release's (as `strata update --feed`; a folder is a local test feed)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have (a folder with every shard: "
                                       "<name>-00001-of-0000N.gguf ... -0000N-of-0000N.gguf)")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="install another model or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the installed engine, Python packages and model settings as a start would, without "
                         "starting the model (UPDATE.bat / update.sh run it after a git pull)")
    ap.add_argument("--rollback-engine", action="store_true",
                    help="put back the engine an update replaced (kept in engine/.previous), and keep the current one there")
    ap.add_argument("--build", action="store_true", help="compile the engine instead of using the ready-made one")
    ap.add_argument("--source", choices=SOURCES, default=None,
                    help="where the model files come from: auto (default: Hugging Face), huggingface or modelscope "
                         "(mainland China: the same files, checked against ModelScope's published SHA-256; "
                         "STRATA_SOURCE)")
    ap.add_argument("--inspect", nargs="+", metavar=("SOURCE", "VARIANT"),
                    help="what a GGUF is and whether Strata runs it, from its headers only (no download): a file, a "
                         "folder, a URL, ms:owner/repo (ModelScope) or hf:owner/repo, and optionally a variant name")
    ap.add_argument("--cuda", choices=["12", "13", "auto"], default=os.environ.get("STRATA_CUDA") or None,
                    help="NVIDIA: the CUDA toolkit of this model's engine. auto (default): CUDA 13, the ready-made "
                         "engine; CUDA 12 (experimental) when a chosen card is older than CUDA 13 supports (Pascal, "
                         "Volta). 12 also runs with an older driver (Windows 528+, Linux 525+). docs/OLDER_GPUS.md")
    ap.add_argument("--prebuilt", default=os.environ.get("STRATA_PREBUILT_URL", PREBUILT_URL),
                    help="where the ready-made engine is (a URL folder or a local folder)")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's settings for this PC (about 15-30 minutes, longer on a slow card), then start the model")
    ap.add_argument("--draft-vocab", choices=list(DRAFT_VOCABS),
                    help="the draft layer's tokens: cjk = with Chinese, Japanese and Korean (default), en = English "
                         "and code only (~110 MiB less VRAM, English answers 1-2%% faster), cyrillic = English, code "
                         "and the Cyrillic script (Ukrainian, Russian... answers decode ~30%% faster), fr = English, "
                         "code and French (French answers: 18%% more drafts accepted)")
    ap.add_argument("--low-ram", choices=["auto", "on", "off", "resident", "mmap"], default="auto",
                    help="read the model's experts from one file in its folder instead of copying them all into RAM "
                         "(for a PC with a big GPU and little RAM); auto: when the experts would not fit the RAM. In "
                         "this mode the experts the GPU does not hold are copied into RAM once when they fit (resident), "
                         "else read through the OS file cache (mmap); resident / mmap force one of the two")
    ap.add_argument("--resident-budget-gib", type=float, metavar="N",
                    help="UD-Q4_K_XL, UD-IQ4_XS: the GiB of its experts kept in RAM (default: the RAM less 24 GB, 40 on 64 GB; "
                         "more is kept as you choose, with a note)")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the engine leaves free for other programs (a game, another model; the engine's "
                         "default: 700); the expert cache takes that much less")
    ap.add_argument("--mtp", choices=["on", "off"],
                    help="the draft layer (MTP): off starts the model without it, which frees about 1 GB of graphics memory "
                         "for the expert cache and takes less RAM, at the price of speed; it is kept in the run config "
                         "(\"mtp\": false) and a later setup run does not undo it. on: back to the default")
    ap.add_argument("--parallel", type=int, metavar="N",
                    help="up to N requests decode together (batch slots, opt-in; default: one at a time, the others "
                         "wait). Each slot takes VRAM from the expert cache; setup says what it recommends")
    ap.add_argument("--replicas", type=int, metavar="N",
                    help="several GPUs: run N independent engines, each on its own equal group of the chosen cards (2 "
                         "replicas on 4 cards = two 2-card engines), requests spread over them (opt-in; more total "
                         "throughput for a model that fits fewer cards; each replica holds its own copy of the model)")
    ap.add_argument("--replica-profile", action="store_true", default=False,
                    help="two or more GPUs, Qwen3.8 IQ3_S/IQ3_XXS/Q2_0: one engine per card behind the server instead of one "
                         "layer split, the experts shared in RAM, the batch switches on (opt-in; about 1.8 times the "
                         "throughput at 8-16 requests at once on 4 Radeon AI PRO R9700; needs about 61 GB of RAM for 2 "
                         "engines and 66 GB for 4; a layer split you chose, replicas or --parallel always win)")
    ap.add_argument("--kv-streaming", choices=["auto", "on", "off"], default="auto",
                    help="from a 64K context: keep the KV cache in RAM and only the attention's window in VRAM (more "
                         "experts fit on the GPU); auto: when the RAM has room for it")
    ap.add_argument("--backend", choices=["cuda", "hip", "sycl"],
                    help="cuda = NVIDIA (default), hip = AMD RX 7900 / 7800 / 7700 XT, RX 9060 XT / 9070 / AI PRO R9700 on "
                         "Linux or Windows (chosen by itself when the PC has no NVIDIA card Strata can use), "
                         "sycl = Intel Arc, EXPERIMENTAL: Linux, built from source (docs/INTEL_ARC.md)")
    ap.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.model and a.model not in MODELS and a.family not in PREVIEW_FAMILIES and a.family not in MULTI_FILE_FAMILIES:
        ap.error(f"--model {a.model} is a size of " + ", ".join(f"--family {f}" for f, d in {**PREVIEW_FAMILIES, **MULTI_FILE_FAMILIES}.items()
                                                              if a.model in d["models"]))
    if a.source:
        os.environ["STRATA_SOURCE"] = a.source
    if a.inspect:                                      # headers only: nothing is installed
        sys.exit(subprocess.run([sys.executable, str(ROOT / "tools" / "strata_inspect.py"), *a.inspect[:2]]).returncode)
    if sys.platform == "darwin" and CONTROL:           # the app on a Mac: model files only; the Metal engine is a component
        if a.family in MULTI_FILE_FAMILIES:            # DeepSeek-V4.1-Flash: its own engine package, the generic DeepSeek install
            return mac_dsv41_setup(a)
        return mac_control_setup(a)
    if a.backend == "sycl":                            # Intel Arc: the SYCL port's own setup (sycl/setup_intel.py)
        return sycl_setup(sys.argv[1:])
    if a.resident_budget_gib is not None and not a.resident_budget_gib > 0:
        ap.error("--resident-budget-gib takes a number of GiB above 0, e.g. --resident-budget-gib 32")
    if a.vision_tokens is not None and a.vision_tokens < 1:
        ap.error("--vision-tokens takes a number of image tokens, 1 or more, e.g. --vision-tokens 768")
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.gpu is not None:                             # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as nvidia-smi numbers them, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    a._roles = None
    if a.gpu_roles:
        if a.gpu is not None or a.gpus:
            ap.error("use --gpu-roles (a role for each card) or --gpu / --gpus (which cards), not both")
        from strata_control import gpu_roles as GR
        try:
            named = GR.parse_roles(a.gpu_roles)
        except ValueError as e:
            ap.error(f"--gpu-roles: {e}")
        on = [i for i, r in named.items() if r != "off"]
        if not on:
            ap.error("--gpu-roles: setup needs at least one card that is Main or Helper")
        mains = [i for i, r in named.items() if r == "main"] or [sorted(on)[-1]]
        helpers = sorted(i for i in on if i != mains[0])
        a._roles_explicit = True
        a._roles = {**{i: "off" for i in named}, **{i: "helper" for i in helpers}, mains[0]: "main"}
        order = helpers + [mains[0]]
        if len(order) > 1:
            a.gpus = ",".join(str(i) for i in order)
        else:
            a.gpu = order[0]
    say("Strata - " + (MULTI_FILE_FAMILIES[a.family]["title"] if a.family in MULTI_FILE_FAMILIES else
                       OWN_PATH_FAMILIES[a.family]["title"] if a.family in OWN_PATH_FAMILIES else "Qwen3.8-Flash-Next")
        + " on a normal PC (a GPU + system RAM + CPU)")
    have = installed_configs()
    # A plain start of an installed model finds everything where it already is: it creates, moves and copies nothing
    # (the model paths are in the run config; the data folder is the one settings.json remembers).  Setting up, checking,
    # tuning and updating look for and gather the model files (data_folder).
    plain_start = bool(have) and not (a.setup or a.model or a.family or a.check or a.no_start or a.update
                                      or a.calibrate or a.rollback_engine or a.data_dir)
    models_dir_given = a.models_dir is not None
    data, elsewhere = data_folder_in_place() if plain_start else data_folder_for(a)   # the model files
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")

    if a.rollback_engine:                              # #670
        return rollback_engine(12 if a.cuda == "12" else 13)

    # ---- 0. already installed: just start it
    if a.update:                                       # #475: UPDATE.bat / update.sh - never starts the model
        return update_install(have, a)
    explicit = a.setup or a.model or a.family or a.check or a.no_start
    adopted = None                                     # #629: the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already here
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.family, a.model, a.context = ch["family"], ch["model"], a.context or ch["context"]
                adopted = prev
                a.kv = a.kv or ch["kv"]
                a.vision = a.vision or ch["vision"]
                a.experimental_speed_projection = a.experimental_speed_projection or ch["esp"]
                a.host, a.api_key = a.host or ch["host"], a.api_key or ch["api_key"]
                a.port = a.port or ch["port"]
                if a.vram_reserve_mib is None:          # #493: an explicit reserve set up before
                    a.vram_reserve_mib = ch.get("vram_reserve_mib")
                if a.cuda is None and ch.get("cuda") == 12:   # the experimental CUDA 12 engine, as before
                    a.cuda = "12"
                if isinstance(ch.get("gpu"), list):     # a layer split: set up across the same cards again
                    a.gpus = a.gpus or ",".join(str(g) for g in ch["gpu"])
                    a.layer_split = a.layer_split or ch.get("layer_split")
                else:
                    if earlier_card_kept(prev):
                        a.gpu = a.gpu if a.gpu is not None else ch.get("gpu")
                a.yes = True
    global GPU_PICK, OLD_GPUS
    # starting an installed model: --gpus 0,2 (or all) saves those cards for it and starts on them (it used to start
    # on the first one alone unless given with --setup), --gpu N runs this start on one card; neither: the saved
    # choice, and asked once when the PC has cards that could share the model
    run_gpu = start_gpus(a.gpus) or a.gpu
    port = a.port or 8080                              # a new install's port (issue #32: --port for an existing one)
    if a.calibrate and (a.setup or a.check or not have):    # #2036: said, not dropped
        why = ("--setup" if a.setup else "--check" if a.check else "no model is installed yet")
        warn(f"--calibrate does nothing with {why}: it tunes a model that is already installed. Run setup first, "
             "then " + ("START-HERE.bat" if WIN else "./setup.sh") + " --calibrate [--model NAME]")
    if have and a.calibrate and not (a.setup or a.check):
        targets = calibration_targets(have, a.model, a.family)
        if not targets:                                # #2036: --model names a model that is not installed
            names = ", ".join(sorted({choices_from_config(c)["model"] or c.stem for c in have}))
            fail(f"--calibrate: {a.model or a.family} is not installed (installed: {names})",
                 "calibrate an installed model, or set this one up first without --calibrate")
        have = targets
        engine_update_note()
        pick_cfg = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
            pick_cfg = have[int(ask("Tune which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if not calibrate_config(pick_cfg):             # #447: said again where it is not lost above the start
            say()
            warn("this PC is NOT tuned: the tuning failed (the reason is above); the model "
                 + ("keeps" if a.no_start else "starts with") + " the default settings")
        return 0 if a.no_start else start(pick_cfg, a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
    if have and not (a.setup or a.model or a.family or a.check or a.no_start):
        engine_update_note()
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {json.loads(c.read_text(encoding='utf-8-sig')).get('model_name', c.stem)}")
        say(f"  {len(have) + 1}) install another model / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, yes=a.yes, layer_split=a.layer_split,
                     keep={"host": a.host, "api_key": a.api_key, "draft_vocab": a.draft_vocab,
                           "vram_reserve_mib": a.vram_reserve_mib, "open_browser": a.browser})
        data, elsewhere = data_folder_for(a)      # "install another model": now the model files are gathered
        roots = [data, *elsewhere]
        if not models_dir_given:
            a.models_dir = str(data / "models")

    # ---- 1. the PC
    step(1, "checking your PC")
    found = gpus()
    amd = amd_gpus()
    amd_ok = [g for g in amd if amd_problem(g) is None]
    # older NVIDIA GPUs (Pascal / Volta): the experimental CUDA 12 engine, when chosen (docs/OLDER_GPUS.md)
    OLD_GPUS = OLD_GPUS or old_gpus_opt_in(found, named_gpus(a.gpu, a.gpus), a.cuda,
                                           other=bool(amd_ok) or a.backend == "hip")
    if OLD_GPUS and a.backend != "hip":
        warn(f"older NVIDIA GPUs (Pascal / Volta) can be used ({OLD_GPUS}): experimental, through a second engine "
             "built with CUDA 12 (docs/OLDER_GPUS.md)")
    nv_ok = any(gpu_problem(g) is None for g in found)
    if not nv_ok and not amd_ok and a.backend != "hip" and maxwell_note(found):    # #3218: a warning with the way forward
        warn(maxwell_note(found))
    hip = a.backend == "hip" or (a.backend is None and not nv_ok and bool(amd_ok))
    if a.backend is None and nv_ok and amd_ok:
        # both kinds of card: asked (a first run on such a PC used to take NVIDIA without mentioning the Radeon)
        say()
        say("  This PC has NVIDIA and AMD cards Strata can use:")
        say("  1) NVIDIA: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB)" for g in found if gpu_problem(g) is None)
            + "   (recommended)")
        say("  2) AMD: " + ", ".join(f"{g['name']} ({g['vram_gb']:.0f} GB{', unified memory' if g.get('uma') else ''})"
                               for g in amd_ok)
            + f"   ({'the ready-made AMD engine' if WIN else 'compiled here'}, pictures on"
              " - docs/AMD_HIP.md)")
        say("  One engine drives the cards of one make: a model is never split across NVIDIA and AMD cards, and two models at the same")
        say("  time, one on each, is not supported yet.")
        from strata_control import gpu_fit
        bigger = gpu_fit.other_vendor_note([{**g, "vendor": "nvidia", "usable": gpu_problem(g) is None} for g in found] +
                                           [{**g, "vendor": "amd", "usable": True} for g in amd_ok], "nvidia")
        if bigger:
            say("  " + bigger)
        hip = ask("Which cards?", ["1", "2"], "1", a.yes or a.check) == "2"
        if a.check and not hip:
            say(f"  (the AMD card: {'START-HERE.bat' if WIN else './setup.sh'} --backend hip)")
    cuda_tk = 13                                       # NVIDIA: the toolkit of this model's engine (cuda_choice)
    if hip:                                            # AMD: compiled here; Windows: ready-made
        if WIN and a.gpus:
            fail("several AMD cards sharing one model (--gpus) is Linux-only for now", "use one card: --gpu N")
        say("  Your AMD GPUs:" if amd else "  No AMD GPU found (" + ("Windows lists no AMD display adapter)." if WIN
                                                                   else "the amdgpu driver's KFD topology is empty)."))
        for g in amd:
            say(f"    GPU {g['index']}: {g['name']}, {amd_mem_text(g)} - " + (amd_problem(g) or "can be used"))
        if not WIN:                                    # WSL: the Linux HIP engines hang there; a warning, the run goes on
            from strata_control.engine_component import wsl_amd_note
            wsl_note = wsl_amd_note("hip", wsl=is_wsl())
            if wsl_note:
                warn(wsl_note)
        usable = [g for g in amd if amd_problem(g) is None]
        if not WIN and (amd or amd_pci_devices()):     # a warning only: recommend, never force
            acc = amd_device_access_problem()
            if acc:
                warn(acc)
        if not amd and not WIN:                        # the KFD topology is empty: name a Strix Halo the kernel sees
            for d in amd_pci_devices():
                if d["pci_id"] in STRIX_HALO_PCI_IDS:
                    say(f"    The kernel lists an AMD Strix Halo (PCI 1002:{d['pci_id']:04x}) but /dev/kfd's topology is "
                        "empty: ROCm cannot use it yet (docs/STRIX_HALO.md: the amdgpu driver and /dev/kfd access, "
                        "your user in the render and video groups)")
        if not usable:
            fail("no AMD GPU Strata can use", f"the AMD backend runs on {AMD_CARDS}")
        a._cards_named = bool(a.gpu is not None or a.gpus)     # before setup's own pick is written into a.gpu below
        if a.gpus:                                     # a layer split across these cards, the first one the main
            chosen = amd_parse_gpus(a.gpus, amd)
            gpu = chosen[0]
        elif a.gpu is not None:
            gpu = next((g for g in usable if g["index"] == a.gpu), None)
            if gpu is None:
                fail(f"AMD GPU {a.gpu} cannot be used", "use one of: " + ", ".join(f"--gpu {g['index']}" for g in usable))
            chosen = [gpu]
        else:
            gpu = min(usable, key=amd_rank)            # a discrete card before an APU, then the most memory
            chosen = [gpu]
            if len(usable) > 1 and any(g.get("uma") for g in usable) != all(g.get("uma") for g in usable):
                others = [g for g in usable if g is not gpu]
                say(f"  Using GPU {gpu['index']} ({gpu['name']}); setup recommends the card with its own memory over "
                    "the APU's shared memory. To use another: "
                    + ", ".join(f"--gpu {g['index']} ({g['name']})" for g in others))
        # the engine is compiled for every chosen card's architecture
        gpu = {**gpu, "count": len(amd), "archs": sorted({g["arch"] for g in chosen})}
        chosen = [gpu] + chosen[1:]
        sel = [g["index"] for g in chosen]
        multi = sel if len(sel) > 1 else []
        a.gpu = gpu["index"] if len(amd) > 1 else a.gpu
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, {amd_mem_text(gpu) if gpu.get('uma') else format(gpu['vram_gb'], '.1f') + ' GB VRAM'}, "
           f"{gpu['arch']} (AMD: docs/{'STRIX_HALO' if is_strix_halo(gpu) else 'AMD_HIP'}.md)")
        if WIN and str(gpu.get("arch") or "").startswith("gfx12"):   # only a pointer; no default changes
            say("  If Windows resets the AMD driver (VIDEO_ENGINE_TIMEOUT_DETECTED, flicker, the engine dies mid-answer): "
                "docs/TROUBLESHOOTING.md, \"Windows AMD: the driver resets\"")
        if gpu.get("uma"):
            for line in igpu_notes(gpu, ram_gb()):
                (warn if line.startswith("!") else say)(line.lstrip("!"))
        for line in unused_apu_notes(chosen, amd, ram_gb()):
            (warn if line.startswith("!") else say)(line.lstrip("!"))
        for line in rdna2_notes(gpu):
            say(line)
    else:
        if not found:
            fail("no NVIDIA GPU found (nvidia-smi did not answer)",
                 "install the NVIDIA driver from https://www.nvidia.com/drivers and restart the PC"
                 + (f"; AMD ({', '.join(AMD_ARCHS)}): --backend hip" if amd else ""))
        if len(found) > 1 or gpu_problem(found[0]) is not None:
            gpu_table(found)
        sel = choose_gpus(a, found)                    # asked when two or more cards can share the model
        multi = sel if len(sel) > 1 else []
        a.gpu = sel[-1] if getattr(a, "_roles", None) else sel[0]   # the main GPU: the checks and the sizing below are its (roles: Main is last)
        GPU_PICK = a.gpu
        gpu = gpu_info(a.gpu)
        chosen = [gpu_info(i) for i in sel]
        gpu["archs"] = sorted({x["arch"] for x in chosen})  # the engine needs code for every one of them
        if multi:
            ok("GPUs: " + " + ".join(gpu_name(x) for x in chosen) + " together (the model's layers are split across them)")
        ok(f"GPU: {gpu['name']}, " + (f"unified memory ({gpu['vram_gb']:.0f} GB usable by the GPU)" if gpu.get("uma") else f"{gpu['vram_gb']:.1f} GB VRAM")
           + f", compute capability {cc(gpu)}, driver {gpu['driver']}")
        cuda_tk, why = cuda_choice(gpu["archs"], a.cuda, driver_major(gpu))   # one engine per model: its oldest card decides
        if why:
            (warn if cuda_tk == 13 or str(a.cuda) == "12" else ok)(f"CUDA {cuda_tk}: {why}")
        min_driver = CUDA12_COMPAT_DRIVER if cuda_tk == 12 else MIN_DRIVER
        if driver_major(gpu) < min_driver:
            fail(f"the NVIDIA driver is too old ({gpu['driver']}; {min_driver} or newer is needed)"
                 + ("" if cuda_tk == 12 else f" (the CUDA 12 engine package needs {cuda12_min_driver()[1]} or newer, {cuda12_min_driver()[2]})"),
                 "update it with the NVIDIA App or from https://www.nvidia.com/drivers, restart, and run this again")
        if cuda_tk == 12:
            cuda12_min_driver(feed=True)                   # the CUDA 12 package's driver floor: the installed package, else the signed feed (#4247)
            if note := cuda12_driver_warning(driver_major(gpu)):
                warn(note)
    if gpu["vram_gb"] < 11:
        warn("less than 12 GB of " + ("GPU memory (this APU's carve-out + shared memory)" if gpu.get("uma") else "VRAM")
             + ": Strata will run, but most experts stay on the CPU and it will be slow")
    ram = ram_gb()
    cpu, avx2, avx512 = cpu_info()
    need = min(d["ram_gb"] for d in MODELS.values() if not d.get("mem_total"))   # Gyro counts VRAM + RAM: not a RAM floor
    low_ok = low_ram_fits("IQ1_M", ram, low_ram_vram(gpu)) and a.low_ram != "off"   # the smallest model, mapped
    if ram < need - 4 and not a.check and not low_ok:
        # every model keeps ALL its experts in RAM (23+ GB); VRAM only holds a copy of the most-used ones, so a
        # bigger GPU does not lower this.  The owner's rule: a stop by default, a risk the user can take (--model
        # with --yes, or y)
        confirm_risk(f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB: Strata keeps all of "
                     "the model's experts in RAM (23-50 GB, whatever the GPU), so the OS will page them from disk. "
                     "Expect it to be very slow, and it may not start at all.", bool(a.model), a.yes,
                     f"RAM: {ram:.0f} GB - the smallest model (the Coder) needs about {need} GB",
                     "Strata keeps all of the model's experts in RAM (23-50 GB, whatever the GPU) and the GPU holds a "
                     "copy of the most-used ones: it needs 32 GB of RAM or more (48 GB for the full model); --model "
                     "NAME --yes installs one anyway")
        warn(f"going on with {ram:.0f} GB of RAM, as you chose")
    ram_msg = (f"RAM: {ram:.0f} GB" if ram >= need - 4 else f"RAM: {ram:.0f} GB (less than the {need} GB the smallest model needs)"
               + ("; the GPU's VRAM makes up for it (the low-RAM mode)" if ram < need - 4 and low_ok else ""))
    (ok if ram >= need - 4 or low_ok else warn)(ram_msg)       # #977: a RAM below every model's floor is not [ok]
    pf = page_file_gb()
    if pf is not None and pf < 4:
        warn(page_file_advice(pf, page_file_setting()))
    ok(f"CPU: {cpu} ({'ARM64' if is_arm64() else 'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    # (a unified-memory GPU - a DGX Spark's GB10 - is on the CPU's own link, not PCIe: nvidia-smi's "1.0 x1" there means nothing)
    link = None if hip or gpu.get("uma") else pcie_link(int(gpu.get("index", 0)))
    if link is not None:
        line, problem = pcie_lines(link)
        ok(line)
        if problem:
            warn(problem)
    floor = cpu_floor(avx2)
    if floor == "unsupported":
        fail("this CPU has neither AVX2 nor SSE4.2; Strata needs at least SSE4.2 (Intel Nehalem, 2008, or newer)")
    if is_arm64():
        # DGX Spark / GB10 (#3622): the engine is the linux-aarch64 package of the signed feed when the release has one, else compiled here
        # (spark/build_engine.sh). A warning, not a stop.
        warn("ARM64 CPU (a DGX Spark): the CPU's share of the experts runs on ggml-cpu's NEON kernels. Tested: Qwen3.8-Flash-Next IQ3_S and "
             "the NVFP4 pack, and DeepSeek-V4.1 2.00 bpw, on one DGX Spark (GB10). The Q2_0 and Gyro packs need the x86 kernels and do not "
             "run on this CPU")
    elif not avx2:
        # #394 #595 #623: the ready-made engine is AVX2; an older CPU gets one compiled here, whose CPU experts run on
        # ggml-cpu's kernels for this CPU.  Experimental: measured only on newer CPUs with the older path forced, and by
        # users on a few Xeons.  A warning, not a stop.
        warn(f"this CPU has no AVX2: Strata support for it is EXPERIMENTAL and slow. Setup compiles the engine on this "
             f"PC for {'AVX' if floor == 'avx' else 'SSE4.2'} (STRATA_ISA_FLOOR={floor}; 10-20 minutes, once), and the "
             "CPU's share of the experts runs on ggml-cpu's kernels, a few times slower than on an AVX2 CPU. "
             "See \"Older CPUs\" in docs/INSTALL.md")
        if hip and WIN:
            fail("the older-CPU engine is compiled from source, and setup compiles the AMD engine on Linux only",
                 "use Linux for an AMD card on this CPU, or an NVIDIA card")
        a.build = True
    if a.check:
        say()
        any_fits = False
        for m, d in MODELS.items():
            have_gb = mem_have(m, ram, gpu["vram_gb"])
            verdict = "fits" if have_gb >= d["ram_gb"] else "tight" if have_gb >= d["ram_gb"] - 8 else "does not fit"
            if d.get("mem_total"):          # Gyro: VRAM + RAM together; the rest of the download is read from the SSD
                verdict += f" ({ram:.0f} GB RAM + {gpu['vram_gb']:.0f} GB VRAM)"
            if d.get("budget") and not unsloth_budget(m, ram, gpu):     # Phase 2: all of them in RAM
                verdict = ("EXPERIMENTAL, " if d.get("experimental") else "") + "fits (all of its experts in RAM)"
            elif d.get("budget"):
                verdict = (("EXPERIMENTAL, " if d.get("experimental") else "") +
                           f"fits with {resident_budget_gib(m, ram)} GiB of its experts in RAM, the rest "
                           "read from the SSD" if ram >= d["ram_gb"] else
                           f"slow here: {resident_budget_gib(m, ram)} GiB of its "
                           f"{int(d['arena_gb'] / 1.073741824)} GiB of experts in RAM, the rest read from the SSD "
                           "(setup asks first)" if ram >= UNSLOTH_MIN_RAM_GB else "does not fit")
            elif low_ram_needed(m, ram) and low_ram_fits(m, ram, low_ram_vram(gpu)) and a.low_ram != "off":
                verdict = (f"fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, low_ram_vram(gpu)):.0f}% "
                           "of its experts, " + ("the rest stays in RAM)" if low_ram_resident(m, ram, low_ram_vram(gpu))
                                                 else "the rest is read from the SSD as needed)"))
            if d.get("budget") and hip and d.get("nvidia_only"):   # #429: not run on AMD yet
                verdict += " - NVIDIA only so far, untested on AMD"
            if hip and d.get("repo"):                  # the ready-made NVFP4 packs: no AMD kernels at all
                verdict = "NVIDIA only (the AMD engine has no NVFP4 kernels)"
            else:
                any_fits = any_fits or (not verdict.startswith("does not fit") and not d.get("mem_total"))
            say(f"  {m:8s} needs ~{d['ram_gb']} GB {'of VRAM + RAM' if d.get('mem_total') else 'RAM'}: {verdict}")
        try:                                           # DeepSeek-V4.1-Flash: a line, from the real numbers (strata_control/dsv41_fit.py)
            from strata_control import dsv41_fit
            cards_ok = amd_ok if hip else [g for g in found if gpu_problem(g) is None]
            for mname in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-Q8_K_XL"):
                ds = dsv41_fit.plan(cards_ok, ram, None, model=mname)
                tail = {"fits": f"fits ({ds['experts_in_vram']:,} of {ds['experts_total']:,} experts in graphics memory, the rest on the CPU)",
                        "tight": ("runs, with the experts RAM cannot hold read from the SSD (about 5-7 tokens a second measured)"
                                  if ds.get("variant") else "fits, but RAM is tight"),
                        "spills": (f"works but slow: less RAM than it was measured with (46-64 GB)" if ds.get("variant")
                                   else f"about {ds['spill_gb']:.0f} GB of its experts would be read from the SSD (slow)"),
                        "no_card": "needs one or more graphics cards"}.get(ds["verdict"], "not checked")
                say(f"  DeepSeek-V4.1-Flash {mname}: a {ds['download_gb']:.0f} GB download: {tail}   (--family deepseek41 --model {mname})")
        except Exception:                              # noqa: BLE001 - a line for information: never break the check
            pass
        if not any_fits:
            # #977: every size says "does not fit", so the verdict says so too (and the exit code, for scripts). It only
            # reports: --model NAME --yes still installs one anyway.
            say(f"\nThis PC cannot run Strata yet: no model size fits {ram:.0f} GB of RAM (the smallest needs about "
                f"{need} GB). --model NAME --yes installs one anyway, slowly.")
            return 1
        say("\nThis PC can run Strata. Run it again without --check to install.")
        return 0

    # ---- 2. the questions
    step(2, "your choices")
    fams = menu_families(hip)
    if a.family:
        family = a.family
    else:
        tier = tier_pick(ram, low_ram_vram(gpu))
        rec_fam = STRIX_HALO_FAMILY if strix_halo_recommends(gpu, ram) and STRIX_HALO_FAMILY in fams else             tier[0] if tier and tier[0] in fams else fams[0]
        for i, f in enumerate(fams, 1):
            d = FAMILIES[f]
            say(f"  {i}) {d['title']:20s} {d['by']} - {d['about']}" + ("   [experimental]" if d.get("experimental") else "")
                + family_menu_note(f, hip, gpu, avx2, avx512)
                + (f"   (recommended for Strix Halo: {STRIX_HALO_MODEL})" if f == rec_fam and rec_fam != fams[0] else ""))
        for j, f in enumerate([*OWN_PATH_FAMILIES, *MULTI_FILE_FAMILIES], len(fams) + 1):   # families with an engine path of their own
            d = {**OWN_PATH_FAMILIES, **MULTI_FILE_FAMILIES}[f]
            say(f"  {j}) {d['title']:20s} {d['by']} - {d['about']}   [new]")
        allf = fams + list(OWN_PATH_FAMILIES) + list(MULTI_FILE_FAMILIES)
        if tier_note(ram, low_ram_vram(gpu)) and rec_fam == "swift":
            say("  " + tier_note(ram, low_ram_vram(gpu)))
        family = allf[int(ask("Which model?", [str(i) for i in range(1, len(allf) + 1)],
                              str(allf.index(rec_fam) + 1), a.yes)) - 1]
    if family in OWN_PATH_FAMILIES:                    # Qwen3.6: its own engine path, its own steps 2-7
        return install_own_path(a, family, gpu, hip, amd, cuda_tk, multi, port, ram, adopted)
    if family in MULTI_FILE_FAMILIES:                  # DeepSeek-V4.1-Flash: nine shards, its own engine, several cards
        all_cards = (None if (WIN or getattr(a, "_cards_named", False)) else dsv41_auto_cards(amd, ROOT / "engine")) if hip else [g for g in found if gpu_problem(g) is None]
        gpu = {**gpu, "cards": chosen}
        return install_dsv41(a, family, gpu, hip, amd, cuda_tk, multi, port, ram, adopted, all_cards)
    fam = FAMILIES[family]
    if fam.get("ready") and hip:
        # not a risk the user could take: the HIP engine has no NVFP4 expert kernels at all, so it would not start
        fail(f"{fam['title']} runs on NVIDIA cards only: the AMD engine has no NVFP4 kernels",
             "UD-IQ4_XS is about the same size and runs on AMD: --family unsloth --model UD-IQ4_XS")
    if fam.get("ready") and a.gguf_dir:
        warn(f"--gguf-dir does not apply to {fam['title']}: its files are ready-made packs, not GGUF shards")
        a.gguf_dir = None
    ok(f"model: {fam['title']}")
    if fam.get("ready") and nvfp4_cpu_note(avx2, avx512):   # #2347: a note (the AVX2 rows work), not a refusal
        warn(nvfp4_cpu_note(avx2, avx512))
    if fam.get("license"):
        say(f"  Its license: {fam['license']}")
    say()
    names = [m for m in MODELS if family in MODELS[m].get("families", ("qwen", "swift"))]
    names.sort(key=lambda m: bool(MODELS[m].get("experimental")))   # an experimental size last, never the default
    if a.model and a.model not in names:
        # #444: say which family has that size, and (with --gguf-dir) which files Strata can run at all
        elsewhere_fams = [f for f in FAMILIES if f in MODELS[a.model].get("families", ("qwen", "swift"))]
        fail(f"{fam['title']} has no {a.model} model file", "choose one of: " + ", ".join(names)
             + (f" (or {a.model}: " + ", ".join(f"--family {f} --model {a.model}" for f in elsewhere_fams) + ")"
                if elsewhere_fams else "")
             + (f".\n       {SUPPORTED_GGUFS}" if a.gguf_dir else ""))
    installed = installed_size(family, names, have)
    for i, m in enumerate(names, 1):
        d = MODELS[m]
        fit = "" if ram >= d["ram_gb"] else f"   <- needs {d['ram_gb']} GB RAM, you have {ram:.0f}"
        if d.get("mem_total"):             # Gyro: the download includes the n-gram table, the memory figure does not
            have_gb = mem_have(m, ram, gpu["vram_gb"])
            fit = "" if have_gb >= d["ram_gb"] else \
                f"   <- needs {d['ram_gb']} GB of VRAM + RAM together, you have {have_gb:.0f}"
            say(f"  {i}) {m}{size_tag(family, m, installed)} {d['about']}; download {d['download_gb']:.1f} GB "
                f"(the n-gram table is in the file), memory {d['ram_gb']} GB{fit}")
            continue
        if d.get("budget"):
            keeps = (f"~{resident_budget_gib(m, ram)} GB" if unsloth_budget(m, ram, gpu, a.resident_budget_gib)
                     else "all")
            say(f"  {i}) {m}{size_tag(family, m, installed)} {d['about']}; download {d['download_gb']:.0f} GB, keeps {keeps} of its "
                f"{d['arena_gb']:.0f} GB of experts in RAM{fit}")
            continue
        if low_ram_needed(m, ram) and low_ram_fits(m, ram, low_ram_vram(gpu)) and a.low_ram != "off":
            fit = (f"   <- fits in the low-RAM mode (the GPU holds ~{100 * low_ram_gpu_share(m, low_ram_vram(gpu)):.0f}%, "
                   + ("the rest in RAM)" if low_ram_resident(m, ram, low_ram_vram(gpu)) else "the rest from the SSD)"))
        say(f"  {i}) {m:8s}{size_tag(family, m, installed)} {d['about']}; download {d['download_gb']:.0f} GB, uses ~{d['arena_gb']:.0f} GB of RAM{fit}")
    rec = str(names.index(installed or recommended_size(family, names, ram, low_ram_vram(gpu))) + 1)
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)], rec, a.yes)) - 1]
    budget, q4_split = None, False
    in_budget = bool(MODELS[model].get("budget")) and unsloth_budget(model, ram, gpu, a.resident_budget_gib)
    if MODELS[model].get("budget"):
        # Unsloth's files: every expert in RAM where they fit with UNSLOTH_RESIDENT_BESIDE_GB beside them (Phase 2,
        # unsloth_budget), as ISTA's.  Else a RAM budget of experts, the rest from the GGUF on the SSD - not the low-RAM
        # mode (no experts.bin: it would be another 77 GB on the disk), and one GPU (the budget mode had no layer split)
        # unless the RAM holds the GGUFs and 24 GB more: then several, without the budget, if asked for (#498).  #642:
        # from RESIDENT_SPLIT_ENGINE several GPUs, when asked for, keep the budget (unsloth_together)
        if MODELS[model].get("experimental"):
            warn(f"{model} is EXPERIMENTAL (docs/UNSLOTH_Q4.md)" + (
                 ": most of its experts are read from the SSD while it answers, so it is several times slower than the "
                 "2-3-bit models" if in_budget else "") + "; quality checked against llama.cpp")
        if hip and MODELS[model].get("nvidia_only"):
            # #429 (jkuepker): checked before the 111 GB download.  The HIP engine has no prompt kernels for its
            # Q4_K / Q5_K experts (STRATA_MMQ_KQUANTS is CUDA-only) and it has not been run on AMD: asked, not refused
            confirm_risk(f"{model} has not been run on AMD cards yet: its prompt kernels are NVIDIA-only, so on "
                         f"{gpu_name(gpu)} long prompts read much more slowly, and it may not work at all",
                         bool(a.model), a.yes, f"{model} is NVIDIA-only so far", "choose one of the 2-3-bit models, "
                         f"or --model {model} --yes to try it on AMD anyway", "  Try it anyway?")
            warn(f"installing {model} on an AMD card, as you chose (please report how it runs)")
        if ram < MODELS[model]["ram_gb"]:
            have = resident_budget_gib(model, ram)
            total = int(MODELS[model]["arena_gb"] / 1.073741824)
            confirm_risk(f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB: "
                         f"it would keep {have} GiB of its {total} GiB of experts in RAM and read the other "
                         f"{total - have} GiB from the SSD while it answers (single-digit tokens/s on most SSDs), and "
                         "it may run out of RAM", bool(a.model), a.yes,
                         f"{model} needs {MODELS[model]['ram_gb']} GB of RAM or more; this PC has {ram:.0f} GB",
                         f"choose a smaller size, or --model {model} --yes to install it anyway", "  Install it anyway?")
            warn(f"installing {model} with {ram:.0f} GB of RAM, as you chose")
        if in_budget:
            budget = budget_choice(model, ram, a.resident_budget_gib)
            if multi and not unsloth_together(a, model, ram, fallback_card(a, gpu, chosen), chosen):
                gpu = fallback_card(a, gpu, chosen)
                multi, sel, chosen = [], [gpu["index"]], [gpu]
            q4_split = bool(multi) and not resident_split()   # #498: on several GPUs without the RAM budget (#642: kept)
            if not q4_split:
                ok(f"RAM budget: {budget:g} GiB of {model}'s experts in RAM, the rest read from the model files on the SSD")
        else:
            ok(f"all {MODELS[model]['arena_gb']:.0f} GB of {model}'s experts in RAM (they and "
               f"{UNSLOTH_RESIDENT_BESIDE_GB} GiB more fit this PC's {ram:.0f} GB; --resident-budget-gib N keeps only "
               "N GiB of them)")
        if a.low_ram not in ("auto", "off"):
            warn(f"--low-ram {a.low_ram} does not apply to {model}: it always reads part of its experts from the files")
    elif a.resident_budget_gib is not None:
        warn(f"--resident-budget-gib is for UD-Q4_K_XL and UD-IQ4_XS: {model} keeps all of its experts in RAM or in "
             "the low-RAM mode")
    low_ram = low_ram_wanted(model, ram, a.low_ram)
    # #642: the engines from RESIDENT_SPLIT_ENGINE run the low-RAM mode on the chosen cards as on one (decided below)
    if low_ram and multi and not resident_split() and not low_ram_together(a, model, ram, fallback_card(a, gpu, chosen), chosen):
        gpu = fallback_card(a, gpu, chosen)
        multi, sel, chosen = [], [gpu["index"]], [gpu]
    # (the low-RAM mode's variant is decided once the context is known, below; on several GPUs it is the mapped one
    # before RESIDENT_SPLIT_ENGINE, #642)
    if MODELS[model].get("mem_total"):
        if mem_have(model, ram, gpu["vram_gb"]) < MODELS[model]["ram_gb"]:   # recommend, never force: a warning
            warn(f"{model} needs {MODELS[model]['ram_gb']} GB of VRAM and RAM together; this PC has "
                 f"{mem_have(model, ram, gpu['vram_gb']):.0f} GB ({ram:.0f} GB RAM + {gpu['vram_gb']:.0f} GB VRAM): it "
                 "may be slow, run out of memory or not start")
    elif not low_ram and budget is None and ram < MODELS[model]["ram_gb"] - 4:
        confirm_paging(model, ram, a.low_ram, a.yes, bool(a.model))
    ok(f"size: {model}")
    if MODELS[model].get("license"):                   # the ready-made packs: each size has its own license
        say(f"  Its license: {MODELS[model]['license']}")
    tag = fam["tag"] + model                           # names of the pack, config and start script
    small = min(x["vram_gb"] for x in chosen)         # each card keeps its layers' KV of the whole context
    rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072
    if budget is not None:                             # UD-Q4_K_XL: every GB of KV is a GB fewer of cached experts
        rec_ctx = 8192 if small < 14 else 32768
    # #406: the RAM rule is part of the recommendation (the smaller of the two), no longer a cap over the user's choice
    rec_ctx = min(rec_ctx, ram_ctx(model, ram, low_ram))
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context length = how much text the model can see at once (your chat, files, tool output).")
        say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
        for i, c in enumerate(CONTEXTS, 1):
            need_c = ctx_ram_need(model, c, low_ram)
            note = ("   (recommended for your GPU)" if c == rec_ctx else "") + \
                   ("   (experimental: setup adds rope scaling)" if c > 262144 else "") + \
                   (f"   (needs ~{need_c:.0f} GB RAM, this PC has {ram:.0f}: may run out of memory)"
                    if c > 131072 and need_c is not None and need_c > ram else "")
            say(f"  {i}) {c // 1024}K tokens{note}")
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    # #406 #364: a context past the RAM rule (an explicit --context, a pick in the list, or the earlier install's) is
    # kept, with what it risks.  It used to become 128K: users ran 256K fine where setup's estimate said no.
    need_gb = ctx_ram_need(model, ctx, low_ram)
    if need_gb is not None and ram < need_gb and ctx > 131072:
        warn(f"{ctx // 1024}K with {model} needs ~{need_gb:.0f} GB of RAM by setup's estimate "
             f"({MODELS[model]['arena_gb']:.0f} GB of experts + the context + room for the rest); this PC has "
             f"{ram:.0f}. Kept as you chose: it may be slower or run out of RAM under load. {rec_ctx // 1024}K is the "
             "recommended size.")
    scaling = a.rope_scaling
    if ctx > 262144 and scaling is None and not a.yes:
        # the interactive path: one question, yarn preselected (llama.cpp's extension method, recall-tested
        # here at 512K).  With --yes nothing prints: resolve_rope takes yarn below and the ok() line says so.
        say()
        say(f"  A {ctx // 1024}K context runs the model past its trained 262,144 positions: the rotary angles")
        say("  get rescaled (llama.cpp's RoPE extension). yarn keeps the trained angles on the high-frequency")
        say("  pairs and corrects the magnitudes; linear shrinks every angle. Override any time with")
        say("  --rope-scaling.")
        scaling = ask("RoPE extension method?", ["yarn", "linear"], "yarn", a.yes)
    try:
        scaling, rope_scale = resolve_rope(ctx, scaling, a.rope_scale)
    except ValueError as e:
        fail(str(e))
    if scaling is not None:
        origin = ("final context / trained 262144; override with --rope-scale" if a.rope_scale is None
                  else "as requested")
        ok(f"rope scaling: {scaling}, factor {rope_scale:g} ({origin})")
    ok(f"context: {ctx} tokens")
    # the KV cache (the model's memory of the conversation): 8-bit, or 4-bit after a Hadamard rotation (PR #21)
    kv = "fp16" if ctx <= 8192 else (a.kv or "int8")
    if ctx > 8192 and not a.kv and not a.yes:
        say()
        say("  KV cache precision (the model's memory of the conversation):")
        say("  1) 8-bit   (recommended: what every published number was measured with)")
        say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
        say("             documents; long-context lookups (needle tests) still pass")
        kv = ["int8", "q4_0"][int(ask("KV cache?", ["1", "2"], "1", a.yes)) - 1]
    if ctx > 8192:
        ok(f"KV cache: {KV_WORDS.get(kv, kv)}")
    if MODELS[model].get("vision", fam.get("vision")) is False:     # UD-IQ4_XS: images, unlike UD-Q4_K_XL
        vision = "none"
        if a.vision not in (None, "no", "none"):
            warn(f"images are not available with {model} yet: off")
    elif hip:
        # Unsloth's files on AMD: pictures on by default, the reader by the 24 GB rule (hip_vision; docs/QUANTS.md)
        vision = hip_vision(a.vision if a.vision is not None or not images_default(model, family, ram, gpu)
                            else "yes", float((gpu or {}).get("vram_gb") or 0.0))
    elif a.vision:
        vision = flash_vision_encoder(a.vision, gpu) if a.vision != "no" else "none"
        vision = "none" if vision in ("no", "none") else vision
    else:
        small = 0 < float(gpu.get("vram_gb") or 0.0) < IMAGES_GPU_ENCODER_MIN_VRAM_GB
        say()
        say("  Images: the model can also read pictures (screenshots, photos, scanned pages). This adds a 0.9 GB")
        if small:
            say("  download. On a card under 24 GB the encoder runs on the CPU (a few seconds per picture, nothing taken")
            say("  from the GPU, pictures read at up to 300 tokens; --vision-tokens 1024 for more detail): the GPU")
            say("  encoder keeps ~1.4 GB of VRAM free, which costs 3-5% of text speed on 16 GB cards (measured) and")
            say("  more on some PCs. --vision gpu if you want pictures in under a second.")
        else:
            say("  download and keeps ~1.4 GB of VRAM free for the image encoder, so text is a few % slower.")
        # on by default wherever the model can read pictures (the reader by the 24 GB rule)
        default = "y" if images_default(model, family, ram, gpu) else "n"
        vision = flash_vision_encoder("yes", gpu) if ask("Do you want images?", ["y", "n"], default, a.yes) == "y" else "none"
    ok("images: " + {"none": "off", "gpu": "on", "cpu": "on (encoder on the CPU)"}[vision])
    if vision != "none" and MODELS[model].get("vision_untested"):
        warn(f"images with {model} are untested: users report them working, but we have not run this file with "
             "images (#967); tell us if the answers look wrong")
    # The low-RAM mode's two variants.  resident: the experts the GPU's cache does not hold (and, as far as RAM allows,
    # the ones the prompt path borrows cache room from) are copied from the pack's experts.bin into RAM once, so
    # nothing is read from the SSD while it answers (engine 0.1.30, --resident-experts; the engine falls back to mmap
    # with a warning when they do not fit the RAM it finds free).  mmap: they are read through the OS file cache.
    # The GPU's share: its VRAM less the dense weights and buffers, this context's KV cache and the image encoder's room.
    resident = False
    if low_ram and not fam.get("ready"):               # NVFP4's packs: the RAM-budget mode instead (step 7)
        arena = MODELS[model]["arena_gb"]
        vram = low_ram_vram(gpu) - (VISION[vision]["reserve_mib"] / 1024 if vision != "none" else 0)
        with_mtp = a.mtp != "off"                       # the draft layer's ~1 GB comes out of the expert cache
        share = low_ram_gpu_share(model, vram, ctx, kv, with_mtp)
        rest = arena - low_ram_gpu_gb(model, vram, ctx, kv, with_mtp)
        resident = a.low_ram == "resident" or (a.low_ram != "mmap" and low_ram_resident(model, ram, vram, ctx, kv, with_mtp))
        if multi:      # every chosen card's share (the image encoder on the main one); #364 #384: the mapped variant
            held = min(arena, low_ram_gpu_gb(model, vram, ctx, kv, with_mtp) +
                       sum(low_ram_gpu_gb(model, low_ram_vram(x), ctx, kv, with_mtp) for x in chosen[1:]))
            share, rest = held / arena, arena - held
            # #642: the resident variant on a split, by the same rule as on one card - what no card holds fits the RAM
            resident = resident_split() and (a.low_ram == "resident" or
                                             (a.low_ram != "mmap" and ram >= rest + LOW_RAM_HEADROOM_GB))
            if resident:
                ok(f"low-RAM mode on {len(chosen)} GPUs: they hold ~{100 * share:.0f}% of {model}'s experts "
                   f"({arena:.0f} GB) and the other ~{rest:.0f} GB stay in RAM ({ram:.0f} GB), read once from a copy "
                   "in the model folder")
            else:
                ok(f"low-RAM mode on {len(chosen)} GPUs: {model}'s experts ({arena:.0f} GB) are read from the model "
                   f"folder through the OS file cache instead of a copy in RAM ({ram:.0f} GB); the GPUs hold "
                   f"~{100 * share:.0f}% of them")
                if share < 0.6:
                    warn("most of the experts are read from the SSD while it answers: expect it to be much slower "
                         "than with enough RAM (a faster SSD and a smaller size help)")
        elif resident:
            ok(f"low-RAM mode: the GPU holds ~{100 * share:.0f}% of {model}'s experts ({arena:.0f} GB) and the other "
               f"~{rest:.0f} GB stay in RAM ({ram:.0f} GB), read once from a copy in the model folder")
        else:
            ok(f"low-RAM mode: {model}'s experts ({arena:.0f} GB) are read from the model folder through the OS file "
               f"cache instead of a copy in RAM ({ram:.0f} GB); the GPU holds ~{100 * share:.0f}% of them")
            if share < 0.6:
                warn("most of the experts are read from the SSD while it answers: expect it to be much slower than "
                     "with enough RAM (a faster SSD and a smaller size help)")
    # EXPERIMENTAL: the experimental-speed-projection control vector (data/experimental-speed-projection), off unless
    # chosen here; with it loaded, the web app and the API switch it off per request
    esp = None
    esp_choice = (a.experimental_speed_projection or "").strip()
    if family in ("qwen", "coder"):                   # the Coder: the same model's residual stream
        if not esp_choice:
            say()
            say("  EXPERIMENTAL - speed projection: a small control vector applied while the model runs (layers 4-44).")
            say("  It changes how the model answers: its package describes it as a refusal-direction projection (the")
            say("  model declines far fewer requests). Off unless you choose it; when on, the web app can switch it off")
            say("  per chat. Details: data/experimental-speed-projection/README.md")
            esp_choice = "on" if ask("Turn on the experimental speed projection?", ["y", "n"], "n", a.yes) == "y" else "off"
        if esp_choice.lower() not in ("off", "no", "n", "0"):
            esp = ESP_VECTOR if esp_choice.lower() in ("on", "yes", "y", "1") else Path(esp_choice).expanduser().resolve()
            if not esp.is_file():
                fail(f"the experimental speed projection's vector is missing: {esp}")
        ok("experimental speed projection: " + ("ON (experimental)" if esp else "off"))
    elif esp_choice.lower() not in ("", "off", "no", "n", "0"):
        warn("the experimental speed projection is made for the original Qwen3.8-Flash-Next, not Swift 1.5: left off"
             if family == "swift" else f"the experimental speed projection is not tested with {model}: left off")
    ready = bool(fam.get("ready"))                     # NVFP4: ready-made packs (NVFP4_REPOS), no GGUF shards
    if ready:
        models_dir = Path(a.models_dir)
        for r in elsewhere:                            # already downloaded and checked in a Strata folder elsewhere
            if all(ready_verified(ready_where(r / "models", f), f[5]) for f in ready_needed(model, r / "models")):
                models_dir = r / "models"
                ok(f"model files found in {models_dir}")
                break
        shards = [ready_where(models_dir, f) for f in ready_files(model)]
    else:
        models_dir = Path(a.gguf_dir) if a.gguf_dir else Path(a.models_dir) / tag
        shards = gguf_dir_shards(models_dir, fam, model) if a.gguf_dir else \
            [models_dir / model_file(fam, model, i) for i in range(1, model_shards(fam, model) + 1)]
    problem = gguf_dir_problem(models_dir, shards[0], fam, model) if a.gguf_dir else None
    if problem:                                        # #444: files Strata cannot run, or another choice's files
        fail(*problem)
    if not ready and not a.gguf_dir and not all(sh.exists() and done(sh) for sh in shards):
        for r in elsewhere:                            # already downloaded in a Strata folder on another drive
            cand = [r / "models" / tag / sh.name for sh in shards]
            if all(c.exists() and done(c) for c in cand):
                models_dir, shards = cand[0].parent, cand
                ok(f"model files found in {models_dir}")
                break
    for s in ([] if ready else shards):                # #173: a whole file copied in by hand has no finish mark
        if s.exists() and not done(s) and whole_shard(s):
            try:
                mark(s, "whole (checked against its own tensor directory)")
            except OSError as e:                       # a read-only folder (--gguf-dir on a share): the file is still whole
                warn(f"{s.name} is whole but its finish mark cannot be written ({e})")
    have_model = all(ready_verified(ready_where(models_dir, f), f[5]) for f in ready_needed(model, models_dir)) if ready else \
        all(s.exists() and (done(s) or a.gguf_dir) for s in shards)
    # #425 (jctaborda): a download that resumes needs room only for what is still missing - the finished shards and
    # the .part files already on the disk count
    on_disk = sum(f.stat().st_size for s in shards for f in (s, s.with_name(s.name + ".part")) if f.is_file()) / 1e9
    to_fetch = 0 if a.gguf_dir or have_model else max(MODELS[model]["download_gb"] - on_disk, 0)
    if ready and not have_model and MODELS[model].get("prepare") and prepared_done(model, models_dir):
        to_fetch = sum(f[4] for f in ready_needed(model, models_dir)
                       if not ready_verified(ready_where(models_dir, f), f[5])) / 1e9   # the small files only
    # count only what step 6 will still write: a pack whose experts.bin is already there (the AVX-512 Q2_0
    # conversion, or the low-RAM mode's copy) and an existing MTP draft layer need no new room
    pack_now = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    pack_bin = ready or ((pack_now / "experts.bin").exists() and (pack_now / "index.txt").exists())   # ready: in the download
    mtp_have = find_in(roots, "mtp/rt/experts.bin") is not None
    q2_avx = model == "Q2_0" and avx512 and family == "qwen"
    prep_gb = MODELS[model].get("prepare_gb", 0.0) if ready and MODELS[model].get("prepare") and \
        not prepared_done(model, models_dir) else 0.0
    margin = 2 if mtp_have else 8                      # working room: a guess, not bytes that will be written
    mtp_fetch = 5.2 if (not mtp_have and not ready and a.mtp != "off") else 0.0      # the draft layer's tensors step 6 fetches (strata_control.discovery.MTP_FETCH_GB)
    real = to_fetch + prep_gb + mtp_fetch + \
        (40 if q2_avx and not pack_bin else 0) + (1 if vision != "none" else 0) + \
        (MODELS[model]["arena_gb"] + 1 if low_ram and not q2_avx and not pack_bin else 0)
    need = real + margin
    free_now = free_gb(models_dir)
    if free_now < real:
        fail(f"not enough free disk space in {models_dir}: need ~{real:.0f} GB, {free_now:.0f} GB free" +
             (f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1 and not have_model else ""),
             ("use --data-dir on a bigger drive (strata launch --data-dir)" if CONTROL else "use --models-dir on a bigger drive"))
    if free_now < need:
        warn(f"little free disk space in {models_dir}: {free_now:.0f} GB free for about {real:.0f} GB of files. "
             "Setup goes on and should fit; free up space if it stops")

    # ---- 3. python packages
    step(3, "Python packages")
    pip_install(runtime_packages(),
                "numpy, jinja2, regex, pyyaml, tqdm, requests, pillow, psutil, textual (the screens)")

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama, eng, meta, lib_dirs, gpu, vision = engine_step(a, gpu, hip, amd, cuda_tk, vision)
    reader = None
    if hip and vision != "none":                       # AMD: the engine folder's reader (GPU, else CPU), settled now the engine is known
        vision, reader = hip_reader(eng, vision, gpu)
    if (ready or family == "gyro") and meta.get("source") != "local" and \
            tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit()) < MODELS[model]["engine"]:
        # a ready-made engine from before NVFP4 / Gyro cannot read these experts: one compiled from this source can
        warn(f"the ready-made engine {meta.get('version')} has no {'NVFP4' if ready else 'Gyro'} experts yet: compiling "
             "the engine from this source")
        eng = build_engine(gpu, vision, a.yes, llama, toolkit=cuda_tk)
        meta = json.loads((eng / "BUILD.json").read_text(encoding="utf-8"))
        lib_dirs = meta.get("lib_dirs") or meta.get("cuda_dirs") or cuda_lib_dirs(cuda_tk)
    engine_ver = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
    need_engine = MODELS[model].get("engine", UNSLOTH_ENGINE)
    # checked before the 94-188 GB download
    if (MODELS[model].get("budget") or family == "gyro") and engine_ver < need_engine:
        fail(f"{model} needs engine {'.'.join(map(str, need_engine))} or newer; this one is {meta.get('version')}",
             "update Strata (or compile the engine with --build) and run setup again")
    ok("engine: the engine component (looked up when the model loads)" if CONTROL else f"engine: {eng / EXE}")

    # ---- 5. the model files
    step(5, f"downloading {fam['title']} {model}")
    if ready:                                          # pinned files of the pack's repository, each SHA-256-checked
        say(f"  The model files go in {models_dir / MODELS[model]['repo'].replace('/', '--')}")
        if model_source() == "modelscope":
            warn("these packs are on Hugging Face only: downloading them from " + hf_endpoint())
        ready_fetch(model, models_dir)
    elif not a.gguf_dir:
        missing = [s.name for s in shards if not (s.exists() and done(s))]
        if missing:                                    # #495: files downloaded by hand go here, or --gguf-dir
            say(f"  The model files go in {models_dir}")
            say(f"  Files you already have: put them here with their original names ({', '.join(missing)}), or use "
                "--gguf-dir <their folder>.")
            if model_source() == "modelscope":
                say(f"  Downloading from ModelScope ({ms_endpoint()}); --source huggingface downloads from Hugging Face")
                warn("ModelScope serves the repositories' current files, not the pinned revisions: each file is checked "
                     "against the SHA-256 ModelScope itself publishes (self-attested); the MTP tensors against the "
                     "pinned checkpoint's own hashes")
            elif hf_endpoint() != HF_DEFAULT:
                say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
        for s in shards:
            if s.exists() and done(s):
                ok(f"{s.name} already downloaded")
                continue
            # the original's shard 2 is the same file for all its sizes and the Coder: reuse one that is already here
            other = [p for p in Path(a.models_dir).glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf") if done(p)]
            if family in ("qwen", "coder") and s.name.endswith("00002-of-00002.gguf") and other and not s.exists():
                try:
                    os.link(other[0], s)
                    mark(s)
                    ok(f"{s.name} shared with {other[0].parent.name} (identical file)")
                    continue
                except OSError:
                    pass
            download(fam["hf"].format(q=model) + s.name, s)
    if not ready:
        check_shards(shards)
    for s in ([] if ready else shards):                # the Unsloth files: pinned sizes and SHA-256
        if s.name in fam.get("sha256", {}):
            verify_sha256(s, *fam["sha256"][s.name], own=bool(a.gguf_dir))
    ok("model files present")
    mmproj = Path(a.models_dir) / fam.get("mmproj", MMPROJ)
    if not mmproj.exists():                            # an earlier install's copy (the same file for every family)
        mmproj = find_in(roots, f"models/{fam['mmproj']}") or mmproj
    if vision != "none":
        if not mmproj.exists() and a.gguf_dir and (Path(a.gguf_dir) / fam["mmproj"]).exists():
            mmproj = Path(a.gguf_dir) / fam["mmproj"]
        else:
            download(fam["mmproj_hf"] + fam["mmproj"], mmproj, "vision encoder")
        if fam.get("mmproj_sha256"):                   # the NVFP4 family pins it by size and SHA-256 too
            verify_sha256(mmproj, *fam["mmproj_sha256"])
        ok(f"vision encoder: {mmproj}")
        if vision == "cpu" and "BF16" in mmproj.name:       # a tip only (recommend, never force; #625)
            say("       tip: on the CPU a Q8_0 copy of this encoder is a third smaller and about as exact (embedding "
                "cosine 0.999 vs BF16); see 'A Q8_0 encoder' in docs/DETAILS.md")

    # ---- 6. the pack and the MTP draft layer
    step(6, "preparing the model for Strata")
    pack = find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    prepared = bool(ready and MODELS[model].get("prepare"))
    if prepared and not prepared_done(model, models_dir):
        say("  NVIDIA's checkpoint is turned into a Strata pack on this PC (once; 20-40 minutes, reads the 102 GB, writes about")
        say(f"  {MODELS[model]['prepare_gb']:.0f} GB). Nothing of NVIDIA's weights leaves this PC.")
        ensure_nvfp4_local_packages()
        run([sys.executable, str(ROOT / "tools" / "nvfp4_local_pack.py"), "prepare",
             "--model", str(ready_where(models_dir, ("src", MODELS[model]["repo"], "", "", 0, ""))),
             "--out", str(prepared_dir(model, models_dir))], env=env)
        ok("prepared; the downloaded checkpoint is no longer needed and can be deleted to get its 102 GB back: "
           + str(ready_where(models_dir, ("src", MODELS[model]["repo"], "", "", 0, ""))))
    if ready:                                          # the downloaded pack folder is the pack, experts.bin included
        pack = ready_paths(model, models_dir)["pack"]
    elif model == "Q2_0" and avx512 and family == "qwen":
        # the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a one-time ~40 GB conversion
        if not (pack / "index.txt").exists() or not (pack / "experts.bin").exists():   # index.txt is written last
            say("  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...")
            run([sys.executable, str(ROOT / "tools" / "strata_pack.py"), "build", "--gguf", str(shards[0]),
                 "--out", str(pack), "--skip-hash", "--force"], env=env)   # --force: #634 refuses an occupied pack
            run([sys.executable, str(ROOT / "tools" / "pack_index.py"), "--pack", str(pack)], env=env)
        if not (pack / "tokenizer" / "vocab.json").exists():
            run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]),
                 "--out", str(pack)], env=env)   # writes <pack>/tokenizer/
    elif not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        # every tensor as the GGUF stores it; the experts are read from the GGUF at start (seconds to build)
        # (UD-Q4_K_XL: --compat-bf16 - its Q8_0 hyper-connection projections become BF16, the form the engine reads)
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             *fam.get("pack_args", [])], env=env)
    if low_ram and not ready and not (pack / "experts.bin").exists():
        say(f"  Writing the experts into one file for the low-RAM mode (one time, {MODELS[model]['arena_gb']:.0f} GB) ...")
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             "--experts-bin", *fam.get("pack_args", [])], env=env)       # #3193: Gyro needs --compat-bf16 here too
    ok(f"model prepared: {pack}")
    mtp = ready_paths(model, models_dir)["mtp"] if prepared else \
        (find_in(roots, "mtp/rt/experts.bin") or data / "mtp/rt/experts.bin").parent.parent
    rt = mtp / "rt"
    corrupt = (rt / "experts.bin").exists() and not prepared and mtp_corrupt(mtp, env)
    if corrupt:
        warn("some MTP tensors are not the checkpoint's (a download mirror that ignored range requests, #327): "
             "fetching them again and rebuilding the draft layer")
    if a.mtp == "off" and not (rt / "experts.bin").exists():
        ok("the draft layer (MTP) is off: its ~5 GB are not downloaded (`--mtp on` fetches it later)")   # round 8, N3
    elif corrupt or not (rt / "experts.bin").exists():
        say("  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:")
        say("  only its ~5 GB of MTP tensors are downloaded.")
        fetch_mtp(mtp, env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(rt)],
            env=env)
    # a setup run again without --draft-vocab keeps the subset this model's config chose before (cyrillic, fr, en)
    draft_vocab = a.draft_vocab or saved_draft_vocab(home() / f"strata-{tag.lower()}.json")
    if (rt / "experts.bin").exists():
        refresh_draft_vocab(rt, draft_vocab or "cjk")
        ok(f"MTP draft layer: {rt}")
    for line in draft_vocab_note(gpu.get("vram_gb", 0.0), draft_vocab):   # #474: a recommendation, nothing changes
        say("  " + line)

    # ---- 7. the start script
    step(7, "writing the start script")
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile                   # the PLE table's shard: shard 2 (original) or 1 (Swift)
    if ready:   # the dense GGUF as --native and --native-dense-gguf (--native alone would look for a second shard),
        rp = ready_paths(model, models_dir)            # the FP8 n-gram table, the BF16 token embedding
        ple = rp["ple"]
        model_args = ["--pack", str(pack), "--native", str(rp["dense"]), "--native-dense-gguf", str(rp["dense"]),
                      "--ple-gguf", str(ple), *(["--embd-gguf", str(rp["embd"])] if "embd" in rp else [])]
    else:
        ple = next((s for s in shards if any(t.name == "per_layer_token_embd.weight" for t in GGUFFile(s).tensors)), None)
        if ple is None:
            fail("the model has no per_layer_token_embd tensor (is this a Qwen3.8-Flash-Next GGUF?)")
        # (a 4-shard file: the engine finds the PLE table's shard itself from shard 1, the measured setup)
        model_args = ["--pack", str(pack), "--native", str(shards[0]),
                      *(["--ple-gguf", str(ple)] if len(shards) <= 2 else [])]
    args = [*model_args,
            "--expert-profile", str(ROOT / "data" / fam.get("profile", "expert-profile.bin")), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(rt),
            "--max-context", str(ctx)]
    if scaling is not None:     # the resolved config: explicit flags as given, or the automatic yarn+factor
        args += ["--rope-scaling", scaling, "--rope-scale", f"{rope_scale:g}"]
    if ctx > 8192:
        args += ["--kv", kv]
    if resident and a.low_ram != "resident" and engine_ver < RESIDENT_ENGINE:
        resident = False                               # an engine from before --resident-experts would refuse it
        ok(f"low-RAM mode: engine {meta.get('version')} has no resident variant yet; the experts are read through "
           "the OS file cache (run setup again after the next engine update)")
    if low_ram and ready:
        # NVFP4 (68 GB of experts) on a PC whose RAM does not hold them: the RAM-budget mode, as Unsloth's files - the
        # most-used experts outside VRAM pinned within RAM less 24 GB, the rest read from experts.bin.  Mapped
        # (--mmap-experts) on a 64 GB Windows PC, the experts it read stayed in the engine's working set: free RAM
        # fell to 0.6 GB on the RTX 5070 PC (2026-10-13), where the budget mode kept 22 GB free
        rb = resident_budget_gib(model, ram)
        args += ["--resident-budget-gib", str(rb)]
        ok(f"low-RAM mode: {rb} GiB of {model}'s experts kept in RAM (the most used), the rest read from the pack's "
           "experts.bin on the SSD as needed")
    elif low_ram:   # the experts from the pack's experts.bin: the ones the GPU does not hold copied into RAM, or mapped
        args += ["--resident-experts" if resident else "--mmap-experts"]
    elif is_gb10(gpu):
        # a DGX Spark: the RAM copy of the experts and the GPU's cache are the same memory. Mapped, every expert fits in the cache (NVFP4
        # on a 128 GB GB10: 24576 of 24576 slots, not 13702; chat decode 44.4 against 40.6 tok/s, 16K prompt 1471 against 1397)
        args += ["--mmap-experts"]
        ok("unified memory: the experts are read from the model files in place (--mmap-experts), so the GPU's cache holds them "
           "instead of a second copy in RAM")
        # the measured settings of this part (docs/DGX_SPARK_LINUX.md): the 32,768-token prompt chunk (auto's 8,192 measured 2-6% slower
        # here); the draft layer as 4-bit copies (+3.5% decode, 3 paired rounds, answers identical: the drafter's proposals are verified
        # by the target); one norm per hyper-connection stream in the draft layer (llama.cpp's graph: acceptance 0.60 -> 0.63, decode +4.3%
        # [1.009, 1.078], 5 of 5 greedy answers identical)
        if "--prefill" in args:
            args[args.index("--prefill") + 1] = "auto:32768"
        if "--mtp" in args:
            args += ["--mtp-q4", "all", "--mtp-hnorm", "stream"]
            ok("prompt chunk 32,768 (--prefill auto:32768), the draft layer as 4-bit copies (--mtp-q4 all) and one norm per "
               "hyper-connection stream in it (--mtp-hnorm stream): measured faster on this part, the answers are unchanged")
        # the language is the user's choice: the default subset keeps Chinese, Japanese and Korean drafts, the English one measured +3.1%
        # decode on English prompts here (16 samples a cell, 90% interval [1.018, 1.049])
        say("  Tip: for English or code only, --draft-vocab en measured +3.1% decode on this part (the default keeps Chinese, Japanese "
            "and Korean drafts).")
    disk = None if is_wsl() else rotational_disk(ple)  # #605 (WSL's virtual disk says rotational)
    if disk:
        tensor = None if ready else next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
        size = ple.stat().st_size if ready else getattr(tensor, "expected_bytes", lambda: None)()
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 4:
            args += ["--ple-io", "ram"]
            ok(f"the model is on a rotational disk ({disk}): its {table_gb:.0f} GB n-gram table is kept in RAM "
               "(--ple-io ram) - read from the disk at random, it can stall prompts for minutes (#605)")
        else:
            warn(f"the model is on a rotational disk ({disk}): its n-gram table is read from it at random, which can "
                 f"stall prompts for minutes (#605). An SSD is recommended; with ~{table_gb:.0f} GB more RAM, "
                 "--ple-io ram in the config's args keeps the table in RAM instead")
    elif not low_ram and ple_ram_platform_ok():
        # 147-verify: on an SSD too, a PC with RAM to spare keeps the n-gram table in RAM: a new prompt's first chunk then
        # does not wait for its table rows (65,536 random SSD reads at 4K tokens).  RTX PRO 5000 (Linux, NVMe, fresh engine
        # per prompt, 3 rounds, the same bits): prompt 4K 1298 -> 1059 ms (-18%), 20K 4644 -> 4223 ms (-9%), 98K 20447 ->
        # 20027 ms (-2%), decode +0.8%.  On a card whose prompt is slow (R9700) the read hides behind layer 0 and nothing
        # changes.  It costs the table's size in RAM, so only with 16 GB to spare beyond the model's own need; a smaller PC
        # keeps reading from the SSD as before.
        try:
            tensor = None if ready else next((t for t in GGUFFile(ple).tensors if t.name == "per_layer_token_embd.weight"), None)
            size = ple.stat().st_size if ready else getattr(tensor, "expected_bytes", lambda: None)()
        except Exception:
            size = None
        table_gb = size / 1e9 if size else 28.8
        if ram >= MODELS[model]["ram_gb"] + table_gb + 16:
            args += ["--ple-io", "ram"]
            ok(f"{ram:.0f} GB of RAM: the {table_gb:.0f} GB n-gram table is kept in RAM (--ple-io ram), so a new prompt does "
               "not wait for its rows from the SSD (a 4K prompt about 18% sooner on a fast card); remove --ple-io ram from "
               "the config's args to read it from the SSD instead")
    # KV streaming: from 64K up the whole KV cache lives in RAM and only the part the attention reads (32K positions
    # per layer) stays in VRAM; the VRAM it frees holds more experts (+6% at 128K, +23% at 262K with Q2_0). It
    # costs ~13.7 KB of RAM per context token with 8-bit KV (1.7 GB at 128K), 10.6 KB with K8V4, 7.5 KB with 4-bit, so only
    # when it fits.
    kv_ram_gb = kv_streaming_ram_gb(ctx, kv)      # the branches below are kv_streaming_wanted, with its messages
    # --kv-streaming on|off overrides the RAM test (the owner's rule); WSL stays off - it cannot stream.
    stream_fits = ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1
    if is_wsl() and ctx >= 65536:
        ok("WSL: KV streaming off (the driver pins only about 1 GB of RAM); the KV cache stays in VRAM")
        if a.kv_streaming == "on":
            warn("--kv-streaming on: WSL cannot stream the KV cache (its RAM copy must be pinned, and the driver pins "
                 "only about 1 GB there): off")
    elif ctx >= 65536 and a.kv_streaming == "off":
        ok("KV streaming off, as you chose (--kv-streaming off): the KV cache stays in VRAM")
    elif ctx >= 65536 and (stream_fits or a.kv_streaming == "on"):
        args += ["--kv-resident", "32768"]
        ok(f"KV streaming on: the context's KV cache lives in RAM ({kv_ram_gb:.1f} GB), more experts fit in VRAM")
        if not stream_fits:
            warn(f"KV streaming needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
                 f"uses; this PC has {ram:.0f}. Kept as you chose (--kv-streaming on): it may page or run out of RAM "
                 "under load")
        if q4_split:                                   # #498: no budget to take it out of
            pass
        elif budget is not None and a.resident_budget_gib is None:   # its RAM comes out of the experts' budget
            budget = resident_budget_gib(model, ram, kv_ram_gb)
            ok(f"RAM budget: {budget} GiB (less the KV cache's RAM)")
        elif budget is not None and budget > resident_budget_gib(model, ram, kv_ram_gb):
            warn(f"the KV cache's {kv_ram_gb:.1f} GB of RAM come on top of your {budget:g} GiB RAM budget (setup "
                 f"would take them out of it: {resident_budget_gib(model, ram, kv_ram_gb)} GiB); kept as you chose")
    elif ctx >= 65536:   # #620: say why, so a regenerated config that lost --kv-resident is not a surprise
        ok(f"KV streaming off: it needs ~{kv_ram_gb:.1f} GB of RAM beside the ~{MODELS[model]['ram_gb']} GB {model} "
           f"uses, and this PC has {ram:.0f}; the KV cache stays in VRAM (fewer cached experts). --kv-streaming on "
           "turns it on anyway")
    elif a.kv_streaming == "on":
        warn("--kv-streaming on: a context under 64K is not streamed (the attention's window holds all of it): off")
    if budget is not None and not q4_split:   # UD-Q4_K_XL: the experts read from the GGUF in place, the most-used N
        args += ["--resident-budget-gib", f"{budget:g}"]   # GiB kept in RAM (#498: a layer split had no budget)
        if gpu.get("uma") and "--expert-cache" in args:
            # #1715: the cache and the budget draw on the same RAM: a number that leaves the OS its room, not `auto`
            slots = uma_expert_cache(model, ram, budget, kv_ram_gb if "--kv-resident" in args else 0.0)
            if slots is not None:
                args[args.index("--expert-cache") + 1] = str(slots)
                ok(f"expert cache: {slots} slots (not auto): this GPU shares the RAM with the {budget:g} GiB budget, "
                   f"so setup leaves {UNSLOTH_RAM_LEFT_GB} GB of the RAM for the OS and the desktop (#1715). "
                   "--expert-cache N in the config's args sets it")
            else:
                warn(f"this GPU shares the RAM, and the {budget:g} GiB RAM budget leaves almost none for the expert "
                     "cache: it stays on auto, which may take the room the OS needs (#1715). A smaller "
                     "--resident-budget-gib leaves room")
    if vision != "none":
        args += ["--vision"]
        if VISION[vision]["reserve_mib"]:
            args += ["--vram-reserve-mib", str(VISION[vision]["reserve_mib"])]
        if vision == "gpu" and a.vram_reserve_mib is None and 0 < gpu.get("vram_gb", 0.0) <= 12.5:
            # a tip only (recommend, never force): on a 12 GB card the encoder's 700 MiB can leave ~200 MiB free
            print(f"  tip: images on a {gpu['vram_gb']:.0f} GB card can leave little VRAM free; if a request stalls, "
                  f"run setup again with --vram-reserve-mib {VISION_GPU_SMALL_RESERVE_MIB}")
    if a.vram_reserve_mib is not None:                 # #493: VRAM left free for other programs (only when given)
        if "--vram-reserve-mib" in args:
            i = args.index("--vram-reserve-mib") + 1
            if vision == "gpu" and a.vram_reserve_mib < int(args[i]):
                warn(f"--vram-reserve-mib {a.vram_reserve_mib}: the image encoder on the GPU needs ~{args[i]} MiB of "
                     "it; kept as you chose (it may run out of VRAM when it reads a picture)")
            args[i] = str(a.vram_reserve_mib)
        else:
            args += ["--vram-reserve-mib", str(a.vram_reserve_mib)]
        ok(f"VRAM kept free for other programs: {a.vram_reserve_mib} MiB (--vram-reserve-mib; the expert cache takes "
           "that much less)")
    if not multi and family == "unsloth" and 0 < gpu.get("vram_gb", 0.0) <= UD_FIT_CARD_GB:
        # a card of about 8 GB: the config setup would write may not start (measured on an A750).  Pick the least
        # harmful one that fits and say so; what the user chose with --context / --kv / --vram-reserve-mib is kept
        res_arg = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args[:-1] else None
        vg = gpu["vram_gb"]
        rt_mib = unsloth_runtime_mib(gpu)
        fits = unsloth_vram_slack_mib(vg, ctx, kv, UD_FIT_ENGINE_RESERVE_MIB if res_arg is None else res_arg, True,
                                      rt_mib) >= UD_FIT_MIN_SLACK_MIB
        plan = None
        if not fits:
            plan = unsloth_small_card_plan(vg, ctx, kv, res_arg, ctx_fixed=bool(a.context), kv_fixed=bool(a.kv),
                                           runtime_mib=rt_mib)
            if plan is not None:
                nctx, nkv, nmtp, slack = plan
                changes = []
                if nctx != ctx:
                    args[args.index("--max-context") + 1] = str(nctx)
                    changes.append(f"context {ctx} -> {nctx}")
                    ctx = nctx
                if nkv != kv:
                    if "--kv" in args[:-1]:
                        args[args.index("--kv") + 1] = nkv
                    else:
                        args += ["--kv", nkv]
                    changes.append(f"KV cache {kv} -> {nkv}")
                    kv = nkv
                if not nmtp and "--mtp" in args[:-1]:
                    i = args.index("--mtp")
                    del args[i:i + 2]
                    changes.append("no MTP draft layer on the GPU (about 0.8 GB more for the expert cache; decode is slower)")
                warn(f"with the usual settings {model} would not start on this {vg:.0f} GB card (setup's estimate: no VRAM "
                     "left for the expert cache). Setup chose the least harmful start that fits: " + ", ".join(changes)
                     + ". A bigger context, 8-bit KV or the draft layer need a card with more VRAM (or "
                     "--vram-reserve-mib 0, which leaves nothing for other programs).")
        for line in unsloth_small_card_lines(model, vg, plan, fits):
            warn(line)
    if not multi and 0 < gpu.get("vram_gb", 0.0) < SMALL_CARD_GB:
        # #496: on a 6 GB card the expert cache can get no room at all; the engine lowers its own reserve when that
        # is what it takes, and says what is short when even that is not enough.  Setup only says what helps.
        for line in small_card_note(ctx, draft_vocab):   # a recommendation: nothing changes
            say("  " + line)
    elif hip and a.vram_reserve_mib is None and linux_desktop():
        for line in desktop_reserve_note():              # #560 #516: a recommendation: nothing changes
            say("  " + line)
    if not hip and not multi and a.vram_reserve_mib is None and gpu_drives_display(gpu):
        say("  tip: this card drives a display. If the PC freezes or the screen goes black once the model is loaded "
            f"(#779), keep more VRAM free: run setup again with --vram-reserve-mib {DISPLAY_RESERVE_MIB}")   # a tip only
    if not hip:
        for gi in ([g for g in multi] if multi else [gpu.get("index", 0)]):
            gi = gi["index"] if isinstance(gi, dict) else gi
            w = compute_mode_warning(gi, gpu_compute_mode(gi))
            if w:
                warn(w)                                   # #1445: a warning only, never a refusal
    if esp is not None:
        # the package's profile, with llama.cpp's flags (the engine takes the same ones)
        args += ["--control-vector-scaled", f"{esp}:1.0", "--control-vector-layer-range", "4", "44",
                 "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    cfg = {"exe": engine_exe(eng), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{model.lower()}", "log": str(home() / f"strata-{tag.lower()}.log"),
           "lib_dirs": lib_dirs, "port": port}
    if cuda_tk == 12:                                  # the experimental CUDA 12 engine (engine-cuda12/)
        cfg["cuda"] = 12
    if (uenv := unsloth_default_env(family, model, hip, in_budget)):
        cfg["env"] = uenv
    if hip:
        cfg["backend"] = "hip"
        # the dense prompt GEMMs through hipBLASLt with kernels measured on this GPU generation (tools/hip; +40-60%
        # prompt speed on the 7900 XTX): only a table for this card's arch AND the installed hipBLASLt version (the
        # engine refuses any other one and falls back to plain hipBLAS)
        if CONTROL:
            pass    # the app's engine is a package: the server picks its tuning/ table for the card when it starts (strata_control/hipblaslt_tuning.py)
        else:
            table = hipblaslt_table(gpu["arch"], lib_dirs, meta.get("hipblaslt_version"), [eng / "tuning"])
            if table:
                cfg["env"] = {"STRATA_HIPBLASLT_TUNING": str(table)}
        if GFX1103_OPT_IN and gfx_arch_is(gpu["arch"], "gfx1103") and not WIN:
            # the 780M's KFD queues are evicted (a GPU reset) while transparent huge pages move the pinned expert arena
            cfg.setdefault("env", {})["STRATA_NO_ARENA_THP"] = "1"
            # the exact gfx1151 speed switches measured on this chip (GFX1103_SWITCHES): the engine has no arch defaults
            # for gfx1103, so setup writes them into the config.  They are setup's own (SETUP_ENV), so a re-run refreshes
            # them and turning the opt-in off takes them out again.
            cfg.setdefault("env", {}).update(GFX1103_SWITCHES)
        if GFX1150_OPT_IN and gfx_arch_is(gpu["arch"], "gfx1150"):
            # the exact gfx1151 speed switches measured on this chip (GFX1150_SWITCHES): the engine has no arch defaults
            # for gfx1150, so setup writes them into the config.  They are setup's own (SETUP_ENV), so a re-run refreshes
            # them and turning the opt-in off takes them out again.
            cfg.setdefault("env", {}).update(GFX1150_SWITCHES)
        # RDNA2 (gfx103x): the prompt's FP16 GEMMs through rocBLAS's best kernel per shape (tools/hip/tune_rocblas;
        # the GDN input projection 6x faster below 1,152 tokens on a 6900 XT).  The ids are valid for one rocBLAS
        # build, which only the engine can read, so setup points it at the directory and the engine picks the file
        # <arch>-rocblas-<version>.txt for its own library, or runs the default kernels and says so in its log.
        rocblas_tables = sorted(p.name for p in (ROOT / "tools" / "hip").glob(f"{gpu['arch']}-rocblas-*.txt"))
        if rocblas_tables:
            cfg.setdefault("env", {})["STRATA_ROCBLAS_TUNING"] = str(ROOT / "tools" / "hip")
            ok(f"rocBLAS tuning tables for {gpu['arch']}: {', '.join(rocblas_tables)} (the engine takes the one of its rocBLAS build)")
        if resident:   # ROCm: large page-locked host allocations can fail or be slow for the CPU; keep the copy pageable
            cfg.setdefault("env", {})["STRATA_RESIDENT_PIN"] = "0"
    if gpu["count"] > 1 or a.gpu is not None:
        cfg["gpu"] = gpu["index"]                      # the engine is told this card (issue #51)
        cfg["gpus_asked"] = True                       # chosen at setup: not asked again at start
    if multi:                                          # a layer split across these cards (the server adds the flag)
        cfg["gpu"] = multi
        cfg["layer_split"] = a.layer_split or "auto"
        if getattr(a, "_roles", None):                 # roles chosen: the card list is Main last, in this order, and the server keeps it
            cfg["gpu_order"] = "as_given"
        ok(f"layer split across GPUs {multi} ({cfg['layer_split']})")
        recommend_remote_expert_opt(cfg, off=a.no_remote_expert_opt)
        replica_done = False
        if a.replicas is None and a.layer_split in (None, "auto") and (getattr(a, "replica_profile", False) or
                                                                       os.environ.get("STRATA_REPLICA_PROFILE")):
            try:                                       # the opt-in replica profile (strata_control/replica_profile.py)
                from strata_control import replica_profile
                if getattr(a, "replica_profile", False):
                    cfg["replica_profile"] = True
                lines = replica_profile.apply([{"name": c.get("name"), "vram_gb": c.get("vram_gb"), "vendor": c.get("vendor"),
                                                "index": c.get("index")} for c in chosen], cfg, float(ram),
                                              parallel_given=a.parallel is not None)
                replica_done = bool(cfg.get("replicas"))
                for line in lines:
                    (ok if replica_done else say)(line)
            except Exception as e:                     # noqa: BLE001 - a profile is never a reason to stop setup
                say(f"  (the replica profile was not applied: {e})")
        if hip and a.layer_split in (None, "auto") and not replica_done:    # four Radeon AI PRO R9700 cards and Qwen3.8 IQ3_S: the measured serving profile
            try:                                       # (strata_control/amd_four_card.py); what the person chose always wins
                from strata_control import amd_four_card
                for line in amd_four_card.apply([{"name": c.get("name"), "vram_gb": c.get("vram_gb"), "vendor": "amd"} for c in chosen],
                                                cfg, parallel_given=a.parallel is not None):
                    ok(line)
            except Exception as e:                     # noqa: BLE001 - a profile is never a reason to stop setup
                say(f"  (the four-card profile was not applied: {e})")
    if getattr(a, "_roles", None):                     # the roles themselves (strata_control/gpu_roles.py); "gpu" above is their mirror
        final = cfg["gpu"] if isinstance(cfg.get("gpu"), list) else [cfg["gpu"]]      # what the checks above left (a model that cannot share keeps one card)
        main = final[-1]
        cfg["gpu_roles"] = {str(i): ("main" if i == main else "helper" if i in final else "off") for i in sorted(set(a._roles) | set(final))}
        cfg["gpus_asked"] = True
        cfg.pop("gpu_none", None)
        if isinstance(cfg.get("gpu"), list) and len(final) > 1:
            cfg["gpu_order"] = "as_given"
    if a.replicas is not None and a.replicas >= 2:      # data-parallel replicas (serve/replicas.py), opt-in
        cards = multi if multi else []
        if not cards or len(cards) % a.replicas:
            warn(f"--replicas {a.replicas}: needs the chosen GPUs ({cards or 'one'}) to divide into {a.replicas} equal "
                 "groups; not set (choose the cards with --gpu 0,1,2,3, or write the groups into the config: "
                 '"replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}])')
        else:
            cfg["replicas"] = a.replicas
            per = len(cards) // a.replicas
            ok(f"{a.replicas} replicas of {per} GPU(s) each: {' | '.join(','.join(map(str, cards[i * per:(i + 1) * per])) for i in range(a.replicas))}")
            if per > 1 and a.layer_split not in (None, "auto"):
                warn("--layer-split is one engine's value; with replicas every group would get it: leave it at auto")
            if (os.cpu_count() or 0) < 8 * a.replicas:
                warn("each replica loads its own copy of the model into RAM and runs its own host threads: check the RAM")
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if draft_vocab:
        cfg["draft_vocab"] = draft_vocab
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    if a.mtp is not None:                              # written only when given (else an earlier choice is carried over)
        if a.mtp == "off":
            cfg["mtp"] = False
            removed = draft_layer.apply(cfg)
            ok("the draft layer (MTP) is off for this model" + (f" ({' '.join(removed)} left out)" if removed else "") +
               ": less graphics memory and RAM, slower words per second. `--mtp on` turns it back on")
        else:
            cfg["mtp"] = True                          # explicit, so carry_over does not bring an earlier false back
            ok("the draft layer (MTP) is on")
    # #465: requests at once - written only when given (else an earlier "parallel" is carried over); a recommendation
    streaming = "--kv-resident" in args
    if a.parallel is not None:
        if a.parallel >= 2:
            cfg["parallel"] = a.parallel
            for i, line in enumerate(parallel_note(a.parallel, [g.get("vram_gb", 0.0) for g in chosen],
                                                   MODELS[model]["arena_gb"], ctx, kv, streaming)):
                (ok if i == 0 else warn)(line)
        else:
            cfg["parallel"] = 1
            ok("parallel requests: one at a time (--parallel 1)")
    if vision != "none":
        old_cfg = home() / f"strata-{tag.lower()}.json"
        vt = vision_tokens(a.vision_tokens, vision, old_cfg if old_cfg.is_file() else adopted)
        cfg["vision"] = {"exe": str(eng / VEXE), "mmproj": str(mmproj),     # the text model, opened vocab-only:
                         "model": str(ready_paths(model, models_dir)["dense"] if ready else shards[0]),
                         "gpu": vision == "gpu", "max_tokens": vt}
        if reader is not None:                          # AMD: the HIP reader with the CPU reader behind it, or the CPU reader
            from strata_control import vision_reader as VR
            cfg["vision"] = VR.vision_section(reader, cfg["vision"])
        if vision == "cpu":
            cfg["vision"]["threads"] = vision_threads()
    elif a.vision_tokens is not None:
        warn("--vision-tokens: images are off for this model, so it is not used")
    if a.images:                                       # only when given: else an earlier "images" is carried over
        cfg["images"] = setup_images(a, gpu, hip, Path(a.models_dir))
    cfg_path = home() / f"strata-{tag.lower()}.json"
    cal = setup_calibration(cfg, hip)                  # #566: Linux HIP too; the tuning is offered on NVIDIA only
    if cal is not None:
        sys.path.insert(0, str(ROOT / "tools"))
        import calibrate as CAL
        cfg["args"] = CAL.apply(cfg["args"], cal.get("settings") or {})
        if "env" in cal:                               # a calibration from before the env step leaves "env" as it is
            cfg["env"] = CAL.apply_env(cfg.get("env"), cal["env"])
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else ""))
    else:                                              # #642: measured counts (a calibration) win over the rule
        cfg["args"] = recommend_pool_workers(cfg["args"])
        for line in pool_workers_note(cfg["args"]):
            warn(line)
        for line in two_socket_note(cpu_sockets()):    # bench tips: text only, the config is not touched
            say("  " + line)
    for line in bench_tips(cfg["args"], cfg.get("env"), ram, MODELS[model]["ram_gb"], gpu.get("vram_gb", 0.0), vision, WIN):
        say("  " + line)
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None, explicit_args(a))
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    # offered only when someone answers: --yes installs and adopted earlier installs are not held up by it
    if cal is None and not hip and not a.no_start and not a.yes and ask(
            "Tune Strata for this PC now? It measures a few engine settings (about 15-30 minutes, longer on a slow card; the PC is busy "
            "meanwhile; later: START-HERE --calibrate)", ["y", "n"], "y", a.yes) == "y":
        tuned = calibrate_config(cfg_path)
    else:
        tuned = None                                   # not asked for: nothing to repeat below
    ok(f"start script: {script.name}")

    say()
    say("All set.")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run {'START-HERE.bat' if WIN else './setup.sh'} (or {script.name}) - it starts right away")
    if vision != "none":
        say("  Images:           send them in the chat page, in chat.py (/image <path>) or over the API")
    if cfg.get("images"):
        say(f"  Image generation: POST http://127.0.0.1:{port}/v1/images/generations (OpenAI's images API)")
    if a.parallel is None:                             # #465: the opt-in, said once (nothing changes)
        for line in parallel_note(None, [g.get("vram_gb", 0.0) for g in chosen], MODELS[model]["arena_gb"], ctx, kv,
                                  "--kv-resident" in cfg["args"]):
            say("  " + line)
    if tuned is False:                                 # #447: a failed tuning is repeated here, not only above
        say("  Tuning:           FAILED (the reason is above): the default settings stay - "
            f"{'START-HERE.bat' if WIN else './setup.sh'} --calibrate tries again")
    if a.no_start:
        return 0
    # offered only when someone answers (--yes asks nothing), default no; the benchmark starts its own server and stops it
    if not a.yes and sys.stdin.isatty() and ask(
            "Run a quick benchmark now? It measures this install for about 10 minutes and writes a report you can share "
            f"(later: {'BENCH.bat' if WIN else './bench.sh'})", ["y", "n"], "n", a.yes) == "y":
        subprocess.call([sys.executable, str(ROOT / "tools" / "strata_bench.py"), "--config", str(cfg_path), "--port", str(port)])
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except acquisition.AcquisitionRefused as e:         # a start that would have installed something: said, not done
        say(f"\n  [X]  {e}")
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)

