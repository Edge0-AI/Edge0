"""tools/image/install.py - image generation (Qwen-Image-2.1-Turbo) for an existing Strata install, or on its own:
compiles sd-cli for the backend (setup.py's build_image_engine), downloads and checks the three model files, and
writes the "images" section into a run config (or prints it).  setup.py --images qwen-image does the same inside
the normal setup; this is the way to add it without running the whole setup again.

    python tools/image/install.py --config strata-q2_0.json                 (backend and size picked for this PC)
    python tools/image/install.py --backend vulkan --quant Q8_0 --models-dir ~/models --print
    python tools/image/install.py --backend hip --arch gfx1103 --build-only

--gpu-policy auto|keep|shrink|unload: what the text model does during an image job (serve/images.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "image"))
import qimage as QI  # noqa: E402


def detect_gpu(S) -> tuple[dict | None, bool]:
    """setup.py's view of this PC's GPU: (gpu dict, AMD?) - an NVIDIA card first, else an AMD one, else none."""
    try:
        g = S.gpu_info()
        if g:
            return {**g, "vendor": "nvidia"}, False
    except Exception:                                   # noqa: BLE001 - no NVIDIA driver
        pass
    try:
        amd = S.amd_gpus_win() if S.WIN else S.amd_gpus()
        if amd:
            g = min(amd, key=S.amd_rank)
            return S.amd_apply_uma(dict(g)), True
    except Exception:                                   # noqa: BLE001
        pass
    return None, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=QI.BACKENDS)
    ap.add_argument("--arch", help="the GPU architecture to compile for (CUDA: 86, ...; HIP: gfx1103, ...)")
    ap.add_argument("--quant", choices=QI.QUANTS)
    ap.add_argument("--gpu-gb", type=float, help="the GPU memory the quant is picked for (default: detected)")
    ap.add_argument("--models-dir", help="where the model files go (default: the data folder's models)")
    ap.add_argument("--config", help="a run config (strata-<model>.json) to write the \"images\" section into")
    ap.add_argument("--gpu-policy", choices=["auto", "keep", "shrink", "unload"], default="auto")
    ap.add_argument("--gpu", type=int, help="the GPU's number for the image job (default: the backend's first)")
    ap.add_argument("--native", choices=["on", "off"], default="off",
                    help="EXPERIMENTAL: Strata's own DiT runner on CUDA / HIP (the config's \"native\")")
    ap.add_argument("--build-only", action="store_true", help="compile sd-cli only")
    ap.add_argument("--print", action="store_true", help="print the images section instead of writing it")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    import setup as S
    gpu, amd = detect_gpu(S) if not (a.backend and (a.arch or a.backend in ("vulkan", "cpu"))) else (None, False)
    gpu = dict(gpu or {})
    if a.arch:
        gpu["arch"] = a.arch
    if a.gpu_gb is not None:
        gpu["vram_gb"] = a.gpu_gb
    ns = argparse.Namespace(images="qwen-image", image_backend=a.backend, image_quant=a.quant,
                            image_gpu_policy=a.gpu_policy, yes=a.yes)
    backend, quant, gb = S.image_backend_choice(ns, gpu or None, amd or a.backend == "hip")
    if backend in ("cuda", "hip") and not gpu.get("arch"):
        S.fail(f"the {backend} build needs the GPU's architecture", "pass --arch (e.g. 86 or gfx1103)")
    S.ok(f"image generation: {quant} on {backend}" + (f" ({gb:.0f} GB GPU memory)" if gb else ""))
    exe, lib_dirs = S.build_image_engine(backend, gpu)
    if a.build_only:
        return 0
    if a.models_dir:
        models = Path(a.models_dir).expanduser()
    else:
        data, _ = S.data_folder(None)
        models = data / "models"
    S.say("  " + QI.LICENSE_NOTE)
    for name in (quant, "vae", "llm"):
        repo, path, size, sha = QI.FILES[name]
        dst = QI.local_path(models, name)
        S.download(QI.file_url(name, S.hf_endpoint()), dst, f"image model: {Path(path).name}")
        S.verify_sha256(dst, size, sha)
    section = QI.image_config(backend, quant, exe, models, uma=QI.is_apu(gpu), gpu=a.gpu, lib_dirs=lib_dirs,
                              policy=a.gpu_policy, native=a.native == "on")
    if a.print or not a.config:
        print(json.dumps({"images": section}, indent=1))
        return 0
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    cfg["images"] = section
    S.write_config(path, cfg)
    S.ok(f"{path.name}: image generation added (POST /v1/images/generations once the server starts again)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
