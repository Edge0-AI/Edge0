"""Write a local service configuration for the Q4_0 latent Metal backend."""

import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--latent-dense", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--mtp-dir", type=Path, required=True)
    parser.add_argument("--mtp-gguf", type=Path, required=True)
    parser.add_argument("--expert-profile", type=Path)
    parser.add_argument("--port", type=int, default=8127)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    source = Path(__file__).resolve().parents[1]
    binary = options.build_dir.resolve() / "strata"
    model = options.model_dir.resolve()
    pack = model / "latent_a_pack"
    dense = options.latent_dense.resolve()
    mtp = options.mtp_dir.resolve()
    mtp_gguf = options.mtp_gguf.resolve()
    tokenizer = options.tokenizer.resolve()
    files = [
        binary, pack / "native_experts.txt", pack / "experts.bin",
        pack / "labels.bin", pack / "index.txt", pack / "dense.bin",
        model / "model-metal-shell.gguf", model / "ple-fp8.gguf",
        dense, mtp_gguf, mtp / "draft_vocab.bin",
    ]
    if options.expert_profile:
        files.append(options.expert_profile.resolve())
    for path in files:
        if not path.is_file():
            parser.error(f"missing file: {path}")
    for path in (mtp, tokenizer):
        if not path.is_dir():
            parser.error(f"missing directory: {path}")

    manifest = (pack / "native_experts.txt").read_text()
    match = re.search(r"expert_embd (1280|1792)\b", manifest)
    if not match:
        parser.error("the expert manifest must declare latent width 1280 or 1792")
    width = int(match[1])
    dense_bytes = 48 * 2 * 8 * 2560 * (width // 32) * 18
    if dense.stat().st_size != dense_bytes:
        parser.error(f"Q4_0 D/U at width {width} requires {dense_bytes} bytes")
    for layer in range(48):
        scale = pack / "scales" / f"l{layer}.f32"
        if not scale.is_file() or scale.stat().st_size != width * 4:
            parser.error(f"expected {width} float32 scales in {scale}")

    args = [
        "--pack", str(pack),
        "--native", str(model / "model-metal-shell.gguf"),
        "--native-dense-gguf", str(model / "model-metal-shell.gguf"),
        "--ple-gguf", str(model / "ple-fp8.gguf"),
        "--ple-io", "direct",
        "--expert-cache", "64",
        "--mmap-experts",
        "--prefill", "auto",
        "--max-context", "32768",
        "--spec", "4",
        "--suffix-draft", "3",
        "--mtp", str(mtp),
        "--spec-min-p", "0.5",
    ]
    if options.expert_profile:
        args += ["--expert-profile", str(options.expert_profile.resolve())]
    env = {
        "STRATA_METAL_LATENT": "1",
        "STRATA_LATENT_DENSE": str(dense),
        "STRATA_LATENT_DENSE_FMT": "4",
        "STRATA_LATENT_A": str(pack / "experts.bin"),
        "STRATA_LATENT_LABELS": str(pack / "labels.bin"),
        "STRATA_HC_Q8": "0",
        "STRATA_ROUTE_TAIL_SKIP": "0",
        "STRATA_PREFILL_CPU_SHARE": "0",
        "STRATA_METAL_LATENT_GROUP_U": "1",
        "STRATA_METAL_LATENT_PROMPT": "1",
        "STRATA_METAL_PROMPT_CHUNK": "8192",
        "STRATA_METAL_LATENT_DECODE_TILES": "1",
        "STRATA_METAL_MATH": "fast",
        "STRATA_METAL_LATENT_PROMPT_QSA_MM": "1",
        "STRATA_METAL_GDN_SCAN": "gdn_scanr_4_8",
        "STRATA_METAL_MTP_GGUF": str(mtp_gguf),
        "STRATA_METAL_DRAFT_VOCAB": str(mtp / "draft_vocab.bin"),
        "STRATA_DECODE_TIMING": "1",
        "STRATA_METAL_LATENT_VERIFY_FAST": "1",
        "STRATA_METAL_LATENT_PROMPT_TUNE": "1",
    }
    output = options.output.resolve()
    config = {
        "exe": str(binary), "cwd": str(source), "args": args, "env": env,
        "tokenizer": str(tokenizer), "model_name": f"strata-l{width}-q3k-metal",
        "backend": "metal", "lib_dirs": [], "host": "127.0.0.1",
        "port": options.port, "log": str(output.with_suffix(".engine.log")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
