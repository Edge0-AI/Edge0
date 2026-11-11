#!/usr/bin/env python3
"""Re-run the 2026-10-08 RTX 4070/IQ2_XS benchmark from provenance.json.

Standard-library only. No network, download, model redistribution, or automatic
publication. Run "check" (SHA-256) *separately* from "run" to avoid warming the
OS file cache immediately before a timed measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

MANIFEST = Path(__file__).with_name("provenance.json")
SOURCE_COMMIT = "9ec3806058cf32ab27a55e4377daf7cf0d087dec"
PATH_OPTIONS = {"--pack", "--native", "--ple-gguf", "--mtp",
                "--expert-profile", "--tokens-file"}
DECODE_RE = re.compile(
    r"^decode\s+(\d+) tokens in\s+([\d.]+) ms\s+->\s+([\d.]+) tok/s",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def git_head(directory: Path) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checked_path(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes data root: " + rel)
    return candidate


def expected_files(manifest: dict) -> list[dict]:
    rows = []
    for entry in manifest["model"]["native_shards"]:
        rows.append({"role": "model", **entry})
    rows.extend(manifest["artifacts"])
    return rows


def verify(manifest: dict, data_root: Path, engine: Path, source: Path | None,
           mode: str, permit_different_binary: bool,
           overrides: dict[str, Path]) -> tuple[list[str], list[str]]:
    problems, warnings = [], []
    head = git_head(source) if source else None
    if source:
        if head is None:
            problems.append(f"Cannot determine Git HEAD in source directory: {source}")
        elif head.lower() != SOURCE_COMMIT:
            problems.append(f"Wrong source HEAD: {head}; expected {SOURCE_COMMIT}")
        try:
            dirty = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if dirty:
                warnings.append("Source working tree has uncommitted changes")
        except (OSError, subprocess.CalledProcessError):
            warnings.append("Cannot check whether source tree is clean")
    else:
        warnings.append("Source Git checkout not verified (provide --source-root)")
    if not engine.is_file():
        problems.append(f"Engine executable missing: {engine}")
    else:
        expected = manifest["source"]
        if engine.stat().st_size != expected["binary_bytes"]:
            msg = (f"Engine size mismatch: {engine.stat().st_size} vs "
                   f"{expected['binary_bytes']} bytes")
            (warnings if permit_different_binary else problems).append(msg)
        digest = sha256(engine)
        if digest != expected["binary_sha256"].upper():
            msg = f"Engine binary SHA-256 mismatch: {digest}"
            (warnings if permit_different_binary else problems).append(msg)
    for item in expected_files(manifest):
        if ((item["path"] == "profiles/learned-heart.bin" and "--expert-profile" in overrides) or
            (item["path"] == "prompts/neuro.tokens" and "--tokens-file" in overrides)):
            continue  # Local profile/prompt are checked separately, without matching historical SHA.
        target = checked_path(data_root, item["path"])
        if not target.is_file():
            problems.append(f"MISSING [{item['role']}]: {target}")
            continue
        if target.stat().st_size != item["bytes"]:
            problems.append(f"SIZE MISMATCH [{item['role']}]: {target}")
            continue
        if mode == "sha256" and sha256(target) != item["sha256"].upper():
            problems.append(f"SHA256 MISMATCH [{item['role']}]: {target}")
    for flag, target in overrides.items():
        if not target.is_file():
            problems.append(f"OVERRIDE MISSING [{flag}]: {target}")
    if "--tokens-file" in overrides and overrides["--tokens-file"].is_file():
        content = overrides["--tokens-file"].read_text(encoding="utf-8")
        if not re.fullmatch(r"[\d+,\s-]+", content):
            problems.append("User prompt must be pretokenized integer IDs (comma/whitespace-separated)")
        else:
            count = len(re.findall(r"[-+]?\d+", content))
            target_count = int(manifest["primary_command"]["actual_prompt_tokens"])
            if count != target_count:
                problems.append(f"Prompt length {count} differs from {target_count} historical input tokens")
    if mode == "size":
        warnings.append("Data file SHA-256 was NOT checked; only sizes were compared")
    if overrides:
        warnings.append("Comparable-workload reproduction: custom prompt/profile; NOT an exact historical replay")
    return problems, warnings


def make_command(manifest: dict, data_root: Path, engine: Path,
                 overrides: dict[str, Path]) -> list[str]:
    original = manifest["primary_command"]["args"]
    command = [str(engine)]
    path_value = None
    for arg in original:
        if path_value is not None:
            command.append(str(overrides[path_value] if path_value in overrides else
                               checked_path(data_root, arg)))
            path_value = None
        else:
            command.append(arg)
            path_value = arg if arg in PATH_OPTIONS else None
    if path_value:
        raise ValueError("Path flag at end of provenance argument vector")
    return command


def gpu_snapshot(path: Path) -> None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,name,memory.total,memory.used,"
             "memory.free,driver_version", "--format=csv"],
            capture_output=True, text=True, timeout=20,
        )
        path.write_text(p.stdout if p.returncode == 0 else p.stderr,
                        encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
        path.write_text(str(exc), encoding="utf-8")


def run_benchmark(manifest: dict, data_root: Path, engine: Path,
                  output: Path, repetitions: int, source: Path | None,
                  validation_mode: str, allow_binary_mismatch: bool,
                  overrides: dict[str, Path]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    command = make_command(manifest, data_root, engine, overrides)
    env = os.environ.copy()
    env.update(manifest["primary_command"]["environment"])
    # The source installer records CUDA runtime DLL directories in BUILD.json.
    # A bare strata.exe launch requires those directories on PATH on Windows.
    build_json = engine.parent / "BUILD.json"
    dll_dirs = []
    if build_json.is_file():
        build = json.loads(build_json.read_text(encoding="utf-8"))
        dll_dirs = [str(Path(d)) for key in ("lib_dirs", "cuda_dirs")
                    for d in (build.get(key) or []) if Path(d).is_dir()]
    if dll_dirs:
        env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])
    result_rows = []
    expected_tokens = int(manifest["primary_command"]["output_tokens"])
    for index in range(1, repetitions + 1):
        folder = output / f"run-{index:02d}"
        folder.mkdir()
        # Do not publish raw stdout: the engine prints the full prompt and
        # output token IDs, potentially encoding private text.
        (folder / "command.json").write_text(
            json.dumps({"argv": command,
                        "environment_overrides": manifest["primary_command"]["environment"],
                        "cuda_library_paths_from_BUILD_json": dll_dirs,
                        "started_utc": datetime.now(timezone.utc).isoformat()},
                       indent=2), encoding="utf-8")
        gpu_snapshot(folder / "gpu-before.csv")
        with (
            (folder / "stdout.log").open("w", encoding="utf-8") as stdout,
            (folder / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            proc = subprocess.run(command, cwd=data_root, env=env,
                                  stdout=stdout, stderr=stderr, check=False)
        gpu_snapshot(folder / "gpu-after.csv")
        output_text = (folder / "stdout.log").read_text(encoding="utf-8")
        matches = DECODE_RE.findall(output_text)
        row = {"run": index, "exit_code": proc.returncode,
               "accepted_decode_tok_s": None, "output_tokens": None,
               "decode_ms": None, "valid": False}
        if proc.returncode == 0 and len(matches) == 1:
            tokens, ms, rate = matches[0]
            row.update({"output_tokens": int(tokens), "decode_ms": float(ms),
                        "accepted_decode_tok_s": float(rate),
                        "valid": int(tokens) == expected_tokens})
        (folder / "summary.json").write_text(json.dumps(row, indent=2),
                                               encoding="utf-8")
        result_rows.append(row)
        print(f"run {index}/{repetitions}: {row}", flush=True)
    success = [r["accepted_decode_tok_s"] for r in result_rows if r["valid"]]
    report = {
        "source_commit_required": SOURCE_COMMIT,
        "source_git_head_observed": git_head(source) if source else None,
        "binary_sha256_expected": manifest["source"]["binary_sha256"],
        "binary_sha256_observed": sha256(engine),
        "binary_mismatch_explicitly_allowed": allow_binary_mismatch,
        "data_validation_mode": validation_mode,
        "environment_overrides": manifest["primary_command"]["environment"],
        "cuda_library_paths_from_BUILD_json": dll_dirs,
        "comparison_type": "comparable user-supplied workload (not historical exact)" if overrides else
                           "historical inputs if all file hashes match",
        "prompt_file_sha256": sha256(overrides["--tokens-file"]) if "--tokens-file" in overrides else
                              sha256(checked_path(data_root, "prompts/neuro.tokens")),
        "profile_sha256": sha256(overrides["--expert-profile"]) if "--expert-profile" in overrides else
                          sha256(checked_path(data_root, "profiles/learned-heart.bin")),
        "prompt_tokens": int(manifest["primary_command"]["actual_prompt_tokens"]),
        "results": result_rows,
        "valid_runs": len(success),
        "median_accepted_decode_tok_s": statistics.median(success) if success else None,
        "mean_accepted_decode_tok_s": statistics.mean(success) if success else None,
        "min_accepted_decode_tok_s": min(success) if success else None,
        "max_accepted_decode_tok_s": max(success) if success else None,
        "historical_primary_median": manifest["run_sets"]["primary_e4_s24"]["median_tok_s"],
        "note": "GPU snapshots are before/after, NOT an in-run peak VRAM measurement",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2),
                                        encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if len(success) != repetitions:
        raise SystemExit("One or more runs failed or produced the wrong output token count")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "run"])
    p.add_argument("--data-root", type=Path, required=True,
                   help="Folder containing models/, packs/, mtp/, profiles/, prompts/")
    p.add_argument("--engine", type=Path, required=True,
                   help="Path to the compiled strata.exe executable")
    p.add_argument("--source-root", type=Path,
                   help="The pinned Git checkout to verify with git rev-parse HEAD")
    p.add_argument("--hash-mode", choices=["sha256", "size"],
                   help="Defaults to sha256 for check, size for run (avoid pre-run cache warming)")
    p.add_argument("--allow-binary-mismatch", action="store_true",
                   help="Run a rebuild that differs in bytes; not a byte-identical engine reproduction")
    p.add_argument("--profile", type=Path,
                   help="Locally generated expert profile; no historical profile download required")
    p.add_argument("--prompt-file", type=Path,
                   help="Your own pretokenized, 28,912-token prompt; historical prompt not required")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--out", type=Path, help="New output directory for raw logs (never overwritten)")
    args = p.parse_args()
    if args.action == "run" and (not args.out or args.runs < 1):
        p.error("run requires --out and --runs >= 1")
    data_root, engine = args.data_root.resolve(), args.engine.resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mode = args.hash_mode or ("sha256" if args.action == "check" else "size")
    overrides = {}
    if args.profile:
        overrides["--expert-profile"] = args.profile.resolve()
    if args.prompt_file:
        overrides["--tokens-file"] = args.prompt_file.resolve()
    problems, warnings = verify(manifest, data_root, engine,
                                 args.source_root.resolve() if args.source_root else None,
                                 mode, args.allow_binary_mismatch, overrides)
    for line in warnings:
        print("WARNING:", line, file=sys.stderr)
    if problems:
        for line in problems:
            print("ERROR:", line, file=sys.stderr)
        return 2
    print(f"Preflight OK ({mode} data validation)")
    if args.action == "run":
        run_benchmark(manifest, data_root, engine, args.out.resolve(), args.runs,
                      args.source_root.resolve() if args.source_root else None,
                      mode, args.allow_binary_mismatch, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
