#!/usr/bin/env python3
"""Download a tier's model directory (base checkpoint + trained adapters).

The published Hugging Face repos co-locate the base checkpoint and the
trained LoRA / prerouter adapters in ONE directory, so a single snapshot
download produces a ready-to-run model directory that ``edge0 demo`` /
``edge0 serve`` accept directly.

Requirements:
    pip install 'edge0[fetch]'        # or: pip install huggingface_hub

Repo ids come from the environment (set them once, e.g. in ~/.zshrc):

    export EDGE0_35B_REPO=<your-hf-org>/edge0-35b
    export EDGE0_10B_REPO=<your-hf-org>/edge0-10b

Usage:
    python scripts/fetch_models.py --tier edge0-35b --target-dir models
    python scripts/fetch_models.py --tier all --target-dir models

Afterwards point the tier names at what you downloaded:

    export EDGE0_35B_MODEL=$PWD/models/edge0-35b
    edge0 demo edge0-35b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TIER_REPOS = {
    "edge0-35b": "EDGE0_35B_REPO",
    "edge0-10b": "EDGE0_10B_REPO",
}


def _repo_for(tier: str) -> str:
    env = TIER_REPOS[tier]
    repo = os.environ.get(env, "")
    if not repo:
        raise SystemExit(
            f"[fetch] {tier} needs a repo id.\n"
            f"Set {env} to your Hugging Face repo, e.g.\n"
            f"    export {env}=<your-hf-org>/{tier}\n"
            f"then re-run this script.")
    return repo


def _download(tier: str, target_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "[fetch] huggingface_hub is required: pip install 'edge0[fetch]' "
            "or pip install huggingface_hub")
    repo = _repo_for(tier)
    dest = target_dir / tier
    print(f"[fetch] {tier} <- {repo}  (-> {dest})", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))
    print(f"[fetch] done: {dest}\n"
          f"        export {TIER_REPOS[tier].replace('_REPO','_MODEL')}"
          f"={dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True,
                    choices=list(TIER_REPOS) + ["all"])
    ap.add_argument("--target-dir", default="models",
                    help="directory to write <tier>/ under (default: models)")
    args = ap.parse_args()

    target = Path(args.target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    tiers = list(TIER_REPOS) if args.tier == "all" else [args.tier]
    for tier in tiers:
        _download(tier, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
