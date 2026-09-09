#!/usr/bin/env python3
"""Download a tier's model directory (base checkpoint + trained adapters).

The published Hugging Face repos co-locate the base checkpoint and the
trained LoRA / prerouter adapters in ONE directory, so a single snapshot
download produces a ready-to-run model directory that ``edge0 demo`` /
``edge0 serve`` accept directly.

Requirements:
    pip install 'edge0[fetch]'        # or: pip install huggingface_hub

Repo ids default to the official releases (override via the environment
if you mirror them):

    export EDGE0_35B_REPO=Edge0/Edge0-35b-a3b-preview   # default
    export EDGE0_8B_REPO=Edge0/Edge0-8B-A1B-preview   # default

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
    "edge0-35b": ("EDGE0_35B_REPO", "Edge0/Edge0-35b-a3b-preview"),
    "edge0-8b": ("EDGE0_8B_REPO", "Edge0/Edge0-8B-A1B-preview"),
}


def _repo_for(tier: str) -> str:
    env, default = TIER_REPOS[tier]
    repo = os.environ.get(env, default)
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
