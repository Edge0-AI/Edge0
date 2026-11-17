#!/usr/bin/env python3
"""Print the size and SHA-256 Hugging Face's API metadata gives for the files of a repository at a pinned commit.
Nothing is downloaded: https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=1 lists each LFS file with
`lfs.oid` (the SHA-256) and `size`.

    python tools/hf_pin_fetch.py ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF ed59f92082b1e93c0e96d60a8b11aab089b52f09
    python tools/hf_pin_fetch.py --json <repo> <revision>     # {path: [size, sha256]} as JSON
    python tools/hf_pin_fetch.py --setup                      # every repository in setup.py's HF_REVISIONS, as JSON
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_control import tls  # noqa: E402

tls.install()       # #3722: the operating system's certificates


def tree(repo: str, revision: str) -> dict:
    """{path: (size, sha256)} of the repository's LFS files at `revision` (the API pages its answer with Link headers)."""
    out = {}
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1&expand=0"
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata-pin"}), timeout=60) as r:
            for e in json.load(r):
                if e.get("type") == "file" and e.get("lfs"):
                    out[e["path"]] = (int(e["lfs"]["size"]), e["lfs"]["oid"])
            link = r.headers.get("Link", "")
        url = next((p.split(";")[0].strip(" <>") for p in link.split(",") if 'rel="next"' in p), None)
    return out


def main(argv: list) -> int:
    if argv and argv[0] == "--setup":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import setup
        print(json.dumps({r: {p: list(v) for p, v in tree(r, rev).items()} for r, rev in setup.HF_REVISIONS.items()}, indent=1))
        return 0
    as_json = bool(argv) and argv[0] == "--json"
    repo, revision = argv[1:3] if as_json else argv[:2]
    t = tree(repo, revision)
    if as_json:
        print(json.dumps({p: list(v) for p, v in t.items()}, indent=1))
    else:
        for p, (size, sha) in sorted(t.items()):
            print(f"{size:>14,}  {sha}  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
