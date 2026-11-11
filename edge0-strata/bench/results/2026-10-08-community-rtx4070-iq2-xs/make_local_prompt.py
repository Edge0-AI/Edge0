#!/usr/bin/env python3
"""Create a private, pretokenized input for comparable IQ2_XS benchmarking.

This script does NOT upload, distribute or print the input text. It uses the
tokenizer of the published Strata source checkout and makes exactly 28,912
tokens. A different text is a comparable test workload, not an exact replay
of the historical medical prompt.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True,
                    help="Checkout of pinned benchmark source commit")
    ap.add_argument("--gguf", type=Path, required=True,
                    help="IQ2_XS model GGUF shard 1")
    ap.add_argument("--text", type=Path, required=True,
                    help="Local UTF-8 text of your choice; never uploaded")
    ap.add_argument("--out", type=Path, required=True,
                    help="New output file containing integer token IDs")
    ap.add_argument("--tokens", type=int, default=28912)
    args = ap.parse_args()
    if args.tokens < 2:
        ap.error("--tokens must be >= 2")
    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "tools"))
    from strata_tokenizer import Tokenizer

    text = args.text.read_text(encoding="utf-8").strip()
    if not text:
        ap.error("--text input is empty")
    tokenizer = Tokenizer.from_gguf(args.gguf.resolve())
    ids = tokenizer.encode(text)
    if not ids:
        ap.error("Tokenizer returned zero tokens")
    # Repeat a user's locally chosen document if it is shorter than the target.
    # Re-tokenize the concatenation to preserve BPE boundaries.
    if len(ids) < args.tokens:
        chunk = text + "\n\n"
        count = max(2, args.tokens // len(ids) + 2)
        while True:
            ids = tokenizer.encode(chunk * count)
            if len(ids) >= args.tokens:
                break
            count *= 2
    ids = ids[:args.tokens]
    target = args.out.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (" ".join(str(i) for i in ids) + "\n").encode("utf-8")
    with target.open("xb") as f:
        f.write(data)
    print(f"Wrote {len(ids)} tokens to {target}")
    print(f"SHA256: {hashlib.sha256(data).hexdigest().upper()}")
    print("Input is local and differs from historical benchmark workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
