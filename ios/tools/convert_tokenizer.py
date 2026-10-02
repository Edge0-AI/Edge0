#!/usr/bin/env python3
"""Convert `tokenizer.json` into a compact binary the device can load quickly.

The shipped file is 20 MB of JSON holding 248,044 vocabulary entries and 247,587 merge
rules. Parsing that on a phone at launch is slow and allocates far more than the data
needs. This rewrites it into the same shape the weights use — a small header plus flat
arrays that can be read directly:

    vocab_blob     all token strings concatenated, no separators
    vocab_offsets  uint32[n+1], so token i is blob[offsets[i]..<offsets[i+1]]
    merge_pairs    uint32[2m], the two token ids of each merge in rank order
    specials       ids and strings of the 33 added tokens

Merges are stored as **token id pairs rather than strings**, so the device never has to
look a merge up by text: the BPE loop can key on a packed pair of ids. That is the single
biggest difference from parsing the JSON directly, and it is why the merge table costs
about 2 MB instead of 20.

The pre-tokenizer regex is carried across verbatim rather than reimplemented. It is the
GPT-4 pattern, and the one place where a subtle difference produces a tokenizer that works
on English and quietly mis-splits CJK.

Usage:
    convert_tokenizer.py <model-dir> <out-file>
"""

import json
import struct
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    spec = json.load(open(model_dir / "tokenizer.json"))
    model = spec["model"]
    if model["type"] != "BPE":
        raise SystemExit(f"expected BPE, found {model['type']}")

    vocab = model["vocab"]                      # token string -> id
    merges = model["merges"]                    # [[left, right], ...] in rank order
    added = spec.get("added_tokens", [])

    # Tokens are addressed by id, so build the inverse mapping and check it is dense.
    by_id = {i: s for s, i in vocab.items()}
    for entry in added:
        by_id[entry["id"]] = entry["content"]
    size = max(by_id) + 1
    missing = [i for i in range(size) if i not in by_id]
    if missing:
        raise SystemExit(f"vocabulary has {len(missing)} holes, first at {missing[0]}")

    blob = bytearray()
    offsets = [0]
    for index in range(size):
        blob.extend(by_id[index].encode("utf-8"))
        offsets.append(len(blob))

    # Merges as id pairs. A merge naming a token that is not in the vocabulary would
    # silently never fire, so it is an error rather than a skip.
    pairs = []
    for rank, (left, right) in enumerate(merges):
        if left not in vocab or right not in vocab:
            raise SystemExit(f"merge {rank} references an unknown token: {left!r} {right!r}")
        pairs.append((vocab[left], vocab[right]))

    special_ids = [entry["id"] for entry in added]
    special_blob = bytearray()
    special_offsets = [0]
    for entry in added:
        special_blob.extend(entry["content"].encode("utf-8"))
        special_offsets.append(len(special_blob))

    regex = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

    header = json.dumps({
        "format": "edge0-tokenizer-v1",
        "vocab_size": size,
        "merge_count": len(pairs),
        "special_count": len(added),
        "regex": regex,
        "normalizer": (spec.get("normalizer") or {}).get("type"),
        "byte_level": True,
        "sections": ["vocab_offsets", "vocab_blob", "merge_pairs",
                     "special_ids", "special_offsets", "special_blob"],
    }, separators=(",", ":")).encode()

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(header)))
        out.write(header)
        out.write(struct.pack(f"<{len(offsets)}I", *offsets))
        out.write(bytes(blob))
        flat = [value for pair in pairs for value in pair]
        out.write(struct.pack(f"<{len(flat)}I", *flat))
        out.write(struct.pack(f"<{len(special_ids)}I", *special_ids))
        out.write(struct.pack(f"<{len(special_offsets)}I", *special_offsets))
        out.write(bytes(special_blob))

    total = out_path.stat().st_size
    print(f"  vocab {size:,}   merges {len(pairs):,}   specials {len(added)}")
    print(f"  ✓ {total / 2**20:.1f} MiB → {out_path.name}  "
          f"(from {(model_dir / 'tokenizer.json').stat().st_size / 2**20:.1f} MiB of JSON)")
    print(f"  regex: {regex[:70]}…")


if __name__ == "__main__":
    main()
