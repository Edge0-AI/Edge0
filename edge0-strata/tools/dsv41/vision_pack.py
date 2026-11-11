# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/vision_pack.py - DeepSeek-V4.1's vision tower (the part of the model that reads pictures, which the GGUFs
leave out) for a GGUF install: shard 1 of DeepSeek's own checkpoint (model-00001-of-00048, the tower's vision.* and the
aligner's aligner.* tensors, BF16, nothing else) and config.json (their geometry) from download_gguf.json's "vision"
entry, plus the picture span's three marker rows (image_start, image_newline, image_end: BF16 [5120]), which sit beside
the embedding in shard 2 - only their 30 KB are read from it, with an HTTP range request, never the 1.3 GB file.

    python tools/ds4/vision_pack.py --dir <folder with shard 1 and config.json>     -> vision.bin + vision.json there

vision.bin holds every tensor exactly as the checkpoint has it (copied, 256-byte aligned) and then the three marker
rows; it is read back and compared with the shard bit for bit before it is kept.  vision.json lists every tensor (name,
dtype, offset, bytes, shape) as tensors.txt does, and the geometry the engine reads (vision.<key> from config.json's
vision_config, image_token_id); pack_gguf.py adds both to the pack.

The marker rows: the shard's header (a range request), then the rows (one range request), each row's sha256 checked
against the manifest's.  A server that answers a range request with the whole file (status 200, not 206) is cut off at
once, the request is tried once more, then this stops with the reason - nothing large is ever downloaded here.
--marks-url takes another URL, or a local file (a mirror, the tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import time
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "download_gguf.json"
HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
ALIGN = 256
MARKS = ("image_start", "image_newline", "image_end")   # (the order in vision.bin)
TRIES = 2                                               # a range request and one retry


class RangeError(Exception):
    pass


def manifest_entry(path: pathlib.Path = MANIFEST) -> dict:
    v = json.loads(path.read_text(encoding="utf-8")).get("vision") or {}
    if not v.get("files") or not v.get("marks"):
        raise SystemExit(f"{path}: no \"vision\" entry")
    return v


def read_range(url: str, start: int, n: int, total: int | None = None, timeout: int = 60) -> bytes:
    """Bytes start .. start + n - 1 of a file: a URL with a range request (exactly those bytes, status 206 - anything
    else is cut off unread: RangeError), or a local file / file:// URL read at the offset."""
    if not url.startswith(("http://", "https://")):
        p = pathlib.Path(url[7:] if url.startswith("file://") else url)
        with p.open("rb") as f:
            f.seek(start)
            b = f.read(n)
        if len(b) != n:
            raise RangeError(f"{p}: {len(b)} of {n} bytes at {start}")
        return b
    end = start + n - 1
    req = urllib.request.Request(url, headers={"User-Agent": "maya-setup", "Range": f"bytes={start}-{end}",
                                               "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range", "")
            want = f"bytes {start}-{end}/"
            if r.status != 206 or not cr.startswith(want) or (total and cr != f"{want}{total}"):
                # (the whole file, or other bytes: the connection is closed here without reading the body)
                raise RangeError(f"the server answered status {r.status} ({cr or 'no Content-Range'}) to a request for "
                                 f"bytes {start}-{end}, not just those bytes")
            b = r.read(n + 1)
    except OSError as e:
        raise RangeError(str(e)) from e
    if len(b) != n:
        raise RangeError(f"{len(b)} bytes arrived for a range of {n}")
    return b


def fetch_marks(url: str, marks: dict, tries: int = TRIES) -> dict:
    """The marker rows {name: bytes} from shard 2 at `url`: its header (where they are), then the rows, every row's
    sha256 checked.  Each step is tried `tries` times; then SystemExit with the reason."""
    rows = marks["rows"]
    total = marks.get("bytes")
    why = ""
    for attempt in range(tries):
        try:
            n = struct.unpack("<Q", read_range(url, 0, 8, total))[0]
            if not 0 < n < 1 << 24:
                raise RangeError(f"a safetensors header of {n} bytes")
            head = json.loads(read_range(url, 8, n, total))
            base = 8 + n
            at = {}
            for name, (off, nb, _sha) in rows.items():
                e = head.get(name) or {}
                a, b = e.get("data_offsets") or (None, None)
                if e.get("dtype") != "BF16" or a is None or base + a != off or b - a != nb:
                    raise RangeError(f"{name} is not where the manifest says ({e or 'not in the header'})")
                at[name] = (off, nb)
            lo = min(o for o, _ in at.values())
            hi = max(o + nb for o, nb in at.values())
            if hi - lo > 1 << 20:
                raise RangeError(f"the marker rows span {hi - lo} bytes: not read in one piece")
            blob = read_range(url, lo, hi - lo, total)
            out = {}
            for name, (off, nb) in at.items():
                out[name] = blob[off - lo:off - lo + nb]
                if hashlib.sha256(out[name]).hexdigest() != rows[name][2]:
                    raise RangeError(f"{name}: the sha256 does not match the manifest's")
            return out
        except (RangeError, ValueError) as e:
            why = str(e)
            if attempt + 1 < tries:
                print(f"vision: the marker rows: {why}; trying once more ...", file=sys.stderr)
                time.sleep(2)
    raise SystemExit(f"vision: the picture markers (30 KB of {marks['file']}) could not be read: {why}.  Nothing large "
                     "was downloaded.  Check the connection and run the setup again (or --no-vision: text only).")


def shard(path: pathlib.Path):
    raw = np.memmap(path, np.uint8, mode="r")
    n = struct.unpack("<Q", bytes(raw[:8]))[0]
    head = json.loads(bytes(raw[8:8 + n]))
    head.pop("__metadata__", None)
    return raw, head, 8 + n


def geometry(cfg: dict) -> dict:
    """pack.py's vision keys: vision.<key> for config.json's numeric vision_config entries, and image_token_id."""
    out = {}
    for k, v in (cfg.get("vision_config") or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out["vision.%s" % k] = v
    if isinstance(cfg.get("image_token_id"), int):
        out["image_token_id"] = cfg["image_token_id"]
    if not out.get("vision.num_hidden_layers") or "image_token_id" not in out:
        raise SystemExit("config.json: no vision_config / image_token_id")
    return out


def pack(d: pathlib.Path, man: dict, marks_url: str | None = None) -> dict:
    names = [n for n in man["files"] if n.endswith(".safetensors")]
    if len(names) != 1:
        raise SystemExit("the manifest's vision entry: one safetensors shard expected")
    src = d / names[0]
    if not src.is_file() or src.stat().st_size != man["files"][names[0]][0]:
        raise SystemExit(f"{src}: missing or not whole")
    geo = geometry(json.loads((d / "config.json").read_text(encoding="utf-8")))
    mk = man["marks"]
    url = marks_url or HF.format(repo=man["repo"], revision=man["revision"], path=mk["file"])
    rows = fetch_marks(url, mk)
    raw, head, base = shard(src)
    order = sorted(head, key=lambda t: head[t]["data_offsets"][0])
    for t in order:
        if not t.startswith(("vision.", "aligner.")):
            raise SystemExit(f"{src.name}: {t} is not the vision tower's")
    entries, off = [], 0
    part = d / "vision.bin.part"
    with part.open("wb") as f:
        def put(name, dtype, shape, b):
            nonlocal off
            f.write(b)
            entries.append([name, dtype, off, len(b), list(shape)])
            off += len(b)
            pad = (-off) % ALIGN
            f.write(b"\0" * pad)
            off += pad
        for t in order:
            e = head[t]
            a, b = e["data_offsets"]
            put(t, e["dtype"], e["shape"], raw[base + a:base + b].tobytes())
        for name in MARKS:
            put(name, "BF16", [len(rows[name]) // 2], rows[name])
    # read back: every tensor as the checkpoint has it, bit for bit
    back = np.memmap(part, np.uint8, mode="r")
    for name, _dt, o, nb, _shape in entries:
        if name in rows:
            ok = back[o:o + nb].tobytes() == rows[name]
        else:
            a, b = head[name]["data_offsets"]
            ok = np.array_equal(back[o:o + nb], raw[base + a:base + b])
        if not ok:
            raise SystemExit(f"vision.bin: {name} differs from the checkpoint after the copy")
    del back, raw
    part.replace(d / "vision.bin")
    info = {"from": {"repo": man["repo"], "revision": man["revision"], "files": sorted(man["files"]),
                     "marks": mk["file"]}, "config": geo, "tensors": entries}
    (d / "vision.json").write_text(json.dumps(info))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the folder with shard 1 and config.json")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--marks-url", help="shard 2 elsewhere (a URL or a local file; default: Hugging Face, pinned)")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    info = pack(d, manifest_entry(pathlib.Path(a.manifest)), a.marks_url)
    nb = sum(e[3] for e in info["tensors"])
    print(f"vision.bin: {len(info['tensors'])} tensors ({nb / 1e9:.2f} GB), every one equal to the checkpoint's; "
          f"{info['config']['vision.num_hidden_layers']} blocks, image token {info['config']['image_token_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
