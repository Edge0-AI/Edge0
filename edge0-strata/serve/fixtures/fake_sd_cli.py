"""A stand-in for stable-diffusion.cpp's sd-cli (serve/test_images.py): reads the arguments serve/images.py passes and
writes one small PNG per image whose pixels encode the seed, size and prompt, so a test can tell the images apart.
FAKE_SD_FAIL=1: exits 1 after a log line.  FAKE_SD_SLEEP=<s>: waits that long first (cancel tests).
FAKE_SD_ARGS=<file>: writes its argument list there (JSON)."""
import hashlib
import json
import os
import struct
import sys
import time
import zlib


def png(w: int, h: int, seed: int, text: str) -> bytes:
    digest = hashlib.sha256(f"{seed}|{w}x{h}|{text}".encode()).digest()
    row = b"\x00" + digest[:3] * 4
    raw = row * 4
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
            + chunk(b"tEXt", f"parameters\x00{text}, Seed: {seed}, Size: {w}x{h}".encode("latin-1", "replace"))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main(argv):
    a = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            a[argv[i]] = True
            i += 1
    if os.environ.get("FAKE_SD_ARGS"):
        with open(os.environ["FAKE_SD_ARGS"], "w", encoding="utf-8") as f:
            json.dump(argv, f)
    time.sleep(float(os.environ.get("FAKE_SD_SLEEP") or 0))
    print("[INFO ] loading tensors completed, taking 0.10s", flush=True)
    if os.environ.get("FAKE_SD_FAIL"):
        print("[ERROR] ggml_vulkan: ErrorDeviceLost", flush=True)
        return 1
    steps = int(a.get("--steps", 20))
    for s in range(1, steps + 1):
        print(f"  |{'=' * s}| {s}/{steps} - 0.01s/it", flush=True)
    print("[INFO ] get_learned_condition completed, taking 12 ms", flush=True)
    print("[INFO ] sampling completed, taking 0.08s", flush=True)
    print("[INFO ] decode_first_stage completed, taking 0.02s", flush=True)
    w, h, seed, n = int(a["-W"]), int(a["-H"]), int(a["-s"]), int(a.get("-b", 1))
    for k in range(n):
        with open(a["-o"] % k, "wb") as f:
            f.write(png(w, h, seed + k, a["-p"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
