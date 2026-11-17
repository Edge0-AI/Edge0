"""Pictures for DeepSeek-V4.1-Flash through the engine's OWN vision tower (strata-dsv41 --vision on, the default).

The Qwen engines get pictures from a separate reader process (strata-vision): it turns a picture into embeddings and the server writes
those into the file a `GENI` line names.  strata-dsv41 carries its own 319 MiB tower, so the server only has to
  1. decode each picture to RGB and bring it to a size the tower takes (both sides a multiple of 42 = patch 14 x merge 3, at most
     `vision.max_image_tokens` image tokens, at least `vision.min_pixels` pixels),
  2. put one run of image tokens into the prompt where the template wrote its single `<｜deepseek_image｜>` placeholder: the run is
     n_h * (n_w + 1) + 2 positions long (the picture's rows, a newline marker per row, the start and end markers), and
  3. write the pictures into the file of a `GENI` line (`DSI1`: int32 count, then per picture int64 start, int32 width, int32 height
     and its height x width x 3 RGB bytes), which the engine reads and encodes on its own GPU.
Decoding needs Pillow (the server asks for it by name when it is missing).  Nothing here starts a process or touches a GPU.
"""
from __future__ import annotations

import io
import os
import struct
import tempfile
from pathlib import Path

PATCH = 14
MERGE = 3
MAX_TOKENS = 1024
MIN_PIXELS = 0
OFF_WORDS = ("off", "no", "false", "0")


def pack_keys(pack_dir) -> dict:
    """The pack's ds4.txt as {key: value-string} ({} when it cannot be read)."""
    out = {}
    try:
        for line in (Path(pack_dir) / "ds4.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            k, _, v = line.partition(" ")
            if k:
                out[k] = v.strip()
    except OSError:
        pass
    return out


def args_after(args, flag):
    """The value that follows `flag` in an engine argument list (None when absent)."""
    for i, a in enumerate(args or []):
        if a == flag and i + 1 < len(args):
            return str(args[i + 1])
    return None


def vision_wanted(args) -> bool:
    """`--vision off` (or no/false/0) turns pictures off; everything else, including no flag at all, leaves them on."""
    v = args_after(args, "--vision")
    return v is None or v.strip().lower() not in OFF_WORDS


def span_length(width: int, height: int, patch: int = PATCH, merge: int = MERGE) -> int:
    """The image tokens one picture takes in the prompt: n_h * (n_w + 1) + 2 (the engine's own count, ds4_model.cu encode)."""
    gh, gw = height // patch, width // patch
    n_h, n_w = -(-gh // merge), -(-gw // merge)
    return n_h * (n_w + 1) + 2


def fit_size(width: int, height: int, max_tokens: int = MAX_TOKENS, min_pixels: int = MIN_PIXELS,
             patch: int = PATCH, merge: int = MERGE) -> tuple[int, int]:
    """The size to resize a width x height picture to: its own shape, both sides a multiple of patch * merge, scaled down until its
    span is within `max_tokens` and up until it has `min_pixels` (when that fits the budget).  Never smaller than one cell."""
    cell = patch * merge
    if width <= 0 or height <= 0:
        raise ValueError("the picture has no size")

    def snap(scale: float) -> tuple[int, int]:
        return max(cell, int(round(width * scale / cell)) * cell), max(cell, int(round(height * scale / cell)) * cell)

    scale = 1.0
    if width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
    w, h = snap(scale)
    while span_length(w, h, patch, merge) > max_tokens:       # shrink in small steps until the span fits the engine's limit
        scale *= 0.95
        w, h = snap(scale)
        if w == cell and h == cell:
            break
    return w, h


class DsPictures:
    """What the Service holds as `vision` for a DeepSeek-V4.1 engine that reads pictures itself.  It has the few members the
    Service asks of a picture encoder (alive, elastic, restart, unload, dir) and does the DeepSeek-specific work in build()."""

    elastic = False
    elastic_mib = 0

    def __init__(self, image_token: int, max_tokens: int = MAX_TOKENS, min_pixels: int = MIN_PIXELS, tmp_dir=None,
                 patch: int = PATCH, merge: int = MERGE):
        self.image_token = int(image_token)
        self.max_tokens = int(max_tokens)
        self.min_pixels = int(min_pixels)
        self.patch, self.merge = int(patch), int(merge)
        self.dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())

    @classmethod
    def from_config(cls, cfg: dict, tmp_dir=None):
        """-> DsPictures, or None: pictures are off (`--vision off`), or the pack has no vision tower (no vision.* keys / image token)."""
        args = cfg.get("args") or []
        if not vision_wanted(args):
            return None
        pack = args_after(args, "--pack")
        if not pack:
            return None
        k = pack_keys(pack)
        try:
            if int(k.get("vision.num_hidden_layers", "0")) <= 0 or "image_token_id" not in k:
                return None
            return cls(int(k["image_token_id"]), int(k.get("vision.max_image_tokens", MAX_TOKENS)),
                       int(float(k.get("vision.min_pixels", MIN_PIXELS))), tmp_dir,
                       int(k.get("vision.patch_size", PATCH)), int(k.get("vision.downsample_ratio", MERGE)))
        except ValueError:
            return None

    # ---- the members the Service asks of any picture encoder
    def alive(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def stop(self) -> None:
        pass

    # ---- the work
    @staticmethod
    def decode(data: bytes):
        """-> a Pillow RGB image (upright by its EXIF tag; transparency over white)."""
        try:
            from PIL import Image, ImageOps
        except ImportError:
            raise ValueError("DeepSeek reads pictures with Pillow, which is not installed here (python -m pip install pillow)") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            im = ImageOps.exif_transpose(im)
        except Exception as e:  # noqa: BLE001 - any decoder failure is the picture's
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            return bg
        return im if im.mode == "RGB" else im.convert("RGB")

    def prepare(self, data: bytes) -> tuple[int, int, bytes]:
        """-> (width, height, RGB bytes) of the picture at the size the tower takes."""
        im = self.decode(data)
        w, h = fit_size(im.width, im.height, self.max_tokens, self.min_pixels, self.patch, self.merge)
        if (w, h) != im.size:
            from PIL import Image
            im = im.resize((w, h), Image.LANCZOS)
        return w, h, im.tobytes()

    def encode(self, source) -> None:
        """Checks that a picture can be read and sized (the Service's pre-check of tool pictures); nothing is stored."""
        data = source if isinstance(source, bytes) else None
        if data is None:
            from serve.server import Vision
            data = Vision.load(source)
        self.prepare(data)

    def build(self, ids: list[int], pictures: list[bytes]) -> tuple[list[int], str]:
        """The prompt ids with each placeholder expanded to its picture's span, and the path of the DSI1 file for the GENI line.
        `pictures`: the bytes of every image of the request, in prompt order."""
        marks = [i for i, t in enumerate(ids) if t == self.image_token]
        if len(marks) != len(pictures):
            raise ValueError("the prompt and its images do not match")
        out: list[int] = []
        entries = []
        last = 0
        for pos, data in zip(marks, pictures):
            w, h, rgb = self.prepare(data)
            out += ids[last:pos]
            entries.append((len(out), w, h, rgb))
            out += [self.image_token] * span_length(w, h, self.patch, self.merge)
            last = pos + 1
        out += ids[last:]
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="strata-ds-pics-", suffix=".dsi", dir=str(self.dir))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(b"DSI1" + struct.pack("<i", len(entries)))
                for start, w, h, rgb in entries:
                    f.write(struct.pack("<qii", start, w, h) + rgb)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return out, path
