"""serve/test_ds_pictures.py: DeepSeek-V4.1 pictures through the engine's own tower (serve/ds_pictures.py): the size rule, the span, the
config switch, the prompt expansion and the DSI1 file the engine reads.  CPU only; the decode tests need Pillow and skip without it."""
import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import ds_pictures as D  # noqa: E402

try:
    from PIL import Image
except ImportError:                       # pragma: no cover
    Image = None

IMG = 129264


def png(w, h, color=(220, 20, 20)) -> bytes:
    im = Image.new("RGB", (w, h), color)
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


class Geometry(unittest.TestCase):
    def test_span_matches_the_engine(self):
        # mkimg.py's 672 x 672 picture: 16 x 17 + 2 = 274 positions
        self.assertEqual(D.span_length(672, 672), 274)
        self.assertEqual(D.span_length(42, 42), 1 * 2 + 2)
        self.assertEqual(D.span_length(14, 14), 1 * 2 + 2)             # a partial merge cell still counts one

    def test_sizes_are_multiples_of_42_within_the_budget(self):
        for w, h in ((100, 100), (672, 672), (4000, 3000), (3000, 4000), (1920, 1080), (50, 3000)):
            nw, nh = D.fit_size(w, h)
            self.assertEqual((nw % 42, nh % 42), (0, 0), (w, h))
            self.assertLessEqual(D.span_length(nw, nh), D.MAX_TOKENS, (w, h))
            self.assertGreaterEqual(min(nw, nh), 42)

    def test_small_pictures_grow_to_min_pixels(self):
        w, h = D.fit_size(200, 200, min_pixels=295936)
        self.assertGreaterEqual(w * h, 295936 * 0.8)                   # rounded to the 42 grid

    def test_a_huge_picture_shrinks_to_the_limit(self):
        w, h = D.fit_size(8000, 6000, 1024)
        self.assertLessEqual(D.span_length(w, h), 1024)
        self.assertGreater(D.span_length(w, h), 800)                   # but not far below it


class Config(unittest.TestCase):
    def pack(self, tower=True):
        d = tempfile.mkdtemp()
        lines = ["hidden_size 5120"]
        if tower:
            lines += ["vision.num_hidden_layers 32", "vision.patch_size 14", "vision.downsample_ratio 3", "vision.max_image_tokens 1024",
                      "vision.min_pixels 295936", f"image_token_id {IMG}"]
        Path(d, "ds4.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return d

    def test_on_by_default(self):
        p = D.DsPictures.from_config({"args": ["--pack", self.pack(), "--max-context", "8192"]})
        self.assertIsNotNone(p)
        self.assertEqual((p.image_token, p.max_tokens, p.min_pixels), (IMG, 1024, 295936))

    def test_vision_on_flag(self):
        self.assertIsNotNone(D.DsPictures.from_config({"args": ["--pack", self.pack(), "--vision", "on"]}))

    def test_vision_off_turns_it_off(self):
        for word in ("off", "OFF", "no", "false", "0"):
            self.assertIsNone(D.DsPictures.from_config({"args": ["--pack", self.pack(), "--vision", word]}), word)

    def test_a_pack_without_a_tower_has_none(self):
        self.assertIsNone(D.DsPictures.from_config({"args": ["--pack", self.pack(tower=False)]}))
        self.assertIsNone(D.DsPictures.from_config({"args": []}))
        self.assertIsNone(D.DsPictures.from_config({"args": ["--pack", "/nonexistent"]}))


@unittest.skipIf(Image is None, "Pillow is not installed")
class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = D.DsPictures(IMG, 1024, 0, self.tmp)

    def read(self, path):
        raw = Path(path).read_bytes()
        self.assertEqual(raw[:4], b"DSI1")
        n = struct.unpack("<i", raw[4:8])[0]
        pos, out = 8, []
        for _ in range(n):
            start, w, h = struct.unpack("<qii", raw[pos:pos + 16])
            pos += 16
            out.append((start, w, h, raw[pos:pos + w * h * 3]))
            pos += w * h * 3
        self.assertEqual(pos, len(raw))
        return out

    def test_one_picture(self):
        ids = [5, 6, IMG, 7, 8]
        out, path = self.p.build(ids, [png(672, 672)])
        self.assertEqual(out[:2], [5, 6])
        self.assertEqual(out.count(IMG), 274)
        self.assertEqual(out[2 + 274:], [7, 8])
        (start, w, h, rgb), = self.read(path)
        self.assertEqual((start, w, h), (2, 672, 672))
        self.assertEqual(rgb[:3], bytes((220, 20, 20)))
        self.assertEqual(out[start], IMG)

    def test_two_pictures_keep_their_order_and_starts(self):
        ids = [1, IMG, 2, 3, IMG, 4]
        out, path = self.p.build(ids, [png(42, 42, (0, 255, 0)), png(84, 42, (0, 0, 255))])
        a, b = self.read(path)
        self.assertEqual((a[1], a[2]), (42, 42))
        self.assertEqual((b[1], b[2]), (84, 42))
        self.assertEqual(a[0], 1)
        self.assertEqual(b[0], 1 + D.span_length(42, 42) + 2)
        self.assertEqual(a[3][:3], bytes((0, 255, 0)))
        self.assertEqual(b[3][-3:], bytes((0, 0, 255)))
        self.assertEqual(out[-1], 4)

    def test_counts_must_match(self):
        with self.assertRaises(ValueError):
            self.p.build([1, IMG, 2], [png(42, 42), png(42, 42)])
        with self.assertRaises(ValueError):
            self.p.build([1, 2], [png(42, 42)])

    def test_odd_sizes_and_transparency_are_brought_to_the_grid(self):
        im = Image.new("RGBA", (100, 61), (0, 0, 0, 0))                # fully transparent: white
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        w, h, rgb = self.p.prepare(buf.getvalue())
        self.assertEqual((w % 42, h % 42), (0, 0))
        self.assertEqual(rgb[:3], b"\xff\xff\xff")

    def test_a_bad_image_is_a_value_error(self):
        with self.assertRaises(ValueError):
            self.p.prepare(b"not an image")

    def test_the_engine_limit_holds_for_a_huge_picture(self):
        w, h, rgb = self.p.prepare(png(3000, 2000))
        self.assertLessEqual(D.span_length(w, h), 1024)
        self.assertEqual(len(rgb), w * h * 3)


# ---- the server side: a picture reaches the engine as a GENI file, text requests are untouched, pictures off is the old 400
import base64  # noqa: E402
import json  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from serve import frontend  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import Service, default_model_name, serve  # noqa: E402
from serve.test_dsv41 import TEMPLATE, DsEngine, mini_tokenizer  # noqa: E402


class SeenEngine(DsEngine):
    """Keeps the picture file's bytes while the engine is 'reading' it (the server deletes the file after the run)."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.seen_file = Path(embeddings).read_bytes() if embeddings else None
        self.seen_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


@unittest.skipIf(Image is None, "Pillow is not installed")
class Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = mini_tokenizer()
        cls.img = cls.tok.encode("<｜deepseek_image｜>", parse_special=True)[0]

    def start(self, pictures=True):
        engine = SeenEngine(self.tok, "ok")
        vision = D.DsPictures(self.img, 1024, 0, tempfile.mkdtemp()) if pictures else None
        svc = Service(engine, self.tok, ChatTemplate(TEMPLATE), model_name=default_model_name({"arch": "deepseek41"}), vision=vision)
        svc.model_arch = "deepseek41"
        svc.set_family("deepseek41")
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(frontend.use_family, None)
        self.engine, self.svc = engine, svc
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def picture_body(self):
        url = "data:image/png;base64," + base64.b64encode(png(672, 672)).decode()
        return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": "What colour is this?"}]}], "max_tokens": 8}

    def test_a_picture_reaches_the_engine_as_a_geni_file(self):
        self.start()
        code, out = self.post(self.picture_body())
        self.assertEqual(code, 200, out)
        self.assertEqual(self.engine.seen_ids.count(self.img), 274)
        raw = self.engine.seen_file
        self.assertEqual(raw[:4], b"DSI1")
        n, = struct.unpack("<i", raw[4:8])
        start, w, h = struct.unpack("<qii", raw[8:24])
        self.assertEqual((n, w, h), (1, 672, 672))
        self.assertEqual(self.engine.seen_ids[start], self.img)
        self.assertEqual(len(raw), 24 + 672 * 672 * 3)
        left = [p for p in Path(self.svc.vision.dir).glob("strata-ds-pics-*.dsi")]
        self.assertEqual(left, [], "the picture file is removed after the run")

    def test_text_requests_are_the_same_with_pictures_on(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        self.start(pictures=True)
        self.assertEqual(self.post(body)[0], 200)
        with_pics = list(self.engine.seen_ids)
        self.assertIsNone(self.engine.seen_file)
        self.start(pictures=False)
        self.assertEqual(self.post(body)[0], 200)
        self.assertEqual(self.engine.seen_ids, with_pics)

    def test_pictures_off_keeps_the_400(self):
        self.start(pictures=False)
        code, out = self.post(self.picture_body())
        self.assertEqual(code, 400)
        self.assertIn("without pictures for DeepSeek", json.dumps(out))

    def test_a_follow_up_with_the_picture_in_history_works(self):
        self.start()
        body = self.picture_body()
        body["messages"] += [{"role": "assistant", "content": "red and blue"}, {"role": "user", "content": "and the middle?"}]
        code, out = self.post(body)
        self.assertEqual(code, 200, out)
        self.assertEqual(self.engine.seen_ids.count(self.img), 274)


if __name__ == "__main__":
    unittest.main()
