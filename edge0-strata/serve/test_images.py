"""POST /v1/images/generations (serve/images.py): the request shape, the sd-cli command, the GPU turn and its policies,
over HTTP with a stand-in sd-cli (serve/fixtures/fake_sd_cli.py) and the mock engine.

    python -m serve.test_images
"""
import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import images as images_api
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "serve" / "fixtures" / "fake_sd_cli.py"
REAL_ARGS = images_api.sd_cli_args


def fake_args(cfg, job, out):
    """sd-cli's command, run through Python (the stand-in is a script)."""
    return [sys.executable, str(FAKE), *REAL_ARGS(cfg, job, out)[1:]]


def image_cfg(d, **kw):
    files = {}
    for k in ("diffusion_model", "vae", "llm"):
        p = Path(d) / f"{k}.bin"
        p.write_bytes(b"x" * 1024)
        files[k] = str(p)
    return {"exe": str(FAKE), "backend": "vulkan", "quant": "Q8_0", "args": ["--diffusion-fa"],
            "out_dir": str(Path(d) / "out"), **files, **kw}


class Request(unittest.TestCase):
    CFG = {"max_n": 4, "max_side": 2048}

    def test_defaults(self):
        j = images_api.parse_request({"prompt": "a cat"}, self.CFG)
        self.assertEqual((j["width"], j["height"], j["n"], j["steps"], j["response_format"]),
                         (1024, 1024, 1, 8, "b64_json"))
        self.assertTrue(0 <= j["seed"] < 2**31)

    def test_fields(self):
        j = images_api.parse_request({"prompt": "a cat", "size": "1536x1024", "n": 2, "steps": 4, "seed": 7,
                                      "response_format": "url", "quality": "hd", "model": "dall-e-3"}, self.CFG)
        self.assertEqual((j["width"], j["height"], j["n"], j["steps"], j["seed"], j["response_format"]),
                         (1536, 1024, 2, 4, 7, "url"))
        self.assertEqual(images_api.parse_request({"prompt": "x", "size": "auto"}, self.CFG)["width"], 1024)

    def test_refused(self):
        bad = [{}, {"prompt": ""}, {"prompt": 5}, {"prompt": "x", "n": 0}, {"prompt": "x", "n": 5},
               {"prompt": "x", "n": True}, {"prompt": "x", "size": "1000x1000"}, {"prompt": "x", "size": "128x128"},
               {"prompt": "x", "size": "4096x1024"}, {"prompt": "x", "size": "big"}, {"prompt": "x", "steps": 0},
               {"prompt": "x", "seed": -1}, {"prompt": "x", "seed": "1"}, {"prompt": "x", "response_format": "png"},
               {"prompt": "x", "output_format": "jpeg"}, {"prompt": "a\0b"}]
        for req in bad:
            with self.subTest(req=req), self.assertRaises(images_api.ImageRequestError):
                images_api.parse_request(req, self.CFG)


class Command(unittest.TestCase):
    def test_the_model_cards_command(self):
        """Unsloth's stable-diffusion.cpp command for Turbo: 8 steps, CFG 1, euler, the Turbo sigmas, --diffusion-fa."""
        cfg = {"exe": "sd-cli", "diffusion_model": "dit.gguf", "vae": "vae.st", "llm": "te.gguf",
               "args": ["--diffusion-fa", "--vae-tiling"]}
        job = {"prompt": "a sloth", "width": 1024, "height": 768, "n": 1, "steps": 8, "seed": 42}
        self.assertEqual(images_api.sd_cli_args(cfg, job, "o/img_%d.png"), [
            "sd-cli", "--diffusion-model", "dit.gguf", "--vae", "vae.st", "--llm", "te.gguf", "-p", "a sloth",
            "--steps", "8", "--cfg-scale", "1.0", "--sampling-method", "euler", "-W", "1024", "-H", "768",
            "--sigmas", "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0",
            "-s", "42", "-o", "o/img_%d.png", "--diffusion-fa", "--vae-tiling"])

    def test_other_steps_and_batch(self):
        cfg = {"exe": "sd-cli", "diffusion_model": "d", "vae": "v", "llm": "l"}
        cmd = images_api.sd_cli_args(cfg, {"prompt": "p", "width": 512, "height": 512, "n": 3, "steps": 4,
                                           "seed": 1}, "x")
        self.assertNotIn("--sigmas", cmd)          # Turbo's sigmas are an 8-step schedule
        self.assertEqual(cmd[cmd.index("-b") + 1], "3")

    def test_native_switch(self):
        cfg = {"exe": "sd-cli", "diffusion_model": "d", "vae": "v", "llm": "l", "native": True}
        job = {"prompt": "p", "width": 512, "height": 512, "n": 1, "steps": 8, "seed": 1}
        self.assertEqual(images_api.sd_cli_args(cfg, job, "x")[-2:], ["--model-args", "qwen_image_2_1_native=1"])
        cfg["native"] = False
        self.assertNotIn("--model-args", images_api.sd_cli_args(cfg, job, "x"))

    def test_env(self):
        env = images_api.image_env({"backend": "cuda", "gpu": 1, "env": {"X": 2}}, {"PATH": "p"})
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"], env["X"]), ("1", "PCI_BUS_ID", "2"))
        self.assertEqual(images_api.image_env({"backend": "cuda", "gpu": [1, 0]}, {})["CUDA_VISIBLE_DEVICES"], "1,0")     # a list is a device list
        with tempfile.TemporaryDirectory() as d:
            env = images_api.image_env({"backend": "hip", "gpu": 0, "lib_dirs": [d, d + "-missing"]}, {})
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["PATH" if os.name == "nt" else "LD_LIBRARY_PATH"], d)
        self.assertEqual(images_api.image_env({"backend": "vulkan", "gpu": 2}, {})["GGML_VK_VISIBLE_DEVICES"], "2")

    def test_config_checked(self):
        with self.assertRaises(ValueError):
            images_api.ImageGen({"exe": "x", "diffusion_model": "d", "vae": "v"})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                images_api.ImageGen(image_cfg(d, gpu_policy="sometimes"))


class FakeEngine:
    """An engine with the parts the GPU turn uses: alive/unload/restart, vram (elastic) and session files."""

    def __init__(self, elastic=False):
        self.up, self.calls, self.info = True, [], {"vram_elastic": elastic}
        self.spawn = (["strata"], ["--vram-elastic"] if elastic else [])
        self.unloaded = False

    def alive(self):
        return self.up

    def exit_code(self):
        return 0

    def unload(self):
        self.calls.append("unload")
        self.up, self.unloaded = False, True

    def restart(self):
        self.calls.append("restart")
        self.up, self.unloaded = True, False

    def vram(self, reserve):
        self.calls.append(("vram", reserve))
        return {"vram_free_mib": 9000}

    def session_file(self, action, path):
        self.calls.append((action, os.path.basename(path)))
        if action == "save":
            Path(path).write_bytes(b"kv")
        else:
            assert Path(path).read_bytes() == b"kv"
        return {"tokens": 123, "bytes": 2, "ms": 1.0}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        return iter(())


class Policy(unittest.TestCase):
    def svc(self, engine):
        tok = ByteTokenizer()
        return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_resolve(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageGen(image_cfg(d))
            eng = FakeEngine()
            svc = self.svc(eng)
            with mock.patch.object(svc, "free_vram_mib", return_value=20000):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            with mock.patch.object(svc, "free_vram_mib", return_value=None):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            with mock.patch.object(svc, "free_vram_mib", return_value=4000):
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "unload")
                svc.engine = FakeEngine(elastic=True)
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "shrink")
                gen.cfg["uma"] = True
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
                gen.cfg["uma"] = False
                gen.policy = "unload"
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "unload")
                svc.engine.up = False                   # nothing loaded: nothing to give back
                self.assertEqual(images_api.resolve_policy(svc, gen, 15000), "keep")
            self.assertEqual(images_api.resolve_policy(self.svc(MockEngine(ByteTokenizer(), "x")), gen, 1), "keep")

    def test_unload_saves_and_the_next_start_restores(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageGen(image_cfg(d, gpu_policy="unload"))
            eng = FakeEngine()
            svc = self.svc(eng)
            job = {"width": 1024, "height": 1024}
            with images_api.gpu_turn(svc, gen, job) as policy:
                self.assertEqual(policy, "unload")
                self.assertTrue(svc.fifo.locked())          # no chat turn meanwhile
                self.assertFalse(eng.up)
                self.assertTrue(svc.status["busy"])
            self.assertFalse(svc.fifo.locked())
            self.assertFalse(svc.status["busy"])
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls], ["save", "unload"])
            self.assertTrue(svc.restore_after_load)
            with svc.fifo:
                svc.ensure_loaded()                          # the next request: start, then the conversation back
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls],
                             ["save", "unload", "restart", "restore"])
            self.assertIsNone(svc.restore_after_load)
            self.assertEqual(list(Path(gen.dir).glob("llm-session-*")), [])

    def test_eager_reload(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageGen(image_cfg(d, gpu_policy="unload", reload="eager"))
            eng = FakeEngine()
            svc = self.svc(eng)
            with images_api.gpu_turn(svc, gen, {"width": 512, "height": 512}):
                pass
            self.assertEqual([c if isinstance(c, str) else c[0] for c in eng.calls],
                             ["save", "unload", "restart", "restore"])
            self.assertTrue(eng.up)

    def test_shrink_gives_the_cache_back(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageGen(image_cfg(d, gpu_policy="shrink"))
            eng = FakeEngine(elastic=True)
            svc = self.svc(eng)
            svc.vram_reserve = 512
            with images_api.gpu_turn(svc, gen, {"width": 1024, "height": 1024}):
                self.assertEqual(eng.calls[0][0], "vram")
                self.assertEqual(eng.calls[0][1], 6 * 1024)     # 2 GiB + 4 GiB per megapixel (+ 3 KiB of files)
            self.assertEqual(eng.calls[-1], ("vram", 512))
            self.assertTrue(eng.up)

    def test_keep_touches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            gen = images_api.ImageGen(image_cfg(d, gpu_policy="keep"))
            eng = FakeEngine()
            svc = self.svc(eng)
            with images_api.gpu_turn(svc, gen, {"width": 1024, "height": 1024}):
                pass
            self.assertEqual(eng.calls, [])


@mock.patch.object(images_api, "sd_cli_args", fake_args)
class OverHttp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nok")
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.images = images_api.ImageGen(image_cfg(self.tmp.name, keep_files=3))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def post(self, body, path="/v1/images/generations"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=30) as r:
                return r.status, r.read(), r.headers.get("Content-Type")
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read(), None

    def test_b64(self):
        code, body = self.post({"prompt": "a red fox", "seed": 11, "n": 2, "size": "512x768"})
        self.assertEqual(code, 200, body)
        self.assertEqual(len(body["data"]), 2)
        pngs = [base64.b64decode(x["b64_json"]) for x in body["data"]]
        sys.path.insert(0, str(FAKE.parent))
        import fake_sd_cli
        self.assertEqual(pngs, [fake_sd_cli.png(512, 768, 11, "a red fox"), fake_sd_cli.png(512, 768, 12, "a red fox")])
        self.assertEqual([x["seed"] for x in body["data"]], [11, 12])
        self.assertEqual(body["size"], "512x768")
        self.assertEqual(body["strata"]["steps"], 8)
        self.assertEqual(body["strata"]["sampling_s"], 0.08)
        self.assertEqual(body["strata"]["gpu_policy"], "keep")
        self.assertEqual(self.svc.images.totals["images"], 2)
        self.assertEqual([p.name for p in Path(self.svc.images.dir).iterdir()], [])   # nothing left behind

    def test_url(self):
        names = []
        for i in range(4):
            code, body = self.post({"prompt": "a lighthouse", "seed": i, "response_format": "url"})
            self.assertEqual(code, 200, body)
            url = body["data"][0]["url"]
            self.assertTrue(url.startswith(self.base + "/v1/images/files/"), url)
            names.append(url.rsplit("/", 1)[1])
            code, data, ctype = self.get(url[len(self.base):])
            self.assertEqual((code, ctype), (200, "image/png"))
            self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual(self.get("/v1/images/files/" + names[0])[0], 404)   # keep_files 3: the oldest went
        self.assertEqual(self.get("/v1/images/files/../../setup.py")[0], 404)
        self.assertEqual(self.get("/v1/images/files/" + "0" * 32 + ".png")[0], 404)

    def test_bad_request_and_failure(self):
        self.assertEqual(self.post({"prompt": "x", "size": "999x999"})[0], 400)
        with mock.patch.dict(os.environ, {"FAKE_SD_FAIL": "1"}):
            code, body = self.post({"prompt": "x"})
        self.assertEqual(code, 500)
        self.assertIn("ErrorDeviceLost", body["error"]["message"])
        self.assertEqual(self.svc.images.totals["failed"], 1)
        self.assertFalse(self.svc.status["busy"])
        self.assertFalse(self.svc.fifo.locked())

    def test_listed(self):
        code, data, _ = self.get("/v1/models")
        ids = [m["id"] for m in json.loads(data)["data"]]
        self.assertIn("qwen-image-2.1-turbo", ids)
        st = json.loads(self.get("/v1/status")[1])
        self.assertTrue(st["image_generation"]["enabled"])
        self.assertTrue(json.loads(self.get("/health")[1])["image_generation"])

    def test_not_set_up(self):
        self.svc.images = None
        code, body = self.post({"prompt": "x"})
        self.assertEqual((code, body["error"]["code"]), (404, "images_not_set_up"))

    def test_chat_waits_for_the_image(self):
        """One job at a time on the GPU: a chat request sent during an image job is answered after it."""
        done = {}
        with mock.patch.dict(os.environ, {"FAKE_SD_SLEEP": "1.5"}):
            t = threading.Thread(target=lambda: done.setdefault("img", (self.post({"prompt": "x"}), time.time())))
            t.start()
            time.sleep(0.5)
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
                {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.assertEqual(r.status, 200)
            chat_at = time.time()
            t.join()
        (code, _), img_at = done["img"]
        self.assertEqual(code, 200)
        self.assertGreaterEqual(chat_at, img_at - 0.05)


if __name__ == "__main__":
    unittest.main()
