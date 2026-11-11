"""The elastic image encoder (the vision section's "elastic": true with "vram_elastic"): the first picture makes the
expert cache keep the encoder's VRAM free and then starts it, later pictures use it as it is, an idle encoder stops and
the cache takes its VRAM back, a POST /v1/vram reserve stacks on the encoder's share, and the engine's load never starts
it on its own."""
import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGE = {"type": "image", "source": "x.png"}


class ElasticVision:
    """An encoder process stand-in: starts, stops, and turns every picture into three image tokens."""
    def __init__(self, d, elastic=True):
        self.elastic, self.elastic_mib, self.elastic_idle_s = elastic, 2048, 60.0
        self.up, self.starts, self.unloads = False, 0, 0
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def encode(self, source):
        assert self.up, "a picture reached an encoder that was not started"
        return self.rows, 3


class Elastic(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.tok = ByteTokenizer()
        self.engine = VramEngine(self.tok)
        self.engine.loaded = True
        self.vision = ElasticVision(d)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.addCleanup(self.drop_embeddings)

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, "embeddings", None), "path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def picture(self):
        self.svc.prepare([{"role": "user", "content": [{"type": "text", "text": "what is it"}, IMAGE]}], None, {})

    def test_the_first_picture_makes_room_then_starts_the_encoder(self):
        self.picture()
        self.assertEqual(self.engine.calls, [2048])         # the cache keeps the encoder's share free first
        self.assertEqual(self.vision.starts, 1)
        self.assertTrue(self.svc.vision_held)
        self.picture()                                      # the next one uses the running encoder
        self.assertEqual(self.engine.calls, [2048])
        self.assertEqual(self.vision.starts, 1)

    def test_an_idle_encoder_stops_and_the_cache_grows_back_whole(self):
        self.picture()
        self.assertEqual(self.svc._vision_idle_down(), "busy")             # a picture came just now
        self.svc.last_image_at = time.time() - 61
        with self.svc.fifo:                                                # a request holds the turn
            self.assertEqual(self.svc._vision_idle_down(), "busy")
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertFalse(self.vision.up)
        self.assertFalse(self.svc.vision_held)
        self.assertEqual(self.engine.calls, [2048, 0])      # 0, not None: the full size, not the start's reserve
        self.assertEqual(self.svc._vision_idle_down(), "nothing")
        self.picture()                                      # and the next picture starts it again
        self.assertEqual(self.vision.starts, 2)
        self.assertEqual(self.engine.calls, [2048, 0, 2048])

    def test_a_vram_reserve_stacks_on_the_encoders_share(self):
        self.svc.vram(4000)
        self.picture()
        self.assertEqual(self.engine.calls, [4000, 6048])
        self.svc.vram(1000)                                 # asked while the encoder runs
        self.assertEqual(self.engine.calls[-1], 3048)
        self.svc.last_image_at = 0
        self.svc._vision_idle_down()
        self.assertEqual(self.engine.calls[-1], 1000)       # back to the asked reserve, not to 0

    def test_an_engine_that_cannot_shrink_still_gets_its_encoder(self):
        self.engine.elastic = False
        self.picture()
        self.assertEqual(self.vision.starts, 1)
        self.assertFalse(self.svc.vision_held)
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertEqual(self.engine.calls, [2048])         # the refused shrink only: nothing to give back

    def test_an_encoder_that_fails_to_start_is_a_503_not_a_dropped_connection(self):
        def fail():
            raise RuntimeError("the vision encoder did not start: cudaMalloc failed: out of memory")
        self.vision.restart = fail
        with self.assertRaises(EngineDied) as e:            # the handlers answer EngineDied with a 503
            self.picture()
        self.assertIn("the image encoder failed to start", str(e.exception))
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertEqual(self.engine.calls, [2048, 0])      # the share made for it goes back

    def test_the_stop_is_logged_when_the_cache_never_made_room(self):
        self.engine.elastic = False                         # a layer split: the engine refuses --vram-elastic
        self.picture()
        self.svc.last_image_at = 0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertIn("the image encoder stopped after 60 s without a picture", out.getvalue())
        self.assertNotIn("expert cache", out.getvalue())

    def test_the_load_never_starts_an_elastic_encoder(self):
        self.assertFalse(self.svc._vision_down())
        self.svc.vision = ElasticVision(tempfile.mkdtemp(), elastic=False)
        self.assertTrue(self.svc._vision_down())            # a non-elastic encoder that is down is started again

    def test_an_unload_lets_go_of_the_share(self):
        self.picture()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.svc.vision_held)
        self.assertFalse(self.vision.up)

    def test_the_config_keys(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True, "elastic_mib": 3000,
                    "elastic_idle_s": 30}, lazy=False)          # elastic: nothing starts with the server
        self.assertTrue(v.elastic and v.elastic_mib == 3000 and v.elastic_idle_s == 30 and not v.alive())
        v = Vision({"exe": "e", "mmproj": "m", "model": "t"}, lazy=True)
        self.assertFalse(v.elastic)
        self.assertEqual((v.elastic_mib, v.elastic_idle_s), (2048, 120.0))


if __name__ == "__main__":
    unittest.main()
