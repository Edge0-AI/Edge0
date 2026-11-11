"""I6 (tools and images review, 2026-10-22): a picture a tool returned, with an elastic image encoder that is not running,
used to write to a process that did not exist (AttributeError, then a 400 "malformed request" that every later turn of the
conversation repeated).  It starts the encoder like a picture of the user does; when it cannot start, the picture becomes
an "[image omitted: why]" note.

    python -m serve.test_vision_tool_image
"""
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_vision_elastic import ElasticVision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]


def tool_turn():
    return [{"role": "user", "content": "look at it"},
            {"role": "tool", "tool_call_id": "t1", "content": [{"type": "image", "source": "x.png"}]}]


class FailingStart(ElasticVision):
    def restart(self):
        raise RuntimeError("the vision encoder did not start: it ended without saying anything (exit code 1)")


class ToolImage(unittest.TestCase):
    def service(self, vision):
        tok = ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)

    def test_an_elastic_encoder_is_started_for_a_tool_picture(self):
        vision = ElasticVision(tempfile.mkdtemp())
        svc = self.service(vision)
        messages = tool_turn()
        svc._note_unreadable_tool_images(messages)
        self.assertEqual(vision.starts, 1)
        self.assertEqual(messages[1]["content"][0]["type"], "image")      # kept: it was read
        svc._note_unreadable_tool_images(messages)                        # the running encoder is used as it is
        self.assertEqual(vision.starts, 1)

    def test_an_encoder_that_cannot_start_leaves_a_note_not_a_400(self):
        svc = self.service(FailingStart(tempfile.mkdtemp()))
        messages = tool_turn()
        svc._note_unreadable_tool_images(messages)                        # no exception
        part = messages[1]["content"][0]
        self.assertEqual(part["type"], "text")
        self.assertIn("[image omitted:", part["text"])
        self.assertIn("did not start", part["text"])

    def test_prepare_survives_a_tool_picture_with_an_elastic_encoder(self):
        vision = ElasticVision(tempfile.mkdtemp())
        svc = self.service(vision)
        self.addCleanup(lambda: getattr(svc.embeddings, "path", None) and svc.embeddings.path.unlink(missing_ok=True))
        svc.prepare(tool_turn(), None, {})
        self.assertEqual(vision.starts, 1)

    def test_encode_without_a_running_reader_is_a_clear_value_error(self):
        v = Vision.__new__(Vision)
        v.dir, v.cache, v.proc = Path(tempfile.mkdtemp()), {}, None
        import threading
        v.lock = threading.Lock()
        with mock.patch.object(Vision, "normalize", return_value=b"png"), self.assertRaises(ValueError) as cm:
            v.encode(b"bytes")
        self.assertIn("picture reader is not running", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
