"""#4658: after `strata update` the old engine folder stays, so a picture reader written into it kept starting beside the new engine.

    python -m serve.test_reader_follows_engine
"""
import tempfile
import unittest
from pathlib import Path

import serve.server as server


class ReaderFollowsEngine(unittest.TestCase):
    def make(self, root: Path, *folders: str):
        for name in folders:
            d = root / "components" / "engine" / name
            d.mkdir(parents=True)
            (d / "strata.exe").write_bytes(b"x")
            (d / "strata-vision.exe").write_bytes(b"x")
        return root / "components" / "engine"

    def test_a_reader_in_the_older_engine_folder_moves_to_the_engine_in_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make(Path(tmp), "0.1.52-x", "0.1.54-x")
            cfg = {"vision": {"exe": str(engine / "0.1.52-x" / "strata-vision.exe"), "mmproj": "m.gguf"}}
            fixed = server.with_current_reader(cfg, engine / "0.1.54-x")
            self.assertEqual(fixed["vision"]["exe"], str(engine / "0.1.54-x" / "strata-vision.exe"))
            self.assertEqual(fixed["vision"]["mmproj"], "m.gguf")

    def test_a_reader_already_in_the_engine_folder_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make(Path(tmp), "0.1.54-x")
            cfg = {"vision": {"exe": str(engine / "0.1.54-x" / "strata-vision.exe")}}
            self.assertIs(server.with_current_reader(cfg, engine / "0.1.54-x"), cfg)

    def test_a_reader_somewhere_else_is_the_persons_choice_and_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make(Path(tmp), "0.1.54-x")
            mine = Path(tmp) / "mine"
            mine.mkdir()
            (mine / "strata-vision.exe").write_bytes(b"x")
            cfg = {"vision": {"exe": str(mine / "strata-vision.exe")}}
            self.assertIs(server.with_current_reader(cfg, engine / "0.1.54-x"), cfg)

    def test_a_missing_reader_is_still_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make(Path(tmp), "0.1.54-x")
            cfg = {"vision": {"exe": str(engine / "0.1.40-x" / "strata-vision.exe")}}
            fixed = server.with_current_reader(cfg, engine / "0.1.54-x")
            self.assertEqual(fixed["vision"]["exe"], str(engine / "0.1.54-x" / "strata-vision.exe"))


if __name__ == "__main__":
    unittest.main()
