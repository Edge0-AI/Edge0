"""#3469 #3437: the 0.2 releases carry no flat engine zip, so a source checkout (UPDATE.bat, setup) fetches the engine package
for its card from the signed update feed and installs its files in engine/.

    python -m unittest tools.test_setup_feed_engine
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import engine_component as ec  # noqa: E402


class FeedEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.eng = self.root / "engine"
        self.eng.mkdir()
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        (self.eng / setup.EXE).write_bytes(b"old")
        self.package = self.root / "components" / "engine" / "0.1.46-test"
        self.package.mkdir(parents=True)
        (self.package / "BUILD.json").write_text(json.dumps({"version": "0.1.46", "archs": [89]}), encoding="utf-8")
        (self.package / setup.EXE).write_bytes(b"new")
        for name, value in (("home", lambda: self.root), ("say", lambda *a, **k: None), ("ok", lambda *a, **k: None),
                            ("warn", lambda *a, **k: None)):
            patch = mock.patch.object(setup, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(setup.acquisition, "require", lambda what: None)
        patch.start()
        self.addCleanup(patch.stop)

    def run_update(self, code=0):
        return mock.patch.object(setup.subprocess, "run", return_value=types.SimpleNamespace(returncode=code, stdout="", stderr="no feed"))

    def test_the_package_becomes_the_installed_engine(self):
        with self.run_update() as run, mock.patch.object(ec, "resolve", return_value=types.SimpleNamespace(root=self.package)):
            got = setup.engine_from_feed(self.eng, updating=True)
        self.assertEqual(got, self.eng)
        cmd = run.call_args.args[0]
        self.assertIn("update", cmd)
        self.assertIn("--no-control", cmd)
        self.assertNotIn("STRATA_NO_ACQUIRE", run.call_args.kwargs["env"])        # this one step may fetch
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"new")
        meta = json.loads((self.eng / "BUILD.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["version"], meta["source"]), ("0.1.46", "prebuilt"))
        self.assertEqual((self.eng / ".previous" / setup.EXE).read_bytes(), b"old")      # one generation to roll back to

    def test_no_feed_keeps_the_installed_engine(self):
        with self.run_update(code=6):
            self.assertIsNone(setup.engine_from_feed(self.eng, updating=True))
        self.assertEqual((self.eng / setup.EXE).read_bytes(), b"old")

    def test_a_release_without_the_zip_falls_back_to_the_feed(self):
        gpu = {"arch": 89, "archs": [89], "vram_gb": 8.0, "index": 0}
        (self.eng / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt", "archs": [89]}), encoding="utf-8")
        with mock.patch.object(setup, "engine_dir", lambda toolkit=13: self.eng),                 mock.patch.object(setup, "_urlopen", side_effect=OSError("HTTP Error 404")),                 mock.patch.object(setup, "engine_from_feed", return_value=self.eng) as feed:
            got = setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=True)
            self.assertEqual(got, self.eng)
            feed.assert_called_once()
            feed.reset_mock()
            setup.get_prebuilt(setup.PREBUILT_URL, gpu, "none", updating=True, toolkit=12)
            feed.assert_not_called()                      # the experimental CUDA 12 engine has no feed package here

    def test_the_minimum_is_the_0_2_engine(self):
        self.assertLess((0, 1, 45), setup.MIN_ENGINE)
        self.assertGreaterEqual((0, 1, 46), setup.WIN_HIP_MIN_ENGINE)


if __name__ == "__main__":
    unittest.main()
