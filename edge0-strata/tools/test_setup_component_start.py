"""Setup can start a component-backed config without treating the reference as a file.

    python -m unittest tools.test_setup_component_start -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class ComponentStart(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = self.root / "strata-iq3_s.json"
        self.gguf = self.root / "model.gguf"
        self.gguf.write_bytes(b"GGUF")
        self.cards = [{"index": i, "name": f"RTX {i}", "arch": "89", "vram_gb": 24.0} for i in range(2)]

    def run_start(self, exe="component:strata-engine", backend="cuda", gpu=0, win=False, keep=None):
        cfg = {"exe": exe, "backend": backend, "args": ["--native", str(self.gguf)],
               "gpu": gpu, "gpus_asked": True, "model_name": "test", "port": 8080}
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        self.output = io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(contextlib.redirect_stdout(self.output))
            st.enter_context(mock.patch.object(setup, "WIN", win))
            st.enter_context(mock.patch.dict(os.environ, {"STRATA_EXECV": ""}))
            st.enter_context(mock.patch.object(setup, "gpus", return_value=self.cards))
            st.enter_context(mock.patch.object(setup, "amd_gpus", return_value=[
                {"index": i, "name": f"Radeon {i}", "arch": "gfx1201", "vram_gb": 24.0} for i in range(2)]))
            st.enter_context(mock.patch.object(setup, "check_gpus"))
            st.enter_context(mock.patch.object(setup, "split_budget", return_value=False))
            st.enter_context(mock.patch.object(setup, "split_mmap", return_value=False))
            st.enter_context(mock.patch.object(setup, "recommend_remote_expert_opt"))
            st.enter_context(mock.patch.object(setup, "old_gpus_opt_in", return_value=[]))
            for name in ("engine_runs_on", "engine_archs_hip", "hip_runtime_beside_exe"):
                st.enter_context(mock.patch.object(setup, name, side_effect=AssertionError("legacy engine lookup: " + name)))
            self.server = st.enter_context(mock.patch.object(setup.subprocess, "call", return_value=0))
            return setup.start(self.config, None, gpu=gpu, open_browser=False, yes=True, keep=keep)

    def test_component_reaches_the_server_and_keeps_its_reference(self):
        self.assertEqual(self.run_start(), 0)
        self.server.assert_called_once()
        argv = self.server.call_args.args[0]
        self.assertIn(str(setup.ROOT / "serve" / "server.py"), argv)
        self.assertEqual(argv[argv.index("--config") + 1], str(self.config))
        self.assertEqual(self.server.call_args.kwargs["env"][setup.acquisition.ENV], "1")
        self.assertEqual(json.loads(self.config.read_text())["exe"], "component:strata-engine")

    def test_component_uses_saved_gpu_without_looking_for_a_source_build(self):
        self.assertEqual(self.run_start(gpu=None), 0)
        self.server.assert_called_once()

    def test_component_can_save_several_gpus_without_looking_for_a_source_build(self):
        self.assertEqual(self.run_start(gpu=[0, 1]), 0)
        self.assertEqual(json.loads(self.config.read_text())["gpu"], [0, 1])

    def test_windows_hip_component_does_not_repair_the_legacy_engine_folder(self):
        self.assertEqual(self.run_start(backend="hip", win=True), 0)
        self.server.assert_called_once()

    def test_hip_component_does_not_read_source_build_architectures(self):
        self.assertEqual(self.run_start(backend="hip", gpu=[0, 1]), 0)
        self.server.assert_called_once()

    def test_missing_model_file_is_still_reported(self):
        self.gguf.unlink()
        with self.assertRaises(SystemExit):
            self.run_start()
        self.assertIn(str(self.gguf), self.output.getvalue())
        self.server.assert_not_called()

    def test_missing_literal_engine_path_is_still_reported(self):
        missing = str(self.root / "strata")
        with self.assertRaises(SystemExit):
            self.run_start(exe=missing)
        self.assertIn(missing, self.output.getvalue())
        self.server.assert_not_called()

    def test_component_reference_survives_saving_start_settings(self):
        self.assertEqual(self.run_start(keep={"host": "127.0.0.1", "vram_reserve_mib": 1024}), 0)
        saved = json.loads(self.config.read_text())
        self.assertEqual(saved["exe"], "component:strata-engine")
        self.assertEqual(saved["args"][-2:], ["--vram-reserve-mib", "1024"])


if __name__ == "__main__":
    unittest.main()
