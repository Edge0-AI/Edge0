"""#1485: the card an engine update compiles for is one the installed models use (and CUDA 13 can compile for), not
the card with the most VRAM - a Tesla V100 beside two RTX 4090s made ./update.sh build for sm_70 with CUDA 13.

    python -m unittest tools.test_setup_update_gpu -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402

V100 = {"index": 1, "name": "Tesla V100-PCIE-32GB", "vram_gb": 32.0, "arch": "70", "driver": "580"}
R4090 = lambda i: {"index": i, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89", "driver": "580"}  # noqa: E731


class UpdateEngineGpu(unittest.TestCase):
    def pick(self, found, configs, toolkit=13, pinned=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = []
        for n, cfg in enumerate(configs):
            p = Path(tmp.name) / f"strata-{n}.json"
            p.write_text(json.dumps(cfg))
            paths.append(p)
        with mock.patch.object(setup, "gpus", return_value=found), \
                mock.patch.object(setup, "installed_configs", return_value=paths), \
                mock.patch.object(setup, "GPU_PICK", pinned):
            return setup.update_engine_gpu(toolkit)

    def test_the_models_cards_not_the_biggest_one(self):
        g = self.pick([R4090(0), V100, R4090(2)], [{"exe": "x", "args": [], "gpu": [0, 2]}])
        self.assertEqual(g["arch"], "89")
        self.assertEqual(g["count"], 3)

    def test_a_card_cuda_13_cannot_compile_for_is_left_out_without_a_config(self):
        self.assertEqual(self.pick([R4090(0), V100], [])["index"], 0)

    def test_the_cuda_12_engine_may_take_the_volta(self):
        self.assertEqual(self.pick([R4090(0), V100], [], toolkit=12)["index"], 1)

    def test_only_a_volta_is_still_taken(self):
        self.assertEqual(self.pick([V100], [])["index"], 1)

    def test_no_nvidia_card(self):
        self.assertIsNone(self.pick([], []))


if __name__ == "__main__":
    unittest.main()
