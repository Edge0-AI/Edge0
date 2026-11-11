"""Two Intel Arc Pro B65 cards (device id 0xe222): the Terminal must see both and name both for a launch that names none (fake sysfs, no hardware)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_cli.launch import catalog as C, runner as R, settings as S      # noqa: E402


def fake_drm(root: Path, ids: list[str]) -> None:
    for i, did in enumerate(ids):
        dev = root / f"card{i}" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text("0x8086\n")
        (dev / "device").write_text(did + "\n")


class IntelCards(unittest.TestCase):
    def cards(self, ids):
        with tempfile.TemporaryDirectory() as d:
            fake_drm(Path(d), ids)
            return C.intel_arc_cards(d)

    def test_two_b65_are_two_cards_of_32_gb(self):
        cards = self.cards(["0xe222", "0xe222"])
        self.assertEqual([(c["index"], c["vram_gb"]) for c in cards], [(0, 32.0), (1, 32.0)])
        self.assertTrue(all("B65" in c["name"] for c in cards))

    def test_the_other_battlemage_ids_are_listed(self):
        for did in ("0xe221", "0xe211", "0xe212", "0xe20b", "0xe223"):
            self.assertEqual(len(self.cards([did])), 1, did)

    def test_an_integrated_intel_gpu_is_never_a_card(self):
        for did in ("0x46a6", "0x7d55", "0xa7a0", "0x9a49", "0x4680", "0x64a0"):
            self.assertEqual(self.cards([did]), [], did)
        self.assertEqual(len(self.cards(["0x7d55", "0xe222"])), 1)       # an iGPU beside a B65: one card

    def test_both_tables_list_the_same_discrete_ids(self):
        import ast
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "INTEL_ARC")
        theirs = {"0x" + k: v[0].replace("Arc ", "") for k, v in ast.literal_eval(node.value).items()}
        ours = {k: v[0].replace("Arc ", "") for k, v in C.INTEL_ARC.items()}
        self.assertEqual(set(theirs), set(ours))
        self.assertEqual({k: v.replace(" ", "") for k, v in theirs.items()}, {k: v.replace(" ", "") for k, v in ours.items()})

    def test_mixed_sizes_get_roles_like_any_other_pair(self):
        cards = self.cards(["0xe222", "0xe20b"])                         # 32 GB + 12 GB
        hw = {"ram_gb": 128.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": cards}
        cat = C.Catalog(root=Path(tempfile.mkdtemp()), hardware=hw)
        cat.installed_backend = lambda: "sycl"
        got = R.cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_S"), cat, base={})
        self.assertEqual(sorted(got), [0, 1])

    def test_one_card_names_nothing(self):
        hw = {"ram_gb": 128.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": self.cards(["0xe222"])}
        cat = C.Catalog(root=Path(tempfile.mkdtemp()), hardware=hw)
        cat.installed_backend = lambda: "sycl"
        self.assertIsNone(R.cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_S"), cat, base={}))

    def test_a_launch_that_names_no_card_gets_both_b65(self):
        hw = {"ram_gb": 128.0, "os": "Linux", "cpu": {"avx2": True}, "gpus": self.cards(["0xe222", "0xe222"])}
        cat = C.Catalog(root=Path(tempfile.mkdtemp()), hardware=hw)
        cat.installed_backend = lambda: "sycl"
        s = S.LaunchSettings(family="qwen", model="IQ3_S")
        got = R.cards_for_launch(s, cat, base={})
        self.assertEqual(sorted(got), [0, 1])
        cfg = {}
        R.apply_cards(cfg, got)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        self.assertEqual(cfg["layer_split"], "auto")


if __name__ == "__main__":
    unittest.main()
