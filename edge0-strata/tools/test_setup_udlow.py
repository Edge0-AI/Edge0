"""Unsloth's UD-IQ1_S / UD-IQ1_M DeepSeek-V4.1-Flash sets (2026-10-30): pins, the app's file recognition, the fit numbers and the pack call.
Run: python -m pytest tools/test_setup_udlow.py"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("setup_udlow", ROOT / "setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

from strata_control import dsv41_fit, dsv41_sets  # noqa: E402

SETS = {"UD-IQ1_S": (232_979_608_420, 49_120_285_248), "UD-IQ1_M": (250_404_346_724, 49_707_332_128)}


class Pins(unittest.TestCase):
    def test_every_file_is_pinned(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        for model in SETS:
            md = fam["models"][model]
            self.assertEqual(md["recipe"], "udlow")
            names = [*md["shards"], md["engram_shard"]]
            self.assertEqual(len(names), 5)
            for n in names:
                self.assertIn(n, setup.DSV41_FILES)
                self.assertIs(setup.REPO_PINS["unsloth/DeepSeek-V4.1-Flash-1017-GGUF"], setup.DSV41_FILES)
            self.assertEqual(sum(setup.DSV41_FILES[n][0] for n in names), SETS[model][0])
            self.assertEqual(setup.DSV41_FILES[md["shards"][1]][0], SETS[model][1])
            self.assertEqual(setup.DSV41_FILES[md["engram_shard"]][0], 110_595_290_592)

    def test_the_engram_shard_is_the_same_data_as_isas_file_but_its_own_pin(self):
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["UD-IQ1_S"]
        self.assertNotEqual(setup.DSV41_FILES[md["engram_shard"]][1], setup.DSV41_ALL_FILES[setup.DSV41_ISTA_ENGRAM][1])

    def test_ista_sets_are_unchanged(self):
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]
        self.assertEqual(md["recipe"], "ista")
        self.assertEqual(len(md["shards"]), 4)


class Recognised(unittest.TestCase):
    def test_a_whole_set_with_its_own_engram(self):
        for model in SETS:
            t = dsv41_sets.tables()["sets"][model]
            files = {**t["shards"], **t["engram_alt"]}
            rows = [r for r in dsv41_sets.match(files) if r["model"] == model]
            self.assertTrue(rows and rows[0]["complete"], model)
            self.assertEqual(rows[0]["engram"], next(iter(t["engram_alt"])))

    def test_four_files_and_an_ista_engram_is_a_whole_set_too(self):
        t = dsv41_sets.tables()
        files = dict(t["sets"]["UD-IQ1_M"]["shards"])
        files[setup.DSV41_ISTA_ENGRAM] = setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM][0]
        row = dsv41_sets.match(files)[0]
        self.assertEqual((row["model"], row["complete"], row["engram"]), ("UD-IQ1_M", True, setup.DSV41_ISTA_ENGRAM))

    def test_a_cut_shard_is_refused(self):
        t = dsv41_sets.tables()["sets"]["UD-IQ1_S"]
        files = dict(t["shards"])
        name = list(files)[1]
        files[name] -= 1
        row = dsv41_sets.match(files)[0]
        self.assertFalse(row["complete"])
        self.assertEqual(row["wrong_size"], [name])


class Fit(unittest.TestCase):
    def test_download_sizes(self):
        self.assertAlmostEqual(dsv41_fit.ista_download_bytes("UD-IQ1_S") / 1e9, 240.96, places=1)
        self.assertAlmostEqual(dsv41_fit.ista_download_bytes("UD-IQ1_M") / 1e9, 258.38, places=1)
        self.assertAlmostEqual(dsv41_fit.ista_download_bytes("UD-IQ1_M", have_engram=True) / 1e9, 147.8, places=1)

    def test_fits_four_big_cards(self):
        cards = [{"name": "R9700", "vram_gb": 32.0}] * 4
        for model in SETS:
            self.assertEqual(dsv41_fit.plan(cards, 251.0, None, 8192, model=model)["verdict"], "fits")


class PackCall(unittest.TestCase):
    def test_pack_uses_the_sets_first_shard_as_metadata_and_the_shared_engram(self):
        md = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["UD-IQ1_S"]
        folder = Path("/m/UD-IQ1_S")
        shards = [folder / n for n in md["shards"]]
        cmd = setup.dsv41_pack_cmd(folder, [s.name for s in shards], Path("/m/eng.gguf"), shards[0], Path("/m/dspark.gguf"), Path("/p"))
        self.assertEqual(cmd[cmd.index("--engram") + 1], str(Path("/m/eng.gguf")))
        self.assertEqual(cmd[cmd.index("--engram-meta") + 1], str(shards[0]))
        self.assertEqual(len(cmd[cmd.index("--shards") + 1:cmd.index("--engram")]), 4)


if __name__ == "__main__":
    unittest.main()
