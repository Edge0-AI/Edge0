"""`strata launch` for DeepSeek-V4.1-Flash: the model is in the list, `--plan` says where it goes on this computer's cards (the real
numbers of strata_control/dsv41_fit.py), `--model-path` takes the nine shards where they are, and the settings that cannot apply
(parallel requests, pictures, tuning) are said instead of passed to an engine that would refuse them.  Nothing is downloaded or started."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd"}
HW_4X32 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd",
           "gpus": [{**R9700, "index": i} for i in range(4)]}


class LaunchDeepSeek(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(HW_4X32)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err + out)
        return json.loads(out)["plan"]

    def test_the_model_is_in_the_list(self):
        rows = [r for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        self.assertEqual([r.model for r in rows], ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-IQ1_S", "UD-IQ1_M", "UD-Q8_K_XL"])
        self.assertEqual([r.download_gb for r in rows[:3]], [256.8, 273.8, 239.3])
        self.assertEqual(catalog(self.hw).resolve("ist-das-2.25bpw"), ("deepseek41", "IST-DAS-2.25bpw"))
        r = rows[5]
        self.assertEqual((r.id, r.download_gb, r.verdict, r.images), ("deepseek-v4.1-flash-ud-q8_k_xl", 508.4, "fits", True))   # 0.2.7: pictures through the engine's own tower
        self.assertEqual(catalog(self.hw).resolve("ud-q8_k_xl"), ("deepseek41", "UD-Q8_K_XL"))
        self.assertEqual(catalog(self.hw).max_context("deepseek41", "UD-Q8_K_XL"), 8192)

    def test_the_plan_places_the_model_on_the_four_cards(self):
        p = self.plan("--model", "ud-q8_k_xl", "--no-tui")
        pl = p["placement"]
        self.assertEqual(p["settings"]["family"], "deepseek41")
        self.assertEqual(p["settings"]["context"], 8192)
        self.assertEqual(p["settings"]["tune"], "skip")
        self.assertEqual(p["settings"]["vision"], "on")              # pictures are on by default (the engine's own tower)
        self.assertEqual(pl["verdict"], "fits")
        self.assertEqual([c["layers"] for c in pl["cards"]], [[0, 9], [10, 19], [20, 29], [30, 39]])
        self.assertEqual(pl["args"], ["--tp", "2"])
        # measured on this very computer: 6,197 experts (116.5 GB) in graphics memory, 9,163 (172.3 GB) on the CPU
        self.assertAlmostEqual(pl["experts_in_vram"], 6197, delta=100)
        self.assertAlmostEqual(pl["cpu_experts_gb"], 172.3, delta=2.5)
        text = " ".join(pl["lines"])
        self.assertIn("n-gram table", text)
        self.assertIn("stays on the SSD", text)
        self.assertIn("508 GB", text)

    def test_the_human_plan_shows_the_placement(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("placement", out)
        self.assertIn("4 cards, one block of layers on each", out)
        self.assertIn("n-gram table", out)
        self.assertIn("Nothing was changed.", out)

    def test_named_cards_are_the_ones_planned(self):
        p = self.plan("--model", "ud-q8_k_xl", "--gpu", "0,1")
        self.assertEqual([c["index"] for c in p["placement"]["cards"]], [0, 1])
        self.assertLess(p["placement"]["experts_in_vram"], 3000)

    def test_a_small_computer_gets_the_warnings_not_a_refusal(self):
        self.hw = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 100.0, "engine_vendor": "nvidia"}
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("read from the SSD", out)
        self.assertIn("free disk space", out)
        self.assertIn("It is still allowed", out)
        row = next(r for r in catalog(self.hw).rows() if r.family == "deepseek41" and r.model == "UD-Q8_K_XL")
        self.assertEqual(row.verdict, "no")                         # "too big (can still try)": the plan above is printed and nothing is refused

    def test_the_plan_names_dspark_not_mtp(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("drafting  DSpark", out)
        self.assertNotIn("MTP", out)

    def test_the_plan_does_not_promise_a_strata_engine_download_for_deepseek(self):
        p = self.plan("--model", "ud-q8_k_xl", "--no-tui")
        text = json.dumps(p.get("download") or {})
        self.assertNotIn("engine for your graphics card", text)
        self.assertFalse((p.get("download") or {}).get("engine_missing"))

    def test_no_card_is_said(self):
        self.hw = {**HW_12, "gpus": []}
        code, out, err = run_cli(["launch", "--plan", "--model", "ud-q8_k_xl", "--no-tui"])
        self.assertEqual(code, 0, err)
        self.assertIn("No graphics card", out)

    def test_parallel_pictures_and_tuning_are_not_passed_on(self):
        p = self.plan("--model", "ud-q8_k_xl", "--parallel", "4", "--vision", "on", "--tune", "quick")
        s = p["settings"]
        self.assertFalse(s.get("parallel"))
        self.assertEqual((s["vision"], s["tune"]), ("on", "quick"))           # the DeepSeek tune is real now: the choice stays; pictures stay on
        text = " ".join(p["warnings"])
        self.assertIn("one request at a time", text)
        self.assertNotIn("cannot read pictures", text)
        off = self.plan("--model", "ud-q8_k_xl", "--vision", "off")["settings"]
        self.assertEqual(off["vision"], "off")                                 # --vision off still turns them off
        self.assertNotIn("Tuning is not offered", text)

    def test_setup_is_asked_for_the_family_and_the_files(self):
        s = S.LaunchSettings(family="deepseek41", model="UD-Q8_K_XL", context=8192, vision="off", tune="skip",
                             model_path="/data/ds")
        args = Runner(s, catalog=catalog(self.hw)).setup_args()
        self.assertEqual(args[:6], ["--yes", "--no-start", "--family", "deepseek41", "--model", "UD-Q8_K_XL"])
        self.assertIn("--gguf-dir", args)
        self.assertNotIn("--parallel", args)
        self.assertIsNone(Runner(s, catalog=catalog(self.hw)).catalog_parallel())

    def test_the_feed_reaches_setup_so_the_deepseek_engine_comes_from_it(self):
        """`strata launch --feed DIR` must hand the feed to setup: the DeepSeek engine package is fetched by setup, not by `strata update`
        (found on strix, 2026-10-25: the launch read the local feed for its own checks and setup went to GitHub for the engine)."""
        s = S.LaunchSettings(family="deepseek41", model="IST-DAS-2.00bpw", context=8192, vision="off", tune="skip")
        args = Runner(s, catalog=catalog(self.hw), feed="/data/feed").setup_args()
        self.assertIn("--feed", args)
        self.assertEqual(args[args.index("--feed") + 1], "/data/feed")
        self.assertNotIn("--feed", Runner(s, catalog=catalog(self.hw)).setup_args())


class OwnShards(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(HW_4X32)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def folder(self, shards=range(1, 10), nested=False):
        base = Path(os.environ["STRATA_INSTALL_DIR"]) / "ds"
        folder = base / "UD-Q8_K_XL" if nested else base
        folder.mkdir(parents=True, exist_ok=True)
        for i in shards:
            (folder / f"DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-{i:05d}-of-00009.gguf").write_bytes(b"GGUF")
        return base, folder

    def runner(self, path):
        return Runner(S.LaunchSettings(model_path=str(path)), catalog=catalog(self.hw))

    def test_a_folder_of_the_nine_shards_is_the_model(self):
        _base, folder = self.folder()
        r = self.runner(folder)
        r.check_own_files()
        self.assertEqual((r.s.family, r.s.model, r.s.model_path), ("deepseek41", "UD-Q8_K_XL", str(folder)))

    def test_the_repository_folder_with_the_size_inside(self):
        base, folder = self.folder(nested=True)
        r = self.runner(base)
        r.check_own_files()
        self.assertEqual((r.s.family, r.s.model, r.s.model_path), ("deepseek41", "UD-Q8_K_XL", str(folder)))

    def test_one_shard_names_its_folder(self):
        _base, folder = self.folder()
        r = self.runner(folder / "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00003-of-00009.gguf")
        r.check_own_files()
        self.assertEqual(r.s.model_path, str(folder))

    def test_a_missing_shard_is_said_before_anything_is_downloaded(self):
        _base, folder = self.folder(shards=[1, 2, 3, 5, 6, 7, 8, 9])
        with self.assertRaises(LaunchError) as cm:
            self.runner(folder).check_own_files()
        self.assertIn("nine parts", cm.exception.text)
        self.assertIn("00004-of-00009", cm.exception.text)

    def test_the_plan_with_own_files_downloads_nothing(self):
        _base, folder = self.folder()
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(folder), "--no-tui"])
        self.assertEqual(code, 0, err + out)
        self.assertIn("DeepSeek-V4.1-Flash", out)
        self.assertIn("Nothing to download", out)


if __name__ == "__main__":
    unittest.main()
