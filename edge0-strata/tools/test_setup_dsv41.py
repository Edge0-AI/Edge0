"""Tests for DeepSeek-V4.1-Flash in setup.py (MULTI_FILE_FAMILIES: nine pinned shards, its own engine strata-dsv41, several cards) and
for strata_control/dsv41_fit.py (where it fits, from the real numbers).  Mocked engine, downloads and processes - nothing is built,
fetched or started.

    python -m unittest tools.test_setup_dsv41
"""
from __future__ import annotations

import argparse
import contextlib
import os
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import dsv41_fit  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "vendor": "amd"}
FOUR = [{**R9700, "index": i} for i in range(4)]


def args(**kw):
    base = dict(family="deepseek41", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None, context=None,
                rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto", vram_reserve_mib=None,
                host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None, gpus=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        self.assertEqual(setup.HF_REVISIONS[REPO], "8a6ec9cf118e785a362ea94ee123c61c70553be6")
        self.assertRegex(fam["hf"], SHA)
        self.assertTrue(fam["hf"].endswith(f"{REPO}/resolve/8a6ec9cf118e785a362ea94ee123c61c70553be6/UD-Q8_K_XL/"))
        self.assertIs(setup.REPO_PINS[REPO], setup.DSV41_FILES)
        shards = fam["models"]["UD-Q8_K_XL"]["shards"]
        self.assertEqual(len(shards), 9)
        self.assertEqual(sum(setup.DSV41_FILES[n][0] for n in shards), 508_376_763_197)       # the nine shards: the pinned total
        self.assertEqual(setup.DSV41_FILES[shards[7]][0], dsv41_fit.ENGRAM_BYTES)             # shard 8 is the n-gram table
        for n, (size, sha) in setup.DSV41_FILES.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
            self.assertGreater(size, 5e6)
        self.assertEqual(setup.DSV41_FILES["mmproj-BF16.gguf"][0], 972_208_288)

    def test_not_a_pack_or_single_file_family(self):
        self.assertNotIn("deepseek41", setup.FAMILIES)
        self.assertNotIn("deepseek41", setup.OWN_PATH_FAMILIES)
        self.assertNotIn("UD-Q8_K_XL", setup.MODELS)

    def test_config_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-deepseek-v4.1-flash-ud-q8_k_xl.json"
            p.write_text(json.dumps({"args": ["--pack", "x", "--max-context", "8192", "--tp", "2"], "port": 8080, "gpu": [0, 1, 2, 3]}))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"], ch["gpu"]), ("deepseek41", "UD-Q8_K_XL", 8192, [0, 1, 2, 3]))


class Fit(unittest.TestCase):
    def setUp(self):                                  # the AMD rules below are the Linux ones, whatever box the test runs on
        p = mock.patch.object(dsv41_fit, "_on_windows", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def test_the_numbers_of_the_model(self):
        self.assertAlmostEqual(dsv41_fit.WEIGHTS_GB, 305.6, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.EXPERTS_GB, 288.8, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.ENGRAM_GB, 202.8, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.DOWNLOAD_GB, 508.4, delta=0.1)
        self.assertEqual(dsv41_fit.EXPERTS, 15360)

    def test_four_amd_cards_run_the_ista_sets_with_a_plain_layer_split(self):
        # r9700, 2026-10-23: 8K prompt 370 tokens a second with --tp 1 against 196 with the engine's default (2)
        for model in ("IST-DAS-1.75bpw", "IST-DAS-2.00bpw", "IST-DAS-2.25bpw"):
            p = dsv41_fit.plan(FOUR, 251.0, 900.0, model=model)
            self.assertEqual((p["tp"], p["args"]), (1, ["--tp", "1"]), model)
        # two and three cards keep pairs; other makes keep pairs; one card needs no flag; the Unsloth set is not changed
        self.assertEqual(dsv41_fit.plan(FOUR[:2], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        self.assertEqual(dsv41_fit.plan(FOUR[:3], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        nv = [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "vendor": "nvidia", "index": i} for i in range(4)]
        self.assertEqual(dsv41_fit.plan(nv, 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        self.assertEqual(dsv41_fit.plan(FOUR[:1], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], [])
        self.assertEqual(dsv41_fit.plan(FOUR, 251.0, 900.0)["args"], ["--tp", "2"])

    def test_four_r9700_match_the_measured_run(self):
        # r9700, 2026-10-20: 1,558 + 1,559 + 1,575 + 1,505 = 6,197 experts in graphics memory, 9,163 (172.3 GB) on the CPU lane
        p = dsv41_fit.plan(FOUR, 251.0, 900.0)
        self.assertEqual(p["verdict"], "fits")
        self.assertAlmostEqual(p["experts_in_vram"], 6197, delta=100)
        self.assertAlmostEqual(p["cpu_experts_gb"], 172.3, delta=2.5)
        self.assertEqual([c["layers"] for c in p["cards"]], [[0, 9], [10, 19], [20, 29], [30, 39]])
        self.assertEqual(p["args"], ["--tp", "2"])
        self.assertTrue(p["disk_ok"])

    def test_one_card_and_little_ram_are_warnings(self):
        p = dsv41_fit.plan([{**R9700, "index": 0}], 64.0, 100.0)
        self.assertEqual(p["verdict"], "spills")
        self.assertEqual(p["args"], [])
        self.assertGreater(p["spill_gb"], 200)
        text = " ".join(p["warnings"])
        self.assertIn("read from the SSD", text)
        self.assertIn("free disk space", text)
        self.assertIn("still allowed", text)
        self.assertFalse(p["disk_ok"])

    def test_a_card_too_small_for_the_dense_weights_is_said(self):
        p = dsv41_fit.plan([{"index": 0, "name": "RTX 3060", "vram_gb": 12.0}], 128.0)
        self.assertTrue(any("dense weights" in w for w in p["warnings"]))

    def test_no_card(self):
        p = dsv41_fit.plan([], 128.0)
        self.assertEqual(p["verdict"], "no_card")
        self.assertTrue(any("No graphics card" in w for w in p["warnings"]))

    def test_cards_of_two_sizes_share_the_layers_in_proportion(self):
        cards = [{**R9700, "index": 0}, {"name": "RTX A4000", "vram_gb": 16.0, "index": 1}]
        p = dsv41_fit.plan(cards, 256.0)
        a, b = p["cards"]
        self.assertGreater(a["layers"][1] - a["layers"][0], b["layers"][1] - b["layers"][0])
        self.assertEqual(b["layers"][1], 39)
        self.assertGreater(a["experts"], b["experts"])

    def test_split_layers_covers_every_layer(self):
        for w in ([1, 1, 1, 1], [32, 16], [8, 8, 8], [32, 32, 16, 12, 12]):
            r = dsv41_fit.split_layers(w)
            self.assertEqual(r[0][0], 0)
            self.assertEqual(r[-1][1], 39)
            self.assertTrue(all(a[1] + 1 == b[0] for a, b in zip(r, r[1:])))

    def test_options_like_gpu_fit(self):
        o = dsv41_fit.options(FOUR, 251.0, "amd")
        self.assertEqual(o["recommended"], "all")
        self.assertEqual([x["key"] for x in o["options"]], ["all", "single"])
        self.assertEqual(o["options"][0]["cards"], [0, 1, 2, 3])
        self.assertTrue(o["options"][0]["recommended"])


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.github_up = True
        self.cfg = {}
        self.exe = self.root / "engine" / ("strata-dsv41.exe" if setup.WIN else "strata-dsv41")
        eng = self.root / "engine"

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            eng.mkdir(parents=True, exist_ok=True)
            return self.root / "llama", eng, {"version": "0.2.2"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None, explicit=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False, also=(): self.calls.append(("sha", s.name, own))),
                   mock.patch.object(setup, "free_gb", lambda p: 900.0),
                   mock.patch.object(setup, "github_reachable", lambda timeout=10.0: self.github_up),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "ROOT", self.root),
                   mock.patch.object(setup, "home", lambda: self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}).start()
        self.addCleanup(mock.patch.stopall)

    def install(self, ram=251.0, cards=None, engine=True, **kw):
        if engine:
            self.exe.parent.mkdir(parents=True, exist_ok=True)
            self.exe.write_text("x")
        cards = cards or FOUR
        kw.setdefault("model", "UD-Q8_K_XL")
        kw.setdefault("vision", "no")            # the older tests are about the files of the model; pictures-on has its own tests (PicturesByDefault)
        gpu = {"index": 0, "count": len(cards), "vram_gb": 32.0, "name": "R9700", "cards": cards}
        return run(setup.install_dsv41, args(models_dir=str(self.root / "models"), **kw), "deepseek41", gpu, True, [], 13, [], 8080, ram,
                   None, cards)

    def test_installs_the_nine_pinned_shards_and_writes_the_config(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        folder = self.root / "models" / "deepseek-v4.1-flash-UD-Q8_K_XL"
        self.assertEqual([c[2] for c in get], [folder / setup.DSV41_SHARD.format(i=i) for i in range(1, 10)])
        self.assertEqual(get[1][1], setup.hf(REPO) + "UD-Q8_K_XL/" + setup.DSV41_SHARD.format(i=2))
        self.assertEqual(sum(1 for c in self.calls if c[0] == "sha"), 9)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn("pack_gguf.py", " ".join(map(str, pack)))
        self.assertEqual(self.cfg["arch"], "deepseek41")
        self.assertEqual(self.cfg["exe"], str(self.exe))
        self.assertEqual(self.cfg["model_name"], "deepseek-v4.1-flash")
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "8192")
        self.assertEqual(a[a.index("--eos") + 1], "1")
        self.assertEqual(a[a.index("--tp") + 1], "2")
        self.assertTrue(Path(a[a.index("--pack") + 1]).match("*packs/deepseek-v4.1-flash-ud-q8_k_xl"))
        self.assertEqual(self.cfg["gpu"], [0, 1, 2, 3])
        self.assertEqual(self.cfg["engine_ready_s"], 3600)
        self.assertEqual(self.cfg["backend"], "hip")
        self.assertTrue(self.cfg["tokenizer"].endswith("tokenizer"))
        self.assertEqual(a[a.index("--vision") + 1], "off")     # (this test asks for no pictures: they are on by default, see PicturesByDefault)
        self.assertNotIn("--mtp", a)
        self.assertIn("4 cards, one block of layers on each", out)
        self.assertIn("n-gram table", out)

    def test_no_engine_stops_before_the_download(self):
        rc, out = self.install(engine=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("DeepSeek's engine is not available for this computer yet", out)
        self.assertIn("RTX 30, 40 and 50", out)
        self.assertNotIn("cmake", out)
        self.assertNotIn("STRATA_DSV41_EXE", out)
        self.assertIn("Nothing has been downloaded", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_own_files_are_checked_not_downloaded(self):
        own = self.root / "own"
        own.mkdir()
        for i in range(1, 10):
            (own / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])
        self.assertTrue(all(c[2] for c in self.calls if c[0] == "sha"))        # own=True: setup never deletes the person's file
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn(str(own), pack)

    def test_a_missing_own_shard_is_said(self):
        own = self.root / "own"
        own.mkdir()
        for i in (1, 2, 3):
            (own / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 1, out)
        self.assertIn("00004-of-00009.gguf is missing", out)

    def test_the_repository_layout_is_found(self):
        own = self.root / "repo"
        (own / "UD-Q8_K_XL").mkdir(parents=True)
        for i in range(1, 10):
            (own / "UD-Q8_K_XL" / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn(str(own / "UD-Q8_K_XL"), pack)

    def test_little_ram_is_asked_and_kept_with_an_explicit_model(self):
        rc, out = self.install(ram=64.0, cards=FOUR[:1])
        self.assertEqual(rc, 0, out)
        self.assertIn("read from the SSD", out)
        self.assertIn("installing UD-Q8_K_XL with 64 GB of RAM, as you chose", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--max-context") + 1], "8192")
        self.assertNotIn("--tp", self.cfg["args"])

    def test_little_ram_with_yes_alone_stops_where_the_rule_says(self):
        rc, out = self.install(ram=24.0, cards=FOUR[:1], model=None)        # the version was not chosen on the command line
        self.assertEqual(rc, 1, out)
        self.assertIn("more RAM or cards fix it", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_a_disk_that_cannot_hold_the_files_is_the_one_hard_stop(self):
        with mock.patch.object(setup, "free_gb", lambda p: 100.0):
            rc, out = self.install()
        self.assertEqual(rc, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn("an NVMe drive is best", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_pictures_and_flash_next_settings_are_said(self):
        rc, out = self.install(vision="yes", kv="int8", context=131072)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("cannot read pictures", out)                          # 0.2.7: the engine's own tower reads them
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vision") + 1], "on")
        self.assertIn("Flash-Next settings", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--max-context") + 1], "131072")
        self.assertIn("checked at 4K and 8K", out)


class Catalog(unittest.TestCase):
    def setUp(self):                                  # the AMD rules below are the Linux ones, whatever box the test runs on
        p = mock.patch.object(dsv41_fit, "_on_windows", lambda: False)
        p.start()
        self.addCleanup(p.stop)

    def test_the_mcp_catalog_lists_the_size(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table({"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd"})
                   if f["family"] == "deepseek41")
        self.assertEqual([x["model"] for x in fam["sizes"]], ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-IQ1_S", "UD-IQ1_M", "UD-Q8_K_XL"])
        small_sets = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual((small_sets["IST-DAS-2.00bpw"]["download_gb"], small_sets["IST-DAS-2.25bpw"]["download_gb"]), (256.8, 273.8))
        self.assertTrue(small_sets["IST-DAS-2.00bpw"]["on_this_pc"].startswith("fits"))
        size = small_sets["UD-Q8_K_XL"]
        self.assertEqual((size["model"], size["id"], size["download_gb"], size["images"]), ("UD-Q8_K_XL", "deepseek-v4.1-flash-ud-q8_k_xl", 508.4, True))
        self.assertTrue(size["on_this_pc"].startswith("fits"))
        self.assertTrue(size["mem_total"] and size["multi_file"])
        self.assertEqual(size["gpu_options"]["recommended"], "all")
        small = next(f for f in s.model_table({"ram_gb": 32.0, "gpus": []}) if f["family"] == "deepseek41")
        self.assertTrue(all(x["on_this_pc"].startswith("does not fit") for x in small["sizes"]))      # no card

    def test_each_set_states_the_free_disk_it_needs_from_the_set_itself(self):
        """The app's DeepSeek card says "Needs about N GB of free disk space": N is the set's whole download plus the pack and room to work, per set."""
        import strata_mcp as M
        s = M.Strata(ROOT)
        table = s.model_table({"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd"})
        fam = next(f for f in table if f["family"] == "deepseek41")
        for size in fam["sizes"]:
            self.assertEqual(size["disk_gb"], round(size["download_gb"] + dsv41_fit.DISK_HEADROOM_GB, 1), size["model"])
        by = {x["model"]: x["disk_gb"] for x in fam["sizes"]}
        self.assertNotEqual(by["IST-DAS-2.00bpw"], by["IST-DAS-2.25bpw"], "2.00 and 2.25 do not need the same disk")
        self.assertGreater(by["IST-DAS-2.25bpw"], by["IST-DAS-2.00bpw"])
        self.assertGreater(by["IST-DAS-2.00bpw"], by["IST-DAS-1.75bpw"])
        for f in table:
            if f["family"] != "deepseek41":
                self.assertTrue(all(x["disk_gb"] is None for x in f["sizes"]), f["family"])

    def test_the_install_plan(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        s.hardware = lambda fresh=False: {"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd",
                                          "cpu": {"avx2": True}}
        s.disk_free = lambda p: 2000.0
        plan, a = M.Tools(s).install_plan("deepseek41", "UD-Q8_K_XL", None, None, None, None, None, None, None, None, None)
        self.assertEqual((plan["title"], plan["download_gb"], plan["context"], plan["images"]), ("DeepSeek-V4.1-Flash UD-Q8_K_XL", 508.4, 8192, "yes"))   # 0.2.7: pictures on by default
        self.assertEqual(a[:6], ["--yes", "--no-start", "--family", "deepseek41", "--model", "UD-Q8_K_XL"])
        self.assertTrue(plan["ram"].startswith("fits"))
        for flag in ("--kv", "--low-ram", "--experimental-speed-projection"):
            self.assertNotIn(flag, a)
        files = M.expected_files(s.setup_module(), "deepseek41", "UD-Q8_K_XL", Path("/m"), False)
        self.assertEqual(len(files), 9)
        self.assertEqual(sum(f["bytes"] for f in files), 508_376_763_197)


ISTA = "IST-DAS-2.00bpw"
RTX5070 = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 12.0, "vendor": "nvidia"}


class NoNetwork(Install):
    """GitHub cannot be reached (a connection that is never answered): one plain line at once, not minutes of retries; and the Strata engine is not used."""

    def test_unreachable_github_is_one_plain_line_before_anything_is_downloaded(self):
        self.github_up = False
        rc, out = self.install(engine=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("Strata can't reach GitHub to download the DeepSeek engine.", out)
        self.assertIn("Check the internet connection and run the same command again.", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_a_local_feed_needs_no_github(self):
        """A run on a local feed (a folder, a loopback address) downloads nothing from GitHub: a slow or down GitHub must not stop it."""
        self.github_up = False
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_FEED": str(self.root / "feed")}):
            rc, out = self.install(engine=False)
        self.assertNotIn("can't reach GitHub", out)

    def test_a_package_the_person_has_needs_no_github(self):
        self.github_up = False
        self.install(engine=False)                                    # (the first run only creates the folders)
        pk = self.root / "engine-dsv41"
        pk.mkdir(exist_ok=True)
        (pk / "strata-dsv41-0.1.49-x.tar.gz").write_bytes(b"x")
        with mock.patch.object(setup.dsv41_engine_module if hasattr(setup, "dsv41_engine_module") else __import__("strata_control.dsv41_engine", fromlist=["x"]),
                               "install_package", lambda *a, **k: None):
            rc, out = self.install(engine=False)
        self.assertNotIn("can't reach GitHub", out)

    def test_the_strata_engine_is_not_part_of_a_deepseek_install(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in self.calls if c[0] == "engine"])        # engine_step is not called any more
        self.assertIn("DeepSeek engine", out)


class GithubCheck(unittest.TestCase):
    def test_github_reachable_answers_quickly_in_both_directions(self):
        import urllib.error
        with mock.patch.object(setup, "_urlopen", side_effect=urllib.error.URLError("timed out")):
            self.assertFalse(setup.github_reachable(1))
        with mock.patch.object(setup, "_urlopen", side_effect=OSError("unreachable")):
            self.assertFalse(setup.github_reachable(1))
        with mock.patch.object(setup, "_urlopen", side_effect=urllib.error.HTTPError("https://github.com", 403, "no", {}, None)):
            self.assertTrue(setup.github_reachable(1))                       # any answer means it is reachable
        with mock.patch.object(setup, "_urlopen", return_value=mock.MagicMock()):
            self.assertTrue(setup.github_reachable(1))

    def test_a_local_feed_is_told_from_the_signed_one(self):
        for feed in ("C:/feeds/f", "/tmp/feed", "file:///tmp/feed", "http://127.0.0.1:8099/feed.json", "https://localhost/f"):
            self.assertTrue(setup.local_feed(feed), feed)
        for feed in (None, "", "https://github.com/Niko1221/Strata/releases/latest/download/feed.json"):
            with mock.patch.dict("os.environ", {"STRATA_UPDATE_FEED": ""}):
                self.assertFalse(setup.local_feed(feed), feed)
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_FEED": "/tmp/x"}):
            self.assertTrue(setup.local_feed())                              # the environment names it when --feed does not


class OfficialSets(unittest.TestCase):
    """The two official IST-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF): the catalog, the pins, the pack recipe, the fit words."""

    def test_the_pinned_files(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        self.assertEqual(list(fam["models"]), ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-IQ1_S", "UD-IQ1_M", "UD-Q8_K_XL"])         # the official sets first
        repo = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
        self.assertEqual(setup.HF_REVISIONS[repo], "cb02422c900df489cea5833e8a92171224059b3c")
        self.assertIs(setup.REPO_PINS[repo], setup.DSV41_ISTA_FILES)
        self.assertNotIn("helcig/tmp-oct26-2", setup.HF_REVISIONS)           # the first release's repository is only remembered as local names
        self.assertEqual(setup.DSV41_ISTA_REPO, repo)
        self.assertEqual(len(setup.DSV41_ISTA_FILES), 13)                      # 3 x 4 shards + the Engram
        for n, (size, sha) in setup.DSV41_ISTA_FILES.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
        eng = setup.DSV41_ISTA_FILES["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"]
        self.assertEqual(eng, (110_595_290_816, "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"))
        self.assertEqual(setup.DSV41_ALL_FILES["dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"],
                         (7_976_914_240, "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"))
        sums = {m: sum(setup.DSV41_ALL_FILES[n][0] for n in fam["models"][m]["shards"]) for m in list(fam["models"])[:3]}
        self.assertEqual(sums, {"IST-DAS-2.00bpw": 138238893056, "IST-DAS-2.25bpw": 155225824256, "IST-DAS-1.75bpw": 120721120256})
        self.assertEqual(sums["IST-DAS-1.75bpw"], dsv41_fit.ISTA_SETS["IST-DAS-1.75bpw"]["set_bytes"])
        self.assertEqual(sums["IST-DAS-2.00bpw"], dsv41_fit.ISTA_SETS["IST-DAS-2.00bpw"]["set_bytes"])
        self.assertEqual(sums["IST-DAS-2.25bpw"], dsv41_fit.ISTA_SETS["IST-DAS-2.25bpw"]["set_bytes"])
        self.assertEqual(setup.DSV41_ISTA_FILES["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"][0], dsv41_fit.ISTA_ENGRAM_BYTES)
        for m in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw"):                        # both sets name the same shared files: the Engram is downloaded once
            self.assertEqual(fam["models"][m]["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK])
            for n in fam["models"][m]["shared"]:
                self.assertIn(n, setup.DSV41_SHARED)
                self.assertIn(n, setup.DSV41_ALL_FILES)
        self.assertRegex(setup.DSV41_SHARED[setup.DSV41_ISTA_ENGRAM], r"ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/cb02422c[0-9a-f]{32}/$")
        self.assertRegex(fam["models"]["IST-DAS-2.25bpw"]["hf"], r"ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/cb02422c[0-9a-f]{32}/$")

    def test_shipped_profiles_match_their_hashes(self):
        import hashlib
        for n in ("profile-2.00.bin", "profile-2.25.bin"):
            data = (setup.DSV41_PROFILES / n).read_bytes()
            self.assertEqual(len(data), 66048)                                 # uint32 [43][384]
            want = (setup.DSV41_PROFILES / (n + ".sha256")).read_text().split()[0]
            self.assertEqual(hashlib.sha256(data).hexdigest(), want)

    def test_the_pack_recipe_is_the_plain_one(self):
        folder, pack = Path("/m/set"), Path("/d/pack")
        names = [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)]
        cmd = setup.dsv41_pack_cmd(folder, names, Path("/s/engram-IQ4_NL.gguf"), Path("/s/meta.gguf"), Path("/s/dspark.gguf"), pack)
        self.assertTrue(cmd[1].endswith("pack_gguf.py"))
        self.assertEqual(cmd[2:], ["--model", str(folder), "--shards", *names, "--engram", str(Path("/s/engram-IQ4_NL.gguf")),
                                   "--engram-meta", str(Path("/s/meta.gguf")), "--dspark-gguf", str(Path("/s/dspark.gguf")),
                                   "--dspark-noise-token", "128799", "--dspark-markov-rank", "256", "--out", str(pack)])
        for bad in ("--skip-tensors", "--dense-override", "--override-experts", "--tq-experts"):      # these give garbage text on the official sets
            self.assertNotIn(bad, cmd)

    def test_fit_words_use_the_measured_numbers(self):
        for ram, verdict in ((64.0, "tight"), (46.0, "tight"), (24.0, "spills"), (256.0, "fits")):
            p = dsv41_fit.plan([RTX5070], ram, None, model=ISTA)
            self.assertEqual(p["verdict"], verdict, ram)
            self.assertAlmostEqual(p["download_gb"], 256.8, delta=0.1)
            self.assertEqual(p["args"], [])
        p = dsv41_fit.plan([RTX5070], 64.0, 100.0, model=ISTA)
        text = " ".join(p["lines"]) + " ".join(p["warnings"])
        for needle in ("4.7", "7.2", "31 tokens a second", "free disk space", "only once" if False else "downloads it once"):
            self.assertIn(needle, text)
        self.assertFalse(p["disk_ok"])
        p225 = dsv41_fit.plan([RTX5070], 64.0, None, model="IST-DAS-2.25bpw")
        self.assertAlmostEqual(p225["download_gb"], 273.8, delta=0.1)
        self.assertEqual(dsv41_fit.plan([], 64.0, None, model=ISTA)["verdict"], "no_card")
        self.assertEqual(dsv41_fit.plan([RTX5070], 64.0, None, model="UD-Q8_K_XL")["verdict"], "spills")        # the big set is unchanged


class EnginePackage(unittest.TestCase):
    """DeepSeek's engine is its own package beside the engine; the Qwen engine folder is never written."""

    def test_the_package_folder_is_beside_the_engine(self):
        self.assertEqual(setup.dsv41_engine_dir(Path("/d/engine")), Path("/d") / "engine-dsv41")

    def test_the_own_package_is_found_before_the_old_place(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}):
            eng = Path(d) / "engine"
            eng.mkdir()
            name = "strata-dsv41.exe" if setup.WIN else "strata-dsv41"
            self.assertIsNone(setup.dsv41_exe(eng))
            (eng / name).write_text("old")
            self.assertEqual(setup.dsv41_exe(eng), eng / name)                 # an all-in-one engine still works
            (Path(d) / "engine-dsv41" / "0.1.49-x").mkdir(parents=True)
            (Path(d) / "engine-dsv41" / "0.1.49-x" / name).write_text("new")
            self.assertEqual(setup.dsv41_exe(eng), Path(d) / "engine-dsv41" / "0.1.49-x" / name)


class SharedFiles(unittest.TestCase):
    def test_the_drafter_beside_the_sets_folder_is_found_from_the_set_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "off" / "2.00bpw").mkdir(parents=True)
            (root / "dspark").mkdir()
            f = root / "dspark" / setup.DSV41_DSPARK
            f.write_bytes(b"x")
            a = argparse.Namespace(gguf_dir=str(root / "off" / "2.00bpw"))
            with mock.patch.dict(setup.DSV41_ALL_FILES, {setup.DSV41_DSPARK: (1, "0" * 64)}):
                got = setup.dsv41_find_shared(a, setup.DSV41_DSPARK, root / "off" / "2.00bpw", root / "models" / "shared")
            self.assertEqual(got, f)


class LegacyFiles(unittest.TestCase):
    """Files of the first release (model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf, often in a 2.00bpw/ folder) are the same weights: accepted."""

    def test_the_old_pins_are_the_old_files_and_the_unchanged_shards_match_today_s(self):
        for i in (2, 3, 4):                                                   # same bytes: same size and SHA-256 under both names
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=i)
            self.assertEqual(setup.DSV41_ALL_FILES[new], setup.DSV41_LEGACY_FILES["model-2.00bpw-%05d-of-00004.gguf" % i])
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XS", i=i)
            self.assertEqual(setup.DSV41_ALL_FILES[new], setup.DSV41_LEGACY_FILES["model-2.25bpw-%05d-of-00004.gguf" % i])
        self.assertNotEqual(setup.DSV41_ALL_FILES[setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=1)],
                            setup.DSV41_LEGACY_FILES["model-2.00bpw-00001-of-00004.gguf"])        # shard 1 and the Engram differ in their header only
        self.assertNotIn("model-2.00bpw-00001-of-00004.gguf", setup.DSV41_ISTA_FILES)                 # (never downloaded under the old names)
        self.assertNotIn("model-2.00bpw-00001-of-00004.gguf", setup.REPO_PINS["ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"])

    def test_pick_prefers_today_s_name_then_the_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=2)
            self.assertEqual(setup.dsv41_pick(folder, new), folder / new)                  # nothing there: where it will be downloaded
            (folder / "model-2.00bpw-00002-of-00004.gguf").write_bytes(b"x")
            self.assertEqual(setup.dsv41_pick(folder, new).name, "model-2.00bpw-00002-of-00004.gguf")
            (folder / new).write_bytes(b"y")
            self.assertEqual(setup.dsv41_pick(folder, new).name, new)

    def test_the_folder_of_the_first_release_is_found_from_the_parent(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "2.00bpw"
            sub.mkdir()
            (sub / "model-2.00bpw-00001-of-00004.gguf").write_bytes(b"x")
            fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
            a = argparse.Namespace(gguf_dir=d, models_dir=d)
            self.assertEqual(setup.dsv41_folder(a, fam, "t", "IST-DAS-2.00bpw"), sub)
            a2 = argparse.Namespace(gguf_dir=str(sub), models_dir=d)
            self.assertEqual(setup.dsv41_folder(a2, fam, "t", "IST-DAS-2.00bpw"), sub)

    def test_the_old_engram_name_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "2.00bpw"
            sub.mkdir()
            old = sub / "engram-IQ4_NL.gguf"
            old.write_bytes(b"e")
            a = argparse.Namespace(gguf_dir=d)
            with mock.patch.dict(setup.DSV41_ALL_FILES, {"engram-IQ4_NL.gguf": (1, "0" * 64)}):
                got = setup.dsv41_find_shared(a, setup.DSV41_ISTA_ENGRAM, sub, Path(d) / "shared")
            self.assertEqual(got, old)


class SixShardLayout(unittest.TestCase):
    """ISTA-DASLab's six-shard layout (2026-10-27, llama.cpp b11870+) beside the pinned four-shard one: both are accepted from files already here,
    setup still downloads the pinned four-shard files (the owner moves the default later)."""

    def models(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]
        return {m: fam[m] for m in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw")}

    def test_the_download_pin_and_the_four_shard_tables_did_not_move(self):
        repo = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
        self.assertEqual(setup.HF_REVISIONS[repo], "cb02422c900df489cea5833e8a92171224059b3c")
        self.assertEqual(len(setup.DSV41_ISTA_FILES), 13)
        self.assertEqual(setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM][1], "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a")
        for m, d in self.models().items():
            self.assertEqual(len(d["shards"]), 4, m)
            self.assertTrue(all(n.endswith("-of-00004.gguf") for n in d["shards"]), m)

    def test_the_six_shard_tables_are_five_files_a_set_with_the_hub_sizes_and_hashes(self):
        self.assertEqual(len(setup.DSV41_ISTA6_FILES), 15)
        for n, (size, sha) in setup.DSV41_ISTA6_FILES.items():
            self.assertTrue(n.endswith("-of-00006.gguf"), n)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
            self.assertIn(n, setup.DSV41_ALL_FILES)
            self.assertNotIn(n, setup.DSV41_ISTA_FILES)                         # never downloaded: not in the repository pin
            self.assertNotIn(n, setup.REPO_PINS["ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"])
        sums = {m: sum(setup.DSV41_ISTA6_FILES[n][0] for n in d["shards6"]) for m, d in self.models().items()}
        self.assertEqual(sums, {"IST-DAS-2.00bpw": 138238967104, "IST-DAS-2.25bpw": 155225898368, "IST-DAS-1.75bpw": 120721194336})
        for m, d in self.models().items():
            self.assertEqual(len(d["shards6"]), setup.DSV41_ISTA6_SHARDS)
            self.assertTrue(d["shards6"][0].endswith("-00001-of-00006.gguf"))
            self.assertEqual(setup.DSV41_ISTA6_FILES[d["shards6"][0]][0], 5780704)       # the metadata shard: no tensors
        self.assertEqual(setup.DSV41_ISTA6_FILES["DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00005-of-00006.gguf"],
                         (2969539840, "7c8dbee99890f05dbc420eba6f44393f5228d06248fe61e1ae8ed002449c7549"))

    def test_the_converted_engram_has_the_pinned_ones_name_and_size_and_another_hash_that_is_accepted_beside_it(self):
        size, sha = setup.DSV41_ENGRAM_CONVERTED
        pinned = setup.DSV41_ISTA_FILES[setup.DSV41_ISTA_ENGRAM]
        self.assertEqual(size, pinned[0])
        self.assertNotEqual(sha, pinned[1])
        self.assertEqual(setup.DSV41_SHA_ALSO, {setup.DSV41_ISTA_ENGRAM: (sha,)})

    def test_verify_accepts_the_other_engram_hash_and_marks_it(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "e.gguf"
            f.write_bytes(b"x" * 8)
            real = hashlib.sha256(b"x" * 8).hexdigest()
            rc, out = run(setup.verify_sha256, f, 8, "0" * 64, own=True, also=(real,))
            self.assertNotEqual(rc, 1, out)
            self.assertIn("converted version", out)
            self.assertEqual(f.with_name("e.gguf.done").read_text(), "sha256 " + real)
            rc, out = run(setup.verify_sha256, f, 8, "0" * 64, own=True, also=(real,))             # the mark is enough the second time
            self.assertNotEqual(rc, 1, out)
            self.assertNotIn("checking", out)
            f2 = Path(d) / "f.gguf"
            f2.write_bytes(b"y" * 8)
            with mock.patch.object(setup, "own_gguf_is_whole", return_value=False):
                rc, out = run(setup.verify_sha256, f2, 8, "0" * 64, own=True, also=(real,))        # another file is still checked as before
                self.assertEqual(rc, 1, out)

    def test_layout_of_the_files_in_a_folder(self):
        d = self.models()["IST-DAS-2.00bpw"]
        four, six, legacy = list(d["shards"]), list(d["shards6"]), [setup.DSV41_LEGACY_NAMES[n] for n in d["shards"]]
        of = lambda have: setup.dsv41_layout_of(d, set(have).__contains__)
        self.assertEqual(of([]), "four")                                          # nothing there: the pinned files are what setup downloads
        self.assertEqual(of(four), "four")
        self.assertEqual(of(legacy), "four")                                      # the first release's names are four-shard files
        self.assertEqual(of(six), "six")
        self.assertEqual(of(six[:2]), "six")                                      # an incomplete six-shard set is still told as six: the missing ones are named
        self.assertEqual(of(four + six), "four")                                  # both whole: the pinned set, as before
        self.assertEqual(of(four[:2] + six), "six")                               # a stray four-shard file beside the whole six-shard set
        self.assertEqual(of(four[:2] + six[:2]), "four")                          # neither whole

    def test_parts_name_what_is_missing_in_the_layout_that_is_there(self):
        d = self.models()["IST-DAS-2.25bpw"]
        have = set(d["shards6"]) - {d["shards6"][2]}
        layout, names, missing = setup.dsv41_parts(d, have.__contains__)
        self.assertEqual((layout, names, missing), ("six", list(d["shards6"]), [d["shards6"][2]]))
        have = set(d["shards"][:3])
        layout, names, missing = setup.dsv41_parts(d, have.__contains__)
        self.assertEqual((layout, len(names), missing), ("four", 4, [d["shards"][3]]))

    def test_the_pack_says_which_layout_it_was_made_from(self):
        with tempfile.TemporaryDirectory() as t:
            pack = Path(t)
            self.assertIsNone(setup.dsv41_pack_layout(pack))
            (pack / "tensors.txt").write_text("file 0 x-00001-of-00006.gguf\nfile 1 y.gguf\nt a BF16 0 0 1 1\n")
            self.assertEqual(setup.dsv41_pack_layout(pack), "six")
            (pack / "tensors.txt").write_text("file 0 x-00001-of-00004.gguf\nt a BF16 0 0 1 1\n")
            self.assertEqual(setup.dsv41_pack_layout(pack), "four")
            (pack / "tensors.txt").write_text("file 0 model.gguf\nt a BF16 0 0 1 1\n")
            self.assertIsNone(setup.dsv41_pack_layout(pack))

    def test_the_folder_with_six_shard_names_is_found_from_the_parent(self):
        with tempfile.TemporaryDirectory() as t:
            sub = Path(t) / "2.00bpw"
            sub.mkdir()
            (sub / self.models()["IST-DAS-2.00bpw"]["shards6"][0]).write_bytes(b"x")
            fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
            self.assertEqual(setup.dsv41_folder(argparse.Namespace(gguf_dir=t, models_dir=t), fam, "t", "IST-DAS-2.00bpw"), sub)


class Smallest(unittest.TestCase):
    """IST-DAS 1.75 bpw (IQ1_M): the smallest official set, recommended where 2.00 is tight, never forced."""

    def test_the_files_the_pins_and_the_recipe(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        d = fam["models"]["IST-DAS-1.75bpw"]
        self.assertEqual(d["recipe"], "ista")
        self.assertEqual(d["shards"], [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertEqual(d["shared"], fam["models"]["IST-DAS-2.00bpw"]["shared"])              # the same Engram, drafter and metadata shard
        self.assertEqual(setup.DSV41_ISTA_FILES[d["shards"][0]],
                         (39358830464, "6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d"))
        self.assertEqual(setup.DSV41_ISTA_FILES[d["shards"][3]][0], 2188841824)
        self.assertEqual(sum(setup.DSV41_ISTA_FILES[n][0] for n in d["shards"]), 120_721_120_256)
        self.assertTrue(all(n not in setup.DSV41_LEGACY_NAMES.values() for n in d["shards"]))     # no first-release names for it
        for n, (size, sha) in setup.DSV41_NEW_SHARD1.items():                                     # shard 1 after ISTA's 2026-10-26 template update
            self.assertIn(n, setup.DSV41_ISTA_FILES)
            self.assertNotEqual(setup.DSV41_ISTA_FILES[n][0], size)
            self.assertNotEqual(setup.DSV41_ISTA_FILES[n][1], sha)
        cmd = setup.dsv41_pack_cmd(Path("/m"), d["shards"], Path("/s/e.gguf"), Path("/s/m.gguf"), Path("/s/d.gguf"), Path("/p"))
        self.assertIn("--dspark-noise-token", cmd)
        self.assertNotIn("--skip-tensors", cmd)

    def test_it_is_the_smallest_and_warns_never_refuses(self):
        sizes = {m: dsv41_fit.ista_download_bytes(m) / 1e9 for m in dsv41_fit.ISTA_SETS}
        self.assertEqual(min(sizes, key=sizes.get), "IST-DAS-1.75bpw")
        self.assertAlmostEqual(sizes["IST-DAS-1.75bpw"], 239.3, delta=0.1)
        for ram in (24.0, 36.0, 64.0):
            p = dsv41_fit.plan([RTX5070], ram, None, model="IST-DAS-1.75bpw")
            self.assertIn(p["verdict"], ("fits", "tight", "spills"))
            self.assertTrue(p["lines"])
        self.assertEqual(dsv41_fit.plan([RTX5070], 36.0, None, model="IST-DAS-1.75bpw")["verdict"], "tight")      # runs where 2.00 spills
        self.assertEqual(dsv41_fit.plan([RTX5070], 36.0, None, model="IST-DAS-2.00bpw")["verdict"], "spills")

    def test_it_is_recommended_where_2_00_is_tight(self):
        self.assertEqual(dsv41_fit.recommend(64.0, 900.0), "IST-DAS-2.00bpw")
        self.assertEqual(dsv41_fit.recommend(None, None), "IST-DAS-2.00bpw")                   # unknown: the usual one
        self.assertEqual(dsv41_fit.recommend(32.0, 900.0), "IST-DAS-1.75bpw")                  # less RAM than 2.00 was measured with
        self.assertEqual(dsv41_fit.recommend(64.0, 250.0), "IST-DAS-1.75bpw")                  # disk fits 1.75 (239 + 6) but not 2.00 (257 + 6)
        self.assertEqual(dsv41_fit.recommend(64.0, 100.0), "IST-DAS-2.00bpw")                  # fits neither: no reason to prefer it

    def test_the_menu_marks_the_recommendation_and_still_lets_anyone_choose(self):
        import io
        names = list(setup.MULTI_FILE_FAMILIES["deepseek41"]["models"])
        self.assertIn("IST-DAS-1.75bpw", names)
        self.assertEqual(names.index(dsv41_fit.recommend(32.0, 900.0)) + 1, 3)


class InstallOfficial(Install):
    """install_dsv41 for an official set, the engine and the tools mocked."""

    def install_ista(self, model=ISTA, ram=64.0, **kw):
        self.packinfo_ok = True
        self.smoke_ok = True
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (self.packinfo_ok, "files: 6, all present and long enough")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (self.smoke_ok, "The capital of France is Paris.")
                                     if self.smoke_ok else (False, "test question failed: the answer is empty"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return self.install(ram=ram, cards=[RTX5070], model=model, **kw)

    def test_downloads_the_set_the_engram_once_and_the_drafter(self):
        rc, out = self.install_ista()
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        folder = self.root / "models" / "deepseek-v4.1-flash-IST-DAS-2.00bpw"
        shared = self.root / "models" / "deepseek-v4.1-flash-shared"
        self.assertEqual([c[2] for c in get][:4], [folder / f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertEqual([c[2] for c in get][4:], [shared / setup.DSV41_ISTA_ENGRAM, shared / setup.DSV41_SHARD.format(i=1),
                                                  shared / setup.DSV41_DSPARK])
        self.assertTrue(get[0][1].endswith("/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"))
        self.assertNotIn("/2.00bpw/", get[0][1])             # the repository is flat
        self.assertEqual(sum(1 for c in self.calls if c[0] == "sha"), 7)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn("--engram-meta", pack)
        self.assertNotIn("--skip-tensors", pack)

    def test_the_second_set_does_not_download_the_engram_again(self):
        shared = self.root / "models" / "deepseek-v4.1-flash-shared"
        shared.mkdir(parents=True)
        shared_names = (setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK)
        for n in shared_names:
            (shared / n).write_bytes(b"x")
            (shared / (n + ".done")).write_text("ok")
        tiny = {n: (1, setup.DSV41_ALL_FILES[n][1]) for n in shared_names}      # the stand-in files are one byte long
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(model="IST-DAS-2.25bpw")
        self.assertEqual(rc, 0, out)
        got = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertEqual(got, [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-{i:05d}-of-00004.gguf" for i in range(1, 5)])      # only the four shards

    def test_the_first_releases_files_are_used_as_they_are_nothing_is_downloaded_but_the_small_files(self):
        own = self.root / "own"
        sub = own / "2.00bpw"
        sub.mkdir(parents=True)
        for i in range(1, 5):
            (sub / f"model-2.00bpw-{i:05d}-of-00004.gguf").write_bytes(b"GGUF")
        (sub / "engram-IQ4_NL.gguf").write_bytes(b"E")
        (own / "dspark").mkdir()
        (own / "dspark" / setup.DSV41_DSPARK).write_bytes(b"D")
        tiny = {n: (4, "0" * 64) for n in setup.DSV41_LEGACY_FILES if n.startswith("model-2.00bpw")}
        tiny["engram-IQ4_NL.gguf"] = (1, "0" * 64)
        tiny[setup.DSV41_DSPARK] = (1, "0" * 64)
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        got = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertEqual(got, [setup.DSV41_SHARD.format(i=1)])                 # only the 5.8 MB metadata shard
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        shards = pack[pack.index("--shards") + 1:pack.index("--engram")]
        self.assertEqual(shards, [f"model-2.00bpw-{i:05d}-of-00004.gguf" for i in range(1, 5)])      # packed from the files as they are
        self.assertTrue(pack[pack.index("--engram") + 1].endswith("engram-IQ4_NL.gguf"))
        self.assertIn("model", pack[pack.index("--model") + 1] + "model")
        self.assertEqual(Path(pack[pack.index("--model") + 1]), sub)

    def six_shard_folder(self, quant="IQ2_XXS", engram=True):
        """A folder of the person's own with ISTA-DASLab's six-shard files of 2026-10-27 (stand-ins of 4 bytes) and the drafter beside it."""
        own = self.root / "own6"
        own.mkdir(parents=True)
        names = [setup.DSV41_ISTA6_SHARD.format(quant=quant, i=i) for i in range(1, setup.DSV41_ISTA6_SHARDS + 1)]
        for n in names:
            (own / n).write_bytes(b"GGUF")
        if engram:
            (own / setup.DSV41_ISTA_ENGRAM).write_bytes(b"E")
        (own / setup.DSV41_DSPARK).write_bytes(b"D")
        tiny = {n: (4, "0" * 64) for n in names}
        tiny[setup.DSV41_ISTA_ENGRAM] = (1, "0" * 64)
        tiny[setup.DSV41_DSPARK] = (1, "0" * 64)
        return own, names, tiny

    def test_the_persons_six_shard_files_are_used_where_they_are_and_packed_with_the_engram_as_the_sixth(self):
        own, names, tiny = self.six_shard_folder()
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        self.assertIn("six-shard files of 2026-10-27", out)
        self.assertEqual([c[2].name for c in self.calls if c[0] == "get"], [setup.DSV41_SHARD.format(i=1)])      # only the 5.8 MB metadata shard
        self.assertEqual([c[1] for c in self.calls if c[0] == "sha"][:5], names)                                   # the five files are checked by name
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertEqual(pack[pack.index("--shards") + 1:pack.index("--engram")], names)
        self.assertTrue(pack[pack.index("--engram") + 1].endswith(setup.DSV41_ISTA_ENGRAM))
        self.assertEqual(Path(pack[pack.index("--model") + 1]), own)

    def test_six_shard_files_without_the_engram_fetch_the_pinned_engram_and_nothing_else_big(self):
        own, names, tiny = self.six_shard_folder(engram=False)
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(sorted(c[2].name for c in get), sorted([setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1)]))
        engram = next(c for c in get if c[2].name == setup.DSV41_ISTA_ENGRAM)
        self.assertIn("/resolve/cb02422c900df489cea5833e8a92171224059b3c/", engram[1])                              # the pin did not move

    def test_a_folder_with_both_layouts_whole_is_used_as_the_pinned_four_shard_set(self):
        own, names, tiny = self.six_shard_folder()
        four = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shards"]
        for n in four:
            (own / n).write_bytes(b"GGUF")
            tiny[n] = (4, "0" * 64)
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertEqual(pack[pack.index("--shards") + 1:pack.index("--engram")], four)
        self.assertNotIn("six-shard files of 2026-10-27", out)

    def make_pack(self, names):
        pack = self.root / "data" / "packs" / "deepseek-v4.1-flash-ist-das-2.00bpw"
        (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        (pack / "ds4.txt").write_text("vision.num_hidden_layers 32\nimage_token_id 129264\n")       # (the tower is there: only the files differ)
        (pack / "tensors.txt").write_text("".join(f"file {i} {n}\n" for i, n in enumerate(names)) + "t embed.weight BF16 0 0 1 1\n")

    def pack_runs(self):
        return [c for c in self.calls if c[0] == "run" and any(str(x).endswith("pack_gguf.py") for x in c[1])]

    def test_a_pack_made_from_the_other_layout_is_made_again_and_one_from_the_same_layout_is_kept(self):
        own, names, tiny = self.six_shard_folder()
        four = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shards"]
        self.make_pack(four + [setup.DSV41_ISTA_ENGRAM])                        # a pack of the four-shard files, the six-shard files now
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        self.assertIn("made from the four-shard files of this set", out)
        self.assertEqual(len(self.pack_runs()), 1)
        self.make_pack(names + [setup.DSV41_ISTA_ENGRAM])                       # a pack of the six-shard files: kept
        self.calls.clear()
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        self.assertNotIn("building it again", out)
        self.assertEqual(len(self.pack_runs()), 0)

    def test_a_four_shard_pack_is_kept_for_the_pinned_four_shard_files(self):
        self.make_pack(setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shards"] + [setup.DSV41_ISTA_ENGRAM])
        rc, out = self.install_ista()
        self.assertEqual(rc, 0, out)
        self.assertNotIn("building it again", out)
        self.assertEqual(len(self.pack_runs()), 0)

    def test_pack_info_profile_smoke_and_the_launch_flags(self):
        rc, out = self.install_ista()
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "auto")                    # never "off": that still loads the drafter and takes the VRAM
        self.assertNotEqual(a[a.index("--dspark") + 1], "off")
        self.assertNotIn("--prefill-chunk", a)                                  # the engine's own default is the layer-major prompt path
        self.assertEqual(a[a.index("--dspark-conf") + 1], "0.65")                # never a floor below 0.3             # the layer-major prompt path
        self.assertNotIn("--ssd-cache", a)                                      # the store sizes itself
        self.assertEqual(a[a.index("--max-context") + 1], "8192")
        self.assertIn("checked: every file the engine needs is there", out)
        self.assertIn("test question answered sensibly", out)
        self.assertTrue(Path(a[a.index("--pack") + 1]).match("*packs/deepseek-v4.1-flash-ist-das-2.00bpw"))

    def test_an_incomplete_pack_stops_with_the_missing_file(self):
        self.packinfo_ok = False
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (False, "  MISSING x.gguf (1.00 GB wanted)")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (True, "Paris"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc, out = self.install(ram=64.0, cards=[RTX5070], model=ISTA)
        self.assertEqual(rc, 1, out)
        self.assertIn("pack is not complete", out)
        self.assertIn("MISSING", out)

    def test_a_failed_text_check_is_said_plainly_and_stops(self):
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (True, "all present")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (False, "!!!!\ntest question failed: the answer is empty"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc, out = self.install(ram=64.0, cards=[RTX5070], model=ISTA)
        self.assertEqual(rc, 1, out)
        self.assertIn("does not answer sensibly", out)
        self.assertIn("run setup again", out)
        self.assertFalse(self.cfg)                                              # no start script for a model that failed its check

    def test_a_disk_that_cannot_hold_the_set_stops(self):
        with mock.patch.object(setup, "free_gb", lambda p: 100.0):
            rc, out = self.install_ista()
        self.assertEqual(rc, 1, out)
        self.assertIn("not enough free disk space", out)


class InstallSmallest(InstallOfficial):
    def test_downloads_the_iq1_m_files_and_the_shared_ones(self):
        rc, out = self.install_ista(model="IST-DAS-1.75bpw", ram=36.0)
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual([c[2].name for c in get][:4], [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertTrue(get[0][1].startswith(setup.hf(setup.DSV41_ISTA_REPO)))
        self.assertIn("cb02422c900df489cea5833e8a92171224059b3c", get[0][1])
        self.assertIn("239 GB", out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "auto")


if __name__ == "__main__":
    unittest.main()


class PicturesByDefault(Install):
    """0.2.7: DeepSeek reads pictures with its own engine's tower, on by default: setup fetches the tower's weights (mmproj-BF16.gguf, a shared file),
    packs them with the model and writes --vision on; --vision no leaves all three out."""

    def test_default_fetches_the_tower_packs_it_and_says_on(self):
        rc, out = self.install(vision=None)
        self.assertEqual(rc, 0, out)
        got = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertIn(setup.DSV41_MMPROJ, got)
        self.assertEqual(got[-1], setup.DSV41_MMPROJ)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn("--mmproj", pack)
        self.assertTrue(str(pack[pack.index("--mmproj") + 1]).endswith(setup.DSV41_MMPROJ))
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vision") + 1], "on")
        self.assertNotIn("cannot read pictures", out)

    def test_vision_no_leaves_the_tower_out(self):
        rc, out = self.install(vision="no")
        self.assertEqual(rc, 0, out)
        self.assertNotIn(setup.DSV41_MMPROJ, [c[2].name for c in self.calls if c[0] == "get"])
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertNotIn("--mmproj", pack)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vision") + 1], "off")

    def test_gpu_and_cpu_are_taken_as_yes_with_a_note(self):
        rc, out = self.install(vision="cpu")
        self.assertEqual(rc, 0, out)
        self.assertIn("own tower", out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--vision") + 1], "on")

    def test_a_pack_without_the_tower_is_made_again_with_it(self):
        pack = self.root / "data" / "packs" / "deepseek-v4.1-flash-ud-q8_k_xl"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "vocab.json").write_text("{}")
        (pack / "tensors.txt").write_text("x")
        (pack / "ds4.txt").write_text("hidden_size 5120\n")
        self.assertFalse(setup.dsv41_pack_has_tower(pack))
        rc, out = self.install(vision=None)
        self.assertEqual(rc, 0, out)
        self.assertIn("made without the picture reader", out)
        self.assertTrue([c for c in self.calls if c[0] == "run" and "--mmproj" in c[1]])
        pack.mkdir(parents=True, exist_ok=True)
        (pack / "ds4.txt").write_text("vision.num_hidden_layers 32\nimage_token_id 129264\n")
        self.assertTrue(setup.dsv41_pack_has_tower(pack))
