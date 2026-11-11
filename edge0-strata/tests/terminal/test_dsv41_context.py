"""DeepSeek-V4.1-Flash's context sizes, 8K up to 1M (Project Maya #442 by @needmorevram): the choices, the pack's own limit, the refusal of a size the
model cannot address, the words past 256K, and that setup, the Terminal and the app say the same.  Nothing is downloaded or started."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_control import dsv41_context as C
from tests.terminal.launch_support import HW_12, Home, catalog, run_cli

ROOT = Path(__file__).resolve().parents[2]
R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "usable": True, "vendor": "amd"}
HW_4X32 = {**HW_12, "ram_gb": 251.0, "disk_free_gb": 900.0, "engine_vendor": "amd", "gpus": [{**R9700, "index": i} for i in range(4)]}


def pack_with(limit_line: str | None) -> Path:
    d = Path(tempfile.mkdtemp(prefix="strata-ds4-pack-"))
    (d / "ds4.txt").write_text("vocab_size 129280\n" + (limit_line + "\n" if limit_line else "") + "num_hidden_layers 43\n", encoding="utf-8")
    return d


class Choices(unittest.TestCase):
    def test_the_sizes_go_up_to_one_million_and_are_labelled_512K_and_1M(self):
        self.assertEqual(C.CHOICES, (8192, 32768, 65536, 131072, 262144, 524288, 1048576))
        self.assertEqual(C.labels(C.CHOICES), ["8K", "32K", "64K", "128K", "256K", "512K", "1M"])
        self.assertEqual([S.context_label(c) for c in C.CHOICES], ["8K", "32K", "64K", "128K", "256K", "512K", "1M"])
        self.assertEqual(S.context_label(2097152), "2M")
        self.assertEqual(S.parse_context("1m"), 1048576)
        self.assertEqual(S.parse_context("512k"), 524288)
        self.assertEqual(S.context_label(S.parse_context("1M")), "1M")

    def test_one_million_is_accepted_and_two_million_refused(self):
        self.assertEqual(C.problem(524288), "")
        self.assertEqual(C.problem(1048576), "")
        self.assertIn("2M is more than DeepSeek-V4.1-Flash reads (1M)", C.problem(2097152))
        self.assertNotEqual(C.problem(1048577), "", "one position more than the model reads")
        self.assertNotEqual(C.problem(512), "", "too small to be useful")

    def test_the_pack_limit_is_read_from_ds4_txt_and_is_the_last_word(self):
        full = pack_with("max_position_embeddings 1048576")
        small = pack_with("max_position_embeddings 262144")
        none = pack_with(None)
        for d in (full, small, none):
            self.addCleanup(shutil.rmtree, d, True)
        self.assertEqual(C.read_limit(full), 1048576)
        self.assertEqual(C.read_limit(small), 262144)
        self.assertIsNone(C.read_limit(none), "a pack without the line behaves as before")
        self.assertIsNone(C.read_limit(full / "nowhere"))
        self.assertIsNone(C.read_limit(None))
        self.assertEqual(C.choices(C.read_limit(full)), list(C.CHOICES))
        self.assertEqual(C.choices(C.read_limit(small)), [8192, 32768, 65536, 131072, 262144])
        self.assertEqual(C.choices(None), list(C.CHOICES), "setup and the Terminal know the model: up to 1M without a pack")
        self.assertEqual(C.choices(None, unknown_cap=C.TESTED), [8192, 32768, 65536, 131072, 262144], "the app: up to 256K without a pack")
        self.assertEqual(C.problem(524288, C.read_limit(small)).split(" is ")[0], "512K")
        self.assertEqual(C.problem(262144, C.read_limit(small)), "")
        cfg = {"args": ["--pack", str(small), "--max-context", "8192"]}
        self.assertEqual(C.limit_of_config(cfg), 262144)
        self.assertIsNone(C.limit_of_config({"args": ["--native", "m.gguf"]}))

    def test_past_256K_the_note_says_it_is_untested_and_what_it_costs(self):
        self.assertEqual(C.note(262144), "")
        self.assertEqual(C.note(8192), "")
        for size in (524288, 1048576):
            self.assertEqual(C.note(size), "Not tested yet. A context this long takes graphics memory from the model and reads long prompts slowly.")


class Terminal(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(HW_4X32))
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args, ok=True):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        if ok:
            self.assertEqual(code, 0, err + out)
        return code, out, err

    def test_the_context_step_offers_8K_to_1M_for_deepseek_and_keeps_the_others(self):
        cat = catalog(HW_4X32)
        self.assertEqual(cat.contexts_for("deepseek41", "UD-Q8_K_XL"), list(C.CHOICES))
        self.assertEqual(cat.contexts_for("qwen", "IQ3_S"), cat.contexts(), "Qwen's choices stay as they are")
        self.assertNotIn(1048576, cat.contexts_for("qwen", "IQ3_S"))

    def test_the_flag_takes_512k_and_1m(self):
        for text, tokens in (("512k", 524288), ("1m", 1048576), ("524288", 524288), ("1048576", 1048576)):
            code, out, err = self.plan("--model", "ud-q8_k_xl", "--context", text, "--no-tui")
            plan = json.loads(out)["plan"]
            self.assertEqual(plan["settings"]["context"], tokens, text)
            self.assertTrue(any("Not tested yet" in n["text"] for n in plan["notices"]) if "notices" in plan else "Not tested yet" in out, text)

    def test_two_million_is_refused_with_a_plain_sentence(self):
        code, out, err = self.plan("--model", "ud-q8_k_xl", "--context", "2m", "--no-tui", ok=False)
        self.assertNotEqual(code, 0)
        self.assertIn("2M is more than DeepSeek-V4.1-Flash reads (1M)", out + err)

    def test_256k_has_no_untested_note_and_qwen_keeps_its_rope_scaling_words(self):
        code, out, err = self.plan("--model", "ud-q8_k_xl", "--context", "256k", "--no-tui")
        self.assertNotIn("Not tested yet", out + err)
        notes = S.check_context(S.LaunchSettings(context=524288))
        self.assertIn("rope scaling", notes[0].next)
        deep = S.check_context(S.LaunchSettings(context=524288), deepseek=True)
        self.assertEqual([n.level for n in deep], ["warn"])
        self.assertIn("Not tested yet", deep[0].text)
        self.assertEqual(S.check_context(S.LaunchSettings(context=262144), deepseek=True), [])
        self.assertEqual([n.level for n in S.check_context(S.LaunchSettings(context=2097152), deepseek=True)], ["error"])
        self.assertEqual([n.level for n in S.check_context(S.LaunchSettings(context=524288), deepseek=True, pack_limit=262144)], ["error"])


class SetupAndApp(unittest.TestCase):
    def test_nobody_gets_a_bigger_context_than_before_unless_they_choose_it(self):
        """The default stays 8K everywhere: setup writes it, the Terminal recommends it, the app starts a new DeepSeek version at it and shows an installed one at
        the context it was prepared with (so a 0.2.5 / 0.2.6 user's load is the saved file's, as it was)."""
        import setup
        self.assertEqual(setup.DSV41_CONTEXT, 8192)
        with Home(), mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_4X32)):
            self.assertEqual(catalog(HW_4X32).max_context("deepseek41", "UD-Q8_K_XL"), 8192)
        screen = (ROOT / "apps" / "host" / "src" / "components" / "NativeModels.tsx").read_text(encoding="utf-8")
        self.assertRegex(screen, r"if \(selectedSize\.multi_file\)\s+setContext\(8192\)")
        self.assertRegex(screen, r"if \(installed\.context\)\s+setContext\(installed\.context\)")

    def test_setup_stops_before_any_download_and_checks_the_pack_again(self):
        source = (ROOT / "setup.py").read_text(encoding="utf-8")
        early = source.index("ctx = a.context or DSV41_CONTEXT")
        self.assertIn("dsv41_context.problem(ctx)", source[early:early + 400])
        self.assertLess(early, source.index("folder = dsv41_folder(a, fam, tag, model)", early) + 1)
        self.assertIn("dsv41_context.read_limit(pack)", source)

    def test_the_install_tool_refuses_a_size_the_model_cannot_read(self):
        source = (ROOT / "tools" / "strata_mcp.py").read_text(encoding="utf-8")
        self.assertIn("dsv41_context.problem(int(context))", source)

    def test_the_engine_does_not_clamp_max_context(self):
        main = (ROOT / "src" / "dsv41" / "ds4_main.cpp").read_text(encoding="utf-8", errors="replace")
        self.assertIn('o.max_context = std::atoll(val("--max-context").c_str());', main)
        self.assertNotRegex(main, r"max_context\s*=\s*std::(min|clamp)")
        model = (ROOT / "src" / "dsv41" / "ds4_model.cu").read_text(encoding="utf-8", errors="replace")
        self.assertNotRegex(model, r"max_context\s*=\s*std::(min|clamp)")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_the_app_says_the_same(self):
        script = ("import {CONTEXT_CHOICES,TESTED_CONTEXT,NATIVE_CONTEXT,UNTESTED_NOTE} from './apps/host/src/lib/deepseek-context.ts';"
                  "console.log(JSON.stringify({CONTEXT_CHOICES,TESTED_CONTEXT,NATIVE_CONTEXT,UNTESTED_NOTE}))")
        out = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            self.skipTest("this node cannot load TypeScript: " + out.stderr[-200:])
        app = json.loads(out.stdout)
        self.assertEqual(app["CONTEXT_CHOICES"], list(C.CHOICES))
        self.assertEqual((app["TESTED_CONTEXT"], app["NATIVE_CONTEXT"], app["UNTESTED_NOTE"]), (C.TESTED, C.NATIVE, C.NOTE))


if __name__ == "__main__":
    unittest.main()
