"""`strata launch` without screens: the settings rules, the catalog, presets, parameters, access, connect, and every flag."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import access, harness, params, settings as S, store
from strata_cli.launch.catalog import Catalog, CatalogUnavailable, fit_on, verdict
from strata_cli.launch.runner import Result, Runner, derive_config, desc_matches, tune_words
from tests.terminal.launch_support import HW_12, HW_24, HW_2X16, Home, catalog, run_cli


class SettingsRules(unittest.TestCase):
    def test_parse_context(self):
        self.assertEqual(S.parse_context("128k"), 131072)
        self.assertEqual(S.parse_context("128K"), 131072)
        self.assertEqual(S.parse_context("32 k"), 32768)
        self.assertEqual(S.parse_context("65536"), 65536)
        with self.assertRaises(ValueError):
            S.parse_context("lots")
        with self.assertRaises(ValueError):
            S.parse_context("100")

    def test_fit_parallel_follows_the_spec(self):
        self.assertEqual([S.fit_parallel(v) for v in (8, 12, 16, 20, 24, 32, 48)], [1, 1, 1, 2, 3, 4, 4])
        self.assertEqual(S.fit_parallel(None), 1)

    def test_words_and_labels(self):
        self.assertEqual(S.context_label(131072), "128K")
        self.assertEqual(S.words_for(131072), "about 98,000 words")

    def test_parallel_warning_is_a_warning_not_an_error(self):
        notes = S.check_parallel(S.LaunchSettings(parallel=4), vram_gb=12)
        self.assertEqual([n.level for n in notes], ["warn"])
        self.assertIn("still start", notes[0].line())
        self.assertEqual(S.check_parallel(S.LaunchSettings(parallel=9), vram_gb=12)[0].level, "error")
        self.assertEqual(S.check_parallel(S.LaunchSettings(parallel=2), vram_gb=24), [])

    def test_no_key_warnings_are_plain_and_stronger_in_public(self):
        lan = S.check_access(S.LaunchSettings(access="lan", no_key=True))
        self.assertEqual([n.level for n in lan], ["warn"])
        self.assertIn("anyone who can reach this port can use the model and its tools", lan[0].line())
        pub = S.check_access(S.LaunchSettings(access="cloudflare", no_key=True))
        self.assertEqual([n.level for n in pub], ["warn"])
        self.assertIn("public address", pub[0].line())
        self.assertIn("anyone on the internet", pub[0].line())
        self.assertEqual(S.check_access(S.LaunchSettings(access="lan", api_key="k")), [])
        self.assertEqual(S.check_access(S.LaunchSettings(access="local", no_key=True)), [])

    def test_access_notes(self):
        self.assertEqual(S.check_access(S.LaunchSettings(access="local")), [])
        lan = S.check_access(S.LaunchSettings(access="lan"))
        self.assertEqual([n.level for n in lan], ["warn"])
        self.assertIn("API key", lan[0].line())
        bad = S.check_access(S.LaunchSettings(access="lann"))
        self.assertEqual(bad[0].level, "error")
        self.assertIn("lan", bad[0].line())
        self.assertEqual(S.check_port("abc").level, "error")
        self.assertEqual(S.check_port(70000).level, "error")
        self.assertEqual(S.check_port(80).level, "warn")
        self.assertIsNone(S.check_port(8080))

    def test_split_setting(self):
        self.assertEqual(S.split_setting("prefill=65536"), ("--prefill", "65536"))
        self.assertEqual(S.split_setting("--kv-grow"), ("--kv-grow", ""))
        self.assertEqual(S.split_setting("STRATA_PIN_BY_NEED=1"), ("STRATA_PIN_BY_NEED", "1"))
        self.assertEqual(S.split_setting("expert_cache=auto"), ("--expert-cache", "auto"))
        self.assertEqual(S.split_setting("IGC_EnableDPEmulation=1"), ("IGC_EnableDPEmulation", "1"))        # environment names keep their case
        self.assertEqual(S.split_setting("ONEAPI_DEVICE_SELECTOR=level_zero:gpu"), ("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"))
        self.assertEqual(S.split_setting("--prefill=4096"), ("--prefill", "4096"))
        with self.assertRaises(ValueError):
            S.split_setting("no spaces allowed=1")

    def test_flags_text_round_trips_to_the_same_choices(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=131072, parallel=2, tune="quick", access="lan", port=9000,
                             extra={"--prefill": "65536", "STRATA_PIN_BY_NEED": "1"})
        text = S.flags_text(s)
        for part in ("--model iq3_xxs", "--context 128k", "--parallel 2", "--tune quick", "--access lan", "--port 9000",
                     "--set prefill=65536", "--set STRATA_PIN_BY_NEED=1"):
            self.assertIn(part, text)
        self.assertIn('--preset "Fast"', S.flags_text(s, preset="Fast"))

    def test_slug(self):
        self.assertEqual(S.slug("IQ3_XXS 128K - Full speed"), "iq3_xxs-128k-full-speed")
        self.assertTrue(S.slug("123 go")[0].isalpha())

    def test_roundtrip_dict(self):
        s = S.LaunchSettings(name="a", model="Q2_0", family="qwen", context=32768, extra={"--spec": "3"}, tuned_preset="abc")
        again = S.LaunchSettings.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(again, s)
        self.assertEqual(s.to_dict(with_key=False)["api_key"], None)
        self.assertEqual(S.LaunchSettings(api_key="sk-x").to_dict(with_key=False)["api_key"], "set")


class CatalogRules(unittest.TestCase):
    def test_one_source_of_truth_and_fit_numbers(self):
        with Home():
            cat = catalog(HW_12)
            rows = cat.rows()
            ids = {(r.family, r.model) for r in rows}
            self.assertIn(("qwen", "IQ3_XXS"), ids)
            models, families, contexts, source = cat.tables()
            self.assertEqual(source, "setup.py")                       # the same tables setup.py and the app read
            for r in rows:
                if r.family == "deepseek41":                           # its sizes are in MULTI_FILE_FAMILIES (shards + shared files), not in MODELS
                    self.assertAlmostEqual(r.download_gb, {"UD-Q8_K_XL": 508.4, "IST-DAS-2.00bpw": 256.8, "IST-DAS-2.25bpw": 273.8, "IST-DAS-1.75bpw": 239.3, "UD-IQ1_S": 241.0, "UD-IQ1_M": 258.4}[r.model], delta=0.1)
                    continue
                self.assertAlmostEqual(r.download_gb, models[r.model]["download_gb"])
            rec = [r for r in rows if r.recommended]
            self.assertEqual([(r.family, r.model) for r in rec], [("qwen", "IQ3_S")])      # 64 + 12 GB: the 64 GB tier

    def test_the_recommendation_follows_the_memory_tiers(self):
        # the owner's tiers count RAM + VRAM: 44 GB is the 40 GB tier (Swift 1.5 IQ2_XS, Gyro-M named), 60 GB the 48 GB tier
        from tests.terminal.launch_support import HW_32GB_RAM
        with Home():
            rec = catalog(HW_32GB_RAM).recommendation(HW_32GB_RAM)
            self.assertEqual((rec["family"], rec["model"]), ("swift", "IQ2_XS"))
            self.assertIn("Gyro-M", rec["why"])
            hw = {**HW_32GB_RAM, "ram_gb": 48.0}
            rec = catalog(hw).recommendation(hw)
            self.assertEqual((rec["family"], rec["model"]), ("qwen", "IQ3_XXS"))

    def test_fit_on_matches_native_fit_ts(self):
        # apps/host/src/lib/native-fit.ts: memory = experts + 1; GPU holds experts up to vram - 5; RAM holds up to ram - 10; the rest spills
        f = fit_on({"experts_gb": 42.9, "ram_needed_gb": 60}, 64.0, 12.0)
        self.assertEqual((f["memoryGb"], f["vramUseGb"], f["spillGb"]), (43.9, 12.0, 0.0))
        self.assertAlmostEqual(f["ramUseGb"], 35.9, places=1)
        spill = fit_on({"experts_gb": 98.4, "ram_needed_gb": 108}, 64.0, 12.0)
        self.assertGreater(spill["spillGb"], 30)
        self.assertFalse(fit_on({"experts_gb": 10}, None, 12)["known"])
        gyro = fit_on({"mem_total": True, "ram_needed_gb": 27.6}, 64, 12)
        self.assertEqual(gyro["memoryGb"], 27.6)

    def test_a_mac_is_judged_by_unified_memory_and_the_words_follow(self):
        hw = {"ram_gb": 32.0, "os": "Darwin", "cpu": {"avx2": None, "name": "Apple M4"}, "mac": {"chip": "Apple M4", "macos": "27.2", "unified_gb": 32.0},
              "gpus": [{"index": 0, "name": "Apple M4", "vendor": "apple", "vram_gb": 0.0, "unified_gb": 32.0, "usable": True}]}
        with Home():
            cat = catalog(hw)
            rows = {r.model: r for r in cat.rows()}
            self.assertEqual(set(rows), {"Q2_0", "IQ3_XXS", "IQ3_S", "Gyro-S", "Gyro-M", "IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "UD-IQ1_S", "UD-IQ1_M"})   # DeepSeek (0.2.7.1): the two IST-DAS sets run on a Mac, UD-IQ1_S since f-039
            self.assertEqual([r.model for r in rows.values() if r.recommended], ["Q2_0"])
            for name in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw"):            # a 32 GB Mac: said plainly (mostly read from the SSD), never refused
                self.assertEqual(rows[name].family, "deepseek41")
                self.assertEqual(rows[name].verdict, "tight")
                self.assertFalse(rows[name].images)
                self.assertGreater(rows[name].spill_gb, 100)
            self.assertEqual((rows["Q2_0"].verdict, rows["Q2_0"].words), ("fits", "fits, part read from SSD"))
            self.assertEqual(rows["IQ3_S"].verdict, "tight")
            self.assertGreater(rows["IQ3_S"].spill_gb, 10)
            self.assertEqual(cat.max_context("qwen", "Q2_0", hw), 32768)
            self.assertEqual(cat.computer().line(), "Apple M4 · 32 GB unified memory · " + cat.computer().line().split(" · ")[-1])
            # no "0 GB" of video memory in the card part of the line (the free-disk figure after it may end in a zero: "60 GB free")
            self.assertNotIn("0 GB", " · ".join(cat.computer().line().split(" · ")[:2]))

    def test_verdict_words_never_block(self):
        self.assertEqual(verdict("fits")[0], "fits")
        self.assertEqual(verdict("does not fit")[0], "no")
        self.assertIn("can still try", verdict("does not fit")[1])
        self.assertEqual(verdict("tight")[0], "tight")
        self.assertEqual(verdict("fits in the low-RAM mode (slower)")[0], "fits")

    def test_resolve_names_and_closest(self):
        with Home():
            cat = catalog()
            self.assertEqual(cat.resolve("iq3_xxs"), ("qwen", "IQ3_XXS"))
            self.assertEqual(cat.resolve("IQ1_M"), ("coder", "IQ1_M"))
            self.assertEqual(cat.resolve("iq3_xxs", "swift"), ("swift", "IQ3_XXS"))
            with self.assertRaises(ValueError) as e:
                cat.resolve("iq3xxs")
            self.assertIn("Did you mean 'IQ3_XXS'", str(e.exception))
            with self.assertRaises(ValueError) as e:
                cat.resolve("q2_0", "swiift")
            self.assertIn("swift", str(e.exception))
            with self.assertRaises(ValueError) as e:
                cat.resolve("IQ1_M", "swift")
            self.assertIn("has no size", str(e.exception))

    def test_small_computer_gets_warnings_not_blocks(self):
        with Home():
            rows = catalog({**HW_12, "ram_gb": 32.0}).rows()
            big = [r for r in rows if r.model == "IQ3_S" and r.family == "qwen"][0]
            self.assertIn(big.verdict, ("no", "tight"))

    def test_kv_figures(self):
        with Home():
            cat = catalog()
            self.assertAlmostEqual(cat.kv_gb(131072, "int8"), 1.799, places=2)
            self.assertLess(cat.kv_gb(131072, "q4_0"), cat.kv_gb(131072, "int8"))
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", HW_12), 32768)
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", HW_24), 131072)

    def test_missing_installer_files_are_said_in_words(self):
        with Home():
            with self.assertRaises(CatalogUnavailable) as e:
                Catalog(src="/nonexistent-folder", hardware=HW_12)
            self.assertIn("one-line installer", str(e.exception))


class PresetStore(unittest.TestCase):
    def test_save_load_list_delete(self):
        with Home():
            s = S.LaunchSettings(name="IQ3_XXS 128K", family="qwen", model="IQ3_XXS", context=131072, api_key="sk-secret")
            store.save(s)
            store.save(S.LaunchSettings(name="Coder 64K", family="coder", model="IQ1_M", context=65536))
            names = [p.name for p in store.list_presets()]
            self.assertEqual(sorted(names), ["Coder 64K", "IQ3_XXS 128K"])
            back = store.load("IQ3_XXS 128K")
            self.assertEqual((back.model, back.context, back.api_key), ("IQ3_XXS", 131072, "sk-secret"))
            store.touch_used("IQ3_XXS 128K")
            self.assertTrue(store.load("IQ3_XXS 128K").last_used)
            self.assertEqual(store.list_presets()[0].name, "IQ3_XXS 128K")          # most recently used first
            store.delete("Coder 64K")
            self.assertEqual([p.name for p in store.list_presets()], ["IQ3_XXS 128K"])
            with self.assertRaises(store.PresetError) as e:
                store.load("IQ3 XXS 128")
            self.assertIn("Did you mean", str(e.exception))
            with self.assertRaises(store.PresetError):
                store.delete("nope")

    def test_key_is_not_world_readable_on_posix(self):
        if os.name == "nt":
            self.skipTest("POSIX modes")
        with Home():
            store.save(S.LaunchSettings(name="k", api_key="sk-secret"))
            path = next(store.preset_dir().glob("*.json"))
            self.assertEqual(path.stat().st_mode & 0o077, 0)


class Advanced(unittest.TestCase):
    HELP = """strata generate --pack DIR [options]

  --pack DIR           the pack directory (default pack/full)
  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes, half the
                       VRAM; default fp16 until the gate accepts int8
  --kv q4_0            4-bit K/V after a Hadamard rotation
  --kv-grow            the K/V takes VRAM only for the cells the requests reach
  --native-rope        experimental pinned text-only CUDA rotary arithmetic
  --prefill N          prompt chunk size (default 2048)
"""

    def test_parse_help(self):
        ps = {p.name: p for p in params.parse_help(self.HELP)}
        self.assertEqual(set(ps), {"--pack", "--kv", "--kv-grow", "--native-rope", "--prefill"})
        self.assertEqual(ps["--pack"].metavar, "DIR")
        self.assertEqual(ps["--pack"].help_default, "pack/full")
        self.assertEqual(ps["--kv"].values, ["fp16", "int8"])
        self.assertIn("4-bit", ps["--kv"].description)               # a flag listed twice keeps both texts
        self.assertTrue(ps["--native-rope"].experimental)
        self.assertFalse(ps["--prefill"].experimental)
        self.assertEqual(ps["--kv-grow"].metavar, "")
        self.assertLessEqual(len(ps["--prefill"].short), 70)

    def test_rows_show_default_tuned_and_yours(self):
        rows = {r.name: r for r in params.build_rows(self.HELP, ["--kv", "int8", "--prefill", "auto"], ["--kv", "int8", "--prefill", "auto:32768"],
                                                     {}, {}, {"--prefill": "65536"})}
        self.assertEqual((rows["--prefill"].default, rows["--prefill"].tuned, rows["--prefill"].yours), ("auto", "auto:32768", "65536"))
        self.assertEqual(rows["--kv"].yours, "")
        self.assertIn("--native-rope", rows)
        names = list(rows)
        self.assertEqual(names[:2], ["--kv", "--prefill"])             # what this launch uses comes first

    def test_unknown_names_warn_and_are_kept(self):
        rows = params.build_rows(self.HELP, [], None, {}, {}, {"--prefil": "1"})
        r = next(x for x in rows if x.name == "--prefil")
        self.assertFalse(r.known)
        warn = params.check_name("--prefil", ["--prefill", "--kv"])
        self.assertIn("Did you mean --prefill", warn)
        self.assertIsNone(params.check_name("--prefill", ["--prefill"]))
        self.assertIsNone(params.check_name("--prefil", None))        # no list to check against: no warning

    def test_extra_becomes_argv_and_env(self):
        a, e = S.as_argv_changes({"--prefill": "65536", "--kv-grow": "", "--spec": "off", "STRATA_PIN_BY_NEED": "1"}, frozenset({"--kv-grow"}))
        self.assertEqual(a, {"--prefill": "65536", "--kv-grow": True, "--spec": None})
        self.assertEqual(e, {"STRATA_PIN_BY_NEED": "1"})


class RunConfig(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--native", "x.gguf", "--max-context", "32768", "--kv", "int8", "--prefill", "auto"],
            "port": 8095, "model_name": "m", "env": {}}

    def test_derive_applies_access_key_and_advanced(self):
        s = S.LaunchSettings(access="lan", port=9001, api_key="sk-k", extra={"--prefill": "65536", "--kv-grow": "", "STRATA_PIN_BY_NEED": "1"})
        cfg = derive_config(self.BASE, s, bool_flags=frozenset({"--kv-grow"}))
        self.assertEqual((cfg["host"], cfg["port"], cfg["api_key"]), ("0.0.0.0", 9001, "sk-k"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--prefill") + 1], "65536")
        self.assertIn("--kv-grow", args)
        self.assertEqual(cfg["env"], {"STRATA_PIN_BY_NEED": "1"})
        self.assertEqual(self.BASE["args"][-1], "auto")               # the installed config is never edited

    def test_mtp_off_leaves_the_draft_layer_out_of_this_launch_only(self):
        base = {**self.BASE, "args": [*self.BASE["args"], "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt"]}
        cfg = derive_config(base, S.LaunchSettings(mtp="off"), bool_flags=frozenset())
        self.assertIs(cfg["mtp"], False)
        for flag in ("--mtp", "--spec-min-p", "--batch-mtp"):
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--spec") + 1], "4")      # prompt lookup keeps its windows
        self.assertIn("--mtp", base["args"])                                      # the installed config is never edited
        on = derive_config({**base, "mtp": False}, S.LaunchSettings(mtp="on"), bool_flags=frozenset())
        self.assertIs(on["mtp"], True)                                            # an explicit on beats an installed off
        self.assertEqual(derive_config(base, S.LaunchSettings(), bool_flags=frozenset())["args"], base["args"])

    def test_local_and_cloudflare_stay_on_loopback_without_a_key(self):
        for kind in ("local", "cloudflare"):
            cfg = derive_config(self.BASE, S.LaunchSettings(access=kind), bool_flags=frozenset())
            self.assertEqual(cfg["host"], "127.0.0.1")
            self.assertNotIn("api_key", cfg)

    def test_no_key_is_written_as_none_and_a_key_wins(self):
        cfg = derive_config(self.BASE, S.LaunchSettings(access="lan", no_key=True), bool_flags=frozenset())
        self.assertEqual((cfg["host"], cfg["api_key"]), ("0.0.0.0", "none"))
        cfg = derive_config(self.BASE, S.LaunchSettings(access="lan", no_key=True, api_key="sk-x"), bool_flags=frozenset())
        self.assertEqual(cfg["api_key"], "sk-x")
        cfg = derive_config(self.BASE, S.LaunchSettings(access="local", no_key=True), bool_flags=frozenset())
        self.assertNotIn("api_key", cfg)

    def test_installed_config_matching(self):
        d = {"context": 32768, "kv": "int8", "images": "off"}
        self.assertTrue(desc_matches(d, S.LaunchSettings(context=32768)))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=65536)))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=32768, vision="on")))
        self.assertFalse(desc_matches({**d, "images": "cpu"}, S.LaunchSettings(context=32768, vision="off")))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=32768, kv="q4_0")))

    def test_tune_words(self):
        self.assertEqual(tune_words({"outcome": "improved", "gain_pct": 4.2})[0], "faster")
        self.assertEqual(tune_words({"outcome": "default_confirmed"})[0], "already fastest")
        self.assertEqual(tune_words({"outcome": "inconclusive", "reason_code": "noisy"})[0], "busy PC")
        self.assertEqual(tune_words({"outcome": "base_failed", "message": "Not tuned: no memory"})[0], "failed")
        self.assertEqual(tune_words(None)[0], "failed")
        self.assertEqual(tune_words({"outcome": "cancelled"})[0], "stopped")


class SetupArgs(unittest.TestCase):
    def runner(self, **kw):
        with Home():
            cat = catalog(HW_12)
            kw.setdefault("vision", "off")
            s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, kv="q4_0", **kw)
            return Runner(s, catalog=cat).setup_args()

    def args_for(self, **kw):
        values = dict(family="qwen", model="IQ3_XXS", context=32768, kv="q4_0", vision="off")
        values.update(kw)
        with Home():
            return Runner(S.LaunchSettings(**values), catalog=catalog(HW_12)).setup_args()

    def test_a_lower_case_tag_from_a_saved_preset_reaches_setup_as_its_key(self):
        """#4209: a preset keeps "model": "iq3_xxs", and setup's --model choices are the table's own keys
        ("IQ3_XXS"). Starting such a preset with any changed choice reruns setup, which then refused it."""
        args = self.args_for(model="iq3_xxs")
        self.assertEqual(args[args.index("--model") + 1], "IQ3_XXS")
        self.assertEqual(args[args.index("--family") + 1], "qwen")
        for fam, model, want in (("coder", "iq1_m", "IQ1_M"), ("gyro", "gyro-s", "Gyro-S"), ("qwen", "IQ3_S", "IQ3_S")):
            args = self.args_for(family=fam, model=model)
            self.assertEqual(args[args.index("--family") + 1], fam)
            self.assertEqual(args[args.index("--model") + 1], want)

    def test_an_unknown_model_is_still_setup_to_report(self):
        args = self.args_for(model="not-a-model")
        self.assertEqual(args[args.index("--model") + 1], "not-a-model")

    def test_a_model_list_that_cannot_be_read_leaves_the_values_alone(self):
        """The resolver is an improvement, never a new way for the launch to fail: when it cannot answer, the
        preset's own values go to setup as they did before #4209."""
        with Home():
            cat = catalog(HW_12)
            r = Runner(S.LaunchSettings(family="qwen", model="iq3_xxs", context=32768, vision="off"), catalog=cat)
            with mock.patch.object(cat, "resolve", side_effect=RuntimeError("the model list cannot be read")):
                args = r.setup_args()
        self.assertEqual(args[args.index("--model") + 1], "iq3_xxs")
        self.assertEqual(args[args.index("--family") + 1], "qwen")

    def test_pictures_default_on_the_cpu_reader_under_24_gb_and_the_gpu_reader_from_24_gb_on_every_make(self):
        from tests.terminal.launch_support import HW_12 as NV
        card = lambda name, gb, vendor: {**NV, "gpus": [{"index": 0, "name": name, "vram_gb": gb, "usable": True, "vendor": vendor}]}   # noqa: E731
        amd = card("AMD Radeon RX 9070 XT", 16.0, "amd")
        intel = card("Intel Arc B580", 12.0, "intel")
        with Home():
            for hw, want in ((NV, "cpu"), (amd, "cpu"), (intel, "cpu"), (card("NVIDIA GeForce RTX 4090", 24.0, "nvidia"), "gpu"),
                             (card("AMD Radeon AI PRO R9700", 32.0, "amd"), "gpu")):
                cat = catalog(hw)
                r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768), catalog=cat)
                self.assertEqual(r.setup_args()[r.setup_args().index("--vision") + 1], want, hw["gpus"][0]["vendor"])
                self.assertEqual(r.s.vision, want)
            off = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off"), catalog=catalog(NV))
            self.assertEqual(off.setup_args()[off.setup_args().index("--vision") + 1], "no")

    def test_an_arc_b_series_card_is_allowed_with_pictures_on_the_cpu_reader_and_one_plain_line(self):
        """Owner decision 2026-10-22: Arc B-series (Battlemage) runs like any Arc. An Arc Pro B70 (32 GB) gets pictures on, with the CPU reader (the GPU
        reader has not run on a 24 GB+ Arc), and a warn-level note that it is new and how to report a problem. An A-series card gets no such note."""
        from strata_cli.launch import cmd
        for name, vram, note in (("Intel Arc Pro B70", 32.0, True), ("Intel Arc B580", 12.0, True), ("Intel Arc A750", 8.0, False)):
            hw = {"ram_gb": 64.0, "os": "Linux", "cpu": {"avx2": True, "name": "x"},
                  "gpus": [{"index": 0, "name": name, "vendor": "intel", "vram_gb": vram, "usable": True}]}
            with Home():
                from strata_cli.launch.catalog import Computer
                cat = catalog(hw)
                comp = Computer(ram_gb=64.0, gpus=hw["gpus"], disk_free_gb=500.0, os="Linux", avx2=True, raw=hw, engine_vendor="intel")
                with mock.patch.object(type(cat), "reader_state", lambda self: "present"), mock.patch.object(type(cat), "computer", lambda self, fresh=False: comp):
                    self.assertEqual(cat.picture_default(True), "cpu", name)
                self.assertEqual(bool(comp.new_card_note), note, name)
                if note:
                    text, nxt = comp.new_card_note
                    self.assertIn("new to Strata", text)
                    self.assertIn("--vision off", nxt)
                    self.assertIn("github.com/Niko1221/Strata/issues", nxt)

    def test_a_mac_defaults_pictures_to_the_metal_reader_whatever_its_memory(self):
        for gb in (16.0, 32.0, 128.0):
            hw = {"ram_gb": gb, "os": "Darwin", "cpu": {"avx2": None, "name": "Apple M4"},
                  "mac": {"chip": "Apple M4", "macos": "27.2", "unified_gb": gb},
                  "gpus": [{"index": 0, "name": "Apple M4", "vendor": "apple", "vram_gb": 0.0, "unified_gb": gb, "usable": True}]}
            with Home():
                cat = catalog(hw)
                with mock.patch.object(type(cat), "reader_state", lambda self: "present"):
                    self.assertTrue(cat.is_mac())
                    self.assertEqual(cat.picture_default(True), "gpu", gb)
                    r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=cat)
                    with mock.patch.object(sys, "platform", "darwin"):
                        args = r.setup_args()
                    self.assertEqual(args[args.index("--vision") + 1], "gpu", gb)
                with mock.patch.object(type(cat), "reader_state", lambda self: "missing"):
                    self.assertEqual(cat.picture_default(True), "off")

    def test_a_model_that_cannot_read_pictures_stays_off_without_a_question(self):
        import dataclasses
        with Home():
            cat = catalog(HW_12)
            real = cat.rows()
            rows = [dataclasses.replace(r, images=False) for r in real]
            with mock.patch.object(type(cat), "rows", lambda self, *a, **k: rows):
                r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768), catalog=cat)
                self.assertEqual(r.resolved_vision(), "off")

    def test_an_engine_without_the_reader_defaults_to_off(self):
        with Home():
            cat = catalog(HW_12)
            with mock.patch.object(type(cat), "reader_state", lambda self: "missing"):
                self.assertEqual(cat.picture_default(True), "off")

    def test_the_command_defaults_pictures_on_and_off_still_turns_them_off(self):
        from strata_cli.launch.cmd import apply_defaults
        with Home():
            cat = catalog(HW_12)
            s, _ = apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS"), cat)
            self.assertEqual(s.vision, "cpu")                                 # a 12 GB card: pictures ON, read on the CPU
            s, _ = apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="gpu"), cat)
            self.assertEqual(s.vision, "gpu")                                 # what was asked for is kept
            s, _ = apply_defaults(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="off"), cat)
            self.assertEqual(s.vision, "off")

    def test_the_apps_own_installer_arguments(self):
        import tempfile
        own = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, own, True)
        a = self.runner(data_dir="/d", model_path=own, gpus=[1])
        self.assertEqual(a[:8], ["--yes", "--no-start", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32768"])
        for pair in (["--vision", "no"], ["--kv", "q4_0"], ["--experimental-speed-projection", "off"], ["--data-dir", "/d"], ["--gguf-dir", own],
                     ["--gpu", "1"]):
            self.assertIn(pair[0], a)
            self.assertEqual(a[a.index(pair[0]) + 1], pair[1])
        self.assertIn("--set-default-data-dir", a)             # a folder the person chose is the remembered one (the app and the next start use it)

    def test_a_mac_gets_only_the_options_its_setup_knows(self):
        with mock.patch.object(sys, "platform", "darwin"):
            a = self.runner(data_dir="/d", gpus=[0])
        self.assertEqual(a, ["--yes", "--no-start", "--family", "qwen", "--model", "IQ3_XXS", "--context", "32768", "--vision", "no", "--data-dir", "/d",
                             "--set-default-data-dir"])


class AdoptRules(unittest.TestCase):
    def files(self, root: Path, shards=2, pack=True, tokenizer=True, mtp=True):
        d = root / "Strata-data"
        m = d / "models" / "IQ3_XXS"
        m.mkdir(parents=True)
        for i in range(1, shards + 1):
            (m / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{i}-of-0000{shards}.gguf").write_bytes(b"GGUF")
        p = d / "packs" / "iq3_xxs"
        p.mkdir(parents=True)
        if pack:
            (p / "experts.bin").write_bytes(b"x")
        if tokenizer:
            (p / "tokenizer").mkdir()
        if mtp:
            (d / "mtp" / "rt").mkdir(parents=True)
        return d, m

    def test_setup_detection(self):
        from strata_cli.launch import adopt
        self.assertTrue(adopt.setup_cannot_handle_this_card("no NVIDIA GPU found (nvidia-smi did not answer)"))
        self.assertTrue(adopt.setup_cannot_handle_this_card("no AMD GPU Strata can use"))
        self.assertFalse(adopt.setup_cannot_handle_this_card("not enough free disk space in /x"))

    def test_find_shards_needs_the_whole_set(self):
        from strata_cli.launch import adopt
        with Home() as h:
            d, m = self.files(h.base)
            self.assertEqual(len(adopt.find_shards(m)), 2)
            (m / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf").unlink()
            self.assertEqual(adopt.find_shards(m), [])

    def test_plan_writes_the_settings_setup_would(self):
        from strata_cli.launch import adopt
        with Home() as h:
            d, m = self.files(h.base)
            src = Path(__file__).resolve().parents[2]
            cfg, problems = adopt.plan(family="qwen", model="IQ3_XXS", model_dir=m, data_dir=d, src=src, context=32768, kv=None,
                                       model_name="qwen3.8-flash-next-iq3_xxs", family_def={}, log=h.base / "x.log", backend="sycl")
            self.assertEqual(problems, [])
            a = cfg["args"]
            self.assertEqual(a[a.index("--max-context") + 1], "32768")
            self.assertEqual(a[a.index("--kv") + 1], "int8")
            self.assertIn("--ple-gguf", a)
            self.assertIn("--mtp", a)
            self.assertEqual((cfg["exe"], cfg["backend"]), ("component:strata-engine", "sycl"))
            self.assertTrue(cfg["tokenizer"].endswith("tokenizer"))

    def test_plan_intel_flags_follow_the_driver(self):
        from unittest import mock
        from strata_cli.launch import adopt
        for driver, want, other in (("i915", ["--ple-io", "ram"], "--stream-experts"), ("xe", ["--stream-experts"], "--ple-io")):
            with Home() as h, mock.patch.object(adopt, "intel_driver", return_value=driver):
                d, m = self.files(h.base)
                for ctx, reserve in ((32768, "1024"), (65536, "2048")):
                    cfg, problems = adopt.plan(family="qwen", model="IQ3_XXS", model_dir=m, data_dir=d, src=Path(__file__).resolve().parents[2],
                                               context=ctx, kv=None, model_name="x", family_def={}, log=h.base / "x.log", backend="sycl")
                    self.assertEqual(problems, [])
                    a = cfg["args"]
                    for w in want:
                        self.assertIn(w, a)
                    self.assertNotIn(other, a)
                    self.assertEqual(a[a.index("--vram-reserve-mib") + 1], reserve)

    def test_intel_arc_is_read_from_the_kernel_and_gets_a_small_context(self):
        from strata_cli.launch import catalog as C
        with Home() as h:
            card = h.base / "drm" / "card0" / "device"
            card.mkdir(parents=True)
            (card / "vendor").write_text("0x8086")
            (card / "device").write_text("0x56a1")
            cards = C.intel_arc_cards(str(h.base / "drm"))
            self.assertEqual([(c["name"], c["vram_gb"]) for c in cards], [("Intel Arc A750", 8.0)])
            self.assertEqual(C.intel_arc_cards(str(h.base / "nothing")), [])
            hw = dict(HW_12)
            hw["gpus"] = cards
            cat = catalog(hw)
            self.assertEqual(cat.computer().line().split(" 8 GB")[0], "Arc A750".replace("Arc", "Intel Arc"))
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", hw), 8192)
            hw["gpus"] = [{"vendor": "intel", "name": "Intel Arc Pro B70", "vram_gb": 32.0, "usable": True}]
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", hw), 32768)

    def test_plan_names_what_is_missing(self):
        from strata_cli.launch import adopt
        with Home() as h:
            d, m = self.files(h.base, pack=False, tokenizer=False)
            cfg, problems = adopt.plan(family="qwen", model="IQ3_XXS", model_dir=m, data_dir=d, src=Path(__file__).resolve().parents[2], context=8192,
                                       kv=None, model_name="x", family_def={}, log=h.base / "x.log", backend=None)
            self.assertIsNone(cfg)
            self.assertTrue(any("prepared pack" in p for p in problems))
            self.assertTrue(any("tokenizer" in p for p in problems))
            cfg, problems = adopt.plan(family="unsloth", model="UD-Q4_K_XL", model_dir=m, data_dir=d, src=Path("."), context=8192, kv=None, model_name="x",
                                       family_def={}, log=h.base / "x.log", backend=None)
            self.assertIn("only for the Qwen3.8-Flash-Next", problems[0])
            self.assertNotIn("..", problems[0] + ".")                           # round 8, N11: the caller adds the full stop

    def test_pictures_on_is_setups_yes_and_setup_picks_the_encoder(self):
        with Home() as h:
            cat = catalog(HW_12)
            for vision, want in (("on", "yes"), ("off", "no"), ("gpu", "gpu"), ("cpu", "cpu")):
                r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision=vision), catalog=cat)
                self.assertEqual(r.vision_word(), want, vision)
            argv = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="on"), catalog=cat).setup_args()
            self.assertEqual(argv[argv.index("--vision") + 1], "yes")

    def test_setup_control_mode_keeps_pictures_when_the_engine_carries_the_reader(self):
        from types import SimpleNamespace
        with Home() as h:
            mod = catalog(HW_12).strata.setup_module()
            if mod is None or not hasattr(mod, "control_engine_step"):
                self.skipTest("no setup.py in this checkout")
            engine = h.base / "engine"
            engine.mkdir()
            found = SimpleNamespace(root=engine, version="0.1.45", backend="cuda", platform="linux-x86_64")
            with mock.patch.object(mod, "get_llama_cpp", return_value=engine), mock.patch("strata_control.engine_component.resolve", return_value=found):
                self.assertEqual(mod.control_engine_step({}, "cpu")[5], "none")           # no strata-vision beside the engine: off, with a note
                (engine / mod.VEXE).write_text("x")
                self.assertEqual(mod.control_engine_step({}, "cpu")[5], "cpu")            # the component carries it: the choice is kept
                self.assertEqual(mod.control_engine_step({}, "none")[5], "none")

    def test_turning_pictures_on_for_an_installed_model_downloads_only_the_reader(self):
        with Home() as h:
            cat = catalog(HW_12)
            cfgfile = h.base / "strata-x.json"
            cfgfile.write_text("{}")
            for vision, images, ctx, want in (("on", "off", 32768, 0.9), ("on", "cpu", 32768, 0.0), ("off", "off", 32768, 0.0), ("off", "off", 65536, 0.0)):
                r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision=vision, context=ctx), catalog=cat)
                desc = {"context": 32768, "kv": "int8", "images": images}
                with mock.patch.object(r, "installed_state", return_value=(cfgfile, desc)):
                    self.assertAlmostEqual(r.download_gb(), want, msg=(vision, images, ctx))

    def test_a_missing_picture_reader_file_is_fetched_again(self):
        from strata_cli.launch.runner import files_present
        with Home() as h:
            mm = h.base / "mmproj-x.gguf"
            cfg = {"args": [], "vision": {"mmproj": str(mm)}}
            self.assertEqual(files_present(cfg), [str(mm)])
            mm.write_text("x")
            self.assertEqual(files_present(cfg), [])
            mm.unlink()
            cat = catalog(HW_12)
            cfgfile = h.base / "strata-x.json"
            cfgfile.write_text(json.dumps(cfg))
            r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="on", context=32768), catalog=cat)
            with mock.patch.object(r, "installed_state", return_value=(cfgfile, {"context": 32768, "kv": "int8", "images": "cpu"})):
                self.assertAlmostEqual(r.download_gb(), 0.9)

    def test_the_cards_of_this_launch_go_into_its_config_without_reinstalling(self):
        from tests.terminal.launch_support import HW_2X16
        from strata_cli.launch.runner import apply_cards, cards_for_launch
        from strata_control import tune_knobs
        base = {"exe": "component:strata-engine", "args": ["--max-context", "32768"], "gpu": 0, "port": 8080}
        with Home() as h:
            # an installed single-card config, then the split chosen now
            cfg = derive_config(base, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS, gpus=[0, 1])
            self.assertEqual((cfg["gpu"], cfg["layer_split"]), ([0, 1], "auto"))
            self.assertTrue(cfg["gpus_asked"])
            # an installed split config, then one named card now
            split_base = {**base, "gpu": [0, 1], "layer_split": "auto"}
            cfg = derive_config(split_base, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS, gpus=[1])
            self.assertEqual(cfg["gpu"], 1)
            self.assertNotIn("layer_split", cfg)
            # nothing chosen and nothing to recommend: the installed choice stays
            cfg = derive_config(split_base, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS)
            self.assertEqual(cfg["gpu"], [0, 1])
            self.assertEqual(base["gpu"], 0)                                   # the installed config itself is never changed
            aged = derive_config({**base, "backend": "hip", "hip_ordinal": 1}, S.LaunchSettings(), bool_flags=tune_knobs.BOOL_FLAGS, gpus=[0])
            self.assertNotIn("hip_ordinal", aged)                              # a card named now is not overruled by an old config's ordinal
            # which cards: named ones win (one card too: --gpu 1 on a box that has them), else the fit rule on a box with several cards
            two = catalog(HW_2X16)
            self.assertEqual(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpus=[1, 0]), two), [1, 0])
            self.assertEqual(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS"), two), [0, 1])     # roles: card 0 Main (last), card 1 Helper
            one = catalog(HW_12)
            self.assertEqual(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpus=[1]), one), [1])
            self.assertIsNone(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS"), one))
            # the launch's own config is written with them (Runner.build_config), the installed file is untouched
            import json
            installed = h.base / "strata-iq3_xxs.json"
            installed.write_text(json.dumps(base))
            r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", name="cards", gpus=[1]), catalog=two)
            out = r.build_config(installed)
            self.assertEqual(json.loads(out.read_text())["gpu"], 1)
            self.assertEqual(json.loads(installed.read_text())["gpu"], 0)

    def test_a_card_the_installed_model_was_set_to_is_not_replaced_by_the_automatic_pick(self):
        """Two equal cards: the fit rule recommends card 0, so a person's earlier choice (setup --gpu 1, the app's card pick: "gpus_asked")
        must survive a `strata launch` that names no cards - in its order, with "gpu_order": "as_given" too."""
        import json
        from tests.terminal.launch_support import HW_2X16
        from strata_cli.launch.runner import cards_for_launch, installed_cards
        with Home() as h:
            two = catalog(HW_2X16)
            plain = S.LaunchSettings(family="qwen", model="IQ3_XXS")
            chosen = {"exe": "component:strata-engine", "args": ["--max-context", "32768"], "gpu": 1, "gpus_asked": True, "port": 8080}
            self.assertEqual(installed_cards(chosen), [1])
            self.assertEqual(installed_cards({**chosen, "gpu": [1, 0], "gpu_order": "as_given"}), [1, 0])
            self.assertEqual(installed_cards({**chosen, "gpu": "1,0"}), [1, 0])
            self.assertIsNone(installed_cards({**chosen, "gpus_asked": False}))               # never asked: setup's default, not a choice
            self.assertIsNone(installed_cards({"gpu": 1}))
            self.assertIsNone(cards_for_launch(plain, two, base=chosen))                      # the installed choice stays
            self.assertEqual(cards_for_launch(plain, two, base={**chosen, "gpus_asked": False}), [0, 1])    # no choice recorded: the recommended roles
            self.assertEqual(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpus=[0]), two, base=chosen), [0])   # --gpu 0 wins
            if True:
                installed = h.base / "strata-iq3_xxs.json"
                for gpu, extra in ((1, {}), ([1, 0], {"gpu_order": "as_given", "layer_split": "auto"})):
                    base = {**chosen, "gpu": gpu, **extra}
                    installed.write_text(json.dumps(base))
                    r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", name="keep"), catalog=two)
                    cfg = json.loads(r.build_config(installed).read_text())
                    self.assertEqual(cfg["gpu"], gpu)
                    self.assertEqual(cfg.get("gpu_order"), extra.get("gpu_order"))

    def test_pictures_asked_but_still_off_is_said_in_plain_words(self):
        with Home() as h:
            cat = catalog(HW_12)
            seen = []
            r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", vision="on"), catalog=cat, emit=seen.append)
            path = h.base / "cfg.json"
            path.write_text("{}")
            with mock.patch.object(cat.strata, "describe_config", return_value={"images": "off"}):
                r.check_pictures(path)
            self.assertTrue(any("Pictures could not be switched on" in e.text and e.level == "warn" for e in seen))
            seen.clear()
            with mock.patch.object(cat.strata, "describe_config", return_value={"images": "cpu"}):
                r.check_pictures(path)
            self.assertEqual(seen, [])

    def test_tuning_that_found_the_standard_settings_fastest_is_not_run_again(self):
        # "already fastest" saves no tuning preset, only tuned_at: the next start (a container restart with TUNE=quick) must not measure for 20 minutes again
        with Home():
            seen = []
            s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, tune="quick", tuned_at="2026-10-20")
            self.assertTrue(s.is_tuned)
            r = Runner(s, catalog=catalog(HW_12), emit=seen.append)
            with mock.patch.object(r, "_spawn", side_effect=AssertionError("tuned again")):
                r.tune()
            self.assertTrue(any("Already tuned on 2026-10-20" in e.text for e in seen))
            self.assertFalse(S.LaunchSettings(tune="quick").is_tuned)
            store.save(s, "Fastest")
            self.assertTrue(store.summary(store.load("Fastest"))["tuned"])

    def test_runner_adopts_when_setup_cannot_handle_the_card(self):
        with Home() as h:
            d, m = self.files(h.base)
            cat = catalog(HW_12)
            s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, data_dir=str(d))
            r = Runner(s, catalog=cat)
            path = r.adopt_prepared_files("no NVIDIA GPU found (nvidia-smi did not answer)")
            self.assertTrue(path.is_file())
            self.assertTrue(r.notices and "wrote the run settings itself" in r.notices[0])
            self.assertIsNone(r.adopt_prepared_files("not enough free disk space"))
            from strata_cli.launch.runner import LaunchError
            r2 = Runner(S.LaunchSettings(family="qwen", model="IQ3_S", context=32768, data_dir=str(d)), catalog=cat)
            with self.assertRaises(LaunchError) as e:
                r2.adopt_prepared_files("no NVIDIA GPU found")
            self.assertIn("cannot set up this graphics card yet", e.exception.text)


class AccessRules(unittest.TestCase):
    def test_hosts_and_keys(self):
        self.assertEqual(access.host_for("local"), "127.0.0.1")
        self.assertEqual(access.host_for("cloudflare"), "127.0.0.1")
        self.assertEqual(access.host_for("lan"), "0.0.0.0")
        self.assertTrue(access.new_key().startswith("sk-strata-"))
        self.assertNotIn(access.new_key(), access.mask_key(access.new_key()))
        self.assertEqual(access.mask_key(None), "off")

    def test_private_ranges(self):
        for ip in ("10.1.2.3", "192.168.88.21", "172.16.0.9", "100.101.1.1"):
            self.assertTrue(access._is_private(ip), ip)
        for ip in ("8.8.8.8", "172.32.0.1", "1.2.3.4"):
            self.assertFalse(access._is_private(ip), ip)

    def test_missing_helpers_say_how_to_install(self):
        with mock.patch.object(access, "find_cloudflared", return_value=None):
            msg = access.missing_helper("cloudflare")
            self.assertIn("cloudflared is not installed", msg)
            self.assertIn("quick tunnel", msg)
        with mock.patch.object(access, "find_tailscale", return_value=None):
            self.assertIn("Tailscale is not installed", access.missing_helper("tailscale"))
        with mock.patch.object(access, "tailscale_status", return_value={"installed": True, "running": False, "ip": None, "name": None}):
            self.assertIn("tailscale up", access.missing_helper("tailscale"))
        self.assertIsNone(access.missing_helper("local"))

    def test_options_and_urls(self):
        with mock.patch.object(access, "lan_ips", return_value=["192.168.1.5"]), \
                mock.patch.object(access, "find_cloudflared", return_value="/bin/cloudflared"), \
                mock.patch.object(access, "tailscale_status", return_value={"installed": True, "running": True, "ip": "100.64.0.1", "name": "pc.tail.ts.net"}):
            opts = {o.kind: o for o in access.options(8080)}
            self.assertEqual(opts["lan"].url, "http://192.168.1.5:8080")
            self.assertTrue(opts["cloudflare"].ready)
            self.assertEqual(opts["tailscale"].url, "http://pc.tail.ts.net:8080")
            urls = access.urls_for(S.LaunchSettings(access="lan", port=8080))
            self.assertEqual(urls, ["http://127.0.0.1:8080", "http://192.168.1.5:8080"])
            self.assertEqual(access.urls_for(S.LaunchSettings(access="cloudflare", port=8080), tunnel_url="https://a.trycloudflare.com")[0],
                             "https://a.trycloudflare.com")

    def test_the_windows_firewall_question_is_announced_before_it_happens(self):
        with mock.patch.object(access, "WINDOWS", True):
            self.assertIn("choose Allow for private networks", access.firewall_note("lan"))
            self.assertIn("still works on this computer", access.firewall_note("tailscale"))
            self.assertIsNone(access.firewall_note("local"))
            self.assertIsNone(access.firewall_note("cloudflare"))
        with mock.patch.object(access, "WINDOWS", False):
            self.assertIsNone(access.firewall_note("lan"))

    def test_helper_output_filter_drops_login_lines(self):
        self.assertEqual(access.filter_line("Please open this URL to login: https://dash.cloudflare.com/argotunnel?token=abc"), "")
        self.assertEqual(access.filter_line("Your quick Tunnel has been created: https://x.trycloudflare.com"),
                         "Your quick Tunnel has been created: https://x.trycloudflare.com")
        self.assertEqual(access.TUNNEL_URL.search("| https://calm-river-1234.trycloudflare.com |").group(0),
                         "https://calm-river-1234.trycloudflare.com")


class FlagsCommand(unittest.TestCase):
    """Every `strata launch` flag, run through the real parser with the computer faked and nothing installed."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err or out)
        return json.loads(out)["plan"]

    def test_help_lists_everything_with_examples(self):
        code, out, err = run_cli(["launch", "--help"])
        self.assertEqual(code, 0)
        for flag in ("--preset", "--model", "--family", "--model-path", "--data-dir", "--context", "--kv", "--vision", "--parallel", "--gpu",
                     "--tune", "--tune-allow-small", "--set", "--access", "--port", "--api-key", "--api-key-env", "--new-key", "--save-preset",
                     "--no-tui", "--plan", "--feed", "--list-presets", "--delete-preset", "--status", "--stop", "--show-key", "examples:"):
            self.assertIn(flag, out)
        code, out, _ = run_cli(["--help"])
        self.assertIn("launch", out)
        self.assertIn("connect", out)

    def test_mtp_flag_is_listed_planned_and_copyable(self):
        code, out, err = run_cli(["launch", "--help"])
        self.assertIn("--mtp", out)
        p = self.plan("--model", "q2_0", "--context", "32k", "--vision", "off", "--mtp", "off")
        self.assertEqual(p["settings"]["mtp"], "off")
        self.assertIn("--mtp off", p["flags"])
        code, out, err = run_cli(["launch", "--plan", "--mtp", "maybe"])
        self.assertNotEqual(code, 0)

    def test_vision_help_says_the_picture_rule(self):
        code, out, err = run_cli(["launch", "--help"])
        text = " ".join(out.split())
        self.assertIn("CPU reader under 24 GB", text)
        self.assertIn("GPU reader from 24 GB", text)
        self.assertIn("off turns pictures off", text)
        self.assertNotIn("for NVIDIA and AMD", text)

    def test_model_context_parallel_vision_tune_access_port(self):
        p = self.plan("--model", "iq3_xxs", "--context", "128k", "--parallel", "2", "--vision", "off", "--tune", "quick", "--access", "lan",
                      "--port", "9090")
        s = p["settings"]
        self.assertEqual((s["family"], s["model"], s["context"], s["parallel"], s["vision"], s["tune"], s["access"], s["port"]),
                         ("qwen", "IQ3_XXS", 131072, 2, "off", "quick", "lan", 9090))
        self.assertEqual(s["api_key"], "set")                               # shared beyond this computer: a key was made
        self.assertTrue(any("2 requests" in w and "12 GB" in w for w in p["warnings"]))

    def test_defaults_come_from_the_recommendation(self):
        p = self.plan()
        s = p["settings"]
        self.assertEqual((s["family"], s["model"], s["context"], s["access"], s["tune"]), ("qwen", "IQ3_S", 32768, "local", "skip"))
        self.assertTrue(any("picked" in w for w in p["warnings"]))

    def test_family_and_kv_and_gpu_and_data_dir_and_set(self):
        p = self.plan("--model", "iq3_xxs", "--family", "swift", "--kv", "q4_0", "--gpu", "0,1", "--data-dir", str(self.home.base / "data-x"),
                      "--set", "prefill=65536", "--set", "STRATA_PIN_BY_NEED=1", "--tune-allow-small")
        s = p["settings"]
        self.assertEqual((s["family"], s["kv"], s["gpus"], s["data_dir"], s["tune_allow_small"]), ("swift", "q4_0", [0, 1], str(self.home.base / "data-x"), True))
        self.assertEqual(s["extra"], {"--prefill": "65536", "STRATA_PIN_BY_NEED": "1"})
        self.assertIn("--set prefill=65536", p["flags"])

    def test_pool_cores_is_an_experimental_advanced_setting(self):
        code, out, err = run_cli(["launch", "--help"])
        self.assertIn("--pool-cores", out)
        p = self.plan("--model", "q2_0", "--pool-cores", "2,4, 6,8")
        self.assertEqual(p["settings"]["extra"], {"--pool-cores": "2,4,6,8"})
        self.assertIn("--set pool-cores=2,4,6,8", p["flags"])                # the copyable line says the same
        p = self.plan("--model", "q2_0", "--pool-cores", "2-14:2")
        self.assertEqual(p["settings"]["extra"], {"--pool-cores": "2-14:2"})
        p = self.plan("--model", "q2_0")                                    # off by default: nothing is added
        self.assertNotIn("--pool-cores", p["settings"]["extra"])
        p = self.plan("--model", "q2_0", "--pool-cores", "two,four")        # a list that cannot be read: a warning, never a refusal
        self.assertNotIn("--pool-cores", p["settings"]["extra"])
        self.assertTrue(any("pool-cores" in w for w in p["warnings"]))

    def test_dspark_on_a_mac_is_warned_about_and_not_refused(self):
        # the drafter has no Metal kernels yet: --dspark on is passed on as asked, with a plain warning (recommend, never force)
        import sys
        from unittest import mock
        with mock.patch.object(sys, "platform", "darwin"):
            p = self.plan("--model", "q2_0", "--dspark", "on")
        self.assertEqual(p["settings"]["extra"].get("--dspark"), "on")
        self.assertTrue(any("DSpark" in w and "Mac" in w for w in p["warnings"]), p["warnings"])
        p = self.plan("--model", "q2_0", "--dspark", "auto")
        self.assertFalse(any("DSpark" in w and "Mac" in w for w in p["warnings"]))

    def test_dspark_takes_a_word_and_the_word_reaches_the_engine_line(self):
        # --set dspark=on used to start strata-dsv41 with a lone "--dspark" ("--dspark needs a value", #3959)
        from strata_cli.launch import settings as S
        from strata_control import tune_knobs
        base = ["--pack", "p", "--max-context", "262144"]
        for word in ("on", "off", "auto", "none"):
            a, _ = S.as_argv_changes({"--dspark": word}, tune_knobs.BOOL_FLAGS)
            self.assertEqual(a, {"--dspark": word})
            out = tune_knobs.apply_to_argv(base + ["--dspark", "auto"], a)
            self.assertEqual(out[out.index("--dspark") + 1], word, word)
            self.assertEqual(out.count("--dspark"), 1)
        a, _ = S.as_argv_changes({"--dspark": "ON", "--dspark-conf": "0.7"}, tune_knobs.BOOL_FLAGS)
        self.assertEqual(a, {"--dspark": "on", "--dspark-conf": "0.7"})
        self.assertEqual(S.as_argv_changes({"--dspark": ""}, tune_knobs.BOOL_FLAGS)[0], {})
        code, out, _err = run_cli(["launch", "--help"])
        self.assertIn("--dspark", out)
        p = self.plan("--model", "q2_0", "--dspark", "on")
        self.assertEqual(p["settings"]["extra"], {"--dspark": "on"})
        self.assertIn("--set dspark=on", p["flags"])

    def test_api_key_flags(self):
        p = self.plan("--model", "q2_0", "--access", "lan", "--api-key", "sk-mine")
        self.assertEqual(p["settings"]["api_key"], "set")
        self.assertFalse(any("new API key was made" in w for w in p["warnings"]))
        with mock.patch.dict(os.environ, {"MY_KEY": "sk-env"}):
            p = self.plan("--model", "q2_0", "--api-key-env", "MY_KEY")
            self.assertEqual(p["settings"]["api_key"], "set")
        code, out, err = run_cli(["launch", "--plan", "--api-key-env", "NOT_SET_ANYWHERE"])
        self.assertEqual(code, 2)
        self.assertIn("NOT_SET_ANYWHERE", err)
        p = self.plan("--model", "q2_0", "--new-key")
        self.assertEqual(p["settings"]["api_key"], "set")

    def test_no_api_key_flag(self):
        p = self.plan("--model", "q2_0", "--access", "lan", "--no-api-key")
        self.assertIsNone(p["settings"]["api_key"])
        self.assertFalse(any("new API key was made" in w for w in p["warnings"]))
        self.assertTrue(any("anyone who can reach this port can use the model and its tools" in w for w in p["warnings"]))
        p = self.plan("--model", "q2_0", "--access", "cloudflare", "--no-api-key")
        self.assertTrue(any("public address" in w and "anyone on the internet" in w for w in p["warnings"]))
        p = self.plan("--model", "q2_0", "--access", "local", "--no-api-key")
        self.assertEqual([w for w in p["warnings"] if "API key" in w], [])
        # scripts that name LAN without a key flag keep getting a key, as before
        p = self.plan("--model", "q2_0", "--access", "lan", "--yes")
        self.assertEqual(p["settings"]["api_key"], "set")
        self.assertTrue(any("new API key was made" in w for w in p["warnings"]))
        # a key given as well wins over --no-api-key, with a note
        p = self.plan("--model", "q2_0", "--access", "lan", "--api-key", "sk-a", "--no-api-key")
        self.assertEqual(p["settings"]["api_key"], "set")

    def test_model_path_is_used_instead_of_a_catalog_model(self):
        import tempfile
        from tests.terminal.test_own_files import IQ3XXS, gguf_bytes
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "IQ3_XXS"
            folder.mkdir()
            for n in IQ3XXS:
                (folder / n).write_bytes(gguf_bytes(["IQ3_XXS"]))
            s = self.plan("--model-path", str(folder))["settings"]
        self.assertEqual(s["model_path"], str(folder))
        self.assertEqual((s["family"], s["model"]), ("qwen", "IQ3_XXS"))      # the files say which model they are

    def test_no_choices_and_no_terminal_does_not_start_anything(self):
        with mock.patch.dict(os.environ, {"CI": "1"}), mock.patch.object(Runner, "run", side_effect=AssertionError("must not run")):
            code, out, err = run_cli(["launch"])
        self.assertEqual(code, 2)
        self.assertIn("not a terminal", err)
        self.assertIn("--preset NAME", err)

    def test_git_bash_without_a_tty_goes_on_in_plain_questions(self):
        fake = Result("launch-x", ["http://127.0.0.1:8080"], None, None, [], "/x")
        with mock.patch.dict(os.environ, {"MSYSTEM": "MINGW64"}), mock.patch("builtins.input", return_value="y"),                 mock.patch.object(Runner, "run", return_value=fake) as run:
            code, out, err = run_cli(["launch"])
        self.assertEqual(code, 0, err)
        run.assert_called_once()
        self.assertIn("cannot show Strata's screens", out + err)
        self.assertIn("winpty strata launch", out + err)

    def test_git_bash_answering_no_changes_nothing(self):
        with mock.patch.dict(os.environ, {"MSYSTEM": "MINGW64"}), mock.patch("builtins.input", return_value="n"),                 mock.patch.object(Runner, "run", side_effect=AssertionError("must not run")):
            code, out, err = run_cli(["launch"])
        self.assertEqual(code, 0, err)
        self.assertIn("Nothing was changed", out + err)

    def test_a_pipe_with_nobody_to_answer_still_says_what_to_name(self):
        with mock.patch.dict(os.environ, {"MSYSTEM": "MINGW64"}), mock.patch("builtins.input", side_effect=EOFError),                 mock.patch.object(Runner, "run", side_effect=AssertionError("must not run")):
            code, out, err = run_cli(["launch"])
        self.assertEqual(code, 2)
        self.assertIn("--preset NAME", err)

    def test_unknown_model_names_the_closest_one(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3xxs"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'IQ3_XXS'", err)
        self.assertNotIn("Traceback", err)

    def test_unknown_flag_names_the_closest_one(self):
        code, out, err = run_cli(["launch", "--modle", "iq3_xxs"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean '--model'", err)

    def test_bad_values_are_plain_sentences(self):
        for argv, word in ((["--tune", "quik"], "quick"), (["--access", "lann"], "lan"), (["--context", "huge"], "context"),
                           (["--port", "99999"], "1 to 65535"), (["--parallel", "12"], "1 to 8"), (["--vision", "maybe"], "off")):
            code, out, err = run_cli(["launch", "--plan", *argv])
            self.assertEqual(code, 2, argv)
            self.assertIn(word, err, argv)

    def test_preset_with_flags_on_top(self):
        store.save(S.LaunchSettings(name="Office", family="qwen", model="IQ3_XXS", context=65536, parallel=2, access="local"))
        p = self.plan("--preset", "Office", "--context", "128k")
        s = p["settings"]
        self.assertEqual((s["name"], s["context"], s["parallel"]), ("Office", 131072, 2))
        code, out, err = run_cli(["launch", "--plan", "--preset", "Offce"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'Office'", err)

    def test_vision_is_turned_off_for_a_model_without_pictures(self):
        # every catalog model reads pictures now (Gyro and NVFP4-QAD too): a row that does not stands in for a future one
        import dataclasses
        real_rows = Catalog.rows

        def rows(self, *args, **kwargs):
            return [dataclasses.replace(r, images=False) if r.model == "Gyro-S" else r for r in real_rows(self, *args, **kwargs)]

        with mock.patch.object(Catalog, "rows", rows):
            p = self.plan("--model", "gyro-s", "--vision", "on")
        self.assertEqual(p["settings"]["vision"], "off")
        self.assertTrue(any("cannot read pictures" in w for w in p["warnings"]))

    def test_does_not_fit_is_a_warning_with_the_reason(self):
        p = self.plan("--model", "nvfp4")
        self.assertTrue(any("does not fit" in w for w in p["warnings"]))

    def test_human_plan_changes_nothing(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "q2_0", "--save-preset", "X"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing was changed.", out)
        self.assertEqual(store.list_presets(), [])

    def test_the_access_help_names_the_flags_for_showing_and_setting_a_key(self):             # #3713 #3951
        from strata_cli.launch import screens
        self.assertIn("strata launch --show-key", screens.AccessScreen.help_text)
        self.assertIn("--api-key YOUR-KEY", screens.AccessScreen.help_text)

    def test_the_summary_says_where_the_engine_log_is(self):                                  # #3656
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "x.json"
            cfg.write_text(json.dumps({"log": str(Path(d) / "office.engine.log"), "args": []}), encoding="utf-8")
            fake = Result(profile="launch-office", urls=["http://127.0.0.1:8080"], key=None, tune_word=None, notices=[], config=str(cfg))
            with mock.patch.object(Runner, "run", return_value=fake):
                code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0"])
        self.assertEqual(code, 0, err)
        self.assertIn("Engine log", out)
        self.assertIn("office.engine.log", out)
        self.assertIn("strata logs --follow", out)

    def test_run_without_screens_prints_the_address_and_saves_the_preset(self):
        fake = Result(profile="launch-office", urls=["http://127.0.0.1:8080", "http://192.168.1.5:8080"], key="sk-strata-abcdefghijkl",
                      tune_word=None, notices=[], config="/x")
        with mock.patch.object(Runner, "run", return_value=fake):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--access", "lan", "--save-preset", "Office"])
        self.assertEqual(code, 0, err)
        self.assertIn("Strata is serving.", out)
        self.assertIn("http://127.0.0.1:8080/v1", out)
        self.assertIn("http://192.168.1.5:8080/v1", out)
        self.assertNotIn("sk-strata-abcdefghijkl", out)                      # the key is masked
        self.assertIn("strata connect", out)
        self.assertEqual([p.name for p in store.list_presets()], ["Office"])
        code, out, err = run_cli(["launch", "--list-presets"])
        self.assertIn("Office", out)
        code, out, _ = run_cli(["launch", "--show-key", "--preset", "Office"])
        self.assertEqual(code, 0)                                              # the preset's own key
        self.assertTrue(out.strip().startswith("sk-strata-"))

    def test_a_setup_line_joined_after_the_model_could_not_be_installed_starts_with_a_capital(self):
        from strata_cli.launch.runner import setup_failure
        e = setup_failure([], "the model does not answer sensibly after the preparation: no answer", "", None, False)
        self.assertTrue(e.text.startswith("The model could not be installed. The model does not answer sensibly"), e.text)

    def test_a_reason_joined_after_a_sentence_starts_with_a_capital(self):
        from strata_cli.launch.runner import capital
        self.assertEqual(capital("the update feed could not be read: no route"), "The update feed could not be read: no route")
        self.assertEqual(capital("Already fine."), "Already fine.")
        self.assertEqual(capital(""), "")

    def test_launch_error_is_one_plain_sentence(self):
        from strata_cli.launch.runner import LaunchError
        with mock.patch.object(Runner, "run", side_effect=LaunchError("The engine could not be installed.", "Check your internet.", 6)):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0"])
        self.assertEqual(code, 6)
        self.assertIn("The engine could not be installed. Check your internet.", err)
        self.assertNotIn("Traceback", err)

    def test_list_delete_status_stop_without_anything_running(self):
        store.save(S.LaunchSettings(name="A", family="qwen", model="Q2_0", context=32768))
        code, out, _ = run_cli(["launch", "--list-presets"])
        self.assertIn("A", out)
        code, out, _ = run_cli(["launch", "--delete-preset", "A"])
        self.assertEqual(code, 0)
        self.assertIn("Models and tuning results were not touched", out)
        code, out, _ = run_cli(["launch", "--list-presets"])
        self.assertIn("No saved presets", out)
        code, out, _ = run_cli(["launch", "--status"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing from `strata launch` is running.", out)
        code, out, _ = run_cli(["launch", "--stop"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing", out)
        code, out, err = run_cli(["launch", "--delete-preset", "gone"])
        self.assertEqual(code, 2)
        self.assertIn("no saved preset", err)

    def test_feed_flag_reaches_the_runner(self):
        seen = {}

        def fake(self, *a, **k):
            seen["feed"] = self.feed
            return Result("launch-x", ["http://127.0.0.1:8080"], None, None, [], "/x")

        with mock.patch.object(Runner, "run", fake):
            code, _, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--feed", "http://127.0.0.1:9/feed"])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen["feed"], "http://127.0.0.1:9/feed")

    def test_output_json_for_scripts(self):
        fake = Result("launch-x", ["http://127.0.0.1:8080"], "sk-k", "faster", ["a note"], "/x")
        with mock.patch.object(Runner, "run", return_value=fake):
            code, out, err = run_cli(["launch", "--no-tui", "--output", "json", "--model", "q2_0"])
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["urls"], ["http://127.0.0.1:8080"])
        self.assertEqual(data["api_key"], "set")                                # never the key itself
        self.assertEqual(data["tuned"], "faster")


class ConnectCommand(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_list_and_unknown_names(self):
        code, out, _ = run_cli(["connect"])
        self.assertEqual(code, 0)
        for h in ("opencode", "codex", "claude-code", "omp", "pi", "hermes", "openclaw", "deepseek-harness"):
            self.assertIn(h, out)
        code, out, err = run_cli(["connect", "opncode"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'opencode'", err)
        code, out, _ = run_cli(["connect", "--output", "json"])
        self.assertEqual(len(json.loads(out)["harnesses"]), 8)

    def test_prints_the_config_with_a_masked_key(self):
        code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8123", "--key", "sk-strata-abcdefghij1234", "--model", "m1"])
        self.assertEqual(code, 0, err)
        self.assertIn("http://127.0.0.1:8123/v1", out)
        self.assertIn("opencode.json", out)
        self.assertIn("limit", out)
        self.assertNotIn("abcdefghij1234", out)
        code, out, _ = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8123", "--key", "sk-strata-abcdefghij1234", "--show-key"])
        self.assertIn("sk-strata-abcdefghij1234", out)

    def test_claude_code_uses_the_anthropic_variables_without_a_login(self):
        code, out, _ = run_cli(["connect", "claude-code", "--url", "http://127.0.0.1:8080", "--key", "sk-x1234567890abc", "--show-key"])
        self.assertIn("ANTHROPIC_BASE_URL", out)
        self.assertIn("ANTHROPIC_AUTH_TOKEN", out)
        self.assertNotIn("/v1", out.split("ANTHROPIC_BASE_URL")[1].splitlines()[0])      # the Anthropic client adds /v1 itself

    def test_a_pasted_v1_address_is_cut_back_to_the_server_root(self):
        """The Anthropic client adds /v1/messages itself, so ANTHROPIC_BASE_URL must never end in /v1 (that gave /v1/v1/messages, a 404);
        the OpenAI-style apps get exactly one /v1."""
        code, out, _ = run_cli(["connect", "claude-code", "--url", "http://127.0.0.1:8080/v1/", "--key", "sk-x1234567890abc", "--show-key", "--output", "json"])
        data = json.loads(out)
        self.assertEqual(data["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertIn('"ANTHROPIC_BASE_URL": "http://127.0.0.1:8080"', data["files"][0]["text"])
        code, out, _ = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8080/v1", "--output", "json"])
        text = json.loads(out)["files"][0]["text"]
        self.assertIn("http://127.0.0.1:8080/v1", text)
        self.assertNotIn("/v1/v1", text)
        code, out, _ = run_cli(["connect", "claude-code", "--url", "http://127.0.0.1:8080"])
        self.assertIn("without /v1", out)

    def test_write_keeps_a_backup_and_merges(self):
        target = Path(os.environ["HOME"]) / ".config" / "opencode" / "opencode.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"theme": "dark", "provider": {"other": {"a": 1}}}))
        code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8080", "--key", "sk-x1234567890abc", "--write", "--yes"])
        self.assertEqual(code, 0, err)
        data = json.loads(target.read_text())
        self.assertEqual(data["theme"], "dark")
        self.assertIn("other", data["provider"])
        self.assertIn("strata", data["provider"])
        backups = list(target.parent.glob("opencode.json.strata-bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text())["theme"], "dark")
        self.assertIn("backup", out)

    def test_write_asks_first_without_yes(self):
        # no keyboard: stdin is a pipe whatever the shell that runs the tests is; input() must never be reached
        pipe = io.StringIO("")
        with mock.patch.object(sys, "stdin", pipe), mock.patch("builtins.input", side_effect=AssertionError("asked on a pipe")):
            code, out, err = run_cli(["connect", "opencode", "--write"])
        self.assertEqual(code, 2)
        self.assertIn("asks first", err)

    def test_write_answered_by_nobody_writes_nothing(self):
        # a terminal whose input ends (Ctrl-D, a closed console) is "no", not a traceback
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch("builtins.input", side_effect=EOFError):
            code, out, err = run_cli(["connect", "opencode", "--write"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing was written", out)

    def test_built_in_harness_has_nothing_to_write(self):
        code, out, _ = run_cli(["connect", "deepseek-harness", "--write", "--yes"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing to write", out)

    def render(self, name, **kw):
        h = harness.find(name)
        t = harness.target(kw.get("url", "http://127.0.0.1:18900"), kw.get("key"), "qwen3.8-flash-next-q2_0", kw.get("context", 32768))
        return harness.render(h, t)

    def test_all_eight_are_tested_configs(self):
        self.assertEqual([h.id for h in harness.HARNESSES],
                         ["opencode", "codex", "claude-code", "omp", "pi", "hermes", "openclaw", "deepseek-harness"])
        for h in harness.HARNESSES:
            self.assertTrue(h.tested, h.id)
            self.assertIn("tested", h.status)

    def test_opencode_has_limit_input_and_the_15k_compaction_buffer(self):
        cfg = json.loads(self.render("opencode")["files"][0]["text"])
        m = cfg["provider"]["strata"]["models"]["qwen3.8-flash-next-q2_0"]
        self.assertEqual(m["limit"], {"context": 32768, "input": 28672, "output": 4096})
        self.assertEqual(cfg["compaction"]["buffer"], 13672)                  # 28672 - 15000
        self.assertEqual(cfg["provider"]["strata"]["options"]["baseURL"], "http://127.0.0.1:18900/v1")
        self.assertEqual(cfg["provider"]["strata"]["options"]["apiKey"], "strata-local")

    def test_codex_uses_responses_and_the_key_variable(self):
        out = self.render("codex", key="sk-real")
        text = out["files"][0]["text"]
        self.assertIn('wire_api = "responses"', text)
        self.assertIn("model_auto_compact_token_limit = 15000", text)
        self.assertIn('base_url = "http://127.0.0.1:18900/v1"', text)
        self.assertEqual(out["env"], {"STRATA_API_KEY": "sk-real"})

    def test_claude_code_sets_window_and_output_cap_and_has_no_login(self):
        out = self.render("claude-code")
        env = out["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:18900")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "32768")
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "4096")
        self.assertEqual(env["CLAUDE_CODE_SIMPLE"], "1")                      # the full prompt is 15.6K tokens
        self.assertEqual(json.loads(out["files"][0]["text"])["env"], env)
        big = self.render("claude-code", context=131072)["env"]
        self.assertEqual(big["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "131072")
        self.assertNotIn("CLAUDE_CODE_SIMPLE", big)
        self.assertTrue(any("CLAUDE_CODE_MAX_OUTPUT_TOKENS" in n for n in out["limits"]))

    def test_pi_omp_hermes_openclaw_numbers(self):
        pi = self.render("pi")["files"]
        self.assertEqual(json.loads(pi[1]["text"])["compaction"]["reserveTokens"], 17768)       # 32768 - 15000
        self.assertEqual(json.loads(pi[0]["text"])["providers"]["strata"]["models"][0]["contextWindow"], 32768)
        omp = self.render("omp")["files"]
        self.assertIn("thresholdTokens: 15000", omp[1]["text"])
        self.assertIn("contextWindow: 32768", omp[0]["text"])
        hermes = self.render("hermes")
        self.assertIn("provider: lmstudio", hermes["files"][0]["text"])
        self.assertIn("context_length: 65536", hermes["files"][0]["text"])
        self.assertTrue(any("WARNING" in n and "64K" in n for n in hermes["limits"]))
        self.assertFalse(any("WARNING" in n for n in self.render("hermes", context=131072)["limits"]))
        oc = json.loads(self.render("openclaw")["files"][0]["text"])
        self.assertEqual(oc["models"]["providers"]["strata"]["models"][0]["contextTokens"], 23192)
        self.assertTrue(any("Node 24.16" in n for n in self.render("openclaw")["limits"]))

    def test_limits_are_in_plain_words_for_every_app(self):
        for h in harness.HARNESSES:
            self.assertTrue(self.render(h.id)["limits"], h.id)

    def test_deepseek_harness_is_the_app(self):
        out = self.render("deepseek-harness")
        self.assertEqual(out["files"], [])
        self.assertIn("Strata app", " ".join(out["limits"]))

    def test_check_sends_the_apps_own_wire_format_and_reports_pass_or_fail(self):
        from strata_cli.launch import check
        seen = []

        def good(url, key, body, headers=None, timeout=0):
            seen.append(url)
            if url.endswith("/v1/responses"):
                if "tools" in body:
                    return {"output": [{"type": "function_call", "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}]}
                return {"output_text": "pong"}
            if url.endswith("/v1/messages"):
                if "tools" in body:
                    return {"content": [{"type": "tool_use", "name": "get_weather", "input": {"city": "Paris"}}]}
                return {"content": [{"type": "text", "text": "pong"}]}
            if "tools" in body:
                return {"choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}}]}
            return {"choices": [{"message": {"content": "Pong."}}]}

        for h, ending in (("opencode", "/v1/chat/completions"), ("codex", "/v1/responses"), ("claude-code", "/v1/messages")):
            seen.clear()
            res = check.run_checks(h, "http://127.0.0.1:1", "k", "m", poster=good)
            self.assertEqual([r.ok for r in res], [True, True], h)
            self.assertTrue(all(u.endswith(ending) for u in seen), (h, seen))

        def wrong(url, key, body, headers=None, timeout=0):
            if "tools" in body:
                return {"choices": [{"message": {"content": "It is sunny.", "tool_calls": None}}]}
            return {"choices": [{"message": {"content": "hello there"}}]}

        res = check.run_checks("pi", "http://127.0.0.1:1", "k", "m", poster=wrong)
        self.assertEqual([r.ok for r in res], [False, False])
        self.assertIn("not the word asked for", res[0].detail)
        self.assertIn("did not call the tool", res[1].detail)

        import urllib.error

        def refused(url, key, body, headers=None, timeout=0):
            raise urllib.error.HTTPError(url, 401, "no", None, None)
        res = check.run_checks("pi", "http://127.0.0.1:1", "", "m", poster=refused)
        self.assertIn("refused the key", res[0].detail)

    def test_connect_check_flag_prints_pass_lines(self):
        from strata_cli.launch import check
        ok = [check.Check("prompt", True, "It answered: 'pong'.", 1.0), check.Check("tool call", True, "It called get_weather with {}.", 2.0)]
        with mock.patch.object(check, "run_checks", return_value=ok):
            code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:1", "--check"])
        self.assertEqual(code, 0, err)
        self.assertIn("PASS prompt", out)
        self.assertIn("PASS tool call", out)
        bad = [check.Check("prompt", False, "x", 1.0), ok[1]]
        with mock.patch.object(check, "run_checks", return_value=bad):
            code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:1", "--check"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL prompt", out)

    def test_context_flag_reaches_the_config(self):
        code, out, _ = run_cli(["connect", "pi", "--url", "http://127.0.0.1:8080", "--context", "64k", "--show-key"])
        self.assertIn('"contextWindow": 65536', out)
        self.assertIn('"reserveTokens": 33394', out)             # 65536 - 32142: the tested 52.3% share, not a flat 15K

    def test_the_compaction_point_scales_with_the_window(self):
        """A wider server must not be summarised every 15,000 tokens: the share the tested 32K configs used carries over."""
        wide = harness.target("http://127.0.0.1:18900", None, "qwen3.8-flash-next-q2_0", 262144)
        self.assertEqual(harness.compact_at(wide), 135000)                     # 52.3% of the 258048 input limit
        cfg = json.loads(self.render("opencode", context=262144)["files"][0]["text"])
        self.assertEqual(wide.input_limit - cfg["compaction"]["buffer"], 135000)
        self.assertIn("model_auto_compact_token_limit = 135000", self.render("codex", context=262144)["files"][0]["text"])
        self.assertIn("thresholdTokens: 135000", self.render("omp", context=262144)["files"][1]["text"])
        # the window every config was tested on keeps exactly the value it was tested at
        tested = harness.target("http://127.0.0.1:18900", None, "qwen3.8-flash-next-q2_0", 32768)
        self.assertEqual(harness.compact_at(tested), 15000)


class GpuRolesFlag(unittest.TestCase):
    """--gpu-roles: a role for each card (strata_control/gpu_roles.py); unnamed cards are Off."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_2X16, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def test_roles_reach_the_settings_and_the_plan_says_who_does_what(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "iq3_xxs", "--gpu-roles", "1=main,0=helper"])
        self.assertEqual(code, 0, err or out)
        s = json.loads(out)["plan"]["settings"]
        self.assertEqual(s["gpu_roles"], {"1": "main", "0": "helper"})
        self.assertIsNone(s["gpus"])
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3_xxs", "--gpu-roles", "1=main,0=helper"])
        self.assertEqual(code, 0, err or out)
        self.assertIn("cards     Main: GPU 2 RTX A4000. Helper: GPU 1 RTX A4000.", out)
        self.assertIn("--gpu-roles 0=helper,1=main", out)                 # the copyable line says the same

    def test_one_card_named_leaves_the_other_off(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3_xxs", "--gpu-roles", "1=main"])
        self.assertEqual(code, 0, err or out)
        self.assertIn("Main: GPU 2 RTX A4000. Off: GPU 1 RTX A4000.", out)

    def test_bad_roles_are_plain_sentences(self):
        for argv, word in ((["--gpu-roles", "1=boss"], "main, helper, off"), (["--gpu-roles", "0=main,1=main"], "only one card can be Main"),
                           (["--gpu-roles", "1"], "number=role"), (["--gpu", "0", "--gpu-roles", "1=main"], "not both")):
            code, out, err = run_cli(["launch", "--plan", *argv])
            self.assertEqual(code, 2, argv)
            self.assertIn(word, err, argv)
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3_xxs", "--gpu-roles", "7=main"])
        self.assertIn("There is no card 7", out + err)

    def test_the_launch_config_gets_the_roles_and_the_mirror_with_main_last(self):
        from strata_cli.launch.runner import cards_for_launch
        two = catalog(HW_2X16)
        mirror = cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpu_roles={"1": "main", "0": "helper"}), two)
        self.assertEqual((mirror["gpu"], mirror["gpu_order"]), ([0, 1], "as_given"))
        self.assertEqual(mirror["gpu_roles"], {"0": "helper", "1": "main"})
        base = {"exe": "x", "args": [], "gpu": 1, "gpus_asked": True, "hip_ordinal": 0, "gpu_roles": {"0": "off", "1": "main"}}
        cfg = derive_config(base, S.LaunchSettings(family="qwen", model="IQ3_XXS"), bool_flags=frozenset(), gpus=mirror)
        self.assertEqual((cfg["gpu"], cfg["layer_split"], cfg["gpu_order"]), ([0, 1], "auto", "as_given"))
        self.assertNotIn("hip_ordinal", cfg)
        one = cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpu_roles={"0": "main"}), two)
        cfg = derive_config(base, S.LaunchSettings(family="qwen", model="IQ3_XXS"), bool_flags=frozenset(), gpus=one)
        self.assertEqual((cfg["gpu"], cfg["gpu_roles"], "layer_split" in cfg), (0, {"0": "main", "1": "off"}, False))
        none = cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS", gpu_roles={"0": "off", "1": "off"}), two)
        self.assertEqual((none["gpu"], none["gpu_none"]), ([], True))

    def test_a_plain_card_list_replaces_an_earlier_roles_choice_and_the_installed_roles_are_kept(self):
        from strata_cli.launch.runner import apply_cards, cards_for_launch, installed_cards
        cfg = {"gpu": 1, "gpu_roles": {"0": "off", "1": "main"}, "gpu_cards": {}, "gpus_asked": True}
        apply_cards(cfg, [0, 1])
        self.assertEqual((cfg["gpu"], "gpu_roles" in cfg), ([0, 1], False))
        saved = {"gpu": [0, 1], "gpu_order": "as_given", "gpu_roles": {"0": "helper", "1": "main"}}
        self.assertEqual(installed_cards(saved), [0, 1])
        self.assertIsNone(cards_for_launch(S.LaunchSettings(family="qwen", model="IQ3_XXS"), catalog(HW_2X16), base=saved))


if __name__ == "__main__":
    unittest.main()
