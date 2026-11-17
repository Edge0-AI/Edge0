"""The preset store: save, name, apply on load, stale detection, undo."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from strata_control import presets, tune_knobs

CFG = {"exe": "strata", "model_name": "swift-iq2_xs", "ui_resource_profile": "work", "vram_reserve_mib": 1536,
       "args": ["--pack", "P", "--native", "Swift-IQ3_S-00001-of-00002.gguf", "--expert-cache", "auto", "--prefill", "auto",
                "--spec", "4", "--max-context", "131072", "--kv", "int8"]}


def identity(driver="610.47", sha="aaa", epoch=0, gpu="RTX 5070", ram=64):
    return {"hardware": {"hard": {"gpus": [{"name": gpu, "vram_mib": 12227}], "cpu": "cpu", "logical": 12, "physical": 6,
                                  "hybrid": False, "ram_gib": ram, "pcie": None, "os": "Windows"},
                         "soft": {"driver": [driver], "os_release": "11"}},
            "engine": {"version": "0.1.45", "backend": "cuda", "sha256": "e1", "tuning_epoch": epoch},
            "model": {"sha256_shard1": sha, "bytes": 1}}


RESULT = {"outcome": "improved", "seconds": 1100, "settings": {"args": {"--host-core": "last"}, "env": {}, "per_request": {}},
          "answers": {"class": "exact", "fields": []}, "base": {"decode_tok_s": 60.0, "pp_tok_s": 1500.0},
          "chosen": {"decode_tok_s": 65.0, "pp_tok_s": 1500.0, "rss_peak_mib": 12000, "vram_used_mib": 9000},
          "gain_pct": 8.3, "pairs": 2, "faster_pairs": 2, "noise": {"aa_gap": 0.004}}


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["STRATA_PRESET_DIR"] = str(Path(self.tmp.name) / "presets")
        self.addCleanup(lambda: os.environ.pop("STRATA_PRESET_DIR", None))
        self.addCleanup(self.tmp.cleanup)
        self.choices = presets.choices_from_config(CFG)

    def make(self, name=None, ident=None, result=None, choices=None):
        return presets.create(result or RESULT, choices=choices or self.choices, identity=ident or identity(), name=name)

    def test_default_name_is_plain_words(self):
        self.assertEqual(self.choices["quant"], "IQ3_S")
        self.assertEqual(presets.default_name(self.choices), "IQ3_S 128K – Work")
        full = dict(self.choices, goal="speed")
        self.assertEqual(presets.default_name(full), "IQ3_S 128K – Full speed")

    def test_create_list_rename_delete(self):
        p = self.make()
        self.assertTrue(p["name"].startswith("IQ3_S 128K"))
        self.assertEqual(len(presets.list_presets()), 1)
        self.assertTrue(presets.list_presets()[0]["active"])
        q = presets.rename(p["id"], "My fast one")
        self.assertEqual(q["name"], "My fast one")
        with self.assertRaises(presets.PresetError):
            presets.rename(p["id"], "")
        p2 = self.make(name="My fast one")                   # a duplicate name gets a number, never overwrites
        self.assertEqual(p2["name"], "My fast one (2)")
        presets.delete("My fast one (2)")
        self.assertEqual([x["name"] for x in presets.list_presets()], ["My fast one"])

    def test_activation_keeps_the_previous_one_and_undo_restores_it(self):
        a = self.make(name="A")
        b = self.make(name="B")
        self.assertEqual(presets.find("B")["id"], b["id"])
        slot = a["slot"]
        self.assertEqual(presets._index()[slot]["active"], b["id"])
        self.assertEqual(presets.load(b["id"])["previous"], a["id"])
        restored = presets.undo(slot)
        self.assertEqual(restored["id"], a["id"])
        self.assertEqual(presets._index()[slot]["active"], a["id"])
        self.assertIsNone(presets.undo(slot))                 # undo once more: back to the standard setup
        self.assertIsNone(presets._index()[slot]["active"])
        with self.assertRaises(presets.PresetError):
            presets.undo(slot)

    def test_unsafe_settings_are_refused(self):
        bad = copy.deepcopy(RESULT)
        bad["settings"]["args"] = {"--expert-cache": 0}
        with self.assertRaises(presets.PresetError):
            self.make(result=bad)

    def test_a_value_outside_the_tested_range_is_kept_with_a_warning(self):
        # #3844: --spec 5 is not a value the tuner tries (its safe values are 3 and 4), but nobody measured it as bad on every
        # card; a user who chose it keeps it, and is told once. The never list (--spec 6, --expert-cache 0 ...) still refuses.
        res = copy.deepcopy(RESULT)
        res["settings"]["args"] = {"--spec": 5}
        self.make(result=res)                                               # saved, not refused
        out, info = presets.overlay(CFG, now=identity())
        self.assertTrue(info["applied"], info)
        self.assertEqual(tune_knobs._current(out["args"], "--spec"), "5")
        self.assertEqual(info["warnings"], ["--spec 5: not tried by the tuner, kept because it is your setting"])
        self.assertNotIn("why_not", info)
        self.assertEqual(tune_knobs._current(CFG["args"], "--spec"), "4")   # the input is not modified
        # an in-range preset has no warnings
        presets.delete(presets.list_presets()[0]["id"])
        self.make()
        _, info2 = presets.overlay(CFG, now=identity())
        self.assertTrue(info2["applied"])
        self.assertNotIn("warnings", info2)

    def test_the_never_list_still_refuses_but_a_range_does_not(self):
        ctx = tune_knobs.Context()
        never, bounds = tune_knobs.check_setting_parts({"--spec": 5, "--pcie-frac": 0.6}, {}, ctx)
        self.assertEqual(never, [])
        self.assertEqual(len(bounds), 2)
        never, bounds = tune_knobs.check_setting_parts({"--spec": 6}, {}, ctx)
        self.assertTrue(never and bounds)                                   # slower (never list) and outside the values
        # the tuner's own check is the two lists together, in the same order as before
        self.assertEqual(tune_knobs.check_setting({"--spec": 6}, {}, ctx), never + bounds)
        bad = copy.deepcopy(RESULT)
        bad["settings"]["args"] = {"--spec": 6}
        with self.assertRaises(presets.PresetError):
            self.make(result=bad)

    def test_overlay_applies_the_active_preset_without_measuring(self):
        self.make()
        out, info = presets.overlay(CFG, now=identity())
        self.assertTrue(info["applied"])
        self.assertEqual(info["state"], "fresh")
        self.assertEqual(out["args"][-2:], ["--host-core", "last"])
        self.assertEqual(out["args"][:len(CFG["args"])], CFG["args"])          # the user's own flags are untouched
        self.assertNotIn("--host-core", CFG["args"])                           # the input is not modified
        self.assertIn("identical", info["notice"])

    def test_overlay_other_choices_get_no_preset(self):
        self.make()
        other = copy.deepcopy(CFG)
        other["args"][other["args"].index("--max-context") + 1] = "32768"
        out, info = presets.overlay(other, now=identity())
        self.assertFalse(info["applied"])
        self.assertEqual(out, other)
        full = copy.deepcopy(CFG)
        full["ui_resource_profile"] = "performance"
        self.assertFalse(presets.overlay(full, now=identity())[1]["applied"])

    def test_hard_change_does_not_apply_and_says_why(self):
        self.make()
        for kw, word in ((dict(gpu="RTX 4090"), "graphics card"), (dict(sha="zzz"), "model file")):
            out, info = presets.overlay(CFG, now=identity(**kw))
            if "model file" in word:
                self.assertFalse(info["applied"]) if info.get("state") else None
            self.assertFalse(info["applied"], kw)

    def test_soft_change_applies_and_asks_for_a_recheck(self):
        self.make()
        out, info = presets.overlay(CFG, now=identity(driver="620.1"))
        self.assertTrue(info["applied"])
        self.assertEqual(info["state"], "soft")
        self.assertTrue(any("driver" in r for r in info["reasons"]))
        out, info = presets.overlay(CFG, now=identity(epoch=1))
        self.assertEqual(info["state"], "soft")

    def test_staleness_rules(self):
        p = self.make()
        self.assertEqual(presets.staleness(p, identity())["state"], "fresh")
        self.assertEqual(presets.staleness(p, identity(ram=32))["state"], "hard")
        self.assertEqual(presets.staleness(p, identity(ram=62))["state"], "fresh")     # within 10%
        old = dict(p, created_epoch=time.time() - 100 * 86400)
        self.assertEqual(presets.staleness(old, identity())["state"], "soft")

    def test_per_request_defaults_reach_the_sampling_block(self):
        res = copy.deepcopy(RESULT)
        res["settings"]["per_request"] = {"spec_min_p": 0.7}
        self.make(result=res)
        out, info = presets.overlay(CFG, now=identity())
        self.assertEqual(out["sampling"]["strata_tune"], {"spec_min_p": 0.7})

    def test_a_broken_store_never_breaks_a_start(self):
        self.make()
        for f in Path(os.environ["STRATA_PRESET_DIR"]).glob("*.json"):
            f.write_text("{not json")
        out, info = presets.overlay(CFG, now=identity())
        self.assertEqual(out, CFG)
        self.assertFalse(info["applied"])

    def test_export_import_roundtrip(self):
        p = self.make(name="Shared")
        data = presets.export_preset(p["id"])
        presets.delete(p["id"])
        q = presets.import_preset(data)
        self.assertEqual(q["settings"], p["settings"])
        self.assertNotEqual(q["id"], p["id"])
        data["settings"]["args"] = {"--spec": 6}
        with self.assertRaises(presets.PresetError):
            presets.import_preset(data)
        data["settings"]["args"] = {"--spec": 5}                            # #3844: only outside the tuner's range: kept
        self.assertEqual(presets.import_preset(data)["settings"]["args"], {"--spec": 5})

    def test_recheck_keep_refreshes_the_identity(self):
        p = self.make()
        out = presets.record_recheck(p["id"], {"verdict": "keep", "identity": identity(driver="620.1"),
                                              "measured": {"decode_tok_s": 70.0}})
        self.assertEqual(presets.staleness(out, identity(driver="620.1"))["state"], "fresh")
        self.assertEqual(out["measured"]["decode_tok_s"], 70.0)      # better by more than 3%: stored
        self.assertIsNotNone(out["recheck"]["last_ok"])

    def test_trial_history_blocks_a_candidate_that_did_not_start(self):
        presets.record_trials([{"label": "host_core_last", "args": ["--x"], "env": {}, "status": "OOM", "at": 1}],
                              hw_hash="h", model_digest="m", engine_sha="e")
        self.assertEqual(presets.bad_signatures(hw_hash="h", model_digest="m", engine_sha="e"), {"host_core_last"})
        self.assertEqual(presets.bad_signatures(hw_hash="h2", model_digest="m", engine_sha="e"), set())
        self.assertEqual(presets.bad_signatures(hw_hash="h", model_digest="m", engine_sha="e2"), set())

    def test_a_command_that_passed_is_not_switched_off_by_a_later_failure(self):
        # #3502: the plain base command passed four times, then a busy card wrote OOM rows
        for status in ("PASS", "OOM"):
            presets.record_trials([{"label": "base", "args": ["--x"], "env": {}, "status": status, "at": 1}],
                                  hw_hash="h", model_digest="m", engine_sha="e")
        self.assertEqual(presets.bad_signatures(hw_hash="h", model_digest="m", engine_sha="e"), set())


if __name__ == "__main__":
    unittest.main()
