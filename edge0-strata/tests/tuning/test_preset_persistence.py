"""A saved tuning is applied at every start: after an app restart (a new process), after an unload and a load, and it can be seen
(`presets effective`, `presets status`) and switched off and on again."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cli(env, *args):
    out = subprocess.run([sys.executable, str(ROOT / "strata.py"), *args, "--output", "json"], capture_output=True, text=True,
                         env=env, cwd=str(ROOT), timeout=120)
    return json.loads(out.stdout[out.stdout.index("{"):]), out


def fresh_process(env, code):
    """A new Python process: what an app restart or a new server start is."""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        (t / "model.gguf").write_bytes(b"gguf" * 1000)
        (t / "engine.exe").write_bytes(b"x")
        (t / "tok").mkdir()
        self.cfg_path = t / "strata-model.json"
        self.cfg = {"exe": str(t / "engine.exe"), "tokenizer": str(t / "tok"), "model_name": "iq3_s-test", "ui_resource_profile": "work",
                    "args": ["--native", str(t / "model.gguf"), "--expert-cache", "auto", "--spec", "4", "--max-context", "65536",
                             "--kv", "int8", "--kv-resident", "32768"]}
        self.cfg_path.write_text(json.dumps(self.cfg))
        self.env = dict(os.environ, STRATA_PRESET_DIR=str(t / "presets"), PYTHONPATH=str(ROOT), STRATA_NO_PRESET="")
        self.env.pop("STRATA_NO_PRESET")
        # the tune's result, saved as a preset by the library, the way `strata tune auto` does
        sys.path.insert(0, str(ROOT))
        os.environ["STRATA_PRESET_DIR"] = str(t / "presets")
        self.addCleanup(lambda: os.environ.pop("STRATA_PRESET_DIR", None))
        from strata_control import presets
        self.presets = presets
        ident = presets.current_identity(self.cfg, exe=self.cfg["exe"])
        result = {"outcome": "improved", "seconds": 900, "settings": {"args": {"--kv-resident": 65536}, "env": {}, "per_request": {}},
                  "answers": {"class": "exact", "fields": []}, "base": {"decode_tok_s": 60.0}, "chosen": {"decode_tok_s": 64.0},
                  "gain_pct": 6.6, "pairs": 2, "faster_pairs": 2, "noise": {}}
        self.preset = presets.create(result, choices=presets.choices_from_config(self.cfg), identity=ident, name="IQ3_S 64K – Work")

    CODE = ("import json,sys;sys.path.insert(0,'.');from serve import server;"
            "c=json.load(open(%r));out=server.apply_saved_tuning(c,say=lambda l:None);"
            "print(json.dumps(out['args']))")

    def args_at_start(self):
        return fresh_process(self.env, self.CODE % str(self.cfg_path))

    def test_applied_after_restart_and_after_unload_and_load(self):
        for _ in range(3):                  # three starts = a restart, an unload, a load: every one is a new server process
            args = self.args_at_start()
            self.assertEqual(args[args.index("--kv-resident") + 1], "65536")
            self.assertEqual(args[args.index("--spec") + 1], "4")            # the user's own flags are untouched
        self.assertEqual(json.loads(self.cfg_path.read_text())["args"], self.cfg["args"])    # the saved run config itself never changes

    def test_a_tune_worker_measures_the_configuration_as_given(self):
        env = dict(self.env, STRATA_NO_PRESET="1")
        args = fresh_process(env, self.CODE % str(self.cfg_path))
        self.assertEqual(args[args.index("--kv-resident") + 1], "32768")

    def test_expert_mode_can_read_what_will_run(self):
        out, _ = cli(self.env, "presets", "effective", str(self.cfg_path))
        self.assertTrue(out["applied"])
        self.assertEqual(out["preset"]["name"], "IQ3_S 64K – Work")
        self.assertEqual(out["changed"], [{"flag": "--kv-resident", "was": "32768", "now": "65536"}])
        self.assertEqual(out["args"][out["args"].index("--kv-resident") + 1], "65536")

    def test_which_tuning_is_in_use_and_switching_it_off_and_on(self):
        st, _ = cli(self.env, "presets", "status", str(self.cfg_path))
        self.assertEqual(st["active"]["name"], "IQ3_S 64K – Work")
        self.assertFalse(st["switched_off"])
        off, _ = cli(self.env, "presets", "off", str(self.cfg_path))
        self.assertIsNone(off["active"])
        args = self.args_at_start()
        self.assertEqual(args[args.index("--kv-resident") + 1], "32768")        # back to the standard settings
        eff, _ = cli(self.env, "presets", "effective", str(self.cfg_path))
        self.assertFalse(eff["applied"])
        st, _ = cli(self.env, "presets", "status", str(self.cfg_path))
        self.assertIsNone(st["active"])
        self.assertTrue(st["switched_off"])
        self.assertEqual(st["last_active"]["name"], "IQ3_S 64K – Work")          # kept, and listed, so it can be switched on again
        on, _ = cli(self.env, "presets", "apply", st["last_active"]["id"])
        self.assertTrue(on["ok"])
        args = self.args_at_start()
        self.assertEqual(args[args.index("--kv-resident") + 1], "65536")

    def test_a_changed_gpu_is_not_applied_and_says_why(self):
        p = self.presets.load(self.preset["id"])
        p["identity"]["hardware"]["hard"]["gpus"] = [{"name": "Another GPU", "vram_mib": 1}]
        self.presets._write_json(self.presets._path(p["id"]), p)
        out, _ = cli(self.env, "presets", "effective", str(self.cfg_path))
        self.assertFalse(out["applied"])
        self.assertIn("graphics card", out["why_not"])


class OutcomeCodesTest(unittest.TestCase):
    def test_codes_and_values(self):
        from strata_control import autotune
        mk = autotune.TuneResult
        r = mk(outcome="improved", gain_pct=7.2, seconds=1200, waited_s=120, pairs=5, faster_pairs=4)
        r.base, r.chosen = {"decode_tok_s": 50.0}, {"decode_tok_s": 53.6}
        autotune.Tuner.finalize(r)
        self.assertEqual(r.code, "faster")
        self.assertEqual(r.values["gain_pct"], 7.2)
        self.assertEqual(r.values["minutes"], 18.0)
        for outcome, reason, code in (("default_confirmed", "", "already_fastest"), ("inconclusive", "noisy", "busy_pc"),
                                      ("inconclusive", "drift", "busy_pc"), ("inconclusive", "not_tried", "busy_pc"),
                                      ("base_failed", "start_failed", "failed"), ("cancelled", "", "stopped")):
            x = mk(outcome=outcome, reason_code=reason)
            autotune.Tuner.finalize(x)
            self.assertEqual((x.code, x.reason_code), (code, reason), outcome)


if __name__ == "__main__":
    unittest.main()
