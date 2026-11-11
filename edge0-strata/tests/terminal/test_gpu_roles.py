"""strata_control/gpu_roles.py: Main / Helper / Off for every graphics card (the rules the app, `strata launch`, setup and the server share)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from strata_control import gpu_roles as R

ROOT = Path(__file__).resolve().parents[2]
V = json.loads((ROOT / "tests" / "data" / "gpu_roles_vectors.json").read_text(encoding="utf-8"))


def roles(d):
    return None if d is None else {int(k): v for k, v in d.items()}


class Vectors(unittest.TestCase):
    """tests/data/gpu_roles_vectors.json is also run by apps/host/tests/gpu-roles.test.mjs against gpu-roles.ts."""

    def test_recommend(self):
        for c in V["recommend"]:
            with self.subTest(c["hw"]):
                r = R.recommend(V["hw"][c["hw"]])
                self.assertEqual(r["roles"], roles(c["roles"]))
                self.assertEqual((r["main"], r["why"]), (c["main"], c["why"]))

    def test_set_role(self):
        for c in V["set_role"]:
            with self.subTest(c):
                out, note = R.set_role(roles(c["roles"]), V["hw"][c["hw"]], c["card"], c["role"])
                self.assertEqual(out, roles(c["after"]))
                self.assertEqual(note, c["note"])

    def test_order(self):
        for c in V["order"]:
            with self.subTest(c):
                self.assertEqual(R.order(roles(c["roles"]), V["hw"][c["hw"]]), c["gpu"])

    def test_warnings(self):
        for c in V["warnings"]:
            with self.subTest(c):
                w = R.warnings(roles(c["roles"]), V["hw"][c["hw"]], spill_gb=c.get("spill_gb", 0.0))
                self.assertEqual([x["code"] for x in w], c["codes"])
                if c.get("fix"):
                    self.assertEqual(w[0]["fix"], c["fix"])

    def test_saved(self):
        for c in V["saved"]:
            with self.subTest(c):
                got, problem = R.saved_roles(c["cfg"])
                self.assertEqual(got, roles(c["roles"]))
                if c.get("problem"):
                    self.assertIn(c["problem"], problem or "")

    def test_legacy(self):
        for c in V["legacy"]:
            with self.subTest(c):
                self.assertEqual(R.from_legacy(c["gpu"], V["hw"][c["hw"]], c.get("gpu_order")), roles(c["roles"]))


class Words(unittest.TestCase):
    def test_warnings_are_one_short_line_and_say_only_what_a_person_cannot_know(self):
        two = V["hw"]["two"]
        # a slower card as Main is NOT warned about: people understand it (owner rule)
        self.assertEqual(R.warnings({0: "main", 1: "helper"}, two), [])
        for case in V["warnings"]:
            for w in R.warnings(roles(case["roles"]), V["hw"][case["hw"]], spill_gb=case.get("spill_gb", 0.0)):
                self.assertLess(len(w["text"]), 90, w["text"])
                self.assertNotIn("slower", w["text"].replace("is slow as Main", ""))

    def test_cpu_only_line(self):
        w = R.warnings({0: "off", 1: "off"}, V["hw"]["two"], engine_runs_cpu_only=True)
        self.assertEqual(w[0]["text"], "Performance is degraded on CPU only.")
        w = R.warnings({0: "off", 1: "off"}, V["hw"]["two"])
        self.assertEqual(w[0]["text"], R.NO_CARD_LINE)

    def test_sentence_and_describe(self):
        self.assertEqual(R.sentence(V["hw"]["one"]), "Strata will use your RTX 4070 Ti.")
        s = R.sentence(V["hw"]["three"])
        self.assertIn("RTX 4070 Ti is Main: it is faster.", s)
        self.assertIn("RTX 2060 starts Off: 8 GB or less.", s)
        self.assertEqual(R.describe({0: "helper", 1: "main"}, V["hw"]["two"]), "Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.")

    def test_cards_with_8_gb_or_less_start_off_but_a_lone_8_gb_card_is_main(self):
        self.assertEqual(R.recommend([{"index": 0, "name": "RTX 4060", "vram_gb": 8}])["roles"], {0: "main"})
        r = R.recommend([{"index": 0, "name": "A", "vram_gb": 12, "speed": 9}, {"index": 1, "name": "B", "vram_gb": 8, "speed": 5}, {"index": 2, "name": "C", "vram_gb": 10, "speed": 5}])
        self.assertEqual(r["roles"], {0: "main", 1: "off", 2: "helper"})


class MirrorAndParse(unittest.TestCase):
    def test_mirror_writes_roles_and_the_legacy_fields_with_main_last(self):
        m = R.mirror({0: "helper", 1: "main"}, V["hw"]["two"])
        self.assertEqual(m["gpu_roles"], {"0": "helper", "1": "main"})
        self.assertEqual((m["gpu"], m["gpu_order"]), ([0, 1], "as_given"))
        self.assertNotIn("gpu_none", m)
        one = R.mirror({0: "off", 1: "main"}, V["hw"]["two"])
        self.assertEqual(one["gpu"], 1)                                   # one card is a number, never a list (#3203)
        none = R.mirror({0: "off", 1: "off"}, V["hw"]["two"])
        self.assertEqual((none["gpu"], none["gpu_none"]), ([], True))

    def test_parse_roles(self):
        cards = V["hw"]["three"]
        self.assertEqual(R.parse_roles("1=main, 0=helper,2=off", cards), {1: "main", 0: "helper", 2: "off"})
        self.assertEqual(R.roles_from_text("1=main,0=helper", cards), {0: "helper", 1: "main", 2: "off"})      # unnamed cards are Off
        self.assertEqual(R.roles_from_text("0=helper", cards), {0: "main", 1: "off", 2: "off"})                 # a lone card on is Main
        for bad in ("", "1", "1=boss", "x=main", "0=main,1=main", "7=main"):
            with self.subTest(bad), self.assertRaises(ValueError):
                R.parse_roles(bad, cards)

    def test_normalize(self):
        two = V["hw"]["two"]
        self.assertEqual(R.normalize({0: "helper", 1: "helper"}, two), {0: "helper", 1: "main"})
        self.assertEqual(R.normalize({0: "main", 1: "main"}, two), {0: "main", 1: "helper"})
        self.assertEqual(R.normalize({0: "helper", 1: "off"}, two), {0: "main", 1: "off"})

    def test_spill_uses_the_cards_that_are_on(self):
        two = [{"index": 0, "name": "A", "vram_gb": 12}, {"index": 1, "name": "B", "vram_gb": 12}]
        a = R.spill_for({0: "main", 1: "helper"}, two, 44.0, 32.0)
        b = R.spill_for({0: "main", 1: "off"}, two, 44.0, 32.0)
        self.assertLess(a, b)
        self.assertEqual(R.spill_for({0: "main", 1: "off"}, two, None, 32.0), 0.0)


if __name__ == "__main__":
    unittest.main()
