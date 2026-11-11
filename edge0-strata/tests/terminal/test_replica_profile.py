"""strata_control/replica_profile.py: the opt-in replica serving profile (one engine per card, shared expert arena)."""

from __future__ import annotations

import unittest

from strata_control import replica_profile as P

R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 31.9, "vendor": "amd"}


def cards(n, **kw):
    return [{**R9700, **kw, "index": i} for i in range(n)]


def cfg(**kw):
    c = {"model_name": "qwen3.8-flash-next-iq3_s", "gpu": [0, 1, 2, 3], "layer_split": "auto",
         "args": ["--pack", "/p/iq3_s", "--spec", "4"], "replica_profile": True}
    c.update(kw)
    return c


class Request(unittest.TestCase):
    def test_opt_in_only(self):
        self.assertFalse(P.requested({}, {}))
        self.assertTrue(P.requested({"replica_profile": True}, {}))
        self.assertTrue(P.requested({}, {"STRATA_REPLICA_PROFILE": "1"}))

    def test_an_explicit_no_wins(self):
        self.assertFalse(P.requested({"replica_profile": False}, {"STRATA_REPLICA_PROFILE": "1"}))

    def test_without_the_request_nothing_changes(self):
        c = cfg(replica_profile=None)
        c.pop("replica_profile")
        before = dict(c)
        self.assertEqual(P.apply(cards(4), c, 256.0, platform="linux", environ={}), [])
        self.assertEqual(c, before)


class Count(unittest.TestCase):
    def test_ram_sets_the_number_of_engines(self):
        self.assertEqual(P.replicas_for(4, 50.3, 251.0, True), 4)
        self.assertEqual(P.replicas_for(4, 50.3, 128.0, True), 4)
        self.assertEqual(P.replicas_for(4, 50.3, 96.0, True), 4)
        self.assertEqual(P.replicas_for(4, 50.3, 80.0, True), 2)     # 80 - 12 - 50.3 = 17.7 -> 2
        self.assertEqual(P.replicas_for(4, 50.3, 64.0, True), 0)     # one engine's room only
        self.assertEqual(P.replicas_for(2, 50.3, 251.0, False), 2)   # private copies: 2 x 58.3 + 12
        self.assertEqual(P.replicas_for(4, 50.3, 200.0, False), 3)
        self.assertEqual(P.replicas_for(2, 50.3, 100.0, False), 1)


class Apply(unittest.TestCase):
    def test_four_cards_with_plenty_of_ram(self):
        c = cfg()
        notes = P.apply(cards(4), c, 251.0, platform="linux", environ={})
        self.assertTrue(notes)
        self.assertEqual(c["replicas"], [{"gpus": [0]}, {"gpus": [1]}, {"gpus": [2]}, {"gpus": [3]}])
        self.assertIs(c["replica_shared_arena"], True)
        self.assertEqual(c["parallel"], 4)
        self.assertEqual(c["env"]["STRATA_CKPT_POOL"], "1")
        self.assertEqual(c["env"]["STRATA_QSA_SLOTS"], "1")

    def test_two_cards_get_eight_slots_each(self):
        c = cfg(gpu=[0, 1])
        P.apply(cards(2), c, 128.0, platform="linux", environ={})
        self.assertEqual(c["parallel"], 8)
        self.assertEqual(len(c["replicas"]), 2)

    def test_less_ram_drops_cards_and_says_so(self):
        c = cfg()
        notes = P.apply(cards(4), c, 80.0, platform="linux", environ={})
        self.assertEqual(c["replicas"], [{"gpus": [0]}, {"gpus": [1]}])
        self.assertEqual(c["gpu"], [0, 1])
        self.assertTrue(any("left out" in n for n in notes))

    def test_too_little_ram_keeps_the_layer_split(self):
        c = cfg()
        notes = P.apply(cards(4), c, 64.0, platform="linux", environ={})
        self.assertNotIn("replicas", c)
        self.assertEqual(c["layer_split"], "auto")
        self.assertTrue(notes and "layer split stays" in notes[0])

    def test_windows_shares_through_a_section(self):
        c = cfg(gpu=[0, 1])
        P.apply(cards(2), c, 80.0, platform="win32", environ={})
        self.assertEqual(len(c["replicas"]), 2)
        self.assertIs(c["replica_shared_arena"], True)

    def test_an_os_that_cannot_share_needs_the_private_copies_to_fit(self):
        c = cfg(gpu=[0, 1])
        self.assertEqual(P.apply(cards(2), c, 100.0, platform="darwin", environ={})[0].split(":")[0], "replica profile")
        self.assertNotIn("replicas", c)
        c = cfg(gpu=[0, 1])
        P.apply(cards(2), c, 160.0, platform="darwin", environ={})
        self.assertEqual(len(c["replicas"]), 2)
        self.assertNotIn("replica_shared_arena", c)

    def test_what_the_person_chose_wins(self):
        for kw in ({"replicas": 2}, {"layer_split": "13,26,39"}, {"args": ["--pack", "/p/iq3_s", "--peer-device", "1"]},
                   {"args": ["--pack", "/p/iq3_s", "--layer-split", "20,40"]}, {"arch": "deepseek41"}):
            c = cfg(**kw)
            before = dict(c)
            P.apply(cards(4), c, 251.0, platform="linux", environ={})
            self.assertEqual(c.get("replicas"), before.get("replicas"), kw)
            self.assertEqual(c.get("layer_split"), before.get("layer_split"), kw)

    def test_parallel_and_environment_variables_of_the_person_win(self):
        c = cfg(parallel=6)
        P.apply(cards(4), c, 251.0, platform="linux", environ={"STRATA_CKPT_POOL": "0"})
        self.assertEqual(c["parallel"], 6)
        self.assertNotIn("STRATA_CKPT_POOL", c["env"])
        c = cfg()
        P.apply(cards(4), c, 251.0, parallel_given=True, platform="linux", environ={})
        self.assertNotIn("parallel", c)

    def test_only_the_measured_hardware_and_model(self):
        for kw in ({"model_name": "qwen3.6-35b-a3b-q4"}, {"model_name": "qwen3.8-flash-next-iq2_xs", "args": ["--pack", "/p/x"]}):
            c = cfg(**kw)
            P.apply(cards(4), c, 251.0, platform="linux", environ={})
            self.assertNotIn("replicas", c, kw)
        c = cfg(gpu=[0, 1])
        P.apply(cards(2, vram_gb=16.0), c, 251.0, platform="linux", environ={})
        self.assertNotIn("replicas", c)
        c = cfg(gpu=[0])
        P.apply(cards(1), c, 251.0, platform="linux", environ={})
        self.assertNotIn("replicas", c)


if __name__ == "__main__":
    unittest.main()
