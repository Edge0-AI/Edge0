"""#4662: the server says, in one plain line, when a card in use is of an architecture the compiled engine has no code for.

    python -m unittest serve.test_engine_arch_note
"""
import unittest

import serve.server as server

CARDS = {0: 75, 1: 86}


class ArchNote(unittest.TestCase):
    def test_a_card_the_build_does_not_cover_is_said(self):
        line = server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75]})
        self.assertIn("sm_75", line)
        self.assertIn("sm_86", line)
        self.assertIn("update", line)

    def test_a_covered_pair_says_nothing(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75, 86]}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": ["sm_75", "sm_86"]}))

    def test_only_the_cards_the_run_uses_count(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0]}, CARDS, {"archs": [75]}))
        self.assertIsNotNone(server.engine_arch_note({}, CARDS, {"archs": [75]}))     # no "gpu": every card

    def test_a_build_with_ptx_covers_newer_cards(self):
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": [75], "ptx": True}))

    def test_unknown_things_say_nothing(self):
        self.assertIsNone(server.engine_arch_note({"backend": "hip"}, CARDS, {"archs": [75]}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {}))
        self.assertIsNone(server.engine_arch_note({"gpu": [0, 1]}, CARDS, {"archs": ["x"]}))
        self.assertIsNone(server.engine_arch_note({"exe": "component:strata-engine"}))


if __name__ == "__main__":
    unittest.main()
