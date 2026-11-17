"""Config "env_to_user": opencode's <env> block (working directory, date) leaves the system message for the first user
message, so the system turn is the same in every session and the prefix cache keeps it.  Off (the default), nothing moves."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.frontend as fe  # noqa: E402

SYS = ("You are opencode.\n"
       "Here is some useful information about the environment you are running in:\n"
       "<env>\n"
       "  Working directory: C:/a\n"
       "  Today's date: Mon Oct 12 2026\n"
       "</env>\n"
       "Instructions that never change.")


class EnvBlockToUser(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, fe, "ENV_TO_USER", fe.ENV_TO_USER)

    def convert(self, on, system=SYS, user="hi"):
        fe.ENV_TO_USER = on
        msgs, _, _ = fe.openai_to_messages({"messages": [{"role": "system", "content": system},
                                                         {"role": "user", "content": user}]})
        return msgs

    def test_moved_to_the_first_user_message(self):
        msgs = self.convert(True)
        self.assertEqual(msgs[0]["content"], "You are opencode.\nInstructions that never change.")
        self.assertTrue(msgs[1]["content"].startswith(
            "Here is some useful information about the environment you are running in:\n<env>\n  Working directory: C:/a"))
        self.assertTrue(msgs[1]["content"].endswith("</env>\n\nhi"))

    def test_two_folders_share_the_system_turn(self):
        a = self.convert(True)[0]["content"]
        b = self.convert(True, system=SYS.replace("C:/a", "D:/b"))[0]["content"]
        self.assertEqual(a, b)

    def test_content_parts_get_a_leading_text_part(self):
        fe.ENV_TO_USER = True
        out = fe._env_to_user([{"role": "system", "content": SYS},
                               {"role": "user", "content": [{"type": "image", "image": "x.png"}, {"type": "text", "text": "hi"}]}])
        self.assertEqual(out[1]["content"][0]["type"], "text")
        self.assertIn("<env>", out[1]["content"][0]["text"])
        self.assertEqual(out[1]["content"][1:], [{"type": "image", "image": "x.png"}, {"type": "text", "text": "hi"}])

    def test_text_parts_are_read_as_one_text(self):
        msgs = self.convert(True, user=[{"type": "text", "text": "hi"}])
        self.assertIn("<env>", msgs[1]["content"] if isinstance(msgs[1]["content"], str) else msgs[1]["content"][0]["text"])

    def test_anthropic_requests_too(self):
        fe.ENV_TO_USER = True
        msgs, _, _ = fe.anthropic_to_messages({"system": SYS, "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("<env>", msgs[0]["content"])
        self.assertIn("<env>", msgs[1]["content"])

    def test_off_and_no_block_change_nothing(self):
        self.assertEqual(self.convert(False)[0]["content"], SYS)
        self.assertEqual(self.convert(True, system="Plain system prompt.")[0]["content"], "Plain system prompt.")


if __name__ == "__main__":
    unittest.main()
