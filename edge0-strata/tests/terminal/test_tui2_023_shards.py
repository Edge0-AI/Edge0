"""0.2.3 RC stress test #9 (C05, #3478): a missing shard is named, with its file name, in the CLI and in the TUI's own-files message."""

from strata_control import models
from tests.terminal.test_own_files import Base, Q2


class MissingShardIsNamed(Base):
    def test_the_plan_names_the_missing_second_shard(self):
        folder = self.put(self.base / "half", Q2[:1], ["Q2_0"])
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], ["incomplete_or_invalid_shard_set"])
        self.assertEqual(len(plan["missing_shards"]), 1)
        name = plan["missing_shards"][0]
        self.assertRegex(name, r"-00002-of-0000[0-9]\.gguf$")
        self.assertEqual(models.problem_words(plan["issues"], missing=plan["missing_shards"]), f"{name} is missing")
        self.assertEqual(plan["plain_issues"], [f"{name} is missing"])

    def test_without_a_name_the_old_sentence_stays(self):
        self.assertEqual(models.problem_words(["incomplete_or_invalid_shard_set"]), "some parts of the model are missing")

    def test_several_missing_shards_are_counted(self):
        self.assertEqual(models.missing_words(["a-00002-of-00004.gguf", "a-00003-of-00004.gguf"]),
                         "a-00002-of-00004.gguf and a-00003-of-00004.gguf are missing")
        self.assertEqual(models.missing_words(["a-2", "a-3", "a-4", "a-5"]), "a-2, a-3 and 2 more files are missing")

    def test_launch_plan_message_carries_the_file_name(self):
        from tests.terminal.launch_support import run_cli
        folder = self.put(self.base / "half2", Q2[:1], ["Q2_0"])
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(folder)])
        self.assertNotEqual(code, 0)
        self.assertRegex(err + out, r"-00002-of-0000[0-9]\.gguf is missing")
