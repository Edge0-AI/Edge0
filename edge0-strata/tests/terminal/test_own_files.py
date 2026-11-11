"""0.2.2: "use my own model files" accepts every layout 0.1.45's `setup.py --gguf-dir` accepted (issues #3316 #3292 #3190 #3209 #3331).

The checker in front of setup (strata_control/models.plan_import) used to be stricter than setup itself: a split set whose first shard
holds only the metadata (Unsloth's), a link to the file (Hugging Face cache, Ollama), a folder holding several sizes, a folder above the
model folder (a whole repository, a 0.1.45 data folder), more than 128 files in a folder. Tiny GGUF files with the real shard names
stand for the real ones (the shard names and the expert types are what the table checks)."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from strata_control import gguf_preflight, models
from strata_control.profiles import ProfileError
from tools.gguf_reader import BLOCK_GEOMETRY

TYPE_IDS = {"Q8_0": 8, "IQ3_S": 21, "IQ4_NL": 20, "IQ4_XS": 23, "IQ3_XXS": 18, "Q2_0": 42, "IQ2_S": 22, "IQ2_XS": 17,
            "I8": 24, "I16": 25, "I32": 26, "I64": 27}
# One value per block. Kept here rather than read from the reader's table: the test has to build such a file whether or
# not that table sizes the type, and a wrong row there must fail the test instead of moving the fixture with it.
INTEGER_BYTES = {"I8": 1, "I16": 2, "I32": 4, "I64": 8}


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_bytes(experts: list[str], *, metadata: bool = True) -> bytes:
    """A tiny qwen4exp GGUF with one routed-expert tensor per type named; `experts=[]` is the metadata-only first shard of a split set."""
    pairs = [("general.architecture", 8, string("qwen4exp")), ("tokenizer.ggml.model", 8, string("gpt2")),
             ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")),
             ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b"))] if metadata else []
    header = b"GGUF" + struct.pack("<IQQ", 3, len(experts), len(pairs)) + \
        b"".join(string(k) + struct.pack("<I", kind) + value for k, kind, value in pairs)
    offset = 0
    for i, name in enumerate(experts):
        block, size = (1, INTEGER_BYTES[name]) if name in INTEGER_BYTES else BLOCK_GEOMETRY[name]
        header += string(f"blk.{i}.ffn_gate_exps.weight") + struct.pack("<IQIQ", 1, block, TYPE_IDS[name], offset)
        offset += -(-size // 32) * 32
    header += b"\x00" * (-len(header) % 32)
    return header + b"\x00" * offset


UNSLOTH_Q4 = ["Qwen3.8-Flash-Next-UD-Q4_K_XL-0000%d-of-00004.gguf" % i for i in (1, 2, 3, 4)]
UNSLOTH_IQ4 = ["Qwen3.8-Flash-Next-UD-IQ4_XS-0000%d-of-00003.gguf" % i for i in (1, 2, 3)]
Q2 = ["Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000%d-of-00002.gguf" % i for i in (1, 2)]
IQ3XXS = ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000%d-of-00002.gguf" % i for i in (1, 2)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-own-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def put(self, folder: Path, names: list[str], experts: list[str]) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            (folder / n).write_bytes(gguf_bytes(experts))
        return folder

    def gsq(self, folder: Path, names: list[str], kind: str) -> Path:
        return self.put(folder, names, [kind])

    def unsloth(self, folder: Path, names: list[str], kinds: list[str]) -> Path:
        """The first shard has only the metadata (what llama.cpp's gguf-split makes), the others carry tensors."""
        folder.mkdir(parents=True, exist_ok=True)
        (folder / names[0]).write_bytes(gguf_bytes([]))
        for n in names[1:]:
            (folder / n).write_bytes(gguf_bytes(kinds, metadata=False))
        return folder


class MetadataOnlyFirstShard(Base):
    def test_unsloth_split_set_is_recognised(self):
        folder = self.unsloth(self.base / "u", UNSLOTH_Q4, ["IQ4_NL"])
        plan = models.plan_import(str(folder))
        # the tiny files carry IQ4_NL experts, which UD-Q4_K_XL does not list: the set is found, the quantization is what is flagged
        self.assertNotIn("unsupported_or_missing_architecture", plan["issues"])
        self.assertNotIn("unsupported_or_incomplete_tokenizer_metadata", plan["issues"])
        self.assertNotIn("GGUF tensor or metadata count", " ".join(plan["issues"]))
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("unsloth", "UD-Q4_K_XL"))

    def test_unsloth_iq4_xs_passes_with_its_real_expert_types(self):
        folder = self.unsloth(self.base / "u", UNSLOTH_IQ4, ["IQ3_S", "IQ4_NL", "IQ4_XS", "Q8_0"])
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "UD-IQ4_XS")
        self.assertEqual(plan["model_dir"], str(folder))

    def test_a_shard_with_no_tensors_and_no_metadata_is_still_refused(self):
        f = self.base / "x.gguf"
        f.write_bytes(gguf_bytes([], metadata=False))
        plan = models.plan_import(str(f))
        self.assertIn("unsupported_or_missing_architecture", plan["issues"])


class Links(Base):
    def make_links(self):
        blobs = self.base / "blobs"
        snap = self.base / "snapshots" / "abc" / "Q2_0"
        blobs.mkdir()
        snap.mkdir(parents=True)
        for i, n in enumerate(Q2):
            (blobs / f"sha{i}").write_bytes(gguf_bytes(["Q2_0"]))
            try:
                os.symlink(blobs / f"sha{i}", snap / n)
            except (OSError, NotImplementedError):
                self.skipTest("this account cannot create symbolic links")
        return snap

    def test_a_hugging_face_cache_snapshot_is_used_through_its_links(self):
        snap = self.make_links()
        plan = models.plan_import(str(snap))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")
        self.assertEqual(plan["file_count"], 2)

    def test_a_link_to_the_folder_is_followed(self):
        real = self.gsq(self.base / "real", Q2, "Q2_0")
        link = self.base / "link"
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symbolic links")
        self.assertEqual(models.plan_import(str(link))["issues"], [])

    def test_a_link_to_a_file_that_is_gone_says_so(self):
        snap = self.base / "s"
        snap.mkdir()
        try:
            os.symlink(self.base / "gone", snap / Q2[0])
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symbolic links")
        with self.assertRaises(ProfileError) as ctx:
            gguf_preflight.real_model_file(snap / Q2[0])
        self.assertIn("link to a file that is not there", str(ctx.exception))

    def test_a_missing_file_is_missing_not_a_link(self):
        with self.assertRaises(ProfileError) as ctx:
            gguf_preflight.real_model_file(self.base / "nope.gguf")
        self.assertIn("is missing", str(ctx.exception))

    def test_run_configs_keep_the_no_link_rule(self):
        cfg = self.base / "real.json"
        cfg.write_text("{}")
        link = self.base / "strata-x.json"
        try:
            os.symlink(cfg, link)
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symbolic links")
        with self.assertRaises(ProfileError) as ctx:
            models._read(link)                               # noqa: SLF001
        self.assertIn("symlink", str(ctx.exception))


class IntegerTensors(Base):
    """The IST-DAS DeepSeek-V4.1 first shard keeps the Engram's hash parameters as integer tensors instead of metadata
    (tools/dsv41/pack_gguf.py: I32 and I64), so the table in front of setup and the app sizes them instead of refusing the file."""

    def test_a_tensor_directory_with_integer_tensors_is_read(self):
        p = self.base / "DeepSeek-V4.1-Flash-IST-DAS-2.00bpw-00001-of-00004.gguf"
        p.write_bytes(gguf_bytes(["I32", "I64"]))
        self.assertEqual(gguf_preflight.inspect_metadata(p)["tensor_types"], {"I32": 1, "I64": 1})

    def test_each_integer_type_is_sized_at_one_value_a_block(self):
        for kind, size in INTEGER_BYTES.items():
            p = self.base / f"integer-{kind}.gguf"
            p.write_bytes(gguf_bytes([kind]))
            self.assertEqual(gguf_preflight.inspect_metadata(p)["payload_bytes_required"], size, kind)


class Folders(Base):
    def test_many_other_gguf_files_in_the_folder_do_not_refuse(self):
        folder = self.gsq(self.base / "m", Q2, "Q2_0")
        for i in range(140):
            (folder / f"other-{i:03d}.gguf").write_bytes(b"GGUF")
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")
        self.assertEqual(plan["file_count"], 2)

    def test_several_sizes_in_one_folder_name_them_and_ask_for_one_file(self):
        folder = self.gsq(self.base / "many", Q2, "Q2_0")
        self.gsq(folder, IQ3XXS, "IQ3_XXS")
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], ["multiple_model_groups"])
        self.assertIsNone(plan["candidate_selection"])
        self.assertEqual(sorted(h["model"] for h in plan["folder_holds"]), ["IQ3_XXS", "Q2_0"])
        self.assertTrue(any("holds 2 models" in m and "type the path of its .gguf file" in m for m in plan["messages"]))

    def test_one_known_size_among_other_files_is_used_and_the_rest_is_named(self):
        folder = self.gsq(self.base / "mix", Q2, "Q2_0")
        (folder / "some-other-model-Q4_K_M.gguf").write_bytes(gguf_bytes(["IQ3_S"]))
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")
        self.assertEqual(plan["file_count"], 2)
        self.assertTrue(any("some-other-model-Q4_K_M.gguf" in m and "leaves the others alone" in m for m in plan["messages"]))

    def test_a_shard_file_picks_its_set_out_of_a_folder_of_several(self):
        folder = self.gsq(self.base / "many", Q2, "Q2_0")
        self.gsq(folder, IQ3XXS, "IQ3_XXS")
        plan = models.plan_import(str(folder / IQ3XXS[1]))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "IQ3_XXS")
        self.assertEqual(plan["model_dir"], str(folder))

    def test_a_folder_above_the_model_folder_is_searched(self):
        data = self.base / "Strata-data"
        self.gsq(data / "models" / "q2_0", Q2, "Q2_0")
        (data / "packs").mkdir()
        for start in (data, data / "models"):
            plan = models.plan_import(str(start))
            self.assertEqual(plan["issues"], [], start)
            self.assertEqual(plan["model_dir"], str(data / "models" / "q2_0"))
            self.assertTrue(any("The model files are in" in m for m in plan["messages"]))

    def test_a_whole_repository_with_one_folder_per_size_asks_which(self):
        repo = self.base / "repo"
        self.gsq(repo / "Q2_0", Q2, "Q2_0")
        self.gsq(repo / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        plan = models.plan_import(str(repo))
        self.assertIn("multiple_model_groups", plan["issues"])
        self.assertEqual(sorted((Path(h["folder"]).name, h["model"]) for h in plan["folder_holds"]),
                         [("IQ3_XXS", "IQ3_XXS"), ("Q2_0", "Q2_0")])
        self.assertTrue(plan["file_set_status"] == "invalid_or_incomplete")

    def test_one_known_folder_next_to_unrelated_ones_is_used(self):
        top = self.base / "all"
        self.gsq(top / "models" / "q2_0", Q2, "Q2_0")
        (top / "stuff").mkdir()
        (top / "stuff" / "notes.txt").write_text("x")
        plan = models.plan_import(str(top))
        self.assertEqual(plan["candidate_selection"]["model"], "Q2_0")

    def test_a_folder_with_no_gguf_anywhere_says_so(self):
        (self.base / "empty" / "a" / "b").mkdir(parents=True)
        with self.assertRaises(ProfileError) as ctx:
            models.plan_import(str(self.base / "empty"))
        self.assertIn("no GGUF files found", str(ctx.exception))

    def test_a_folder_name_with_spaces_and_a_second_drive_style_path(self):
        folder = self.gsq(self.base / "my models" / "Qwen 3.8 files", Q2, "Q2_0")
        self.assertEqual(models.plan_import(str(folder))["issues"], [])

    def test_a_missing_second_shard_is_one_plain_issue(self):
        folder = self.put(self.base / "half", Q2[:1], ["Q2_0"])
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], ["incomplete_or_invalid_shard_set"])


class Launcher(Base):
    """`strata launch --model-path` and the TUI's "use my own files" (strata_cli/launch/runner.py, screens.py)."""

    def runner(self, **kw):
        from strata_cli.launch import settings as S
        from strata_cli.launch.catalog import Catalog
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import HW_12
        return Runner(S.LaunchSettings(**kw), catalog=Catalog(hardware=HW_12))

    def test_a_folder_above_the_model_is_resolved_to_the_model_folder(self):
        data = self.base / "Strata-data"
        self.gsq(data / "models" / "q2_0", Q2, "Q2_0")
        r = self.runner(model_path=str(data))
        r.resolve_own_files()
        self.assertEqual((r.s.family, r.s.model), ("qwen", "Q2_0"))
        self.assertEqual(Path(r.s.model_path), data / "models" / "q2_0")
        argv = r.setup_args()
        self.assertEqual(Path(argv[argv.index("--gguf-dir") + 1]), data / "models" / "q2_0")

    def test_a_named_model_in_a_folder_of_several_sizes_is_never_refused_here(self):
        folder = self.gsq(self.base / "many", Q2, "Q2_0")
        self.gsq(folder, IQ3XXS, "IQ3_XXS")
        r = self.runner(model_path=str(folder), family="qwen", model="IQ3_XXS")
        r.resolve_own_files()
        self.assertEqual(Path(r.s.model_path), folder)               # setup picks the IQ3_XXS files by name, as in 0.1.45
        self.assertEqual((r.s.family, r.s.model), ("qwen", "IQ3_XXS"))

    def test_a_named_model_is_found_in_the_folder_that_holds_it(self):
        repo = self.base / "repo"
        self.gsq(repo / "Q2_0", Q2, "Q2_0")
        self.gsq(repo / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        r = self.runner(model_path=str(repo), family="qwen", model="IQ3_XXS")
        r.resolve_own_files()
        self.assertEqual(Path(r.s.model_path), repo / "IQ3_XXS")

    def test_two_models_and_none_named_says_which_in_plain_words(self):
        from strata_cli.launch.runner import LaunchError
        repo = self.base / "repo"
        self.gsq(repo / "Q2_0", Q2, "Q2_0")
        self.gsq(repo / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        r = self.runner(model_path=str(repo))
        with self.assertRaises(LaunchError) as ctx:
            r.resolve_own_files()
        text = str(ctx.exception)
        self.assertIn("more than one model", text)
        self.assertIn("holds 2 models", text)

    def test_a_linked_model_file_keeps_its_name(self):
        snap = Links.make_links(self)
        r = self.runner(model_path=str(snap / Q2[0]))
        r.resolve_own_files()
        self.assertEqual((r.s.family, r.s.model), ("qwen", "Q2_0"))
        self.assertEqual(Path(r.s.model_path).name, Q2[0])           # not the blob's hash name


class Words(unittest.TestCase):
    def test_a_file_problem_names_the_file_and_says_why(self):
        self.assertEqual(models.plain_issue("model-00002-of-00002.gguf: GGUF metadata is truncated"),
                         "model-00002-of-00002.gguf: the file is cut short (a download or copy that did not finish)")
        self.assertIn("shorter than its own header", models.plain_issue("a.gguf: GGUF tensor payload is shorter than its directory requires"))
        self.assertIn("old format (version 2)", models.plain_issue("a.gguf: Strata metadata preflight requires GGUF v3"))
        self.assertEqual(models.plain_issue("incomplete_or_invalid_shard_set"), "some parts of the model are missing")

    def test_the_plan_carries_the_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / Q2[0]
            f.write_bytes(gguf_bytes(["Q2_0"])[:60])
            plan = models.plan_import(str(f))
        self.assertTrue(plan["plain_issues"])
        self.assertTrue(any(x.startswith(Q2[0] + ":") or x.startswith(Q2[1] + ":") for x in plan["plain_issues"]), plan["plain_issues"])


class Table(unittest.TestCase):
    def test_every_size_setup_installs_is_in_the_table(self):
        """Setup can install it from --gguf-dir, so the checker in front of it must know it too (0.2.0 knew 7 of 10)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("strata_setup_for_test", models.ROOT / "setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        table = json.loads((models.ROOT / "data" / "gguf_fingerprints.json").read_text(encoding="utf-8"))
        missing = []
        for family, fam in setup.FAMILIES.items():
            if fam.get("ready"):
                continue
            for model, row in setup.MODELS.items():
                if family not in row.get("families", ("qwen", "swift")):
                    continue
                names = {setup.model_file(fam, model, i) for i in range(1, setup.model_shards(fam, model) + 1)}
                key = f"{family}/{model}"
                have = {Path(n).name for n in (table.get(key) or {}).get("files", [])}
                if key not in models.SUPPORTED_SELECTIONS or have != names:
                    missing.append(key)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
