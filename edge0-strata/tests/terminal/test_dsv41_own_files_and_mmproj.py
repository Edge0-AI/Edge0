"""#3927 / #3917: a Qwen vision projector in the shared models folder must never be taken for DeepSeek's (the pack is made without a vision
tower, no error).  #3914: DeepSeek files the person already has are recognised by the pinned shard names - in the guided screen too - instead of
by the generic GGUF check, which does not know DeepSeek (and called its n-gram shard 'could not be read')."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402
import setup  # noqa: E402


def gguf(path: Path, **kv) -> Path:
    """A GGUF with only metadata (string and uint32 values)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, v in kv.items():
        k = k.replace("__", ".")
        out += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            out += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out))
    return path


class SharedFolderMmproj(unittest.TestCase):
    def layout(self, d: Path):
        models = d / "models"
        model = models / "deepseek-v4.1-flash-IST-DAS-2.25bpw"
        model.mkdir(parents=True)
        return models, model

    def test_a_qwen_projector_in_the_shared_folder_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            models, model = self.layout(Path(d))
            gguf(models / "mmproj-Qwen3.8-Flash-Next-BF16.gguf", general__architecture="clip", clip__projector_type="qwen3vl_merger")
            gguf(models / "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", general__architecture="clip", clip__projector_type="qwen3vl_merger")
            self.assertIsNone(PG.find_mmproj(model))

    def test_a_deepseek_projector_is_found_even_after_a_qwen_one(self):
        with tempfile.TemporaryDirectory() as d:
            models, model = self.layout(Path(d))
            gguf(models / "mmproj-Qwen3.8-Flash-Next-BF16.gguf", general__architecture="clip", clip__projector_type="qwen3vl_merger")
            ds = gguf(models / "mmproj-zz-DeepSeek-BF16.gguf", general__architecture="clip", clip__projector_type="deepseek41v")
            self.assertEqual(PG.find_mmproj(model), ds)

    def test_a_broken_or_not_gguf_mmproj_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            models, model = self.layout(Path(d))
            (models / "mmproj-broken.gguf").write_bytes(b"not a gguf at all")
            (models / "mmproj-empty.gguf").write_bytes(b"")
            self.assertIsNone(PG.find_mmproj(model))

    def test_no_mmproj_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            _models, model = self.layout(Path(d))
            self.assertIsNone(PG.find_mmproj(model))

    def test_setup_does_not_name_an_mmproj_for_the_official_sets(self):
        cmd = setup.dsv41_pack_cmd(Path("/m/set"), ["a.gguf"], Path("/s/e.gguf"), Path("/s/m.gguf"), Path("/s/d.gguf"), Path("/p"))
        self.assertNotIn("--mmproj", cmd)                                  # the folder scan above decides, by the projector type


class OwnDeepSeekFiles(unittest.TestCase):
    def touch_set(self, folder: Path, names):
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            (folder / n).write_bytes(b"x")

    def test_the_pinned_shard_names_are_recognised_in_the_folder_above_and_in_the_first_releases_folder(self):
        from strata_cli.launch.runner import Runner
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        new = fam["models"]["IST-DAS-2.00bpw"]["shards"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.touch_set(d / "a", new)
            self.assertEqual(Runner.deepseek_set_in(setup, d / "a")[0], "IST-DAS-2.00bpw")
            old = ["model-2.25bpw-%05d-of-00004.gguf" % i for i in range(1, 5)]
            self.touch_set(d / "b" / "2.25bpw", old)
            m, _entry, folder = Runner.deepseek_set_in(setup, d / "b")
            self.assertEqual((m, folder), ("IST-DAS-2.25bpw", d / "b" / "2.25bpw"))
            ud = [setup.DSV41_SHARD.format(i=i) for i in range(1, 10)]
            self.touch_set(d / "c" / "UD-Q8_K_XL", ud)
            self.assertEqual(Runner.deepseek_set_in(setup, d / "c")[0], "UD-Q8_K_XL")

    def test_the_six_shard_names_are_recognised_like_the_pinned_ones(self):
        from strata_cli.launch.runner import Runner
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.touch_set(d / "a", fam["models"]["IST-DAS-2.00bpw"]["shards6"])
            self.assertEqual(Runner.deepseek_set_in(setup, d / "a")[0], "IST-DAS-2.00bpw")
            self.touch_set(d / "b" / "2.25bpw", fam["models"]["IST-DAS-2.25bpw"]["shards6"])
            m, _entry, folder = Runner.deepseek_set_in(setup, d / "b")
            self.assertEqual((m, folder), ("IST-DAS-2.25bpw", d / "b" / "2.25bpw"))

    def deepseek_files(self, folder: Path):
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner
        catalog = mock.MagicMock()
        catalog.strata.setup_module.return_value = setup
        r = Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog)
        return r, r.deepseek_files()

    def test_launch_model_path_takes_a_whole_six_shard_set_and_names_what_is_missing_from_an_incomplete_one(self):
        from strata_cli.launch.runner import LaunchError
        six = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-1.75bpw"]["shards6"]
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d) / "whole", six)
            r, got = self.deepseek_files(Path(d) / "whole")
            self.assertTrue(got)
            self.assertEqual((r.s.family, r.s.model), ("deepseek41", "IST-DAS-1.75bpw"))
            self.touch_set(Path(d) / "cut", [n for n in six if "-00003-" not in n])
            with self.assertRaises(LaunchError) as cm:
                self.deepseek_files(Path(d) / "cut")
            self.assertIn("is in five parts and 1 of them are not in", str(cm.exception))
            self.assertIn("-00003-of-00006.gguf", str(cm.exception))

    def test_the_guided_screen_takes_a_six_shard_set_and_names_a_missing_part(self):
        from strata_cli.launch import settings as S
        six = setup.MULTI_FILE_FAMILIES["deepseek41"]["models"]["IST-DAS-2.00bpw"]["shards6"]
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d) / "whole", six)
            cls, f, said = self.screen(S.LaunchSettings())
            self.assertTrue(cls.check_own_deepseek(f, str(Path(d) / "whole")))
            self.assertEqual((f.ctx.settings.family, f.ctx.settings.model), ("deepseek41", "IST-DAS-2.00bpw"))
            self.touch_set(Path(d) / "cut", six[:4])
            cls, f, said = self.screen(S.LaunchSettings())
            self.assertTrue(cls.check_own_deepseek(f, str(Path(d) / "cut")))
            self.assertIn("1 of 5 parts are not in", " ".join(said))

    def test_other_models_files_are_not_deepseeks(self):
        from strata_cli.launch.runner import Runner
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d), ["Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"])
            self.assertIsNone(Runner.deepseek_set_in(setup, Path(d)))

    def screen(self, settings):
        from strata_cli.launch.screens import ModelScreen
        f = mock.MagicMock()
        f.ctx.settings = settings
        f.ctx.catalog.strata.setup_module.return_value = setup
        said = []
        f.say = lambda text, err=False: said.append(text)
        return ModelScreen, f, said

    def test_the_guided_screen_takes_the_ud_files_as_deepseek_instead_of_saying_a_shard_could_not_be_read(self):
        from strata_cli.launch import settings as S
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d), [setup.DSV41_SHARD.format(i=i) for i in range(1, 10)])
            cls, f, said = self.screen(S.LaunchSettings())
            self.assertTrue(cls.check_own_deepseek(f, d))
            s = f.ctx.settings
            self.assertEqual((s.family, s.model, s.model_path), ("deepseek41", "UD-Q8_K_XL", str(Path(d))))
            self.assertIn("DeepSeek-V4.1-Flash", " ".join(said))
            self.assertNotIn("cannot be used", " ".join(said))

    def test_a_part_that_is_missing_is_said_plainly(self):
        from strata_cli.launch import settings as S
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d), [setup.DSV41_SHARD.format(i=i) for i in range(1, 9)])      # shard 9 is missing
            cls, f, said = self.screen(S.LaunchSettings())
            self.assertTrue(cls.check_own_deepseek(f, d))
            self.assertIn("1 of 9 parts are not in", " ".join(said))
            self.assertIsNone(f.ctx.settings.model_path)

    def test_files_of_another_model_go_on_to_the_usual_check(self):
        from strata_cli.launch import settings as S
        with tempfile.TemporaryDirectory() as d:
            self.touch_set(Path(d), ["x-00001-of-00002.gguf"])
            cls, f, said = self.screen(S.LaunchSettings())
            self.assertFalse(cls.check_own_deepseek(f, d))
            self.assertEqual(said, [])


if __name__ == "__main__":
    unittest.main()
