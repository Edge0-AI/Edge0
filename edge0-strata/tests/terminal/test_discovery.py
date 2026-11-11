"""Tester report 2026-10-18, T03 / T04 / T05: existing models are found the same way by the launch screens, `strata models list`, the
own-files import and `strata launch` (strata_control/discovery.py), with a mock of the desktop app's folders and a Gyro-S file header
(TQK6 / TQK7 experts, the one-file layout)."""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.launch import modelstate, settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import Runner
from strata_control import discovery, models
from tests.terminal.launch_support import HW_12

GYRO_S = "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf"
GYRO_M = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
TQ2_T, TQK6, TQK7 = 144, 145, 146


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_with(tensors: list[tuple[str, int, int, int]]) -> bytes:
    """A tiny qwen4exp GGUF: (name, ggml type id, elements, payload bytes) per tensor, laid out one after another."""
    pairs = [("general.architecture", 8, string("qwen4exp")), ("tokenizer.ggml.model", 8, string("gpt2")),
             ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")),
             ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b"))]
    header = (b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(pairs)) +
              b"".join(string(k) + struct.pack("<I", kind) + value for k, kind, value in pairs))
    offset = 0
    for name, type_id, elements, nbytes in tensors:
        header += string(name) + struct.pack("<IQIQ", 1, elements, type_id, offset)
        offset += -(-nbytes // 32) * 32
    header += b"\x00" * (-len(header) % 32)
    return header + b"\x00" * offset


def gyro_s_bytes() -> bytes:
    return gguf_with([("blk.0.ffn_gate_exps.weight", TQK6, 256, 52), ("blk.0.ffn_down_exps.weight", TQK7, 256, 60)])


def gyro_m_bytes() -> bytes:
    return gguf_with([("blk.0.ffn_gate_exps.weight", TQ2_T, 256, 68)])


class GuiLayout(unittest.TestCase):
    """The desktop app's folders, as the Windows (and Linux) app lays them out, plus an empty Terminal folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-discovery-")
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.base = base
        self.env = {"LOCALAPPDATA": str(base / "local"), "XDG_DATA_HOME": str(base / "local"), "APPDATA": str(base / "roaming"),
                    "XDG_CONFIG_HOME": str(base / "config"), "USERPROFILE": str(base / "home"), "HOME": str(base / "home"),
                    "XDG_STATE_HOME": str(base / "state"), "XDG_DOCUMENTS_DIR": ""}
        settings = Path(self.env["APPDATA"]) / "Strata" if os.name == "nt" else Path(self.env["XDG_CONFIG_HOME"]) / "strata"
        self.settings_path = settings / "settings.json"
        patch = mock.patch.dict(os.environ, self.env)
        patch.start()
        self.addCleanup(patch.stop)
        for key in ("STRATA_INSTALL_DIR", "STRATA_SETTINGS_FILE", "STRATA_COMPONENTS_DIR", "STRATA_DOCUMENTS_DIR"):
            patch = mock.patch.dict(os.environ)
            patch.start()
            self.addCleanup(patch.stop)
            os.environ.pop(key, None)
        self.terminal = base / "terminal"
        (self.terminal / "data").mkdir(parents=True)
        shutil.copy(Path(models.__file__).resolve().parents[1] / "data" / "gguf_fingerprints.json", self.terminal / "data")
        p = mock.patch.object(models, "ROOT", self.terminal)
        p.start()
        self.addCleanup(p.stop)
        self.install = discovery.app_install_dir()
        self.docs = base / "home" / "Documents" / "Strata"
        gyro = self.docs / "Models" / "gyro-Gyro-S"
        gyro.mkdir(parents=True)
        self.gguf = gyro / GYRO_S
        self.gguf.write_bytes(gyro_s_bytes())
        (gyro / (GYRO_S + ".done")).write_text("done")
        (self.docs / "packs" / "gyro-gyro-s" / "tokenizer").mkdir(parents=True)
        (self.docs / "packs" / "gyro-gyro-s" / "dense.bin").write_bytes(b"x")
        (self.docs / "mtp" / "rt").mkdir(parents=True)
        (self.docs / "mtp" / "rt" / "experts.bin").write_bytes(b"x")
        self.mmproj = self.docs / "Models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
        self.mmproj.write_bytes(b"x")
        self.install.mkdir(parents=True)
        engine = base / "engine.exe"
        engine.write_bytes(b"e")
        tok = self.docs / "packs" / "gyro-gyro-s" / "tokenizer"
        (tok / "vocab.json").write_text("{}")
        self.cfg = {"exe": "component:strata-engine", "tokenizer": str(tok), "model_name": "qwen3.8-flash-next-gyro-gyro-s",
                    "args": ["--pack", str(self.docs / "packs" / "gyro-gyro-s"), "--native", str(self.gguf), "--ple-gguf", str(self.gguf),
                             "--mtp", str(self.docs / "mtp" / "rt"), "--max-context", "131072", "--kv", "int8", "--vision"],
                    "vision": {"exe": str(engine), "mmproj": str(self.mmproj), "model": str(self.gguf), "gpu": False}}
        self.config_path = self.install / "strata-gyro-gyro-s.json"
        self.config_path.write_text(json.dumps(self.cfg))
        (self.install.parent / "app-settings.json").write_text(json.dumps({"data_dir": str(self.docs), "installs": [str(self.install)]}))

    def catalog(self) -> Catalog:
        return Catalog(hardware=HW_12)

    # ---- T04: one answer for list and the wizard
    def test_models_list_finds_the_app_install_by_default(self):
        result = models.list_installed()
        self.assertEqual([m["id"] for m in result["models"]], ["gyro-gyro-s"])
        self.assertEqual(Path(result["install_dir"]).resolve(), self.install.resolve())
        self.assertEqual(result["install_dir_source"], "app")

    def test_list_and_the_wizard_agree_on_the_install(self):
        cat = self.catalog()
        self.assertEqual(cat.root.resolve(), models.resolve_install()[0].resolve())
        self.assertIsNotNone(cat.installed_config("gyro-gyro-s"))
        row = next(r for r in cat.rows() if (r.family, r.model) == ("gyro", "Gyro-S"))
        self.assertTrue(row.installed)

    def test_cli_list_names_where_it_looked_when_nothing_is_found(self):
        self.config_path.unlink()
        out = io.StringIO()
        from strata_cli.cli import main
        with redirect_stdout(out):
            code = main(["models", "list"])
        self.assertEqual(code, 0)
        self.assertIn("No installed models found. Looked in:", out.getvalue())
        self.assertIn("--install-dir", out.getvalue())

    def test_an_explicit_install_dir_is_the_only_place(self):
        other = self.base / "other"
        other.mkdir()
        self.assertEqual(models.list_installed(str(other))["models"], [])
        self.assertEqual([p.source for p in discovery.places(explicit=str(other))], ["explicit"])

    def test_a_remembered_older_install_and_the_app_install_are_both_listed(self):
        old = self.base / "T-Strata"
        old.mkdir()
        (old / "strata-qwen-iq3_xxs.json").write_text(json.dumps({"exe": "e", "tokenizer": "t", "args": []}))
        settings = self.settings_path
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"installs": [str(old)]}))
        got = models.list_installed()
        self.assertEqual(sorted(m["id"] for m in got["models"]), ["gyro-gyro-s", "qwen-iq3_xxs"])
        self.assertEqual(got["install_dir_source"], "remembered")

    def test_inside_the_app_an_older_install_is_still_not_adopted(self):
        old = self.base / "T-Strata"
        old.mkdir()
        (old / "strata-x.json").write_text(json.dumps({"exe": "e", "tokenizer": "t", "args": []}))
        app_settings = self.install.parent / "app-settings.json"
        with mock.patch.dict(os.environ, {"STRATA_SETTINGS_FILE": str(app_settings)}):
            settings = self.settings_path
            settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps({"installs": [str(old)]}))
            self.assertEqual([m["id"] for m in models.list_installed()["models"]], ["gyro-gyro-s"])

    # ---- T03: the own-files import knows Gyro
    def test_import_recognises_a_gyro_s_folder_and_its_prepared_files(self):
        plan = models.plan_import(str(self.gguf.parent))
        self.assertEqual(plan["issues"], [])
        self.assertEqual((plan["candidate_selection"]["family"], plan["candidate_selection"]["model"]), ("gyro", "Gyro-S"))
        self.assertEqual(sorted(plan["observed_expert_types"]), ["TQK6", "TQK7"])
        done = plan["already_set_up"]
        self.assertEqual(Path(done["config"]).resolve(), self.config_path.resolve())
        self.assertEqual(Path(done["pack"]).resolve(), (self.docs / "packs" / "gyro-gyro-s").resolve())
        self.assertEqual(Path(done["mtp"]).resolve(), (self.docs / "mtp" / "rt").resolve())
        self.assertEqual(Path(done["data_dir"]).resolve(), self.docs.resolve())

    def test_import_recognises_the_file_itself(self):
        plan = models.plan_import(str(self.gguf))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "Gyro-S")

    def test_import_recognises_gyro_m(self):
        folder = self.base / "gyro-m"
        folder.mkdir()
        (folder / GYRO_M).write_bytes(gyro_m_bytes())
        plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["model"], "Gyro-M")

    def test_a_folder_with_two_models_says_which_and_asks_for_one_file(self):
        folder = self.base / "both"
        folder.mkdir()
        (folder / GYRO_S).write_bytes(gyro_s_bytes())
        (folder / GYRO_M).write_bytes(gyro_m_bytes())
        plan = models.plan_import(str(folder))
        self.assertIn("multiple_model_groups", plan["issues"])
        self.assertEqual(sorted((h["family"], h["model"]) for h in plan["folder_holds"]), [("gyro", "Gyro-M"), ("gyro", "Gyro-S")])
        from strata_cli.cli import _import_problem
        text = _import_problem(plan)
        self.assertIn("more than one model", text)
        self.assertIn("holds 2 models", text)
        self.assertIn("gyro Gyro-S", text)
        self.assertIn("type the path of its .gguf file", text)
        self.assertNotIn("multiple_model_groups", text)
        self.assertNotIn("not a model size", text)

    def test_import_still_refuses_gyro_s_with_the_wrong_expert_format(self):
        folder = self.base / "wrong"
        folder.mkdir()
        (folder / GYRO_S).write_bytes(gyro_m_bytes())               # TQ2_T experts under the Gyro-S name
        plan = models.plan_import(str(folder))
        self.assertIn("missing_or_unexpected_expert_quantization", plan["issues"])

    def test_a_renamed_file_is_known_by_its_header(self):
        data = gyro_s_bytes()
        folder = self.base / "renamed"
        folder.mkdir()
        (folder / "my-copy.gguf").write_bytes(data)
        fake_root = self.base / "fake-root"
        (fake_root / "data").mkdir(parents=True)
        payload = 52 + 60 + 12                                     # 32-byte aligned offsets: 64 + 64 -> see below
        table = {"gyro/Gyro-S": {"arch": "qwen4exp", "tensors": 2, "bytes": None, "expert_bytes": {"TQK6": 52, "TQK7": 60},
                                 "files": [GYRO_S], "repo": "x"}}
        first = models.plan_import(str(folder / "my-copy.gguf"))
        self.assertIsNone(first["candidate_selection"])           # not in the real table under another name
        table["gyro/Gyro-S"]["bytes"] = first["metadata"]["payload_bytes_required"]
        (fake_root / "data" / "gguf_fingerprints.json").write_text(json.dumps(table))
        with mock.patch.object(models, "ROOT", fake_root):
            plan = models.plan_import(str(folder / "my-copy.gguf"))
        self.assertEqual(plan["candidate_selection"]["basis"], "header_size_and_tensors")
        self.assertEqual(plan["candidate_selection"]["model"], "Gyro-S")
        self.assertEqual(payload, 124)

    def test_a_fine_tune_with_the_layout_of_a_known_build_is_used_not_refused(self):        # #3651 #3777
        folder = self.base / "finetune"
        folder.mkdir()
        (folder / "Qwen3.8-Flash-Next-Gyro-S-abliterated-TQ1_0.gguf").write_bytes(gyro_s_bytes())
        fake_root = self.base / "fake-root2"
        (fake_root / "data").mkdir(parents=True)
        table = {"gyro/Gyro-S": {"arch": "qwen4exp", "tensors": 2, "bytes": 999_999, "expert_bytes": {"TQK6": 52, "TQK7": 60},
                                 "files": [GYRO_S], "repo": "x"},
                 "gyro/Gyro-M": {"arch": "qwen4exp", "tensors": 2, "bytes": 888_888, "expert_bytes": {"TQ2_T": 40},
                                 "files": [GYRO_S], "repo": "x"}}
        (fake_root / "data" / "gguf_fingerprints.json").write_text(json.dumps(table))
        with mock.patch.object(models, "ROOT", fake_root):
            plan = models.plan_import(str(folder))
        self.assertEqual(plan["issues"], [])
        self.assertEqual(plan["candidate_selection"]["basis"], "same_layout_other_build")
        self.assertEqual(plan["candidate_selection"]["model"], "Gyro-S")           # the closest by expert formats
        self.assertTrue(any("not exactly a build Strata has tested" in m for m in plan["messages"]))
        self.assertIn("unrecognized_or_unsupported_model_variant", models.plan_import(str(folder))["issues"])   # the real table has 1224 tensors

    # ---- T05: where it is, what is missing
    def test_the_folder_defaults_to_where_the_model_is(self):
        cat = self.catalog()
        self.assertEqual(cat.data_dir().resolve(), self.docs.resolve())
        self.assertEqual(cat.data_dir_for("gyro", "Gyro-S").resolve(), self.docs.resolve())
        self.assertEqual(cat.data_dir_choice()[1], "app")

    def test_an_own_setting_beats_the_apps_folder(self):
        mine = self.base / "mine"
        mine.mkdir()
        settings = self.settings_path
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(json.dumps({"data_dir": str(mine)}))
        self.assertEqual(self.catalog().data_dir().resolve(), mine.resolve())

    def test_installed_with_everything_there_is_already_downloaded(self):
        st = modelstate.assess(self.catalog(), "gyro", "Gyro-S", vision="cpu")
        self.assertTrue(st.installed)
        self.assertEqual(st.remaining_gb, 0.0)
        self.assertIn("Already downloaded", st.headline())
        self.assertIn("Nothing more to download", st.headline())

    def test_only_the_draft_layer_missing_says_so_with_its_size(self):
        (self.docs / "mtp" / "rt" / "experts.bin").unlink()
        cat = self.catalog()
        st = modelstate.assess(cat, "gyro", "Gyro-S", vision="cpu")
        self.assertEqual([p.kind for p in st.parts], ["mtp"])
        self.assertAlmostEqual(st.remaining_gb, discovery.MTP_FETCH_GB)
        text = st.headline()
        self.assertIn("Only the draft layer is missing (5.2 GB)", text)
        self.assertNotIn("download the model", text.lower())
        title, body = st.question()
        self.assertIn("draft layer", title)
        self.assertIn("not downloaded again", body)

    def test_draft_layer_and_picture_reader_are_itemised(self):
        (self.docs / "mtp" / "rt" / "experts.bin").unlink()
        self.mmproj.unlink()
        st = modelstate.assess(self.catalog(), "gyro", "Gyro-S", vision="cpu")
        self.assertEqual(sorted(p.kind for p in st.parts), ["mmproj", "mtp"])
        self.assertIn("the draft layer (5.2 GB) and the picture reader (0.9 GB)", st.headline())
        self.assertAlmostEqual(st.remaining_gb, 5.2 + 0.9)

    def test_pictures_off_never_asks_for_the_reader(self):
        self.mmproj.unlink()
        st = modelstate.assess(self.catalog(), "gyro", "Gyro-S", vision="off")
        self.assertEqual(st.parts, [])

    def test_weights_gone_downloads_the_model_again_and_says_so(self):
        self.gguf.unlink()
        st = modelstate.assess(self.catalog(), "gyro", "Gyro-S", vision="off")
        self.assertFalse(st.installed)
        self.assertGreater(st.remaining_gb, 50)

    def test_runner_counts_only_the_missing_helper_and_gives_setup_the_apps_folder(self):
        (self.docs / "mtp" / "rt" / "experts.bin").unlink()
        cat = self.catalog()
        r = Runner(S.LaunchSettings(family="gyro", model="Gyro-S", context=131072, kv="int8", vision="cpu"), catalog=cat)
        self.assertAlmostEqual(r.download_gb(), 5.2)
        argv = r.setup_args()
        self.assertEqual(Path(argv[argv.index("--data-dir") + 1]).resolve(), self.docs.resolve())

    def test_own_files_use_the_prepared_pack_and_folder(self):
        r = Runner(S.LaunchSettings(model_path=str(self.gguf.parent), family="gyro", model="Gyro-S", vision="off"), catalog=self.catalog())
        self.assertEqual(r.download_gb(), 0.0)
        self.assertEqual(Path(r.existing_data_dir()).resolve(), self.docs.resolve())


if __name__ == "__main__":
    unittest.main()
