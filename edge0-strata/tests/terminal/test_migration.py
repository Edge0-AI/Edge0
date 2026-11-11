"""An install made by today's Strata is found and used in place by the terminal.

The fixture is today's layout: a Strata folder with strata-<model>.json run configs (absolute paths into it and into
the data folder), an engine/ folder, run-<model> scripts, a .venv, and a separate data folder (Strata-data) holding the
GGUF files, prepared packs and the MTP layer; the user's settings.json remembers the data folder and the install. The
terminal archive is a NEW folder. Nothing may be copied, moved, created or rewritten in the old layout, and the real
settings file and model folders are never touched: everything lives in a temporary directory, the settings file
location is redirected (conftest.py does the same for the whole session).
"""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup
from strata_cli.cli import main
from strata_control import models, profiles

from tests.terminal.test_models import gguf_fixture


def snapshot(base: Path) -> dict:
    out = {}
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base).as_posix()
        if path.is_dir():
            out[rel] = ("dir",)
        else:
            out[rel] = ("file", path.stat().st_size, path.stat().st_mtime_ns,
                        hashlib.sha256(path.read_bytes()).hexdigest())
    return out


class ExistingInstallIsFoundInPlace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.appdata = base / "appdata"
        self.old = base / "Strata"                       # today's install (a checkout or an unzipped release)
        self.data = base / "Strata-data"                 # the data folder settings.json remembers
        self.terminal = base / "Strata-terminal"         # the new archive, extracted elsewhere
        self.profiles = base / "profiles"                # profile storage is outside both layouts
        for d in (self.appdata, self.old, self.data, self.terminal):
            d.mkdir()
        env = mock.patch.dict(os.environ, {"APPDATA": str(self.appdata), "XDG_CONFIG_HOME": str(self.appdata),
                                           "STRATA_PROFILE_DIR": str(self.profiles),
                                           "STRATA_RUNTIME_DIR": str(base / "runtime"),
                                           # the desktop app's folders of whoever runs the tests are not part of this fixture
                                           "LOCALAPPDATA": str(base / "local"), "XDG_DATA_HOME": str(base / "localshare"),
                                           "STRATA_COMPONENTS_DIR": ""})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STRATA_INSTALL_DIR", None)

        exe_name = "strata.exe" if os.name == "nt" else "strata"
        (self.old / "engine").mkdir()
        (self.old / "engine" / exe_name).write_bytes(b"engine")
        (self.old / "engine" / exe_name).chmod(0o700)
        (self.old / "engine" / "BUILD.json").write_text(json.dumps({"version": "0.1.45", "source": "prebuilt"}))
        (self.old / "setup.py").write_text("# the installer of the old folder\n")
        (self.old / ".venv").mkdir()
        (self.old / ".venv" / "pyvenv.cfg").write_text("home = elsewhere\n")
        tok = self.data / "packs" / "tokenizer"
        tok.mkdir(parents=True)
        for name in ("vocab.json", "merges.txt", "token_type.json"):
            (tok / name).write_bytes(b"placeholder")
        self.models = []
        for tag, ctx in (("q2_0", 32768), ("iq3_xxs", 65536)):
            folder = self.data / "models" / tag.upper()
            folder.mkdir(parents=True)
            gguf = folder / f"Qwen3.8-Flash-Next-{tag}-00001-of-00001.gguf"
            gguf.write_bytes(gguf_fixture())
            pack = self.data / "packs" / tag
            pack.mkdir()
            (pack / "experts.bin").write_bytes(b"experts")
            mtp = self.data / "mtp" / tag
            mtp.mkdir(parents=True)
            cfg = self.old / f"strata-{tag}.json"
            cfg.write_text(json.dumps({
                "exe": str(self.old / "engine" / exe_name), "tokenizer": str(tok), "cwd": str(self.old),
                "args": ["--native", str(gguf), "--pack", str(pack), "--mtp", str(mtp), "--max-context", str(ctx)],
                "backend": "cuda", "model_name": tag.upper(), "port": 8080, "log": str(self.old / f"{tag}.log")}),
                encoding="utf-8")
            (self.old / f"strata-{tag}.shared-settings.json").write_text("{}")
            (self.old / (f"run-{tag}.bat" if os.name == "nt" else f"run-{tag}.sh")).write_text("run\n")
            self.models.append(cfg)
        (self.terminal / "VERSION").write_text("0.1.45\n")
        # never the real settings file or model folders: everything below lives in the temporary directory
        self.assertTrue(str(setup.settings_path()).startswith(str(base)), setup.settings_path())
        setup.settings_path().parent.mkdir(parents=True)
        setup.settings_path().write_text(json.dumps({"data_dir": str(self.data), "installs": [str(self.old)]}))
        self.before = snapshot(base)
        self.base = base
        root = mock.patch.object(models, "ROOT", self.terminal)
        root.start()
        self.addCleanup(root.stop)

    def cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([*args, "--output", "json"])
        return code, json.loads(out.getvalue())

    def assertNothingChanged(self):
        after = snapshot(self.base)
        created = sorted(set(after) - set(self.before))
        self.assertEqual([c for c in created if not c.startswith("profiles")], [])
        for rel, was in self.before.items():
            self.assertEqual(after.get(rel), was, rel)

    def test_the_terminal_finds_the_old_install_through_settings_json(self):
        self.assertEqual(models.resolve_install(), (self.old.resolve(), "remembered"))
        code, listed = self.cli("models", "list", "--installed")
        self.assertEqual(code, 0)
        self.assertEqual(listed["install_dir"], str(self.old.resolve()))
        self.assertEqual(listed["install_dir_source"], "remembered")
        self.assertEqual([m["id"] for m in listed["models"]], ["iq3_xxs", "q2_0"])
        self.assertEqual({m["component_status"] for m in listed["models"]}, {"present"})
        self.assertEqual({m["context_tokens"] for m in listed["models"]}, {32768, 65536})
        code, inspected = self.cli("models", "inspect", "q2_0")
        self.assertEqual(code, 0)
        self.assertTrue(inspected["components"]["ok"])
        self.assertNothingChanged()

    def test_a_profile_and_serve_plan_use_the_old_engine_and_models_where_they_are(self):
        found = models.install_dir()
        config = found / "strata-q2_0.json"
        profiles.create("migrated", str(config))
        code, planned = self.cli("serve", "--profile", "migrated", "--plan")
        self.assertEqual(code, 0)
        plan = planned["plan"]
        self.assertEqual(Path(plan["source_config"]), config.resolve())
        self.assertEqual(Path(plan["engine_exe"]).parent, self.old / "engine")
        argv = plan["engine_argv"]
        native = argv[argv.index("--native") + 1]
        self.assertEqual(Path(native).parent.parent.parent, self.data)           # Strata-data/models/Q2_0/<gguf>
        self.assertNothingChanged()

    def test_a_folder_with_its_own_configs_and_an_explicit_dir_win_over_the_remembered_install(self):
        (self.terminal / "strata-local.json").write_bytes((self.old / "strata-q2_0.json").read_bytes())
        self.assertEqual(models.resolve_install(), (self.terminal, "here"))
        (self.terminal / "strata-local.json").unlink()
        self.assertEqual(models.resolve_install(str(self.old)), (self.old.resolve(), "explicit"))
        with mock.patch.dict(os.environ, {"STRATA_INSTALL_DIR": str(self.old)}):
            self.assertEqual(models.resolve_install()[1], "explicit")

    def test_nothing_remembered_means_this_folder_and_no_error(self):
        # nothing found anywhere: the folder a new model would be installed into (the desktop app's, which the Terminal shares), no error
        from strata_control import discovery
        default = (discovery.app_install_dir(), "default")
        setup.settings_path().unlink()
        self.assertEqual(models.resolve_install(), default)
        setup.settings_path().write_text("not json")
        self.assertEqual(models.resolve_install(), default)
        setup.settings_path().write_text(json.dumps({"installs": ["relative/path", 7, str(self.base / "gone")]}))
        self.assertEqual(models.resolve_install(), default)
        self.assertEqual(models.list_installed()["models"], [])

    def test_the_settings_path_is_the_one_setup_writes(self):
        self.assertEqual(models.settings_path(), setup.settings_path())

    def test_a_plain_start_from_the_old_folder_uses_the_remembered_data_folder_and_moves_nothing(self):
        started = []
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", self.old), mock.patch.object(sys, "argv", ["setup.py", "--yes"]), \
                mock.patch.object(setup, "gpus", lambda: []), \
                mock.patch.object(setup.subprocess, "call", lambda cmd, *a, **k: started.append(cmd) or 0), \
                contextlib.redirect_stdout(out):
            self.assertEqual(setup.main(), 0, out.getvalue())
        self.assertEqual(len(started), 1)
        self.assertIn("server.py", " ".join(map(str, started[0])))
        self.assertEqual(setup.data_folder_in_place(), (self.data, []))
        # starting touches the most recently used config (its mtime) and nothing else
        after = snapshot(self.base)
        changed = [rel for rel in self.before if after.get(rel) != self.before[rel]]
        self.assertEqual([c for c in changed if not c.startswith("Strata/strata-")], [])
        self.assertEqual(sorted(set(after) - set(self.before)), [])

    def test_setup_from_a_new_copy_reuses_the_data_folder_and_moves_no_model_file(self):
        """The explicit setup path (which does gather files) finds the model files already in the data folder."""
        new_copy = self.base / "Strata-new"
        new_copy.mkdir()
        (new_copy / "setup.py").write_text("# a new copy\n")
        models_before = {k: v for k, v in self.before.items() if k.startswith("Strata-data/")}
        with mock.patch.object(setup, "ROOT", new_copy), contextlib.redirect_stdout(io.StringIO()):
            data, elsewhere = setup.data_folder(None)
        self.assertTrue(data.samefile(self.data))
        self.assertEqual(elsewhere, [])
        after = snapshot(self.base)
        self.assertEqual({k: v for k, v in after.items() if k.startswith("Strata-data/")}, models_before)
        self.assertEqual({k: v for k, v in after.items() if k.startswith("Strata/")},
                         {k: v for k, v in self.before.items() if k.startswith("Strata/")})
        saved = json.loads(setup.settings_path().read_text())
        self.assertTrue(Path(saved["data_dir"]).samefile(self.data))
        self.assertIn(str(self.old), saved["installs"])                    # the old install stays remembered


if __name__ == "__main__":
    unittest.main()
