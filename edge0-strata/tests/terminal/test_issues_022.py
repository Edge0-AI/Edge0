"""Small fixes from the 0.2.2 issue sweep, one test class per issue number."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Issue3298MtpDepth(unittest.TestCase):
    """Qwen3.6 passes `--mtp 2` (a draft depth); it is not a folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_doctor_does_not_look_for_a_folder_named_2(self):
        from strata_control import doctor
        engine = self.root / "engine"
        engine.write_text("x", encoding="utf-8")
        engine.chmod(0o700)
        tok = self.root / "tok"
        tok.mkdir()
        for n in ("vocab.json", "merges.txt", "token_type.json"):
            (tok / n).write_text("{}", encoding="utf-8")
        model = self.root / "m.gguf"
        model.write_bytes(b"x")
        cfg = self.root / "run.json"
        cfg.write_text(json.dumps({"exe": str(engine), "args": ["--native", str(model), "--mtp", "2", "--vision"],
                                   "cwd": str(self.root), "tokenizer": str(tok), "model_name": "q36", "port": 8095}),
                       encoding="utf-8")
        report = doctor.inspect(cfg)
        self.assertNotIn("mtp", [c["name"] for c in report["components"]])
        self.assertEqual([c["name"] for c in report["components"] if c["status"] not in ("ok",)
                          and c.get("required") and c["name"] in ("native", "mtp", "pack")], [])

    def test_doctor_still_checks_an_mtp_folder(self):
        from strata_control import doctor
        engine = self.root / "engine"
        engine.write_text("x", encoding="utf-8")
        engine.chmod(0o700)
        tok = self.root / "tok"
        tok.mkdir()
        for n in ("vocab.json", "merges.txt", "token_type.json"):
            (tok / n).write_text("{}", encoding="utf-8")
        cfg = self.root / "run.json"
        cfg.write_text(json.dumps({"exe": str(engine), "args": ["--mtp", str(self.root / "gone")],
                                   "cwd": str(self.root), "tokenizer": str(tok), "model_name": "x", "port": 8095}),
                       encoding="utf-8")
        report = doctor.inspect(cfg)
        mtp = [c for c in report["components"] if c["name"] == "mtp"]
        self.assertEqual(len(mtp), 1)
        self.assertEqual(mtp[0]["status"], "missing")

    def test_launch_files_present_ignores_a_depth_number(self):
        from strata_cli.launch.runner import files_present
        model = self.root / "m.gguf"
        model.write_bytes(b"x")
        self.assertEqual(files_present({"args": ["--native", str(model), "--mtp", "2"]}), [])
        self.assertEqual(files_present({"args": ["--mtp", str(self.root / "gone")]}), [str(self.root / "gone")])


class Issue3403MacBuildTools(unittest.TestCase):
    """The Mac setup compiles without setup.cmake_build(), so cmake and ninja must be installed for it too."""

    def test_missing_build_tools_are_installed_before_the_mac_compile(self):
        from unittest import mock
        import setup
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metal"))
        import setup_mac as M
        installed = {"cmake": None, "ninja": None}

        def find(name):
            return installed[name]

        def pip(packages, what):
            self.assertTrue(all(p.split("==")[0] in ("cmake", "ninja") for p in packages), packages)
            for p in packages:
                installed[p.split("==")[0]] = "/venv/bin/" + p.split("==")[0]

        with mock.patch.object(setup, "find_tool", side_effect=find), mock.patch.object(setup, "pip_install", side_effect=pip) as pi,                 mock.patch.object(setup, "_tool_version", return_value=(True, (3, 31, 0))):
            self.assertEqual(setup.ensure_build_tools(), ("/venv/bin/cmake", "/venv/bin/ninja"))
            self.assertEqual(pi.call_count, 1)
            setup.ensure_build_tools()                       # present now: nothing more is installed
            self.assertEqual(pi.call_count, 1)

    def test_both_mac_builds_ask_for_the_tools_first(self):
        import inspect
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metal"))
        import setup_mac as M
        for fn in (M.build_engine, M.build_vision):
            self.assertIn("S.ensure_build_tools()", inspect.getsource(fn), fn.__name__)


@unittest.skipUnless(sys.platform == "win32", "ACLs are Windows only")
class WindowsRuntimeFolderRefused(unittest.TestCase):
    """The owner's fresh Windows PC: `strata launch` stopped with "runtime directory must be owner-only" on a folder that was already
    there with another account in its ACL.  Fake SDDL strings for each way an ACL can look."""

    def setUp(self):
        from unittest import mock
        from strata_control import private_files as pf
        self.pf, self.mock = pf, mock
        self.me = "S-1-5-21-1-2-3-1001"
        patch = mock.patch.object(pf, "current_user_sid", return_value=self.me)
        patch.start()
        self.addCleanup(patch.stop)

    def offenders(self, sddl):
        return self.pf.unshared_offenders(sddl)

    def test_the_usual_acls_are_fine(self):
        for sddl in ("D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)",
                     "D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;S-1-5-21-1-2-3-1001)",
                     "D:PAI(A;OICI;FA;;;LA)(A;OICI;FA;;;SY)",               # the built-in Administrator's alias
                     "D:P(A;OICI;FA;;;S-1-5-18)(A;OICI;FA;;;S-1-5-32-544)(A;OICIIO;GA;;;CO)(A;OICI;FA;;;OW)"):
            self.assertEqual(self.offenders(sddl), [], sddl)

    def test_another_account_is_named(self):
        self.assertEqual(self.offenders("D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)(A;OICI;FR;;;BU)"), ["BU"])
        self.assertEqual(self.offenders("D:PAI(A;;FA;;;WD)(A;;FA;;;AU)"), ["WD", "AU"])
        self.assertEqual(self.offenders("D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1002)"), ["S-1-5-21-1-2-3-1002"])

    def test_an_inherit_only_entry_is_not_for_the_folder_itself(self):
        self.assertEqual(self.offenders("D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)(A;OICIIO;FR;;;BU)(A;CIIO;GR;;;WD)"), [])
        # the same entry without the IO flag does apply
        self.assertEqual(self.offenders("D:PAI(A;OICI;FR;;;BU)"), ["BU"])

    def test_deny_entries_do_not_make_it_reachable(self):
        self.assertEqual(self.offenders("D:PAI(D;;FA;;;BU)(A;OICI;FA;;;S-1-5-21-1-2-3-1001)"), [])

    def test_plain_names(self):
        self.assertEqual(self.pf.account_name("BU"), "Users")
        self.assertEqual(self.pf.account_name("WD"), "Everyone")
        self.assertEqual(self.pf.account_name("S-1-5-32-545"), "Users")

    def test_runtime_dir_repairs_a_folder_that_names_other_accounts(self):
        from strata_control import runtime
        pf = self.pf
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "runtime"
            base.mkdir()
            state = {"fixed": False}

            def sddl(_path):
                return "D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)" if state["fixed"] else \
                    "D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)(A;OICI;0x1200a9;;;BU)"

            def restrict(path, directory=None):
                state["fixed"] = True

            env = {"STRATA_RUNTIME_DIR": str(base)}
            with self.mock.patch.dict(os.environ, env), self.mock.patch.object(pf, "dacl_sddl", side_effect=sddl), \
                    self.mock.patch.object(pf, "restrict_to_current_user", side_effect=restrict) as r, \
                    self.mock.patch.object(pf, "private_mkdir"), self.mock.patch.object(runtime, "_private_mkdir"):
                self.assertEqual(runtime.runtime_dir(), base)
                self.assertEqual(r.call_count, 1)

    def test_runtime_dir_still_refuses_and_says_who_when_the_repair_fails(self):
        from strata_control import profiles, runtime
        pf = self.pf
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "runtime"
            base.mkdir()
            with self.mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(base)}), \
                    self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)(A;OICI;FR;;;BU)"), \
                    self.mock.patch.object(pf, "restrict_to_current_user", side_effect=OSError("denied")), \
                    self.mock.patch.object(runtime, "_private_mkdir"):
                with self.assertRaises(profiles.ProfileError) as cm:
                    runtime.runtime_dir()
            self.assertIn("Users", str(cm.exception))
            self.assertIn(str(base), str(cm.exception))
            self.assertIn("Properties", str(cm.exception))


@unittest.skipUnless(sys.platform == "win32", "ACLs are Windows only")
class WindowsRuntimeFolderMoreCases(unittest.TestCase):
    """The cases beyond the first repair (audit-refusals-022.md item 5): a data drive's inherited Users entries, an ACL that cannot be
    read, a folder made by an elevated run, a drive without permissions, and a message in plain words."""

    ME = "S-1-5-21-1-2-3-1001"

    def run_dir(self, *, sddl, restrict=None, repair=None, acls=True):
        from unittest import mock
        from strata_control import private_files as pf, runtime
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "runtime"
            base.mkdir()
            calls = {"restrict": 0, "repair": 0}

            def r(path, directory=None):
                calls["restrict"] += 1
                if restrict:
                    return restrict(calls)
            def rp(path, tree=True):
                calls["repair"] += 1
                return repair(calls) if repair else False
            with mock.patch.dict(os.environ, {"STRATA_RUNTIME_DIR": str(base)}), \
                    mock.patch.object(pf, "current_user_sid", return_value=self.ME), \
                    mock.patch.object(pf, "dacl_sddl", side_effect=sddl), \
                    mock.patch.object(pf, "restrict_to_current_user", side_effect=r), \
                    mock.patch.object(pf, "repair_access", side_effect=rp), \
                    mock.patch.object(pf, "volume_has_acls", return_value=acls), \
                    mock.patch.object(runtime, "_private_mkdir"), mock.patch.object(runtime, "_NO_ACL_WARNED", set()), \
                    mock.patch("sys.stderr", new_callable=lambda: __import__("io").StringIO()) as err:
                try:
                    return runtime.runtime_dir(), calls, None, err.getvalue()
                except Exception as exc:     # noqa: BLE001 - the test looks at the ProfileError text
                    return None, calls, exc, err.getvalue()

    def test_a_data_drive_folder_with_users_and_authenticated_users_is_repaired_once(self):
        state = {"fixed": False}
        def sddl(_):
            return f"D:PAI(A;OICI;FA;;;{self.ME})" if state["fixed"] else \
                f"D:PAI(A;OICI;FA;;;{self.ME})(A;OICI;0x1301bf;;;AU)(A;OICI;0x1200a9;;;BU)"
        base, calls, exc, _ = self.run_dir(sddl=sddl, restrict=lambda c: state.update(fixed=True))
        self.assertIsNone(exc)
        self.assertEqual(calls["restrict"], 1)

    def test_an_acl_that_cannot_be_read_is_repaired_then_refused_in_plain_words(self):
        def sddl(_):
            raise OSError("access denied")
        base, calls, exc, _ = self.run_dir(sddl=sddl)
        self.assertIsNotNone(exc)
        self.assertEqual((calls["restrict"], calls["repair"]), (1, 1))      # tried both, once each
        self.assertIn("cannot read the permissions", str(exc))
        self.assertNotIn("0700", str(exc))

    def test_a_folder_made_by_an_elevated_run_gets_ownership_taken_back(self):
        state = {"fixed": False}
        def sddl(_):
            if not state["fixed"]:
                raise OSError("access denied")
            return f"D:PAI(A;OICI;FA;;;{self.ME})"
        def restrict(calls):
            raise PermissionError("denied")
        base, calls, exc, _ = self.run_dir(sddl=sddl, restrict=restrict, repair=lambda c: state.update(fixed=True) or True)
        self.assertIsNone(exc)
        self.assertEqual((calls["restrict"], calls["repair"]), (1, 1))

    def test_a_drive_without_permissions_is_used_with_one_warning(self):
        def sddl(_):
            raise AssertionError("FAT has no ACL to read")
        base, calls, exc, err = self.run_dir(sddl=sddl, acls=False)
        self.assertIsNone(exc)
        self.assertEqual((calls["restrict"], calls["repair"]), (0, 0))
        self.assertIn("no file permissions", err)

    def test_the_refusal_names_the_accounts_and_never_says_mode_0700(self):
        sddl = lambda _: f"D:PAI(A;OICI;FA;;;{self.ME})(A;OICI;FR;;;AU)(A;OICI;FR;;;WD)"   # noqa: E731
        base, calls, exc, _ = self.run_dir(sddl=sddl)
        text = str(exc)
        self.assertIn("Authenticated Users", text)
        self.assertIn("Everyone", text)
        self.assertNotIn("0700", text)
        self.assertEqual(calls["restrict"], 1)

    def test_the_real_system_drive_has_permissions(self):
        from strata_control import private_files as pf
        self.assertTrue(pf.volume_has_acls(os.environ.get("SystemRoot", "C:\\Windows")))


class Issue3410TrashIsNotAnInstall(unittest.TestCase):
    def test_trash_folders_are_recognised(self):
        from strata_control.discovery import in_trash
        for p in ("/home/u/.local/share/Trash/files/strata", "/mnt/d/.Trash-1000/files/Strata", "/Users/u/.Trash/Strata",
                  "C:\\$Recycle.Bin\\S-1-5-21\\$R123\\Strata", "/home/u/.local/share/trash/files/strata/x"):
            self.assertTrue(in_trash(p), p)
        for p in ("/home/u/bin/strata", "/home/u/Trash-talk/strata", "/home/u/trash/strata", "/opt/Strata",
                  "C:\\Users\\u\\Strata"):
            self.assertFalse(in_trash(p), p)

    def test_remembered_installs_in_the_trash_are_not_listed(self):
        from strata_control import discovery
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "bin" / "strata"
            trash = Path(tmp) / ".local" / "share" / "Trash" / "files" / "strata"
            got = discovery._installs_in({"installs": [str(trash), str(good)]})
            self.assertEqual(got, [good])

    def test_setup_does_not_pass_a_trash_folder_on(self):
        import setup
        with tempfile.TemporaryDirectory() as tmp:
            trash = Path(tmp) / ".local" / "share" / "Trash" / "files" / "strata"
            trash.mkdir(parents=True)
            (trash / "setup.py").write_text("x")
            from unittest import mock
            nowhere = Path(tmp) / "none" / "none" / "home"            # no Strata* folders beside it on this PC
            with mock.patch.object(setup, "home", return_value=nowhere):
                self.assertEqual(setup.other_installs({"installs": [str(trash)]}), [])
                self.assertEqual(setup.other_installs({"installs": [str(trash)] + [str(trash.parent.parent)]}), [])


class Issue2971WebToolRowFitsPhones(unittest.TestCase):
    """A tool row widened the whole page on a phone (752 px page at a 412 px viewport, measured in Chrome). The three rules
    that keep it inside the viewport; a long tool name is cut with an ellipsis instead of pushing the page."""

    def test_the_rules_are_in_the_stylesheet(self):
        import re
        css = (Path(__file__).resolve().parents[2] / "serve" / "web" / "app.css").read_text(encoding="utf-8")

        def rule(selector):
            m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
            self.assertIsNotNone(m, selector)
            return m.group(1)

        self.assertIn("min-width: 0", rule(".view"))
        self.assertIn("overflow: hidden", rule(".tool-call > summary"))
        name = rule(".tool-call__name")
        for part in ("min-width: 0", "overflow: hidden", "text-overflow: ellipsis"):
            self.assertIn(part, name)


class Issue3188BenchStartsItsOwnServer(unittest.TestCase):
    """tools/strata_bench.py named `port` in the start-a-server branch without ever setting it."""

    def test_the_start_branch_runs_with_the_config_port(self):
        from unittest import mock
        from tools import strata_bench as sb
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "run.json"
            cfg.write_text(json.dumps({"port": 18999, "args": []}), encoding="utf-8")
            started = []
            with mock.patch.object(sb, "is_strata", return_value=False), \
                    mock.patch.object(sb, "try_json", return_value=None), \
                    mock.patch.object(sb, "start_server", side_effect=lambda c, p, g, log: started.append(p) or object()), \
                    mock.patch.object(sb, "wait_ready", return_value=False), \
                    mock.patch.object(sb, "stop_server"):
                out = io.StringIO()
                with redirect_stdout(out), mock.patch("sys.stderr", io.StringIO()):
                    code = sb.main(["--config", str(cfg)])
                self.assertEqual(code, 1)
                self.assertEqual(started, [18999])
                self.assertIn("on port 18999", out.getvalue())
                started.clear()
                with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    sb.main(["--config", str(cfg), "--port", "18777"])
                self.assertEqual(started, [18777])


@unittest.skipUnless(sys.platform == "win32", "SDDL aliases exist on Windows only")
class Issue3302AdministratorAlias(unittest.TestCase):
    """SDDL prints the built-in Administrator (RID 500) as `LA`; the owner-only checks must still accept it."""

    def setUp(self):
        from unittest import mock
        from strata_control import private_files as pf
        self.pf = pf
        self.mock = mock
        # the built-in Administrator's string SID on this machine: resolve the alias the same way the fix does
        import ctypes
        from ctypes import wintypes
        pointer = ctypes.c_void_p()
        pf._advapi.ConvertStringSidToSidW("LA", ctypes.byref(pointer))
        text = wintypes.LPWSTR()
        pf._advapi.ConvertSidToStringSidW(pointer.value, ctypes.byref(text))
        self.admin_sid = text.value
        self.assertTrue(self.admin_sid.endswith("-500"))

    def test_alias_matches_the_account_it_names(self):
        self.assertTrue(self.pf._same_sid("LA", self.admin_sid))
        self.assertTrue(self.pf._same_sid(self.admin_sid, self.admin_sid))
        self.assertFalse(self.pf._same_sid("LA", "S-1-5-21-1-2-3-1001"))
        self.assertFalse(self.pf._same_sid("WD", self.admin_sid))
        self.assertFalse(self.pf._same_sid("S-1-5-21-1-2-3-1001", self.admin_sid))

    def test_private_and_unshared_hold_for_the_built_in_administrator(self):
        pf = self.pf
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;LA)"):
            self.assertTrue(pf.is_private("x"))
            self.assertTrue(pf.is_unshared("x"))
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)"):
            self.assertFalse(pf.is_private("x"))
            self.assertFalse(pf.is_unshared("x"))
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;LA)(A;OICI;FR;;;WD)"):
            self.assertFalse(pf.is_private("x"))
            self.assertFalse(pf.is_unshared("x"))



class Issue3488IntegrityRetry(unittest.TestCase):
    """A hash that reads wrong once (file just unpacked, or held by a virus scanner) must not stick."""

    def test_second_read_decides(self):
        from unittest import mock
        from strata_control import engine_component as ec
        path = Path("x")
        with mock.patch.object(ec, "_sha256", side_effect=["bad", "good"]) as sha, mock.patch.object(ec.time, "sleep"):
            self.assertEqual(ec._hash_settled(path, "good"), "good")
            self.assertEqual(sha.call_count, 2)
        with mock.patch.object(ec, "_sha256", side_effect=[PermissionError(), "good"]), mock.patch.object(ec.time, "sleep"):
            self.assertEqual(ec._hash_settled(path, "good"), "good")
        with mock.patch.object(ec, "_sha256", return_value="bad"), mock.patch.object(ec.time, "sleep"):
            self.assertEqual(ec._hash_settled(path, "good"), "bad")      # still wrong after the retry: refuse



class Issue3501TrashOnPosix(unittest.TestCase):
    """A Windows Recycle Bin path read on Linux: Path does not split backslashes there."""

    def test_windows_path_on_a_posix_path_class(self):
        from pathlib import PurePosixPath
        from unittest import mock
        from strata_control import discovery
        with mock.patch.object(discovery, "Path", PurePosixPath):
            self.assertTrue(discovery.in_trash(r"C:\$Recycle.Bin\S-1-5-21\$R123\Strata"))
            self.assertTrue(discovery.in_trash("/mnt/d/$RECYCLE.BIN/S-1/Strata"))
            self.assertFalse(discovery.in_trash(r"C:\Users\a\Strata"))



class Issue3504MultiValueArgs(unittest.TestCase):
    """Advanced settings: a two-value option is two argv entries, `off` removes all of its values, an empty value is no bare flag."""

    ARGS = ["--max-context", "131072", "--control-vector-layer-range", "4", "44", "--cvec-dir", "per-layer",
            "--conversation-cache-mib", "8192"]

    def test_two_values_are_two_entries(self):
        from strata_control import tune_knobs
        for typed in ("4,44", "4 44", "4-44"):
            out = tune_knobs.apply_to_argv(self.ARGS, {"--control-vector-layer-range": typed})
            i = out.index("--control-vector-layer-range")
            self.assertEqual(out[i + 1:i + 3], ["4", "44"], typed)
            self.assertEqual(out.count("--control-vector-layer-range"), 1)

    def test_off_removes_the_option_and_every_value(self):
        from strata_cli.launch import settings as S
        from strata_control import tune_knobs
        extra = {"--control-vector-layer-range": "off", "--cvec-dir": "off"}
        a, _ = S.as_argv_changes(extra, tune_knobs.BOOL_FLAGS)
        out = tune_knobs.apply_to_argv(self.ARGS, a)
        self.assertEqual(out, ["--max-context", "131072", "--conversation-cache-mib", "8192"])

    def test_an_empty_value_is_no_bare_flag(self):
        from strata_cli.launch import settings as S
        from strata_control import tune_knobs
        a, _ = S.as_argv_changes({"--cvec-dir": "", "--kv-grow": ""}, tune_knobs.BOOL_FLAGS)
        self.assertEqual(a, {"--kv-grow": True})
        self.assertEqual(tune_knobs.apply_to_argv(self.ARGS, a)[:-1].count("--cvec-dir"), 1)

    def test_set_flag_and_the_table_keep_both_values(self):
        from strata_cli.launch import params
        from strata_control import user_args
        self.assertEqual(user_args.set_flag([], "--control-vector-layer-range", "4,44"),
                         ["--control-vector-layer-range", "4", "44"])
        self.assertEqual(params.argv_to_dict(self.ARGS)["--control-vector-layer-range"], "4 44")


if __name__ == "__main__":
    unittest.main()
