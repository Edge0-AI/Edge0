r"""The Windows installer must never touch the user's data. The app lives in %LOCALAPPDATA%\Programs\Strata; %LOCALAPPDATA%\Strata
holds engines, model run configs, tuning. A build that installed the app into the data folder is migrated by removing only its app
files. Exercised with makensis on a temp layout (the template's own migration macro, extracted and run); skipped without NSIS."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "apps" / "desktop" / "src-tauri" / "nsis" / "installer.nsi"
MAKENSIS = Path(os.environ.get("LOCALAPPDATA", "")) / "tauri" / "NSIS" / "makensis.exe"
DATA = ["components/engine/0.1.45/strata.exe", "install/strata-iq3_s.json", "procs/p.json", "runtime/r.json", "time-skip/t.json",
        "presets/p.json"]
APP = ["strata-desktop.exe", "uninstall.exe", "host/server/index.mjs", "node/node.exe", "control/python/python.exe"]
BS = "\\"


def touch(base: Path, names):
    for name in names:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


class TemplateTests(unittest.TestCase):
    text = TEMPLATE.read_text(encoding="utf-8")

    def test_the_default_folder_is_programs_strata_and_the_data_folder_is_never_remembered_as_the_app(self):
        self.assertIn(r'StrCpy $INSTDIR "$LOCALAPPDATA\Programs\${PRODUCTNAME}"', self.text)
        self.assertIn(r'StrCmp $4 "$LOCALAPPDATA\${PRODUCTNAME}" restore_done 0', self.text)
        self.assertIn("StrIsUnder", self.text)          # a remembered folder under Temp is ignored too

    def test_the_install_folder_page_is_shown_except_in_an_update(self):
        # #3186 #3190: the person can put the app on another drive; the app's own updater (passive) and silent installs keep the folder
        # the earlier install wrote to the registry, so an update never asks again and never moves the app
        text = self.text.replace("\r\n", "\n")
        match = re.search(r"!define MUI_PAGE_CUSTOMFUNCTION_PRE (\w+)\n!insertmacro MUI_PAGE_DIRECTORY", text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "SkipIfPassive")
        self.assertIn('WriteRegStr SHCTX "${MANUPRODUCTKEY}" "" $INSTDIR', text)
        self.assertIn("Call RestorePreviousInstallLocation", text)
        conf = (ROOT / "apps" / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
        self.assertIn('"installMode": "currentUser"', conf)

    def test_the_uninstaller_removes_files_one_by_one_and_only_empty_folders(self):
        uninstall = self.text[self.text.index("Section Uninstall"):]
        self.assertNotRegex(uninstall, r'RMDir /r "\$INSTDIR"')
        self.assertNotIn(r'RmDir /r "$LOCALAPPDATA\${PRODUCTNAME}"', self.text)
        self.assertIn(r'RmDir /r "$LOCALAPPDATA\${BUNDLEID}"', self.text)


@unittest.skipUnless(MAKENSIS.is_file(), "NSIS (the Tauri bundler's copy) is not installed here")
class MigrationTests(unittest.TestCase):
    def run_nsis(self, body: str, work: Path):
        template = TEMPLATE.read_text(encoding="utf-8")
        block = chr(10).join(re.search(rf"; STRATA-{name}-BEGIN.*?; STRATA-{name}-END", template, re.S).group(0) for name in ("MIGRATE", "STOP", "CLEAN"))
        script = work / "t.nsi"
        script.write_text('Unicode true\nOutFile "t.exe"\nRequestExecutionLevel user\nSilentInstall silent\n' + block + "\nSection\n" + body + "\nSectionEnd\n", encoding="utf-8")
        done = subprocess.run([str(MAKENSIS), "/V1", str(script)], cwd=work, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        subprocess.run([str(work / "t.exe")], cwd=work, check=True, timeout=60)

    def layout(self, base: Path) -> Path:
        old = base / "Strata"
        touch(old, DATA + APP)
        return old

    def test_migration_removes_only_the_old_app_files_and_keeps_every_bit_of_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = self.layout(base)
            new = base / "Programs" / "Strata"
            self.run_nsis(f'!insertmacro StrataMigrateOldInstall "{old}" "{new}"', base)
            for name in DATA:
                self.assertTrue((old / name).is_file(), f"data kept: {name}")
            for name in APP:
                self.assertFalse((old / name).exists(), f"app file removed: {name}")
            for folder in ("host", "node", "control"):
                self.assertFalse((old / folder).exists())

    def test_migration_does_nothing_when_old_and_new_are_the_same_or_there_is_no_old_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = self.layout(base)
            self.run_nsis(f'!insertmacro StrataMigrateOldInstall "{old}" "{old}"', base)
            for name in DATA + APP:
                self.assertTrue((old / name).is_file(), name)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            touch(base / "Strata", DATA)                   # data only: no strata-desktop.exe
            self.run_nsis(f'!insertmacro StrataMigrateOldInstall "{base / "Strata"}" "{base / "Programs" / "Strata"}"', base)
            for name in DATA:
                self.assertTrue((base / "Strata" / name).is_file(), name)

    def test_an_uninstall_of_the_template_kind_leaves_data_in_a_shared_folder(self):
        """The uninstaller deletes named files and uses non-recursive RMDir: in a folder that also holds data (a legacy install) the
        data and so the folder stay."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = self.layout(base)
            lines = [f'Delete "{old}{BS}{n.replace("/", BS)}"' for n in APP]
            lines += [f'RMDir "{old}{BS}{d}"' for d in ("host" + BS + "server", "host", "node", "control" + BS + "python", "control")]
            lines += [f'RMDir "{old}"']
            self.run_nsis("\n".join(lines), base)
            for name in DATA:
                self.assertTrue((old / name).is_file(), name)
            self.assertFalse((old / "strata-desktop.exe").exists())


    def test_the_uninstaller_stops_only_the_processes_started_from_the_install_folder(self):
        ping = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "PING.EXE"
        if not ping.is_file():
            self.skipTest("no PING.EXE to stand in for a running Strata process")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inst, other = base / "Programs" / "Strata", base / "Elsewhere"
            for folder in (inst / "node", other):
                folder.mkdir(parents=True)
            mine = inst / "node" / "node.exe"
            theirs = other / "node.exe"
            for target in (mine, theirs):
                target.write_bytes(ping.read_bytes())
            procs = [subprocess.Popen([str(p), "-t", "127.0.0.1"], stdout=subprocess.DEVNULL) for p in (mine, theirs)]
            try:
                self.run_nsis(f'!insertmacro StrataStopOwnProcesses "{inst}"', base)
                procs[0].wait(timeout=20)
                self.assertIsNotNone(procs[0].poll(), "the process from the install folder was stopped")
                self.assertIsNone(procs[1].poll(), "a process of the same name elsewhere is left running")
            finally:
                for proc in procs:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()


    def test_files_the_app_made_in_its_own_folders_are_removed_and_the_data_beside_them_is_not(self):
        """Strix, standard user: after the app had run, Python bytecode caches under control\\ kept the program folder alive."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inst = base / "Programs" / "Strata"
            touch(inst, ["strata-desktop.exe", "control/python/python.exe", "control/python/Lib/__pycache__/argparse.cpython-312.pyc",
                         "control/serve/__pycache__/server.cpython-312.pyc", "host/server/index.mjs", "host/logs/made-at-run.txt", "node/node.exe",
                         "harness/BUNDLE.json", "components/engine/keep.txt", "runtime/keep.json"])
            self.run_nsis(f'!insertmacro StrataRemoveAppFolders "{inst}"', base)
            for folder in ("control", "host", "node", "harness"):
                self.assertFalse((inst / folder).exists(), folder)
            for name in ("components/engine/keep.txt", "runtime/keep.json", "strata-desktop.exe"):
                self.assertTrue((inst / name).is_file(), f"kept: {name}")

    def test_the_uninstaller_clears_those_folders_before_the_last_rmdir(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        uninstall = text[text.index("Section Uninstall"):]
        self.assertIn('!insertmacro StrataRemoveAppFolders "$INSTDIR"', uninstall)
        self.assertLess(uninstall.index("StrataRemoveAppFolders"), uninstall.index('RMDir "$INSTDIR"'))
        self.assertNotRegex(uninstall, r'RMDir /r "\$INSTDIR"')


class StopOrderTests(unittest.TestCase):
    def test_the_uninstaller_stops_them_before_it_removes_files(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        uninstall = text[text.index("Section Uninstall"):]
        self.assertLess(uninstall.index("StrataStopOwnProcesses"), uninstall.index(r'Delete "$INSTDIR\${MAINBINARYNAME}.exe"'))


if __name__ == "__main__":
    unittest.main()
