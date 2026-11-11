"""One models folder for the Terminal and the desktop app: the folder changed last wins, the other settings file is updated, nothing moves."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import setup
from strata_control import discovery
from tests.terminal.test_discovery import GuiLayout


class SharedModelsFolder(GuiLayout):
    def files(self):
        mine = self.settings_path
        mine.parent.mkdir(parents=True, exist_ok=True)
        return mine, discovery.app_settings_file()

    def put(self, file: Path, folder: Path, at: float) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"data_dir": str(folder), "data_dir_at": at, "installs": ["keep"]}))

    def read(self, file: Path) -> dict:
        return json.loads(file.read_text())

    def test_the_terminal_choice_made_last_wins_and_the_app_file_follows(self):
        mine, app = self.files()
        old, new = self.base / "on-c", self.base / "on-d"
        self.put(app, old, 1000)
        self.put(mine, new, 2000)
        self.assertEqual(discovery.data_dir(self.base / "default"), (new, "settings"))
        self.assertEqual(Path(self.read(app)["data_dir"]), new)
        self.assertEqual(self.read(app)["data_dir_at"], 2000)
        self.assertEqual(self.read(app)["installs"], ["keep"])        # merged, not replaced
        self.assertEqual(list(old.iterdir()), [])                     # nothing moved

    def test_the_app_choice_made_last_wins_and_the_terminal_file_follows(self):
        mine, app = self.files()
        old, new = self.base / "on-c", self.base / "on-d"
        self.put(mine, old, 1000)
        self.put(app, new, 3000)
        self.assertEqual(discovery.data_dir(self.base / "default")[0], new)
        self.assertEqual(Path(self.read(mine)["data_dir"]), new)
        self.assertEqual(self.read(mine)["data_dir_at"], 3000)
        self.assertEqual(self.catalog().data_dir().resolve(), new.resolve())

    def test_an_app_folder_on_a_disk_that_is_gone_is_not_adopted(self):
        mine, app = self.files()
        keep = self.base / "kept"
        self.put(mine, keep, 1000)
        gone = self.base / "unplugged"
        app.write_text(json.dumps({"data_dir": str(gone), "data_dir_at": 5000}))
        self.assertEqual(discovery.data_dir(self.base / "default"), (keep, "settings"))
        self.assertEqual(Path(self.read(app)["data_dir"]), gone)      # the app's file is left alone

    def test_the_same_folder_in_both_changes_nothing(self):
        mine, app = self.files()
        folder = self.base / "same"
        self.put(mine, folder, 1000)
        self.put(app, folder, 2000)
        before = app.read_bytes()
        self.assertEqual(discovery.data_dir(self.base / "default")[0], folder)
        self.assertEqual(app.read_bytes(), before)

    def test_files_without_a_time_use_the_file_time(self):
        mine, app = self.files()
        old, new = self.base / "a", self.base / "b"
        old.mkdir(), new.mkdir()
        app.write_text(json.dumps({"data_dir": str(old)}))
        os.utime(app, (time.time() - 500, time.time() - 500))
        mine.write_text(json.dumps({"data_dir": str(new)}))
        self.assertEqual(discovery.data_dir(self.base / "default")[0], new)

    def test_the_apps_own_processes_do_not_sync(self):
        mine, app = self.files()
        self.put(mine, self.base / "t", 9000)
        self.put(app, self.base / "a", 1000)
        os.environ["STRATA_SETTINGS_FILE"] = str(app)
        self.assertIsNone(discovery.sync_data_dir())
        self.assertEqual(Path(self.read(app)["data_dir"]), self.base / "a")

    def test_setup_stamps_a_changed_data_folder(self):
        mine, _ = self.files()
        setup.save_settings({"data_dir": str(self.base / "x")})
        first = self.read(mine)["data_dir_at"]
        self.assertGreater(first, 0)
        time.sleep(0.02)
        setup.save_settings({**self.read(mine), "installs": ["y"]})   # same folder: the time stays
        self.assertEqual(self.read(mine)["data_dir_at"], first)
