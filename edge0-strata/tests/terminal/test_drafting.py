"""Which drafter a model uses (MTP, DFlash2 or none): the suggestion table, the Terminal flag, the engine arguments, the download and the
fallback words. strata_control/drafting.py is the one place; the Terminal and the app read it."""

from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import acquisition, drafting, drafting_fetch
from tests.terminal.launch_support import run_cli

ROOT = Path(__file__).resolve().parents[2]
GYRO = "qwen3.8-flash-next-gyro-gyro-s"


def serving_dflash2():
    """The engine of this release runs DFlash2 only in its CLI run (drafting.ENGINE_SERVES_DFLASH2): the tests turn the switch on."""
    return mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", True)


class SuggestionTable(unittest.TestCase):
    def test_every_row_is_a_real_drafter_with_evidence(self):
        for klass, row in drafting.SUGGESTED.items():
            self.assertIn(row.drafter, ("mtp", "dflash2"), klass)
            self.assertTrue(row.evidence.strip(), f"{klass}: say what was measured")

    def test_auto_is_dflash2_only_where_it_won_on_every_tested_prompt(self):
        # The owner's rule (2026-10-22): "suggest DFlash2 where the win is 100%". Today no card class qualifies.
        for klass in drafting.SUGGESTED:
            self.assertEqual(drafting.suggest(klass), "mtp", klass)
        for klass in ("strix_halo", "blackwell", "rdna4", "ampere", "other", "never-heard-of-it"):
            self.assertEqual(drafting.suggest(klass), "mtp")
        # a row that says DFlash2 but not "faster everywhere" is still the MTP; only both together switch Auto
        with mock.patch.dict(drafting.SUGGESTED, {"blackwell": drafting.Suggestion("dflash2", False, "wins on code only")}):
            self.assertEqual(drafting.suggest("blackwell"), "mtp")
        with mock.patch.dict(drafting.SUGGESTED, {"blackwell": drafting.Suggestion("dflash2", True, "faster on chat, code, json and tool_call")}):
            self.assertEqual(drafting.suggest("blackwell"), "dflash2")
            self.assertEqual(drafting.suggest("strix_halo"), "mtp", "the other classes are unchanged")

    def test_the_measured_losses_are_written_down(self):
        text = {k: v.evidence for k, v in drafting.SUGGESTED.items()}
        self.assertIn("-6%", text["strix_halo"])
        self.assertIn("tool_call", text["strix_halo"])
        self.assertIn("-1%", text["blackwell"])
        self.assertIn("chat", text["blackwell"])
        self.assertIn("every prompt", text["rdna4"])
        self.assertIn("every prompt", text["ampere"])

    def test_cards_are_classified_by_their_architecture(self):
        self.assertEqual(drafting.card_class(["sm_120"]), "blackwell")
        self.assertEqual(drafting.card_class(["gfx1151"]), "strix_halo")
        self.assertEqual(drafting.card_class(["gfx1201"]), "rdna4")
        self.assertEqual(drafting.card_class(["sm_86"]), "ampere")
        self.assertEqual(drafting.card_class([]), "other")
        self.assertEqual(drafting.card_class(["sm_61"]), "other")
        self.assertEqual(drafting.card_class(["sm_61", "sm_120"]), "blackwell", "the first card the table knows")
        self.assertEqual(drafting.card_class("SM_120"), "blackwell")

    def test_both_sides_know_the_same_modes(self):
        self.assertEqual(drafting.MODES, S.DRAFTING_MODES)
        self.assertEqual(drafting.normalize(None), "auto")
        self.assertEqual(drafting.normalize("DFlash2"), "dflash2")
        self.assertEqual(drafting.normalize("fastest"), "auto", "an unknown value is Auto, never an error")


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def ready(self):
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))

    def test_auto_and_mtp_run_the_mtp_with_no_extra_flags(self):
        with serving_dflash2():
            for mode in ("auto", "mtp", None):
                d = drafting.decide(mode, klass="blackwell", model_name=GYRO, data_dir=self.data)
                self.assertEqual((d.drafter, d.flags, d.note), ("mtp", (), ""), mode)

    def test_off_is_off(self):
        self.assertEqual(drafting.decide("off", klass="blackwell", model_name=GYRO).drafter, "off")

    def test_dflash2_needs_the_drafter_file_and_gets_the_engine_flags_when_it_is_there(self):
        with serving_dflash2():
            gone = drafting.decide("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data)
            self.assertEqual((gone.drafter, gone.note), ("mtp", drafting.NOT_DOWNLOADED))
            self.assertTrue(drafting.needs_pack("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data))
            self.ready()
            d = drafting.decide("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data)
            self.assertEqual(d.drafter, "dflash2")
            self.assertEqual(d.flags, ("--draft", "dflash2", "--draft-model", str(drafting.pack_path(self.data))))
            self.assertFalse(drafting.needs_pack("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data))

    def test_dflash2_is_for_qwen38_flash_next_only(self):
        with serving_dflash2():
            self.ready()
            for name in ("qwen3.6-35b-a3b-q4", "qwen3.8-flash-next-coder-iq3_xxs", "qwen3.8-flash-next-swift-iq2_xs", None):
                d = drafting.decide("dflash2", klass="blackwell", model_name=name, data_dir=self.data)
                self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SUPPORTED), name)
                self.assertFalse(drafting.needs_pack("dflash2", klass="blackwell", model_name=name, data_dir=self.data))
            for name in (GYRO, "qwen3.8-flash-next-q2_0", "qwen3.8-flash-next-iq3_xxs"):
                self.assertEqual(drafting.decide("dflash2", klass="blackwell", model_name=name, data_dir=self.data).drafter, "dflash2", name)
            self.assertTrue(drafting.supported_model(family="qwen") and drafting.supported_model(family="gyro"))
            self.assertFalse(drafting.supported_model(family="coder") or drafting.supported_model(family="swift") or drafting.supported_model(family="nvfp4"))

    def test_while_the_engine_cannot_serve_it_the_choice_says_so_and_nothing_is_downloaded(self):
        self.assertFalse(drafting.ENGINE_SERVES_DFLASH2, "src/program/generate.cpp: --draft dflash2 is off with --serve in this release")
        d = drafting.decide("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SERVED))
        self.assertFalse(drafting.needs_pack("dflash2", klass="blackwell", model_name=GYRO, data_dir=self.data))
        self.assertEqual(drafting.decide("auto", klass="blackwell", model_name=GYRO, data_dir=self.data).note, "")

    def test_auto_follows_the_table(self):
        with serving_dflash2(), mock.patch.dict(drafting.SUGGESTED, {"blackwell": drafting.Suggestion("dflash2", True, "wins everywhere")}):
            self.ready()
            self.assertEqual(drafting.decide("auto", klass="blackwell", model_name=GYRO, data_dir=self.data).drafter, "dflash2")
            self.assertEqual(drafting.decide("auto", klass="strix_halo", model_name=GYRO, data_dir=self.data).drafter, "mtp")
            self.assertEqual(drafting.decide("auto", klass="blackwell", model_name="qwen3.6-35b-a3b-q4", data_dir=self.data).note, "",
                             "Auto never nags about a model DFlash2 is not for")


class EngineArguments(unittest.TestCase):
    ARGS = ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", "/d/mtp/rt", "--spec", "4", "--spec-min-p", "0.5"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def cfg(self, **extra):
        return {"model_name": GYRO, "args": list(self.ARGS), **extra}

    def test_a_config_nobody_touched_keeps_its_arguments_exactly(self):
        c = self.cfg()
        drafting.apply(c, klass="blackwell", data_dir=self.data)
        self.assertEqual(c["args"], self.ARGS)
        self.assertNotIn("mtp", c)

    def test_dflash2_adds_the_flags_and_replaces_older_ones(self):
        pack = drafting.pack_path(self.data)
        pack.parent.mkdir(parents=True)
        pack.write_bytes(b"x" * (2 << 20))
        with serving_dflash2():
            c = self.cfg(drafting="dflash2", args=[*self.ARGS, "--draft", "mtp", "--draft-model", "/old.pack"])
            d = drafting.apply(c, klass="blackwell", data_dir=self.data)
        self.assertEqual(d.drafter, "dflash2")
        self.assertEqual(c["args"], [*self.ARGS, "--draft", "dflash2", "--draft-model", str(pack)])
        self.assertEqual(c["args"].count("--draft"), 1)
        self.assertIn("--mtp", c["args"], "the MTP stays as the fallback the engine names")

    def test_mtp_removes_a_drafter_that_was_written_by_hand(self):
        c = self.cfg(drafting="mtp", args=[*self.ARGS, "--draft", "dflash2", "--draft-model", "/p"])
        drafting.apply(c, klass="blackwell", data_dir=self.data)
        self.assertEqual(c["args"], self.ARGS)

    def test_off_is_the_draft_layers_own_switch(self):
        c = self.cfg(drafting="off")
        d = drafting.apply(c, klass="blackwell", data_dir=self.data)
        self.assertEqual(d.drafter, "off")
        self.assertIs(c["mtp"], False)
        for flag in ("--mtp", "--spec-min-p", "--batch-mtp", "--draft"):
            self.assertNotIn(flag, c["args"])
        self.assertEqual(c["args"][c["args"].index("--spec") + 1], "4", "prompt lookup keeps its windows")

    def test_a_choice_that_cannot_be_applied_starts_the_mtp_and_says_why(self):
        c = self.cfg(drafting="dflash2", model_name="qwen3.8-flash-next-coder-iq3_xxs")
        with serving_dflash2():
            d = drafting.apply(c, klass="blackwell", data_dir=self.data)
        self.assertEqual((d.drafter, d.note), ("mtp", drafting.NOT_SUPPORTED))
        self.assertEqual(c["args"], self.ARGS)

    def test_the_server_applies_it_at_every_start(self):
        from serve.server import engine_args
        pack = drafting.pack_path(Path("/d"))
        with mock.patch("serve.server.drafting_card_class", return_value="blackwell"), serving_dflash2(), \
                mock.patch.object(drafting, "pack_ready", return_value=True), mock.patch.object(drafting, "data_dir_of", return_value=Path("/d")):
            args = engine_args(self.cfg(drafting="dflash2"))
            self.assertEqual(args[-4:], ["--draft", "dflash2", "--draft-model", str(pack)])
        with mock.patch("serve.server.drafting_card_class", return_value="blackwell"):
            self.assertEqual(engine_args(self.cfg()), self.ARGS, "no choice: nothing changes")
            off = engine_args(self.cfg(drafting="off"))
            self.assertNotIn("--mtp", off)
            self.assertEqual(drafting.take_note(), "")
            engine_args(self.cfg(drafting="dflash2"))
            self.assertEqual(drafting.take_note(), drafting.NOT_SERVED, "the engine that starts next is told why")

    def test_data_folder_of_an_argument_list(self):
        self.assertEqual(drafting.data_dir_of(["--pack", "/d/packs/x"]), Path("/d"))
        self.assertEqual(drafting.data_dir_of(["--mtp", "/d/mtp/rt"]), Path("/d"))
        self.assertIsNone(drafting.data_dir_of(["--mtp", "4"]))
        self.assertIsNone(drafting.data_dir_of([]))


class WhatTheEngineSaid(unittest.TestCase):
    ARGS = ["--mtp", "/m", "--draft", "dflash2", "--draft-model", "/p"]

    def test_no_room_in_graphics_memory(self):
        log = ("strata generate: --draft dflash2: could not allocate the drafter's buffers. DFlash2 needs about 1.2 GiB of VRAM (the pack 0.76 GiB, "
               "its buffers and K/V ring 0.25 GiB, a draft head 0.2 GiB); start with --vram-reserve-mib 1900 to keep it. The MTP drafts instead.\n")
        info = drafting.engine_info(self.ARGS, log)
        self.assertEqual(info, {"drafting": "mtp", "drafting_note": "Drafting: MTP (DFlash2 didn't fit)"})

    def test_a_mode_this_build_does_not_run(self):
        log = "strata generate: --draft dflash2 is off: it is for one card and the CLI run (not --serve, --batch or a layer split) in this build; the MTP drafts instead\n"
        self.assertEqual(drafting.engine_info(self.ARGS, log)["drafting_note"], "Drafting: MTP (DFlash2 isn't available in this mode)")

    def test_it_ran(self):
        log = "strata generate: DFlash2 drafter on (/p, graphs captured in 3 ms), the MTP layer stays as the fallback\n"
        self.assertEqual(drafting.engine_info(self.ARGS, log), {"drafting": "dflash2", "drafting_note": ""})
        self.assertEqual(drafting.fallback_note(log), "")

    def test_choices_that_never_reached_the_engine_keep_their_own_words(self):
        self.assertEqual(drafting.engine_info(["--mtp", "/m"], "", drafting.NOT_SUPPORTED)["drafting_note"],
                         "Drafting: MTP (DFlash2 is for Qwen3.8-Flash-Next only)")
        self.assertEqual(drafting.engine_info(["--mtp", "/m"]), {"drafting": "mtp", "drafting_note": ""})
        self.assertEqual(drafting.engine_info(["--spec", "4"]), {"drafting": "off", "drafting_note": ""})

    def test_summary_line(self):
        self.assertEqual(drafting.summary_line(drafting.Decision("auto", "mtp")), "Drafting: MTP")
        self.assertEqual(drafting.summary_line(drafting.Decision("dflash2", "dflash2")), "Drafting: DFlash2")
        self.assertEqual(drafting.summary_line(drafting.Decision("off", "off")), "Drafting: Off")
        self.assertEqual(drafting.summary_line(drafting.Decision("dflash2", "mtp", drafting.NOT_DOWNLOADED)),
                         "Drafting: MTP (the DFlash2 drafter isn't downloaded yet)")


class WordsMatchTheApp(unittest.TestCase):
    def test_every_line_the_engine_start_can_say_is_in_the_apps_catalog_in_all_eight_languages(self):
        folder = ROOT / "apps" / "host" / "src" / "locales"
        catalogs = {l: json.loads((folder / f"{l}.json").read_text(encoding="utf-8")) for l in ("en", "de", "es", "fr", "ja", "ko", "pt", "ru", "zh")}
        ids = {v: k for k, v in catalogs["en"].items()}
        self.assertEqual(len(drafting.REASONS), 4)
        for reason in drafting.REASONS:
            line = drafting.mtp_line(reason)
            self.assertIn(line, ids, line)
            for lang, catalog in catalogs.items():
                self.assertTrue(catalog[ids[line]].strip(), f"{line} in {lang}")
        # and what the engine's log makes of them
        self.assertEqual(drafting.fallback_note("--draft dflash2: out of memory"), drafting.mtp_line(drafting.DIDNT_FIT))
        self.assertEqual(drafting.fallback_note("--draft dflash2 is off: it is for one card and the CLI run"), drafting.mtp_line(drafting.NOT_HERE))


class TerminalFlag(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--pack", "/d/packs/x", "--mtp", "/d/mtp/rt"], "port": 8080, "model_name": GYRO}

    def test_the_flag_is_listed_planned_copyable_and_saved_in_the_launch_config(self):
        code, out, _ = run_cli(["launch", "--help"])
        self.assertIn("--drafting", out)
        self.assertIn("auto|mtp|dflash2|off", out)
        cfg = derive_config(self.BASE, S.LaunchSettings(drafting="dflash2"), bool_flags=frozenset())
        self.assertEqual(cfg["drafting"], "dflash2")
        self.assertNotIn("drafting", derive_config(self.BASE, S.LaunchSettings(), bool_flags=frozenset()), "Auto writes nothing: nobody chose")
        self.assertIn("--drafting dflash2", S.flags_text(S.LaunchSettings(drafting="dflash2")))
        self.assertNotIn("--drafting", S.flags_text(S.LaunchSettings()))
        code, out, err = run_cli(["launch", "--plan", "--drafting", "fastest"])
        self.assertNotEqual(code, 0)

    def test_a_saved_preset_keeps_the_choice(self):
        s = S.LaunchSettings.from_dict(S.LaunchSettings(drafting="mtp").to_dict())
        self.assertEqual(s.drafting, "mtp")
        self.assertIsNone(S.LaunchSettings.from_dict({}).drafting)


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    hits: list = []

    def do_GET(self):
        body = self.files.get(self.path.rsplit("/", 1)[-1])
        self.hits.append(self.path)
        if body is None:
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])

    def log_message(self, *_):
        pass


class FetchAndPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location("t_pack", ROOT / "tools" / "test_dflash2_pack.py")
        cls.synth = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.synth)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.synth.write_safetensors(self.dir / "m.safetensors", self.synth.tensors())
        (self.dir / "c.json").write_text(json.dumps(self.synth.CFG))
        import hashlib
        self.blobs = {"config.json": (self.dir / "c.json").read_bytes(), "model.safetensors": (self.dir / "m.safetensors").read_bytes()}
        self.files = tuple((n, len(b), hashlib.sha256(b).hexdigest()) for n, b in self.blobs.items())
        _Handler.files, _Handler.hits = dict(self.blobs), []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.data = self.dir / "data"
        self.steps: list = []
        # the packed test file is a few hundred kB, not the real 0.76 GB: "ready" asks for more than 1 MiB by default
        patcher = mock.patch.object(drafting, "pack_ready", lambda d: bool(d) and drafting.pack_path(d).is_file())
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, **kw):
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_ALLOW_HTTP": "1"}):
            return drafting_fetch.ensure(self.data, progress=lambda *a: self.steps.append(a), base_url=self.base, files=self.files, **kw)

    def test_the_pinned_files_are_the_repositorys_own(self):
        self.assertEqual(drafting.REVISION, "a22a1744e85c17364dcc54a6a420af7135d4c7f6")
        self.assertTrue(drafting.BASE_URL.endswith(f"/{drafting.REVISION}/"), "a revision, never a branch name")
        names = {n: (size, sha) for n, size, sha in drafting.FILES}
        self.assertEqual(names["model.safetensors"], (1206140008, "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"))
        self.assertEqual(len(names["config.json"][1]), 64)
        self.assertEqual(drafting.LICENSE, "Apache-2.0")

    def test_it_downloads_checks_packs_and_removes_the_source(self):
        out = self.fetch()
        self.assertEqual(out, drafting.pack_path(self.data))
        self.assertEqual(out.read_bytes()[:4], b"DF2P")
        self.assertFalse((out.parent / "src").exists(), "the 1.2 GB source file does not stay on disk")
        words = [s[0] for s in self.steps]
        self.assertIn("Downloading the DFlash2 drafter…", words)
        self.assertIn("Packing the DFlash2 drafter for the engine…", words)
        self.assertEqual(words[-1], "The DFlash2 drafter is ready.")
        self.assertEqual(max(s[1] for s in self.steps if s[0].startswith("Downloading")), sum(f[1] for f in self.files))

    def test_a_drafter_that_is_here_is_not_fetched_again(self):
        self.fetch()
        _Handler.hits.clear()
        self.steps.clear()
        self.fetch()
        self.assertEqual(_Handler.hits, [])
        self.assertEqual(self.steps[0][0], "The DFlash2 drafter is ready.")

    def test_a_file_that_does_not_match_its_hash_is_deleted_and_the_words_are_plain(self):
        bad = tuple((n, size, "0" * 64) for n, size, _h in self.files)
        with mock.patch.dict("os.environ", {"STRATA_UPDATE_ALLOW_HTTP": "1"}):
            with self.assertRaises(drafting_fetch.DraftingError) as ctx:
                drafting_fetch.ensure(self.data, base_url=self.base, files=bad)
        self.assertIn("did not pass the safety check", str(ctx.exception))
        self.assertFalse(drafting.pack_path(self.data).exists())
        self.assertEqual(list((drafting.pack_path(self.data).parent / "src").glob("*.part")), [])

    def test_an_interrupted_download_is_continued(self):
        name, size, _sha = self.files[1]
        src = drafting.pack_path(self.data).parent / "src"
        src.mkdir(parents=True)
        (src / (name + ".part")).write_bytes(self.blobs[name][: size // 2])
        self.fetch()
        ranges = [h for h in _Handler.hits if h.endswith(name)]
        self.assertEqual(len(ranges), 1)
        self.assertTrue(drafting.pack_path(self.data).is_file())

    def test_a_start_never_downloads(self):
        with acquisition.forbidden():
            with self.assertRaises(acquisition.AcquisitionRefused):
                drafting_fetch.ensure(self.data, base_url=self.base, files=self.files)
        self.assertFalse(drafting.pack_path(self.data).exists())

    def test_the_plan_names_the_size_and_where_it_comes_from(self):
        with mock.patch.object(drafting, "pack_ready", return_value=False):
            plan = drafting_fetch.plan(self.data)
        self.assertEqual(plan["gb"], 1.2)
        self.assertIn("1.2 GB to download", plan["text"])
        self.assertIn("0.76 GB", plan["text"])
        self.assertIn("incoai/Qwen3.8-Flash-Next-DFlash2", plan["text"])
        self.assertIn("Apache-2.0", plan["text"])
        self.assertIn("starts with the MTP instead", plan["text"], "and what happens if it does not fit")
        with mock.patch.object(drafting, "pack_ready", return_value=True):
            self.assertEqual(drafting_fetch.plan(self.data)["gb"], 0.0)

    def test_the_command_line_says_the_same(self):
        code, out, err = run_cli(["drafting", "plan", "--data-dir", str(self.data), "--output", "json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual((payload["ok"], payload["revision"]), (True, drafting.REVISION))
        self.assertEqual(payload["gb"], 1.2)


class TheAppAsksControl(unittest.TestCase):
    """The app calls `strata drafting plan|fetch --config <run config> --mode <choice>` before a load; Control decides, the app never reads the table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "data"
        self.cfg = Path(self.tmp.name) / "strata-gyro.json"
        self.cfg.write_text(json.dumps({"model_name": GYRO, "args": ["--pack", str(self.data / "packs" / "gyro"), "--mtp", str(self.data / "mtp" / "rt")]}))

    def plan(self, *extra):
        code, out, err = run_cli(["drafting", "plan", "--config", str(self.cfg), "--output", "json", *extra])
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_nothing_is_needed_for_auto_mtp_or_off(self):
        for mode in ("auto", "mtp", "off"):
            plan = self.plan("--mode", mode)
            self.assertFalse(plan["needed"], mode)
            self.assertEqual(plan["gb"], 0.0)
        self.assertEqual(self.plan("--mode", "off")["drafter"], "off")

    def test_dflash2_is_needed_once_the_engine_can_serve_it_and_not_before(self):
        before = self.plan("--mode", "dflash2")
        self.assertFalse(before["needed"], "while the engine cannot serve it nothing is downloaded")
        self.assertEqual(before["line"], "Drafting: MTP (DFlash2 isn't available in this mode)")
        with serving_dflash2():
            plan = self.plan("--mode", "dflash2")
            self.assertTrue(plan["needed"])
            self.assertEqual(plan["gb"], 1.2)
            self.assertEqual(Path(plan["pack"]), drafting.pack_path(self.data), "the data folder comes from the run config's own arguments")
            self.cfg.write_text(json.dumps({"model_name": "qwen3.8-flash-next-coder-iq3_xxs", "drafting": "dflash2", "args": ["--pack", str(self.data / "packs" / "c")]}))
            other = self.plan()
            self.assertFalse(other["needed"])
            self.assertEqual(other["note"], drafting.NOT_SUPPORTED)

    def test_a_model_that_is_not_installed_yet_is_asked_about_by_its_family(self):
        with serving_dflash2():
            for family, needed in (("gyro", True), ("qwen", True), ("coder", False), ("swift", False)):
                code, out, err = run_cli(["drafting", "plan", "--family", family, "--mode", "dflash2", "--data-dir", str(self.data), "--output", "json"])
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads(out)["needed"], needed, family)

    def test_fetch_if_needed_does_nothing_when_nothing_is_needed(self):
        code, out, err = run_cli(["drafting", "fetch", "--config", str(self.cfg), "--mode", "auto", "--if-needed", "--output", "jsonl"])
        self.assertEqual(code, 0, err)
        last = json.loads(out.strip().splitlines()[-1])
        self.assertEqual((last["ok"], last["skipped"], last["pack"]), (True, True, None))
        self.assertFalse(drafting.pack_path(self.data).exists())


class TheDownloadPlanListsIt(unittest.TestCase):
    """`strata launch --plan --drafting dflash2` adds the drafter to the download when it would be used."""

    def test_the_plan_line_and_the_part(self):
        from tests.terminal.launch_support import HW_12, Home, catalog
        from strata_cli.launch.runner import Runner
        with Home() as home:
            runner = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off", drafting="dflash2"), catalog=catalog(HW_12))
            line = runner.drafting_plan_line()
            self.assertEqual(line, "Drafting: MTP (DFlash2 isn't available in this mode)")
            self.assertNotIn("dflash2", [p["kind"] for p in runner.download_plan()["parts"]], "nothing to fetch while the engine cannot serve it")
            with serving_dflash2():
                kinds = {p["kind"]: p for p in runner.download_plan()["parts"]}
                self.assertEqual(kinds["dflash2"]["gb"], 1.2)
                self.assertIn("the DFlash2 drafter (1.2 GB)", runner.download_plan()["text"])
                self.assertIn("--drafting mtp", runner.download_plan()["text"])
                auto = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768, vision="off"), catalog=catalog(HW_12))
                self.assertNotIn("dflash2", [p["kind"] for p in auto.download_plan()["parts"]], "Auto is the MTP: no extra download")
                other = Runner(S.LaunchSettings(family="coder", model="IQ3_XXS", context=32768, vision="off", drafting="dflash2"), catalog=catalog(HW_12))
                self.assertNotIn("dflash2", [p["kind"] for p in other.download_plan()["parts"]], "not for this model: hidden, not offered")


if __name__ == "__main__":
    unittest.main()


class ControlReportsWhetherTheEngineServesIt(unittest.TestCase):
    def test_the_plan_says_it_so_the_app_can_hide_the_choice(self):
        for served in (False, True):
            with mock.patch.object(drafting, "ENGINE_SERVES_DFLASH2", served):
                code, out, err = run_cli(["drafting", "plan", "--mode", "auto", "--family", "gyro", "--data-dir", tempfile.gettempdir(), "--output", "json"])
                self.assertEqual(code, 0, err)
                self.assertIs(json.loads(out)["serves"], served)
