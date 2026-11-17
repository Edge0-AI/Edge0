"""The owner's rule for setup (#403 #406 #364 #384): setup RECOMMENDS, it never forces - and the recommended answers
stay exactly what they were.  setup.main() with --yes (and with Enter for every question) on a few PCs writes the same
run config, byte for byte, as 0.1.32 plus the fixes on top of it did (tools/test_setup_golden.json, written by that
setup).  Every outside effect is mocked: no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_golden
    python tools/test_setup_golden.py --write      (only to record a new baseline on purpose)
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GOLDEN = Path(__file__).with_name("test_setup_golden.json")


def card(i, name, vram, arch):
    return {"index": i, "name": name, "vram_gb": vram, "arch": arch, "driver": "580.97"}


# the PCs: total RAM as ram_gb() reads it (a little under the label), the cards as nvidia-smi lists them
PROFILES = {
    "64GB-1x32GB": (63.7, [card(0, "NVIDIA GeForce RTX 5090", 31.8, "120")]),                  # #406
    "32GB-2x24GB": (31.9, [card(i, "NVIDIA GeForce RTX 3090", 24.0, "86") for i in range(2)]),  # #364
    "47GB-2x16GB": (47.0, [card(i, "NVIDIA GeForce RTX 5060 Ti", 15.9, "120") for i in range(2)]),   # #384
    "96GB-1x16GB": (95.8, [card(0, "NVIDIA GeForce RTX 5080", 15.9, "120")]),
    "128GB-1x24GB": (127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")]),
}
# (family, model): None = the recommended size
MODELS = [("qwen", None), ("qwen", "Q2_0"), ("qwen", "IQ3_XXS"), ("qwen", "IQ3_S"), ("unsloth", "UD-Q4_K_XL")]


class FakeGGUF:
    """gguf_reader.GGUFFile: shard 2 holds the PLE table."""
    def __init__(self, path):
        names = ["per_layer_token_embd.weight"] if "00002-of" in str(path) else ["blk.0.ffn_up_exps.weight"]
        self.tensors = [types.SimpleNamespace(name=n) for n in names]


def normalize(v, t: Path):
    """The config with the temp folder and the platform's names taken out (it compares on Windows and Linux)."""
    if isinstance(v, dict):
        return {k: normalize(x, t) for k, x in v.items()}
    if isinstance(v, list):
        return [normalize(x, t) for x in v]
    if isinstance(v, str):
        v = v.replace(str(t), "<T>").replace("\\", "/")
        # the platform's own names of the programs (strata.exe or strata, strata-vision.exe or strata-vision, ...) so the baseline compares on Windows and Linux (#3325)
        for name, label in ((setup.HVEXE, "<HVEXE>"), (setup.VEXE, "<VEXE>"), (setup.EXE, "<EXE>")):
            v = re.sub(rf"(?<![w.-]){re.escape(name)}(?![w.-])", label, v)   # #973: only the whole name, not "strata-x.log" on Linux
        return v
    return v


def install(ram, found, argv, answers=None, extra=(), avx512=False, configs=()):
    """setup.main() on a mocked PC -> (exit code, printed text, the written config or None, the questions asked).
    answers: None = --yes (a question fails the run), "" = Enter for every question, a list of answers in order, or
    {words of a question: its answer} (Enter for the others).
    extra: more patches (mock.patch objects).  configs: (name, dict) run configs already in the Strata folder."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "data" / "mtp" / "rt").mkdir(parents=True)
        (t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        eng = t / "engine"
        eng.mkdir()
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.32", "source": "local", "cuda_dirs": ["<cuda>"]}))
        for name, c in configs:
            (t / name).write_text(json.dumps(c))
        asked = []

        def fake_input(prompt=""):
            asked.append(prompt)
            if answers is None:
                raise AssertionError(f"asked {prompt!r} with --yes")
            if isinstance(answers, list):
                return answers.pop(0) if answers else ""
            if isinstance(answers, dict):                  # {words of the question: answer}, else Enter
                return next((v for k, v in answers.items() if k in prompt), "")
            return answers

        def fake_download(url, dst, what=None):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        patches = [
            mock.patch.object(setup, "ROOT", t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d, *a, **k: (t / "data", [])),
            mock.patch.object(setup, "load_settings", lambda: {}),
            mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "card_speeds", lambda: {}),
            mock.patch.object(setup, "amd_gpus", lambda: []),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, avx512)),
            mock.patch.object(setup, "cpu_cores", lambda: None),   # #642: not a hybrid CPU
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "rotational_disk", lambda p: None),   # #605: the test PC's own disk
            mock.patch.object(setup, "ple_ram_platform_ok", lambda: False),   # 147-verify: the golden configs as on Windows
            mock.patch.object(setup, "free_gb", lambda p: 900.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", lambda *a, **k: eng),
            mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda *a, **k: None),
            mock.patch.object(setup, "run", lambda *a, **k: None),
            mock.patch.object(setup, "mtp_corrupt", lambda *a, **k: False),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: t / f"run-{tag}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "calibrate_config", mock.Mock(side_effect=AssertionError("calibrated"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", "--models-dir", str(t / "models"), *argv]
                              + (["--yes"] if answers is None else [])),
            mock.patch("builtins.input", fake_input),
            *extra,
        ]
        out = io.StringIO()
        code = None
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            with contextlib.redirect_stdout(out):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        written = sorted(t.glob("strata-*.json"), key=lambda p: p.stat().st_mtime)
        cfg = None
        if written and code == 0:
            cfg = normalize(json.loads(written[-1].read_text(encoding="utf-8")), t)
            # "args_written" (strata_control/user_args.py): the arguments as setup wrote them, so a later hand edit can be told; the golden configs keep their own keys
            assert cfg.pop("args_written", cfg.get("args")) == cfg.get("args"), "args_written is not the written args"
        return code, out.getvalue(), cfg, asked


def argv_for(family, model):
    return ["--family", family, "--no-start"] + (["--model", model] if model else [])


def record() -> dict:
    """Every profile x model with --yes: the exit code and the config (the baseline this test compares against)."""
    res = {}
    for prof, (ram, found) in PROFILES.items():
        for family, model in MODELS:
            code, out, cfg, _ = install(ram, found, argv_for(family, model))
            res[f"{prof} {family} {model or 'recommended'}"] = {"code": code, "config": cfg}
    return res


class Golden(unittest.TestCase):
    """--yes and Enter: the same config as the baseline wherever the baseline wrote one."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def cases(self):
        for prof, (ram, found) in PROFILES.items():
            for family, model in MODELS:
                key = f"{prof} {family} {model or 'recommended'}"
                if self.golden[key]["code"] == 0:
                    yield key, ram, found, family, model

    def test_the_baseline_covers_every_kind_of_pc(self):
        written = [k for k, v in self.golden.items() if v["code"] == 0]
        self.assertGreaterEqual(len(written), 20)
        lowram_multi = self.golden["32GB-2x24GB qwen IQ3_XXS"]["config"]   # #364 -> #642: low-RAM, two cards -> both
        self.assertIn("--resident-experts" if "--resident-experts" in lowram_multi["args"] else "--mmap-experts",
                      lowram_multi["args"])
        self.assertEqual(lowram_multi.get("gpu"), [0, 1])                   # 0.1.40 (#642): resident on both cards; roles: Main 1 goes last, as before

    def test_yes(self):
        for key, ram, found, family, model in self.cases():
            with self.subTest(key):
                code, out, cfg, asked = install(ram, found, argv_for(family, model))
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg, self.golden[key]["config"], out[-3000:])
                self.assertEqual(asked, [])

    def test_enter_for_every_question(self):
        for key, ram, found, family, model in self.cases():
            if "UD-Q4_K_XL" in key and ram < 48:               # --yes consents to the risk; Enter declines it (the
                continue                                       # "Install it anyway? [n]" question), by design
            with self.subTest(key):
                code, out, cfg, asked = install(ram, found, argv_for(family, model), answers="")
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg, self.golden[key]["config"], out[-3000:])

    def test_the_refusals_with_yes_alone_stay(self):
        """A size the baseline refused with --yes and no explicit --model still stops (--yes alone is no consent)."""
        for prof, (ram, found) in PROFILES.items():
            for family, model in MODELS:
                key = f"{prof} {family} {model or 'recommended'}"
                if self.golden[key]["code"] != 0 and model is None:
                    with self.subTest(key):
                        self.assertNotEqual(install(ram, found, argv_for(family, model))[0], 0)


if __name__ == "__main__":
    if "--write" in sys.argv:
        GOLDEN.write_text(json.dumps(record(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {GOLDEN}")
    else:
        unittest.main()
