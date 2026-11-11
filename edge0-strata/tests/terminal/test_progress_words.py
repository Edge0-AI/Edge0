"""0.2.2: the words for a model download and its preparation are plain and true (strata_control/progress_words.py), and the app's
and the Terminal's progress read the same lines the same way (tools/strata_mcp.py install_progress / combined_progress).

The excerpts are from real setup logs (Gyro-S on the RTX 5070 PC, 2026-10-19) and from the steps a Q2_0 install adds."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strata_mcp  # noqa: E402
from strata_control import progress_words as W  # noqa: E402

HEAD = ["Strata - Qwen3.8-Flash-Next on a normal PC", "=== Step 1: checking your PC ===", "  [ok] GPU: NVIDIA GeForce RTX 5070",
        "=== Step 2: your choices ===", "  [ok] size: Gyro-S", "=== Step 3: Python packages ===", "  [ok] numpy ...",
        "=== Step 4: the Strata engine ==="]
STEP5 = ["=== Step 5: downloading Qwen3.8-Flash-Next (Gyro) Gyro-S ===",
         "  The model files go in C:\\Users\\x\\Documents\\Strata\\models\\gyro-Gyro-S"]
MODEL_PROGRESS = "  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  30.53 / 58.49 GB (52%)   "
CHECK = "  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf (SHA-256, 58.5 GB) ..."
STEP6 = ["=== Step 6: preparing the model for Strata ==="]


def prog(lines):
    p = strata_mcp.install_progress(lines)
    return p, p.get("label"), p.get("phase")


class StepWords(unittest.TestCase):
    def test_checking_the_pc(self):
        self.assertEqual(prog(HEAD[:3])[1:], ("Checking your computer", "check"))

    def test_the_engine_and_the_helper_download_have_their_own_words(self):
        self.assertEqual(prog(HEAD + ["  llama.cpp source:   0.01 / 0.04 GB (21%)   "])[2], "helpers")
        self.assertIn("small tools", prog(HEAD + ["  llama.cpp source:   0.01 / 0.04 GB (21%)   "])[1])
        self.assertEqual(prog(HEAD + ["  Strata engine:   0.20 / 0.90 GB (22%)   "])[1], "Downloading the engine for this PC")
        self.assertEqual(prog(HEAD)[2], "engine")

    def test_the_model_download(self):
        _p, label, phase = prog(HEAD + STEP5 + [MODEL_PROGRESS])
        self.assertEqual((label, phase), ("Downloading the model files", "download"))

    def test_a_figure_from_an_earlier_step_does_not_leak_into_the_next(self):
        """Step 4's 'llama.cpp source' line must not make step 5 'Downloading ...' before any model line is printed."""
        p, label, phase = prog(HEAD + ["  llama.cpp source:   0.04 / 0.04 GB (100%)   "] + STEP5)
        self.assertNotIn("download", p)
        self.assertEqual(phase, "download")
        self.assertEqual(label, "Getting the model files")

    def test_the_hash_check_is_checking_not_downloading(self):
        lines = HEAD + STEP5 + [MODEL_PROGRESS, "  [ok] downloaded", CHECK]
        p, label, phase = prog(lines)
        self.assertEqual(phase, "verify")
        self.assertIn("Checking that the model files are complete", label)
        self.assertNotIn("Downloading", label)
        # with its own figure
        p, label, phase = prog(lines + ["  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "])
        self.assertEqual(phase, "verify")
        self.assertEqual(p["download"]["percent"], 20)

    def test_a_damaged_file_is_repaired_in_plain_words_and_the_second_check_is_checking(self):
        damaged = "  Some parts of Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf were damaged; downloading only those parts again."
        lines = HEAD + STEP5 + [MODEL_PROGRESS, CHECK, damaged, "  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "]
        _p, label, phase = prog(lines)
        self.assertEqual(phase, "repair")
        self.assertEqual(label, "Some parts of a model file were damaged; downloading only those parts again")
        done = lines + ["  2 of 1186 parts were different and have been written again; checking the whole file again ..."]
        self.assertEqual(prog(done)[2], "download")

    def test_the_picture_reader(self):
        lines = HEAD + STEP5 + ["  [ok] model files present", "  vision encoder:   0.30 / 0.91 GB (33%)   "]
        self.assertEqual(prog(lines)[1:], ("Downloading the picture reader", "pictures"))

    def test_files_already_there(self):
        lines = HEAD + STEP5 + ["  [ok] Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf already downloaded"]
        self.assertEqual(prog(lines)[2], "verify")

    def test_step_six_packing_the_experts(self):
        lines = HEAD + STEP5 + STEP6 + ["  Converting the Q2_0 experts for the AVX-512 kernel (one time, ~40 GB written, 2-5 min) ...",
                                        "  > python tools/strata_pack.py build --gguf x --out y --skip-hash --force"]
        self.assertEqual(prog(lines)[1:], ("Packing the model's experts for your processor (one time, a few minutes)", "pack"))
        self.assertEqual(prog(lines + ["  > python tools/pack_index.py --pack y"])[1], "Indexing the packed experts")
        self.assertEqual(prog(lines + ["  > python tools/strata_tokenizer.py --gguf x --out y"])[1], "Preparing the tokenizer")

    def test_step_six_the_draft_layer_download_then_its_preparation(self):
        fetch = ["  [ok] model prepared: y", "  The MTP draft layer (speculative decoding, ~2x faster output) comes from the original Qwen checkpoint:",
                 "  only its ~5 GB of MTP tensors are downloaded.", "  > python tools/mtp_fetch.py fetch --out z",
                 "  MTP draft layer:   1.65 / 5.21 GB (31%)"]
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch)
        self.assertEqual(phase, "draft_download")
        self.assertIn("Downloading the draft layer", label)
        # the download is over: the numbers it printed are stale, the next tool is running
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch + ["  MTP draft layer:   5.21 / 5.21 GB (100%)", "  > python tools/mtp_pack.py --src z"])
        self.assertEqual((label, phase), ("Preparing the draft layer", "draft"))
        p, label, phase = prog(HEAD + STEP5 + STEP6 + fetch + ["  > python tools/mtp_rt.py --gguf g --out rt"])
        self.assertEqual(phase, "draft")

    def test_step_six_with_everything_already_there(self):
        lines = HEAD + STEP5 + STEP6 + ["  [ok] model prepared: y", "  [ok] MTP draft layer: z"]
        self.assertEqual(prog(lines)[2], "draft")

    def test_low_memory_mode_packing(self):
        lines = HEAD + STEP5 + STEP6 + ["  Writing the experts into one file for the low-memory mode (one time, 80 GB) ...",
                                        "  > python tools/iq_pack.py --gguf x --out y --experts-bin"]
        self.assertEqual(prog(lines)[1], "Packing the experts into one file for low-memory mode (one time)")

    def test_the_last_step(self):
        lines = HEAD + STEP5 + STEP6 + ["=== Step 7: writing the start script ===", "  [ok] KV streaming on"]
        self.assertEqual(prog(lines)[1:], ("Saving the model's settings", "settings"))

    def test_the_heading_is_found_even_after_hundreds_of_tool_lines(self):
        lines = HEAD + STEP5 + STEP6 + ["  > python tools/iq_pack.py --gguf x --out y"] + [f"  layer {i}/40" for i in range(900)]
        self.assertEqual(prog(lines)[2], "pack")

    def test_no_step_yet(self):
        self.assertEqual(prog(["Strata - Qwen3.8-Flash-Next"])[1:], ("Getting your model ready", "working"))

    def test_with_numbers(self):
        w = W.describe(5, STEP5, {"file": "x.gguf", "done_gb": 12.34, "total_gb": 58.49, "percent": 21})
        self.assertEqual(W.with_numbers(w, {"done_gb": 12.34, "total_gb": 58.49}), "Downloading the model files: 12.3 of 58.5 GB")
        self.assertEqual(W.with_numbers(w, None), "Downloading the model files")


class CombinedFigure(unittest.TestCase):
    def test_a_part_file_the_pieces_filled_ahead_is_not_a_finished_download(self):
        """The parallel download writes the last pieces early, so the .part is full length while the pieces done are 50%: the log's
        figure is the truth (the bar used to jump to 99%)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            part = folder / "m-00001-of-00002.gguf.part"
            part.write_bytes(b"\0" * 1000)
            files = [{"path": str(folder / "m-00001-of-00002.gguf"), "bytes": 1_000_000_000, "role": "model"}]
            prog_ = {"step": "5/7: downloading X", "download": {"file": "m-00001-of-00002.gguf", "done_gb": 0.5, "total_gb": 1.0, "percent": 50}}
            # the .part says 1000 bytes here; make it say 1 GB without writing it
            import os
            with open(part, "r+b") as f:
                f.truncate(1_000_000_000)
            out = strata_mcp.files_progress(prog_, {"total_gb": 1.0}, files)
            self.assertAlmostEqual(out["download"]["done_gb"], 0.5, places=2)
            self.assertEqual(out["download"]["percent"], 50)

    def test_the_hash_check_has_its_own_figure(self):
        lines = HEAD + STEP5 + [CHECK, "  checking Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf:  12.00 / 58.49 GB (20%)   "]
        p = strata_mcp.install_progress(lines)
        out = strata_mcp.combined_progress(p, {"folder": None, "total_gb": 58.5})
        self.assertEqual(out["download"]["file"], "checking files")
        self.assertEqual(out["download"]["percent"], 20)


if __name__ == "__main__":
    unittest.main()
