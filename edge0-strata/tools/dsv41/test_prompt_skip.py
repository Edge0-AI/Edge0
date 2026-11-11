"""The long-prompt layer skip (STRATA_DS4_PROMPT_SKIP=1) against the full work: same answers within noise.

Needs a GPU, a DeepSeek-V4.1 pack and the golden ids file, so it runs only when they are named:

    STRATA_DSV41_BIN=.../strata-dsv41 STRATA_DSV41_PACK=/data/pack2.00 STRATA_DSV41_IDS=.../logpos.ids.json \
        python -m unittest tools.dsv41.test_prompt_skip

The golden segments are joined, the first DS4_SCORE_PREFIX tokens (4200) are read as one prompt (the path a long prompt
takes: logits of the last row only, where the skip applies), and the positions after them are scored teacher-forced.
The skip changes only what the layers 20-39 compute for the prompt's early rows, so the scored rows' KL against the
full work must stay at the noise of the experts' summing order.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import quant_quality  # noqa: E402

BIN = os.environ.get("STRATA_DSV41_BIN")
PACK = os.environ.get("STRATA_DSV41_PACK")
IDS = os.environ.get("STRATA_DSV41_IDS")


def keep_rows(nl=40, yoco=20, win=128):
    """The rows a long prompt keeps on layers yoco.. (the engine's forward_chunks): each layer looks back one window
    more than the layer before, so the last (nl - yoco) * (win - 1) + win rows."""
    return (nl - yoco) * (win - 1) + win


class PromptSkip(unittest.TestCase):
    def test_keep_rows_is_the_chain_of_windows(self):
        self.assertEqual(keep_rows(), 2668)

    @unittest.skipUnless(BIN and PACK and IDS, "needs STRATA_DSV41_BIN / _PACK / _IDS")
    def test_scored_rows_after_a_skipped_prompt_match_the_full_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            outs = {}
            for mode in ("0", "1"):
                out = pathlib.Path(tmp) / f"skip{mode}.tsv"
                env = dict(os.environ, STRATA_DS4_PROMPT_SKIP=mode, DS4_SCORE_PREFIX="4200", DS4_CHUNK="1024")
                subprocess.run([BIN, "--pack", PACK, "--tp", "1", "--threads", "off", "--max-context", "6144", "--vision",
                                "off", "--dspark", "off", "--expert-profile", "none", "--window", "128", "--score", IDS,
                                "--score-out", str(out)], check=True, env=env)
                outs[mode] = quant_quality.read_run(str(out))
            res = quant_quality.compare(outs["0"], outs["1"])["all"]
            self.assertGreater(res["positions"], 500)
            self.assertLess(res["kl"], 0.002)
            self.assertGreaterEqual(res["top1"], 0.99)


if __name__ == "__main__":
    unittest.main()
