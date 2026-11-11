"""sycl/setup_intel.py: the prompt chunk size setup recommends for an Intel card (--prefill).

A Battlemage card (xe driver, not an A-series) with 24 GB or more and a context over 32K (the prompt borrows cache slots) gets
8192; everything else keeps 4096; a chunk size the user put in the config is kept."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
si = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(si)

B70 = dict(vram_gb=32.0, driver="xe", devid="e223")
A770 = dict(vram_gb=16.0, driver="xe", devid="56a0")
A750 = dict(vram_gb=8.0, driver="i915", devid="56a1")


def prefill_of(ctx, card, extra=()):
    cfg = {"args": ["--max-context", str(ctx), "--kv", "int8", "--prefill", "auto", *extra]}
    out = si.to_sycl(cfg, ROOT / "build-sycl-aot" / "strata", 64.0, {}, **card)
    return si.flag(out["args"], "--prefill")


def test_helper_values():
    assert si.recommended_prefill(32.0, 131072, "xe", "e223")[0] == 8192
    assert si.recommended_prefill(32.0, 131072, "xe", "e223")[1]          # a plain line says why
    assert si.recommended_prefill(24.0, 65536, "xe", "e221")[0] == 8192
    assert si.recommended_prefill(32.0, 32768, "xe", "e223") == (4096, "")   # own prompt buffers: a bigger chunk costs cache slots
    assert si.recommended_prefill(16.0, 131072, "xe", "e212")[0] == 4096     # under 24 GB
    assert si.recommended_prefill(16.0, 131072, "xe", "56a0")[0] == 4096     # Alchemist
    assert si.recommended_prefill(8.0, 131072, "i915", "56a1")[0] == 4096


def test_config_battlemage_long_context():
    assert prefill_of(131072, B70) == "8192"


def test_config_battlemage_32k_keeps_4096():
    assert prefill_of(32768, B70) == "4096"


def test_config_other_cards_unchanged():
    assert prefill_of(131072, A770) == "4096"
    assert prefill_of(131072, A750) == "4096"


def test_user_pick_is_kept():
    cfg = {"args": ["--max-context", "131072", "--kv", "int8", "--prefill", "16384"]}
    out = si.to_sycl(cfg, ROOT / "build-sycl-aot" / "strata", 64.0, {}, **B70)
    assert si.flag(out["args"], "--prefill") == "16384"
