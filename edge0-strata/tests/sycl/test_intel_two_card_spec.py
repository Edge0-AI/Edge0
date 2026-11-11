"""strata_control/intel_spec.py: --spec 2 for a run config on exactly two Arc B-series (Battlemage) cards.

The rule changes only the generated default (--spec 4).  Single cards, Alchemist cards, three cards, a config with no draft
layer and anything the person set stay as they are.  Also checks the two places that apply it: sycl/setup_intel.py's
config rewrite (through apply_to_config) and the Intel server's engine_args wrapper."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from strata_control import intel_spec  # noqa: E402

B65 = {"driver": "xe", "devid": "e222"}
B70 = {"driver": "xe", "devid": "e223"}
B50 = {"driver": "xe", "devid": "e212"}
B580 = {"driver": "xe", "devid": "e20b"}
A770 = {"driver": "xe", "devid": "56a0"}
A750 = {"driver": "i915", "devid": "56a1"}
IGPU = {"driver": "xe", "devid": "64a0"}
ARGS = ["--pack", "/p", "--native", "/m.gguf", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/m/rt",
        "--max-context", "32768", "--kv", "int8"]


def spec_of(args):
    return args[args.index("--spec") + 1]


def test_battlemage_by_pci_id():
    for c in (B65, B70, B50, B580, {"driver": "xe", "devid": "0xE223"}, {"driver": "xe", "devid": "E20C"}):
        assert intel_spec.is_battlemage(c["driver"], c["devid"]), c
    assert not intel_spec.is_battlemage("xe", "56a0")        # an A770 on xe is Alchemist
    assert not intel_spec.is_battlemage("i915", "56a1")      # the A750
    assert not intel_spec.is_battlemage("i915", "e223")      # the driver decides too
    assert not intel_spec.is_battlemage("xe", "64a0")        # an integrated Xe2 part is not a B-series card
    assert not intel_spec.is_battlemage("xe", "")
    assert not intel_spec.is_battlemage("xe", "zzzz")


def test_two_b_series_cards_get_spec_2():
    for pair in ([B65, B65], [B70, B70], [B65, B50], [B580, B580]):
        args, note = intel_spec.plan(ARGS, None, pair)
        assert spec_of(args) == "2" and note, pair
        assert spec_of(ARGS) == "4"                           # the input list is not changed
    # nothing else moved
    args, _ = intel_spec.plan(ARGS, None, [B65, B65])
    assert [a for i, a in enumerate(args) if args[i - 1:i] != ["--spec"]] == [a for i, a in enumerate(ARGS) if ARGS[i - 1:i] != ["--spec"]]


def test_everything_else_stays():
    for cards in ([], [B65], [A750], [A770], [A750, A750], [A770, A770], [B65, A770], [A750, B65], [B65, IGPU],
                  [B65, B65, B65], [B65, B65, B65, B65]):
        args, note = intel_spec.plan(ARGS, None, cards)
        assert args == ARGS and note == "", cards


def test_the_persons_own_spec_wins():
    # in user_args, whatever its value
    for val in ("4", "3", "2", "6"):
        args, note = intel_spec.plan(ARGS, ["--spec", val], [B65, B65])
        assert args == ARGS and note == "", val
    # unrelated user options do not stop the rule
    args, note = intel_spec.plan(ARGS, ["--vram-reserve-mib", "800"], [B65, B65])
    assert spec_of(args) == "2" and note
    # a value in args that is not setup's own 4 was chosen by somebody
    for val in ("3", "2", "5", "8"):
        mine = list(ARGS)
        mine[mine.index("--spec") + 1] = val
        args, note = intel_spec.plan(mine, None, [B65, B65])
        assert args == mine and note == "", val
    # no --spec at all: nothing is added
    no_spec = [a for a in ARGS if a not in ("--spec", "4")]
    args, note = intel_spec.plan(no_spec, None, [B65, B65])
    assert args == no_spec and note == ""


def test_no_draft_layer_no_change():
    no_mtp = ARGS[:ARGS.index("--mtp")] + ARGS[ARGS.index("--mtp") + 2:]
    assert intel_spec.plan(no_mtp, None, [B65, B65]) == (no_mtp, "")
    assert intel_spec.plan(ARGS, None, [B65, B65], mtp_off=True) == (ARGS, "")


def test_apply_to_config_keeps_the_written_copy_in_step():
    cfg = {"args": list(ARGS), "args_written": list(ARGS), "env": {}}
    note = intel_spec.apply_to_config(cfg, [B65, B65])
    assert note and spec_of(cfg["args"]) == "2" and spec_of(cfg["args_written"]) == "2"
    # a second call (setup run again) changes nothing and says nothing
    assert intel_spec.apply_to_config(cfg, [B65, B65]) == ""
    # a config of the person's: user_args with --spec, "mtp": false, one card
    own = {"args": list(ARGS), "args_written": list(ARGS), "user_args": ["--spec", "4"]}
    assert intel_spec.apply_to_config(own, [B65, B65]) == "" and spec_of(own["args"]) == "4"
    off = {"args": list(ARGS), "mtp": False}
    assert intel_spec.apply_to_config(off, [B65, B65]) == "" and spec_of(off["args"]) == "4"
    one = {"args": list(ARGS)}
    assert intel_spec.apply_to_config(one, [B65]) == "" and spec_of(one["args"]) == "4"


def test_sysfs_cards_in_card_order(tmp_path):
    def card(n, vendor, devid, driver):
        d = tmp_path / f"card{n}" / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
        (d / "device").write_text(devid + "\n")
        drv = tmp_path / "drivers" / driver
        drv.mkdir(parents=True, exist_ok=True)
        try:
            (d / "driver").symlink_to(drv, target_is_directory=True)
        except OSError:                                       # no symlinks (Windows without rights): nothing to read
            return False
        return True
    ok = card(0, "0x8086", "0xe222", "xe")
    card(1, "0x8086", "0xe222", "xe")
    card(2, "0x10de", "0x2d05", "nvidia")
    (tmp_path / "card0-DP-1").mkdir()
    if not ok:
        return
    cards = intel_spec.sysfs_cards(tmp_path)
    assert [c["devid"] for c in cards] == ["e222", "e222"] and [c["index"] for c in cards] == [0, 1]
    assert intel_spec.exactly_two_battlemage(cards)


def _server_intel():
    sys.path.insert(0, str(ROOT / "sycl" / "serve"))
    spec = importlib.util.spec_from_file_location("server_intel_under_test", ROOT / "sycl" / "serve" / "server_intel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_server_wrapper_applies_the_rule_at_start(monkeypatch):
    mod = _server_intel()
    S = mod.S
    seen = []
    monkeypatch.setattr(S, "engine_args", lambda cfg: list(cfg["args"]))
    monkeypatch.setattr(intel_spec, "sysfs_cards", lambda *a, **k: [dict(B65, index=0), dict(B65, index=1)])
    mod.install_two_card_spec()
    cfg = {"backend": "sycl", "gpu": [0, 1], "args": list(ARGS)}
    assert spec_of(S.engine_args(cfg)) == "2"
    assert spec_of(cfg["args"]) == "4"                        # the config in memory is not rewritten by a start
    assert spec_of(S.engine_args({**cfg, "user_args": ["--spec", "4"]})) == "4"
    assert spec_of(S.engine_args({**cfg, "gpu": 0})) == "4"   # one card
    assert spec_of(S.engine_args({**cfg, "backend": "hip"})) == "4"
    assert spec_of(S.engine_args({**cfg, "args": ARGS + ["--peer-device", "1"]})) == "4"   # not a layer split
    assert spec_of(S.engine_args({**cfg, "mtp": False})) == "4"


def test_server_wrapper_one_a_series_card_stays(monkeypatch):
    mod = _server_intel()
    S = mod.S
    monkeypatch.setattr(S, "engine_args", lambda cfg: list(cfg["args"]))
    monkeypatch.setattr(intel_spec, "sysfs_cards", lambda *a, **k: [dict(B65, index=0), dict(A750, index=1)])
    mod.install_two_card_spec()
    cfg = {"backend": "sycl", "gpu": [0, 1], "args": list(ARGS)}
    assert spec_of(S.engine_args(cfg)) == "4"



def _derive(base, settings, cards):
    from unittest import mock
    from strata_cli.launch.runner import derive_config
    with mock.patch.object(intel_spec, "sysfs_cards", lambda *a, **k: cards):
        return derive_config(base, settings, bool_flags=frozenset())


def test_launcher_writes_spec_2_for_two_b_series_cards():
    from strata_cli.launch import settings as L
    base = {"backend": "sycl", "gpu": [0, 1], "gpus_asked": True, "layer_split": "auto", "exe": "component:strata-engine",
            "args": list(ARGS), "port": 8095}
    two = [dict(B65, index=0), dict(B65, index=1)]
    cfg = _derive(base, L.LaunchSettings(), two)
    assert spec_of(cfg["args"]) == "2" and spec_of(base["args"]) == "4"            # the installed config is never edited
    # the person's choices come after the rule and win: --set --spec 4 on the command line, user_args in the config
    assert spec_of(_derive(base, L.LaunchSettings(extra={"--spec": "4"}), two)["args"]) == "4"
    assert spec_of(_derive(base, L.LaunchSettings(extra={"--spec": "3"}), two)["args"]) == "3"
    assert spec_of(_derive({**base, "user_args": ["--spec", "4"]}, L.LaunchSettings(), two)["args"]) == "4"
    # no draft layer for this launch: nothing to pipeline, nothing changed
    assert "--spec" not in _derive(base, L.LaunchSettings(mtp="off"), two)["args"] or         spec_of(_derive(base, L.LaunchSettings(mtp="off"), two)["args"]) == "4"
    # one card, an A-series card in the pair, another backend
    assert spec_of(_derive({**base, "gpu": 0}, L.LaunchSettings(), two)["args"]) == "4"
    assert spec_of(_derive(base, L.LaunchSettings(), [dict(B65, index=0), dict(A750, index=1)])["args"]) == "4"
    assert spec_of(_derive({**base, "backend": "hip"}, L.LaunchSettings(), two)["args"]) == "4"
