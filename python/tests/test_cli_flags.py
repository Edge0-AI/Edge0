"""CLI wiring for the prefill switches (no GPU needed)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from edge0.cli import TIER_ENV, _build_parser, _engine_kwargs


@pytest.mark.parametrize("cmd", ["demo", "chat", "serve"])
def test_prefill_ondemand_reaches_the_engine(cmd):
    args = _build_parser().parse_args([cmd, "edge0-8b", "--prefill-ondemand"])
    assert _engine_kwargs(args) == {"prefill_ondemand": True}


@pytest.mark.parametrize("cmd", ["demo", "chat", "serve"])
def test_prefill_ondemand_defaults_off(cmd):
    args = _build_parser().parse_args([cmd, "edge0-8b"])
    assert "prefill_ondemand" not in _engine_kwargs(args)


def test_existing_engine_flags_still_map():
    args = _build_parser().parse_args(
        ["serve", "edge0-8b", "--no-prerouter", "--no-lora",
         "--history-slots"])
    assert _engine_kwargs(args) == {"prerouter": None, "lora": "",
                                    "history_slots": True}


def test_serve_keeps_its_own_flags():
    args = _build_parser().parse_args(
        ["serve", "edge0-8b", "--host", "0.0.0.0", "--port", "9"])
    assert (args.host, args.port) == ("0.0.0.0", 9)


@pytest.mark.parametrize("cmd", ["demo", "chat", "serve"])
@pytest.mark.parametrize("tier", ["edge0-8b", "edge0-35b"])
@pytest.mark.parametrize("no_prerouter", [False, True])
def test_entrypoint_preserves_prerouter_contract(
        cmd, tier, no_prerouter, monkeypatch, tmp_path):
    """Check actual CLI dispatch without constructing an inference engine."""
    from edge0 import AutoEngine

    class EngineConstructionReached(Exception):
        pass

    factory = Mock(side_effect=EngineConstructionReached)
    monkeypatch.setattr(AutoEngine, "from_pretrained", factory)
    monkeypatch.setenv(TIER_ENV[tier], str(tmp_path))
    argv = [cmd, tier]
    if no_prerouter:
        argv.append("--no-prerouter")
    args = _build_parser().parse_args(argv)

    with pytest.raises(EngineConstructionReached):
        args.fn(args)

    expected = {}
    if no_prerouter or (cmd == "demo" and tier == "edge0-8b"):
        expected["prerouter"] = None
    factory.assert_called_once_with(str(tmp_path), name=tier, **expected)
