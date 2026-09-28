"""CLI wiring for the prefill switches (no GPU needed)."""

from __future__ import annotations

import pytest

from edge0.cli import _build_parser, _engine_kwargs


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
