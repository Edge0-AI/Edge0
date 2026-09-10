"""CLI parse tests (no checkpoints, no HTTP)."""

from __future__ import annotations

from edge0.cli import build_parser


def test_serve_defaults_to_loopback():
    args = build_parser().parse_args(["serve", "/tmp/model"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_serve_accepts_explicit_non_loopback_host():
    args = build_parser().parse_args(
        ["serve", "/tmp/model", "--host", "0.0.0.0", "--port", "8080"])
    assert args.host == "0.0.0.0"
    assert args.port == 8080
