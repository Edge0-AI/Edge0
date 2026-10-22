"""CLI/report contracts for the MLX benchmark harness; no scalar numerical oracle."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "examples" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def invocation(bench, monkeypatch):
    from edge0 import AutoEngine

    for key in ("BENCH_PROMPT", "BENCH_LONG", "BENCH_NTOK"):
        monkeypatch.delenv(key, raising=False)
    engine = SimpleNamespace(name="edge0-35b", close=Mock())
    factory = Mock(return_value=engine)
    monkeypatch.setattr(AutoEngine, "from_pretrained", factory)
    report = {"runs": [], "mean_tok_s": 7.5, "peak_gib": 3.25,
              "mean_prefill_tok_s": 125.5}

    def measured(*args, **kwargs):
        print("measurement progress")
        return report

    run = Mock(side_effect=measured)
    monkeypatch.setattr(bench, "run_bench", run)
    return engine, factory, run, report


def test_submission_json_is_one_document_and_matches_template(
        bench, invocation, monkeypatch, capsys):
    engine, factory, run, report = invocation
    monkeypatch.setattr(bench, "_hardware_metadata", lambda: {
        "device": "Mac fixture", "chip": "Apple fixture", "ram_gb": None,
        "os": "macOS fixture", "mlx_version": "fixture"})
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission", "--json",
                                    "--reporter", "@tester", "--storage", "external SSD",
                                    "--on-battery", "no", "--notes", "first\nsecond"])
    assert bench.main() == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert len(output.out.splitlines()) == 1
    assert "measurement progress" in output.err
    assert data["tier"] == engine.name
    assert data["runtime"] == "python-mlx"
    assert data["decode_tok_s"] == report["mean_tok_s"]
    assert data["prefill_tok_s"] == report["mean_prefill_tok_s"]
    assert data["peak_active_gib"] == report["peak_gib"]
    assert data["ram_gb"] is None
    assert data["reporter"] == "@tester"
    assert data["on_battery"] == "no"
    assert data["storage"] == "external SSD"
    assert "first second" in data["notes"]
    assert "MLX allocator" in data["notes"]
    assert set(data) == {"tier", "runtime", "device", "chip", "ram_gb", "os", "storage",
                         "mlx_version", "decode_tok_s", "prefill_tok_s", "peak_active_gib",
                         "on_battery", "reporter", "notes"}
    run.assert_called_once_with(engine, 200, 10, submission=True)
    factory.assert_called_once_with("checkpoint", name=None)
    engine.close.assert_called_once()


def test_submission_text_has_pasteable_colon_fields(bench, invocation, monkeypatch, capsys):
    monkeypatch.setattr(bench, "_hardware_metadata", lambda: {})
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission"])
    assert bench.main() == 0
    text = capsys.readouterr().out
    assert "tier:" in text and "runtime:" in text and "reporter:" in text
    assert "None" not in text


def test_default_cli_retains_existing_benchmark_call(bench, invocation, monkeypatch, capsys):
    engine, _, run, _ = invocation
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint"])
    assert bench.main() == 0
    run.assert_called_once_with(engine, 200, 10)
    engine.close.assert_called_once()
    output = capsys.readouterr()
    assert "[bench] tier=edge0-35b" in output.out
    assert "measurement progress" in output.out
    assert output.err == ""


@pytest.mark.parametrize("flags", [["--json"], ["--submission", "--ntok", "0"],
                                  ["--submission", "--warmup", "-1"],
                                  ["--submission", "--on-battery", "maybe"]])
def test_invalid_flags_fail_before_loading_model(bench, invocation, monkeypatch, flags):
    _, factory, _, _ = invocation
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", *flags])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    factory.assert_not_called()


def test_submission_rejects_custom_prompt_before_model_load(bench, invocation, monkeypatch):
    _, factory, _, _ = invocation
    monkeypatch.setenv("BENCH_PROMPT", "custom prompt")
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    factory.assert_not_called()


def test_json_failure_does_not_emit_partial_document_and_closes_engine(
        bench, invocation, monkeypatch, capsys):
    engine, _, run, _ = invocation
    run.side_effect = RuntimeError("fixture failure")
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission", "--json"])
    with pytest.raises(RuntimeError, match="fixture failure"):
        bench.main()
    assert capsys.readouterr().out == ""
    engine.close.assert_called_once()


def test_json_routes_engine_load_and_cleanup_progress_to_stderr(
        bench, invocation, monkeypatch, capsys):
    engine, factory, _, _ = invocation

    def load(*args, **kwargs):
        print("loading fixture")
        return engine

    factory.side_effect = load
    engine.close.side_effect = lambda: print("closing fixture")
    monkeypatch.setattr(bench, "_hardware_metadata", lambda: {})
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission", "--json"])
    assert bench.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["tier"] == engine.name
    assert "loading fixture" in output.err and "closing fixture" in output.err


def test_json_rejects_nonfinite_measurements(bench, invocation, monkeypatch, capsys):
    _, _, _, report = invocation
    report["mean_tok_s"] = float("nan")
    monkeypatch.setattr(bench, "_hardware_metadata", lambda: {})
    monkeypatch.setattr("sys.argv", ["bench.py", "checkpoint", "--submission", "--json"])
    with pytest.raises(ValueError):
        bench.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("submission", [False, True])
def test_mlx_harness_retains_reset_warmup_decode_shape(bench, monkeypatch, submission):
    from edge0.backends import core

    for key in ("BENCH_TEMP", "BENCH_SEED", "BENCH_PROMPT", "BENCH_LONG"):
        monkeypatch.delenv(key, raising=False)

    class TinyMLXEngine:
        name = "edge0-35b"
        _tok = None
        cfg = SimpleNamespace(gen=SimpleNamespace(temperature=0.7, top_k=2, top_p=1.0,
                                                  repetition_penalty=1.0))

        def __init__(self):
            self.resets = 0
            self.steps = 0
            self.prefills = []
            self.prompts = []
            self.logits = core.zeros((4,))

        def encode_chat(self, messages, *, think):
            assert think is False
            self.prompts.append(messages[0]["content"])
            return [1, 2, 3]

        def reset(self):
            self.resets += 1

        def prefill(self, ids):
            self.prefills.append(list(ids))
            core.eval(self.logits)

        def next_logits(self):
            return self.logits

        def step(self, token):
            self.steps += 1
            return self.logits

    engine = TinyMLXEngine()
    report = bench.run_bench(engine, 2, 1, submission=submission)
    assert engine.resets == 2
    assert engine.steps == 6
    assert engine.prefills == [[1, 2, 3], [1, 2, 3]]
    assert engine.prompts == [bench._long_prompt() if submission else bench.PROMPTS[engine.name]]
    assert len(report["runs"]) == 2
    keys = {"runs", "mean_tok_s", "peak_gib"}
    run_keys = {"prefill_s", "decode_s", "ntok", "peak_gib", "tok_s"}
    if submission:
        keys.add("mean_prefill_tok_s")
        run_keys.add("prefill_tok_s")
    assert set(report) == keys
    assert all(set(run) == run_keys for run in report["runs"])
    json.dumps(report, allow_nan=False)


def test_submission_metadata_failure_leaves_unknown_values(bench, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("fixture")

    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(bench.platform, "mac_ver", lambda: ("fixture", (), ""))
    monkeypatch.setattr(bench.subprocess, "run", unavailable)
    monkeypatch.setattr(bench, "version", Mock(side_effect=bench.PackageNotFoundError))
    assert bench._hardware_metadata() == {"device": "", "chip": "", "ram_gb": None,
                                          "os": "macOS fixture", "mlx_version": ""}


def test_submission_alias_uses_existing_environment_resolution(bench, invocation, monkeypatch):
    engine, factory, _, _ = invocation
    monkeypatch.setenv("EDGE0_35B_MODEL", "fixture-checkpoint")
    monkeypatch.setattr(bench, "_hardware_metadata", lambda: {})
    monkeypatch.setattr("sys.argv", ["bench.py", "edge0-35b", "--submission", "--json"])
    assert bench.main() == 0
    factory.assert_called_once_with("fixture-checkpoint", name=None)
    engine.close.assert_called_once()
