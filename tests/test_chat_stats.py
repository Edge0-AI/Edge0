import json
from types import SimpleNamespace

from edge0.chat_stats import ChatStats


def test_request_deltas_and_token_timing(monkeypatch, capsys):
    import edge0.chat_stats as module
    times = iter([10., 12., 12., 14., 15.])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(module, "memory_stats", lambda: {"rss_bytes": 100})
    counters = {"cache_hits": 10, "cache_misses": 5, "checkpoint_bytes_read": 1000, "kernel_dispatches": 8, "batch_retirements": 4}
    engine = SimpleNamespace(stats=lambda: dict(counters))
    stats = ChatStats(engine)
    counters.update(cache_hits=13, cache_misses=6, checkpoint_bytes_read=1500, kernel_dispatches=11, batch_retirements=6)
    stats.on_token(1)
    stats.on_token(2)
    stats.emit("complete", {"usage": {"prompt_tokens": 4}})
    lines = capsys.readouterr().err.splitlines()
    result = json.loads(lines[-1].removeprefix("[edge0 stats] "))
    assert result["cache_hits"] == 3
    assert result["kernel_dispatches"] == 3 and result["batch_retirements"] == 2
    assert result["checkpoint_bytes_read"] == 500
    assert result["resident_cache_hit_fraction"] == .75
    assert result["time_to_first_token_s"] == 2
    assert result["decode_tokens_per_s_after_first"] == .5
