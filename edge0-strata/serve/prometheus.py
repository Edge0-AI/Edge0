"""GET /metrics in the Prometheus text format, with vLLM's metric names, so the dashboards and alerts written for a
vLLM server read a Strata server unchanged.  Everything comes from the server's own record (Service.metrics(), the
same dict the JSON /metrics returns) plus three latency histograms observed when a request finishes.

What vLLM has no name for is the JSON's own facts under `strata:`, named after the JSON's keys (live.tok_s ->
strata:live_tok_s, totals.decode_ms -> strata:totals_decode_seconds_total, hardware.gpu_util -> strata:gpu_util), so
the two formats say the same thing and a panel can be moved from the Monitor tab to Grafana by its key."""
import threading

# vLLM's buckets for the request latencies (seconds)
BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0,
           80.0, 160.0, 640.0, 2560.0)
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def wants_prometheus(accept: str, query: str) -> bool:
    """Prometheus asks with Accept: text/plain or application/openmetrics-text; a person can add ?format=prometheus.
    Anything else (the Monitor tab, curl without a header) keeps the JSON."""
    return "format=prometheus" in (query or "") or "openmetrics" in (accept or "") or "text/plain" in (accept or "")


class Histogram:
    def __init__(self):
        self.counts = [0] * len(BUCKETS)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float, times: int = 1):
        if v is None or times <= 0:
            return
        for i, b in enumerate(BUCKETS):
            if v <= b:
                self.counts[i] += times
        self.sum += v * times
        self.n += times


class Latencies:
    """Time to first token, the time between tokens and the whole request, per finished request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ttft, self.itl, self.e2e = Histogram(), Histogram(), Histogram()

    def observe(self, ttft_s, output_tokens, decode_s, total_s):
        with self.lock:
            self.ttft.observe(ttft_s)
            if output_tokens and output_tokens > 1 and decode_s:
                # the mean gap, once per gap: the engine reports the decode time, not each token's
                self.itl.observe(decode_s / (output_tokens - 1), output_tokens - 1)
            self.e2e.observe(total_s)

    def snapshot(self):
        with self.lock:
            return {k: (list(h.counts), h.sum, h.n) for k, h in
                    (("ttft", self.ttft), ("itl", self.itl), ("e2e", self.e2e))}


def _num(v, default=0):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def render(m: dict, lat: dict) -> str:
    """The text for one scrape.  `m` is Service.metrics(), `lat` Latencies.snapshot()."""
    model = str((m.get("engine") or {}).get("model") or "strata").replace("\\", "\\\\").replace('"', '\\"')
    lab = f'model_name="{model}"'
    live, totals = m.get("live") or {}, m.get("totals") or {}
    out = []

    def metric(name, kind, help_, value):
        out.append(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name}{{{lab}}} {value}")

    busy = live.get("state") in ("reading", "generating")
    ctx = _num((m.get("engine") or {}).get("max_context"), 0)
    used = _num(live.get("prompt_tokens")) + _num(live.get("generated")) if busy else 0
    # "parallel": live.running / live.waiting count the requests together (waiting: for a slot or the control lines)
    metric("vllm:num_requests_running", "gauge", "Requests reading their prompt or generating.",
           _num(live.get("running"), int(busy)))
    metric("vllm:num_requests_waiting", "gauge", "Requests waiting for their turn.",
           _num(live.get("queued")) + _num(live.get("waiting")))
    metric("vllm:kv_cache_usage_perc", "gauge", "The running request's share of the context (1 = full).",
           round(min(1.0, used / ctx), 4) if ctx else 0)
    metric("vllm:prompt_tokens_total", "counter", "Prompt tokens of the finished requests.",
           _num(totals.get("prompt_tokens")))
    metric("vllm:generation_tokens_total", "counter", "Generated tokens.", _num(totals.get("output_tokens")))
    metric("vllm:request_success_total", "counter", "Finished requests.", _num(totals.get("requests")))
    metric("vllm:prefix_cache_queries_total", "counter", "Prompt tokens looked up in the prompt cache.",
           _num(totals.get("prompt_tokens")))
    metric("vllm:prefix_cache_hits_total", "counter", "Prompt tokens the prompt cache already held.",
           _num(totals.get("reused")))
    metric("vllm:spec_decode_num_draft_tokens_total", "counter", "MTP draft tokens offered.",
           _num(totals.get("drafts_offered")))
    metric("vllm:spec_decode_num_accepted_tokens_total", "counter", "MTP draft tokens accepted.",
           _num(totals.get("drafts_accepted")))
    metric("vllm:num_preemptions_total", "counter", "Preempted requests (Strata does not preempt).", 0)
    for name, key, help_ in (("vllm:time_to_first_token_seconds", "ttft", "Time to the first token."),
                             ("vllm:inter_token_latency_seconds", "itl", "Time between two tokens."),
                             ("vllm:e2e_request_latency_seconds", "e2e", "A request from start to end.")):
        counts, total, n = lat[key]
        out.append(f"# HELP {name} {help_}\n# TYPE {name} histogram")
        for b, c in zip(BUCKETS, counts):
            out.append(f'{name}_bucket{{{lab},le="{b}"}} {c}')
        out.append(f'{name}_bucket{{{lab},le="+Inf"}} {n}')
        out.append(f"{name}_sum{{{lab}}} {round(total, 6)}\n{name}_count{{{lab}}} {n}")
    strata(m, out, lab)
    return "\n".join(out) + "\n"


def strata(m: dict, out: list, lab: str):
    """The JSON's facts vLLM has no name for, under `strata:` and the JSON's key."""
    live, totals = m.get("live") or {}, m.get("totals") or {}
    hw, eng = m.get("hardware") or {}, m.get("engine") or {}
    reqs = m.get("requests") or []

    typed = set()

    def metric(name, kind, help_, value, labels=""):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return                                       # unknown here (no GPU telemetry, an older engine): no sample
        if name not in typed:                            # one HELP / TYPE per family, before its first sample
            typed.add(name)
            out.append(f"# HELP strata:{name} {help_}\n# TYPE strata:{name} {kind}")
        out.append(f"strata:{name}{{{lab}{labels}}} {value}")

    for state in ("unloaded", "idle", "reading", "generating"):
        metric("live_state", "gauge", "1 for what the engine is doing (live.state).", int(live.get("state") == state),
               f',state="{state}"')
    metric("live_tok_s", "gauge", "Decode rate over the last seconds (live.tok_s).", live.get("tok_s") or 0)
    metric("live_prefill_tok_s_mean", "gauge", "Prompt reading rate of the running request (live.prefill_tok_s_mean).",
           live.get("prefill_tok_s_mean") or 0)
    metric("live_prompt_read", "gauge", "Prompt tokens read so far by the running request (live.prompt_read).",
           live.get("prompt_read") or 0)
    for state in ("idle", "reading", "generating"):
        slots = live.get("slots")
        if isinstance(slots, list) and slots:
            metric("live_slots", "gauge", "The batch slots in each state (live.slots).",
                   sum(1 for x in slots if isinstance(x, dict) and x.get("state") == state), f',state="{state}"')
    metric("engine_max_context", "gauge", "The engine's context (engine.max_context).", eng.get("max_context"))
    metric("totals_prompt_seconds_total", "counter", "Time spent reading prompts (totals.prompt_ms).",
           round((totals.get("prompt_ms") or 0) / 1000, 3))
    metric("totals_decode_seconds_total", "counter", "Time spent generating (totals.decode_ms).",
           round((totals.get("decode_ms") or 0) / 1000, 3))
    last = reqs[0] if reqs else {}
    metric("last_hit_rate", "gauge", "The last request's expert cache hit rate (requests[0].hit_rate).",
           last.get("hit_rate"))
    metric("last_decode_tok_s", "gauge", "The last request's decode rate (requests[0].decode_tok_s).",
           last.get("decode_tok_s"))
    gpus = hw.get("gpus") or ([{"index": 0, "util": hw.get("gpu_util"), "mem_used": hw.get("gpu_mem_used"),
                                "mem_total": hw.get("gpu_mem_total"), "temp": hw.get("gpu_temp"),
                                "power": hw.get("gpu_power")}] if hw.get("gpu_util") is not None else [])
    for name, key, help_ in (("gpu_util", "util", "GPU busy, percent (hardware.gpu_util)."),
                             ("gpu_mem_used_bytes", "mem_used", "GPU memory in use (hardware.gpu_mem_used)."),
                             ("gpu_mem_total_bytes", "mem_total", "GPU memory (hardware.gpu_mem_total)."),
                             ("gpu_temp_celsius", "temp", "GPU temperature (hardware.gpu_temp)."),
                             ("gpu_power_watts", "power", "GPU power draw (hardware.gpu_power).")):
        for g in gpus:                                   # a family's samples together, one per card
            metric(name, "gauge", help_, g.get(key), f',gpu="{g.get("index")}"')
    metric("cpu", "gauge", "CPU busy, percent (hardware.cpu).", hw.get("cpu"))
    metric("ram_used_bytes", "gauge", "RAM in use (hardware.ram_used).", hw.get("ram_used"))
    metric("ram_total_bytes", "gauge", "RAM (hardware.ram_total).", hw.get("ram_total"))
