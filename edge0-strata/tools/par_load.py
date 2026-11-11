#!/usr/bin/env python3
"""Shared load harness for the parallelism program (research tool, standard library only).

Wraps the OpenAI-compatible server the way tools/serve_load.py does, and adds what an SLO-capacity harness needs:

  * closed-loop concurrency staircases (fixed work, shared wall window incl. the final drain),
  * open-loop seeded Poisson arrivals (offered rate independent of completions; late/unfinished requests are failures),
  * token counts from the server's `usage` (completion_tokens / prompt_tokens), NEVER from SSE chunks; a request with
    no usage is marked `tokens_source: "sse"` and the run is flagged invalid,
  * per request: TTFT, TPOT, max gap between content chunks (a chunk can hold several tokens when the engine flushes
    speculative windows, so "gap" is a chunk gap), client scheduling lag, finish reason, cached prompt tokens,
  * aggregate tok/s = total committed completion tokens / ONE shared wall-clock window (first submit to last completion),
  * the workloads W1 solo, W2 short concurrent, W4 long joiner, W6 shared-prefix cohorts,
  * one JSON ledger per run (args, server status before/after, per-request records, per-level blocks).

    python3 tools/par_load.py http://127.0.0.1:8080 --workload w2 --conc 1,2,4,8 --rounds 3 --out run.json
    python3 tools/par_load.py URL --workload w1 --prompt-tokens 512 --max-tokens 512 --rounds 3 --out w1.json
    python3 tools/par_load.py URL --workload w4 --conc 4 --joiner-tokens 32768 --out w4.json
    python3 tools/par_load.py URL --workload w6 --conc 4 --share 0,0.5,0.9 --out w6.json
    python3 tools/par_load.py URL --workload w2 --open-rates 0.25,0.5,1 --duration 120 --slo-ttft 10 --out open.json

Labelling for the paired analysis (tools/par_analyze.py): --arm A|B --pair N --label "free text".  The ledger's
`blocks` list holds one entry per (workload, level) with the headline metrics.  See Desktop 30thOctober results/par/HARNESS.md.
Prompts are deterministic from --seed and carry a unique nonce, so the prompt cache cannot help unless W6 asks for a shared prefix.
"""
import argparse, http.client, json, math, os, platform, random, statistics, sys, threading, time, urllib.parse, urllib.request

HARNESS_VERSION = "1.0"
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand "
         "part place case week company system program question work government number night point home water room mother area "
         "money story fact month lot right study book eye job word business issue side kind head house service friend father "
         "power hour game line end member law car city community name president team minute idea kid body information back "
         "parent face others level office door health person art war history party result change morning reason research girl "
         "guy moment air teacher force education foot boy age policy music market sense nation plan college interest death "
         "experience effect use class control care field development role effort rate heart drug show leader light voice wife "
         "police mind price report decision son view relationship town road arm difference value building action model season "
         "society tax director position player record paper space ground form event official matter center couple site project "
         "activity star table need court oil situation cost industry figure street image phone data picture practice piece "
         "land product doctor wall patient worker news test movie north love support technology step baby computer type "
         "attention film tree source organization hair window culture chance brother energy period course summer plant").split()
TOPICS = ["the history of lighthouses", "how a mechanical watch keeps time", "why bridges are built as arches",
          "the water cycle and droughts", "how bread rises", "the design of a medieval castle",
          "how a transistor switches", "the life of a honey bee colony"]


# ----------------------------------------------------------------------------------------------------------- prompts
class Prompter:
    """Deterministic filler text of a target token count.  The words-per-token ratio is calibrated once against the
    server's own usage.prompt_tokens (a max_tokens=1 probe), so the target counts tokens after the real template."""

    def __init__(self, url, key, model, seed, think, calib=None):
        self.url, self.key, self.model, self.seed, self.think = url, key, model, seed, think
        self.tok_per_word = (calib or {}).get("tok_per_word")
        self.overhead = (calib or {}).get("overhead", 40)
        self.calib = calib or {}

    def words(self, n, rng):
        return " ".join(rng.choice(WORDS) for _ in range(n))

    def body_text(self, key, n_words):
        return self.words(n_words, random.Random(f"{self.seed}/{key}"))

    def calibrate(self):
        if self.tok_per_word:
            return
        n = 1500
        text = "[calibration] " + self.body_text("calib", n) + "\nSay ok."
        u = chat_once(self.url, self.key, self.model, text, 1, self.think)
        pt = u.get("usage", {}).get("prompt_tokens")
        if not pt:
            self.tok_per_word, self.overhead = 1.3, 40            # server gave no usage: a guess, the ledger says so
            self.calib = {"tok_per_word": 1.3, "overhead": 40, "guess": True}
            return
        # prompt = template overhead (about 25-40 tokens) + n words; assume the overhead is the same as the tail
        self.overhead = 40
        self.tok_per_word = max(0.5, (pt - self.overhead) / n)
        self.calib = {"tok_per_word": self.tok_per_word, "overhead": self.overhead, "probe_prompt_tokens": pt, "probe_words": n}

    def make(self, key, target_tokens, prefix=None, share=0.0, tail="Now write a long, detailed essay about "):
        """A prompt of ~target_tokens.  With prefix (a stable string) and share: the first share*target tokens are
        the prefix's words (the same text for the whole cohort), the rest is unique to `key`."""
        self.calibrate()
        nw = max(8, int((target_tokens - self.overhead) / self.tok_per_word))
        topic = TOPICS[hash_int(key) % len(TOPICS)]
        nonce = f"[req {key}] "
        if prefix is not None and share > 0:
            ns = int(nw * share)
            pre = self.body_text("prefix/" + prefix, ns)
            rest = self.body_text(key, nw - ns)
            return pre + "\n" + nonce + rest + f"\n{tail}{topic}."
        return nonce + self.body_text(key, nw) + f"\n{tail}{topic}."


def hash_int(s):
    h = 1469598103934665603
    for ch in s.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ------------------------------------------------------------------------------------------------------------ request
def _conn(url):
    u = urllib.parse.urlparse(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=1800), u.path.rstrip("/")


def _body(model, text, max_tokens, think, stream, extra=None):
    b = {"model": model, "stream": stream, "max_tokens": max_tokens, "temperature": 0,
         "messages": [{"role": "user", "content": text}]}
    if stream:
        b["stream_options"] = {"include_usage": True}
    if not think:
        b["reasoning_effort"] = "none"
        b["chat_template_kwargs"] = {"enable_thinking": False}
    if extra:
        b.update(extra)
    return b


def chat_once(url, key, model, text, max_tokens, think):
    c, base = _conn(url)
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    c.request("POST", base + "/v1/chat/completions", json.dumps(_body(model, text, max_tokens, think, False)), h)
    r = c.getresponse()
    d = json.loads(r.read() or b"{}")
    c.close()
    return d


def stream_request(url, key, model, text, max_tokens, think, sched_t0, extra=None, keep_times=False, keep_text=False):
    """One streamed chat request.  Times are seconds on the perf_counter clock of this process; the record also
    carries them relative to the run start (t_rel_*) by the caller."""
    rec = {"t_sched": sched_t0, "t_send": None, "t_first": None, "t_last": None, "t_end": None, "completion_tokens": None,
           "prompt_tokens": None, "cached_tokens": 0, "finish": None, "tokens_source": None, "chunks": 0, "max_gap": 0.0,
           "gaps_ge1s": 0, "err": None, "timings": None, "reasoning_chars": 0, "content_chars": 0}
    times = [] if keep_times else None
    texts = [] if keep_text else None
    rec["t_send"] = time.perf_counter()
    last = None
    sse_tokens = 0
    try:
        c, base = _conn(url)
        h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
        c.request("POST", base + "/v1/chat/completions", json.dumps(_body(model, text, max_tokens, think, True, extra)), h)
        r = c.getresponse()
        if r.status != 200:
            rec["err"] = f"http {r.status}: {r.read(300).decode('utf-8', 'replace')}"
        else:
            while True:
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    j = json.loads(d)
                except ValueError:
                    continue
                now = time.perf_counter()
                if j.get("error"):
                    rec["err"] = str(j["error"])[:300]
                for ch in j.get("choices", []):
                    dl = ch.get("delta", {})
                    a, b = dl.get("content") or "", dl.get("reasoning_content") or ""
                    if a or b:
                        if rec["t_first"] is None:
                            rec["t_first"] = now
                        elif last is not None:
                            g = now - last
                            if g > rec["max_gap"]:
                                rec["max_gap"] = g
                            if g >= 1.0:
                                rec["gaps_ge1s"] += 1
                        last = now
                        rec["t_last"] = now
                        rec["chunks"] += 1
                        sse_tokens += 1
                        rec["content_chars"] += len(a)
                        if texts is not None:
                            texts.append(a + b)
                        rec["reasoning_chars"] += len(b)
                        if times is not None:
                            times.append(now)
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
                u = j.get("usage")
                if u:
                    rec["completion_tokens"] = u.get("completion_tokens")
                    rec["prompt_tokens"] = u.get("prompt_tokens")
                    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                    rec["tokens_source"] = "usage"
                if j.get("timings"):
                    rec["timings"] = j["timings"]
        c.close()
    except Exception as e:  # noqa: BLE001
        rec["err"] = repr(e)[:300]
    rec["t_end"] = time.perf_counter()
    if rec["completion_tokens"] is None:
        rec["completion_tokens"] = sse_tokens
        rec["tokens_source"] = "sse"                      # NOT a valid token count; the ledger marks the run invalid
    if times is not None:
        rec["chunk_times"] = times
    if texts is not None:
        rec["text"] = "".join(texts)
    return rec


def finish_record(rec, t0):
    """Derived per-request metrics, and times relative to the run start."""
    ttft = rec["t_first"] - rec["t_send"] if rec["t_first"] is not None else None
    n = rec["completion_tokens"] or 0
    tpot = (rec["t_last"] - rec["t_first"]) / (n - 1) if (ttft is not None and n > 1 and rec["t_last"] is not None) else None
    out = {k: rec[k] for k in ("completion_tokens", "prompt_tokens", "cached_tokens", "finish", "tokens_source", "chunks", "max_gap",
                               "gaps_ge1s", "err", "timings", "reasoning_chars", "content_chars")}
    out.update({"t_sched": rec["t_sched"] - t0, "t_send": rec["t_send"] - t0, "t_end": rec["t_end"] - t0,
                "t_first": (rec["t_first"] - t0) if rec["t_first"] is not None else None,
                "lag": rec["t_send"] - rec["t_sched"], "ttft": ttft, "tpot": tpot, "latency": rec["t_end"] - rec["t_send"]})
    if "text" in rec:
        out["text"] = rec["text"]
    if "chunk_times" in rec:
        out["chunk_t"] = [round(x - t0, 4) for x in rec["chunk_times"]]
    return out


# --------------------------------------------------------------------------------------------------------- aggregates
def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def aggregate(recs, window_s, slo=None):
    """Level metrics.  window_s = one shared wall window for all requests (first submit to last completion)."""
    ok = [r for r in recs if not r["err"]]
    toks = sum(r["completion_tokens"] or 0 for r in ok)
    agg = {"requests": len(recs), "errors": len(recs) - len(ok), "window_s": window_s, "tokens": toks,
           "agg_tps": toks / window_s if window_s > 0 else None,
           "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in ok),
           "invalid_token_source": sum(1 for r in recs if r["tokens_source"] != "usage"),
           "length_stops": sum(1 for r in ok if r["finish"] == "length"), "early_stops": sum(1 for r in ok if r["finish"] == "stop")}
    for k in ("ttft", "tpot", "max_gap", "latency", "lag"):
        v = [r[k] for r in ok]
        agg[k + "_p50"], agg[k + "_p95"], agg[k + "_max"] = pct(v, .5), pct(v, .95), (max(x for x in v if x is not None) if any(x is not None for x in v) else None)
    per = [(r["completion_tokens"] - 1) / (r["t_end"] - r["t_first"]) for r in ok
           if r["t_first"] is not None and (r["completion_tokens"] or 0) > 1 and r["t_end"] > r["t_first"]]
    agg["per_req_tps_p50"] = pct(per, .5)
    if slo:
        good = [r for r in ok if (slo.get("ttft") is None or (r["ttft"] is not None and r["ttft"] <= slo["ttft"]))
                and (slo.get("gap") is None or r["max_gap"] <= slo["gap"])]
        agg["slo"] = slo
        agg["slo_pass"] = len(good)
        agg["goodput_tps"] = sum(r["completion_tokens"] or 0 for r in good) / window_s if window_s > 0 else None
    return agg


# --------------------------------------------------------------------------------------------------------- runners
class Ctx:
    def __init__(self, a, prompter):
        self.a, self.p = a, prompter


def closed_loop(ctx, level, n_requests, make_job, label):
    """level concurrent clients pull jobs 0..n_requests-1 from one queue; the window is first submit -> last end."""
    a = ctx.a
    jobs = list(range(n_requests))
    lock = threading.Lock()
    recs = [None] * n_requests
    t0 = time.perf_counter()

    def worker():
        while True:
            with lock:
                if not jobs:
                    return
                i = jobs.pop(0)
            text, mt, extra = make_job(i)
            recs[i] = stream_request(a.url, a.key, a.model, text, mt, a.think, time.perf_counter(), extra, a.keep_times, a.keep_text)

    th = [threading.Thread(target=worker) for _ in range(level)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    t1 = max(r["t_end"] for r in recs)
    fr = [finish_record(r, t0) for r in recs]
    return fr, t1 - t0


def run_closed(ctx, wl, levels):
    a = ctx.a
    blocks, allreq = [], []
    for c in levels:
        n = max(c * a.rounds, a.min_requests)
        tag = f"{a.nonce}-{wl}-c{c}"

        def make_job(i, c=c, tag=tag):
            return ctx.p.make(f"{tag}-{i}", a.prompt_tokens), a.max_tokens, None

        if a.warm_requests:                                      # warmup outside the measured window
            for w in range(a.warm_requests):
                stream_request(a.url, a.key, a.model, ctx.p.make(f"{tag}-warm{w}", a.prompt_tokens), min(a.max_tokens, 64), a.think, time.perf_counter())
        fr, wall = closed_loop(ctx, c, n, make_job, tag)
        agg = aggregate(fr, wall)
        agg.update({"workload": wl, "level": c, "mode": "closed", "prompt_target": a.prompt_tokens, "max_tokens": a.max_tokens})
        blocks.append(agg)
        for i, r in enumerate(fr):
            r.update({"workload": wl, "level": c, "idx": i})
        allreq += fr
        print(f"{wl} c={c}: {agg['tokens']} tokens / {wall:.1f}s = {agg['agg_tps']:.1f} tok/s  ttft p50/p95 "
              f"{fmt(agg['ttft_p50'])}/{fmt(agg['ttft_p95'])}  tpot p50 {fmt(agg['tpot_p50'], 4)}  max_gap p95 {fmt(agg['max_gap_p95'])}"
              f"  err {agg['errors']}", flush=True)
    return blocks, allreq


def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def run_w4(ctx, levels):
    """W4: c W2-style decoders running, then one long joiner (joiner_tokens) arrives joiner_at seconds after they start."""
    a = ctx.a
    blocks, allreq = [], []
    for c in levels:
        for rd in range(a.rounds):
            tag = f"{a.nonce}-w4-c{c}-r{rd}"
            res = {}
            t0 = time.perf_counter()

            def dec(i):
                res[("d", i)] = stream_request(a.url, a.key, a.model, ctx.p.make(f"{tag}-d{i}", a.prompt_tokens), a.decoder_tokens, a.think, time.perf_counter(), None, a.keep_times or True)

            def joiner():
                time.sleep(a.joiner_at)
                res[("j", 0)] = stream_request(a.url, a.key, a.model, ctx.p.make(f"{tag}-j", a.joiner_tokens), a.joiner_out, a.think, time.perf_counter(), None, True)

            th = [threading.Thread(target=dec, args=(i,)) for i in range(c)] + [threading.Thread(target=joiner)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = max(r["t_end"] for r in res.values()) - t0
            fr = {k: finish_record(r, t0) for k, r in res.items()}
            j = fr[("j", 0)]
            ts_j = j["t_send"]
            te_j = j["t_first"] if j["t_first"] is not None else j["t_end"]
            # incumbent gaps between the joiner's submit and its first token (the interference window)
            gaps = []
            for (k, i), r in fr.items():
                if k != "d":
                    continue
                ct = [x for x in r.get("chunk_t", [])]
                for x, y in zip(ct, ct[1:]):
                    if y >= ts_j and x <= te_j + 1.0:
                        gaps.append(y - x)
            agg = aggregate(list(fr.values()), wall)
            allreq += [dict(r, workload="w4", level=c, round=rd, role=k[0]) for k, r in fr.items()]
            for r in allreq[-len(fr):]:
                r.pop("chunk_t", None) if not a.keep_times else None
            decs = [r for k, r in fr.items() if k[0] == "d"]
            agg.update({"workload": "w4", "level": c, "mode": "closed", "round": rd, "joiner_tokens": a.joiner_tokens,
                        "joiner_ttft": j["ttft"], "joiner_prompt_tokens": j["prompt_tokens"],
                        "incumbent_gap_max_during_join": max(gaps) if gaps else None,
                        "incumbent_gap_p99_during_join": pct(gaps, .99),
                        "incumbent_tps_p50": pct([(d["completion_tokens"] - 1) / max(d["t_end"] - d["t_first"], 1e-9) for d in decs if d["t_first"] is not None and d["completion_tokens"] and d["completion_tokens"] > 1], .5),
                        "makespan_s": wall})
            blocks.append(agg)
            print(f"w4 c={c} r={rd}: makespan {wall:.1f}s agg {fmt(agg['agg_tps'], 1)} tok/s, joiner ttft {fmt(j['ttft'])}s ({j['prompt_tokens']} prompt tok), "
                  f"incumbent max gap during join {fmt(agg['incumbent_gap_max_during_join'])}s", flush=True)
    return blocks, allreq


def run_w6(ctx, levels):
    a = ctx.a
    blocks, allreq = [], []
    for c in levels:
        for share in [float(x) for x in a.share.split(",")]:
            tag = f"{a.nonce}-w6-c{c}-s{share}"
            prefix = f"{tag}"
            if a.prime and share > 0:                           # prefix-warm: one priming request, not measured
                stream_request(a.url, a.key, a.model, ctx.p.make(f"{tag}-prime", a.prompt_tokens, prefix, share), 8, a.think, time.perf_counter())
            n = max(c * a.rounds, a.min_requests)

            def make_job(i, tag=tag, prefix=prefix, share=share):
                return ctx.p.make(f"{tag}-{i}", a.prompt_tokens, prefix, share), a.max_tokens, None

            fr, wall = closed_loop(ctx, c, n, make_job, tag)
            agg = aggregate(fr, wall)
            agg.update({"workload": "w6", "level": c, "share": share, "primed": bool(a.prime), "mode": "closed",
                        "cached_tokens_mean": statistics.mean(r["cached_tokens"] for r in fr) if fr else None})
            blocks.append(agg)
            for i, r in enumerate(fr):
                r.update({"workload": "w6", "level": c, "share": share, "idx": i})
            allreq += fr
            print(f"w6 c={c} share={share}: {fmt(agg['agg_tps'], 1)} tok/s ttft p50 {fmt(agg['ttft_p50'])} cached mean {agg['cached_tokens_mean']:.0f}", flush=True)
    return blocks, allreq


def run_open(ctx, wl, rates):
    """Open loop: seeded exponential inter-arrival times, independent of completions.  Requests still running at
    duration + drain_s are failures (kept in the ledger as unfinished)."""
    a = ctx.a
    blocks, allreq = [], []
    for rate in rates:
        rng = random.Random(f"{a.seed}/open/{rate}")
        sched, t = [], 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= a.duration:
                break
            sched.append(t)
        tag = f"{a.nonce}-{wl}-open{rate}"
        t0 = time.perf_counter()
        res = [None] * len(sched)
        th = []

        def one(i, at):
            res[i] = stream_request(a.url, a.key, a.model, ctx.p.make(f"{tag}-{i}", a.prompt_tokens), a.max_tokens, a.think, t0 + at, None, False)

        def launcher():
            for i, at in enumerate(sched):
                d = t0 + at - time.perf_counter()
                if d > 0:
                    time.sleep(d)
                t_ = threading.Thread(target=one, args=(i, at), daemon=True)
                t_.start()
                th.append(t_)

        lt = threading.Thread(target=launcher)
        lt.start()
        lt.join()
        deadline = t0 + a.duration + a.drain_s
        for t_ in th:
            t_.join(max(0.0, deadline - time.perf_counter()))
        done = [r for r in res if r is not None and r["t_end"] is not None]
        unfinished = len(sched) - len(done)
        window = max([r["t_end"] for r in done], default=t0) - t0
        fr = [finish_record(r, t0) for r in done]
        slo = {"ttft": a.slo_ttft, "gap": a.slo_gap}
        # tokens finished inside the arrival window over that window; the drain is reported separately
        inwin = [r for r in fr if r["t_end"] <= a.duration]
        agg = aggregate(fr, window, slo)
        agg["tps_in_window"] = sum(r["completion_tokens"] or 0 for r in inwin if not r["err"]) / a.duration
        agg.update({"workload": wl, "mode": "open", "offered_rate": rate, "offered_requests": len(sched), "unfinished": unfinished,
                    "duration": a.duration, "drain_s": a.drain_s, "achieved_rate": len(inwin) / a.duration,
                    "slo_pass_frac": (agg.get("slo_pass", 0) / len(sched)) if sched else None})
        blocks.append(agg)
        allreq += [dict(r, workload=wl, offered_rate=rate, idx=i) for i, r in enumerate(fr)]
        print(f"{wl} open rate={rate}/s: offered {len(sched)}, done {len(done)}, unfinished {unfinished}, slo pass {agg.get('slo_pass')}, "
              f"goodput {fmt(agg.get('goodput_tps'), 1)} tok/s, ttft p95 {fmt(agg['ttft_p95'])}, lag max {fmt(agg['lag_max'], 3)}", flush=True)
    return blocks, allreq


# ------------------------------------------------------------------------------------------------------------- main
def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--workload", default="w2", choices=["w1", "w2", "w3", "w4", "w6"])
    ap.add_argument("--conc", default="1,2,4,8", help="closed-loop concurrency levels (w1 forces 1)")
    ap.add_argument("--rounds", type=int, default=3, help="requests per client per level (a level runs conc*rounds requests)")
    ap.add_argument("--min-requests", type=int, default=0)
    ap.add_argument("--warm-requests", type=int, default=0, help="unmeasured short requests before each level")
    ap.add_argument("--prompt-tokens", type=int, default=None, help="default: w1 512, w2 1024, w3 8192, w6 2048")
    ap.add_argument("--max-tokens", type=int, default=None, help="default: w1 512, w2 256, w3 128, w6 128")
    ap.add_argument("--joiner-tokens", type=int, default=28000)
    ap.add_argument("--joiner-out", type=int, default=64)
    ap.add_argument("--joiner-at", type=float, default=6.0, help="seconds after the decoders start")
    ap.add_argument("--decoder-tokens", type=int, default=512)
    ap.add_argument("--share", default="0,0.5,0.9", help="w6 shared-prefix fractions")
    ap.add_argument("--prime", action="store_true", help="w6: one unmeasured request primes the prefix (prefix-warm)")
    ap.add_argument("--open-rates", default=None, help="open-loop offered rates (req/s), e.g. 0.25,0.5,1")
    ap.add_argument("--duration", type=float, default=120.0, help="open loop: arrival window (s)")
    ap.add_argument("--drain-s", type=float, default=120.0, help="open loop: extra time to finish before counting failures")
    ap.add_argument("--slo-ttft", type=float, default=None)
    ap.add_argument("--slo-gap", type=float, default=None)
    ap.add_argument("--seed", default="par29")
    ap.add_argument("--nonce", default=str(int(time.time())))
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--keep-times", action="store_true", help="keep every chunk timestamp in the ledger (large)")
    ap.add_argument("--keep-text", action="store_true", help="keep every answer's text in the ledger (identity checks)")
    ap.add_argument("--calib", default=None, help="a ledger/calibration JSON to reuse the words-per-token calibration from")
    ap.add_argument("--arm", default="")
    ap.add_argument("--pair", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--settings", default=None, help="file with the effective engine settings/provenance to embed (launcher writes it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.url = a.url.rstrip("/")
    d = {"w1": (512, 512), "w2": (1024, 256), "w3": (8192, 128), "w4": (1024, 256), "w6": (2048, 128)}[a.workload]
    a.prompt_tokens = a.prompt_tokens or d[0]
    a.max_tokens = a.max_tokens or d[1]
    levels = [1] if a.workload == "w1" else [int(x) for x in a.conc.split(",")]
    calib = None
    if a.calib and os.path.exists(a.calib):
        calib = json.load(open(a.calib)).get("calibration")
    p = Prompter(a.url, a.key, a.model, a.seed, a.think, calib)
    ctx = Ctx(a, p)
    status0 = get_json(a.url + "/v1/status")
    started = time.time()
    if a.open_rates:
        blocks, reqs = run_open(ctx, a.workload, [float(x) for x in a.open_rates.split(",")])
    elif a.workload == "w4":
        blocks, reqs = run_w4(ctx, levels)
    elif a.workload == "w6":
        blocks, reqs = run_w6(ctx, levels)
    else:
        blocks, reqs = run_closed(ctx, a.workload, levels)
    status1 = get_json(a.url + "/v1/status")
    settings = None
    if a.settings and os.path.exists(a.settings):
        settings = open(a.settings, errors="replace").read()[:20000]
    for b in blocks:
        b.update({"arm": a.arm, "pair": a.pair, "label": a.label})
    invalid = sum(b.get("invalid_token_source", 0) for b in blocks)
    ledger = {"harness": "par_load", "version": HARNESS_VERSION, "started": started, "host": platform.node(), "argv": sys.argv[1:],
              "args": {k: v for k, v in vars(a).items() if k != "key"}, "calibration": p.calib, "valid": invalid == 0,
              "invalid_reason": None if invalid == 0 else f"{invalid} requests without server usage: token counts are SSE chunks",
              "status_before": status0, "status_after": status1, "settings": settings, "blocks": blocks, "requests": reqs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(ledger, open(a.out, "w"), indent=1)
    print("ledger:", a.out, "valid" if invalid == 0 else "INVALID (no usage)")
    return 0 if invalid == 0 and all(b["errors"] == 0 for b in blocks) else 1


if __name__ == "__main__":
    sys.exit(main())
