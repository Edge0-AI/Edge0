"""Strata 服务端测速（沿用 NInfer-Offload 的 4 个提示词，便于对比）。

  python3 strata_bench.py PORT LABEL [--reps 2] [--max-tokens 512] [--prefill-tokens 3300]

每个请求：贪心（temperature 0），不思考（enable_thinking=False），记录
解码 tok/s、预填充 tok/s、草稿提供/接受数、GPU 专家缓存命中率、PCIe 份额（来自 GET /metrics）。
结果每行一个 JSON，最后打印各提示词平均。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import EN, PROMPTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("label")
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--prefill-tokens", type=int, default=0, help="另测一个约这么多 token 的长提示词的预填充（0 = 不测）")
a = ap.parse_args()
base = f"http://127.0.0.1:{a.port}"


def get(path):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return json.loads(r.read())


model = get("/v1/models")["data"][0]["id"]


def req(content, n):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as resp:
        d = json.loads(resp.read())
    wall = time.time() - t0
    m = (get("/metrics").get("requests") or [{}])[0]
    return d, m, wall


def record(name, rep, d, m, wall):
    t = d.get("timings") or {}
    rec = {"cfg": a.label, "rep": rep, "prompt": name,
           "prompt_n": t.get("prompt_n"), "reused": m.get("reused"),
           "prefill_tps": t.get("prompt_per_second"),
           "gen_tokens": d["usage"]["completion_tokens"], "decode_tps": t.get("predicted_per_second"),
           "drafts_offered": m.get("drafts_offered"), "drafts_accepted": m.get("drafts_accepted"),
           "hit_rate": m.get("hit_rate"), "pcie_share": m.get("pcie_share"),
           "ram_blobs": m.get("ram_blobs"), "file_blobs": m.get("file_blobs"),
           "wall_s": round(wall, 1), "head": (d["choices"][0]["message"].get("content") or "")[:60].replace("\n", " ")}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


req("你好", 16)  # 预热
recs = []
for rep in range(a.reps):
    for name, p in PROMPTS:
        recs.append(record(name, rep, *req(p, a.max_tokens)))
if a.prefill_tokens:
    # EN 一段约 90 token；只要大致长度，实际 token 数见 prompt_n
    sep = chr(10) * 2
    long_p = sep.join([f"[{i}] " + EN for i in range(max(1, a.prefill_tokens // 90))]) + sep + "用一句话总结上文。"
    for rep in range(a.reps):
        recs.append(record("prefill", rep, *req(long_p + f"（第 {rep} 次，编号 {time.time()}）", 16)))

print("---- 平均（每个提示词 %d 遍）" % a.reps)
for name in dict.fromkeys(r["prompt"] for r in recs):
    rs = [r for r in recs if r["prompt"] == name]
    avg = lambda k: round(sum(r[k] or 0 for r in rs) / len(rs), 2)  # noqa: E731
    acc = sum(r["drafts_accepted"] or 0 for r in rs) / max(1, sum(r["drafts_offered"] or 0 for r in rs))
    print(json.dumps({"cfg": a.label, "prompt": name, "decode_tps": avg("decode_tps"), "prefill_tps": avg("prefill_tps"),
                      "hit_rate": avg("hit_rate"), "pcie_share": avg("pcie_share"), "draft_accept": round(acc, 3),
                      "prompt_n": rs[0]["prompt_n"]}, ensure_ascii=False))
dec = [r["decode_tps"] or 0 for r in recs if r["prompt"] != "prefill"]
print(json.dumps({"cfg": a.label, "decode_tps_mean_4prompts": round(sum(dec) / max(1, len(dec)), 2)}))
