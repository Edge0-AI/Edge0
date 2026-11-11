#!/usr/bin/env python3
"""复现并诊断 needle_bench 的 512k 档失败（HTTP 400），同时兼作重跑。
失败时打印**响应体**（官方工具只打印 "Bad Request"，看不到原因）。
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.environ.get("STRATA_TOOLS", "tools"))   # run from the repository root
import needle_bench as nb  # noqa: E402

URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8081").rstrip("/") + "/v1/chat/completions"
KEY = os.environ.get("STRATA_KEY", "")
TARGET = int(512 * 1024 * 0.98)          # 工具的口径：512k × 0.98
# ⚠️ 官方工具按 CHARS_PER_TOKEN=3.2 估算**偏乐观**：512k 档目标 513,802 tok 实造 1,644,166 字符，
#    服务端实测 **524,392 tokens**（真实比率 ≈3.135）⇒ prompt+40 > 524,288，被 **400 拒**（never truncated）。
#    ⇒ 这里再打 0.98 折（≈1,611,000 字符），预计落到 ~515K tokens，才是名副其实的「512K 档」。
CHARS = int(TARGET * nb.CHARS_PER_TOKEN * 0.98)
WORD = "harbor-velvet-512"

text = nb.haystack(CHARS)
cut = text.rfind("\n", 0, len(text) // 2) + 1
needle = f"\nThe secret code word for this text is: {WORD}. Remember it.\n"
prompt = (text[:cut] + needle + text[cut:] +
          "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
print(f"目标 {TARGET:,} tok · haystack {CHARS:,} 字符 · 实际 prompt {len(prompt):,} 字符", flush=True)

body = {"model": "strata", "max_tokens": 40, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]}
data = json.dumps(body).encode()
print(f"请求体 {len(data):,} 字节", flush=True)
req = urllib.request.Request(URL, data=data,
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ans = (out["choices"][0]["message"].get("content") or "")
    tim = out.get("timings") or {}
    print(f"✅ HTTP 200 · {dt:.0f} s · prompt_tokens={out['usage']['prompt_tokens']:,} "
          f"· prompt_tok_s={tim.get('prompt_per_second')}")
    print(f"   {'FOUND' if WORD in ans else 'MISSED'}  answer={ans.strip()[:80]!r}")
except urllib.error.HTTPError as e:
    print(f"❌ HTTP {e.code} · {time.time()-t0:.1f} s")
    print("响应体:", e.read().decode("utf-8", "replace")[:1000])
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
