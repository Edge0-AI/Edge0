#!/usr/bin/env python3
"""mkprompts.py: the 10 greedy prompts of the llama.cpp reference run (runs/base1/answers_{chat,think}) as token ids.

Renders each request with the GGUF's chat template (jinja2, thinking_mode chat | thinking), tokenizes with llama-tokenize
(--no-bos (special tokens parse by default): the template writes the BOS text itself), and keeps the reference's greedy output ids.
Writes OUT/prompts.json: [{"id", "mode", "ids": [...], "expect_ids": [...], "expect_text", "finish"}].
"""
import json, os, subprocess, sys, tempfile
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

ref_dir, template, shard, tokenize, out_dir = sys.argv[1:6]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
def raise_exception(m):
    raise jinja2.exceptions.TemplateError(m)
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(template, encoding="utf-8").read())
libs = os.environ.get("LD_LIBRARY_PATH", "")

def tok(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        p = f.name
    r = subprocess.run([tokenize, "-m", shard, "-f", p, "--ids", "--no-bos", "--no-escape"],
                       capture_output=True, text=True, env=dict(os.environ, LD_LIBRARY_PATH=libs))
    os.unlink(p)
    line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
    return json.loads(line)

out = []
for mode_dir, mode in (("answers_chat", "chat"), ("answers_think", "thinking")):
    d = os.path.join(ref_dir, mode_dir)
    for fn in sorted(os.listdir(d)):
        if fn == "summary.json" or not fn.endswith(".json"):
            continue
        r = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        b = r["request_body"]
        text = tpl.render(messages=b["messages"], tools=b.get("tools"), thinking_mode=mode, reasoning_effort=None,
                          add_generation_prompt=True)
        ids = tok(text)
        ch = r["response"]["choices"][0]
        out.append({"id": fn[:-5], "mode": mode, "ids": ids, "max_tokens": b.get("max_tokens", 256),
                    "expect_ids": [t["id"] for t in ch["logprobs"]["content"]],
                    "expect_text": ch["message"].get("content"), "finish": ch["finish_reason"]})
        print(mode, fn[:-5], len(ids), "prompt tokens;", len(out[-1]["expect_ids"]), "expected")
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, "prompts.json"), "w"))
