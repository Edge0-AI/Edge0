#!/usr/bin/env python3
"""mkimg.py OUT_DIR: a 672x672 test picture (left half red, right half blue, a black disc in the middle), its DSI1 file and
the prompt ids (chat mode, GGUF template) with the picture's span (16 x 17 + 2 = 274 positions of the image token) in them.
Writes OUT_DIR/img.dsi and OUT_DIR/img_prompts.json in serve_prompts.py's format ("pic": file)."""
import json, os, struct, subprocess, sys, tempfile
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
out, template, shard, tokenize = sys.argv[1:5]
W = H = 672
rows = bytearray()
for y in range(H):
    for x in range(W):
        if (x - 336) ** 2 + (y - 336) ** 2 < 80 ** 2: rows += bytes((0, 0, 0))
        elif x < 336: rows += bytes((220, 20, 20))
        else: rows += bytes((20, 20, 220))
n_h = n_w = 672 // 42
span = n_h * (n_w + 1) + 2
os.makedirs(out, exist_ok=True)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(m))
tpl = env.from_string(open(template, encoding="utf-8").read())
text = tpl.render(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}},
                  {"type": "text", "text": "What colors are the left half and the right half of this picture? Answer in one short sentence."}]}],
                  tools=None, thinking_mode="chat", reasoning_effort=None, add_generation_prompt=True)
PH = "<\uff5cdeepseek_image\uff5c>"
assert text.count(PH) == 1, text[:300]
a, b = text.split(PH)
def tok(t):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(t); p = f.name
    r = subprocess.run([tokenize, "-m", shard, "-f", p, "--ids", "--no-bos", "--no-escape"], capture_output=True, text=True)
    os.unlink(p)
    return json.loads([l for l in r.stdout.splitlines() if l.startswith("[")][-1])
ia, ib = tok(a), tok(b)
ids = ia + [129264] * span + ib
open(os.path.join(out, "img.dsi"), "wb").write(b"DSI1" + struct.pack("<i", 1) + struct.pack("<qii", len(ia), W, H) + bytes(rows))
json.dump([{"id": "image-colors", "mode": "chat", "ids": ids, "max_tokens": 40, "expect_ids": [],
            "pic": os.path.join(out, "img.dsi")}], open(os.path.join(out, "img_prompts.json"), "w"))
print("prompt", len(ids), "tokens, picture span at", len(ia), "length", span)
