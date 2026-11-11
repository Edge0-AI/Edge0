"""The four decode-speed evaluation prompts (essay, code edit, translation, explanation)."""

CODE = '''import os, json

def load(path):
    with open(path) as f:
        return json.load(f)

def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out

def find_files(root, ext):
    result = []
    for d, _, files in os.walk(root):
        for name in files:
            if name.endswith(ext):
                result.append(os.path.join(d, name))
    return sorted(result)

def main(root):
    cfg = {}
    for p in find_files(root, ".json"):
        cfg = merge(cfg, load(p))
    save(os.path.join(root, "merged.json"), cfg)
    return len(cfg)
'''
EN = ("Mixture-of-experts models activate only a small subset of their parameters for each token, "
      "which makes them attractive for consumer hardware: the total parameter count can exceed the "
      "available GPU memory, while the per-token compute stays modest. The main difficulty is memory "
      "bandwidth. When expert weights live in system RAM, every generated token must stream the selected "
      "experts through the CPU memory bus or across PCIe, and this transfer, rather than arithmetic, "
      "usually determines decoding speed.")
PROMPTS = [
    ("essay", "用中文写一篇大约600字的散文，主题是秋天的城市。"),
    ("code_edit", "给下面代码的每个函数加上类型注解和一行中文 docstring，其他内容保持不变，输出完整代码：\n\n" + CODE),
    ("translate", "把下面这段英文翻译成中文：\n\n" + EN),
    ("explain", "详细解释 TCP 三次握手和四次挥手的过程，以及为什么需要这样设计。"),
]
