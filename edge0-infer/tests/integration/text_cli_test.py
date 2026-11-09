"""Public CLI checks against committed, immutable text references."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

cli, api, fixtures = sys.argv[1:]
root = os.environ.get("MOE_TINY")
if not root:
    raise SystemExit("MOE_TINY is required")
model = Path(root) / "tiny-a"
fixtures = Path(fixtures)

def run(*args, code=0):
    result = subprocess.run([cli, *map(str, args)], capture_output=True)
    assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)
    return result

def tokenize(text, *flags):
    return run("tokenize", "--model", model, *flags, text).stdout

corpus = json.loads((fixtures / "tokenize_golden.json").read_text())

# Exercise both parse-special policies, raw non-NFC input, and escaped pieces.
rows = corpus["rows"]
selected = [r for r in rows if r["id"] in {"manual-000", "manual-043"}]
selected += [r for r in rows if "<|im_end|>" in r["text"]][:2]
for row in selected:
    flags = ["--parse-special"] if row["parse_special"] else []
    actual = tokenize(row["text"], *flags)
    assert [int(i) for i in actual.split()] == row["llama"], row["id"]

# The escaping is deliberately ASCII and reversible even for partial UTF-8 bytes.
def unescape(value):
    out = bytearray()
    i = 0
    while i < len(value):
        if value[i] != 92:
            out.append(value[i]); i += 1
        elif value[i + 1] == ord("x"):
            out.append(int(value[i + 2:i + 4], 16)); i += 4
        else:
            out.append({92: 92, ord("n"): 10, ord("r"): 13, ord("t"): 9}[value[i + 1]])
            i += 2
    return bytes(out)

piece_input = "A\n\r\t\\" + chr(0x4f60) + chr(0x1f600)
piece_output = tokenize(piece_input, "--pieces")
piece_rows = [line.split(b"\t", 1) for line in piece_output.splitlines()]
assert all(len(row) == 2 and all(c < 128 for c in row[1]) for row in piece_rows)
assert b"".join(unescape(row[1]) for row in piece_rows) == piece_input.encode()
assert [int(row[0]) for row in piece_rows] == [int(i) for i in tokenize(piece_input).split()]
assert b"\\n" in piece_output and b"\\r" in piece_output and b"\\t" in piece_output
assert tokenize("<|im_end|>", "--parse-special", "--pieces").endswith(b"\t<|im_end|>\n")
assert tokenize("") == b"\n"
assert tokenize("", "--pieces") == b""

templates = json.loads((fixtures / "template_golden.json").read_text())["rows"]
for row in templates:
    # CLI represents the specified system + single user input; richer histories use the API.
    messages = row["messages"]
    if not (len(messages) in (1, 2) and messages[-1]["role"] == "user"
            and (len(messages) == 1 or messages[0]["role"] == "system")):
        continue
    args = ["prompt", "--model", model]
    if len(messages) == 2:
        args += ["--system", messages[0]["content"]]
    args += ["--user", messages[-1]["content"]]
    assert run(*args).stdout == row["text"].encode(), row["id"]
    with_tokens = run(*args, "--tokens").stdout
    expected_ids = " ".join(map(str, row["tokens"])).encode()
    assert with_tokens == row["text"].encode() + b"\n" + expected_ids + b"\n", row["id"]

for command in ("tokenize", "prompt"):
    for args in ([], ["--model"], ["--model", model], ["--bad"],
                 ["--model", model, "--model", model],
                 ["--model", model, "--pieces", "--pieces", "x"]):
        assert run(command, *args, code=2).stdout == b""
run("tokenize", "--model", model, "one", "two", code=2)
run("prompt", "--model", model, "--user", "x", "--user", "y", code=2)
run("tokenize", "--model", model / "missing", "x", code=1)

subprocess.run([api, str(model)], check=True)

# Copy every file before mutation: never write through to the immutable external asset.
with tempfile.TemporaryDirectory(prefix="moe-text-cli-") as temporary:
    altered = Path(temporary) / "tiny-a"
    shutil.copytree(model, altered)
    template = altered / "chat_template.jinja"
    original = template.read_bytes()
    manifest = (altered / "assets.tsv").read_bytes()
    template.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert template.stat().st_size == len(original)
    assert (altered / "assets.tsv").read_bytes() == manifest
    for command, args in (("inspect", []), ("tokenize", ["hello"])):
        result = run(command, "--model", altered, *args, code=1)
        assert b"incompatible" in result.stderr and b"template identity mismatch" in result.stderr
        assert b"expected" in result.stderr and b"actual" in result.stderr
        assert result.stdout == b""
    template.write_bytes(original)
    run("inspect", "--model", altered)
    assert run("tokenize", "--model", altered, "hello").stdout == tokenize("hello")
    assert (altered / "assets.tsv").read_bytes() == manifest
print("CLI golden tokens, prompt bytes, escaping, errors, and template tamper checks passed")
