"""Builds the chat-templated prompt ids for the DFlash2 acceptance experiment (same tokenizer and template the server uses)."""
import json, sys
from pathlib import Path

REPO = Path(r"C:\Users\AI-Server\Desktop\Strata\GitHub\wt-022-dflash2")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))
import strata_tokenizer as ST
from serve.frontend import ChatTemplate

tp = Path(r"C:\dflash2-work\tok")
vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = (tp / "merges.txt").read_text(encoding="utf-8").split("\n")
types = json.loads((tp / "token_type.json").read_text())
tok = ST.Tokenizer(tokens, merges, types)
tpl = ChatTemplate(tp / "chat_template.jinja")

readme = (REPO / "README.md").read_text(encoding="utf-8")[:2600]

PROMPTS = {
    "chat_explain": "Explain how a hash map works and why lookups are O(1) on average. Then say when that guarantee breaks down. Keep it clear for a junior developer.",
    "code_py": "Write a Python function that reads a web server log file, parses the timestamps (ISO-8601) and returns the five busiest one-minute windows with their request counts. Include a small test with pytest.",
    "code_cpp": "Implement an LRU cache in C++17 with O(1) get and put using std::list and std::unordered_map. Show the class and a short main() that exercises it.",
    "math": "A train leaves city A at 9:00 traveling at 80 km/h. A second train leaves city B, 450 km away, at 10:30 traveling toward A at 95 km/h. At what time do they meet, and how far from city A? Work it out step by step and check the result.",
    "summarize": "Summarize the following text in exactly five bullet points, then give a one-sentence takeaway.\n\n" + readme,
    "json_extract": "Extract the data below as JSON with the keys name, role, city, start_date (ISO format) and skills (a list). Output only the JSON.\n\nMaria Gonzalez joined the company on the 3rd of March 2021 as a senior data engineer. She works from the Lisbon office and her main tools are Spark, dbt, Airflow and a bit of Rust. Her colleague Tom Becker, a product designer from Vienna, started a year later, on 14 June 2022, and is good at Figma, user research and prototyping.",
    "creative": "Write a short story (about 400 words) about a lighthouse keeper who discovers that the light has been answering messages from the sea. Give it a quiet, slightly eerie tone.",
    "chat_advice": "I have a 12 GB graphics card and 64 GB of RAM. What are the trade-offs between running a large mixture-of-experts model with most of its experts in RAM versus a smaller dense model fully on the GPU? Give me a recommendation for daily coding help.",
}

out = {}
for name, text in PROMPTS.items():
    s = tpl.render([{"role": "user", "content": text}])
    ids = tok.encode(s, parse_special=True)
    out[name] = {"ids": ids, "text": s}
    print(name, len(ids), repr(s[-60:]))
json.dump(out, open(r"C:\dflash2-work\prompts.json", "w"))
