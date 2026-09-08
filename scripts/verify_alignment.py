#!/usr/bin/env python
"""Adapter alignment check: reference engines vs edge0.

Runs the SAME prompt through a reference deployment engine (an unmodified
upstream serving stack, production env) and through edge0, both greedy,
and writes both token sequences + decoded text to one JSON result file
(scripts/alignment_results.json, gitignored local evidence).  Identical
out-ids => the adapters mount identically.

Usage:
    python verify_alignment.py demo qwen
    python verify_alignment.py edge0 qwen
    python verify_alignment.py demo ling
    python verify_alignment.py edge0 ling

Result file: scripts/alignment_results.json (each run merges its entry).
"""
import argparse
import json
import os
import sys

N = 14  # max new tokens (greedy)
PROMPTS = {
    "qwen": "Hello! Write one short sentence about the seaside.",
    "ling": "你好，用一句话介绍海滨城市。",
}
QMODEL = os.environ.get("EDGE0_35B_MODEL", "/path/to/qwen35/model")
LMODEL = os.environ.get("EDGE0_10B_MODEL", "/path/to/ling/model")
QDIR = os.path.dirname(QMODEL)
LDIR = os.path.dirname(LMODEL)
RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "alignment_results.json")


def save(tag, payload):
    data = {}
    if os.path.exists(RESULT):
        with open(RESULT) as f:
            data = json.load(f)
    data[tag] = payload
    with open(RESULT, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("saved", tag)


def qwen_demo():
    os.environ.update(
        MODEL_DIR=QMODEL,
        LORA_ADAPTERS=QDIR + "/lora_qwen35.npz",
        LORA_R="16", LORA_ALPHA="32",
        PREGATE_NPZ=QDIR + "/pregate_qwen35.npz",
        QWEN_PG_START="7", QWEN_PG_HIDDEN="512",
        QWEN_STAGED="1", QWEN_STAGED_TOP_K="4", QWEN_STAGED_SLOTS="4",
        QWEN_STAGED_REPLACE="0", QWEN_PREGATE_V7="0",
        QWEN_HISTORY_PREFETCH="0", QWEN_INTRA_STAGE="0", QWEN_ASM_CACHE="0",
    )
    sys.path.insert(0, QDIR)
    import serve  # noqa: E402  (ENGINE built in main(); build it here)
    from mlx_lm.utils import load_tokenizer  # noqa: E402
    tok = load_tokenizer(QMODEL)
    serve.ENGINE = serve.QwenStreamingEngine(
        QMODEL, cache_slots=900, hot_per_layer=0)
    serve.ENGINE.load_tokenizer(tok)
    eng = serve.ENGINE
    tok = eng.tokenizer()
    messages = [{"role": "user", "content": PROMPTS["qwen"]}]
    ids = serve.encode_chat(tok, messages)
    eng.prefill(ids)
    logits = eng.next_logits()
    cur = int(serve.mx.argmax(logits).item())
    out = [cur]
    for _ in range(N - 1):
        logits = eng.step(out[-1])
        serve.mx.eval(logits)
        cur = int(serve.sample(logits, temperature=0.0, top_k=1, top_p=1.0,
                               repetition_penalty=1.0, history=ids + out))
        out.append(cur)
    save("demo_qwen", {"prompt_ids": ids, "out": out, "text": tok.decode(out)})


def edge0_qwen():
    from edge0 import AutoEngine  # noqa: E402
    from edge0.config import GenerationConfig  # noqa: E402
    eng = AutoEngine.from_pretrained(QMODEL)
    tok = eng._tok
    messages = [{"role": "user", "content": PROMPTS["qwen"]}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = list(tok(text)["input_ids"])
    out = eng.generate(ids, GenerationConfig(temperature=0.0, top_k=1,
                                             top_p=1.0,
                                             repetition_penalty=1.0),
                       max_new_tokens=N)
    save("edge0_qwen", {"prompt_ids": ids, "out": out, "text": tok.decode(out)})
    eng.close()


def ling_demo():
    os.environ.update(
        CHECKPOINT_DIR=LMODEL,
        PREGATE_HYBRID="1", PREGATE_INTRA="0", PREGATE_START_LAYER="1",
        PREGATE_FEATURE_TOPK="executed", PREGATE_DTYPE="fp16",
        PREGATE_NPZ=LDIR + "/pregate_r3traces.npz",
        LORA_ADAPTERS=LDIR + "/lora_r3traces.npz",
        THINK_MODE="0", REPETITION_PENALTY="1.0",
        STAGED_DECODE="0", STAGED_SYNC="0",
        STAGED_SLOTS="8", STAGED_LAYERS="1-23", EXPERT_CACHE_SLOTS="64",
        PREFETCH_ENABLE="0", PREFILL_FULL_LAYER="1", PREFILL_PIN="1",
        LING_PREWARM="1", MLX_COMPILE_MOE="1", LING_PREFILL_CHUNK="2048",
    )
    sys.path.insert(0, LDIR)
    import server  # noqa: E402  (engine built in main(); build it here)
    server.engine = server.LingStreamingEngine(
        LMODEL, think=False, prefill_chunk=512)
    eng = server.engine
    msgs = [{"role": "user", "content": PROMPTS["ling"]}]
    ids = eng.encode_chat(msgs, think=False)
    eng.prefill_prompt(ids, chunk_size=512)
    out = []
    cur = int(server.mx.argmax(eng.next_logits(), axis=-1).item())
    out.append(cur)
    for _ in range(N - 1):
        logits = eng.step(out[-1])
        server.mx.eval(logits)
        cur = int(server.mx.argmax(logits, axis=-1).item())
        out.append(cur)
        if cur == 156895:
            break
    save("demo_ling", {"prompt_ids": ids, "out": out,
                       "text": eng.decode(out)})


def edge0_ling():
    from edge0 import AutoEngine  # noqa: E402
    from edge0.config import GenerationConfig  # noqa: E402
    eng = AutoEngine.from_pretrained(LMODEL)
    tok = eng._tok
    messages = [{"role": "user", "content": PROMPTS["ling"]}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    ids = list(tok(text)["input_ids"])
    eng.prefill(ids, chunk_size=512)
    out = []
    cur = int(eng.next_logits().argmax().item())
    out.append(cur)
    for _ in range(N - 1):
        logits = eng.step(out[-1])
        cur = int(logits.argmax().item())
        out.append(cur)
        if cur == 156895:
            break
    save("edge0_ling", {"prompt_ids": ids, "out": out,
                        "text": tok.decode(out)})
    eng.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("side", choices=["demo", "edge0"])
    ap.add_argument("model", choices=["qwen", "ling"])
    args = ap.parse_args()
    fn = {"demo": {"qwen": qwen_demo, "ling": ling_demo},
          "edge0": {"qwen": edge0_qwen, "ling": edge0_ling}}[args.side][args.model]
    fn()


if __name__ == "__main__":
    main()
