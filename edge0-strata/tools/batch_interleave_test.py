#!/usr/bin/env python3
"""Batch slots, the parts batch_test.py does not reach (#465): a long prompt read WHILE other slots decode (its
chunks interleaved with their windows), and a conversation's next turn continued from the slot that holds it.

  1. solo references (GEN): A, then A's next turn (from the live session), B, C (a long prompt)
  2. the same in the slots: A in slot 0, B in slot 1 next to it, then C in slot 2 while A and B decode - every slot's
     tokens must equal its solo tokens
  3. A's next turn (BGEN into slot 0, which still holds A): read from the slot, its tokens must equal the solo next turn
  4. a long prompt D gives way (BYIELD) at its first chunk boundary, a short E is admitted, D goes on from its slot
     while E decodes: both equal their solo tokens.  On a pipelined batch (a layer split, more than one group) a read
     never gives way: D is then read whole into slot 2 and E admitted into slot 1 beside it; both must still equal
     solo, only the give-way is not required (--yield auto; --yield expect requires it anyway)
  6. A in a slot, stopped after 50 tokens and continued on the solo path (GEN of A + its tokens, from the slot): the
     whole equals solo A (what the server does with a request left alone in a slot)
  5. (a measurement) a solo next turn continued from a slot: the drafts accepted
  7. F, A's history as a client sends it back without the reply's thinking, from slot 1's turn checkpoint: equal to
     F solo (which continues from the same checkpoint of the live chain)
     (after step 6 the solo path holds A's start, so F's start comes from the live chain or the prompt cache here)
  7b. A in slot 1 to its end, --flush unrelated solo GENs, then F2 (A's question, another reply) into slot 2: its
     start can only come from slot 1's turn checkpoint; equal to F2 solo.  The engine log line says which source was used.

Exact comparisons need the same settings as batch_test.py (this script sets STRATA_IQ_MT_MIN=1):
  python tools/batch_interleave_test.py --exe build/strata --config strata-<model>.json \\
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
Without --no-prefill-borrow the slots decoding during C's read see the expert cache without the slots C's prompt
borrowed (those experts run on the CPU, which rounds differently): A's and B's text may then drift from solo.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

LONG = ("The history of mathematics is long and full of surprising turns. Early civilizations counted with tally marks, "
        "then with symbols, and later with place-value systems that made arithmetic far easier. ")


def run(eng, out, line, slot=None):
    """Send one GEN/BGEN; returns (tokens of the request line, the BADM flag) - BT/BDONE of other slots are kept."""
    eng.send(line)
    got = []
    for l in out:
        if l.startswith("T "):
            got.append(int(l.split()[1]))
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif slot is None and l.startswith("DONE"):
            return got, None
        elif slot is not None and l.startswith("BADM "):
            return got, l.split()[2] == "1"
    raise SystemExit(f"the engine ended during: {line[:40]} (see {eng.log_path})")


def engine_pipelined(eng, extra):
    """Does the engine run the batch in pipelined groups (a layer split, more than one group)?  Then a BYIELD is
    dropped: a pipelined read never gives way."""
    if "--batch-groups" in extra:
        v = extra[extra.index("--batch-groups") + 1] if extra.index("--batch-groups") + 1 < len(extra) else ""
        if v.isdigit():
            return int(v) > 1
    eng.log.flush()
    m = re.search(r"--batch-groups auto: (\d+) groups? of", Path(eng.log_path).read_text(errors="replace"))
    return bool(m) and int(m.group(1)) > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--long", type=int, default=5000, help="tokens of the long prompt C (several prompt chunks)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--yield", dest="yield_mode", choices=("auto", "expect", "skip"), default="auto",
                    help="step 4's give-way: auto = expect a YIELDED unless the batch is pipelined (the engine log "
                         "says '--batch-groups auto: N groups' with N > 1, or --extra has --batch-groups N > 1), "
                         "where a read never gives way; expect = always required; skip = never required")
    ap.add_argument("--flush", type=int, default=8,
                    help="7b: unrelated solo GENs between A in slot 1 and F, so only the slot holds A's start (0: off)")
    ap.add_argument("--dump", default="", help="write every solo and slot token list as JSON: an A/B of two builds")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    body = LONG * (a.long // max(1, len(tok.encode(LONG))) + 1)
    A = chat("Write a Python function that merges overlapping intervals, then explain it.")
    B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    C = chat(body + "\nSummarize the text above in three sentences.")
    D = chat(body + "\nWhat are the three most important ideas in the text above?")
    E = chat("Describe the life cycle of a star like the Sun.")
    print(f"prompts: A {len(A)}, B {len(B)}, C {len(C)} tokens", flush=True)
    eng = Engine(a.exe, cfg, 3, {"STRATA_IQ_MT_MIN": "1"}, a.extra.split())
    eng.pending = []
    out = eng.lines()
    ids = lambda v: ",".join(map(str, v))
    M = a.max_new

    # 1. solo
    sA, _ = run(eng, out, f"GEN {M} {ids(A)}")
    A2 = A + sA + chat("Now do the same in Rust.")
    sA2, _ = run(eng, out, f"GEN {M} {ids(A2)}")
    # F: A's history as a client sends it back WITHOUT the reply's thinking - it shares A only up to A's last turn
    # boundary (the checkpoint there); solo it continues from that checkpoint of the live chain
    turn_at = max(i for i, t in enumerate(A) if t == A[0])          # A[0] is <|im_start|>
    F = A[:turn_at] + tok.encode("<|im_start|>assistant\nA short answer.<|im_end|>\n", parse_special=True) + \
        chat("Now in Go.")
    sF, _ = run(eng, out, f"GEN {M} {ids(F)}")
    # F2 (step 7b): another reply to A's question - it too shares A only up to A's turn checkpoint
    F2 = A[:turn_at] + tok.encode("<|im_start|>assistant\nAnother short answer.<|im_end|>\n", parse_special=True) + \
        chat("Now in C.")
    sF2, _ = run(eng, out, f"GEN {M} {ids(F2)}") if a.flush > 0 else ([], None)
    sB, _ = run(eng, out, f"GEN {M} {ids(B)}")
    sC, _ = run(eng, out, f"GEN {M} {ids(C)}")
    sD, _ = run(eng, out, f"GEN {M} {ids(D)}")
    sE, _ = run(eng, out, f"GEN {M} {ids(E)}")
    print(f"solo: A {len(sA)}, A2 {len(sA2)}, B {len(sB)}, C {len(sC)}, D {len(sD)}, E {len(sE)} tokens", flush=True)

    # 2. the slots: A, B, then the long C while they decode
    got = {0: [], 1: [], 2: []}
    done = {}
    t0 = time.time()
    for s, P in ((0, A), (1, B), (2, C)):
        first, cont = run(eng, out, f"BGEN {s} {M} {ids(P)}", slot=s)
        got[s] += first
        if not cont:
            done[s] = True
    t_admit = time.time() - t0
    during_c = sum(1 for l in eng.pending if l.startswith("BT "))   # windows the slots ran while C was admitted

    def drain():
        while eng.pending or len(done) < 3:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT "):
                _, s, y = l.split()
                got[int(s)].append(int(y))
            elif l.startswith("BDONE "):
                done[int(l.split()[1])] = True
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
    drain()
    print(f"slots: admissions {t_admit:.1f}s; {during_c} slot tokens arrived while the admissions read their prompts",
          flush=True)
    ok = True
    for s, ref, name in ((0, sA, "A"), (1, sB, "B"), (2, sC, "C")):
        same = got[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(ref))) if got[s][k] != ref[k]), None)
        print(f"slot {s} ({name}): {len(got[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 3. A's next turn from slot 0 (the engine's log says "slot 0 gave back ...")
    first, cont = run(eng, out, f"BGEN 0 {M} {ids(A2)}", slot=0)
    got2, done = first, {}
    if cont:
        while True:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 0 "):
                got2.append(int(l.split()[2]))
            elif l.startswith("BDONE 0 "):
                break
    same = got2 == sA2
    ok &= same
    d = next((k for k in range(min(len(got2), len(sA2))) if got2[k] != sA2[k]), None)
    print(f"A's next turn from its slot: {len(got2)} tokens, solo {len(sA2)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 4. a long prompt D gives way (BYIELD) to a short E at its first chunk boundary, then goes on from its slot.
    # On a pipelined batch (a layer split with more than one group) a read never gives way (generate.cpp: the BYIELD
    # is dropped): D is then admitted whole into slot 2 and E goes into the free slot 1 while D decodes - both must
    # still equal solo, only the give-way itself is not checked (--yield)
    pipelined = engine_pipelined(eng, a.extra.split())
    expect_yield = a.yield_mode == "expect" or (a.yield_mode == "auto" and not pipelined)
    eng.send(f"BGEN 2 {M} {ids(D)}")
    eng.send("BYIELD 2")
    yielded, firstD, contD = None, [], None
    for l in out:
        if l.startswith("T "):
            firstD.append(int(l.split()[1]))
        elif l.startswith("YIELDED "):
            yielded = l
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif l.startswith("BADM 2 "):
            contD = l.split()[2] == "1"
            break
    if yielded is not None:
        print(f"D gave way: {yielded}", flush=True)
    elif expect_yield:
        print(f"D gave way: None (expected a YIELDED: batch {'pipelined' if pipelined else 'not pipelined'})",
              flush=True)
        ok = False
    else:
        print(f"D gave way: None - the batch is pipelined ({pipelined}), where a read never gives way; D stays in "
              f"slot 2 ({len(firstD)} first tokens, {'continues' if contD else 'done'}), E goes into slot 1 beside it",
              flush=True)
    got4 = {1: [], 2: []}
    first, cont1 = run(eng, out, f"BGEN 1 {M} {ids(E)}", slot=1)
    got4[1] += first
    if yielded is not None:
        first, cont2 = run(eng, out, f"BGEN 2 {M} {ids(D)}", slot=2)
        got4[2] += first
    else:   # D never left slot 2: its tokens so far are the read's first ones plus the BT 2 lines kept in pending
        got4[2] += firstD
        cont2 = contD
    done4 = {s for s, c in ((1, cont1), (2, cont2)) if not c}
    while len(done4) < 2:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT "):
            _, s, y = l.split()
            if int(s) in got4 and int(s) not in done4:
                got4[int(s)].append(int(y))
        elif l.startswith("BDONE "):
            done4.add(int(l.split()[1]))
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
    for s, ref, name in ((1, sE, "E (short, admitted while D waited)"), (2, sD, "D (gave way, then went on)" if yielded else "D (read whole, E admitted beside it)")):
        same = got4[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got4[s]), len(ref))) if got4[s][k] != ref[k]), None)
        print(f"slot {s} {name}: {len(got4[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 6. back to the solo path (what the server does with a request left alone in a slot): A in slot 1, BSTOP after
    # 50 tokens, then GEN of A + what it produced - the engine continues from the slot with MTP drafts again
    first, cont = run(eng, out, f"BGEN 1 {M} {ids(A)}", slot=1)
    got6, sent = list(first), False
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 1 "):
            got6.append(int(l.split()[2]))
            if len(got6) >= 50 and not sent:
                eng.send("BSTOP 1")
                sent = True
        elif l.startswith("BDONE 1 "):
            break
    tail6, _ = run(eng, out, f"GEN {M - len(got6)} {ids(A + got6)}") if len(got6) < M else ([], None)
    same = got6 + tail6 == sA
    ok &= same
    both = got6 + tail6
    d = next((k for k in range(min(len(both), len(sA))) if both[k] != sA[k]), None)
    print(f"A in a slot, then solo again after {len(got6)} tokens: {len(both)} tokens, solo {len(sA)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 5. (a measurement, not a check) a SOLO next turn continued from slot 0: its tokens are exact either way, but the
    # draft layer's own K/V was built for another conversation, so fewer drafts may be accepted than in a solo next
    # turn whose drafter read the conversation (A2 in step 1)
    A3 = A2 + got2 + chat("Now in Go.")
    eng.send(f"GEN {M} {ids(A3)}")
    for l in out:
        if l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("DONE"):
            f = l.split()
            print(f"solo next turn from a slot: drafts accepted {f[6]} of {f[7]} ({f[1]} tokens in {f[4]} ms)",
                  flush=True)
            break
    # 7. F, A's history without the reply's thinking, into slot 2 (its start comes from the live chain or the prompt
    # cache: step 6's solo continuation took slot 1's contents back; 7b below makes a slot the only source)
    first, cont = run(eng, out, f"BGEN 2 {M} {ids(F)}", slot=2)
    got7 = list(first)
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 2 "):
            got7.append(int(l.split()[2]))
        elif l.startswith("BDONE 2 "):
            break
    same = got7 == sF
    ok &= same
    d = next((k for k in range(min(len(got7), len(sF))) if got7[k] != sF[k]), None)
    print(f"F (A's history without the reply's thinking) in slot 2: {len(got7)} tokens, solo "
          f"{len(sF)}: {'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 7b. step 7 where only a SLOT can give the start: after step 6 the solo path took slot 1's contents back, so the
    # live chain (or the prompt cache) held A's start in step 7 and F came from there.  Here A runs to its end in
    # slot 1, then --flush unrelated solo GENs push A out of the live session and the prompt cache's checkpoints; F2
    # (A's question, another reply) into slot 2 can then only start from slot 1's turn checkpoint (slot 2 holds F,
    # which shares no more with F2 than A does; the engine log line is printed) - its tokens must equal F2 solo
    got7b = []
    if a.flush > 0:
        first, cont = run(eng, out, f"BGEN 1 {M} {ids(A)}", slot=1)
        gotA = list(first)
        while cont:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 1 "):
                gotA.append(int(l.split()[2]))
            elif l.startswith("BDONE 1 "):
                break
        same = gotA == sA
        ok &= same
        print(f"7b: A in slot 1 to its end: {len(gotA)} tokens, solo {len(sA)}: {'IDENTICAL' if same else 'DIFFERS'}",
              flush=True)
        for k in range(a.flush):
            run(eng, out, f"GEN 8 {ids(chat(f'Unrelated question {k}: name one prime number above {100 * k}.'))}")
        eng.log.flush()
        mark = len(Path(eng.log_path).read_text(errors="replace"))
        first, cont = run(eng, out, f"BGEN 2 {M} {ids(F2)}", slot=2)
        got7b = list(first)
        while cont:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 2 "):
                got7b.append(int(l.split()[2]))
            elif l.startswith("BDONE 2 "):
                break
        tail = Path(eng.log_path).read_text(errors="replace")[mark:]
        src = [l for l in tail.splitlines() if "gave back" in l or "strata serve: prompt" in l][:2]
        same = got7b == sF2
        ok &= same
        d = next((k for k in range(min(len(got7b), len(sF2))) if got7b[k] != sF2[k]), None)
        print(f"7b: F2 after {a.flush} unrelated GENs, from slot 1: {len(got7b)} tokens, solo {len(sF2)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
        for l in src:
            print("  7b log:", l.split("strata ")[-1][:160], flush=True)
    if a.dump:
        Path(a.dump).write_text(json.dumps({
            "solo": {"A": sA, "A2": sA2, "B": sB, "C": sC, "D": sD, "E": sE, "F": sF, "F2": sF2},
            "slots": {"A": got[0], "B": got[1], "C": got[2], "A2": got2, "E": got4[1], "D": got4[2],
                      "A_slot_then_solo": got6 + tail6, "F": got7, "F2": got7b},
            "yielded": yielded is not None, "pipelined": pipelined}), encoding="utf-8")
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = Path(eng.log_path).read_text(errors="replace")
    for l in log.splitlines():
        if "drafts accepted" in l:
            print("  log:", l.split("strata serve: ")[-1][:160], flush=True)
    for key in ("gave back", "its turn checkpoint", "takes", "the prompt was read in", "gives way"):
        print(f"engine log '{key}': {log.count(key)} lines", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
