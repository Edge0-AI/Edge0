"""The draft head's token subset (rt/draft_vocab.bin): the vocabulary rows the MTP draft layer may propose.

The engine reads rt/draft_vocab.bin (int32 token ids) when the draft layer binds and computes the draft logits over
those rows only: a smaller head is faster and takes less VRAM, but a token outside the subset can never be drafted.
The shipped subset was built from English and code and had 27 of the vocabulary's 55,328 Han tokens, so an answer in
Chinese drafted almost nothing (#137).  This adds whole scripts to a subset:

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab.bin --add cjk \
        --out data/draft_vocab.bin

--add takes han, kana, hangul, cjk_punct, cjk (those four) or cyrillic.  The base ids keep their order; the added ones follow in
id order.  --stats prints what a subset holds.

A language written in a script the base already holds (French, Spanish, German... in Latin letters) needs specific
words and word pieces, not a whole script.  --corpus adds the tokens of a text corpus in that language: the most
frequent ones that together cover --coverage (default 0.99) of the corpus's token occurrences (#597):

    python tools/draft_vocab.py --gguf <model>-00001-of-0000N.gguf --base data/draft_vocab_en.bin \
        --corpus fr1.txt fr2.txt ... --coverage 0.99 --out data/draft_vocab_fr.bin

It prints the share of the corpus's occurrences outside the base and outside the new subset.  --add and --corpus
can be combined.
"""
from __future__ import annotations

import argparse
import sys
from array import array
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

SCRIPTS = {
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)],
    "kana": [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    "cjk_punct": [(0x3000, 0x303F), (0xFF00, 0xFF65), (0xFFA0, 0xFFEF)],
    # Ukrainian, Russian, Bulgarian, Serbian...: the shipped subset held 142 of the vocabulary's 18,580 Cyrillic tokens
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
}
GROUPS = {"cjk": ["han", "kana", "hangul", "cjk_punct"]}


def scripts_of(text: str) -> set[str]:
    found = set()
    for ch in text:
        c = ord(ch)
        for name, ranges in SCRIPTS.items():
            if any(a <= c <= b for a, b in ranges):
                found.add(name)
    return found


def token_text(tok, i: int) -> str | None:
    """A token's text, or None when its bytes are not whole UTF-8 characters (a piece of one) or it is special."""
    try:
        return tok.token_bytes(i).decode("utf-8")
    except (KeyError, UnicodeDecodeError, IndexError):
        return None


def corpus_counts(tok, paths) -> Counter:
    """Token occurrences over the corpus files (UTF-8 text, tokenized as plain text: no special tokens)."""
    counts: Counter = Counter()
    for path in paths:
        counts.update(tok.encode(Path(path).read_text(encoding="utf-8")))
    return counts


def covering(counts: Counter, coverage: float) -> list[int]:
    """The most frequent ids (ties: the lower id first) that together cover `coverage` of the occurrences."""
    total = sum(counts.values())
    picked, covered = [], 0
    for i, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if covered >= coverage * total:
            break
        picked.append(i)
        covered += c
    return picked


def outside(counts: Counter, ids) -> float:
    """The share of the occurrences whose token is not in `ids`."""
    total = sum(counts.values())
    have = set(ids)
    return sum(c for i, c in counts.items() if i not in have) / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's (first) GGUF file: its vocabulary")
    ap.add_argument("--base", help="the subset to extend (int32 ids)")
    ap.add_argument("--add", default="", help="comma list: han, kana, hangul, cjk_punct, cjk, cyrillic")
    ap.add_argument("--corpus", nargs="+", default=[], help="UTF-8 text files in the language to add (#597)")
    ap.add_argument("--coverage", type=float, default=0.99,
                    help="with --corpus: the share of its token occurrences the added tokens cover (default 0.99)")
    ap.add_argument("--out", help="where to write the new subset")
    ap.add_argument("--stats", action="store_true", help="print what the base (and the new) subset holds")
    a = ap.parse_args()

    tok = ST.Tokenizer.from_gguf(a.gguf)
    n = len(tok.tokens)
    kinds = [scripts_of(t) if (t := token_text(tok, i)) is not None else set() for i in range(n)]
    base = list(array("i", Path(a.base).read_bytes())) if a.base else []
    if any(i < 0 or i >= n for i in base):
        sys.exit(f"{a.base} has ids outside the vocabulary ({n})")

    want = set()
    for x in filter(None, (s.strip() for s in a.add.split(","))):
        names = GROUPS.get(x, [x])
        if any(nm not in SCRIPTS for nm in names):
            sys.exit(f"--add {x}: expected one of {', '.join([*SCRIPTS, *GROUPS])}")
        want.update(names)
    if not 0 < a.coverage <= 1:
        sys.exit("--coverage: expected a share in (0, 1]")
    counts = corpus_counts(tok, a.corpus) if a.corpus else Counter()
    from_corpus = set(covering(counts, a.coverage)) if counts else set()
    have = set(base)
    added = [i for i in range(n) if i not in have and (kinds[i] & want or i in from_corpus)]
    ids = base + added

    def stats(label, sel):
        counts = {nm: sum(1 for i in sel if nm in kinds[i]) for nm in SCRIPTS}
        full = {nm: sum(1 for k in kinds if nm in k) for nm in SCRIPTS}
        print(f"{label}: {len(sel)} ids; " + ", ".join(f"{nm} {counts[nm]}/{full[nm]}" for nm in SCRIPTS))

    if a.stats or not a.out:
        if base:
            stats("base", base)
        if added:
            stats("new", ids)
    if counts:
        print(f"corpus: {sum(counts.values())} token occurrences, {len(counts)} distinct; "
              f"{len(from_corpus)} tokens cover {a.coverage:g}, {len(from_corpus - have)} of them not in the base")
        print(f"corpus occurrences outside the base: {outside(counts, base):.2%}, outside the new subset: "
              f"{outside(counts, ids):.2%}")
    if a.out:
        Path(a.out).write_bytes(array("i", ids).tobytes())
        print(f"wrote {a.out}: {len(ids)} ids ({len(added)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
