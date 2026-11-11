# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
# Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
"""tools/ds4/engram_hash.py - DeepSeek-V4.1's Engram n-gram hashes (after DeepSeek's inference/engram.py, MIT).

A position is hashed as the n-grams (2 .. max_ngram_size) that end there, each over n_heads heads; every (n-gram
size, head) pair owns a prime-sized bucket range of the layer's table.  Token ids first go through a compressed map
where tokens that normalize alike (NFKC, NFD, accents stripped, lowercased, whitespace runs collapsed) share an id.

    lay = Layout.from_config(text_config)
    tm, n = compressed_token_map(tokenizer_json_path)     # needs the `tokenizers` package
    h = Hasher(lay, tm, multipliers(lay, n), pad=tm[text_config["engram_pad_token_id"]])
    h.ids(tokens)  -> int64 [T, n_layers, (max_ngram_size - 1) * n_heads], absolute table rows
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet (engram.py's find_next_prime)."""
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class Layout:
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram: int
    n_heads: int
    head_dim: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]    # [layer][n-gram size - 2][head]

    @classmethod
    def from_config(cls, t: dict) -> "Layout":
        layer_ids = tuple(t["engram_layer_ids"])
        mx, nh = t["engram_max_ngram_size"], t["engram_n_heads"]
        primes, seen = [], set()
        for _ in layer_ids:
            per = []
            for _ in range(mx - 1):
                sizes, cur = [], t["engram_vocab_size"] - 1
                for _ in range(nh):
                    cur = _next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per.append(tuple(sizes))
            primes.append(tuple(per))
        lay = cls(layer_ids, tuple(t["engram_num_embeddings"]), mx, nh, t["engram_head_dim"], tuple(primes))
        for i, n in enumerate(lay.num_embeddings):
            got = sum(p for per in lay.primes[i] for p in per)
            if got != n:
                raise ValueError("engram layer %d: the primes sum to %d, the config says %d rows" % (i, got, n))
        return lay

    def flat_primes(self) -> np.ndarray:
        """[n_layers, (max_ngram - 1) * n_heads] int64, in hash-column order."""
        return np.array([[p for per in layer for p in per] for layer in self.primes], dtype=np.int64)

    def offsets(self) -> np.ndarray:
        f = self.flat_primes()
        return np.concatenate([np.zeros((f.shape[0], 1), np.int64), np.cumsum(f, axis=1)[:, :-1]], axis=1)


def multipliers(lay: Layout, compressed_vocab: int) -> np.ndarray:
    """[n_layers, max_ngram] int64, odd, bounded so token * multiplier stays inside int64 (engram.py)."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    rows = []
    for lid in lay.layer_ids:
        g = np.random.default_rng(10007 * lid)
        rows.append(g.integers(low=0, high=bound, size=(lay.max_ngram,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def compressed_token_map(tokenizer_json: str) -> tuple[np.ndarray, int]:
    """engram.py's build_compressed_token_map over the raw Rust tokenizer (no clean-up of spaces)."""
    from tokenizers import Regex, Tokenizer, normalizers

    tok = Tokenizer.from_file(str(tokenizer_json))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(n, dtype=np.int64)
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


class Hasher:
    def __init__(self, lay: Layout, token_map: np.ndarray, mult: np.ndarray, pad: int):
        self.lay, self.tm, self.mult, self.pad = lay, token_map, mult, int(pad)
        self.primes = np.array(lay.primes, dtype=np.int64)            # [L, max_ngram - 1, heads]
        self.off = lay.offsets()                                      # [L, cols]

    def ids(self, tokens) -> np.ndarray:
        """Absolute table rows [T, n_layers, cols] for a sequence that starts at position 0 (no dead tokens)."""
        c = self.tm[np.asarray(tokens, dtype=np.int64)]
        T = len(c)
        pos = np.arange(T)
        toks = []
        blocked = np.zeros(T, dtype=bool)
        for s in range(self.lay.max_ngram):
            src = c[np.clip(pos - s, 0, None)]
            blocked = blocked | (pos < s)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)                                # [T, max_ngram]
        prod = toks[:, None, :] * self.mult[None]                     # [T, L, max_ngram]
        rolling, out = prod[..., 0], []
        for i in range(1, self.lay.max_ngram):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, axis=-1) + self.off[None]
