"""tools/make_profile_mix.py - an expert-cache profile from several routing traces, each trace counting the same.

`make_profile.py` adds the traces' raw counts, so the longest trace decides the ranking.  A cache profile should
rank the experts by how often a USER's tokens need them, across the kinds of text that user writes, so here every
trace is first turned into a distribution (its counts / its total) and multiplied by its weight; the ranking is the
weighted sum.  Weights are per trace, so a kind of text with more traces counts more - give a group of N traces the
weight 1/N each to make the group count once.

    python tools/make_profile_mix.py TRACE[:WEIGHT] [TRACE[:WEIGHT] ...] [--out PATH] [--n-expert 512]

A trace is a `--dump-routing` file, or a `STRATA_USAGE_DUMP` file of the SYCL engine (same record format).  The output
is a profile for `--expert-profile`; the pairs no trace uses are appended in make_profile.py's fill order.

Measured on an Arc Pro B70 (IQ3_S, 11.5k cached experts, decode of 250-300 tokens): the shipped profile holds 60% of the
experts 45 chat/code/language prompts route and 85% of those a document summary routes; a profile from the 45 chat prompts
and the 8 summaries (weight 1 : 2 per prompt), tested on prompts left out of it, holds 71% and 84%.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_profile as mp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="TRACE or TRACE:WEIGHT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-expert", type=int, default=mp.N_EXPERT)
    a = ap.parse_args()
    ne = a.n_expert
    score = defaultdict(float)
    for spec in a.traces:
        path, weight = spec, 1.0
        head, sep, tail = spec.rpartition(":")
        if sep:   # TRACE:WEIGHT (a Windows drive letter's colon is not followed by a number)
            try:
                path, weight = head, float(tail)
            except ValueError:
                pass
        freq = mp.read_trace(path, ne)
        total = float(sum(freq.values()))
        if total <= 0:
            print(f"{path}: empty trace, skipped")
            continue
        for p, c in freq.items():
            score[p] += weight * c / total
    ranked = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    for e in range(ne):          # the fill, as make_profile.py: the pairs no trace used
        for layer in range(mp.N_LAYER):
            if (layer, e) not in seen:
                ranked.append((layer, e))
    mp.write_profile(a.out, ranked, ne)
    assert mp.read_profile(a.out, ne) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs, {len(score)} from {len(a.traces)} traces")


if __name__ == "__main__":
    main()
