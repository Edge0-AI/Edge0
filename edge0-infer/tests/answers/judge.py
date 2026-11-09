"""Same-backend judgement of engine output against the repository's own answers.

Tolerances and tie rules: logits within 1e-3 (CPU) or 1e-2 (Metal) of every
saved answer value; routing and token ties within 1e-4 (CPU) or 1e-3 (Metal);
By default, at most one tie case per suite. Explicit cross-kernel comparisons use Metal limits.
"""
import json
from pathlib import Path
import re

import numpy as np

from answer_format import check_assets, read_json, read_logits, read_routes, suite_identity

TOPK = re.compile(r'^ffn_moe_topk-(\d+)$')
PROBS = re.compile(r'^ffn_moe_probs-(\d+)$')


def tolerances(backend: str, cross_kernel: bool = False) -> tuple[float, float]:
    """(tie tolerance, logits limit) for a backend."""
    if backend == 'cpu' and not cross_kernel:
        return 1e-4, 1e-3
    if backend == 'metal' or (backend == 'cpu' and cross_kernel):
        return 1e-3, 1e-2
    raise ValueError('invalid backend')


def load_answer(answer: Path, backend: str, assets: Path | None = None):
    answer = Path(answer)
    identity = suite_identity(answer.parent)
    if identity['backend'] != backend:
        raise ValueError('answer backend differs from requested backend')
    if answer.name not in identity['cases']:
        raise ValueError(f'case not listed in answer identity: {answer.name}')
    if assets is not None:
        check_assets(identity, Path(assets))
    tokens = read_json(answer / 'tokens.json')
    rows = len(tokens['input']) + len(tokens['generated'])
    if tokens['logits_shape'] != [rows, identity['vocabulary']]:
        raise ValueError('answer logits shape differs from its tokens')
    return identity, tokens, read_logits(answer / 'logits.bin', identity, rows)


def own_logits(root: Path, tokens: dict) -> np.ndarray:
    shape = tokens['logits_shape']
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError('invalid logits shape')
    data = np.fromfile(root / 'logits.f32', dtype='<f4')
    if data.size != np.prod(shape) or not np.isfinite(data).all():
        raise ValueError('invalid logits data')
    return data.reshape(shape)


def observed_routes(root: Path) -> dict:
    """Return {(step, layer): (ids[positions, k], probs[positions, experts])} from an observed run."""
    index = read_json(root / 'layers/index.json')
    ids, probs = {}, {}
    for record in index:
        match_ids, match_probs = TOPK.match(record['name']), PROBS.match(record['name'])
        if not match_ids and not match_probs:
            continue
        file = (root / 'layers' / record['file']).resolve()
        if not file.is_relative_to((root / 'layers').resolve()):
            raise ValueError('layer file escapes directory')
        shape = tuple(reversed(record['shape']))
        if match_ids:
            if record['type'] != 26:
                raise ValueError('expert IDs must be I32')
            key = (record['step'], int(match_ids.group(1)))
            if key not in ids:
                ids[key] = np.fromfile(file, dtype='<i4').reshape(shape).reshape(-1, shape[-1]).astype(np.int64)
        else:
            if record['type'] != 0:
                raise ValueError('routing probabilities must be F32')
            key = (record['step'], int(match_probs.group(1)))
            if key not in probs:
                probs[key] = np.fromfile(file, dtype='<f4').reshape(shape).reshape(-1, shape[-1])
    if ids.keys() != probs.keys():
        raise ValueError('observed routing IDs and probabilities differ')
    return {key: (ids[key], probs[key]) for key in ids}


def compare_routes(expected: dict, observed: dict, tolerance: float):
    routes, tie, failed = [], False, False
    for key, answer_ids in expected.items():
        if key not in observed:
            raise ValueError(f'missing observed routing: step {key[0]} layer {key[1]}')
        own_ids, probs = observed[key]
        if own_ids.shape != answer_ids.shape or probs.shape[0] != own_ids.shape[0]:
            raise ValueError(f'route shape differs: step {key[0]} layer {key[1]}')
        if np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=-1), 1, atol=1e-5, rtol=1e-5):
            raise ValueError('invalid routing probabilities')
        if np.any(own_ids < 0) or np.any(own_ids >= probs.shape[-1]) or np.any(answer_ids >= probs.shape[-1]):
            raise ValueError('expert ID outside probability array')
        unequal = answer_ids != own_ids
        differences = np.abs(np.take_along_axis(probs, answer_ids, axis=-1) - np.take_along_axis(probs, own_ids, axis=-1))
        gap = float(differences[unequal].max(initial=0))
        changed = bool(unequal.any())
        tied = changed and gap <= tolerance
        failed |= changed and not tied
        tie |= tied
        worst = None
        if changed:
            position, rank = np.unravel_index(np.argmax(np.where(unequal, differences, -1)), differences.shape)
            a, o = int(answer_ids[position, rank]), int(own_ids[position, rank])
            worst = dict(position=int(position), rank=int(rank), answer_expert=a, ours_expert=o,
                         answer_expert_ours_probability=float(probs[position, a]),
                         ours_expert_probability=float(probs[position, o]), gap=gap)
        routes.append(dict(step=key[0], layer=key[1], mismatches=int(unequal.sum()), max_probability_gap=gap,
                           tie=tied, passed=not (changed and not tied), worst_difference=worst))
    return routes, tie, failed


def compare_logits(own: np.ndarray, rows: np.ndarray, identity: dict, tolerance: float, limit: float) -> dict:
    probes = np.asarray(identity['probes'])
    top = np.take_along_axis(own, rows['tokens'].astype(np.int64), axis=1).astype(np.float64)
    top_delta = np.abs(top - rows['values'].astype(np.float64))
    probe_delta = np.abs(own[:, probes].astype(np.float64) - rows['probes'].astype(np.float64))
    worst = max(float(top_delta.max(initial=0)), float(probe_delta.max(initial=0)))
    different = own.argmax(axis=1) != rows['tokens'][:, 0]
    ambiguous = np.abs(rows['values'][:, 0].astype(np.float64) - rows['values'][:, 1].astype(np.float64)) <= tolerance
    passed = bool(worst <= limit and np.all(~different | ambiguous))
    return dict(max_abs=worst, top_max_abs=float(top_delta.max(initial=0)), probe_max_abs=float(probe_delta.max(initial=0)),
                argmax_differences=int(different.sum()), non_tied_argmax_differences=int((different & ~ambiguous).sum()),
                passed=passed)


def judge(ours: Path, observed: Path, answer: Path, backend: str, assets: Path | None = None, *, cross_kernel: bool = False) -> dict:
    """Forward judgement: `ours` is a plain forward run, `observed` the same run with --observe."""
    tolerance, limit = tolerances(backend, cross_kernel)
    identity, tokens, rows = load_answer(answer, backend, assets)
    for root in (ours, observed):
        actual = read_json(Path(root) / 'tokens.json')
        if actual.get('chunking', 'whole') != identity['chunking']:
            raise ValueError('chunking differs from answer')
        for key in ('input', 'generated', 'logits_shape'):
            if actual[key] != tokens[key]:
                raise ValueError(f'{key} differs from answer')
    steps = [tuple(item) for item in tokens['route_steps']]
    expected = read_routes(Path(answer) / 'routes.bin', identity, steps)
    routes, tie, route_failed = compare_routes(expected, observed_routes(Path(observed)), tolerance)
    logits = compare_logits(own_logits(Path(ours), tokens), rows, identity, tolerance, limit)
    passed = not route_failed and (tie or logits['passed'])
    first = next((item for item in routes if not item['passed']), None)
    if first is None and not tie and not logits['passed']:
        first = dict(name='logits', max_abs=logits['max_abs'])
    return dict(status='passed' if passed else 'failed', tie_case=tie, routes=routes, logits=logits,
                numerical_comparisons_enforced=not tie, first_failure=first)


def judge_suite(results: list[dict]) -> dict:
    count = sum(bool(item['tie_case']) for item in results)
    passed = bool(results) and all(item['status'] == 'passed' for item in results) and count <= 1
    return dict(status='passed' if passed else 'failed', tie_cases=count,
                reason='stop: more than one case has ties' if count > 1 else '', cases=results)


def judge_generation(ours: Path, answer: Path, backend: str, assets: Path | None = None, *, cross_kernel: bool = False,
                     record_only: bool = False) -> dict:
    """Compare an autoregressive continuation only up to its first divergence."""
    tolerance, _ = tolerances(backend, cross_kernel)
    identity, expected, rows = load_answer(answer, backend, assets)
    if identity['chunking'] != 'split-last':
        raise ValueError('answer chunking must be split-last')
    actual = read_json(Path(ours) / 'tokens.json')
    if actual.get('chunking', 'whole') != 'split-last':
        raise ValueError('chunking differs from answer')
    if not expected['input'] or actual['input'] != expected['input']:
        raise ValueError('input differs from answer')
    if not record_only and actual.get('stop_reason') != expected['stop_reason']:
        raise ValueError('stop_reason differs from answer')
    if not record_only and len(actual['generated']) != len(expected['generated']):
        raise ValueError('generated token count differs from answer')
    for tokens in (expected['generated'], actual['generated']):
        if any(type(t) is not int or t < 0 or t >= identity['vocabulary'] for t in tokens):
            raise ValueError('generated token outside vocabulary')
    for i, (a, b) in enumerate(zip(actual['generated'], expected['generated'])):
        if a == b:
            continue
        row = rows[len(expected['input']) - 1 + i]
        listed = {int(t): float(v) for t, v in zip(row['tokens'], row['values'])}
        if b not in listed:
            raise ValueError(f'answer token {b} is not in its own top list')
        if a not in listed:
            return dict(status='failed', tie_cases=0, matched=i, position=i, ours_token=a, answer_token=b,
                        reason=f'token outside answer top {identity["top"]}')
        gap = abs(listed[a] - listed[b])
        tied = gap <= tolerance
        return dict(status='passed' if tied else 'failed', tie_cases=int(tied), matched=i, position=i,
                    ours_token=a, answer_token=b, ours_logit=listed[a], answer_logit=listed[b], gap=gap)
    if actual.get('stop_reason') != expected['stop_reason'] or len(actual['generated']) != len(expected['generated']):
        return dict(status='failed', tie_cases=0, matched=min(len(actual['generated']), len(expected['generated'])),
                    ours_count=len(actual['generated']), answer_count=len(expected['generated']),
                    ours_stop_reason=actual.get('stop_reason'), answer_stop_reason=expected['stop_reason'],
                    reason='generated length or stop reason differs')
    return dict(status='passed', tie_cases=0, matched=len(actual['generated']))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['forward', 'generation'])
    parser.add_argument('--answer', type=Path, required=True)
    parser.add_argument('--backend', required=True, choices=['cpu', 'metal'])
    parser.add_argument('--ours', type=Path, required=True)
    parser.add_argument('--observed', type=Path)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    if args.kind == 'forward':
        report = judge(args.ours, args.observed, args.answer, args.backend, args.assets)
    else:
        report = judge_generation(args.ours, args.answer, args.backend, args.assets)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
