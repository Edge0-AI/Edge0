"""Recompute published measurements without contacting a server or editing files."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf-8-sig'))


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'n': len(values)}


def summarize():
    result = {'prefill': {}, 'coding': {}}
    for prefill in (2048, 4096, 8192, 16384):
        rows = load(f'prefill-{prefill}.json')['results']
        assert len(rows) == 6 and all(r['timings']['cache_n'] == 0 for r in rows)
        result['prefill'][prefill] = {
            label: {
                'wall_seconds': stats([r['wall_seconds'] for r in rows if r['case'] == label]),
                'prompt_tokens': sorted(set(r['usage']['prompt_tokens'] for r in rows if r['case'] == label)),
                'prompt_tps': stats([r['timings']['prompt_per_second'] for r in rows if r['case'] == label]),
                'generation_tps': stats([r['timings']['predicted_per_second'] for r in rows if r['case'] == label]),
                'completion_tokens': stats([r['usage']['completion_tokens'] for r in rows if r['case'] == label]),
                'semantic_passes': sum(r['semantic_pass'] for r in rows if r['case'] == label),
                'strict_json_passes': sum(r['strict_pass'] for r in rows if r['case'] == label),
            } for label in ('recall_8k', 'recall_28k')
        }
    coding = load('coding-per-run.json')['results']
    assert len(coding) == 18
    for model in ('iq2', 'iq3'):
        rows = [r for r in coding if r['model'] == model]
        assert len(rows) == 9 and all(r['success'] and all(r['protected'].values()) for r in rows)
        result['coding'][model] = {
            'wall_seconds': stats([r['wall_s'] for r in rows]),
            'cases': {case: stats([r['wall_s'] for r in rows if r['case'] == case])
                      for case in sorted(set(r['case'] for r in rows))},
            'passed_checks_across_repeats': sum(r['passed'] for r in rows),
            'output_tokens_including_reasoning_and_helpers': sum(r['engine_delta']['output_tokens'] for r in rows),
            'prompt_tokens_including_reused': sum(r['engine_delta']['prompt_tokens'] for r in rows),
            'reused_tokens': sum(r['engine_delta']['reused'] for r in rows),
            'decode_ms': sum(r['engine_delta']['decode_ms'] for r in rows),
            'weighted_generation_tps': sum(r['engine_delta']['output_tokens'] for r in rows) /
                                       (sum(r['engine_delta']['decode_ms'] for r in rows) / 1000),
            'drafts_offered': sum(r['engine_delta']['drafts_offered'] for r in rows),
            'drafts_accepted': sum(r['engine_delta']['drafts_accepted'] for r in rows),
        }
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
