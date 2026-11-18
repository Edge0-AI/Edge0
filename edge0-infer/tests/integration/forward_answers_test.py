"""Two-pass same-backend comparison with the repository answers; stop at the first failed case."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ANSWERS = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers'
sys.path.insert(0, str(ANSWERS.parents[1] / 'answers'))
from judge import judge, judge_suite, own_logits
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu', 'metal'])
    parser.add_argument('--split-last', action='store_true')
    parser.add_argument("--cross-kernel", action="store_true", help="Use explicit cross-kernel answer tolerances")
    parser.add_argument('--record-only', action='store_true', help='Record numerical differences without enforcing thresholds')
    args = parser.parse_args()
    root = Path(os.environ.get('MOE_TINY', '/nonexistent'))
    model = root / 'tiny-a'
    if not (model / 'assets.tsv').exists():
        print('SKIP: missing MOE_TINY tiny-a')
        return 77
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        for role in ['base', 'lora']:
            reports = []
            for case in ['plain-mixed', 'chat-intro', 'long-mixed']:
                suffix = '-split' if args.split_last else ''
                answer = ANSWERS / f'tiny-{role}-{args.backend}-flash{suffix}' / case
                output = workspace / role / case
                observed = workspace / role / (case + '-observed')
                for path, observe in [(output, False), (observed, True)]:
                    command = [args.cli, 'forward', '--model-option', 'routing=actual', '--model', str(model), '--device', args.backend,
                               '--tokens', str(answer / 'tokens.json'), '--out', str(path)]
                    if args.split_last:
                        command.append('--split-last')
                    if role == 'base':
                        command.extend(['--model-option', 'lora=off'])
                    if observe:
                        command.append('--observe')
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode == 77:
                        print(result.stderr)
                        return 77
                    assert result.returncode == 0, result.stdout + result.stderr
                own_logits(observed, json.loads((observed / 'tokens.json').read_text()))
                report = judge(output, observed, answer, args.backend, model, cross_kernel=args.cross_kernel)
                print(json.dumps(dict(role=role, case=case, record_only=args.record_only, **report), indent=2), flush=True)
                if args.split_last and role == 'base' and case == 'chat-intro':
                    own_file = output / 'tokens.json'
                    original = own_file.read_text()
                    own_file.write_text(json.dumps({**json.loads(original), 'chunking': 'whole'}))
                    try:
                        judge(output, observed, answer, args.backend, cross_kernel=args.cross_kernel)
                        raise AssertionError('whole output accepted against split answer')
                    except ValueError as error:
                        assert 'chunking' in str(error)
                    finally:
                        own_file.write_text(original)
                    whole = ANSWERS / f'tiny-{role}-{args.backend}-flash' / case
                    try:
                        judge(output, observed, whole, args.backend, cross_kernel=args.cross_kernel)
                        raise AssertionError('split output accepted against whole answer')
                    except ValueError as error:
                        assert 'chunking' in str(error)
                    print('both chunking mismatch cases rejected', flush=True)
                reports.append(report)
                if not args.record_only and report['status'] != 'passed':
                    print('STOP: numerical comparison failed; thresholds and answers unchanged')
                    return 1
                suite = judge_suite(reports)
                if not args.record_only and suite['status'] != 'passed':
                    print(json.dumps(suite))
                    return 1
                if role == 'lora':
                    base = workspace / 'base' / case
                    base_tokens = json.loads((base / 'tokens.json').read_text())
                    lora_tokens = json.loads((output / 'tokens.json').read_text())
                    assert base_tokens['input'] == lora_tokens['input']
                    count = len(base_tokens['input']) * base_tokens['logits_shape'][1]
                    a = np.fromfile(base / 'logits.f32', dtype='<f4', count=count)
                    b = np.fromfile(output / 'logits.f32', dtype='<f4', count=count)
                    maximum = float(np.abs(a-b).max())
                    if not args.record_only:
                        assert maximum > 1e-3, maximum
                    print(f'{case} T5 prefill_max_abs={maximum}')
            if args.record_only:
                print(json.dumps(dict(role=role, status='recorded', tie_cases=suite['tie_cases'],
                                      comparison_status=suite['status'], reason=suite['reason'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
