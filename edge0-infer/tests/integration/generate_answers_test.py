"""Generation comparison with the repository answers on all tiny cases and both weight modes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ANSWERS = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers'
sys.path.insert(0, str(ANSWERS.parents[1] / 'answers'))
from judge import judge_generation, own_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--backend', default='cpu', choices=['cpu', 'metal'])
    p.add_argument("--cross-kernel", action="store_true", help="Use explicit cross-kernel answer tolerances")
    p.add_argument('--record-only', action='store_true', help='Record answer differences without enforcing them')
    args = p.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (model / 'assets.tsv').exists():
        print('SKIP: missing MOE_TINY tiny-a')
        return 77
    failed = False
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for role in ['base', 'lora']:
            ties = 0
            base = [args.cli, 'run', '--model-option', 'routing=actual', '--model', str(model), '--device', args.backend, '--temperature', '0', '--presence-penalty', '0']
            if role == 'base':
                base += ['--model-option', 'lora=off']
            for name in ['plain-mixed', 'chat-intro', 'long-mixed']:
                answer = ANSWERS / f'tiny-{role}-{args.backend}-flash-split/{name}'
                expected = json.loads((answer / 'tokens.json').read_text())
                output = root / f'{role}-{name}'
                command = base + ['--tokens', str(answer / 'tokens.json'), '--max-tokens', str(len(expected['generated']))]
                a = subprocess.run(command + ['--out', str(output)], capture_output=True, text=True)
                if a.returncode == 77:
                    return 77
                assert a.returncode == 0, a.stderr
                report = judge_generation(output, answer, args.backend, model, cross_kernel=args.cross_kernel, record_only=args.record_only)
                print(json.dumps(dict(role=role, case=name, record_only=args.record_only, **report)), flush=True)
                ties += report['tie_cases']
                failed |= not args.record_only and report['status'] != 'passed'
                fresh = root / f'{role}-{name}-fresh'
                b = subprocess.run(command + ['--out', str(fresh), '--fresh-graphs'], capture_output=True, text=True)
                assert b.returncode == 0, b.stderr
                assert (output / 'tokens.json').read_bytes() == (fresh / 'tokens.json').read_bytes()
                actual = json.loads((output / 'tokens.json').read_text())
                a_logits = own_logits(output, actual)
                b_logits = own_logits(fresh, json.loads((fresh / 'tokens.json').read_text()))
                same = (output / 'logits.f32').read_bytes() == (fresh / 'logits.f32').read_bytes()
                print(json.dumps(dict(role=role, case=name, bitwise=same,
                    max_abs=float(np.max(np.abs(a_logits.astype(float)-b_logits.astype(float)), initial=0)))), flush=True)
                failed |= not same
                summary = json.loads((output / 'result.json').read_text())
                count = len(actual['generated'])
                assert summary['decode_plan_builds'] == int(count > 1) and summary['decode_plan_reuses'] == max(0, count - 2)
                assert summary['e'] == len(expected['input']) + count - 1
                if name == 'chat-intro' and args.record_only:
                    demo_out = root / f'{role}-demo'
                    demo = subprocess.run(base + ['--user', '\u7528\u4e00\u53e5\u8bdd\u4ecb\u7ecd\u4f60\u81ea\u5df1\u3002',
                                          '--max-tokens', '16', '--out', str(demo_out)], capture_output=True, text=True)
                    assert demo.returncode == 0, demo.stderr
                    own_logits(demo_out, json.loads((demo_out / 'tokens.json').read_text()))
                    comparison = judge_generation(demo_out, answer, args.backend, model,
                                                  cross_kernel=args.cross_kernel, record_only=True)
                    print(json.dumps(dict(role=role, case='chat-intro-demo', record_only=True, **comparison)), flush=True)
                if name == 'chat-intro' and not args.record_only:
                    demo = subprocess.run(base + ['--user', '\u7528\u4e00\u53e5\u8bdd\u4ecb\u7ecd\u4f60\u81ea\u5df1\u3002', '--max-tokens', '16', '--check', str(answer)], capture_output=True, text=True)
                    print(f'{role} demo exit={demo.returncode}; ' + demo.stderr.splitlines()[-1], flush=True)
                    failed |= demo.returncode != 0 or 'check: 16/16 tokens match' not in demo.stderr
            print(f'{role}: tie_cases={ties}', flush=True)
            failed |= not args.record_only and ties > 1
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
