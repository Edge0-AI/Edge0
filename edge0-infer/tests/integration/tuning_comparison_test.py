"""Compare generation bytes between two engine binaries/configurations, serially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--backend', choices=['cpu', 'metal'], required=True)
    parser.add_argument('--a-options', default='[]')
    parser.add_argument('--b-options', default='[]')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--matrix', action='store_true', help='Compare cached and direct reads with cached/used')
    parser.add_argument('--long', action='store_true', help='Use the fixed long prompt and eight generated tokens')
    args = parser.parse_args()
    reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
    assets = reference / 'tiny-a' if args.tiny else Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent'))
    answers = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers' / ('tiny-lora-cpu-flash-split' if args.tiny else '35b-lora-cpu-flash-split')
    if not (assets / 'assets.tsv').exists():
        print('SKIP: missing assets')
        return 77
    with tempfile.TemporaryDirectory() as directory:
        if args.long:
            prompt = Path(__file__).resolve().parents[1] / 'fixtures/bench/edge0-long-prompt.txt'
            rendered = subprocess.run([args.a, 'prompt', '--model', str(assets), '--user', prompt.read_text(), '--tokens'],
                                      capture_output=True, text=True, check=True)
            ids = [int(token) for token in rendered.stdout.splitlines()[-1].split()]
            assert len(ids) == 3282
            tokens = Path(directory) / 'long-input.json'
            tokens.write_text(json.dumps({'input': ids}))
            cases = [('long-prompt', tokens, 8)]
        else:
            cases = [(case, answers / case / 'tokens.json',
                      len(json.loads((answers / case / 'tokens.json').read_text())['generated']))
                     for case in ['chat-intro', 'plain-mixed', 'long-mixed']]
        configurations = [('a', args.a, json.loads(args.a_options)), ('b', args.b, json.loads(args.b_options))]
        if args.matrix:
            configurations = []
            for read in ['cached', 'direct']:
                side = 'a' if not configurations else read
                configurations.append((side, args.a if side == 'a' else args.b,
                                       ['--model-option', 'expert_read=' + read,
                                        '--model-option', 'prefill_read=used']))
            configurations.append(('auto', args.b, []))
        for case, tokens, count in cases:
            for side, cli, options in configurations:
                output = Path(directory) / case / side
                command = [cli, 'run', '--model', str(assets), '--device', args.backend, '--tokens', str(tokens),
                           '--temperature', '0', '--presence-penalty', '0', '--context', '4096', '--max-tokens', str(count), '--out', str(output), *options]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 77:
                    print('SKIP: backend unavailable')
                    return 77
                if result.returncode:
                    raise RuntimeError(result.stderr)
            for side, _, _ in configurations[1:]:
                digests = {}
                for name in ['tokens.json', 'logits.f32']:
                    a = (Path(directory) / case / 'a' / name).read_bytes()
                    b = (Path(directory) / case / side / name).read_bytes()
                    assert a == b, (args.backend, case, side, name, 'byte mismatch')
                    digests[name] = hashlib.sha256(a).hexdigest()
                print(json.dumps(dict(backend=args.backend, case=case, configuration=side,
                                      bitwise=True, sha256=digests)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
