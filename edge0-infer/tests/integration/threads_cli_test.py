"""CPU thread counts preserve exact logits across real prefill/decode transitions."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (model / 'assets.tsv').is_file():
        return 77
    if (os.cpu_count() or 512) < 8:
        print('SKIP: the 1/4/8 protocol requires at least eight logical CPUs')
        return 77
    tokens = Path(__file__).resolve().parents[1] / 'fixtures/answers/tiny-lora-cpu-flash-split/chat-intro/tokens.json'
    with tempfile.TemporaryDirectory() as tmp:
        baseline = None
        for decode, prefill in itertools.product((1, 4, 8), repeat=2):
            out = Path(tmp) / f'{decode}-{prefill}'
            result = subprocess.run([args.cli, 'forward', '--model', str(model), '--device', 'cpu',
                                     '--tokens', str(tokens), '--split-last', '--out', str(out),
                                     '--threads', str(decode), '--prefill-threads', str(prefill)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            runtime = json.loads(result.stdout)['runtime']
            assert runtime == dict(cpu_threads=decode, cpu_prefill_threads=prefill,
                                   cpu_threads_source='explicit_value', cpu_prefill_threads_source='explicit_value')
            logits = (out / 'logits.f32').read_bytes()
            if baseline is None:
                baseline = logits
            assert logits == baseline, f'Bitwise mismatch: decode={decode}, prefill={prefill}'
            print(f'decode={decode} prefill={prefill}: bitwise equal', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
