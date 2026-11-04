"""Serial, cooled ABAB sampling benchmark; emit compact measurements only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cli', required=True)
parser.add_argument('--cooldown', type=float, default=30)
args = parser.parse_args()
model = os.environ['MOE_ASSETS_35B']
base = [args.cli, 'bench', '--context', '4096', '--presence-penalty', '0', '--model', model, '--device', 'metal', '--user-file',
        str(Path(__file__).resolve().parents[1] / 'fixtures/bench/library-report-prompt.txt'),
        '--warmup', '10', '--ntok', '200', '--runs', '1', '--breakdown']
for label in ['greedy', 'recommended', 'greedy', 'recommended', 'unlimited']:
    print(json.dumps({'cooldown_seconds': args.cooldown, 'next': label}), flush=True)
    time.sleep(args.cooldown)
    extra = ['--temperature', '0'] if label == 'greedy' else [
        '--temperature', '1', '--top-k', '0' if label == 'unlimited' else '20', '--top-p', '.95', '--seed', '1']
    result = subprocess.run(base + extra, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    row = rows[0]
    physical = row['memory_snapshot']['physical']['values']['peak_bytes']
    print(json.dumps({'configuration': label, 'routing': rows[-1]['model']['loading']['model_options']['routing'], 'decode_tokens_per_second': row['decode_tokens_per_second'],
                      'physical_peak_bytes': physical, 'input_count': row['input_count'],
                      'sampling_seconds': row['breakdown']['fields'].get('sampling_seconds'),
                      'decode_seconds': row['decode_seconds'], 'accepted_tokens': row['accepted_tokens']}), flush=True)
    assert physical <= 4 * 1024**3
