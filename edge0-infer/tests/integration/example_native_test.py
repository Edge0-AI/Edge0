"""Run the public native example against tiny assets."""
import argparse
import os
from pathlib import Path
import re
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--example', required=True)
args = parser.parse_args()
model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
if not (model / 'assets.tsv').is_file():
    print('SKIP: missing tiny-a')
    raise SystemExit(77)
result = subprocess.run([args.example, str(model), 'cpu'], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
counts = re.findall(r'reused_tokens=(\d+)', result.stderr)
assert len(counts) == 2 and int(counts[1]) > 0, result.stderr
assert len(result.stdout.strip().splitlines()) >= 2, result.stdout
print('PASS native example two turns and reuse')
