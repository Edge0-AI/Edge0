"""Public CLI behavior on generated small assets and invalid inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

cli, fixture, root = map(Path, sys.argv[1:])
with tempfile.TemporaryDirectory(prefix='moe-cli-') as directory:
    subprocess.run([fixture, root, directory], check=True)
    def run(*args):
        return subprocess.run([cli, 'inspect', *args], text=True, capture_output=True)
    good = run('--model', directory)
    assert good.returncode == 0, good.stderr
    for line in ['architecture: edge0_35b', 'adapter: edge0_35b', 'gguf.architecture: qwen3next',
                 'integrity: size_only', 'file.base:', 'unused_roles: model provenance', 'model.layers: 2',
                 'model.layers.attention: 1', 'model.layers.deltanet: 0', 'model.experts: 2 used=2',
                 'tensors.base:', 'tensors.lora:', 'expert.0.0.gate:', 'expert.1.1.down:']:
        assert line in good.stdout, line
    report = json.loads(run('--model', directory, '--json').stdout)
    assert report['architecture'] == report['adapter'] == 'edge0_35b'
    assert report['gguf_architecture'] == 'qwen3next'
    assert len(report['expert_slices']) == 12
    for args in [[], ['--model'], ['--bad'], ['--model', directory, '--bad']]:
        assert run(*args).returncode == 2
    result = run('--model', str(Path(directory) / 'absent'))
    assert result.returncode == 1 and not result.stdout and 'asset_invalid' in result.stderr
    subprocess.run([fixture, root, directory, 'blk.0.ffn_gate_exps.weight'], check=True)
    missing = run('--model', directory)
    assert missing.returncode == 1 and not missing.stdout
    assert 'incompatible' in missing.stderr and 'blk.0.ffn_gate_exps.weight' in missing.stderr
    subprocess.run([fixture, root, directory], check=True)
    # Removing a required role must be rejected before any partial report.
    catalog = Path(directory) / 'assets.tsv'
    catalog.write_text(''.join(line for line in catalog.read_text().splitlines(True) if not line.startswith('lora\t')))
    result = run('--model', directory)
    assert result.returncode == 1 and not result.stdout and 'incompatible' in result.stderr and 'lora' in result.stderr
print('CLI text, JSON, usage and failure atomicity passed')

# The description lists the registered assets of a real small model exactly as assets.tsv does,
# and says they were not verified by this load.
tiny = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
if not tiny.is_dir():
    print('MOE_TINY/tiny-a is absent: asset identity in describe was not executed')
    raise SystemExit(77)
described = subprocess.run([cli, 'describe', '--model', str(tiny), '--device', 'cpu'], text=True, capture_output=True)
assert described.returncode == 0, described.stderr
description = json.loads(described.stdout)
rows = [line.split('\t') for line in (tiny / 'assets.tsv').read_text().splitlines()[1:] if line]
expected = sorted(({'role': role, 'filename': name, 'bytes': int(size), 'sha256': digest} for role, size, digest, name in rows),
                  key=lambda item: item['role'])
assert description['assets'] == expected, description['assets']
assert description['assets_verified'] is False
print('describe asset identity matches assets.tsv and is marked unverified')
