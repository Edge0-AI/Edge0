"""Derive invalid fixtures without changing any source asset."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

cli, variant = map(Path, sys.argv[1:])
if not os.environ.get('MOE_TINY'):
    print('not executed: MOE_TINY unset')
    sys.exit(77)
source = Path(os.environ['MOE_TINY']) / 'tiny-a'

def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def hashes():
    return {row.split('\t')[3]: hash_file(source / row.split('\t')[3]) for row in (source / 'assets.tsv').read_text().splitlines()[1:]}

before = hashes()
with tempfile.TemporaryDirectory(prefix='moe-variants-') as directory:
    dest = Path(directory)
    for mode, category, name in [('missing', 'incompatible', 'blk.0.ffn_gate_exps.weight'),
                                 ('extra', 'incompatible', 'unexpected_tensor'),
                                 ('type', 'incompatible', 'blk.0.attn_norm.weight'),
                                 ('metadata', 'incompatible', 'qwen3next.expert_count'),
                                 ('architecture', 'unsupported', 'edge0_35b'),
                                 ('role', 'incompatible', 'lora'), ('payload', 'asset_invalid', 'base.gguf')]:
        shutil.copytree(source, dest, dirs_exist_ok=True)
        if mode in ('missing', 'extra', 'type', 'metadata'):
            subprocess.run([variant, source / 'base.gguf', dest / 'base.gguf', mode], check=True)
        elif mode == 'architecture':
            data = json.loads((dest / 'model.json').read_text())
            data['architecture'] = 'edge0_8b'
            (dest / 'model.json').write_text(json.dumps(data))
        elif mode == 'payload':
            # Last byte belongs to tensor payload or its final padding. Determine
            # the actual expert payload offset from the valid public report.
            valid = subprocess.run([cli, 'inspect', '--model', source, '--json'], capture_output=True, text=True, check=True)
            offset = json.loads(valid.stdout)['expert_slices'][0]['offset']
            with (dest / 'base.gguf').open('r+b') as stream:
                stream.seek(offset)
                byte = stream.read(1)
                stream.seek(offset)
                stream.write(bytes([byte[0] ^ 1]))
        rows = ['role\tbytes\tsha256\tfilename']
        if mode != 'payload':
            for row in (source / 'assets.tsv').read_text().splitlines()[1:]:
                role, _, _, filename = row.split('\t')
                if mode == 'role' and role == 'lora':
                    continue
                path = dest / filename
                rows.append(f'{role}\t{path.stat().st_size}\t{hash_file(path)}\t{filename}')
            (dest / 'assets.tsv').write_text('\n'.join(rows) + '\n')
        command = [cli, 'inspect', '--model', dest]
        if mode == 'payload':
            normal = subprocess.run(command, capture_output=True, text=True)
            assert normal.returncode == 0 and 'integrity: size_only' in normal.stdout, normal.stderr
            assert (dest / 'assets.tsv').read_bytes() == (source / 'assets.tsv').read_bytes()
            command.append('--digest')
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 1 and not result.stdout, (mode, result)
        assert category in result.stderr and name in result.stderr, (mode, result.stderr)
        print(f'{mode}: rejected as {category}, includes {name}')
assert before == hashes(), 'source assets modified'
print('source digests unchanged; payload-only mutation detected only by digest')
