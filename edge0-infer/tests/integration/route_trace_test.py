"""Trace toggling preserves logits; offline counts equal runtime counts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
spec = importlib.util.spec_from_file_location('route_trace', Path(__file__).resolve().parents[1] / 'perf/route_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
backend = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
if not (reference / 'tiny-a/assets.tsv').exists():
    raise SystemExit(77)
for mode, experts in [('actual', 'slots'), ('predicted', 'slots'), ('actual', 'resident'), ('predicted', 'resident')]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokens = root / 'tokens.json'
        tokens.write_text(json.dumps(dict(input=[1, 2, 3])))
        trace = root / 'trace.jsonl'
        for enabled in [False, True]:
            command = [sys.argv[1], 'run', '--device', backend, '--model', str(reference / 'tiny-a'), '--tokens', str(tokens),
                       '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '33', '--model-option', f'routing={mode}', '--model-option', f'experts={experts}', '--out', str(root / str(enabled))]
            if enabled:
                command += ['--route-trace', str(trace)]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode == 77:
                raise SystemExit(77)
            assert result.returncode == 0, result.stderr
        for name in ['tokens.json', 'logits.f32']:
            assert (root / 'False' / name).read_bytes() == (root / 'True' / name).read_bytes(), name
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert len(rows) == 32
        summary = module.summarize(rows)
        activity = json.loads((root / 'True/result.json').read_text())['weight_activity']
        if experts == 'resident':
            assert activity is None
            counts = dict(initiated=0)
        else:
            counts = activity['prefetch']
        assert all('next_layer' not in entry for row in rows for entry in row['layers'])
        assert summary['prepared_experts'] == counts['initiated'], (summary, counts)
        assert summary['consumed_matches'] == summary['consumed_experts'], summary
        if mode == 'predicted':
            assert summary['consumed_experts'] == 31*6*4
        print('Trace toggle bytewise equal; 32 rows; prepared counts and previous-head consumption match')
