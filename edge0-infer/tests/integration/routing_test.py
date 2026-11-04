"""Bytewise routing regressions on independently run model sessions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cli', required=True)
    p.add_argument('--backend', default='cpu')
    p.add_argument('--profile', choices=['tiny', '35b'], default='tiny')
    args = p.parse_args()
    reference = Path(os.environ.get('MOE_TINY', '/nonexistent'))
    answers = Path(__file__).resolve().parents[2] / 'tests/fixtures/answers'
    models = [Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent'))] if args.profile == '35b' else [reference / 'tiny-a', reference / 'tiny-b']
    if any(not (model / 'assets.tsv').exists() for model in models):
        return 77
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tokens = root / 'input.json'
        tokens.write_text(json.dumps(dict(input=[1, 2, 3], generated=[])))
        for model in models:
            cases = ['chat-intro', 'plain-mixed', 'long-mixed'] if args.profile == '35b' else ['tiny']
            for case in cases:
                source = answers / f'35b-lora-cpu-flash-split/{case}/tokens.json' if args.profile == '35b' else tokens
                for slots in ([4] if args.profile == '35b' else [4, 8]):
                    outputs = []
                    modes = ['predicted', 'predicted']
                    for index, mode in enumerate(modes):
                        out = root / f'{model.name}-{case}-{slots}-{index}'
                        command = [args.cli, 'run', '--model', str(model), '--device', args.backend, '--tokens', str(source),
                                   '--temperature', '0', '--presence-penalty', '0', '--max-tokens', '16' if args.profile == '35b' else '33', '--out', str(out),
                                   '--model-option', f'routing={mode}', '--model-option', f'expert_slots={slots}']
                        result = subprocess.run(command, capture_output=True, text=True)
                        if result.returncode == 77:
                            return 77
                        assert result.returncode == 0, result.stderr
                        outputs.append(out)
                    for name in ['tokens.json', 'logits.f32']:
                        assert (outputs[0]/name).read_bytes() == (outputs[1]/name).read_bytes(), (model, case, slots, name)
                    print(model.name, case, slots, 'predicted', 'bytewise passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
