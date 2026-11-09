"""Record the fixed long-prompt chunking experiment for independent answer review."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--read', required=True, choices=['cached', 'direct'])
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite evidence')
    args.raw.mkdir(parents=True, exist_ok=False)
    prompt = Path('tests/fixtures/bench/edge0-long-prompt.txt').read_text()
    lines = ['# 4.2 End-to-end chunking record', '',
             'Metal; temperature 0; max_tokens 128; expert_read=' + args.read + '; prefill_read=used.',
             'Answer quality is reserved for the independent verifier.', '']
    for repetitions in [1, 2]:
        runs = []
        for chunk in [512, None]:
            label = str(chunk) if chunk else 'default'
            folder = args.raw / f'{repetitions}-{label}'
            command = [args.cli, 'run', '--model', args.model, '--device', 'metal',
                       '--user', prompt * repetitions, '--temperature', '0', '--presence-penalty', '0', '--context', '4096', '--max-tokens', '128',
                       '--model-option', 'expert_read=' + args.read, '--model-option', 'prefill_read=used',
                       '--out', str(folder)]
            if repetitions == 2:
                command += ['--context', '8192']
            if chunk:
                command += ['--prefill-chunk', str(chunk)]
            result = subprocess.run(command, capture_output=True, text=True)
            (args.raw / f'{repetitions}-{label}.stdout').write_text(result.stdout)
            (args.raw / f'{repetitions}-{label}.stderr').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'Run {repetitions}-{label} failed: {result.stderr}')
            summary = json.loads((folder / 'result.json').read_text())
            tokens = json.loads((folder / 'tokens.json').read_text())
            assert summary['loading']['prefill_chunk'] == (chunk or 4096)
            runs.append((label, summary, tokens, result.stdout))
        prefix = 0
        for a, b in zip(runs[0][2]['generated'], runs[1][2]['generated']):
            if a != b:
                break
            prefix += 1
        lines += [f'## Prompt repetitions: {repetitions}', '', f'Equal generated-token prefix: {prefix}', '',
                  '| Chunk | Input tokens | Generated tokens | First token (s) | Physical peak (bytes) |',
                  '| --- | --- | --- | --- | --- |']
        for label, summary, tokens, answer in runs:
            lines.append(f"| {label} | {len(tokens['input'])} | {len(tokens['generated'])} | "
                         f"{summary['request_summary']['first_token_seconds']:.9f} | "
                         f"{summary['memory_snapshot']['physical']['values']['peak_bytes']} |")
        for label, summary, tokens, answer in runs:
            fence = '`' * max(4, max((len(part) for part in answer.split() if set(part) == {'`'}), default=0) + 1)
            lines += ['', f'### Answer: {label}', '', fence + 'text', answer, fence, '']
        args.out.write_text('\n'.join(lines))
    if args.out.stat().st_size > 256 * 1024:
        raise ValueError('Evidence exceeds 256 KiB')


if __name__ == '__main__':
    main()
