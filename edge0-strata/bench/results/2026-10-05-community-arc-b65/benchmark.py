#!/usr/bin/env python3
"""Three fresh native runs; only numeric timing and hash receipts persist.

Use under an independent GPU/resource guard, with the model router isolated.
Run: python campaign.py --source SOURCE --profile profile.json --out OUTPUT
RAM logs must live on tmpfs. No generated content is written by this script.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

TASKS = [
 'Write a complete Python standard-library LRU cache module with expiry and locking, validation, iteration, detailed docstrings, demonstrations and boundary-case discussion. Write at least 1500 words of code and explanation.',
 'Write a complete Python standard-library CSV import module with typed validation, error collection, transaction staging, provenance, test examples and extensive documentation. Write at least 1500 words of code and explanation.',
 'Write a detailed story of at least 1500 words about a lighthouse keeper and marine biologist deciphering a research notebook. Develop their distinct voices, observations, uncertainty, and discoveries.',
 'Explain accurate inventory audits, discrepancy investigation, provenance, reconciliation, and periodic checks in at least 1500 words with concrete examples and worked scenarios.',
 'Act as an incident coordinator. Develop a detailed operational response plan for a failed local service: evidence collection, hypotheses, bounded tests, rollback, recovery verification and communication. Include worked scenarios and at least 1500 words.'
]
WARMUP = 'Write a detailed 1500-word guide to maintaining a community garden, with seasonal plans, worked examples and practical checklists.'
FILLER = '\n'.join(f'Archive record {i}: shelf {i%17}, item {(i*37)%997}, status reviewed, source verified, date unknown. Preserve provenance and record uncertainty.' for i in range(900))

def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ram', type=Path, default=Path(os.environ.get('STRATA_TEST_RAM', '/run/strata-community')))
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()
    assert 1 <= args.runs <= 3
    assert str(args.ram.resolve()).startswith('/run/'), 'use a RAM-backed /run directory'
    assert args.ram.is_dir()
    args.out.mkdir(exist_ok=True)
    assert not (args.out / 'requests.json').exists(), 'do not overwrite a consumed run'
    sys.path[:0] = [str(args.source), str(args.source / 'tools')]
    from serve.server import StrataEngine, ENGINE_REQUEST
    from serve.frontend import ChatTemplate
    from strata_tokenizer import Tokenizer

    class MeasuredEngine(StrataEngine):
        def _parse_done(self, line):
            self.done_line = line
            return super()._parse_done(line)

    assert 'STRATA_DBG_NAN' not in os.environ, 'qualification diagnostic must be absent'
    refs=json.loads((Path(__file__).parent/'reference-output-hashes.json').read_text())
    profile = json.loads(args.profile.read_text())
    native_args = [os.path.expandvars(x) for x in profile['args']]
    binary = os.path.expandvars(profile['exe'])
    cwd = os.path.expandvars(profile['cwd'])
    tok = Tokenizer.from_gguf(Path(native_args[native_args.index('--native') + 1]))
    tpl = ChatTemplate(Path(native_args[native_args.index('--pack') + 1]) / 'tokenizer/chat_template.jinja')
    sampling = {'temperature': 0.0, 'seed': 17}
    rows, runs = [], []
    cancel = threading.Event()
    interrupted = False
    engine = None
    failure = None

    def stop(*_):
        nonlocal interrupted
        interrupted = True
        cancel.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        for name, data in [('requests', rows), ('runs', runs)]:
            (args.out / (name + '.json')).write_text(json.dumps(data, indent=2) + '\n')

    def shaped(task, target):
        rendered = tpl.render([
            {'role': 'system', 'content': 'Follow the requested format and provide a thorough answer.'},
            {'role': 'user', 'content': 'PAD_MARKER\n' + task}
        ], enable_thinking=False)
        before, after = rendered.split('PAD_MARKER')
        a, b = tok.encode(before, parse_special=True), tok.encode(after, parse_special=True)
        pad = tok.encode(FILLER)
        assert 0 < target - len(a) - len(b) <= len(pad)
        ids = a + pad[:target - len(a) - len(b)] + b
        assert len(ids) == target
        return ids

    fixtures = {(i, n): shaped(task, n) for n in (512, 7000) for i, task in enumerate(TASKS + [WARMUP])}
    (args.out / 'fixtures.json').write_text(json.dumps([
        {'task_index': i, 'prompt_tokens': n, 'input_ids_sha256': digest(ids), 'warmup': i == 5}
        for (i, n), ids in fixtures.items()
    ], indent=2) + '\n')

    def ask(ids, cap, label, measured=False, expected=None, json_check=False):
        assert not interrupted
        offset = log.stat().st_size
        engine.done_line = None
        start_utc = time.time()
        started = time.monotonic()
        arrivals, output = [], []
        for token in engine.generate(ids, cap, sampling, cancel):
            if token is not None:
                arrivals.append(time.monotonic())
                output.append(token)
        elapsed = time.monotonic() - started
        native = dict(engine.last)
        text = tok.decode(output).replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
        valid = (engine.done_line is not None and len(output) == native['generated']
                 and native['prompt_tokens'] == len(ids) and native['reused'] == 0
                 and native['prompt_read'] == len(ids) and bool(output))
        if expected is not None:
            valid &= text == expected
        if json_check:
            try:
                obj = json.loads(text)
                valid &= set(obj) == {'answer', 'ok'} and type(obj['answer']) is int and obj['ok'] is True
            except (ValueError, TypeError):
                valid = False
        if measured or label.startswith('warm-shape'):
            valid &= len(output) == cap and native['finish'] == 'length'
        with log.open('r') as f:
            f.seek(offset)
            lines = f.read().splitlines()
        timing = [line for line in lines if ENGINE_REQUEST.search(line)]
        row = {'run': len(runs), 'label': label, 'measured': measured,
               'started_utc': start_utc, 'input_tokens': len(ids), 'output_cap': cap,
               'output_tokens': len(output), 'input_ids_sha256': digest(ids),
               'output_ids_sha256': digest(output), 'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'sampling': sampling, 'native': native, 'native_done_line': engine.done_line,
               'native_timing_lines': timing,
               'prompt_tps': native['prompt_read'] * 1000 / native['prompt_ms'],
               'decode_tps': native['generated'] * 1000 / native['decode_ms'],
               'ttft_s': arrivals[0] - started if arrivals else None,
               'post_first_tps': (len(output) - 1) / (arrivals[-1] - arrivals[0]) if len(output) > 1 else None,
               'total_s': elapsed, 'valid': bool(valid)}
        if label == 'json-quality':
            row['math_answer_correct'] = json.loads(text).get('answer') == 50 if valid else False
            row['math_answer_is_runtime_gate'] = False
        if measured:
            ref=refs[label]
            row['matches_reference_output_ids']=all(row[k]==ref[k] for k in ref)
            assert row['matches_reference_output_ids'], 'reference output stream changed: '+label
        rows.append(row)
        save()
        print(json.dumps({'run': len(runs), 'label': label, 'measured': measured,
                          'prompt_tps': row['prompt_tps'], 'decode_tps': row['decode_tps'],
                          'valid': bool(valid)}), flush=True)
        assert valid, 'protocol/length/isolation failure: ' + label
        assert not interrupted, 'guard interruption'

    try:
        for run_index in range(args.runs):
            log = args.ram / f'native-{run_index + 1}.log'
            runs.append({'run': run_index + 1, 'binary_sha256': hashlib.file_digest(Path(binary).open('rb'), 'sha256').hexdigest(),
                         'sampling': sampling, 'args': native_args})
            save()
            started = time.monotonic()
            engine = MeasuredEngine(binary, native_args, cwd=cwd, log=str(log), env=dict(os.environ))
            runs[-1]['startup_ready_s'] = time.monotonic() - started
            runs[-1]['info'] = engine.info
            save()
            for label, prompt, expected, jc in [
                ('isolation-a', 'Reply only with CANARY_MAPLE_73.', 'CANARY_MAPLE_73', False),
                ('isolation-b', 'Reply only with CANARY_RIVER_29.', 'CANARY_RIVER_29', False),
                ('json-quality', 'Calculate 2**5 + 6*3. Return exactly one JSON object with keys answer (the integer result) and ok (true), no markdown.', None, True)
            ]:
                ids = tok.encode(tpl.render([{'role': 'user', 'content': prompt}], enable_thinking=False), parse_special=True)
                ask(ids, 64, label, expected=expected, json_check=jc)
            for n in (512, 7000):
                ask(fixtures[(5, n)], 640, f'warm-shape-{n}')
                for i in range(5):
                    ask(fixtures[(i, n)], 640, f'bench-{n}-{i}', measured=True)
            ids = tok.encode(tpl.render([{'role': 'user', 'content': 'Reply only with CANARY_MAPLE_73.'}], enable_thinking=False), parse_special=True)
            ask(ids, 64, 'post-long-isolation', expected='CANARY_MAPLE_73')
            proc = engine.proc
            started = time.monotonic()
            proc.stdin.write('QUIT\n')
            proc.stdin.flush()
            proc.wait(timeout=75)
            runs[-1]['native_exit'] = proc.returncode
            runs[-1]['quit_exit_s'] = time.monotonic() - started
            assert proc.returncode == 0
            engine.close()
            engine = None
            raw = log.read_text()
            assert not re.search(r'\b[1-9]\d* (?:of \d+ logits )?non-finite\b|\bnonfinite [1-9]\d*', raw)
            runs[-1]['startup_lines'] = [line for line in raw.splitlines()
                if re.match(r'^strata (?:generate|serve): (?:GPU \d+:|PCIe probe:|expert cache \d|\d+ of \d+ experts missing|\d+ MiB of VRAM free)', line)]
            runs[-1]['native_log_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
            save()
            for _ in range(5):
                time.sleep(1)
                assert not interrupted, 'guard interruption during delayed fault observation'
    except BaseException as e:
        failure = {'type': type(e).__name__, 'reason': str(e)}
    finally:
        if engine is not None:
            proc = engine.proc
            if proc is not None and proc.poll() is None:
                proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=75)
            engine.close()
            runs[-1]['native_exit'] = None if proc is None else proc.returncode
        save()
        (args.out / 'result.json').write_text(json.dumps({'failure': failure,
            'run_count': len(runs), 'requests': len(rows), 'measured_requests': sum(r['measured'] for r in rows),
            'generated_content_persisted': False}, indent=2) + '\n')
    if failure:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
