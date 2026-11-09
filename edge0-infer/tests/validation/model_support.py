"""Simplified-baseline effect check through the reference CLI (entry point of the model support validation).

  model_support.py run    --cli <edge0-infer> --model <assets> --out <new results dir> [--device metal]
  model_support.py review --results <dir> --verdicts <json>
  model_support.py render --results <dir> [--perf-report <report.json>]... [--out <file.md>]

Six fixed tasks are run once per routing mode through the official ``run`` entry (single turn, thinking off, greedy,
presence penalty 0). Criteria that a script can decide are decided by the script: the greeting answer is not empty,
the one-character answer is exactly that character, the arithmetic answer is 42 without other numbers than the
operands and the result. For the three tasks that need reading, the script applies only checks whose failure is
certain (empty answer, replacement character, a long fragment repeated four times in a row); everything else is
``pending`` and must be judged by a reviewer other than the implementer, with a written reason.

Outcomes are ``pass``, ``fail``, ``pending`` and ``not_run``. Pending is never counted as pass. ``render`` exits with 1
while any task is pending or not run, otherwise 0.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

if str(Path(__file__).resolve().parents[1] / 'perf') not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perf'))

import report as perf_report_module

TASKS_FILE = Path(__file__).resolve().parents[1] / 'fixtures' / 'validation' / 'simplified-baseline.json'
ROUTINGS = ('actual', 'predicted')
OUTCOMES = ('pass', 'fail', 'pending', 'not_run')
GOOD = '\u597d'            # the single character the eos task asks for
REPLACEMENT = '\ufffd'
OPERANDS_AND_RESULT = {17, 25, 42}
EXIT_DONE, EXIT_UNFINISHED, EXIT_INPUT = 0, 1, 2


class InputError(ValueError):
    """Bad arguments or files (exit code 2)."""


# ---------------------------------------------------------------- judging (pure)

def strip_margin(text):
    """Remove whitespace and punctuation (any script) from both ends."""
    def margin(ch):
        return ch.isspace() or unicodedata.category(ch).startswith('P')
    start, end = 0, len(text)
    while start < end and margin(text[start]):
        start += 1
    while end > start and margin(text[end - 1]):
        end -= 1
    return text[start:end]


def certain_failure(answer):
    """Reason the answer fails for sure, or None. Applies to every task that needs reading."""
    if not answer.strip():
        return 'empty answer'
    if REPLACEMENT in answer:
        return 'contains the replacement character'
    if re.search(r'(.{8,}?)\1{3,}', answer, re.S):
        return 'a fragment of at least 8 characters repeats 4 times in a row'
    return None


def judge(check, answer):
    """(status, reason) of the automatic judgement; status is pass, fail or pending."""
    if check == 'smoke':
        return ('pass', 'answer is not empty (three tokens: smoke only, not proof of usability)') if answer.strip() \
            else ('fail', 'empty answer')
    if check == 'exact_single_character':
        core = strip_margin(answer)
        return ('pass', 'answer is exactly the requested character') if core == GOOD \
            else ('fail', f'answer is not exactly the requested character (core: {core!r})')
    if check == 'arithmetic':
        numbers = [int(n) for n in re.findall(r'[0-9]+', answer)]
        if 42 not in numbers:
            return 'fail', 'answer does not contain 42'
        wrong = sorted({n for n in numbers if n not in OPERANDS_AND_RESULT})
        return ('fail', f'answer contains other numbers: {wrong}') if wrong else ('pass', 'contains 42 and no other numbers than 17, 25, 42')
    if check == 'semantic':
        reason = certain_failure(answer)
        return ('fail', reason) if reason else ('pending', 'needs review against the written criterion')
    raise InputError(f'Unknown check kind: {check}')


# ---------------------------------------------------------------- running

def dump(data):
    """JSON text of a result document with local paths replaced by environment variable names."""
    return perf_report_module.sanitize(json.dumps(data, ensure_ascii=False, indent=1)) + '\n'


def load_tasks(path=TASKS_FILE):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    tasks = document['tasks']
    if len({t['id'] for t in tasks}) != len(tasks):
        raise InputError('Duplicate task ids')
    return document, tasks


def execute(argv):
    # The reference CLI always writes UTF-8. Decode explicitly instead of relying on the OS
    # locale: on a non-UTF-8 console (for example GBK on a Chinese Windows) text mode would fail
    # to decode generated answers and drop stdout. Python's -I flag disables PYTHONUTF8, so the
    # encoding has to be pinned here rather than through the environment.
    result = subprocess.run(argv, capture_output=True, encoding='utf-8')
    return result.returncode, result.stdout, result.stderr


def last_json_line(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def parse_info(text):
    return dict(line.split(': ', 1) for line in text.splitlines() if ': ' in line)


def run_command(cli, model, device, routing, task):
    return [str(cli), 'run', '--model', str(model), '--device', device, '--model-option', f'routing={routing}',
            '--user', task['input'], '--max-tokens', str(task['max_tokens']), '--thinking', 'off',
            '--temperature', '0', '--presence-penalty', '0']


def not_run(task, routing, reason):
    return dict(task=task['id'], routing=routing, check=task['check'], status='not_run', auto_status='not_run', reason=reason,
                answer=None, stop_reason=None, length_stop=None, first_token_seconds=None, input_count=None, reviews=[])


def run_all(cli, model, out, device='metal', routings=ROUTINGS, tasks_path=TASKS_FILE, run=execute):
    """Run every task for every routing and write the results directory. Returns the identity dict."""
    out = Path(out)
    if out.exists():
        raise InputError(f'Refusing to overwrite existing results: {out}')
    document, tasks = load_tasks(tasks_path)
    out.mkdir(parents=True)
    identity = dict(tasks_source_sha256=document['source_sha256'], protocol=document['protocol'], device=device,
                    engine=None, assets=None, routings={})
    unavailable = None
    if not Path(cli).is_file():
        unavailable = f'CLI not found: {Path(cli).name}'
    elif not (Path(model) / 'assets.tsv').is_file():
        unavailable = 'asset directory not found or incomplete'
    if unavailable is None:
        code, stdout, stderr = run([str(cli), 'info'])
        if code:
            unavailable = f'info failed with exit code {code}'
        else:
            info = parse_info(stdout)
            identity['engine'] = {k: info.get(f'engine.{k}') for k in ('version', 'source_commit', 'source_modified')}
            identity['engine']['ggml_version'] = info.get('ggml.version')
    for routing in routings:
        results = []
        for task in tasks:
            if unavailable:
                results.append(not_run(task, routing, unavailable))
                continue
            code, stdout, stderr = run(run_command(cli, model, device, routing, task))
            summary = last_json_line(stderr)
            if code == 77:
                unavailable = f'device {device} unavailable'
                results.append(not_run(task, routing, unavailable))
                continue
            if code or summary is None:
                results.append(dict(not_run(task, routing, ''), status='fail', auto_status='fail',
                                    reason=f'run exited with code {code}' if code else 'no summary line on standard error',
                                    answer=stdout))
                continue
            status, reason = judge(task['check'], stdout)
            summary_time = summary.get('request_summary', {}).get('first_token_seconds')
            results.append(dict(task=task['id'], routing=routing, check=task['check'], status=status, auto_status=status, reason=reason,
                                answer=stdout, stop_reason=summary.get('stop_reason'), length_stop=summary.get('stop_reason') == 'length',
                                first_token_seconds=summary_time, input_count=summary.get('input_count'),
                                configuration=summary.get('configuration'), loading=summary.get('loading'), reviews=[]))
            identity['routings'][routing] = dict(loading=summary.get('loading'))
        (out / routing).mkdir()
        for result in results:
            (out / routing / f"{result['task']}.json").write_text(dump(result), encoding='utf-8')
    if identity['engine'] is not None:
        code, stdout, _ = run([str(cli), 'describe', '--model', str(model), '--device', device])
        if code == 0:
            description = json.loads(stdout)
            identity['assets'] = dict(adapter=description['adapter'], assets=description['assets'], assets_verified=description['assets_verified'])
    (out / 'identity.json').write_text(dump(identity), encoding='utf-8')
    return identity


# ---------------------------------------------------------------- reading results, review, render

def load_results(directory):
    directory = Path(directory)
    if not (directory / 'identity.json').is_file():
        raise InputError(f'Not a results directory: {directory}')
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    results = {}
    for routing_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for path in sorted(routing_dir.glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            results[f"{routing_dir.name}/{data['task']}"] = (path, data)
    if not results:
        raise InputError('No results found')
    return identity, results


def apply_verdicts(results, verdicts):
    """Validate all verdicts, then return the updated result documents. Raises InputError, changing nothing."""
    if not isinstance(verdicts, dict) or not verdicts:
        raise InputError('Verdicts must be a nonempty JSON object keyed by routing/task')
    updated = {}
    for key, entry in verdicts.items():
        if key not in results:
            raise InputError(f'Unknown result: {key}')
        data = json.loads(json.dumps(results[key][1]))
        if data['auto_status'] != 'pending':
            raise InputError(f"{key}: decided automatically ({data['auto_status']}); only pending items accept a verdict")
        if not isinstance(entry, dict) or set(entry) != {'verdict', 'reason', 'reviewer'}:
            raise InputError(f'{key}: expected exactly verdict, reason and reviewer')
        if entry['verdict'] not in ('pass', 'fail'):
            raise InputError(f'{key}: verdict must be pass or fail')
        if not isinstance(entry['reason'], str) or not entry['reason'].strip():
            raise InputError(f'{key}: reason must not be empty')
        if not isinstance(entry['reviewer'], str) or not entry['reviewer'].strip():
            raise InputError(f'{key}: reviewer must not be empty')
        data['reviews'].append(dict(entry))
        data['status'] = entry['verdict']
        data['reason'] = entry['reason']
        updated[key] = data
    return updated


def review(directory, verdicts_path):
    identity, results = load_results(directory)
    try:
        verdicts = json.loads(Path(verdicts_path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f'Cannot read verdicts: {error}')
    updated = apply_verdicts(results, verdicts)
    for key, data in updated.items():
        results[key][0].write_text(dump(data), encoding='utf-8')
    return sorted(updated)


def tally(results, routing):
    counts = dict.fromkeys(OUTCOMES, 0)
    for key, (_, data) in results.items():
        if key.startswith(routing + '/'):
            counts[data['status']] += 1
    return counts


def perf_consistency(identity, perf_report):
    """Lines saying whether a performance report was measured with the same engine commit and assets."""
    problems = []
    engine, assets = (perf_report.get('identity') or {}).get('engine') or {}, (perf_report.get('identity') or {}).get('model') or {}
    if engine.get('source_commit') != (identity.get('engine') or {}).get('source_commit'):
        problems.append(f"engine commit differs: performance report {engine.get('source_commit')}, effect run {(identity.get('engine') or {}).get('source_commit')}")
    mine = (identity.get('assets') or {}).get('assets')
    if assets.get('assets') != mine:
        problems.append('asset digests differ between the performance report and the effect run')
    return problems


def render(directory, perf_reports=()):
    """(markdown, exit code)."""
    identity, results = load_results(directory)
    routings = sorted({key.split('/')[0] for key in results})
    out = ['## Effect check on the simplified baseline', '',
           f"- Engine: {json.dumps(identity.get('engine'), sort_keys=True)}",
           f"- Assets: {json.dumps([(a['role'], a['sha256']) for a in (identity.get('assets') or {}).get('assets', [])])}",
           f"- Protocol: {json.dumps(identity.get('protocol'), sort_keys=True)}", f"- Device: {identity.get('device')}", '']
    unfinished = False
    out += ['| Routing | Pass | Fail | Pending review | Not run | Total |', '| --- | --- | --- | --- | --- | --- |']
    for routing in routings:
        c = tally(results, routing)
        total = sum(c.values())
        unfinished = unfinished or c['pending'] or c['not_run']
        out.append(f"| {routing} | {c['pass']} | {c['fail']} | {c['pending']} | {c['not_run']} | {total} |")
    out.append('')
    out.append('Pending review items are not counted as passed.')
    out.append('')
    for report in perf_reports:
        problems = perf_consistency(identity, report)
        out.append('Performance report identity: ' + ('consistent with this effect run.' if not problems else 'NOT CONSISTENT, the two are not combined: ' + '; '.join(problems)))
    if perf_reports:
        out.append('')
    for routing in routings:
        out += [f'### Routing {routing}', '']
        for key, (_, data) in sorted(results.items(), key=lambda kv: kv[0]):
            if not key.startswith(routing + '/'):
                continue
            out += [f"#### {data['task']} ({data['check']}): {data['status']}", '',
                    f"- Automatic result: {data['auto_status']}; reason: {data['reason']}",
                    f"- Stop reason: {data['stop_reason']}" + (' (length stop)' if data.get('length_stop') else ''),
                    f"- First token seconds: {data.get('first_token_seconds')}"]
            for item in data['reviews']:
                out.append(f"- Review by {item['reviewer']}: {item['verdict']}; {item['reason']}")
            out += ['', 'Answer:', '', '```text', data['answer'] if data['answer'] is not None else '(not run)', '```', '']
    return '\n'.join(out) + '\n', (EXIT_UNFINISHED if unfinished else EXIT_DONE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--cli', required=True, type=Path)
    r.add_argument('--model', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--device', default='metal', choices=['cpu', 'metal'])
    v = sub.add_parser('review')
    v.add_argument('--results', required=True, type=Path)
    v.add_argument('--verdicts', required=True, type=Path)
    d = sub.add_parser('render')
    d.add_argument('--results', required=True, type=Path)
    d.add_argument('--perf-report', action='append', default=[], type=Path)
    d.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            run_all(args.cli, args.model, args.out, args.device)
            return EXIT_DONE
        if args.command == 'review':
            print('reviewed: ' + ', '.join(review(args.results, args.verdicts)))
            return EXIT_DONE
        reports = [json.loads(p.read_text()) for p in args.perf_report]
        text, code = render(args.results, reports)
        if args.out:
            args.out.write_text(text, encoding='utf-8')
        else:
            sys.stdout.write(text)
        return code
    except (InputError, OSError, json.JSONDecodeError, KeyError) as error:
        print(f'model_support: {error}', file=sys.stderr)
        return EXIT_INPUT


if __name__ == '__main__':
    sys.exit(main())
