"""Compare the public interactive CLI with the established scripted-turn oracle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import select
import time
import tempfile


def invoke(command, text=None, success=True):
    # Pin UTF-8: the reference CLI emits UTF-8, but a non-UTF-8 console (for example GBK on a
    # Chinese Windows) would fail to decode generated text and drop output. Python's -I disables
    # PYTHONUTF8, so the encoding must be set here rather than through the environment.
    result = subprocess.run(command, input=text, capture_output=True, encoding='utf-8')
    if result.returncode == 77:
        raise SystemExit(77)
    if success:
        assert result.returncode == 0, (command, result.stderr)
    return result


def read_turns(folder):
    if not folder.exists():
        return []
    paths = sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    assert [p.name for p in paths] == [str(i) for i in range(1, len(paths) + 1)]
    return [(json.loads((p / 'tokens.json').read_text(encoding='utf-8')),
             json.loads((p / 'result.json').read_text(encoding='utf-8')),
             (p / 'answer.txt').read_text(encoding='utf-8')) for p in paths]


def compare(actual, expected, case, *, expected_is_run=True):
    assert len(actual) == len(expected), (case, 'turn count', len(actual), len(expected))
    for index, (a, b) in enumerate(zip(actual, expected), 1):
        for field in ['input', 'generated']:
            assert a[0][field] == b[0][field], (case, index, field, a[0][field], b[0][field])
        for field in ['reuse_method', 'reused_tokens', 'computed_tokens']:
            assert a[1][field] == b[1][field], (case, index, field, a[1][field], b[1][field])
        assert a[2] == b[2], (case, index, 'answer')
        check_fields = {'cancel_point', 'block_snapshot_seconds', 'random_draws', 'host_sampled_steps', 'sampling_readback_bytes', 'vocabulary'}
        assert not set(a[1]) & check_fields, (case, index, 'chat check fields')
        if expected_is_run:
            assert check_fields <= set(b[1]), (case, index, 'run check fields')
            assert set(a[1]) == set(b[1]) - check_fields, (case, index, 'summary schema')
        else:
            assert set(a[1]) == set(b[1]), (case, index, 'summary schema')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--backend', default='cpu', choices=['cpu', 'metal'])
    parser.add_argument('--file-stdin', action='store_true', help='Transport receives all stdin before launching the CLI')
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
    if not (model / 'assets.tsv').is_file():
        print('SKIP: missing tiny-a')
        return 77
    common = ['--model', str(model), '--device', args.backend, '--max-tokens', '4', '--temperature', '0']
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def chat(name, text, extra=()):
            folder = root / name
            result = invoke([args.cli, 'chat', *common, *extra, '--out', str(folder)], text)
            turns = read_turns(folder)
            assert result.stdout == ''.join(t[2] + '\n' for t in turns), (name, 'stdout contamination')
            assert 'Ready: load ' in result.stderr and '; /regen /new /quit\n> ' in result.stderr
            assert 'Generation: temperature=' in result.stderr
            return turns, result

        def oracle(name, turns, extra=(), system=None):
            script = root / (name + '.json')
            document = {'turns': turns}
            if system is not None:
                document['system'] = system
            script.write_text(json.dumps(document))
            folder = root / name
            invoke([args.cli, 'run', *common, *extra, '--turns', str(script), '--out', str(folder)])
            return read_turns(folder)

        prompt_file = root / 'prompt.txt'
        # /file reads this prompt as raw bytes, so keep LF (newline='') and UTF-8; otherwise
        # Windows text mode would write CRLF and change the tokens relative to the LF oracle.
        prompt_file.write_text('second\nline\n', encoding='utf-8', newline='')
        expected = oracle('multiline-oracle', [{'user': 'first\n/slash\nthird'}, {'user': 'second\nline\n'}])
        actual, _ = chat('multiline', '\"\"\"\nfirst\n/slash\nthird\n\"\"\"\n/file ' + str(prompt_file) + '\n/quit\n')
        compare(actual, expected, 'multiline and file')
        actual, boundary = chat('multiline-boundary', '\"\"\"\n\"\"\"\n/file ' + str(root / 'missing') + '\n\"\"\"\ndiscard\n')
        assert not actual and 'Cannot read file' in boundary.stderr

        for fragments in (0,3):
            expected=oracle(f'cancel-oracle-{fragments}', [{'user':'hello','cancel':{'after_fragments':fragments}},{'user':'again'}])
            actual,process=chat(f'cancel-{fragments}','hello\nagain\n/quit\n', ['--sigint-after-fragments',str(fragments)])
            compare(actual,expected,f'cancel-{fragments}')
            assert actual[0][1]['stop_reason']=='cancelled'
            assert bool(actual[0][2])==(fragments!=0)
            assert 'Cancellation latency:' in process.stderr
            if fragments==0:
                fresh=oracle('cancel-empty-fresh',[{'user':'again'}])
                for field in ('input','generated'):
                    assert actual[1][0][field]==fresh[0][0][field]
        # A long prefill keeps the timed signal observable without a latency gate.
        long_input='hello '*256
        expected=oracle('cancel-timer-oracle',[{'user':long_input,'cancel':{'after_fragments':0}},{'user':'again'}])
        actual,process=chat('cancel-timer',long_input+'\nagain\n/quit\n',['--sigint-after-seconds','0'])
        compare(actual,expected,'cancel-timer')
        assert actual[0][1]['stop_reason']=='cancelled' and not actual[0][0]['generated']
        assert 'Cancellation latency:' in process.stderr
        # Delivering a real SIGINT from the parent at the interactive prompt is POSIX-specific test
        # plumbing: select() rejects pipe handles on Windows, a parent cannot raise SIGINT in one
        # child (console Ctrl+C targets a whole group), and the 130 exit convention is POSIX. The
        # cancellation logic itself is exercised on Windows by the --sigint-after-* paths above,
        # which drive the same handler. Marked not executed on Windows per decision 6; see the
        # implementation report (problem 8).
        if args.file_stdin:
            print('NOT EXECUTED: parent-delivered SIGINT at the prompt requires live stdin; transport uses a file', flush=True)
        elif os.name == 'nt':
            print('NOT EXECUTED on Windows: parent-delivered SIGINT at the prompt (decision 6); '
                  'cancellation is covered by the development-flag paths above', flush=True)
        else:
            process=subprocess.Popen([args.cli,'chat',*common,'--no-warmup'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stderr=b'';deadline=time.monotonic()+60
            while not stderr.endswith(b'> '):
                assert time.monotonic()<deadline,'Prompt timeout'
                ready,_,_=select.select([process.stderr],[],[],1)
                if ready:
                    data=os.read(process.stderr.fileno(),4096)
                    assert data,(process.poll(),stderr)
                    stderr+=data
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10)==130
            stdout,tail=process.communicate()
            assert not stdout and b'Cleanup:' not in stderr+tail
            print('SIGINT during output, before compute, and at prompt passed',flush=True)

        turns = [{'user': 'hello'}, {'user': 'again'}, {'regenerate': True}, {'reset': True}, {'user': 'third'}]
        text = '\n \t\n/unknown\nhello\nagain\n /regen \r\n/new\nthird\n/quit\nignored\n'
        expected = oracle('oracle', turns, system='Be brief.')
        cold, cold_process = chat('cold', text, ['--no-warmup', '--system', 'Be brief.'])
        compare(cold, expected, '(a) scripted conversation')
        assert len(cold) == 4 and 'Unknown command' in cold_process.stderr
        warm, _ = chat('warm', text, ['--system', 'Be brief.'])
        compare(warm, cold, '(b) warmup isolation', expected_is_run=False)
        assert warm[0][1]['reuse_method'] == cold[0][1]['reuse_method'] == 'first', '(b) first turn'
        print('(a,b) scripted conversation and warmup isolation passed', flush=True)

        sampled = ['--temperature', '0.8', '--seed', '7']
        expected = oracle('sample-oracle', turns, sampled)
        actual, process = chat('sample-chat', 'hello\nagain\n/regen\n/new\nthird\n/quit\n', sampled)
        compare(actual, expected, 'sampled conversation')
        cold, _ = chat('sample-cold', 'hello\nagain\n/regen\n/new\nthird\n/quit\n', [*sampled, '--no-warmup'])
        compare(actual, cold, 'sampled warmup isolation', expected_is_run=False)
        assert process.stderr.count('seed=7') == len(actual) + 1
        assert all(t[1]['configuration']['seed'] == 7 for t in actual)
        unseeded, process = chat('sample-unseeded', 'hello\n/regen\n/regen\n/quit\n', ['--temperature', '0.8'])
        seeds = [t[1]['configuration']['seed'] for t in unseeded]
        assert len(seeds) == len(set(seeds)) == 3
        assert all(f'seed={seed}' in process.stderr for seed in seeds)
        print('sampled conversation, warmup isolation, per-turn seed and regeneration passed', flush=True)

        capacity = ['--context', '64', '--prefill-chunk', '64', '--model-option', 'prefill_ahead_min_tokens=64']
        expected = oracle('capacity-oracle', [{'user': 'hi'}, {'user': 'ok'}], capacity)
        rejected, process = chat('capacity', 'hi\n' + 'word ' * 80 + '\nok\n/quit\n', capacity)
        assert 'Input exceeds session capacity' in process.stderr and '; use /new' in process.stderr
        compare(rejected, expected, '(c) rejected input rollback')
        assert len(rejected) == 2
        print('(c) rejected input rollback passed', flush=True)

        empty, process = chat('empty', '/regen\n/quit\n')
        assert not empty and 'No answer to regenerate' in process.stderr, '(d) empty regeneration'
        assert not (root / 'empty').exists(), '(d) no result files'
        eof, _ = chat('eof', 'hello\n', ['--no-warmup'])
        compare(eof, oracle('eof-oracle', [{'user': 'hello'}]), '(e) EOF')
        process = invoke([args.cli, 'chat', '--model', str(root / 'missing'), '--device', args.backend], '', success=False)
        assert process.returncode != 0 and process.stderr.strip(), '(f) missing model'
        # Existing output must never be overwritten, and startup failures produce no answer.
        marker = root / 'existing'; marker.mkdir(); (marker / 'sentinel').write_text('keep')
        process = invoke([args.cli, 'chat', *common, '--out', str(marker)], '', success=False)
        assert process.returncode == 1 and 'already exists' in process.stderr and not process.stdout
        assert (marker / 'sentinel').read_text() == 'keep'
        print('(d,e,f) empty regeneration, EOF, and startup failures passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
