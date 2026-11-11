"""chat for a delegated model: an HTTP client of the fake edge0-strata, driven through stdin."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_support import (assert_no_leftovers, environment, gone, make_engine, make_model, process_record,  # noqa: E402
                            strata_args, temp_dir)

ANSWER = 'Hello from fake Strata, answering in several pieces.'


def stats_lines(stderr):
    # A turn without visible reasoning prints its statistics right after the "> " prompt.
    return [line[line.index('[edge0-strata] '):] for line in stderr.splitlines() if '[edge0-strata] ' in line]


def chat(cli, model, engine, record, text, extra=(), **fake):
    result = subprocess.run([str(cli), 'chat', '--model', str(model), *strata_args(engine), *extra], input=text,
                            capture_output=True, encoding='utf-8', env=environment(record, **fake), timeout=120)
    sent = [json.loads(line) for line in (Path(record) / 'requests.jsonl').read_text().splitlines()] \
        if (Path(record) / 'requests.jsonl').exists() else []
    return result, [json.loads(item['body']) for item in sent if item['path'] == '/v1/chat/completions']


def kept_log(record, stderr):
    """A failed edge0-strata's log is kept, named on stderr, and is the only thing left in TMPDIR; removes it."""
    named = [line.split('log kept at ', 1)[1] for line in stderr.splitlines() if 'edge0-strata: log kept at ' in line]
    assert len(named) == 1, stderr
    log = Path(named[0])
    assert log.is_file() and log.parent.parent == temp_dir(record), (log, stderr)
    assert [p.name for p in temp_dir(record).iterdir()] == [log.parent.name], list(temp_dir(record).iterdir())
    text = log.read_text()
    log.unlink()
    log.parent.rmdir()
    return text


def turns(cli, root, engine, model):
    record = root / 'turns'
    attached = root / 'attached.txt'
    attached.write_bytes(('first line\n  indented line\nthird\té\n' + 'long line ' * 300 + '\n').encode())
    script = '\n'.join([
        'hello there',
        'second question',
        '/regen',
        '/think off',
        '"""',
        'block line one',
        '  block line two  ',
        '"""',
        f'/file {attached}',
        '/new',
        'fresh start',
        '/quit',
        'never sent',
    ]) + '\n'
    result, bodies = chat(cli, model, engine, record, script, ['--system', 'Be brief.'], reasoning='considering it')
    assert result.returncode == 0, result.stderr
    assert len(bodies) == 6, bodies
    system = {'role': 'system', 'content': 'Be brief.'}
    user = lambda text: {'role': 'user', 'content': text}  # noqa: E731
    assistant = {'role': 'assistant', 'content': ANSWER}
    assert all(b['stream'] is True and b['model'] == 'edge0_145b' for b in bodies)
    # Multi-turn history.
    assert bodies[0]['messages'] == [system, user('hello there')] and 'chat_template_kwargs' not in bodies[0]
    assert bodies[1]['messages'] == [system, user('hello there'), assistant, user('second question')]
    # /regen resends the conversation without the last answer.
    assert bodies[2]['messages'] == bodies[1]['messages']
    # /think off; the triple-quoted block and the file arrive verbatim.
    history = bodies[1]['messages'] + [assistant]
    assert bodies[3]['chat_template_kwargs'] == {'enable_thinking': False}
    assert bodies[3]['messages'] == history + [user('block line one\n  block line two  ')]
    assert bodies[4]['messages'] == bodies[3]['messages'] + [assistant, user(attached.read_text(encoding='utf-8'))]
    # /new keeps only the system message.
    assert bodies[5]['messages'] == [system, user('fresh start')] and bodies[5]['chat_template_kwargs'] == {'enable_thinking': False}
    # Bodies of 1 KiB or more go out at once, without "Expect: 100-continue" (edge0-strata never answers it).
    seen = [json.loads(line) for line in (record / 'requests.jsonl').read_text().splitlines()]
    assert len(seen[4]['body']) > 3000 and all(item['expect'] is None for item in seen)
    assert result.stdout.count(ANSWER) == 6
    # Reasoning goes to stderr, marked, only while thinking is on.
    assert result.stderr.count('[thinking]') == 3 and 'considering it' in result.stderr and 'considering it' not in result.stdout
    stats = stats_lines(result.stderr)
    assert len(stats) == 6 and all(line.endswith(', end_of_generation') and 'tok/s' in line for line in stats), stats
    assert 'Ready: load ' in result.stderr
    assert_no_leftovers(record)
    print('PASS turns: history, /regen, /think off, """ block, /file, /new, /quit, no process left')


def cancel(cli, root, engine, model):
    record = root / 'cancel'
    result, bodies = chat(cli, model, engine, record, 'tell me\nand then?\n', ['--sigint-after-fragments', '2'], chunk_delay=0.2)
    assert result.returncode == 0, result.stderr
    assert len(bodies) == 2, bodies
    partial = 'Hello from '
    assert result.stdout.startswith(partial) and ANSWER not in result.stdout.split('\n')[0]
    stats = stats_lines(result.stderr)
    assert stats[0].endswith(', cancelled'), stats
    # A cancelled turn with text keeps the partial answer in the history.
    assert bodies[1]['messages'] == [{'role': 'user', 'content': 'tell me'}, {'role': 'assistant', 'content': partial},
                                     {'role': 'user', 'content': 'and then?'}], bodies[1]
    events = [json.loads(line) for line in (record / 'events.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'client_closed' for e in events), 'the cancelled stream was not closed upstream'
    assert_no_leftovers(record)
    if os.name == 'nt':
        from strata_console_ctrl_c import run_console
        run_console(cli, model, engine, root / 'idle-interrupt', 'chat')
        for attempt in range(20):
            run_console(cli, model, engine, root / f'interrupt-eof-{attempt}', 'chat', eof=True)
    else:
        # Ctrl-C while waiting for input stops edge0-strata and exits.
        record = root / 'idle-interrupt'
        process = subprocess.Popen([str(cli), 'chat', '--model', str(model), *strata_args(engine)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment(record))
        line = ''
        while 'Ready: load ' not in line:
            line = process.stderr.readline()
            assert line, 'chat exited before it was ready'
        process.send_signal(signal.SIGINT)
        assert process.wait(60) == 130
        assert_no_leftovers(record)
        # Ctrl-C at the prompt with the input ending at the same moment: the watcher thread and the main thread both
        # reach the engine's stop; it must run once, exit 130, and leave no directory or process (or hang).
        deadline = time.time() + 240
        for attempt in range(20):
            record = root / f'interrupt-eof-{attempt}'
            process = subprocess.Popen([str(cli), 'chat', '--model', str(model), *strata_args(engine)], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment(record))
            line = ''
            while 'Ready: load ' not in line:
                line = process.stderr.readline()
                assert line, 'chat exited before it was ready'
            process.send_signal(signal.SIGINT)
            process.stdin.close()
            try:
                code = process.wait(max(1, min(30, deadline - time.time())))
            except subprocess.TimeoutExpired:
                process.kill()
                raise AssertionError(f'chat hung after Ctrl-C and end of input (attempt {attempt})')
            assert code == 130, (attempt, code, process.stderr.read())
            process.stderr.close()
            process.stdout.close()
            assert_no_leftovers(record)
    print('PASS cancel: partial answer kept, upstream closed, idle Ctrl-C stops edge0-strata and leaves no temporary directory, also when input ends at once (20 times)')


def failures(cli, root, engine, model):
    record = root / 'start-failure'
    result, _ = chat(cli, model, engine, record, 'hi\n', fail_start=4)
    assert result.returncode == 1 and 'exited before it was ready (exit code 4)' in result.stderr, result.stderr
    assert 'edge0-strata: fake strata: failing on purpose' in result.stderr, result.stderr  # the log tail is shown
    assert 'failing on purpose' in kept_log(record, result.stderr)
    assert_no_leftovers(record)
    record = root / 'exited'
    process = subprocess.Popen([str(cli), 'chat', '--model', str(model), *strata_args(engine)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8',
                               env=environment(record, exit_after=1))
    process.stdin.write('one\n')
    process.stdin.flush()
    assert gone(process_record(record)['pid'], timeout=30), 'fake strata did not exit after its first answer'
    stdout, stderr = process.communicate('two\nthree\n', timeout=60)
    # A failure like a failed start: exit code 1, the log's last lines on stderr, the log kept.
    assert process.returncode == 1 and stdout == ANSWER + '\n', (process.returncode, stdout, stderr)
    assert 'edge0-strata has exited (exit code 3); restart chat' in stderr, stderr
    log = kept_log(record, stderr)
    assert log.strip() and log.strip().splitlines()[-1] in stderr, (log, stderr)
    assert len((record / 'requests.jsonl').read_text().splitlines()) == 1
    assert_no_leftovers(record)
    # Exiting while turns are queued: whichever turn finds it gone (before or during its request) ends the chat.
    record = root / 'exited-queued'
    result, bodies = chat(cli, model, engine, record, 'one\ntwo\nthree\nfour\nfive\n', exit_after=1, chunk_delay=0.05)
    assert result.returncode == 1 and 'edge0-strata has exited (exit code 3); restart chat' in result.stderr, result.stderr
    assert 1 <= len(bodies) < 5 and 'did not answer' not in result.stderr, result.stderr
    kept_log(record, result.stderr)
    assert_no_leftovers(record)
    record = root / 'rejected'
    result, _ = chat(cli, model, engine, record, 'hi\n', ['--device', 'metal'])
    assert result.returncode == 2 and 'CUDA GPU' in result.stderr and not record.exists()
    result, _ = chat(cli, model, engine, record, 'hi\n', ['--temperature', '0'])
    assert result.returncode == 2 and '--temperature' in result.stderr and not record.exists()
    print('PASS failures: start failure and engine exit exit 1 with the log tail and a kept log, rejected options start nothing')


def passthrough(cli, root, engine, model):
    tuning = ['--pool-workers', '24', '--pcie-mode', 'dma', '-x', 'direct']
    given = [item for word in tuning for item in ('--strata-arg', word)]
    record = root / 'passthrough'
    result, bodies = chat(cli, model, engine, record, 'hi\n', given)
    assert result.returncode == 0 and len(bodies) == 1, (result.returncode, result.stderr)
    config = json.loads((record / 'config.json').read_text())
    assert config['args'] == ['--pack', str(model), '--ple-io', 'direct', '--max-context', '131072', *tuning], config
    assert_no_leftovers(record)
    for name, words, message in (('empty', ['--pool-workers', ''], 'invalid_argument: --strata-arg needs a value'),
                                 ('template', ['--ple-io', 'ram'], 'invalid_argument: --strata-arg --ple-io: set by the launch template'),
                                 ('context', ['--max-context', '4096'], 'invalid_argument: --strata-arg --max-context: use --context')):
        record = root / ('passthrough-' + name)
        result, _ = chat(cli, model, engine, record, 'hi\n', [item for word in words for item in ('--strata-arg', word)])
        assert result.returncode == 2 and message in result.stderr, (name, result.returncode, result.stderr)
        assert 'log kept at' not in result.stderr, (name, result.stderr)
        assert not record.exists(), ('a process was started', name)
        assert not list(temp_dir(record).iterdir()), ('temporary directories left behind', name)
    print('PASS passthrough: --strata-arg words end the engine args; empty, template options and --max-context start nothing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        engine = make_engine(root / 'engine')
        model = make_model(root / 'model')
        turns(args.cli, root, engine, model)
        cancel(args.cli, root, engine, model)
        failures(args.cli, root, engine, model)
        passthrough(args.cli, root, engine, model)
        fp4 = make_model(root / 'fp4', model_name='edge0-120b-fp4')
        record = root / 'fp4-chat'
        result, bodies = chat(args.cli, fp4, engine, record, 'hi\n')
        assert result.returncode == 0 and bodies[0]['model'] == 'edge0_120b', result.stderr
        config = json.loads((record / 'config.json').read_text())
        assert config['env']['STRATA_LAT_FUSED_Q3K'] == '0' and not (fp4 / 'scales').exists()
        assert config['args'][1] == str(fp4)
        assert_no_leftovers(record)
    print('PASS chat strata')


if __name__ == '__main__':
    main()
