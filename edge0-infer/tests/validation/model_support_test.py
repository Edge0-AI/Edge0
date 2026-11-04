"""Simplified-baseline entry: automatic criteria, review rules, summary and exit codes, with a fake executor."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('model_support', HERE / 'model_support.py')
ms = importlib.util.module_from_spec(spec)
sys.modules['model_support'] = ms
spec.loader.exec_module(ms)

# Characters are written as escapes: this file is a public (English only) path.
GOOD = '\u597d'
GOOD_OF = GOOD + '\u7684'          # "good" plus a particle
NOT_GOOD = '\u4e0d' + GOOD
FULL_STOP = '\u3002'
EXCLAMATION = '\uff01'


class JudgeTest(unittest.TestCase):
    def test_eos_accepts_only_the_character_with_margins(self):
        for text in [GOOD, GOOD + FULL_STOP, ' ' + GOOD + EXCLAMATION, '\n' + GOOD + '.\n', '"' + GOOD + '"']:
            self.assertEqual(ms.judge('exact_single_character', text)[0], 'pass', text)
        for text in [GOOD_OF, NOT_GOOD, '', GOOD + GOOD, 'ok', GOOD + ' ' + GOOD]:
            self.assertEqual(ms.judge('exact_single_character', text)[0], 'fail', text)

    def test_arithmetic(self):
        for text in ['42', '17 + 25 = 42', '42.', 'The result is 42']:
            self.assertEqual(ms.judge('arithmetic', text)[0], 'pass', text)
        for text in ['43', '42, no it is 43', '', 'forty-two', '17 + 25 = 43', '420']:
            self.assertEqual(ms.judge('arithmetic', text)[0], 'fail', text)

    def test_smoke(self):
        self.assertEqual(ms.judge('smoke', 'hello')[0], 'pass')
        self.assertEqual(ms.judge('smoke', ' \n')[0], 'fail')
        self.assertIn('smoke', ms.judge('smoke', 'x')[1])

    def test_semantic_only_fails_for_certain_defects(self):
        self.assertEqual(ms.judge('semantic', 'The sea is calm today.')[0], 'pending')
        self.assertEqual(ms.judge('semantic', '   ')[0], 'fail')
        self.assertEqual(ms.judge('semantic', 'broken \ufffd text')[0], 'fail')
        loop = 'abcdefgh' * 4
        self.assertEqual(ms.judge('semantic', 'start ' + loop + ' end')[0], 'fail')
        self.assertEqual(ms.judge('semantic', 'start ' + 'abcdefgh' * 3 + ' end')[0], 'pending')
        short_loop = 'abc' * 10
        self.assertEqual(ms.judge('semantic', short_loop)[0], 'pending')

    def test_unknown_check_is_an_error(self):
        with self.assertRaises(ms.InputError):
            ms.judge('other', 'x')


class FixtureTest(unittest.TestCase):
    def test_tasks_are_the_six_fixed_ones(self):
        document, tasks = ms.load_tasks()
        self.assertEqual([t['id'] for t in tasks], ['short', 'chinese', 'english', 'long', 'eos', 'arithmetic'])
        self.assertEqual([t['max_tokens'] for t in tasks], [3, 64, 64, 128, 64, 64])
        self.assertEqual({t['check'] for t in tasks}, {'smoke', 'semantic', 'exact_single_character', 'arithmetic'})
        long_input = tasks[3]['input']
        self.assertEqual(long_input.count('\n'), 5)
        record = long_input.split('\n')[1]
        self.assertEqual(long_input.split('\n')[1:5], [record] * 4)
        self.assertEqual(document['source_sha256'][:8], 'd529614c')


ANSWERS = {'short': 'Hi', 'chinese': 'plain answer', 'english': 'The sea is blue.', 'long': 'summary text',
           'eos': GOOD, 'arithmetic': '17 + 25 = 42'}


def fake_executor(answers=None, record=None, info_code=0, run_code=0):
    answers = dict(ANSWERS, **(answers or {}))
    _, tasks = ms.load_tasks()
    by_input = {t['input']: t['id'] for t in tasks}

    def run(argv):
        if record is not None:
            record.append(list(argv))
        if argv[1] == 'info':
            return info_code, 'engine.version: 0.1.0\nengine.source_commit: ' + 'c' * 40 + '\nengine.source_modified: no\nggml.version: v\n', ''
        if argv[1] == 'describe':
            assets = [dict(role='base', filename='base.gguf', bytes=1, sha256='0' * 64)]
            return 0, json.dumps(dict(adapter='edge0_35b', assets=assets, assets_verified=False)), ''
        task = by_input[argv[argv.index('--user') + 1]]
        summary = dict(stop_reason='length' if task in ('chinese', 'english', 'long') else 'end_of_generation', input_count=10,
                       request_summary=dict(first_token_seconds=1.5), loading=dict(directory='m'), configuration=dict(temperature=0))
        return run_code, answers[task], 'log line\n' + json.dumps(summary) + '\n'
    return run


class RunReviewRenderTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-support-')
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.cli = self.root / 'edge0-infer'
        self.cli.write_text('')
        self.model = self.root / 'assets'
        self.model.mkdir()
        (self.model / 'assets.tsv').write_text('role\tbytes\tsha256\tfilename\n')
        self.out = self.root / 'results'

    def run_all(self, **options):
        calls = []
        ms.run_all(self.cli, self.model, self.out, run=fake_executor(record=calls, **options))
        return calls

    def verdicts(self, data):
        path = self.root / 'verdicts.json'
        path.write_text(json.dumps(data))
        return path

    def test_run_judges_and_keeps_raw_outputs(self):
        calls = self.run_all()
        _, results = ms.load_results(self.out)
        self.assertEqual(sorted(results), sorted(f'{r}/{t}' for r in ms.ROUTINGS for t in ANSWERS))
        status = {key: data['status'] for key, (_, data) in results.items()}
        for routing in ms.ROUTINGS:
            self.assertEqual([status[f'{routing}/{t}'] for t in ['short', 'eos', 'arithmetic']], ['pass'] * 3)
            self.assertEqual([status[f'{routing}/{t}'] for t in ['chinese', 'english', 'long']], ['pending'] * 3)
        data = results['actual/chinese'][1]
        self.assertEqual((data['answer'], data['stop_reason'], data['length_stop'], data['first_token_seconds']), ('plain answer', 'length', True, 1.5))
        command = next(c for c in calls if '--user' in c)
        for flag, value in [('--thinking', 'off'), ('--temperature', '0'), ('--presence-penalty', '0'), ('--device', 'metal')]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn('--out', command)
        self.assertEqual({c[c.index('--model-option') + 1] for c in calls if '--model-option' in c}, {'routing=actual', 'routing=predicted'})
        identity = json.loads((self.out / 'identity.json').read_text())
        self.assertEqual(identity['engine']['source_commit'], 'c' * 40)
        self.assertEqual(identity['assets']['assets'][0]['role'], 'base')

    def test_refuses_existing_results_directory(self):
        self.out.mkdir()
        with self.assertRaises(ms.InputError):
            self.run_all()

    def test_automatic_counterexamples(self):
        self.run_all(answers={'eos': GOOD_OF, 'arithmetic': '42, no it is 43'})
        _, results = ms.load_results(self.out)
        self.assertEqual(results['actual/eos'][1]['status'], 'fail')
        self.assertEqual(results['predicted/arithmetic'][1]['status'], 'fail')

    def test_missing_cli_asset_or_device_is_not_run(self):
        self.cli.unlink()
        ms.run_all(self.cli, self.model, self.out, run=fake_executor())
        _, results = ms.load_results(self.out)
        self.assertEqual({data['status'] for _, data in results.values()}, {'not_run'})
        self.assertIn('CLI', next(iter(results.values()))[1]['reason'])
        self.cli.write_text('')
        other = self.root / 'second'
        (self.model / 'assets.tsv').unlink()
        ms.run_all(self.cli, self.model, other, run=fake_executor())
        self.assertEqual({d['status'] for _, d in ms.load_results(other)[1].values()}, {'not_run'})

    def test_device_unavailable_is_not_run(self):
        self.run_all(run_code=77)
        _, results = ms.load_results(self.out)
        self.assertEqual({d['status'] for _, d in results.values()}, {'not_run'})

    def test_review_accepts_only_pending_items(self):
        self.run_all()
        with self.assertRaises(ms.InputError) as raised:
            ms.review(self.out, self.verdicts({'actual/arithmetic': dict(verdict='pass', reason='fine', reviewer='checker')}))
        self.assertIn('decided automatically', str(raised.exception))
        _, results = ms.load_results(self.out)
        self.assertEqual(results['actual/arithmetic'][1]['status'], 'pass')
        self.assertEqual(results['actual/arithmetic'][1]['reviews'], [])

    def test_review_rejects_everything_if_one_entry_is_invalid(self):
        self.run_all()
        good = dict(verdict='pass', reason='explains scattering', reviewer='checker')
        for bad in [dict(good, reason='  '), dict(good, verdict='maybe'), dict(good, reviewer=''), {'verdict': 'pass'}, dict(good, extra=1)]:
            with self.assertRaises(ms.InputError):
                ms.review(self.out, self.verdicts({'actual/chinese': good, 'actual/english': bad}))
        with self.assertRaises(ms.InputError):
            ms.review(self.out, self.verdicts({'actual/unknown': good}))
        _, results = ms.load_results(self.out)
        self.assertEqual(results['actual/chinese'][1]['status'], 'pending')

    def test_review_and_user_override_are_recorded_in_order(self):
        self.run_all()
        ms.review(self.out, self.verdicts({'actual/chinese': dict(verdict='pass', reason='correct scattering', reviewer='checker')}))
        ms.review(self.out, self.verdicts({'actual/chinese': dict(verdict='fail', reason='misses the main point', reviewer='user')}))
        data = ms.load_results(self.out)[1]['actual/chinese'][1]
        self.assertEqual((data['status'], [r['reviewer'] for r in data['reviews']]), ('fail', ['checker', 'user']))

    def test_render_exit_codes_and_counts(self):
        self.run_all()
        text, code = ms.render(self.out)
        self.assertEqual(code, 1)
        self.assertIn('| actual | 3 | 0 | 3 | 0 | 6 |', text)
        self.assertIn('Pending review items are not counted as passed.', text)
        every = {f'{r}/{t}': dict(verdict='pass', reason='ok', reviewer='checker') for r in ms.ROUTINGS for t in ['chinese', 'english', 'long']}
        ms.review(self.out, self.verdicts(every))
        text, code = ms.render(self.out)
        self.assertEqual(code, 0)
        self.assertIn('| actual | 6 | 0 | 0 | 0 | 6 |', text)

    def test_render_flags_mismatching_performance_report(self):
        self.run_all()
        perf = dict(identity=dict(engine=dict(source_commit='d' * 40), model=dict(assets=[])))
        text, _ = ms.render(self.out, [perf])
        self.assertIn('NOT CONSISTENT', text)
        same = dict(identity=dict(engine=dict(source_commit='c' * 40),
                                  model=dict(assets=[dict(role='base', filename='base.gguf', bytes=1, sha256='0' * 64)])))
        text, _ = ms.render(self.out, [same])
        self.assertIn('consistent with this effect run', text)
        self.assertNotIn('NOT CONSISTENT', text)

    def test_command_line_exit_codes(self):
        self.run_all()
        self.assertEqual(ms.main(['render', '--results', str(self.out), '--out', str(self.root / 'o.md')]), 1)
        self.assertEqual(ms.main(['review', '--results', str(self.out), '--verdicts', str(self.verdicts({'actual/short': dict(verdict='pass', reason='x', reviewer='y')}))]), 2)
        self.assertEqual(ms.main(['render', '--results', str(self.root / 'none')]), 2)


if __name__ == '__main__':
    unittest.main()
