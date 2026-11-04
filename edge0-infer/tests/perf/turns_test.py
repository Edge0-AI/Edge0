"""Conversation measurement: process checks, summary statistics and report kinds, with a fake host."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hosts
import report as reports
spec = importlib.util.spec_from_file_location('turns', HERE / 'turns.py')
turns = importlib.util.module_from_spec(spec)
sys.modules['turns'] = turns
spec.loader.exec_module(turns)

ENGINE = dict(version='0.1.0', source_commit='b' * 40, source_modified='no', source_reason='', ggml_origin='o', ggml_version='v')
ASSETS = [dict(role='base', filename='base.gguf', bytes=1, sha256='0' * 64)]
MEMORY = {'ledger': {'peak_bytes': 1}, 'rss': {'values': {'peak_bytes': 2}}, 'physical': {'values': {'peak_bytes': 3}}}


def loading(routing='predicted', context=16384):
    return dict(directory='m', device='MTL0', context=context, prefill_chunk=4096, model_options={'routing': routing}, sources={})


def output(thinking=(False, False, False), first=(6.0, 0.7, 0.8), cold=(6.0, 6.3, 6.4), routing='predicted', count=3,
           methods=('first', 'keep', 'keep'), generated=7, context=16384):
    lines = []
    for t in range(count):
        lines.append(json.dumps(dict(
            type='turn', turn=t + 1, thinking=thinking[t], input_count=3000 + 40 * t, reuse_method=methods[t],
            reused_tokens=0 if t == 0 else 3000, computed_tokens=3000 if t == 0 else 45, checkpoint_restored=False,
            checkpoint_boundary=None, first_token_seconds=first[t], generated_count=5, stop_reason='length',
            input=[1, 2, t], generated=[generated, t], cold=dict(reuse_method='first', reused_tokens=0, computed_tokens=3000 + 40 * t,
                                                                 first_token_seconds=cold[t], stop_reason='length'))))
    lines.append(json.dumps(dict(type='summary', turns=count, engine=ENGINE, memory_snapshot=MEMORY,
                                 model=dict(adapter='edge0_35b', assets=ASSETS, assets_verified=False, loading=loading(routing, context)))))
    return '\n'.join(lines) + '\n'


class FakeHost:
    name = 'fake'

    def __init__(self, outputs, on_run=None):
        self.outputs, self.calls, self.on_run = outputs, 0, on_run

    busy_from = None

    def preflight(self):
        pass

    def check_background_load(self):
        if self.busy_from is not None and self.calls >= self.busy_from:
            raise hosts.BackgroundLoadError('background load: Some Helper (30%) above 15% CPU')

    def device(self):
        return dict(platform='fake', model='M', chip='C', memory_bytes=1, os_version='1')

    def snapshot(self, model):
        return dict(platform='fake', time=0)

    def run(self, argv):
        if self.on_run:
            self.on_run(self.calls)
        item = self.outputs(self.calls, argv) if callable(self.outputs) else self.outputs[self.calls]
        self.calls += 1
        return item if isinstance(item, hosts.CommandResult) else hosts.CommandResult(0, item, '')


class TurnsTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-turns-')
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.binary = self.root / 'bin'
        self.binary.write_bytes(b'x')
        self.plain = self.root / 'plain.json'
        self.plain.write_text(json.dumps({'turns': [{'user': 'a'}, {'user': 'b'}, {'user': 'c'}]}))
        self.thinking = self.root / 'thinking.json'
        self.thinking.write_text(json.dumps({'turns': [{'user': 'a', 'thinking': True}, {'user': 'b', 'thinking': True}, {'user': 'c', 'thinking': True}]}))
        self.combos = {
            'predicted-off': dict(command=[str(self.binary), 'bench', 'p'], routing='predicted', info=turns.conversation_info(self.plain)),
            'predicted-on': dict(command=[str(self.binary), 'bench', 't'], routing='predicted', info=turns.conversation_info(self.thinking)),
            'actual-off': dict(command=[str(self.binary), 'bench', 'a'], routing='actual', info=turns.conversation_info(self.plain))}

    def protocol(self, runs=2):
        return dict(kind='turns', backend='metal', context=16384, runs_per_combination=runs, cooldown_seconds=30, order=list(self.combos))

    def run_with(self, host, runs=2, sleeps=None):
        writer = reports.ReportWriter(self.root / 'out' / 'report')
        self.writer = writer
        return turns.run_turns(host, self.combos, self.protocol(runs), writer, model=Path('m'), raw_dir=self.root / 'raw',
                               tools=[self.binary], root=self.root, sleep=(sleeps.append if sleeps is not None else lambda s: None))

    def correct(self, call, argv):
        name = {'p': 'predicted-off', 't': 'predicted-on', 'a': 'actual-off'}[argv[2]]
        return output(thinking=(True,) * 3 if name == 'predicted-on' else (False,) * 3,
                      routing='actual' if name == 'actual-off' else 'predicted')

    def test_round_robin_and_report(self):
        sleeps = []
        host = FakeHost(self.correct)
        report = self.run_with(host, runs=2, sleeps=sleeps)
        self.assertEqual([c['side'] for c in report['cells']], ['predicted-off', 'predicted-on', 'actual-off'] * 2)
        self.assertEqual(sum(sleeps), 5 * 30)
        self.assertEqual(report['protocol']['kind'], 'turns')
        self.assertEqual(reports.validate(json.loads(self.writer.json_path.read_text())), [])
        self.assertEqual(report['cells'][0]['metrics']['turn2_first_token_seconds'], 0.7)
        self.assertEqual(report['sides']['actual-off']['configuration']['model_options']['routing'], 'actual')

    def test_summary_median_and_range(self):
        firsts = [(6.0, 0.5, 0.9), (6.0, 0.7, 0.8), (6.0, 0.9, 0.7)]
        host = FakeHost(lambda call, argv: output(first=firsts[call]))
        combos = {'predicted-off': self.combos['predicted-off']}
        self.combos = combos
        report = self.run_with(host, runs=3)
        turn2 = report['summary']['predicted-off'][1]
        self.assertEqual((turn2['first_token_seconds']['median'], turn2['first_token_seconds']['minimum'], turn2['first_token_seconds']['maximum']), (0.7, 0.5, 0.9))
        self.assertEqual(turn2['reuse_methods'], {'keep': 3})
        self.assertEqual(turn2['cold_first_token_seconds']['median'], 6.3)
        self.assertTrue(turn2['generated_identical'])
        self.assertEqual(report['summary']['predicted-off'][0]['reuse_methods'], {'first': 3})

    def test_differing_generated_tokens_are_reported_not_hidden(self):
        host = FakeHost(lambda call, argv: output(generated=7 + call))
        self.combos = {'predicted-off': self.combos['predicted-off']}
        report = self.run_with(host, runs=2)
        self.assertFalse(report['summary']['predicted-off'][0]['generated_identical'])

    def stopped_with(self, host, expected_reason, cells_before):
        with self.assertRaises(turns.MeasurementStopped):
            self.run_with(host)
        partial = reports.load_partial(self.writer.partial_path)
        self.assertEqual(len(partial['cells']), cells_before + 1)
        self.assertEqual(partial['cells'][-1]['status'], 'failed')
        self.assertIn(expected_reason, partial['cells'][-1]['reason'])
        self.assertFalse(self.writer.json_path.exists())

    def test_background_load_fails_the_cell_and_stops(self):
        host = FakeHost(self.correct)
        host.busy_from = 2
        self.stopped_with(host, 'background load', 2)

    def test_turn_count_mismatch_fails(self):
        self.stopped_with(FakeHost(lambda call, argv: self.correct(call, argv) if call == 0 else output(count=2, methods=('first', 'keep'),
                                                                                       thinking=(False, False))), 'turn count', 1)

    def test_conversation_digest_mismatch_fails(self):
        def alter(call):
            if call == 0:
                self.plain.write_text(json.dumps({'turns': [{'user': 'changed'}]}))
        self.stopped_with(FakeHost(self.correct, on_run=alter), 'conversation file changed', 2)

    def test_thinking_flag_mismatch_fails(self):
        self.stopped_with(FakeHost(lambda call, argv: output(thinking=(False, False, False)) if argv[2] == 't' else self.correct(call, argv)),
                          'thinking', 1)

    def test_configuration_mismatch_fails(self):
        self.stopped_with(FakeHost(lambda call, argv: output(routing='actual')), 'routing', 0)
        self.setUp()
        self.stopped_with(FakeHost(lambda call, argv: output(context=4096)), 'context', 0)

    def test_nonzero_exit_fails_and_keeps_earlier_cells(self):
        self.stopped_with(FakeHost(lambda call, argv: self.correct(call, argv) if call < 2 else hosts.CommandResult(1, '', 'boom')),
                          'exit code 1', 2)

    def test_changed_engine_identity_fails(self):
        def outputs(call, argv):
            text = self.correct(call, argv)
            return text.replace('"source_modified": "no"', '"source_modified": "yes"') if call == 1 else text
        self.stopped_with(FakeHost(outputs), 'identity changed', 1)

    def test_compare_refuses_other_kinds(self):
        report = self.run_with(FakeHost(self.correct), runs=1)
        abba_like = json.loads(json.dumps(report))
        abba_like['protocol']['kind'] = 'abba'
        self.assertEqual(reports.compare(report, abba_like)[0], 2)
        self.assertEqual(reports.compare(report, report)[0], 2)

    def test_rendering_turns_report(self):
        report = self.run_with(FakeHost(self.correct), runs=1)
        text = Path(self.writer.md_path).read_text()
        self.assertIn('Turns summary', text)
        self.assertIn('predicted-on', text)
        self.assertEqual(reports.render(report), text)


if __name__ == '__main__':
    unittest.main()
