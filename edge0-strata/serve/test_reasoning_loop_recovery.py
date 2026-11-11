"""CPU-only regression for opt-in reasoning recovery: python -m unittest serve.test_reasoning_loop_recovery."""
import threading
import unittest
from serve.server import (Service, ByteTokenizer, MockEngine, HIGH_EFFORT,
                          reasoning_repeat_coverage, focused_recovery_prompt)


class Tokenizer(ByteTokenizer):
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer, scripts):
        super().__init__(tokenizer, scripts, max_context=100000)
        self.calls = []
        self.sampling_calls = []
        self.sampling_objects = []

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls.append((list(ids), max_new))
        self.sampling_calls.append(dict(sampling or {}))
        self.sampling_objects.append(sampling)
        yield from super().generate(ids, max_new, sampling, cancel, **kwargs)


class ReasoningLoopRecovery(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.ids = self.tok.encode('<|im_start|>system\n' + HIGH_EFFORT +
            '<|im_end|>\n<|im_start|>user\nSolve this unchanged task<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n', parse_special=True)
        self.loop = 'Need maybe Agent support agent seven b. ' * 700

    def test_repetition_and_novel_work(self):
        self.assertEqual(reasoning_repeat_coverage(' '.join(str(i) for i in range(4000))), 0)
        self.assertGreater(reasoning_repeat_coverage(self.loop), .9)

    def test_a_loop_written_as_one_long_word(self):
        """#1753: a 24k-digit string cycling an 11-digit pattern is one word to the split and never one token twice."""
        self.assertEqual(reasoning_repeat_coverage('Counting the digits of 9^104. ' + '58747693387' * 3000), 1)
        counted = 'Digits: ' + ''.join(str(i) for i in range(1000))     # 2,893 chars that never repeat
        self.assertEqual(reasoning_repeat_coverage(counted), 0)
        self.assertEqual(reasoning_repeat_coverage('58747693387' * 100), 0)   # a tail too short to judge

    def test_user_instruction_is_never_rewritten(self):
        ids = self.tok.encode('<|im_start|>user\n' + HIGH_EFFORT, parse_special=True)
        self.assertIsNone(focused_recovery_prompt(self.tok, ids, []))

    def test_preserves_task_and_exact_generated_tokens(self):
        generated = self.tok.encode('中文 partially checked code', parse_special=False)
        prompt = focused_recovery_prompt(self.tok, self.ids, generated)
        self.assertEqual(prompt[-len(generated):], generated)
        self.assertIn('Solve this unchanged task', self.tok.decode(prompt))
        self.assertNotIn('</think>', self.tok.decode(prompt))

    def run_case(self, enabled):
        engine = RecordingEngine(self.tok, [self.loop,
            'I have verified the solution.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = enabled
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        return engine, events

    def test_old_explicit_sampler_recovers_without_mutating_request(self):
        engine = RecordingEngine(self.tok, [self.loop,
            'Verified.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        request = {'temperature': .2, 'presence_penalty': 0, 'top_p': .95, 'top_k': 64, 'seed': 2}
        list(service.run(self.ids, True, [], 40000, request, threading.Event()))
        self.assertEqual(engine.sampling_calls[0], request)
        self.assertEqual(engine.sampling_calls[1]['temperature'], 1.0)
        self.assertEqual(engine.sampling_calls[1]['presence_penalty'], 1.5)
        self.assertEqual(engine.sampling_calls[1]['top_k'], 64)      # a client's top_k and top_p are never touched
        self.assertEqual(engine.sampling_calls[1]['top_p'], .95)
        self.assertEqual(engine.sampling_calls[1]['seed'], request['seed'])
        self.assertEqual(request['temperature'], .2)
        self.assertEqual(request['presence_penalty'], 0)

    def test_frozen_cohort_sampler_is_unchanged_during_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Done.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        sampler = {'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5, 'seed': 2}
        list(service.run(self.ids, True, [], 40000, sampler, threading.Event()))
        self.assertEqual(len(engine.sampling_objects), 2)
        self.assertIs(engine.sampling_objects[0], engine.sampling_objects[1])
        self.assertIs(engine.sampling_objects[1], sampler)

    def test_a_lower_top_p_and_top_k_are_never_raised_or_lowered(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Verified.\n</think>\nComplete solution.'])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = "recover"
        list(service.run(self.ids, True, [], 40000, {'temperature': 0, 'top_p': .5, 'top_k': 5}, threading.Event()))
        self.assertEqual((engine.sampling_calls[1]['top_p'], engine.sampling_calls[1]['top_k']), (.5, 5))
        self.assertEqual((engine.sampling_calls[1]['temperature'], engine.sampling_calls[1]['presence_penalty']), (1.0, 1.5))

    def test_the_splice_keeps_a_literal_think_tag_as_text(self):
        """#537: the prompt is spliced as token ids, never decoded and encoded again."""
        from serve.frontend import THINK_TAGS
        head = '<|im_start|>system\n' + HIGH_EFFORT
        marked = self.tok.encode(head + '<|im_end|>\n<|im_start|>user\nquote ' + THINK_TAGS['</think>'] +
                                 ' please<|im_end|>\n<|im_start|>assistant\n<think>\n', parse_special=True)
        spliced = focused_recovery_prompt(self.tok, marked, [])
        tail = marked[len(self.tok.encode(head, parse_special=True)):]
        self.assertEqual(spliced[-len(tail):], tail)             # everything after the sentence is the same ids
        self.assertNotEqual(spliced, marked)

    def test_stop_mode_ends_the_reply_at_the_loop(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = "stop"
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 20000)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_stop_mode_ends_a_loop_written_as_one_long_word(self):
        """#1753: the word count never sees the digits, so only the tail's period ends the reply."""
        engine = RecordingEngine(self.tok, ['9^104 = ' + '58747693387' * 3000])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = "stop"
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 5000)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_disabled_is_one_pass(self):
        engine, events = self.run_case(False)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_default_single_token_guard_does_not_detect_passage_loop(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        events = list(service.run(self.ids, True, [], 16384, {}, threading.Event()))
        self.assertEqual(service.repeat_stop_tokens, 256)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)
        self.assertFalse(any(e.kind == 'content' for kind, e in events if kind == 'event'))

    def test_recovery_is_once_and_never_extends_output_limit(self):
        engine = RecordingEngine(self.tok, [self.loop, self.loop])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, True, [], 16384, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)

    def test_repeated_final_text_does_not_change_sampler_or_prompt(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, False, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_missing_native_high_hint_skips_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        ids = self.tok.encode('<|im_start|>user\nTask<|im_end|>\n', parse_special=True)
        events = list(service.run(ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_cancel_during_reasoning_does_not_start_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Unused'])
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        cancel = threading.Event()
        events = []
        for event in service.run(self.ids, True, [], 40000, {}, cancel):
            events.append(event)
            if event[0] == 'event':
                cancel.set()
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'cancel')
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 0)

    def test_batch_status_and_history_record_recovery(self):
        engine = RecordingEngine(self.tok, [self.loop, 'Done.\n</think>\nAnswer.'])
        engine.batch = 2
        service = Service(engine, self.tok, None)
        service.answer_room = False      # these cases test the loop guards at the cap itself
        service.reasoning_loop_recovery = True
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(service.history[-1]['reasoning_recoveries'], 1)
        self.assertGreaterEqual(service.history[-1]['reasoning_repeat_coverage'], .25)
        self.assertFalse(service.status['busy'])
        self.assertEqual(service.live_reqs, {})

    def test_natural_answer_with_shared_budget_and_preserved_prefix(self):
        engine, events = self.run_case(True)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(events[-1][1]['reasoning_recoveries'], 1)
        self.assertEqual(events[-1][1]['finish'], 'stop')
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        self.assertEqual(content, 'Complete solution.')
        head_size = len(focused_recovery_prompt(self.tok, self.ids, []))
        retained = engine.calls[1][0][head_size:]
        self.assertEqual(retained, self.tok.encode(self.loop)[:len(retained)])
        self.assertEqual(engine.calls[1][1], 40000 - len(retained))
        self.assertNotIn('</think>', self.tok.decode(engine.calls[1][0]))


if __name__ == '__main__':
    unittest.main()
