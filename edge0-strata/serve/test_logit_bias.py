import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from serve.logit_bias import engine_key, normalize, validate_request
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.frontend import ChatTemplate


class LogitBias(unittest.TestCase):
    def test_forms_and_transport(self):
        self.assertEqual(normalize({"7": -100, "9": 2.5}, 10), normalize([[7, False], [9, 2.5]], 10))
        self.assertEqual(engine_key([[9, 2.5], [7, False]]), " logit_bias=7:-100,9:2.5")
        self.assertEqual(StrataEngine.sampling_keys({"logit_bias": {"7": -100}}), " logit_bias=7:-100")
        for value in [None, {}, []]:
            self.assertEqual(engine_key(value), "")

    def test_invalid(self):
        bad = [True, 1, "x", [1], [[1]], [[1, 0, 2]], {"-1": 0}, {"1.0": 0}, {"10": 0},
               {"1": False}, [[1, True]], [[1.0, 0]], [[True, 0]], [[1, 0], ["1", 2]],
               {"1": 101}, {"1": -101}, {"1": float('nan')}, {"1": float('inf')}, {"1": 10**1000}]
        for value in bad:
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                normalize(value, 10)
        with self.assertRaises(ValueError):
            normalize({str(i): -100 for i in range(10)}, 10)

    def test_large_list(self):
        entries = [[i, False] for i in range(103215)]
        self.assertEqual(len(normalize(entries, 248320)), 103215)
        self.assertEqual(engine_key(entries).count(':-100'), 103215)

    def test_capability_and_batch(self):
        for engine in [SimpleNamespace(info={}), SimpleNamespace(info={'logit_bias': 1}, batch=2)]:
            with self.assertRaises(ValueError):
                validate_request({'logit_bias': {'1': -100}}, engine, 10)
            self.assertEqual(validate_request({}, engine, 10), {})
        self.assertEqual(validate_request({'logit_bias': {'1': -100}}, SimpleNamespace(info={'logit_bias': 1}), 10), {1: -100})

    def test_http_rejects_before_stream(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, 'ok', max_context=4096)
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd = serve(service, port=0)
        try:
            for bias in [True, {'1': -100}, [[1, True]]]:
                data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2,
                                   'stream': True, 'logit_bias': bias}).encode()
                req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                                             data=data, headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn('logit_bias', json.loads(response.read())['error']['message'])
        finally:
            httpd.shutdown()
            httpd.server_close()


class LogitBiasCache(unittest.TestCase):
    """A long list is normalized once; the cache must not change what a call returns or raises."""

    def setUp(self):
        from serve import logit_bias
        logit_bias._cache.clear()
        logit_bias._last = None

    def test_same_answers_as_the_plain_function(self):
        from serve.logit_bias import _normalize
        forms = [{str(i): -100 for i in range(200)},
                 [[i, False] for i in range(200)],
                 {str(i): (i % 7) - 3.5 for i in range(200)},
                 [[str(i), 2] for i in range(200)]]
        for vocab in (None, 1000):
            for form in forms:
                want = _normalize(form, vocab)
                for _ in range(3):                   # the first call fills the cache, the others read it
                    self.assertEqual(normalize(form, vocab), want)
                self.assertEqual(normalize(json.loads(json.dumps(form)), vocab), want)   # an equal list, another object
        for form in forms:
            ref = " logit_bias=" + ",".join(f"{i}:{b:g}" for i, b in sorted(_normalize(form).items()))
            self.assertEqual(engine_key(form), ref)
            self.assertEqual(engine_key(form), ref)

    def test_false_is_not_zero(self):
        ban = [[i, False] for i in range(100)]
        zero = [[i, 0] for i in range(100)]
        self.assertTrue(ban == zero)                  # Python calls them equal, so an == key would mix them up
        for first, second in [(ban, zero), (zero, ban)]:
            self.setUp()
            self.assertEqual(set(normalize(first).values()), {-100.0 if first is ban else 0.0})
            self.assertEqual(set(normalize(second).values()), {-100.0 if second is ban else 0.0})

    def test_a_bad_list_is_never_stored(self):
        good = {str(i): -1 for i in range(100)}
        normalize(good, 1000)
        for bad in [{**good, "5": True}, {**good, "5": 101}, {**good, "5": float("nan")}, {**good, "5": "x"}]:
            for _ in range(2):
                with self.assertRaises(ValueError):
                    normalize(bad, 1000)
        pairs = [[i, 1] for i in range(100)]
        normalize(pairs)
        with self.assertRaises(ValueError):           # True == 1, but a bool is not a bias
            normalize([[i, 1] for i in range(99)] + [[99, True]])
        with self.assertRaises(ValueError):           # a duplicate ID
            normalize([[i, 1] for i in range(100)] + [[0, 1]])

    def test_vocabulary_size_is_part_of_the_key(self):
        ids = {str(i): -1 for i in range(100)}
        self.assertEqual(len(normalize(ids, 1000)), 100)
        with self.assertRaises(ValueError):
            normalize(ids, 50)
        self.assertEqual(len(normalize(ids)), 100)
        with self.assertRaises(ValueError):
            normalize(ids, 50)

    def test_the_result_is_a_copy(self):
        ids = {str(i): -1 for i in range(100)}
        first = normalize(ids)
        first[0] = 99.0
        self.assertEqual(normalize(ids)[0], -1.0)
        self.assertEqual(validate_request({"logit_bias": ids}, SimpleNamespace(info={"logit_bias": 1}), 1000)[0], -1.0)

    def test_an_equal_list_in_a_later_request_is_a_hit(self):
        from serve import logit_bias
        a = {str(i): -100 for i in range(100)}
        b = json.loads(json.dumps(a))
        self.assertIsNot(a, b)
        key_a = engine_key(a)
        logit_bias._last = None                       # a later request: the previous object is gone
        key_b = engine_key(b)
        self.assertIs(key_a, key_b)                   # one entry, one key string
        self.assertEqual(len(logit_bias._cache), 1)

    def test_the_cache_is_bounded(self):
        from serve import logit_bias
        for n in range(logit_bias._CACHE_MAX + 3):
            normalize({str(i): -(n + 1) for i in range(100)})
        self.assertEqual(len(logit_bias._cache), logit_bias._CACHE_MAX)
