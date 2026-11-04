"""Hand-calculated prediction statistics, consumption, gaps and empty input."""
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('route_trace', Path(__file__).with_name('route_trace.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def entry(layer, actual, head=(), prepared=(), executed=None):
    return dict(layer=layer, actual=actual, executed=actual if executed is None else executed,
                head=head, prepared=prepared)


class TraceTest(unittest.TestCase):
    def test_hand_calculated(self):
        rows = [dict(position=10, top_k=2, layers=[entry(0, [0, 1], [1, 2, 3, 4]), entry(1, [], prepared=[1, 2])]),
                dict(position=11, top_k=2, layers=[entry(1, [1, 3], executed=[1, 2], prepared=[3, 4])])]
        result = module.summarize(rows)
        self.assertEqual(result['prepared_experts'], 4)
        self.assertEqual((result['consumed_matches'], result['consumed_experts']), (2, 2))
        head = result['predictions']['head_top2k']
        self.assertEqual((head['recall'], head['precision']), (1, .5))
        rows[1]['layers'][0]['executed'] = [2, 1]
        self.assertEqual(module.summarize(rows)['consumed_matches'], 0)
        rows[1]['position'] = 12
        self.assertIsNone(module.summarize(rows)['predictions']['head_top1k']['recall'])
        rows[1]['position'] = 11
        rows[1]['source_position'] = -1
        self.assertEqual(module.summarize(rows)['consumed_experts'], 0)

    def test_empty_and_malformed(self):
        self.assertEqual(module.summarize([])['prepared_experts'], 0)
        with self.assertRaises(ValueError):
            module.summarize([dict(position=0, top_k=0, layers=[])])
        with self.assertRaises(ValueError):
            module.summarize([dict(position=0, top_k=1, layers=[entry(0, []), entry(0, [])])])


if __name__ == '__main__':
    unittest.main()
