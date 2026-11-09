"""Fixed evidence files exercise count derivation without executing a model."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_turn_test import derive_counts


class CountsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def evidence(self, index, tokens, result):
        folder = self.root / str(index)
        folder.mkdir()
        (folder / 'tokens.json').write_text(tokens)
        (folder / 'result.json').write_text(result)

    def first(self, early=False):
        if early:
            self.evidence(1, '{"input":[1,2,3],"generated":[9]}',
                          '{"input_count":3,"generated_count":0,"e":3,"stop_reason":"end_of_generation","tail_is_end":true}')
        else:
            self.evidence(1, '{"input":[1,2,3],"generated":[4,5,6,7]}',
                          '{"input_count":3,"generated_count":4,"e":6,"stop_reason":"length","tail_is_end":false}')

    def test_normal_continuation(self):
        self.first()
        self.evidence(2, '{"input":[1,2,3,4,5,6,7,8,9],"generated":[1,2,3,4]}',
                      '{"input_count":9,"generated_count":4,"e":12,"stop_reason":"length","tail_is_end":false}')
        self.assertEqual(derive_counts(self.root, [('first',0,6,True),('keep',6,6,True)]),
                         [('first',0,6,True),('keep',6,6,True)])

    def test_early_end_changes_following_reuse(self):
        self.first(early=True)
        self.evidence(2, '{"input":[1,2,3,9,6,7],"generated":[8,9]}',
                      '{"input_count":6,"generated_count":1,"e":7,"stop_reason":"end_of_generation","tail_is_end":true}')
        self.evidence(3, '{"input":[1,2,3,9,6,7,8,9,8,9],"generated":[1,2,3,4]}',
                      '{"input_count":10,"generated_count":4,"e":13,"stop_reason":"length","tail_is_end":false}')
        self.assertEqual(derive_counts(self.root, [('first',0,6,True),('keep',6,6,True),('keep',12,6,True)]),
                         [('first',0,3,True),('keep',3,4,True),('keep',7,6,True)])

    def test_restore_uses_input_boundary_not_early_end(self):
        self.first(early=True)
        self.evidence(2, '{"input":[1,2,3],"generated":[9]}',
                      '{"input_count":3,"generated_count":0,"e":3,"stop_reason":"end_of_generation","tail_is_end":true}')
        self.assertEqual(derive_counts(self.root, [('first',0,6,True),('restore',2,4,False)]),
                         [('first',0,3,True),('restore',2,1,False)])

    def test_invalid_stop_or_accounting_is_rejected(self):
        self.first(early=True)
        path = self.root / '1/result.json'
        original = json.loads(path.read_text())
        for key, value in [('stop_reason','length'), ('tail_is_end',False), ('e',4),
                           ('input_count',4), ('generated_count',1)]:
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key:value}))
                with self.assertRaises(AssertionError):
                    derive_counts(self.root, [('first',0,6,True)])


if __name__ == '__main__':
    unittest.main()
