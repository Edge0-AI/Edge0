"""#528: the restore-speed bench's own logic (no GPU): flag overrides and the pass/fail rule."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conversation_restore_speed as R  # noqa: E402


def turn(name, ids, ms, reused=1):
    return {'name': name, 'ids': ids, 'generated': len(ids), 'decode_ms': ms, 'reused': reused}


class Bench(unittest.TestCase):
    def test_override(self):
        args = ['--max-context', '32768', '--kv', 'int8']
        self.assertEqual(R.override(args, R.parse_sets(['--max-context=65536', '--kv-resident=16384', '--kv='])),
                         ['--max-context', '65536', '--kv-resident', '16384'])

    def test_compare(self):
        base = [turn('A1', [1, 2], 100, 0), turn('A2', [3, 4], 100)]
        ok = {'baseline': base, 'candidate': [turn('A1', [1, 2], 100, 0), turn('B2', [9], 10),
                                              turn('A2', [3, 4], 110)]}
        self.assertEqual(R.compare(ok, 0.15), [])
        slow = {'baseline': base, 'candidate': [turn('A1', [1, 2], 100, 0), turn('A2', [3, 4], 400)]}
        self.assertTrue(any('restored decode' in p for p in R.compare(slow, 0.15)))
        differ = {'baseline': base, 'candidate': [turn('A1', [1, 2], 100, 0), turn('A2', [3, 5], 100)]}
        self.assertTrue(any('output differs' in p for p in R.compare(differ, 0.15)))
        unrestored = {'baseline': base, 'candidate': [turn('A1', [1, 2], 100, 0), turn('A2', [3, 4], 100, 0)]}
        self.assertTrue(any('not restored' in p for p in R.compare(unrestored, 0.15)))


if __name__ == '__main__':
    unittest.main()
