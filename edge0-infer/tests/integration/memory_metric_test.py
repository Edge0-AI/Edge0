"""Peak selection keeps the original physical-first criterion."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_metric import process_peak


class MemoryTest(unittest.TestCase):
    def test_priority_and_absence(self):
        s = dict(physical=dict(source='phys_footprint', values=dict(peak_bytes=10)),
                 rss=dict(source='proc_status', values=dict(peak_bytes=20)))
        self.assertEqual(process_peak(s), ('physical', 'phys_footprint', 10))
        s['physical']['values']['peak_bytes'] = None
        self.assertEqual(process_peak(s), ('rss', 'proc_status', 20))
        s['rss']['values'] = None
        with self.assertRaises(ValueError):
            process_peak(s)


if __name__ == '__main__':
    unittest.main()
