"""Evidence readers wait for completed summaries and diagnose missing requests."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration'))
from serve_support import Server


class Process:
    def poll(self):
        return None


class SyncTest(unittest.TestCase):
    def test_expected_count_and_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            server = Server.__new__(Server)
            server.out = Path(temporary)
            server.lines = []
            server.changed = threading.Condition()
            server.process = Process()
            with self.assertRaises(AssertionError):
                server.evidence(1, timeout=0)
            def complete():
                folder = server.out/'1'; folder.mkdir()
                (folder/'tokens.json').write_text('{"input":[1]}')
                (folder/'result.json').write_text('{"stop_reason":"length"}')
                with server.changed:
                    server.lines.append('{"stop_reason":"length"}')
                    server.changed.notify_all()
            worker = threading.Thread(target=complete); worker.start()
            self.assertEqual(server.evidence(1)[0][0], {'input': [1]})
            worker.join()


if __name__ == '__main__':
    unittest.main()
