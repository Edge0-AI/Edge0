import os, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from serve import cache_drop


class CacheDrop(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(cache_drop.enabled({}))
        self.assertFalse(cache_drop.enabled({"STRATA_DROP_CACHE_ON_EXIT": "0"}))

    @unittest.skipIf(os.name == "nt", "POSIX only")
    def test_on(self):
        self.assertTrue(cache_drop.enabled({"STRATA_DROP_CACHE_ON_EXIT": "1"}))

    def test_files_and_advise(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.gguf").write_bytes(b"x")
            os.mkdir(os.path.join(d, "sub"))
            Path(d, "sub", "b.bin").write_bytes(b"y")
            self.assertEqual(len(cache_drop.model_files([d, "/no/such", None])), 2)
            seen = []
            self.assertEqual(cache_drop.drop([d], advise=lambda fd, o, n, how: seen.append(how)), 2)
            self.assertEqual(len(seen), 2)

    def test_not_while_engine_runs_or_not_opted_in(self):
        with tempfile.TemporaryDirectory() as d:
            env = {"STRATA_DROP_CACHE_ON_EXIT": "1"}
            calls = []
            run = lambda *a: calls.append(1)
            alive = SimpleNamespace(proc=SimpleNamespace(poll=lambda: None), spawn=("e", ["--native", d]), model_path=d)
            self.assertEqual(cache_drop.drop_after_exit(alive, env, run), 0)
            gone = SimpleNamespace(proc=SimpleNamespace(poll=lambda: 0), spawn=("e", ["--native", d]), model_path=d)
            self.assertEqual(cache_drop.drop_after_exit(gone, {}, run), 0)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
