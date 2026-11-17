"""Token-file selection through real run, bench, and forward commands."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = sys.argv[1]
MODEL = Path(os.environ.get('MOE_TINY', '/nonexistent')) / 'tiny-a'
if not (MODEL / 'assets.tsv').exists():
    raise SystemExit(77)


class TokenFiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def command(self, mode, source, output):
        args = [CLI, mode, '--model', str(MODEL), '--device', 'cpu',
                '--threads', '1', '--memory-budget', '536870912',
                '--tokens', str(source)]
        if mode == 'run':
            args += ['--temperature', '0', '--max-tokens', '1', '--out', str(output)]
        elif mode == 'forward':
            args += ['--out', str(output)]
        else:
            args += ['--warmup', '0', '--ntok', '1', '--runs', '1']
        return subprocess.run(args, capture_output=True, text=True, timeout=30)

    def test_root_fields(self):
        documents = {
            'plain': '{"input":[1,2],"generated":[3]}',
            'nested': '{"metadata":{"input":[4,5,6],"generated":[7,8]},'
                      '"input":[1,2],"generated":[3]}',
            'escaped': r'{"in\u0070ut":[1,2],"gener\u0061ted":[3]}',
            'metadata': '{"score":0.125,"note":"input generated",'
                        '"generated":[3],"input":[1,2]}',
        }
        for name, document in documents.items():
            source = self.root / (name + '.json')
            source.write_text(document, encoding='utf-8')
            for mode in ['run', 'bench', 'forward']:
                with self.subTest(document=name, mode=mode):
                    output = self.root / (name + '-' + mode)
                    result = self.command(mode, source, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if mode == 'bench':
                        rows = [json.loads(line) for line in result.stdout.splitlines()]
                        self.assertTrue(rows)
                        self.assertTrue(all(row['input_count'] == 2 for row in rows))
                    else:
                        tokens = json.loads((output / 'tokens.json').read_text())
                        self.assertEqual(tokens['input'], [1, 2])
                        if mode == 'forward':
                            self.assertEqual(tokens['generated'], [3])

    def test_invalid_documents(self):
        documents = [
            '{"input":[1,2],"generated":[]',
            '{"input":[1,2],"generated":[]} trailing',
            '{"metadata":{"input":[1,2]},"generated":[]}',
            '[{"input":[1,2],"generated":[]}]',
            '{"input":[-1],"generated":[]}',
            '{"input":[2147483648],"generated":[]}',
            '{"input":[18446744073709551615],"generated":[]}',
            '{"input":[1.0],"generated":[]}',
            '{"input":[true],"generated":[]}',
            '{"input":null,"generated":[]}',
        ]
        for index, document in enumerate(documents):
            with self.subTest(document=document):
                source = self.root / (str(index) + '.json')
                source.write_text(document, encoding='utf-8')
                output = self.root / (str(index) + '-out')
                result = self.command('forward', source, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
        # The generation and benchmark entry points use the same token reader.
        source = self.root / 'trailing.json'
        source.write_text(documents[1], encoding='utf-8')
        for mode in ['run', 'bench']:
            with self.subTest(mode=mode):
                result = self.command(mode, source, self.root / mode)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
