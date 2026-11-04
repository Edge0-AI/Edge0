"""The README translations may contain CJK text; every other public file may not, and the
translations keep every other public check."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('language', Path(__file__).resolve().parents[3] / 'tools/check_language.py')
language = importlib.util.module_from_spec(spec)
spec.loader.exec_module(language)

CJK_TEXT = ''.join(map(chr, (0x4e2d, 0x6587, 0x3002))) + '\n'


class LanguageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        errors = []
        language.check_public(self.root, rel, errors)
        return errors

    def test_translations_are_public(self):
        for rel in language.TRANSLATIONS:
            self.assertEqual(language.classify(rel), 'public')

    def test_translation_allows_cjk(self):
        for rel in language.TRANSLATIONS:
            self.assertEqual(self.check(rel, '# edge0-infer\n\n' + CJK_TEXT), [])

    def test_other_public_files_reject_cjk(self):
        for rel in ('README.md', 'docs/README_zh.md', 'README.public_zh.md', 'examples/README_ja.md'):
            errors = self.check(rel, CJK_TEXT)
            self.assertTrue(errors and errors[0].startswith('cjk '), (rel, errors))

    def test_translation_keeps_link_check(self):
        (self.root / 'CLAUDE.md').write_text('internal\n', encoding='utf-8')
        errors = self.check('README_zh.md', CJK_TEXT + '[x](CLAUDE.md)\n')
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith('link README_zh.md:2'), errors)

    def test_translation_keeps_utf8_check(self):
        path = self.root / 'README_ja.md'
        path.write_bytes(''.join(map(chr, (0x65e5, 0x672c, 0x8a9e))).encode('shift_jis'))
        errors = []
        language.check_public(self.root, 'README_ja.md', errors)
        self.assertEqual(errors, ['non-utf8 README_ja.md'])


if __name__ == '__main__':
    unittest.main()
