"""Exercise the example App design-token check on synthetic trees."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('check_app_tokens', ROOT / 'tools/checks/check_app_tokens.py')
check_app_tokens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_app_tokens)

DESKTOP = 'examples/desktop/src/'
ANDROID = 'examples/android/app/src/main/kotlin/dev/edge0/chat/'
IOS = 'examples/ios/App/'


class AppTokensTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, text):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)

    def run_check(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = check_app_tokens.check(self.root)
        return status, output.getvalue()

    def assert_reports(self, path, text, rule, line=2):
        self.write(path, 'clean\n' + text + '\n')
        status, output = self.run_check()
        self.assertEqual(status, 1, output)
        self.assertIn(f'{path}:{line}: {rule}:', output)
        (self.root / path).unlink()

    def test_positive_rules(self):
        cases = [
            (DESKTOP + 'a.css', '.x { color: #fff; }', 'hex-color'),
            (DESKTOP + 'a.tsx', "const c = '#1b1c1e';", 'hex-color'),
            (DESKTOP + 'a.ts', "const c = '#2f6feb80';", 'hex-color'),
            (DESKTOP + 'a.css', '.x { color: rgb(1, 2, 3); }', 'css-color-function'),
            (DESKTOP + 'a.tsx', "style={{ background: 'rgba(0,0,0,.3)' }}", 'css-color-function'),
            (DESKTOP + 'a.css', '.x { color: hsl(10 20% 30%); }', 'css-color-function'),
            (ANDROID + 'Page.kt', 'val c = Color(0xFF2F6FEB)', 'compose-color-literal'),
            (ANDROID + 'ui/Card.kt', 'val c = Color( 0xFF000000 )', 'compose-color-literal'),
            (IOS + 'Page.swift', 'let c = Color(red: 0.1, green: 0.2, blue: 0.3)', 'swift-color-components'),
            (IOS + 'Views/Card.swift', 'let c = UIColor(red: 1, green: 1, blue: 1, alpha: 1)', 'swift-color-components'),
            (IOS + 'Page.swift', 'let hex = "#FFFFFF"', 'hex-color'),
        ]
        for path, text, rule in cases:
            with self.subTest(text=text):
                self.assert_reports(path, text, rule)

    def test_negative_forms(self):
        self.write(DESKTOP + 'a.css', '.x { color: var(--text); background: transparent; fill: currentColor; }\n'
                   '#root { width: 100%; }\n')
        self.write(DESKTOP + 'a.tsx', '<use href="#i-chat"/>; const id = "#12345"; const e = "&#123;";\n'
                   'const grid = "rgbish"; // background(rgbValue) is not a call to rgb\n')
        self.write(ANDROID + 'Page.kt', 'val c = MaterialTheme.colorScheme.primary\nval t = Color.Transparent\n')
        self.write(IOS + 'Page.swift', 'let c = Color.primary\n#if os(iOS)\nlet d = Tokens.accent\n#endif\n')
        status, output = self.run_check()
        self.assertEqual(status, 0, output)
        self.assertIn('app-tokens: passed', output)

    def test_token_and_test_files_are_skipped(self):
        self.write(DESKTOP + 'theme.css', ':root { --bg: #f4f4f2; --shadow: rgba(0,0,0,.3); }\n')
        self.write(ANDROID + 'ui/theme/Tokens.kt', 'val Bg = Color(0xFFF4F4F2)\n')
        self.write(IOS + 'Tokens.swift', 'static let bg = Color(red: 0.96, green: 0.96, blue: 0.95)\n')
        self.write(DESKTOP + 'format.test.ts', "expect(x).toBe('#fff')\n")
        self.write(DESKTOP + 'Workspace.test.tsx', "expect(x).toBe('rgb(1,2,3)')\n")
        status, output = self.run_check()
        self.assertEqual(status, 0, output)

    def test_unscanned_locations_are_ignored(self):
        self.write('examples/desktop/src-tauri/src/lib.rs', '// #fff\n')
        self.write('examples/desktop/index.html', '<body style="background:#fff">\n')
        self.write('examples/android/core/src/main/kotlin/Core.kt', 'val c = Color(0xFF000000)\n')
        self.write('examples/android/app/src/test/kotlin/A.kt', 'val c = Color(0xFF000000)\n')
        self.write('examples/ios/ChatCore/Sources/ChatCore/Core.swift', 'let c = Color(red: 1, green: 1, blue: 1)\n')
        self.write(DESKTOP + 'node_modules/pkg/index.ts', "const c = '#fff'\n")
        self.write(DESKTOP + 'notes.md', '#fff\n')
        status, output = self.run_check()
        self.assertEqual(status, 0, output)

    def test_reports_every_occurrence_with_line(self):
        self.write(DESKTOP + 'a.css', '.a { color: #fff; }\n.b { color: #000; border-color: rgb(1,1,1); }\n')
        status, output = self.run_check()
        self.assertEqual(status, 1)
        self.assertEqual(output.count('examples/desktop/src/a.css:2:'), 2, output)
        self.assertEqual(output.count('examples/desktop/src/a.css:1:'), 1, output)

    def test_missing_app_directories_pass(self):
        status, output = self.run_check()
        self.assertEqual(status, 0, output)


if __name__ == '__main__':
    unittest.main()
