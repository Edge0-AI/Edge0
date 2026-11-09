"""Build identity generation: clean, modified, nested and non-Git source trees, and no rewrite when unchanged."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

CMAKE, SCRIPT = sys.argv.pop(1), Path(sys.argv.pop(1))
GIT_ENV = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
GIT_ENV.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
IDENTITY = ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false']


def git(directory, *args):
    result = subprocess.run(['git', *IDENTITY, '-C', str(directory), *args], capture_output=True, text=True, env=GIT_ENV)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def generate(source, output):
    result = subprocess.run([CMAKE, f'-DSOURCE_DIR={source}', f'-DOUTPUT={output}', '-P', str(SCRIPT)],
                            capture_output=True, text=True, env=GIT_ENV)
    assert result.returncode == 0, result.stderr
    text = Path(output).read_text()
    fields = dict(re.findall(r'#define MOE_BUILD_SOURCE_(COMMIT|MODIFIED|REASON) (.*)', text))
    return dict(commit=fields['COMMIT'].strip('"'), modified=int(fields['MODIFIED']), reason=fields['REASON'].strip('"'))


class BuildIdentityTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(prefix='moe-build-identity-')
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.output = self.root / 'out' / 'generated' / 'build_identity.h'

    def repository(self):
        directory = self.root / 'source'
        directory.mkdir()
        git(directory, 'init', '-q')
        (directory / 'a.txt').write_text('one\n')
        git(directory, 'add', 'a.txt')
        git(directory, 'commit', '-q', '-m', 'initial')
        return directory

    def test_clean_repository(self):
        directory = self.repository()
        identity = generate(directory, self.output)
        self.assertEqual(identity, dict(commit=git(directory, 'rev-parse', 'HEAD'), modified=0, reason=''))
        self.assertRegex(identity['commit'], r'^[0-9a-f]{40}$')

    def test_tracked_change(self):
        directory = self.repository()
        (directory / 'a.txt').write_text('two\n')
        identity = generate(directory, self.output)
        self.assertEqual((identity['commit'], identity['modified'], identity['reason']),
                         (git(directory, 'rev-parse', 'HEAD'), 1, ''))

    def test_untracked_file_counts_as_modified(self):
        directory = self.repository()
        (directory / 'new.txt').write_text('new\n')
        self.assertEqual(generate(directory, self.output)['modified'], 1)

    def test_not_a_repository(self):
        directory = self.root / 'plain'
        directory.mkdir()
        identity = generate(directory, self.output)
        self.assertEqual((identity['commit'], identity['modified']), ('', 2))
        self.assertNotEqual(identity['reason'], '')

    def test_untracked_subdirectory_does_not_take_the_enclosing_commit(self):
        directory = self.repository()
        (directory / 'sub').mkdir()
        (directory / 'sub' / 'b.txt').write_text('untracked\n')
        identity = generate(directory / 'sub', self.output)
        self.assertEqual((identity['commit'], identity['modified']), ('', 2))
        self.assertNotEqual(identity['reason'], '')

    def tracked_subdirectory(self):
        directory = self.repository()
        (directory / 'sub').mkdir()
        (directory / 'sub' / 'b.txt').write_text('one\n')
        git(directory, 'add', 'sub/b.txt')
        git(directory, 'commit', '-q', '-m', 'sub')
        return directory

    def test_tracked_subdirectory_reports_the_enclosing_commit(self):
        directory = self.tracked_subdirectory()
        identity = generate(directory / 'sub', self.output)
        self.assertEqual(identity, dict(commit=git(directory, 'rev-parse', 'HEAD'), modified=0, reason=''))

    def test_tracked_subdirectory_counts_only_its_own_changes(self):
        directory = self.tracked_subdirectory()
        (directory / 'a.txt').write_text('outside\n')
        (directory / 'outside.txt').write_text('outside\n')
        self.assertEqual(generate(directory / 'sub', self.output)['modified'], 0)
        (directory / 'sub' / 'b.txt').write_text('two\n')
        self.assertEqual(generate(directory / 'sub', self.output)['modified'], 1)
        (directory / 'sub' / 'b.txt').write_text('one\n')
        (directory / 'sub' / 'new.txt').write_text('new\n')
        self.assertEqual(generate(directory / 'sub', self.output)['modified'], 1)

    def test_unchanged_content_is_not_rewritten(self):
        directory = self.repository()
        generate(directory, self.output)
        old = 1_000_000_000
        os.utime(self.output, (old, old))
        generate(directory, self.output)
        self.assertEqual(self.output.stat().st_mtime, old)

    def test_changed_content_is_rewritten(self):
        directory = self.repository()
        self.assertEqual(generate(directory, self.output)['modified'], 0)
        old = 1_000_000_000
        os.utime(self.output, (old, old))
        (directory / 'a.txt').write_text('three\n')
        self.assertEqual(generate(directory, self.output)['modified'], 1)
        self.assertNotEqual(self.output.stat().st_mtime, old)


if __name__ == '__main__':
    unittest.main()
