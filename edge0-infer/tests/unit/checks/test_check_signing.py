"""Signing policy examples use synthetic values, never developer identities."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('signing', Path(__file__).resolve().parents[3] / 'tools/checks/check_signing.py')
signing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signing)


class SigningTest(unittest.TestCase):
    def test_literals(self):
        for key, value in [('DEVELOPMENT_TEAM', 'SYNTHETIC'), ('PRODUCT_BUNDLE_IDENTIFIER', 'example.test.app'),
                           ('CODE_SIGN_IDENTITY', 'Example Identity')]:
            result = signing.inspect('config', '\n' + key + ' = ' + value)
            self.assertTrue(result)
            self.assertIn('config:2: literal-signing-value', result[0])

    def test_environment(self):
        for value in ('$ENV{MOE_IOS_TEAM}', '${MOE_IOS_BUNDLE}', '<TEAM>', 'YOUR_TEAM'):
            self.assertFalse(signing.inspect('config', 'DEVELOPMENT_' + 'TEAM = ' + value))

    def test_bundle_identity_with_team(self):
        # A generated identifier carries the team; only the environment-variable form may be tracked.
        for app in ('edge0-infer', 'edge0-chat'):
            for prefix in ('dev.edge0.', 'dev.moe.'):
                result = signing.inspect('notes.md', 'id ' + prefix + 'ABCDE12345.' + app)
                self.assertIn('notes.md:1: literal-bundle-identity', result[0])
            self.assertFalse(signing.inspect('notes.md', 'id dev.edge0.${MOE_IOS_TEAM}.' + app))

    def test_artifacts(self):
        for path in ('App.xcodeproj/project.pbxproj', 'App.xcworkspace/contents', 'a.mobileprovision', 'xcuserdata/user'):
            self.assertIn('generated-signing-artifact', signing.inspect(path, '')[0])

    def test_evidence(self):
        path = 'openspec/changes/example/evidence/run.log'
        self.assertIn('configured-private-identity', signing.inspect(path, 'synthetic-secret', ['synthetic-secret'])[0])
        self.assertTrue(signing.inspect('config.xcconfig', 'DEVELOPMENT_' + 'TEAM = SYNTHETIC'))


if __name__ == '__main__':
    unittest.main()
