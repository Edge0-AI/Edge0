"""#3722: HTTPS downloads check the server against the operating system's certificates (truststore), fall back to certifi, and a
certificate that still fails gives a plain explanation, never the raw urlopen text and never a way to skip the check."""

import ssl
import sys
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strata_control import tls  # noqa: E402

RAW = ("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate "
       "(_ssl.c:1010)>")


def cert_error():
    inner = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")
    return urllib.error.URLError(inner)


class Install(unittest.TestCase):
    def setUp(self):
        self.mode = tls.mode
        self.context = ssl._create_default_https_context
        tls.mode = None
        self.addCleanup(self.restore)

    def restore(self):
        tls.mode = self.mode
        ssl._create_default_https_context = self.context

    def test_truststore_is_used_when_present(self):
        fake = types.SimpleNamespace(calls=0)
        fake.inject_into_ssl = lambda: setattr(fake, "calls", fake.calls + 1)
        with mock.patch.dict(sys.modules, {"truststore": fake}):
            self.assertEqual(tls.install(), "truststore")
            self.assertEqual(tls.install(), "truststore")          # again: nothing more happens
        self.assertEqual(fake.calls, 1)

    def test_certifi_is_the_fallback_when_truststore_is_missing(self):
        seen = {}
        fake_certifi = types.SimpleNamespace(where=lambda: "bundle.pem")
        with mock.patch.dict(sys.modules, {"truststore": None, "certifi": fake_certifi}), \
                mock.patch.object(ssl, "create_default_context", lambda cafile=None, **k: seen.setdefault("cafile", cafile)):
            self.assertEqual(tls.install(), "certifi")
            ssl._create_default_https_context()
        self.assertEqual(seen["cafile"], "bundle.pem")

    def test_python_default_stays_when_neither_exists(self):
        with mock.patch.dict(sys.modules, {"truststore": None, "certifi": None}):
            self.assertEqual(tls.install(), "default")
        self.assertIs(ssl._create_default_https_context, self.context)

    def test_the_pins_name_truststore_in_requirements_and_every_lock(self):
        self.assertIn("truststore==", (ROOT / "requirements.txt").read_text())
        for lock in (ROOT / "packaging" / "locks").glob("runtime-*-cp312.txt"):
            text = lock.read_text()
            self.assertRegex(text, r"truststore==\S+ \\r?\n\s+--hash=sha256:[0-9a-f]{64}", lock.name)
            self.assertIn("certifi==", text)                       # the fallback stays


class Plain(unittest.TestCase):
    def test_a_failed_certificate_is_recognised_however_it_is_wrapped(self):
        self.assertTrue(tls.is_certificate_error(cert_error()))
        self.assertTrue(tls.is_certificate_error(OSError(RAW)))
        wrapped = RuntimeError("x")
        wrapped.__cause__ = cert_error()
        self.assertTrue(tls.is_certificate_error(wrapped))
        self.assertFalse(tls.is_certificate_error(urllib.error.URLError(ConnectionRefusedError())))
        self.assertFalse(tls.is_certificate_error(None))

    def test_the_message_is_plain_and_never_offers_to_skip_the_check(self):
        text = tls.explain("github.com")
        for word in ("github.com", "antivirus", "proxy", "date and time", "Windows Update"):
            self.assertIn(word, text)
        for raw in ("urlopen", "CERTIFICATE_VERIFY_FAILED", "_ssl.c", "issuer"):
            self.assertNotIn(raw, text)
        self.assertRegex(text, r"will not skip the check")
        self.assertNotRegex(text.lower(), r"verify\s*=\s*false|insecure|disable (the )?verification|turn off (the )?(certificate|verification)")

    def test_setup_urlopen_raises_the_plain_problem(self):
        import setup
        request = types.SimpleNamespace(full_url="https://github.com/x/y")
        with mock.patch.object(setup.acquisition, "require", lambda *a, **k: None), \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=cert_error()):
            with self.assertRaises(tls.CertificateProblem) as cm:
                setup._urlopen(request)
        self.assertIn("github.com", str(cm.exception))
        self.assertNotIn("CERTIFICATE_VERIFY_FAILED", str(cm.exception))

    def test_setup_urlopen_leaves_other_errors_alone(self):
        import setup
        refused = urllib.error.URLError(ConnectionRefusedError())
        with mock.patch.object(setup.acquisition, "require", lambda *a, **k: None), \
                mock.patch.object(setup.urllib.request, "urlopen", side_effect=refused):
            with self.assertRaises(urllib.error.URLError):
                setup._urlopen(types.SimpleNamespace(full_url="https://github.com/x"))

    def test_the_update_reader_gives_the_plain_message(self):
        from strata_control import transfer
        with mock.patch.object(transfer, "check_url_policy", lambda url: None), \
                mock.patch.object(transfer._opener, "open", side_effect=cert_error()):
            with self.assertRaises(transfer.TransferError) as cm:
                transfer.read_small("https://updates.example.org/feed.json")
        self.assertIn("antivirus", str(cm.exception))
        self.assertNotIn("urlopen", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
