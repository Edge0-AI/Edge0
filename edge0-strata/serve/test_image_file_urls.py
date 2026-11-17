"""Local image URIs select the named file before entering the vision pipeline."""
import json
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, Vision, serve
from serve import test_server as fixtures


def pixel(red, green, blue):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes((0, red, green, blue)))) + chunk(b"IEND", b""))


class FileUris(unittest.TestCase):
    def test_local_uris_deliver_the_named_picture_over_http(self):
        class ReadingVision(fixtures.ImageSources.Vision):
            def encode(self, source):
                data = source if isinstance(source, bytes) else Vision.load(source)
                return super().encode(data)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spaced, percent = root / "a b.png", root / "a%20b.png"
            unicode = root / "caf\u00e9-\u56fe.png"
            reserved = root / "hash#percent%.png"
            for path, data in ((spaced, pixel(255, 0, 0)), (percent, pixel(0, 0, 255)),
                               (unicode, pixel(0, 255, 0)), (reserved, pixel(128, 128, 128))):
                path.write_bytes(data)
            vision = ReadingVision(tmp)
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, "</think>\n\nok", max_context=16384), tok,
                          ChatTemplate(fixtures.ROOT / "serve/chat_template.jinja"), vision=vision)
            httpd = serve(svc, port=0)
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            sources = [(str(spaced), spaced), (spaced.as_uri(), spaced), (percent.as_uri(), percent),
                       (unicode.as_uri(), unicode), (reserved.as_uri(), reserved),
                       (spaced.as_uri().replace("file:///", "file://localhost/", 1), spaced)]
            for source, expected in sources:
                for api in ("chat/completions", "responses"):
                    with self.subTest(source=source, api=api):
                        if api == "responses":
                            body = {"input": [{"role": "user", "content": [
                                {"type": "input_text", "text": "What color is this picture?"},
                                {"type": "input_image", "image_url": source}]}], "max_output_tokens": 8}
                        else:
                            body = {"messages": [{"role": "user", "content": [
                                {"type": "text", "text": "What color is this picture?"},
                                {"type": "image_url", "image_url": {"url": source}}]}], "max_tokens": 8}
                        vision.got.clear()
                        request = urllib.request.Request(base + "/v1/" + api, data=json.dumps(body).encode(),
                                                         headers={"Content-Type": "application/json"})
                        try:
                            with urllib.request.urlopen(request, timeout=10) as reply:
                                status, result = reply.status, json.loads(reply.read())
                        except urllib.error.HTTPError as error:
                            with error:
                                status, result = error.code, json.loads(error.read())
                        self.assertEqual(status, 200, result)
                        self.assertEqual(vision.got, [expected.read_bytes()])

    def test_windows_uri_conversion_keeps_drive_spellings(self):
        import nturl2path
        expected = "C:" + chr(92) + "pictures" + chr(92) + "a b.png"
        sources = ("file:///C:/pictures/a%20b.png", "file://C:/pictures/a%20b.png",
                   "file://C:" + chr(92) + "pictures" + chr(92) + "a%20b.png")
        with mock.patch("urllib.request.url2pathname", nturl2path.url2pathname), \
                mock.patch("os.path.isfile", side_effect=lambda path: path == expected) as isfile, \
                mock.patch.object(Path, "read_bytes", return_value=b"picture"):
            for source in sources:
                with self.subTest(source=source):
                    self.assertEqual(Vision.load(source), b"picture")
                    self.assertEqual(isfile.call_args.args[0], expected)

    def test_encoded_network_paths_are_refused_before_filesystem_access(self):
        sources = ("file://remote/share/x.png", "file://localhost/%5c%5chost/share/x.png",
                   "file:///%2fhost/share/x.png", "file://localhost/%5c%3f%3f%5cUNC/host/share/x.png")
        with mock.patch("os.path.isfile") as isfile, mock.patch.object(Path, "read_bytes") as read:
            for source in sources:
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, "network paths"):
                    Vision.load(source)
            isfile.assert_not_called()
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
