"""#1229: a phone JPEG stored turned (EXIF Orientation 6 or 8) is turned upright before the vision encoder reads it."""
import io
import unittest

from serve.server import Vision

try:
    from PIL import Image
except ImportError:                 # Pillow is optional: the image then goes through as it was
    Image = None


def jpeg(size, orientation=None) -> bytes:
    im = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    out = io.BytesIO()
    im.save(out, format="JPEG", exif=exif)
    return out.getvalue()


@unittest.skipIf(Image is None, "Pillow is not installed")
class ExifOrientation(unittest.TestCase):
    def test_a_turned_jpeg_comes_out_upright_as_a_png(self):
        for tag in (6, 8):
            out = Vision.normalize(jpeg((40, 20), tag))
            self.assertEqual(out[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(Image.open(io.BytesIO(out)).size, (20, 40))

    def test_a_jpeg_without_the_tag_or_with_tag_1_passes_through_unchanged(self):
        for tag in (None, 1):
            data = jpeg((40, 20), tag)
            self.assertIs(Vision.normalize(data), data)

    def test_png_passes_through(self):
        out = io.BytesIO()
        Image.new("RGB", (4, 4)).save(out, format="PNG")
        self.assertEqual(Vision.normalize(out.getvalue()), out.getvalue())

    def test_a_broken_jpeg_is_left_as_it_was_sent(self):
        data = b"\xff\xd8\xff" + b"not a jpeg"
        self.assertEqual(Vision.normalize(data), data)


if __name__ == "__main__":
    unittest.main()
