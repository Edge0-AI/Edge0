"""make_big_picture.py OUT : a 1024x1024 test picture (about 1,024 image tokens, a prompt that spans several chunks): a heading, a
few lines of small text and coloured shapes on white."""
import sys
from PIL import Image, ImageDraw, ImageFont

im = Image.new("RGB", (1024, 1024), "white")
d = ImageDraw.Draw(im)


def font(size, bold=True):
    for f in (("arialbd.ttf" if bold else "arial.ttf"), "C:/Windows/Fonts/arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


d.text((90, 60), "INVOICE 20261014", fill="black", font=font(96))
d.text((90, 200), "Total due: 1337 EUR", fill=(20, 20, 160), font=font(64))
for i, line in enumerate(["Item 1   widgets   x 12", "Item 2   gears     x 7", "Item 3   springs   x 30"]):
    d.text((90, 320 + 70 * i), line, fill="black", font=font(48, False))
d.ellipse([560, 600, 960, 1000], fill=(30, 160, 60))
d.rectangle([90, 640, 430, 980], fill=(220, 30, 30))
im.save(sys.argv[1])
print("wrote", sys.argv[1])
