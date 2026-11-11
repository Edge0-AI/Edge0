#!/usr/bin/env python3
"""Correctness check for the vision path: render a code word, ask the model to read it.

This configuration runs the image encoder on the CPU (`VISION=cpu`), so this checks that
path. The picture is rendered with Pillow (no downloaded assets), sent the way an
OpenAI-compatible client sends one (an `image_url` with a `data:` URL), and the answer is
compared with the code that was drawn.
"""
import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CODE = "K7QX-4291"
QUESTION = ("Read the code printed in the image and reply with that code only, "
            "exactly as written, with no other words.")


def render() -> Image.Image:
    """A code word in black on white, big enough to be unambiguous at the encoder's 300 tokens."""
    image = Image.new("RGB", (900, 260), "white")
    ImageDraw.Draw(image).text((40, 60), CODE, fill="black", font=ImageFont.load_default(size=110))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", type=Path, default=Path("/data/bench64"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = render()
    image.save(args.out / "image-check.png")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    body = json.dumps({
        "model": "strata",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": QUESTION},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": 64,
    }).encode()

    request = urllib.request.Request(args.url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    answer = payload["choices"][0]["message"]["content"].strip()
    with urllib.request.urlopen(args.url + "/metrics", timeout=30) as response:
        engine = json.load(response)["requests"][0]

    result = {"expected": CODE, "answer": answer, "found": CODE in answer,
              "usage": payload.get("usage"), "engine": engine}
    (args.out / "image-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
