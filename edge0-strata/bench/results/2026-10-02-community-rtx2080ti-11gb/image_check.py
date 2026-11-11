#!/usr/bin/env python3
"""Image question with an exact expected answer, through a running Strata server with the vision encoder.

Each run draws a new PNG (so the server's cache of encoded images cannot serve it): a code made of letters and digits
in large type, a number of red circles and a number of blue squares, chosen per run. The model is asked for all
three in a fixed format; the answer counts as correct only if all three match exactly. Streaming, greedy, reasoning
off. Client time to the first token includes the image encode (strata-vision, which runs on the CPU in this
configuration and takes its turn in the server's request FIFO) and the prompt read; the engine's prompt_ms covers
only the read of the prompt with the image embeddings, so client TTFT minus prompt_ms approximates the encode time.

    python image_check.py --key-config <server config.json> --runs 3 --engine-log ... --server-log ...
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bench_server import stream
from common import LogTail, Server, api_key, parse_engine, scrub_lines

HERE = Path(__file__).resolve().parent
QUESTION = ("Look at the image. What is the code written in black letters, how many red circles are there, and how "
            "many blue squares are there? Answer in exactly this format and nothing else: CODE; circles; squares")


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def draw(rnd: random.Random):
    code = "".join(rnd.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)) + "-" + str(rnd.randint(10, 99))
    circles, squares = rnd.randint(2, 5), rnd.randint(1, 4)
    w, h = 768, 512
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), code, fill="black", font=font(72))
    boxes = []                                            # non-overlapping cells in the lower part
    cells = [(x, y) for x in range(40, w - 100, 120) for y in (190, 330)]
    rnd.shuffle(cells)
    for i, (x, y) in enumerate(cells[:circles + squares]):
        if i < circles:
            d.ellipse((x, y, x + 90, y + 90), fill=(220, 30, 30), outline="black", width=2)
        else:
            d.rectangle((x, y, x + 90, y + 90), fill=(30, 60, 220), outline="black", width=2)
        boxes.append((x, y))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), code, circles, squares


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key-config")
    ap.add_argument("--engine-log", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=431)
    a = ap.parse_args()
    server = Server(a.url, api_key(a.key_config))
    elog, slog = LogTail(a.engine_log), LogTail(a.server_log)
    rnd = random.Random(a.seed)
    for run in range(1, a.runs + 1):
        png, code, circles, squares = draw(rnd)
        (a.out / f"image-run{run}.png").write_bytes(png)
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        req = {"model": "strata", "temperature": 0, "reasoning_effort": "none", "max_tokens": 40, "stream": True,
               "stream_options": {"include_usage": True},
               "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                         {"type": "text", "text": QUESTION}]}]}
        for attempt in range(1, 4):
            status = server.wait_idle()
            before = server.totals()
            elog.mark(), slog.mark()
            t0 = time.time()
            res = stream(server, req, timeout=900)
            time.sleep(1.5)
            after = server.totals()
            elines, slines = elog.since(), slog.since()
            parsed = parse_engine(elines)
            foreign = (after - before != 1) or len(parsed) != 1
            expected = f"{code}; {circles}; {squares}"
            got = res["text"].strip()
            parts = [p.strip() for p in got.split(";")]
            ok = parts == [code, str(circles), str(squares)]
            eng = parsed[0] if parsed else {}
            row = {"label": f"image-run{run}", "attempt": attempt, "epoch_start": t0, "image": f"image-run{run}.png",
                   "image_px": [768, 512], "expected": expected, "answer": got, "correct": ok, **res,
                   "engine": eng or None, "metrics_record": server.get("/metrics")["requests"][0],
                   "approx_encode_s": (res["client_ttft_s"] - eng["prompt_ms_log"] / 1000) if eng and res["client_ttft_s"] else None,
                   "status_before": status, "totals_before": before, "totals_after": after, "foreign_overlap": foreign,
                   "engine_log": scrub_lines(elines), "server_log": scrub_lines(slines)}
            name = f"image-run{run}" + (f"-discarded-{attempt}" if foreign else "")
            (a.out / f"{name}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n")
            print(f"image run {run}: expected {expected!r} got {got!r} -> {'OK' if ok else 'WRONG'}; "
                  f"ttft {res['client_ttft_s']:.2f} s, {eng.get('line', '')}", flush=True)
            if not foreign:
                break


if __name__ == "__main__":
    main()
