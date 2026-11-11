"""vq.py PORT IMG [IMG2] : the picture questions through the server (chat completions, image_url data URL), greedy,
thinking off.  Prints each answer and the usage."""
import base64, json, sys, time, urllib.request
port, img = sys.argv[1], sys.argv[2]
QS = {"code4217.png": ["What text is written in the image? Answer with the text only.",
                       "Describe the shapes in the image with their colours and positions, one short line each."],
      "shapes.png": ["What text is written in the image? Answer with the text only.",
                     "List every shape in the image with its colour, from left to right. One short line each."]}
name = img.rsplit("/", 1)[-1]
b64 = base64.b64encode(open(img, "rb").read()).decode()
for q in QS.get(name, ["Describe this image."]):
    body = {"model": "x", "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                                                      {"type": "text", "text": q}]}]}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    print("Q:", q)
    print("A:", repr(r["choices"][0]["message"]["content"]), "%.1fs" % (time.time() - t0))
    print("usage", r.get("usage"), flush=True)
