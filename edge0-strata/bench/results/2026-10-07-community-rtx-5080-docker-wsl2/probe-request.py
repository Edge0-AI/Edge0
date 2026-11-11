#!/usr/bin/env python3
"""POST a saved request body and print the server's answer, error body included."""
import json
import sys
import urllib.error
import urllib.request

path = sys.argv[1]
body = open(path, "rb").read()
parsed = json.loads(body)
messages = parsed.get("messages", [])
print("request bytes:", len(body))
print("messages:", len(messages), "| max_tokens:", parsed.get("max_tokens"))
print("prompt chars:", len(messages[0]["content"]) if messages else 0)

req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.load(response)
    print("HTTP 200 | usage:", payload.get("usage"))
except urllib.error.HTTPError as error:
    print("HTTP", error.code)
    print(error.read().decode("utf-8", "replace")[:2000])
