#!/usr/bin/env python3
"""Repeat the benchmark request N times; print decode tok/s per request."""
import json, sys, requests
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for i in range(n):
    r = requests.post(f'http://127.0.0.1:{port}/v1/chat/completions', json={
        'messages': [{'role': 'user', 'content': 'Name the capital of France in one word.'}],
        'max_tokens': 200, 'temperature': 0}, timeout=600)
    t = r.json().get('timings', {})
    print(f'req {i+1}: decode {t.get("predicted_per_second")} tok/s, '
          f'prompt {t.get("prompt_per_second")} tok/s', flush=True)
