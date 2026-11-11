"""Answers through the edge0-infer proxy and directly from edge0-strata, on one running instance.

Usage: compare_outputs.py --cli <edge0-infer> --model <145B directory> [--device cuda:0] [--strata-dir DIR]
       [--max-tokens 200] [--out answers.json]

Starts `edge0-infer serve` (which starts edge0-strata), reads edge0-strata's loopback port from the
log line "edge0-strata: ready (pid N, port P)", and asks three fixed questions greedily with thinking
off (temperature 0, `chat_template_kwargs.enable_thinking=false`, no streaming):

1. directly to edge0-strata, twice, to see whether edge0-strata itself repeats its answers;
2. through the proxy (the Listening port), once.

Exit codes: 0 when edge0-strata repeats itself and every proxied answer equals the direct one byte for
byte; 1 when edge0-strata repeats itself but a proxied answer differs; 2 when edge0-strata does not
repeat itself (then the answers are printed for a manual check and the differing runs are named).
"""
import argparse
import http.client
import json
import re
import signal
import subprocess
import sys
import threading
import time

QUESTIONS = [
    'What is the capital of France? Answer in one sentence.',
    '\u7528\u4e09\u53e5\u8bdd\u4ecb\u7ecd\u4e00\u4e0b\u957f\u57ce\u3002',  # Chinese: introduce the Great Wall in three sentences.
    'Write a Python function that returns the n-th Fibonacci number, with a short docstring.',
]


def ask(port, alias, question, max_tokens):
    body = {'model': alias, 'messages': [{'role': 'user', 'content': question}], 'temperature': 0, 'top_k': 1,
            'max_tokens': max_tokens, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    if response.status != 200:
        raise SystemExit(f'port {port}: HTTP {response.status}: {payload[:300]!r}')
    value = json.loads(payload)
    choice = value['choices'][0]
    return {'content': choice['message'].get('content'), 'reasoning': choice['message'].get('reasoning_content'),
            'finish_reason': choice.get('finish_reason'), 'usage': value.get('usage')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--strata-dir')
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--out', help='write all answers as JSON')
    args = parser.parse_args()
    command = [args.cli, 'serve', '--model', args.model, '--device', args.device, '--port', '0']
    if args.strata_dir:
        command += ['--strata-dir', args.strata_dir]
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    ports = {}
    ready = threading.Event()

    def read():
        for line in server.stderr:
            sys.stderr.write(line)
            match = re.search(r'edge0-strata: ready \(pid \d+, port (\d+)\)', line)
            if match:
                ports['direct'] = int(match[1])
            match = re.match(r'Listening: http://[^:]+:(\d+) model=(\S+)', line)
            if match:
                ports['proxy'], ports['alias'] = int(match[1]), match[2]
                ready.set()
        ready.set()
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        ready.wait(3600)
        if 'proxy' not in ports or 'direct' not in ports:
            raise SystemExit('edge0-infer serve did not become ready')
        results = []
        for question in QUESTIONS:
            first = ask(ports['direct'], ports['alias'], question, args.max_tokens)
            second = ask(ports['direct'], ports['alias'], question, args.max_tokens)
            proxied = ask(ports['proxy'], ports['alias'], question, args.max_tokens)
            results.append({'question': question, 'direct_1': first, 'direct_2': second, 'proxy': proxied})
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(120)
        except subprocess.TimeoutExpired:
            server.kill()
        reader.join(10)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as out:
            json.dump(results, out, ensure_ascii=False, indent=1)
    repeatable = all(r['direct_1']['content'] == r['direct_2']['content'] for r in results)
    identical = all(r['proxy']['content'] == r['direct_1']['content'] for r in results)
    for index, r in enumerate(results, 1):
        print(f'--- question {index}: {r["question"]}')
        print(f'direct 1 == direct 2: {r["direct_1"]["content"] == r["direct_2"]["content"]}; '
              f'proxy == direct 1: {r["proxy"]["content"] == r["direct_1"]["content"]}; '
              f'finish {r["direct_1"]["finish_reason"]}/{r["direct_2"]["finish_reason"]}/{r["proxy"]["finish_reason"]}; '
              f'usage {json.dumps(r["proxy"]["usage"])}')
        print(r['proxy']['content'])
        if r['direct_1']['content'] != r['direct_2']['content']:
            print('>>> direct 2 differs:')
            print(r['direct_2']['content'])
    if not repeatable:
        print('VERDICT MANUAL: edge0-strata does not repeat its own answers; check the answers above by hand')
        sys.exit(2)
    print('VERDICT', 'PASS' if identical else 'FAIL')
    sys.exit(0 if identical else 1)


if __name__ == '__main__':
    main()
