"""Real-weight Portal predictions; failures preserve the original criteria."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import openai
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_support import Server
from memory_metric import process_peak
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/assets"))
from format_gguf import Reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--decode', required=True)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='metal')
    parser.add_argument('--case', default='all')
    args = parser.parse_args()
    model = Path(os.environ.get('MOE_ASSETS_35B', '/nonexistent'))
    if not (model / 'assets.tsv').is_file():
        print('SKIP: missing real assets')
        return 77
    failures = []
    def require(value, label):
        print(('PASS ' if value else 'FAIL ') + label, flush=True)
        if not value:
            failures.append(label)
    index = Reader(model / 'base.gguf')
    architecture = index.metadata['general.architecture'].data
    expert_count = index.metadata[architecture + '.expert_count'].data
    default_slots = index.metadata[architecture + '.expert_used_count'].data
    projections = [tensor for name, tensor in index.tensors.items() if name.endswith(('.ffn_gate_exps.weight','.ffn_up_exps.weight','.ffn_down_exps.weight'))]
    assert projections
    counts = {}
    def report(server, label):
        counts[server] = counts.get(server, 0) + 1
        tokens, result = server.evidence(counts[server])[-1]
        physical = result['memory_snapshot']['physical']
        print(label, json.dumps({k: result[k] for k in ('reuse_method', 'reused_tokens', 'checkpoint_boundary', 'stop_reason', 'reasoning_tokens', 'input_count', 'generated_count')}, ensure_ascii=True), 'physical=', physical, flush=True)
        # The native summary represents absent memory metrics explicitly.
        metric, source, peak = process_peak(result['memory_snapshot'])
        require(peak <= 4 * 1024**3, f'{label} {metric} peak ({source}) <= 4 GiB')
        status_code, body = server.request('/status')
        require(status_code == 200, label + ' status query')
        status = json.loads(body)
        ledger = status['memory']['ledger']
        if args.backend == 'metal':
            devices = [d['name'] for d in status['engine']['devices'] if d['backend'] == 'metal']
            require(any(ledger['by_device'].get(name,{}).get('current_bytes',0) > 0 and ledger['by_device'][name]['memory'] == 'system' for name in devices), label + ' Metal system memory ledger')
        slots = int(result['loading']['model_options'].get('expert_slots',default_slots))
        read = result['loading']['model_options'].get('prefill_read','auto')
        copies = 2 if read in ('auto','ahead') else 1
        # Sum each layer's slot capacity times its one-expert projection bytes,
        # plus the full first-layer expert bank for every configured prefill copy.
        # These 35B projection sizes are already multiples of backend alignment.
        pool_bytes = sum(t.size // expert_count * slots for t in projections)
        prefill_bytes = copies * sum(t.size for t in projections if t.name.startswith('blk.0.'))
        require(ledger['by_purpose']['experts'] == pool_bytes + prefill_bytes > 0, label + ' exact expert reservation')
        require(status['totals']['experts']['demand'] > 0, label + ' cumulative expert demand')
        return tokens, result
    prompts = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/multi_turn/conversations.json').read_text(encoding='utf-8'))[0]['messages'][:2]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for thinking in ((False, True) if args.case in ('all', 'conversations') else ()):
            with Server(args.cli, model, root / ('thinking' if thinking else 'plain'), args.backend, ['--max-tokens', '1024']) as server:
                client = openai.OpenAI(base_url=server.base + '/v1', api_key='local', max_retries=0)
                messages = []
                evidence = []
                for turn, prompt in enumerate(prompts):
                    messages.append({'role': 'user', 'content': prompt})
                    answer, reasoning = '', ''
                    for chunk in client.chat.completions.create(model='test', messages=messages, stream=True,
                            reasoning_effort='medium' if thinking else 'none', stream_options={'include_usage': True}):
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            answer += delta.content or ''
                            reasoning += (delta.model_extra or {}).get('reasoning_content', '')
                    message = {'role': 'assistant', 'content': answer}
                    if thinking:
                        message['reasoning_content'] = reasoning
                    messages.append(message)
                    print('text lengths', len(answer.encode()), len(reasoning.encode()), flush=True)
                    pair = report(server, f'thinking={thinking} turn={turn}')
                    evidence.append(pair)
                    if thinking:
                        require(bool(reasoning) and bool(answer) and pair[1]['reasoning_tokens'] > 0, 'thinking and answer nonempty')
                first, second = evidence
                if not thinking:
                    require(second[1]['reuse_method'] == 'keep', '(k) second turn keep')
                elif second[1]['reuse_method'] == 'keep':
                    require(True, '(l) thinking keep')
                else:
                    require(second[1]['reuse_method'] == 'restore', '(l) thinking restore')
                    prior = (first[0]['input'] + first[0]['generated'])[:first[1]['e']]
                    after = second[0]['input']
                    divergence = next((i for i, (a, b) in enumerate(zip(prior, after)) if a != b), min(len(prior), len(after)))
                    require(second[1]['checkpoint_restored'] and second[1]['reused_tokens'] == first[1]['checkpoint_boundary'] == len(first[0]['input']) - 1, '(l) restored checkpoint boundary')
                    require(len(first[0]['input']) <= divergence < len(prior), '(l) divergence in generated span')
                    marker = subprocess.check_output([args.cli, 'tokenize', '--model', str(model), '--parse-special', '<|im_end|>'], text=True).strip()
                    end = after.index(int(marker), divergence)
                    def decode(tokens):
                        return subprocess.check_output([args.decode, str(model)], input=' '.join(map(str, tokens)).encode())
                    require(decode(prior[divergence:]) == decode(after[divergence:end]), '(l) divergent text byte equality')
        tool = {'name': 'get_weather', 'description': 'Get the current weather for a city.', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}
        prompt = '\u5317\u4eac\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f'
        output = '{"temperature": 23, "condition": "\\u6674"}'
        for protocol in (p for p in ('chat', 'responses') if args.case in ('all', 'tools', p)):
            with Server(args.cli, model, root / protocol, args.backend, ['--max-tokens', '1024']) as server:
                client = openai.OpenAI(base_url=server.base + '/v1', api_key='local', max_retries=0)
                if protocol == 'chat':
                    messages = [{'role': 'user', 'content': prompt}]
                    tools = [{'type': 'function', 'function': tool}]
                    first = client.chat.completions.create(model='test', messages=messages, tools=tools)
                    report(server, '(m) Chat tool first')
                    calls = first.choices[0].message.tool_calls or []
                    require(first.choices[0].finish_reason == 'tool_calls' and len(calls) == 1, '(m) Chat tool end')
                    if not calls:
                        continue
                    call = calls[0]
                    print("tool call", call.model_dump_json(), flush=True)
                    require(call.function.name == 'get_weather' and '\u5317\u4eac' in json.loads(call.function.arguments)['city'], '(m) Chat name and city')
                    messages += [first.choices[0].message.model_dump(exclude_none=True), {'role': 'tool', 'tool_call_id': call.id, 'content': output}]
                    second = client.chat.completions.create(model='test', messages=messages, tools=tools)
                    answer = second.choices[0].message.content or ''
                else:
                    tools = [{'type': 'function', **tool}]
                    first = client.responses.create(model='test', input=prompt, tools=tools)
                    report(server, '(m) Responses tool first')
                    calls = [item for item in first.output if item.type == 'function_call']
                    require(first.status == 'completed' and len(calls) == 1, '(m) Responses tool end')
                    if not calls:
                        continue
                    call = calls[0]
                    print("tool call", call.model_dump_json(), flush=True)
                    require(call.name == 'get_weather' and '\u5317\u4eac' in json.loads(call.arguments)['city'], '(m) Responses name and city')
                    history = [{'role': 'user', 'content': prompt}] + [item.model_dump(exclude_none=True) for item in first.output] + [{'type':'function_call_output', 'call_id':call.call_id, 'output':output}]
                    second = client.responses.create(model='test', input=history, tools=tools)
                    answer = second.output_text
                _, result = report(server, '(m) ' + protocol + ' tool second')
                require('23' in answer and not result['cold'], '(m) ' + protocol + ' result and reuse')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
