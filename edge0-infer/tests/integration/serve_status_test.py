"""Status contract checked against the public field table and native request evidence."""
import datetime
import http.client
import json
import math
from pathlib import Path
import re
import socket
import subprocess
from serve_support import Server


def check_schema(snapshots):
    document = (Path(__file__).resolve().parents[2] / 'docs/openai-compatible-server.md').read_text()
    table = document.split('<!-- status-fields-begin -->')[1].split('<!-- status-fields-end -->')[0]
    fields = {}
    for line in table.splitlines():
        match = re.match(r'\| `([^`]+)` \| ([^|]+) \| (required|optional) \|', line)
        if match:
            fields[match[1]] = (set(match[2].strip().split(' or ')), match[3])
    assert fields
    seen = set()
    def kind(value):
        if value is None: return 'null'
        if isinstance(value, bool): return 'boolean'
        if isinstance(value, int): return 'integer'
        if isinstance(value, float): return 'number'
        if isinstance(value, str): return 'string'
        if isinstance(value, list): return 'array'
        return 'object'
    def walk(value, path=''):
        if path:
            assert path in fields, ('undocumented field', path)
            types = fields[path][0]
            actual = kind(value)
            assert actual in types or actual == 'integer' and 'number' in types, (path, actual, types)
            seen.add(path)
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = '*' if path == 'memory.ledger.by_device' else key
                walk(child, path + '.' + normalized if path else normalized)
            prefix = path + '.' if path else ''
            for field, (_, presence) in fields.items():
                if presence != 'required' or not field.startswith(prefix): continue
                suffix = field[len(prefix):]
                if '.' not in suffix and '[' not in suffix and suffix != '*':
                    assert suffix in value, ('missing field', field)
        elif isinstance(value, list):
            for child in value: walk(child,path+'[]')
    for snapshot in snapshots: walk(snapshot)
    assert seen == set(fields), ('documented fields not exercised', sorted(set(fields)-seen))


def run(cli, model, root, backend):
    snapshots = []
    key = root / 'api-key'; key.write_text('  PRIVATE_AUTH_SECRET_731\n')
    hold = root / 'status-hold'
    addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    address = next((value for value in addresses if not value.startswith(('127.','169.254.'))), None)
    assert address, 'No non-loopback IPv4 address'
    with Server(cli,model,root/'status',backend,['--host','0.0.0.0','--api-key-file',str(key),'--hold-file',str(hold)]) as server:
        def remote(path, token=None):
            connection = http.client.HTTPConnection(address,server.port,timeout=15)
            headers = {} if token is None else {'Authorization':'Bearer '+token}
            connection.request('GET',path,headers=headers)
            response = connection.getresponse(); status = response.status; data = response.read(); connection.close()
            return status,json.loads(data)
        def status():
            code,body = server.request('/status'); assert code == 200
            value = json.loads(body); snapshots.append(value); return value
        initial = status(); instance = initial['instance']['id']
        assert initial['current'] is None and initial['last'] is None and initial['totals']['requests'] == 0
        assert initial['listen'] == {'host':'0.0.0.0','port':server.port,'auth':True}
        assert initial['instance']['pid'] == server.process.pid and initial['instance']['uptime_seconds'] >= 0
        assert initial['engine']['devices'] and initial['model']['directory'] == str(model)
        info = subprocess.run([cli,'info'],check=True,capture_output=True,text=True).stdout
        identity = dict(line.split(': ',1) for line in info.splitlines() if ': ' in line)
        assert initial['engine']['version'] == identity['engine.version']
        assert initial['engine']['source_commit'] == identity['engine.source_commit']
        assert initial['engine']['source_modified'] == identity['engine.source_modified'].split()[0]
        assert initial['engine']['ggml_version'] == identity['ggml.version']
        assert initial['engine']['ggml_origin'] in identity['ggml.origin']
        for device in initial['engine']['devices']:
            assert 'name='+device['name'] in info and device['description'] in info
        assert initial['engine']['backends']
        assert initial['model']['context_length'] == 4096
        for path in ('/v1/models','/v1/models/test'):
            value = json.loads(server.request(path)[1]); item = value['data'][0] if 'data' in value else value
            assert item['context_length'] == 4096
        assert remote('/v1/models','PRIVATE_AUTH_SECRET_731')[0] == 200
        first_code,first_body = server.request('/v1/chat/completions',server.chat())
        assert first_code == 200
        first = json.loads(first_body)
        messages = [{'role':'user','content':'hello'},first['choices'][0]['message'],{'role':'user','content':'again'}]
        hold.touch()
        connection = http.client.HTTPConnection('127.0.0.1',server.port)
        connection.request('POST','/v1/chat/completions',json.dumps(server.chat(messages,stream=True)),{'Content-Type':'application/json','Authorization':'Bearer WRONG_LOCAL_TOKEN'})
        response = connection.getresponse(); assert response.status == 200
        while True:
            line = response.readline().decode(); assert line
            if line.startswith('data: ') and json.loads(line[6:])['choices'][0]['delta'].get('content'): break
        live = status(); current = live['current']
        assert current['phase'] == 'decode' and current['generated_tokens'] >= 1
        assert current['pending_tokens'] == 0 and current['computed_tokens'] + current['reused_tokens'] == current['input_tokens']
        assert current['first_token_seconds'] >= 0 and current['elapsed_seconds'] >= current['first_token_seconds']
        assert current['source'] == '127.0.0.1' and current['endpoint'] == '/v1/chat/completions'
        assert server.request('/v1/chat/completions',server.chat())[0] == 429
        marker = 'PRIVATE_REQUEST_CONTENT_731'
        assert server.request('/v1/chat/completions',server.chat([{'role':'user','content':marker}],promptcachekey='x'))[0] == 400
        assert remote('/status')[0] == 401
        records = status()['recent_requests']
        assert [r['http_status'] for r in records] == [401,400,429,200], records
        assert all(r['source'] and r['total_seconds'] >= 0 for r in records)
        assert remote('/health','PRIVATE_WRONG_TOKEN_731')[0] == 401
        hold.unlink(); response.read(); connection.close()
        server.evidence(expected_requests=2)
        final = status(); assert final['current'] is None
        evidence = server.evidence(); assert len(evidence) == 2
        last = final['last']; summary = evidence[-1][1]
        for key_name in ('first_token_seconds','decode_tokens_per_second','decode_seconds','decode_forwards','total_seconds'):
            assert last[key_name] == summary['request_summary'][key_name]
        assert last['input_tokens'] == len(evidence[-1][0]['input'])
        for key_name in ('reused_tokens','computed_tokens','stop_reason'):
            assert last[key_name] == summary[key_name]
        assert last['generated_tokens'] == summary['generated_count']
        assert last['ended_at'] == last['time']
        totals = final['totals']
        assert [totals[k] for k in ('requests','rejected_busy','rejected_invalid','unauthorized')] == [2,1,1,2]
        for key_name in ('reused_tokens','computed_tokens'):
            assert totals[key_name] == sum(r[key_name] for _,r in evidence)
        assert totals['input_tokens'] == sum(len(t['input']) for t,_ in evidence)
        assert totals['generated_tokens'] == sum(r['generated_count'] for _,r in evidence)
        first_seconds = sum(r['request_summary']['first_token_seconds'] for _,r in evidence)
        prefill = sum(len(t['input'])-r['reused_tokens'] for t,r in evidence)
        decode = sum(r['request_summary']['decode_seconds'] for _,r in evidence)
        forwards = sum(r['request_summary']['decode_forwards'] for _,r in evidence)
        assert math.isclose(totals['first_token_seconds_mean'], first_seconds/2)
        assert math.isclose(totals['prefill_tokens_per_second'], prefill/first_seconds)
        assert math.isclose(totals['decode_tokens_per_second'], forwards/decode)
        assert totals['decode_seconds'] == decode
        assert totals['continued_turns'] == sum(r['reuse_method'] in ('keep','truncate','restore') for _,r in evidence)
        assert totals['checkpoint_restores'] == sum(r['checkpoint_restored'] for _,r in evidence)
        for name in ('demand','hits','loads','bytes'):
            assert totals['experts'][name] == sum(r['weight_activity'][phase][name] for _,r in evidence for phase in ('prefill','decode'))
        ledger = final['memory']['ledger']
        assert sum(ledger['by_purpose'].values()) == sum(d['current_bytes'] for d in ledger['by_device'].values()) == ledger['current_bytes']
        assert ledger['by_purpose']['resident_weights'] > 0 and ledger['by_purpose']['experts'] > 0
        assert all(d['memory'] in ('system','dedicated') for d in ledger['by_device'].values())
        assert ledger['by_device']['host']['memory'] == 'system'
        if backend == 'cpu': assert ledger['by_device']['CPU']['memory'] == 'system'
        assert ledger['limit_bytes'] >= ledger['current_bytes'] > 0 and ledger['peak_bytes'] >= ledger['current_bytes']
        for kind in ('resident','physical'):
            metric = final['memory'][kind]
            assert metric['source'] and metric['scope'] == 'process_lifetime'
            assert metric['values']['current_bytes'] > 0
            if metric['values']['peak_bytes'] is not None: assert metric['values']['peak_bytes'] > 0
            else: assert metric['values']['peak_reason']
        assert final['instance']['uptime_seconds'] >= initial['instance']['uptime_seconds']
        for record in final['recent_requests']:
            datetime.datetime.strptime(record['time'],'%Y-%m-%dT%H:%M:%S.%fZ')
        assert final['system']['memory']['total']['bytes'] > 0
        assert final['system']['memory']['available']['bytes'] <= final['system']['memory']['total']['bytes']
        with server.changed:
            assert server.changed.wait_for(lambda: len([line for line in server.lines if line.startswith('{') and '"http_status"' in line]) >= 6,timeout=30)
        logs = [json.loads(line) for line in server.lines if line.startswith('{') and '"http_status"' in line]
        assert final['recent_requests'] == list(reversed(logs))
        text = json.dumps(final)+''.join(server.lines)
        for private in (marker,'PRIVATE_AUTH_SECRET_731','PRIVATE_WRONG_TOKEN_731','WRONG_LOCAL_TOKEN'): assert private not in text
    with Server(cli,model,root/'status-restart',backend,['--context','1024','--prefill-chunk','1024','--model-option','experts=resident','--model-option','prefill_ahead_min_tokens=1']) as server:
        assert server.request('/v1/chat/completions',server.chat())[0] == 200
        value = json.loads(server.request('/status')[1]); snapshots.append(value)
        assert value['instance']['id'] != instance and not value['listen']['auth']
        assert 'experts' not in value['totals'] and 'experts' not in value['last']
        assert json.loads(server.request('/v1/models')[1])['data'][0]['context_length'] == 1024
    empty = root/'empty-key'; empty.write_text(' \n')
    for extra in (['--host','0.0.0.0'],['--api-key-file',str(empty)],['--api-key-file',str(root/'missing-key')]):
        failed = subprocess.run([cli,'serve','--model',str(root/'missing-model'),*extra],capture_output=True,text=True)
        assert failed.returncode != 0 and 'Listening:' not in failed.stderr
        assert 'key' in failed.stderr.lower() and 'asset' not in failed.stderr.lower(), failed.stderr
    check_schema(snapshots)
    print('PASS status: live progress, native equality, cumulative definitions, LAN auth, shared logs, restart and bidirectional public schema')
