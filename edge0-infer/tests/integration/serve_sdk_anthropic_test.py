"""Exercise Anthropic Messages with the pinned official SDK and accumulator."""
import argparse
from pathlib import Path
import socket
import sys
import tempfile
import anthropic
import httpx2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_support import Server, tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    args = parser.parse_args()
    assert anthropic.__version__ == '2.8.0'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        key = root / 'key'; key.write_text('sdk-secret')
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
        address = next(v for v in addresses if not v.startswith(('127.', '169.254.')))
        with Server(args.cli, tiny(), root / 'server', extra=['--host', '0.0.0.0', '--api-key-file', str(key)]) as server:
            with anthropic.Anthropic(base_url=f'http://{address}:{server.port}', api_key='sdk-secret', max_retries=0,
                                     http_client=httpx2.Client(trust_env=False)) as client:
                body = dict(model='test', messages=[{'role': 'user', 'content': 'hello'}], max_tokens=4, extra_body={'temperature': 0})
                first = client.messages.create(**body)
                text = ''.join(b.text for b in first.content if b.type == 'text')
                assert text and first.stop_reason == 'max_tokens'
                with client.messages.create(**body, stream=True) as stream:
                    streamed = ''.join(e.delta.text for e in stream if e.type == 'content_block_delta' and e.delta.type == 'text_delta')
                assert streamed == text
                with client.messages.stream(**body) as stream:
                    accumulated = ''.join(stream.text_stream)
                    final = stream.get_final_message()
                assert accumulated == text == ''.join(b.text for b in final.content if b.type == 'text')
                assert final.usage.output_tokens == 4 and final.usage.cache_creation_input_tokens is None
                count = client.messages.count_tokens(model='test', messages=body['messages'])
                assert count.input_tokens == server.evidence(3)[-1][1]['input_count']
            with anthropic.Anthropic(base_url=f'http://{address}:{server.port}', api_key='wrong', max_retries=0,
                                     http_client=httpx2.Client(trust_env=False)) as client:
                try:
                    client.messages.create(**body)
                    raise AssertionError('incorrect x-api-key accepted')
                except anthropic.AuthenticationError as error:
                    assert error.body['type'] == 'error' and error.body['error']['type'] == 'authentication_error'
    print('PASS Anthropic 2.8.0 create, stream, accumulator, count_tokens and x-api-key')


if __name__ == '__main__':
    main()
