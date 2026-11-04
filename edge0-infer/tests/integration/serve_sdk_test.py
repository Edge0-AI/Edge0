"""Exercise the pinned official SDK including its streaming accumulators."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import openai
sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_support import Server, tiny


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as root, Server(args.cli, tiny(), root) as server:
        client = openai.OpenAI(base_url=server.base + '/v1', api_key='local', max_retries=0)
        messages = [{'role': 'user', 'content': 'hello'}]
        first = client.chat.completions.create(model='test', messages=messages, logprobs=True, top_logprobs=2)
        assert first.choices[0].message.content and first.choices[0].logprobs.content
        messages += [first.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'again'}]
        with client.chat.completions.stream(model='test', messages=messages) as stream:
            for event in stream:
                pass
            try:
                final = stream.get_final_completion()
            except openai.LengthFinishReasonError as error:
                # Tiny random weights stop at the explicit limit; the SDK raises
                # after successfully accumulating the completion.
                final = error.completion
                assert final.choices[0].finish_reason == 'length' 
            assert final.choices[0].message.content
        text = ''.join(c.choices[0].delta.content or '' for c in client.chat.completions.create(model='test', messages=messages, stream=True) if c.choices)
        assert text
        response = client.responses.create(model='test', input='hello')
        assert response.output_text
        with client.responses.stream(model='test', input='hello', include=['message.output_text.logprobs'], top_logprobs=2) as stream:
            snapshot_text = ''
            final = None
            for event in stream:
                if event.type == 'response.output_text.delta':
                    snapshot_text = event.snapshot
                if event.type in ('response.completed', 'response.incomplete'):
                    final = event.response
            assert final is not None and final.output_text == snapshot_text
            assert final.output[0].content[0].logprobs
            if final.status == 'completed':
                assert stream.get_final_response().output_text == snapshot_text
        # -I ignores PYTHONIOENCODING; -X utf8 forces UTF-8 stdio on Windows GBK consoles.
        subprocess.run([sys.executable, '-I', '-X', 'utf8',
                        str(Path(__file__).resolve().parents[2] / 'examples/openai_client.py'),
                        '--base-url', server.base + '/v1', '--model', 'test'],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('PASS SDK 3.26.1 complete, stream, accumulators, logprobs, example')


if __name__ == '__main__':
    main()
