"""Chat twice and call Responses using a local OpenAI-compatible server."""
import argparse
from openai import OpenAI
parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
parser.add_argument('--model', required=True)
args = parser.parse_args()
client = OpenAI(base_url=args.base_url, api_key='local')
messages = []
for prompt in ('Hello!', 'Please tell me more.'):
    messages.append({'role': 'user', 'content': prompt})
    answer = ''
    for chunk in client.chat.completions.create(model=args.model, messages=messages, stream=True):
        if chunk.choices:
            text = chunk.choices[0].delta.content or ''
            print(text, end='', flush=True)
            answer += text
    print()
    messages.append({'role': 'assistant', 'content': answer})
print(client.responses.create(model=args.model, input='Say hello.').output_text)
