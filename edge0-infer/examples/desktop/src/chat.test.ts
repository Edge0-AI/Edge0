import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Markdown } from './Markdown';
import { describe, expect, it, vi } from 'vitest';
import shared from '../../shared/app.json';
import metricsCases from '../../shared/metrics-cases.json';
import mathCases from '../../shared/math-cases.json';
import { appData, buildRequest, fetchWithRetry, formatMetrics, mapAsset, selectableAssets, protectMath, SseAccumulator, streamChat } from './chat';

describe('shared vectors and bundled app data', () => {
  it('uses the canonical prompt and ordered suggestions', () => {
    expect(appData).toEqual(shared);
    expect(buildRequest([{ role: 'user', content: 'new' }], false, 'alias').messages).toEqual([{ role: 'system', content: shared.system_prompt }, { role: 'user', content: 'new' }]);
  });
  for (const item of metricsCases) it(`metrics: ${item.name}`, () => expect(formatMetrics(item.input)).toBe(item.expected));
  for (const item of mathCases) it(`math: ${item.name}`, () => expect(protectMath(item.input)).toBe(item.expected));
  it('protects formulas starting with numbers', () => expect(protectMath('Use $2_i$ here.')).toBe('Use `$2_i$` here.'));
  it('keeps malformed values absent', () => expect(formatMetrics({ generated_count: NaN, memory_peak_bytes: -1, first_token_seconds: Infinity })).toBe(''));
  it('protects formulas containing backticks without ending the code span', () => expect(protectMath('$x`_i$')).toBe('``$x`_i$``'));
});
it('sends full history including reasoning with explicit thinking and usage controls', () => {
  const history = [{ role: 'user' as const, content: 'first' }, { role: 'assistant' as const, content: 'reply', reasoning_content: 'thought', metrics: { generated_count: 3 } }, { role: 'user' as const, content: 'next' }];
  const request = buildRequest(history, true, 'model');
  expect(request.messages).toEqual([{ role: 'system', content: shared.system_prompt }, { role: 'user', content: 'first' }, { role: 'assistant', content: 'reply', reasoning_content: 'thought' }, { role: 'user', content: 'next' }]);
  expect(request.chat_template_kwargs.enable_thinking).toBe(true); expect(request.stream_options.include_usage).toBe(true);
  expect(buildRequest([], false, 'model').chat_template_kwargs.enable_thinking).toBe(false);
});
it('labels inspected models with their identifier and directory name, separating unavailable entries', () => {
  const valid = mapAsset({ path: '/models/35b/', ok: true, detail: 'ignored', model_id: 'edge0_35b' });
  expect(valid).toEqual({ path: '/models/35b/', selectable: true, reason: '', modelId: 'edge0_35b', directoryName: '35b', label: 'edge0_35b · 35b' });
  const failed = mapAsset({ path: '/models/incomplete', ok: false, detail: 'Missing experts', model_id: null });
  expect(failed.label).toBe('incomplete'); expect(failed.reason).toBe('Missing experts');
  expect(selectableAssets([valid, failed])).toEqual([valid]);
  expect(mapAsset({ path: 'C:\\models\\tiny-a\\', ok: true, detail: '', model_id: 'edge0_fixture' }).label).toBe('edge0_fixture · tiny-a');
  expect(mapAsset({ path: 'c', ok: false, detail: '', model_id: null }).reason).toBeTruthy();
});
const event = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
it('accumulates fragmented SSE content, reasoning and final engine metrics', () => {
  const accumulator = new SseAccumulator();
  const data = ': keepalive\n\n' + event({ choices: [{ delta: { reasoning_content: 'think' } }], metrics: { first_token_seconds: 999 } }) + event({ choices: [{ delta: { content: 'one' } }] }) + event({ choices: [{ delta: { content: ' two' } }] }) + event({ choices: [], usage: { completion_tokens: 32, prompt_tokens: 120, prompt_tokens_details: { cached_tokens: 20 } }, metrics: { first_token_seconds: 2.5, decode_tokens_per_second: 12.25, memory_peak_bytes: 123500000 } }) + 'data: [DONE]\n\n';
  for (const char of data) accumulator.push(char);
  expect(accumulator.message.content).toBe('one two'); expect(accumulator.message.reasoning_content).toBe('think'); expect(accumulator.done).toBe(true);
  expect(formatMetrics(accumulator.message.metrics!)).toBe(metricsCases[0].expected);
});
it('does not manufacture missing metrics or accept intermediate metric fields', () => {
  const accumulator = new SseAccumulator();
  accumulator.push(event({ choices: [{ delta: { content: 'x' } }], metrics: { first_token_seconds: 10 } }));
  expect(accumulator.message.metrics).toBeUndefined();
  accumulator.push(event({ choices: [], usage: { completion_tokens: 2 }, metrics: { first_token_seconds: 2.5, input_tokens: 999, reused_tokens: 0 } }));
  expect(accumulator.message.metrics).toEqual({ generated_count: 2, first_token_seconds: 2.5 });
  expect(formatMetrics(accumulator.message.metrics!)).toBe('2 tokens · TTFT 2.50 s');
});
it('handles CRLF boundaries split across network chunks', () => {
  const accumulator = new SseAccumulator(); const data = event({ choices: [{ delta: { content: 'value' } }] }).replaceAll('\n', '\r\n');
  for (const char of data) accumulator.push(char);
  expect(accumulator.message.content).toBe('value');
});
it('propagates SSE errors', () => expect(() => new SseAccumulator().push(event({ error: { message: 'bad input' } }))).toThrow('bad input'));
it('retries 429 with exponential delays and hides a transient busy response', async () => {
  const fetcher = vi.fn().mockResolvedValueOnce(new Response('', { status: 429 })).mockResolvedValueOnce(new Response('', { status: 429 })).mockResolvedValue(new Response('ok'));
  const wait = vi.fn().mockResolvedValue(undefined), signal = new AbortController().signal;
  const response = await fetchWithRetry(fetcher, 'http://127.0.0.1:1', {}, signal, wait);
  expect(await response.text()).toBe('ok'); expect(wait.mock.calls.map(call => call[0])).toEqual([200, 400]);
});
it('caps retry waiting below ten seconds', async () => {
  const fetcher = vi.fn().mockResolvedValue(new Response('', { status: 429 })), wait = vi.fn().mockResolvedValue(undefined);
  await expect(fetchWithRetry(fetcher, 'http://127.0.0.1:1', {}, new AbortController().signal, wait)).rejects.toThrow('still stopping');
  expect(wait.mock.calls.reduce((sum, call) => sum + call[0], 0)).toBeLessThanOrEqual(10000);
});
it('cancels before requesting and while waiting for a busy service', async () => {
  const controller = new AbortController(); controller.abort(); const fetcher = vi.fn();
  await expect(fetchWithRetry(fetcher, 'http://127.0.0.1:1', {}, controller.signal)).rejects.toThrow(); expect(fetcher).not.toHaveBeenCalled();
  const pending = new AbortController(); const request = fetchWithRetry(async () => new Response('', { status: 429 }), 'http://127.0.0.1:1', {}, pending.signal);
  pending.abort(); await expect(request).rejects.toThrow();
});
it('streams decoded UTF-8 and rejects truncated completion', async () => {
  const bytes = new TextEncoder().encode(event({ choices: [{ delta: { content: '\u03b1' } }] }) + 'data: [DONE]\n\n');
  const update = vi.fn(); const response = () => new Response(new ReadableStream({ start(controller) { for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); controller.close(); } }));
  await streamChat(async () => response(), 'http://127.0.0.1:1', buildRequest([], false, 'model'), new AbortController().signal, update);
  expect(update.mock.lastCall?.[0].content).toBe('\u03b1');
  await expect(streamChat(async () => new Response(event({ choices: [{ delta: { content: 'partial' } }] })), 'http://127.0.0.1:1', buildRequest([], false, 'model'), new AbortController().signal, update)).rejects.toThrow('closed before completion');
});

it('renders Markdown structure while retaining math as literal code', () => {
  const markdown = '# Title\n\n- **bold** and *italic*\n- [link](https://example.com)\n\n```js\nconst x = 1;\n```\n\nUse $x_i * y$ here.\n\nBefore \\[x^2\\] after.';
  const html = renderToStaticMarkup(createElement(Markdown, { children: markdown }));
  for (const tag of ['h1', 'ul', 'li', 'strong', 'em', 'a', 'pre', 'code']) expect(html).toContain('<' + tag);
  expect(html).toContain('$x_i * y$'); expect(html).toContain('\\[x^2\\]');
  expect(html).not.toContain('katex');
});

it('keeps a long generation out of the prefill numerator and requires both usage counts', () => {
  const accumulator = new SseAccumulator();
  accumulator.push(event({ choices: [], usage: { completion_tokens: 512, prompt_tokens: 19, prompt_tokens_details: { cached_tokens: 0 } }, metrics: { first_token_seconds: 2.13 } }));
  expect(formatMetrics(accumulator.message.metrics!)).toBe('512 tokens · TTFT 2.13 s · prefill 8.9 tok/s');
  const missingReuse = new SseAccumulator();
  missingReuse.push(event({ choices: [], usage: { prompt_tokens: 19 }, metrics: { first_token_seconds: 2.13 } }));
  expect(formatMetrics(missingReuse.message.metrics!)).toBe('TTFT 2.13 s');
});

for (const thinking of [false, true]) it(`normal HTTP request changes only temperature with thinking ${thinking}`, async () => {
  const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => new Response('data: [DONE]\n\n'));
  const history = [{ role: 'user' as const, content: 'Give a detailed answer' }];
  await streamChat(fetcher, 'http://127.0.0.1:1234', buildRequest(history, thinking, 'edge0_35b'), new AbortController().signal, () => {});
  expect(fetcher).toHaveBeenCalledOnce();
  const [url, init] = fetcher.mock.calls[0];
  expect(url).toBe('http://127.0.0.1:1234/v1/chat/completions'); expect(init?.method).toBe('POST');
  const body = JSON.parse(String(init?.body));
  expect(Object.keys(body).sort()).toEqual(['chat_template_kwargs', 'messages', 'model', 'stream', 'stream_options', 'temperature']);
  for (const parameter of ['max_tokens', 'max_completion_tokens', 'top_p', 'top_k', 'seed', 'stop', 'presence_penalty', 'frequency_penalty', 'repetition_penalty']) expect(body).not.toHaveProperty(parameter);
  expect(body.temperature).toBe(0.2);
  expect(body.messages).toEqual([{ role: 'system', content: shared.system_prompt }, ...history]);
  expect(body.chat_template_kwargs).toEqual({ enable_thinking: thinking });
});
