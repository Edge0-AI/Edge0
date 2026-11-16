import { expect, it, vi } from 'vitest';
import { buildRequest, SseAccumulator, streamChat } from './chat';

const event = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
const completed = event({ choices: [{ delta: { content: 'answer' } }] }) +
  event({ choices: [], usage: { completion_tokens: 3, prompt_tokens: 7 } }) + 'data: [DONE]\n\n';

it.each([false, true])('ignores data after completion in the same chunk: %s', sameChunk => {
  const accumulator = new SseAccumulator();
  const trailing = event({ choices: [{ delta: { content: 'too late' } }] }) + 'data: invalid JSON\n\n';
  accumulator.push(completed + (sameChunk ? trailing : ''));
  if (!sameChunk) accumulator.push(trailing, true);
  expect(accumulator.done).toBe(true);
  expect(accumulator.message.content).toBe('answer');
  expect(accumulator.message.metrics).toEqual({ generated_count: 3, input_tokens: 7 });
});

it.each([false, true])('finishes and cancels an open response at DONE, fragmented: %s', async fragmented => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const cancel = vi.fn();
  const bytes = new TextEncoder().encode(completed);
  const body = new ReadableStream<Uint8Array>({
    start(source) {
      controller = source;
      for (const chunk of fragmented ? Array.from(bytes, byte => new Uint8Array([byte])) : [bytes]) source.enqueue(chunk);
      // The transport remains open after the protocol's terminal event.
    },
    cancel,
  });
  const update = vi.fn();
  const running = streamChat(async () => new Response(body), 'http://127.0.0.1:1',
    buildRequest([], false, 'model'), new AbortController().signal, update);
  try {
    await vi.waitFor(() => expect(cancel).toHaveBeenCalledOnce(), { timeout: 500, interval: 5 });
    await running;
    expect(body.locked).toBe(false);
    expect(update.mock.lastCall?.[0].content).toBe('answer');
    expect(update.mock.lastCall?.[0].metrics).toEqual({ generated_count: 3, input_tokens: 7 });
  } finally {
    if (!cancel.mock.calls.length) controller.close();
    await running;
  }
});
