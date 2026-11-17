// The metrics a delegated model's stream yields: edge0-strata sends ONE final chunk that carries the finishing
// choice, `usage` and its own `timings`, where Edge0's engine sends a separate choices-less `metrics` event.
// The desktop app must read both, or a delegated model shows no prefill/decode figures.
import { expect, it } from 'vitest';
import { formatMetrics, SseAccumulator } from './chat';

it('reads the figures from a delegated engine final chunk (usage + timings, with a finishing choice)', () => {
  const accumulator = new SseAccumulator();
  accumulator.push('data: ' + JSON.stringify({ choices: [{ index: 0, delta: { content: 'one' }, finish_reason: null }] }) + '\n\n');
  accumulator.push('data: ' + JSON.stringify({
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    usage: { prompt_tokens: 120, completion_tokens: 32, total_tokens: 152, prompt_tokens_details: { cached_tokens: 20 } },
    timings: { cache_n: 20, prompt_n: 120, prompt_ms: 2000, prompt_per_second: 50, predicted_n: 32, predicted_ms: 2560, predicted_per_second: 12.5 },
  }) + '\n\n');
  accumulator.push('data: [DONE]\n\n');
  const text = formatMetrics(accumulator.message.metrics!);
  expect(text).toBe('32 tokens · TTFT 2.00 s · prefill 50.0 tok/s · decode 12.5 tok/s');
  expect(accumulator.done).toBe(true);
});

it('still prefers metrics when the engine sends them', () => {
  const accumulator = new SseAccumulator();
  accumulator.push('data: ' + JSON.stringify({
    choices: [], usage: { prompt_tokens: 120, completion_tokens: 32, prompt_tokens_details: { cached_tokens: 20 } },
    metrics: { first_token_seconds: 4, decode_tokens_per_second: 9 },
    timings: { prompt_ms: 2000, predicted_per_second: 12.5 },
  }) + '\n\n');
  const metrics = accumulator.message.metrics!;
  expect(metrics.first_token_seconds).toBe(4);
  expect(metrics.decode_tokens_per_second).toBe(9);
});
