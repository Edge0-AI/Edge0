// The Current-request card while a delegated model generates. Its /status counts what the proxy forwarded: a
// streaming request has a phase, a fragment count and a first-fragment time, while the prompt counts stay null
// until the engine reports them at the end; a non-streaming request has no phase at all. The card must show what
// is known and never print the word "null".
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import realStatus from './real-status.json';
import { Status } from './pages/Status';
import { aggregateTotals, emptyTotals } from './state/totals';
import type { EngineStatus } from './status';

const base = realStatus as unknown as EngineStatus;

function render(current: unknown): string {
  const status = { ...base, current } as EngineStatus;
  return renderToStaticMarkup(
    createElement(Status, { status, live: true, allTime: aggregateTotals(emptyTotals()), reset: () => {} }),
  );
}

// The fields a delegated model's `current` carries (first_token_seconds only once the first fragment arrived).
const running = {
  endpoint: '/v1/chat/completions', source: '127.0.0.1', started_steady_seconds: 0, elapsed_seconds: 4.2,
  input_tokens: null, reused_tokens: null, computed_tokens: null, pending_tokens: null,
};

it('shows the live decode count without inventing the prompt size', () => {
  const html = render({ ...running, phase: 'decode', generated_tokens: 141, first_token_seconds: 0.35 });
  expect(html).toContain('<b>141</b> tokens');
  expect(html).toContain('Decoding');
  expect(html).toContain('3.9 s');  // decoding for elapsed - first token
  expect(html).toContain('reported when the turn ends');
  expect(html).not.toContain('null');
});

it('shows the prompt being read while prefill runs', () => {
  const html = render({ ...running, phase: 'prefill', generated_tokens: null });
  expect(html).toContain('Reading the prompt');
  expect(html).toContain('4.2 s');
  expect(html).not.toContain('null');
});

it('shows the input count once the stream carries it (Messages)', () => {
  const html = render({ ...running, endpoint: '/v1/messages', phase: 'prefill', generated_tokens: null, input_tokens: 62 });
  expect(html).toContain('Computing <b>62</b> tokens');
  expect(html).not.toContain('null');
});

it('shows only "In progress" and the elapsed time when the phase is unknown (non-streaming)', () => {
  const html = render({ ...running, phase: null, generated_tokens: null });
  expect(html).toContain('In progress');
  expect(html).toContain('4.2 s');
  expect(html).not.toContain('Prefill');
  expect(html).not.toContain('Decode');
  expect(html).not.toContain('null');
});
