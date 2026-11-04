import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { GenerationButton } from './ChatControls';
import { buildRequest } from './chat';
import { canChangeThinking, conversationReducer, emptyConversation, validateServerPort } from './conversation';

it('locks thinking for the entire conversation and unlocks it only in a new chat', () => {
  expect(canChangeThinking(true, emptyConversation)).toBe(true);
  expect(canChangeThinking(false, emptyConversation)).toBe(false);
  const generating = conversationReducer(emptyConversation, { type: 'start', user: 'first' });
  expect(canChangeThinking(true, generating)).toBe(false);
  const completed = conversationReducer(generating, { type: 'finish' });
  expect(canChangeThinking(true, completed)).toBe(false);
  const next = conversationReducer(completed, { type: 'reset' });
  expect(canChangeThinking(true, next)).toBe(true);
  expect(buildRequest([{ role: 'user', content: 'new' }], false, 'model').messages).toHaveLength(2);
});
it('enters Stopping immediately, freezes partial output, marks the reply and permits another turn', () => {
  let state = conversationReducer(emptyConversation, { type: 'start', user: 'long answer' });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'partial', reasoning_content: 'thought', metrics: { generated_count: 2 } } });
  state = conversationReducer(state, { type: 'stop' });
  expect(state.phase).toBe('stopping');
  expect(conversationReducer(state, { type: 'stop' })).toBe(state);
  expect(conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'late data' } })).toBe(state);
  const stopping = renderToStaticMarkup(createElement(GenerationButton, { phase: state.phase, canSend: true, stop() {} }));
  expect(stopping).toContain('disabled'); expect(stopping).toContain('Stopping…'); expect(stopping).toContain('class="stop-spinner"'); expect(stopping).toContain('aria-hidden="true"'); expect(stopping).not.toContain('<progress');
  state = conversationReducer(state, { type: 'finish', failed: true });
  expect(state.phase).toBe('idle');
  expect(state.messages.at(-1)).toEqual({ role: 'assistant', content: 'partial', reasoning_content: 'thought', stopped: true, metrics: undefined });
  const button = renderToStaticMarkup(createElement(GenerationButton, { phase: state.phase, canSend: true, stop() {} }));
  expect(button).toContain('Send'); expect(button).not.toContain('Stopping'); expect(button).not.toContain('disabled');
  expect(buildRequest(state.messages, true, 'model').messages.at(-1)).toEqual({ role: 'assistant', content: 'partial', reasoning_content: 'thought' });
  state = conversationReducer(state, { type: 'start', user: 'next' });
  expect(state.phase).toBe('generating'); expect(state.messages[1].stopped).toBe(true);
});
it('retains normal completion metrics and does not label an unexpected error as user-stopped', () => {
  let state = conversationReducer(emptyConversation, { type: 'start', user: 'test' });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'reply', metrics: { generated_count: 4 } } });
  const completed = conversationReducer(state, { type: 'finish' });
  expect(completed.messages.at(-1)?.metrics).toEqual({ generated_count: 4 });
  expect(completed.messages.at(-1)?.stopped).toBe(false);
  const failed = conversationReducer(state, { type: 'finish', failed: true });
  expect(failed.messages.at(-1)?.metrics).toBeUndefined(); expect(failed.messages.at(-1)?.stopped).toBe(false);
  expect(conversationReducer(state, { type: 'start', user: 'duplicate' })).toBe(state);
});
describe('server port validation before saving settings', () => {
  for (const value of ['', ' ', '\t']) it(`accepts automatic port ${JSON.stringify(value)}`, () => expect(validateServerPort(value)).toEqual({ valid: true, port: null }));
  for (const value of ['1', '8000', '65535']) it(`accepts explicit port ${value}`, () => expect(validateServerPort(value)).toEqual({ valid: true, port: Number(value) }));
  for (const value of ['0', '-1', '65536', '1.5', '1e3', '+80', 'abc', '9999999999999999999999999']) it(`rejects ${value} for saving`, () => expect(validateServerPort(value).valid).toBe(false));
});
