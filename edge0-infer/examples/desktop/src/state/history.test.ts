import { expect, it } from 'vitest';
import { finishGeneration, newConversation, type Conversation } from './history';

const original: Conversation = { ...newConversation(), updatedAt: 123, messages: [
  { role: 'user', content: 'question' },
  { role: 'assistant', content: 'saved answer', reasoning_content: 'saved reasoning', metrics: { generated_count: 9 } },
] };
const started: Conversation = { ...original, messages: [original.messages[0], { role: 'assistant', content: '' }] };
it.each([{ failed: true, stopped: false }, { failed: false, stopped: true }])('restores the existing answer after an empty replacement: %j', result => {
  expect(finishGeneration(original, started, { ...result, replacing: true })).toBe(original);
});
it('also restores the original branch when editing an earlier prompt produces no response', () => {
  const edited: Conversation = { ...started, messages: [{ role: 'user', content: 'edited question' }, started.messages[1]] };
  expect(finishGeneration(original, edited, { failed: true, stopped: false, replacing: true })).toBe(original);
});
it('retains a partial replacement, including reasoning-only output, while clearing failed metrics', () => {
  const partial: Conversation = { ...started, messages: [started.messages[0], { role: 'assistant', content: '', reasoning_content: 'partial thought', metrics: { generated_count: 2 } }] };
  const next = finishGeneration(original, partial, { failed: true, stopped: false, replacing: true });
  expect(next.messages[1].reasoning_content).toBe('partial thought');
  expect(next.messages[1].metrics).toBeUndefined();
});
it('keeps a successful replacement and a failed new user turn', () => {
  const success: Conversation = { ...started, messages: [started.messages[0], { role: 'assistant', content: 'new answer', metrics: { generated_count: 2 } }] };
  expect(finishGeneration(original, success, { failed: false, stopped: false, replacing: true }).messages[1]).toEqual({ ...success.messages[1], stopped: false });
  expect(finishGeneration(original, started, { failed: true, stopped: false, replacing: false }).messages[1].content).toBe('');
});
