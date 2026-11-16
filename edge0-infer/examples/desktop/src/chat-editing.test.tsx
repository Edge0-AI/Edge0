// @vitest-environment happy-dom
import { act, createElement, type ComponentProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { Chat } from './pages/Chat';
import type { Conversation } from './state/history';

let container: HTMLDivElement;
let root: Root;
const noop = () => {};
const conversation = (id: string, prompt: string): Conversation => ({
  id, createdAt: 1, updatedAt: 1, systemPrompt: '', thinking: false,
  messages: [{ role: 'user', content: prompt }, { role: 'assistant', content: 'reply' }],
});
const props: Omit<ComponentProps<typeof Chat>, 'conversation'> = {
  phase: 'idle', loaded: true, model: 'model', supported: false, draft: '',
  setDraft: noop, send: noop, stop: noop, newThinking: noop,
  reasoningExpanded: noop, edit: noop, redo: noop, copy: noop, models: noop,
};
async function render(value: Conversation, edit = props.edit) {
  await act(async () => root.render(createElement(Chat, { ...props, conversation: value, edit })));
}
async function editFirst() {
  await act(async () => container.querySelector<HTMLButtonElement>('[title="Edit and resend"]')!.click());
}
const editor = () => container.querySelector<HTMLTextAreaElement>('[aria-label="Edit message"]');

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  vi.spyOn(HTMLElement.prototype, 'scrollTo').mockImplementation(noop);
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});

it('does not carry an edit into another conversation at the same message index', async () => {
  await render(conversation('first', 'first question')); await editFirst();
  expect(editor()?.value).toBe('first question');
  const edit = vi.fn();
  await render(conversation('second', 'different question'), edit);
  expect(editor()).toBeNull();
  expect(container.textContent).toContain('different question');
  await editFirst(); expect(editor()?.value).toBe('different question');
  await act(async () => container.querySelector<HTMLButtonElement>('.edit-message .primary')!.click());
  expect(edit).toHaveBeenCalledExactlyOnceWith(0, 'different question');
});

it('does not resurrect an earlier edit when the first message arrives in a new chat', async () => {
  await render(conversation('first', 'first question')); await editFirst();
  await render({ ...conversation('new', ''), messages: [] });
  await render(conversation('new', 'new question'));
  expect(editor()).toBeNull();
  expect(container.textContent).toContain('new question');
});

it('preserves the active edit when the same conversation updates', async () => {
  const value = conversation('same', 'question');
  await render(value); await editFirst();
  await render({ ...value, updatedAt: 2, messages: [...value.messages, { role: 'assistant', content: 'updated answer' }] });
  expect(editor()?.value).toBe('question');
});
