import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { Chat } from './Chat';
import { newConversation } from '../state/history';

const noop = () => {};
function render(loaded: boolean, messages = newConversation().messages) {
  return renderToStaticMarkup(createElement(Chat, {
    conversation: { ...newConversation(), messages }, phase: 'idle', loaded, model: 'tiny',
    supported: false, draft: '', setDraft: noop, send: noop, stop: noop, newThinking: noop,
    reasoningExpanded: noop, edit: noop, redo: noop, copy: noop, models: noop,
  }));
}
it('keeps archived messages and copy actions available without a running model', () => {
  const html = render(false, [{ role: 'user', content: 'Saved question' }, { role: 'assistant', content: 'Saved answer' }]);
  expect(html).toContain('Saved question');
  expect(html).toContain('Saved answer');
  expect(html).toContain('title="Copy"');
  expect(html).toContain('title="Regenerate" disabled=""');
  expect(html).toContain('title="Edit and resend" disabled=""');
  expect(html).not.toContain('aria-label="Message"');
});
it('still guides empty unloaded chats and enables the composer only after loading', () => {
  expect(render(false)).toContain('Load a model to start a conversation.');
  expect(render(true)).toContain('aria-label="Message"');
});
