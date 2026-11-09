import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { mapAsset, buildRequest } from './chat';
import { conversationReducer, emptyConversation, scrollContent } from './conversation';
import { currentPage, DeleteConversation, ModelPicker, OtherLocation, PageHeader, ThinkingBlock, ThinkingSwitch, Workspace } from './Workspace';
const noop = () => {};
const good = mapAsset({ path: '/models/35b', ok: true, detail: '', model_id: 'edge0_35b' });
const bad = mapAsset({ path: '/models/incomplete', ok: false, detail: 'Missing weights', model_id: null });
function picker(assets = [good], activePath?: string) {
  return createElement(ModelPicker, { assets, directories: ['/models/35b'], disabled: false, initializing: false, activePath, load: noop, resume: noop,
    children: createElement(OtherLocation, { children: createElement('input', { 'aria-label': 'Asset directory' }) }) });
}
it('shows only model selection before a model is loaded, then enters chat after success', () => {
  const chat = createElement('section', null, 'conversation', createElement('textarea', { 'aria-label': 'Message' }));
  expect(currentPage(false, 'chat')).toBe('models');
  const initial = renderToStaticMarkup(createElement(Workspace, { loaded: false, page: 'chat', models: picker(), chat }));
  expect(initial).toContain('class="models-page"'); expect(initial).toContain('>Load<');
  expect(initial).not.toContain('aria-label="Message"'); expect(initial).not.toContain('conversation');
  const loaded = renderToStaticMarkup(createElement(Workspace, { loaded: true, page: 'chat', models: picker(), chat }));
  expect(loaded).toContain('aria-label="Message"'); expect(loaded).not.toContain('Choose a model');
});
it('keeps Models navigation separate from conversation history and permits returning to the loaded chat', () => {
  let state = conversationReducer(emptyConversation, { type: 'start', user: 'remember this' });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'remembered' } });
  state = conversationReducer(state, { type: 'finish' });
  const history = state.messages;
  const chat = createElement('section', null, history.map(message => message.content).join(' '));
  const models = renderToStaticMarkup(createElement(Workspace, { loaded: true, page: 'models', models: picker([good], good.path), chat }));
  expect(models).toContain('Return to chat'); expect(models).not.toContain('remembered');
  expect(renderToStaticMarkup(createElement(Workspace, { loaded: true, page: 'chat', models: picker(), chat }))).toContain('remembered');
  expect(state.messages).toBe(history);
});
it('renders model rows and confines preparation instructions to the empty available list', () => {
  const available = renderToStaticMarkup(picker([good, bad]));
  expect(available).toContain('edge0_35b · 35b'); expect(available).toContain('title="/models/35b"');
  expect(available).toContain('Unavailable'); expect(available).toContain('Missing weights');
  expect(available.match(/>Load</g)).toHaveLength(1);
  expect(available).not.toContain('Prepare assets');
  const empty = renderToStaticMarkup(picker([bad]));
  expect(empty).toContain('Prepare assets'); expect(empty).toContain('tools/assets'); expect(empty).toContain('README.md'); expect(empty).toContain('/models/35b');
  expect(empty).not.toContain('>Load<');
  expect(empty).toContain('<details class="other-location"><summary>Other location…</summary>');
  expect(empty).not.toContain('<details class="other-location" open');
});
it('defaults the independent thinking block to expanded and hides all reasoning when thinking is off', () => {
  const shown = renderToStaticMarkup(createElement(ThinkingBlock, { enabled: true, content: 'private reasoning', setExpanded: noop }));
  expect(shown).toContain('class="reasoning-block" open'); expect(shown).toContain('private reasoning');
  expect(renderToStaticMarkup(createElement(ThinkingBlock, { enabled: false, content: 'private reasoning', setExpanded: noop }))).toBe('');
  expect(renderToStaticMarkup(createElement(ThinkingBlock, { enabled: true, content: '', setExpanded: noop }))).toBe('');
});
it('preserves manual collapse through streaming updates and completion without changing request history', () => {
  let state = conversationReducer(emptyConversation, { type: 'start', user: 'question' });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: '', reasoning_content: 'thinking' } });
  state = conversationReducer(state, { type: 'reasoning_expanded', index: 1, expanded: false });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'answer', reasoning_content: 'thinking more' } });
  state = conversationReducer(state, { type: 'finish' });
  expect(state.messages[1].reasoningExpanded).toBe(false);
  const collapsed = renderToStaticMarkup(createElement(ThinkingBlock, { enabled: true, content: state.messages[1].reasoning_content!, expanded: state.messages[1].reasoningExpanded, setExpanded: noop }));
  expect(collapsed).not.toContain(' open');
  expect(buildRequest(state.messages, true, 'model').messages[2]).toEqual({ role: 'assistant', content: 'answer', reasoning_content: 'thinking more' });
  state = conversationReducer(state, { type: 'reasoning_expanded', index: 1, expanded: true });
  expect(state.messages[1].reasoningExpanded).toBe(true);
});
for (const disabled of [false, true]) for (const checked of [false, true]) {
  it(`shows readable ${checked ? 'On' : 'Off'} and its color class when ${disabled ? 'locked' : 'editable'}`, () => {
    const html = renderToStaticMarkup(createElement(ThinkingSwitch, { checked, disabled, supported: true, locked: disabled, change: noop }));
    expect(html).toContain(`aria-checked="${checked}"`);
    expect(html).toContain(`switch-track ${checked ? 'is-on' : 'is-off'}`);
    expect(html).toContain(`<span class="switch-state">${checked ? 'On' : 'Off'}</span>`);
    expect(html.includes('is-locked')).toBe(disabled);
    expect(html.includes('disabled=""')).toBe(disabled);
    expect(html.includes('Applies to a new chat')).toBe(disabled);
  });
}
it('places page titles between leading and trailing actions, with loaded-only back navigation', () => {
  const header = (chat: boolean, loaded: boolean) => renderToStaticMarkup(createElement(PageHeader, { chat, loaded, disabled: false, models: noop, newChat: noop, resume: noop, refresh: noop }));
  const chat = header(true, true);
  expect(chat).toMatch(/header-leading.*>Models<.*<h1>Edge0 Chat<.*header-trailing.*>New chat</);
  const chooser = header(false, false);
  expect(chooser).toContain('<h1>Choose a model</h1>'); expect(chooser).toContain('>Refresh<'); expect(chooser).not.toContain('Back to chat');
  expect(header(false, true)).toMatch(/header-leading.*>Back to chat<.*<h1>Choose a model<.*header-trailing.*>Refresh</);
});

it('auto-scroll dependencies ignore disclosure and completion metadata while following actual stream text', () => {
  const changes = (before: readonly unknown[], after: readonly unknown[]) => after.some((value, index) => !Object.is(value, before[index]));
  let state = conversationReducer(emptyConversation, { type: 'start', user: 'question' });
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'answer', reasoning_content: 'thought' } });
  const initial = scrollContent(state.messages);
  state = conversationReducer(state, { type: 'reasoning_expanded', index: 1, expanded: false });
  expect(changes(initial, scrollContent(state.messages))).toBe(false);
  state = conversationReducer(state, { type: 'reasoning_expanded', index: 1, expanded: true });
  expect(changes(initial, scrollContent(state.messages))).toBe(false);
  state = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'answer', reasoning_content: 'thought', metrics: { generated_count: 2 } } });
  expect(changes(initial, scrollContent(state.messages))).toBe(false);
  const withText = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'answer extended', reasoning_content: 'thought' } });
  expect(changes(initial, scrollContent(withText.messages))).toBe(true);
  const withReasoning = conversationReducer(state, { type: 'fragment', message: { role: 'assistant', content: 'answer', reasoning_content: 'thought extended' } });
  expect(changes(initial, scrollContent(withReasoning.messages))).toBe(true);
  state = conversationReducer(state, { type: 'finish' });
  expect(changes(initial, scrollContent(state.messages))).toBe(false);
  expect(changes(initial, scrollContent([...state.messages, { role: 'assistant', content: 'answer', reasoning_content: 'thought' }]))).toBe(true);
});
it('deletes a conversation from a trash button, not a more-actions icon', () => {
  const html = renderToStaticMarkup(createElement(DeleteConversation, { disabled: false, onDelete: noop }));
  expect(html).toContain('aria-label="Delete conversation"'); expect(html).toContain('M10 11v6M14 11v6');
  expect(html).not.toContain('r="1"');
});
