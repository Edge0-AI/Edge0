import { expect, it } from 'vitest';
import { loadStartup } from './startup';
import { defaultSettings } from './settings';
import { newConversation } from './history';
import { emptyTotals } from './totals';

function readers() {
  const conversation = newConversation('saved prompt');
  return {
    settings: async () => ({ ...defaultSettings, theme: 'dark' as const, assets: ['/models'] }),
    history: async () => [{ id: conversation.id, conversation, error: null }],
    totals: async () => emptyTotals(),
    token: async () => 'saved-token',
  };
}
it('retains settings, history and token when the optional totals file is unreadable', async () => {
  const failure = new Error('Cannot read saved totals');
  const state = await loadStartup({ ...readers(), totals: async () => { throw failure; } });
  expect(state.settings.assets).toEqual(['/models']);
  expect(state.settings.theme).toBe('dark');
  expect(state.history[0].conversation?.systemPrompt).toBe('saved prompt');
  expect(state.token).toBe('saved-token');
  expect(state.totals).toEqual(emptyTotals());
  expect(state.errors).toEqual([failure]);
});
it('retains independently readable archives and totals when settings and token fail', async () => {
  const failures = [new Error('Cannot read settings'), new Error('Cannot read API token')];
  const state = await loadStartup({ ...readers(), settings: async () => { throw failures[0]; }, token: async () => { throw failures[1]; } });
  expect(state.settings).toEqual(defaultSettings);
  expect(state.history).toHaveLength(1);
  expect(state.totals).toEqual(emptyTotals());
  expect(state.errors).toEqual(failures);
});
it('keeps successful startup values without reporting errors', async () => {
  const state = await loadStartup(readers());
  expect(state.errors).toEqual([]);
  expect(state.settings.theme).toBe('dark');
  expect(state.history).toHaveLength(1);
  expect(state.token).toBe('saved-token');
});
