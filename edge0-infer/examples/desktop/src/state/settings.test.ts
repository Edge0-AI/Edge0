import { expect, it } from 'vitest';
import { defaultSettings, engineSettingsChanged, type Settings } from './settings';

it.each<Partial<Settings>>([{ executable: '/other/edge0-infer' }, { device: 'vulkan' }, { prefill_device: 'metal' }, { port: 9000 }])('requires a new service for changed engine options: %j', change => {
  expect(engineSettingsChanged(defaultSettings, { ...defaultSettings, ...change })).toBe(true);
});
it('does not invalidate the model for appearance, prompt, sampling, or asset-list changes', () => {
  expect(engineSettingsChanged(defaultSettings, { ...defaultSettings, theme: 'dark', system_prompt: 'new prompt', generation_style: 'creative', assets: ['/new'] })).toBe(false);
  expect(engineSettingsChanged(defaultSettings, { ...defaultSettings })).toBe(false);
});
