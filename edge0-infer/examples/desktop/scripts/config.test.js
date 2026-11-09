import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import { defaultIdentifier, resolveIdentifier } from './config.mjs';
it('uses the tracked fixed identifier without an environment override', () => {
  expect(resolveIdentifier({})).toBe('dev.edge0.example.desktop');
  expect(JSON.parse(readFileSync(new URL('../src-tauri/tauri.conf.json', import.meta.url), 'utf8')).identifier).toBe(defaultIdentifier);
});
it('accepts an optional application identifier override', () => expect(resolveIdentifier({ MOE_DESKTOP_EXAMPLE_APP_ID: 'dev.example.test' })).toBe('dev.example.test'));
it('rejects invalid overrides before launching the build', () => expect(() => resolveIdentifier({ MOE_DESKTOP_EXAMPLE_APP_ID: 'invalid identifier' })).toThrow('reverse-DNS'));
