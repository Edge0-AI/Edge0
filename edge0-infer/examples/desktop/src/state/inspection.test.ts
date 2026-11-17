import { expect, it, vi } from 'vitest';
import { AssetInspections } from './inspection';
import { defaultSettings } from './settings';
import type { Inspection } from '../status';

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
const oldAssets: Inspection[] = [{ path: '/old/tiny', location: '/old', ok: true, detail: '', model_id: 'tiny' }];
it('does not restore a removed location when an earlier scan completes', async () => {
  let current = { ...defaultSettings, assets: ['/old'] };
  const pending = deferred<Inspection[]>(), adopt = vi.fn();
  const scans = new AssetInspections(() => current, () => pending.promise, adopt);
  const result = scans.inspect();
  current = { ...current, assets: [] };
  pending.resolve(oldAssets); await result;
  expect(adopt).not.toHaveBeenCalled();
});
it('keeps the newer refresh when same-settings scans complete out of order', async () => {
  const pending = [deferred<Inspection[]>(), deferred<Inspection[]>()], adopt = vi.fn();
  const scan = vi.fn().mockReturnValueOnce(pending[0].promise).mockReturnValueOnce(pending[1].promise);
  const scans = new AssetInspections(() => defaultSettings, scan, adopt);
  const first = scans.inspect(), second = scans.inspect();
  pending[1].resolve([]); await second;
  pending[0].resolve(oldAssets); await first;
  expect(adopt.mock.calls).toEqual([[[]]]);
});
it('invalidates results from a previous executable but accepts unrelated appearance edits', async () => {
  let current = { ...defaultSettings, executable: '/old-engine' };
  const pending = deferred<Inspection[]>(), adopt = vi.fn();
  const scans = new AssetInspections(() => current, () => pending.promise, adopt);
  const old = scans.inspect(); current = { ...current, executable: '/new-engine' };
  pending.resolve(oldAssets); await old; expect(adopt).not.toHaveBeenCalled();
  const latest = scans.inspect(); current = { ...current, theme: 'dark' };
  await latest; expect(adopt).toHaveBeenCalledWith(oldAssets);
});
