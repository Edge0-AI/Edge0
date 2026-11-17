import { expect, it } from 'vitest';
import { refreshDownloadedAssets } from './main';
import { defaultSettings, type Settings } from './state/settings';

// save_settings keeps download_directory and partial_downloads from disk and writes every other
// field from the object the page sends. assets are one of those written fields.
function savedSettings(incoming: Settings, onDisk: Settings): Settings {
  return { ...incoming, download_directory: onDisk.download_directory, partial_downloads: onDisk.partial_downloads };
}

it('inspects the finished download directory and keeps it when settings are saved afterwards', async () => {
  const directory = 'D:/models/Edge0-120B-q3k';
  const unsaved = 'draft system prompt';
  let current: Settings = { ...defaultSettings, theme: 'dark', system_prompt: unsaved, assets: [], partial_downloads: [{ id: 'edge0-120b-q3k', directory: 'D:/models/Edge0-120B-q3k.partial' }] };
  const onDisk: Settings = { ...defaultSettings, assets: [directory], download_directory: 'D:/models', partial_downloads: [] };
  const inspected: Settings[] = [];
  let loads = 0;
  const load = async () => { loads += 1; return onDisk; };
  const inspect = async (settings: Settings) => { inspected.push(settings); };
  const adopt = (next: Settings) => { current = next; };

  expect(await refreshDownloadedAssets(null, 0, () => current, load, adopt, inspect)).toBeUndefined();
  expect(await refreshDownloadedAssets(0, 0, () => current, load, adopt, inspect)).toBeUndefined();
  expect(loads).toBe(0);
  expect(inspected).toEqual([]);

  const next = await refreshDownloadedAssets(0, 1, () => current, load, adopt, inspect);
  expect(loads).toBe(1);
  expect(inspected).toHaveLength(1);
  expect(inspected[0].assets).toEqual([directory]);
  expect(next).toBe(current);
  expect(current.theme).toBe('dark');
  expect(current.system_prompt).toBe(unsaved);
  expect(current.partial_downloads).toEqual([{ id: 'edge0-120b-q3k', directory: 'D:/models/Edge0-120B-q3k.partial' }]);
  expect(savedSettings(current, onDisk).assets).toEqual([directory]);
  expect(savedSettings(current, onDisk).download_directory).toBe('D:/models');
  expect(savedSettings(current, onDisk).partial_downloads).toEqual([]);
});
