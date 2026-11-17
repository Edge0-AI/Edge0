import { Children, createElement, isValidElement, type ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import { catalog, catalogFor, decimalGb, enoughSpace, fastestSource, presentPath, resumeDirectory, type JobView } from './download-logic';
import { beginDownload, DownloadDialog, openDownload, removeDownload, type DialogState, type DownloadActions } from './download-view';
import { Models } from './pages/Models';

const noop = () => undefined;
const model = catalog[0];
const q3k = catalog[2];
function job(partial: Partial<JobView> & Pick<JobView, 'id' | 'phase'>): JobView {
  return { done_bytes: 0, total_bytes: model.bytes, speed_bps: 0, eta_seconds: null, source: 'ModelScope', error: '', directory: 'C:\\models\\partial', ...partial };
}
function actions(jobs: JobView[] = [], platform = 'windows'): DownloadActions {
  return {
    platform, saveDirectory: 'C:\\Users\\you\\Edge0-models', jobs,
    measure: async () => ({ huggingface_ms: 820, modelscope_ms: 95, chosen: 'modelscope' }),
    disk: async () => ({ free_bytes: 400e9, drive: 'C:' }),
    start: async () => undefined, pause: async () => undefined, remove: async () => undefined,
    browse: async () => undefined,
  };
}
function page(download?: DownloadActions, assets: Parameters<typeof Models>[0]['assets'] = []) {
  return renderToStaticMarkup(createElement(Models, { assets, disabled: false, load: noop, unload: noop, chat: noop, refresh: noop, add: noop, remove: noop, download }));
}
function html(node: ReactNode) { return renderToStaticMarkup(createElement(() => node)); }
function buttonText(children: ReactNode): string {
  return Children.toArray(children).map(child => typeof child === 'string' ? child : isValidElement<{ children?: ReactNode }>(child) ? buttonText(child.props.children) : '').join('');
}
function buttons(node: ReactNode): { text: string; disabled?: boolean; onClick?: () => void }[] {
  return Children.toArray(node).flatMap(child => {
    if (!isValidElement<{ children?: ReactNode; disabled?: boolean; onClick?: () => void }>(child)) return [];
    const nested = buttons(child.props.children);
    return child.type === 'button' ? [{ text: buttonText(child.props.children), disabled: child.props.disabled, onClick: child.props.onClick }, ...nested] : nested;
  });
}

it('shows an available model with its catalog size and repository', () => {
  const markup = page(actions());
  expect(markup).toContain('Edge0 145B · Q3_K');
  expect(markup).toContain('105.4 GB · 93 files');
  expect(markup).toContain('Edge0/Edge0-145B-q3k');
  expect(markup).toContain('Download</button>');
  expect(markup).toContain('Each model downloads into a folder of its own.');
});

it('shows downloading, checking, paused, failed, and on this machine', () => {
  const failed = 'weights.safetensors did not match its checksum twice. Resume to try again, or choose another source.';
  const runningMarkup = page(actions([
    job({ id: model.id, phase: 'running', done_bytes: 41.2e9, total_bytes: model.bytes, speed_bps: 48e6, eta_seconds: 23 * 60 }),
    job({ id: q3k.id, phase: 'paused', done_bytes: 12.8e9, total_bytes: q3k.bytes, source: 'Hugging Face' }),
    job({ id: catalog[3].id, phase: 'failed', done_bytes: 1e9, total_bytes: catalog[3].bytes, error: failed }),
  ]));
  expect(runningMarkup).toContain('41.2 of 105.4 GB · 39% · 48 MB/s · about 23 min left · from ModelScope');
  expect(runningMarkup).toContain('Downloading');
  expect(runningMarkup).toContain('Pause</button>');
  expect(runningMarkup).toContain('Downloaded files are kept; Resume continues where it stopped.');
  expect(runningMarkup).toContain('Paused');
  expect(runningMarkup).toContain(failed);
  expect(runningMarkup).toContain('Failed');
  expect(runningMarkup).toContain('<button class="btn">Pause</button>');
  const downloads = runningMarkup.match(/<button class="btn primary"[^>]*>[\s\S]*?Download<\/button>/g) ?? [];
  const resumes = runningMarkup.match(/<button class="btn primary"[^>]*>Resume<\/button>/g) ?? [];
  expect(downloads.length).toBeGreaterThan(0);
  expect(downloads.every(button => button.includes('disabled'))).toBe(true);
  expect(resumes.every(button => button.includes('disabled'))).toBe(true);
  const checking = page(actions([job({ id: catalog[1].id, phase: 'checking' })]));
  expect(checking).toContain('checking the downloaded files and the model folder');
  expect(checking).toContain('Checking');
  const present = page(actions(), [{ path: 'D:\\models\\Edge0-120B-q3k', ok: true, detail: '', model_id: 'edge0_120b', model_name: 'edge0-120b-q3k' }]);
  expect(present).toContain('92.3 GB · on this machine');
  expect(present).toContain('D:\\models\\Edge0-120B-q3k');
  expect(present).toContain('On this machine');
  expect(present.match(/Download<\/button>/g)).toHaveLength(3);
});

it('lists only the Q3_K models on macOS and all four on Windows', () => {
  const names = ['Edge0 145B · Q3_K', 'Edge0 145B · NVFP4', 'Edge0 120B · Q3_K', 'Edge0 120B · NVFP4'];
  expect(catalogFor('macos').map(entry => entry.name)).toEqual([names[0], names[2]]);
  expect(catalogFor('windows').map(entry => entry.name)).toEqual(names);
  const macos = page(actions([], 'macos'));
  expect(macos).toContain('Each model downloads into a folder of its own.');
  expect(macos).toContain(names[0]);
  expect(macos).toContain(names[2]);
  expect(macos).not.toContain('NVFP4');
  expect(macos.match(/Download<\/button>/g)).toHaveLength(2);
  const windows = page(actions([], 'windows'));
  for (const name of names) expect(windows).toContain(name);
  expect(windows.match(/Download<\/button>/g)).toHaveLength(4);
});

it('hides the download section when no catalog entry applies, including when download props are omitted', () => {
  const linux = page(actions([], 'linux'));
  expect(linux).toContain('Prepare converted assets with the asset tools, then add their location.');
  expect(linux).not.toContain('Each model downloads into a folder');
  const omitted = page();
  expect(omitted).toContain('Converted asset directories found on this machine.');
  expect(omitted).not.toContain('Each model downloads into a folder');
});

it('uses the first-run chooser when this platform has models and none are on the machine', () => {
  const markup = page(actions());
  expect(markup).toContain('Choose a model');
  expect(markup).not.toContain('Other location');
  expect(markup).toContain('Download one below, or add a folder that already holds a model.');
  const calls: string[] = [];
  const tree = Models({ assets: [], disabled: false, load: noop, unload: noop, chat: noop, refresh: () => calls.push('refresh'), add: () => calls.push('add'), remove: noop, download: actions() });
  const head = buttons(tree).filter(button => button.text === 'Refresh' || button.text === 'Add location');
  expect(head.map(button => button.text)).toEqual(['Refresh', 'Add location']);
  head.forEach(button => button.onClick?.());
  expect(calls).toEqual(['refresh', 'add']);
});

it('shows Measuring in the dialog before the measurement returns', async () => {
  let finish: (report: { huggingface_ms: number | null; modelscope_ms: number | null; chosen: 'huggingface' | 'modelscope' | null }) => void = noop;
  const pending = { ...actions(), measure: () => new Promise<{ huggingface_ms: number | null; modelscope_ms: number | null; chosen: 'huggingface' | 'modelscope' | null }>(resolve => { finish = resolve; }) };
  let state: DialogState | null = null;
  const setDialog = (update: DialogState | null | ((current: DialogState | null) => DialogState | null)) => { state = typeof update === 'function' ? update(state) : update; };
  const opened = openDownload(pending, model.id, false, setDialog, noop);
  const shown = (current: DialogState) => html(DownloadDialog({ model, platform: 'windows', resume: current.resume, measuring: current.measuring, report: current.report, source: current.source, directory: current.directory, freeBytes: current.freeBytes, drive: current.drive, onClose: noop, onSource: noop, onMeasure: noop, onChangeDirectory: noop, onStart: noop }));
  await Promise.resolve();
  expect(state).not.toBeNull();
  expect(shown(state!)).toContain('Measuring response time…');
  finish({ huggingface_ms: null, modelscope_ms: 95, chosen: 'modelscope' });
  await opened;
  expect(state!.measuring).toBe(false);
  expect(shown(state!)).not.toContain('Measuring response time…');
});

it('disables the start button while measuring, when space is short, and when both sources are unreachable', () => {
  const common = { model, platform: 'windows', resume: false, directory: 'C:\\Users\\you\\Edge0-models', onClose: noop, onSource: noop, onMeasure: noop, onChangeDirectory: noop, onStart: noop };
  const measuring = DownloadDialog({ ...common, measuring: true, report: { huggingface_ms: null, modelscope_ms: null, chosen: null }, source: null, freeBytes: 400e9, drive: 'C:' });
  expect(html(measuring)).toContain('Measuring response time…');
  expect(buttons(measuring).find(button => button.text === 'Download')?.disabled).toBe(true);
  const low = DownloadDialog({ ...common, measuring: false, report: { huggingface_ms: 820, modelscope_ms: 95, chosen: 'modelscope' }, source: 'modelscope', freeBytes: 64e9, drive: 'C:' });
  expect(html(low)).toContain('64.0 GB free on C: · needs 105.4 GB — not enough space');
  expect(buttons(low).find(button => button.text === 'Download')?.disabled).toBe(true);
  const offline = DownloadDialog({ ...common, measuring: false, report: { huggingface_ms: null, modelscope_ms: null, chosen: null }, source: null, freeBytes: 400e9, drive: 'C:' });
  expect(html(offline)).toContain('Both sources are unreachable. Check the network, then measure again.');
  expect(buttons(offline).find(button => button.text === 'Download')?.disabled).toBe(true);
  const ready = DownloadDialog({ ...common, measuring: false, report: { huggingface_ms: 820, modelscope_ms: 95, chosen: 'modelscope' }, source: 'huggingface', freeBytes: model.bytes, drive: 'C:' });
  expect(html(ready)).toContain('Fastest');
  expect(buttons(ready).find(button => button.text === 'Download')?.disabled).toBe(false);
  const resume = DownloadDialog({ ...common, resume: true, measuring: false, report: { huggingface_ms: 820, modelscope_ms: 95, chosen: 'modelscope' }, source: 'modelscope', freeBytes: 400e9, drive: 'C:', job: job({ id: model.id, phase: 'paused', done_bytes: 12.8e9 }) });
  expect(html(resume)).toContain('12.8 GB already downloaded');
  expect(buttons(resume).find(button => button.text.includes('Change'))?.disabled).toBe(true);
});

it('marks a model present only from an ok inspection with the same model_name', () => {
  expect(presentPath(q3k, [{ path: 'D:\\models\\Edge0-120B-q3k', ok: true, detail: '', model_id: 'edge0_120b', model_name: 'edge0-120b-q3k' }])).toBe('D:\\models\\Edge0-120B-q3k');
  expect(presentPath(q3k, [{ path: 'bad', ok: false, detail: '', model_id: 'edge0_120b', model_name: 'edge0-120b-q3k' }])).toBeUndefined();
  expect(presentPath(q3k, [{ path: 'other', ok: true, detail: '', model_id: 'edge0_120b', model_name: 'edge0-120b-fp4' }])).toBeUndefined();
  expect(fastestSource(200, 200)).toBe('modelscope');
  expect(decimalGb(model.bytes)).toBe('105.4');
});

it('takes a full fit as enough space and the first matching copy, without falling back to the adapter', () => {
  expect(enoughSpace(10, 10)).toBe(true);
  expect(enoughSpace(9, 10)).toBe(false);
  const copy = (path: string, model_name?: string, model_id = 'edge0_120b') => ({ path, ok: true, detail: '', model_id, model_name });
  expect(presentPath(q3k, [copy('first', 'edge0-120b-q3k'), copy('second', 'edge0-120b-q3k')])).toBe('first');
  expect(presentPath(q3k, [copy('native', undefined, 'edge0-120b-q3k')])).toBeUndefined();
});

it('finds the folder a paused download is in', () => {
  expect(resumeDirectory('/X/Edge0-120B-q3k.partial')).toBe('/X');
  expect(resumeDirectory('/X/Edge0-120B-q3k.partial/')).toBe('/X');
  expect(resumeDirectory('D:\\models\\Edge0-120B-q3k.partial')).toBe('D:\\models');
  expect(resumeDirectory('D:\\Edge0-120B-q3k.partial')).toBe('D:\\');
  expect(resumeDirectory('/Edge0-120B-q3k.partial')).toBe('/');
  expect(resumeDirectory('Edge0-120B-q3k.partial')).toBeUndefined();
});

function dialogStore() {
  const store: { state: DialogState | null } = { state: null };
  const setDialog = (update: DialogState | null | ((current: DialogState | null) => DialogState | null)) => { store.state = typeof update === 'function' ? update(store.state) : update; };
  return { store, setDialog };
}

it('resumes in the folder of the paused download, not the current save location', async () => {
  for (const [partial, saved, expected] of [['/X/Edge0-120B-q3k.partial', '/Y', '/X'], ['D:\\old\\Edge0-120B-q3k.partial', 'C:\\Users\\you\\Edge0-models', 'D:\\old']]) {
    const asked: string[] = [];
    const started: string[] = [];
    const paused = { ...actions([job({ id: q3k.id, phase: 'paused', directory: partial })]), saveDirectory: saved, disk: async (path: string) => { asked.push(path); return { free_bytes: 400e9, drive: null }; }, start: async (_id: string, _source: string, directory: string) => { started.push(directory); } };
    const { store, setDialog } = dialogStore();
    await openDownload(paused, q3k.id, true, setDialog, noop);
    expect(store.state!.directory).toBe(expected);
    expect(asked).toEqual([expected]);
    await beginDownload(paused, store.state!, setDialog, noop, { current: false });
    expect(started).toEqual([expected]);
    // A new download still goes to the save location.
    await openDownload(paused, model.id, false, setDialog, noop);
    expect(store.state!.directory).toBe(saved);
  }
});

it('sends one start request until it returns and disables the button meanwhile', async () => {
  let finish: (failure?: Error) => void = noop;
  const calls: string[] = [];
  const failures: unknown[] = [];
  const slow = { ...actions(), start: (id: string) => { calls.push(id); return new Promise<void>((resolve, reject) => { finish = failure => failure ? reject(failure) : resolve(); }); } };
  const { store, setDialog } = dialogStore();
  await openDownload(slow, model.id, false, setDialog, noop);
  const sending = { current: false };
  const shown = () => DownloadDialog({ model, platform: 'windows', resume: false, measuring: false, starting: store.state!.starting, report: store.state!.report, source: store.state!.source, directory: store.state!.directory, freeBytes: store.state!.freeBytes, drive: store.state!.drive, onClose: noop, onSource: noop, onMeasure: noop, onChangeDirectory: noop, onStart: noop });
  expect(buttons(shown()).find(button => button.text === 'Download')?.disabled).toBe(false);
  const first = beginDownload(slow, store.state!, setDialog, error => failures.push(error), sending);
  const second = beginDownload(slow, store.state!, setDialog, error => failures.push(error), sending);
  expect(calls).toEqual([model.id]);
  expect(store.state!.starting).toBe(true);
  expect(buttons(shown()).find(button => button.text === 'Download')?.disabled).toBe(true);
  finish(new Error('Another download is already running'));
  await Promise.all([first, second]);
  expect(failures).toHaveLength(1);
  expect(store.state!.starting).toBe(false);
  expect(buttons(shown()).find(button => button.text === 'Download')?.disabled).toBe(false);
  const retry = beginDownload(slow, store.state!, setDialog, noop, sending);
  expect(calls).toEqual([model.id, model.id]);
  finish();
  await retry;
  expect(store.state).toBeNull();
});

it('sends one delete request per download and keeps its card disabled until it returns', async () => {
  let finish: () => void = noop;
  const calls: string[] = [];
  const slow = { ...actions(), remove: (id: string) => { calls.push(id); return new Promise<void>(resolve => { finish = resolve; }); } };
  let removing: string[] = [];
  const setRemoving = (update: (current: string[]) => string[]) => { removing = update(removing); };
  const sending = new Set<string>();
  const first = removeDownload(slow, q3k.id, setRemoving, noop, sending);
  const second = removeDownload(slow, q3k.id, setRemoving, noop, sending);
  expect(calls).toEqual([q3k.id]);
  expect(removing).toEqual([q3k.id]);
  finish();
  await Promise.all([first, second]);
  expect(removing).toEqual([]);
});
