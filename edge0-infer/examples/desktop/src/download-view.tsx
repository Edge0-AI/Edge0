import { useRef, useState, type ReactNode } from 'react';
import { Icon } from './Icon';
import type { Inspection } from './status';
import {
  catalogFor, decimalGb, downloadBusy, enoughSpace, fastestSource, pausedMeta, presentPath, progressPercent,
  remainingBytes, resumeDirectory, runningMeta, savedInto, sourceHost, sourceLabel, sourceReachable, spaceNote,
  type CatalogModel, type DiskReport, type JobView, type MeasureReport, type SourceId,
} from './download-logic';

export function DownloadingBadge({ active }: { active: boolean }) {
  if (!active) return null;
  return <span className="badge"><span className="dot busy"/>Downloading</span>;
}

export function DownloadCard({ model, job, present, blocked, disabled, onDownload, onPause, onResume, onDelete }: {
  model: CatalogModel; job?: JobView; present?: string; blocked: boolean; disabled: boolean;
  onDownload: () => void; onPause: () => void; onResume: () => void; onDelete: () => void;
}) {
  const phase = present ? 'present' : job?.phase ?? 'none';
  const percent = job ? progressPercent(job.done_bytes, job.total_bytes) : 0;
  const bar = (width: number, color?: string) => <div className="progress"><i style={{ width: `${width}%`, background: color }}/></div>;
  let body: ReactNode;
  let actions: ReactNode = null;
  let tag: ReactNode = null;
  let cls = '';
  if (phase === 'present') {
    cls = 'present';
    tag = <span className="tag ok"><Icon name="check"/>On this machine</span>;
    body = <><p>{decimalGb(model.bytes)} GB · on this machine</p><p className="mono" title={present}>{present}</p></>;
  } else if (phase === 'running') {
    cls = 'active';
    tag = <span className="tag busy"><span className="dot busy"/>Downloading</span>;
    body = <>{bar(percent)}<p className="dl-meta">{job && runningMeta(job)}</p></>;
    actions = <button className="btn" disabled={disabled} onClick={onPause}>Pause</button>;
  } else if (phase === 'checking') {
    cls = 'active';
    tag = <span className="tag busy"><span className="spin"/>Checking</span>;
    body = <>{bar(100)}<p className="dl-meta">{decimalGb(model.bytes)} GB downloaded · checking the downloaded files and the model folder</p></>;
  } else if (phase === 'paused' || phase === 'failed') {
    const failed = phase === 'failed';
    tag = failed ? <span className="tag err">Failed</span> : <span className="tag warn">Paused</span>;
    body = <>{bar(percent, failed ? 'var(--err)' : 'var(--text-3)')}<p className="dl-meta">{job && pausedMeta(job)}</p>{failed ? <p style={{ color: 'var(--err)' }}>{job?.error}</p> : <p>Downloaded files are kept; Resume continues where it stopped.</p>}</>;
    actions = <><button className="btn primary" disabled={disabled || blocked} onClick={onResume}>Resume</button><button className="btn danger" disabled={disabled} onClick={onDelete}>Delete</button></>;
  } else {
    body = <><p>{decimalGb(model.bytes)} GB · {model.files} files</p><p className="mono">{model.repo}</p></>;
    actions = <button className="btn primary" disabled={disabled || blocked} onClick={onDownload}><Icon name="down"/>Download</button>;
  }
  return <div className={`card model-card dl-card${cls ? ` ${cls}` : ''}`}><div className="mi"><Icon name={phase === 'present' ? 'check' : 'down'}/></div><div><h4>{model.name}{tag}</h4>{body}</div><div className="inline">{actions}</div></div>;
}

export function DownloadDialog({ model, job, platform, resume, measuring, starting = false, report, source, directory, freeBytes, drive, onClose, onSource, onMeasure, onChangeDirectory, onStart }: {
  model: CatalogModel; job?: JobView; platform: string; resume: boolean; measuring: boolean; starting?: boolean; report: MeasureReport; source: SourceId | null;
  directory: string; freeBytes: number | null; drive: string | null;
  onClose: () => void; onSource: (source: SourceId) => void; onMeasure: () => void; onChangeDirectory: () => void; onStart: () => void;
}) {
  const fastest = measuring ? null : fastestSource(report.huggingface_ms, report.modelscope_ms);
  const unreachable = !measuring && report.huggingface_ms == null && report.modelscope_ms == null;
  const need = remainingBytes(model, job);
  const space = freeBytes == null ? null : spaceNote(freeBytes, need, drive);
  const reachable = source != null && sourceReachable(source, report);
  const canStart = !starting && !measuring && !unreachable && reachable && space != null && !space.low && enoughSpace(freeBytes ?? 0, need);
  const sources: SourceId[] = ['modelscope', 'huggingface'];
  return <div className="scrim" onClick={event => { if (event.target === event.currentTarget) onClose(); }}><div className="dialog" role="dialog" aria-modal="true">
    <header><h2>{resume ? 'Resume' : 'Download'} {model.name}</h2><p>{model.repo} · {decimalGb(model.bytes)} GB · {model.files} files{job && job.done_bytes > 0 ? ` · ${decimalGb(job.done_bytes)} GB already downloaded` : ''}</p></header>
    <div className="body">
      {unreachable && <div className="banner err">Both sources are unreachable. Check the network, then measure again.</div>}
      <div className="field"><label>Source</label>
        <div className="src-list">{sources.map(id => {
          const ms = id === 'huggingface' ? report.huggingface_ms : report.modelscope_ms;
          return <label key={id} className={`src${source === id ? ' on' : ''}`}><input type="radio" name="src" checked={source === id} onChange={() => onSource(id)}/><span><b>{sourceLabel(id)}</b><small>{sourceHost(id)}</small></span><span className={`ms${measuring ? ' pending' : ''}`}>{measuring ? <span className="spin"/> : ms == null ? 'unreachable' : <>{ms} ms{id === fastest && <> · <span className="tag ok">Fastest</span></>}</>}</span></label>;
        })}</div>
        <div className="src-foot">{measuring ? 'Measuring response time…' : 'Chosen by response time just now; pick the other one to use it instead.'}<span className="sp"/>{!measuring && <button type="button" onClick={onMeasure}>Measure again</button>}</div>
      </div>
      <div className="field last"><label>Save to</label>
        <div className="path-row"><input className="input" value={directory} readOnly disabled={resume}/><button className="btn" type="button" disabled={resume} onClick={onChangeDirectory}><Icon name="folder"/>Change…</button></div>
        <p className="note">Saved into <span className="mono">{savedInto(directory, model.folder, platform)}</span></p>
        {space && <p className={`note${space.low ? ' err' : ''}`}>{space.text}</p>}
      </div>
    </div>
    <footer><button className="btn" type="button" onClick={onClose}>Cancel</button><button className="btn primary" type="button" disabled={!canStart} onClick={onStart}>{resume ? 'Resume' : 'Download'}</button></footer>
  </div></div>;
}

export interface DownloadActions {
  platform: string;
  saveDirectory: string;
  jobs: JobView[];
  measure: (id: string) => Promise<MeasureReport>;
  disk: (path: string) => Promise<DiskReport>;
  start: (id: string, source: string, directory: string) => Promise<void>;
  pause: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  browse: () => Promise<string | undefined>;
  report?: (failure: unknown) => void;
}
export interface DialogState {
  id: string;
  resume: boolean;
  measuring: boolean;
  /** The start request has been sent and has not returned yet. */
  starting: boolean;
  report: MeasureReport;
  source: SourceId | null;
  directory: string;
  freeBytes: number | null;
  drive: string | null;
}
const emptyReport: MeasureReport = { huggingface_ms: null, modelscope_ms: null, chosen: null };
type DialogUpdate = DialogState | null | ((current: DialogState | null) => DialogState | null);
/** Opens the download dialog in its measuring state at once, then fills in the measured sources and free space.
 * Resuming shows the folder the paused download is in, not the current save location. */
export async function openDownload(actions: DownloadActions, id: string, resume: boolean, setDialog: (update: DialogUpdate) => void, fail: (failure: unknown) => void) {
  const partial = resume ? actions.jobs.find(job => job.id === id)?.directory : undefined;
  const directory = (partial && resumeDirectory(partial)) || actions.saveDirectory;
  setDialog({ id, resume, measuring: true, starting: false, report: emptyReport, source: null, directory, freeBytes: null, drive: null });
  try {
    const [measured, space] = await Promise.all([
      actions.measure(id).catch(() => emptyReport),
      actions.disk(directory).catch(() => ({ free_bytes: 0, drive: null })),
    ]);
    setDialog(current => current?.id === id ? { ...current, measuring: false, report: measured, source: measured.chosen, freeBytes: space.free_bytes, drive: space.drive } : current);
  } catch (failure) { fail(failure); }
}

/** Sends one start request for the dialog; further clicks do nothing until it returns. The dialog closes when it
 * succeeds and can be used again when it fails. */
export async function beginDownload(actions: DownloadActions, dialog: DialogState, setDialog: (update: DialogUpdate) => void, fail: (failure: unknown) => void, sending: { current: boolean }) {
  if (!dialog.source || sending.current) return;
  sending.current = true;
  setDialog(current => current?.id === dialog.id ? { ...current, starting: true } : current);
  try {
    await actions.start(dialog.id, dialog.source, dialog.directory);
    setDialog(current => current?.id === dialog.id ? null : current);
  } catch (failure) {
    setDialog(current => current?.id === dialog.id ? { ...current, starting: false } : current);
    fail(failure);
  } finally {
    sending.current = false;
  }
}
/** Sends one delete request per download; its card stays disabled until the request returns. */
export async function removeDownload(actions: DownloadActions, id: string, setRemoving: (update: (current: string[]) => string[]) => void, fail: (failure: unknown) => void, sending: Set<string>) {
  if (sending.has(id)) return;
  sending.add(id);
  setRemoving(current => [...current, id]);
  try {
    await actions.remove(id);
  } catch (failure) {
    fail(failure);
  } finally {
    sending.delete(id);
    setRemoving(current => current.filter(item => item !== id));
  }
}

export function DownloadSection({ assets, disabled, firstRun, actions }: { assets: Inspection[]; disabled: boolean; firstRun: boolean; actions: DownloadActions }) {
  const models = catalogFor(actions.platform);
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string[]>([]);
  const starting = useRef(false);
  const deleting = useRef(new Set<string>());
  if (!models.length) return null;
  const busy = downloadBusy(actions.jobs);
  const jobFor = (id: string) => actions.jobs.find(job => job.id === id);
  const fail = (failure: unknown) => actions.report?.(failure);
  const open = (id: string, resume: boolean) => openDownload(actions, id, resume, setDialog, fail);
  async function changeDirectory() {
    if (!dialog || dialog.resume) return;
    try {
      const picked = await actions.browse();
      if (!picked) return;
      const space = await actions.disk(picked).catch(() => ({ free_bytes: 0, drive: null }));
      setDialog(current => current ? { ...current, directory: picked, freeBytes: space.free_bytes, drive: space.drive } : current);
    } catch (failure) { fail(failure); }
  }
  const selected = dialog ? models.find(model => model.id === dialog.id) : undefined;
  const confirming = confirmId ? models.find(model => model.id === confirmId) : undefined;
  const confirmingJob = confirmId ? jobFor(confirmId) : undefined;
  return <>
    <div className={`section-label${firstRun ? ' first' : ''}`}>Download<span className="sp"/></div>
    <div className="grid">{models.map(model => <DownloadCard key={model.id} model={model} job={jobFor(model.id)} present={presentPath(model, assets)} blocked={busy} disabled={disabled || removing.includes(model.id)} onDownload={() => void open(model.id, false)} onPause={() => void actions.pause(model.id).catch(fail)} onResume={() => void open(model.id, true)} onDelete={() => setConfirmId(model.id)}/>)}</div>
    <p className="help">Each model downloads into a folder of its own. Models already on this machine are marked instead of offered again.</p>
    {dialog && selected && <DownloadDialog model={selected} job={jobFor(selected.id)} platform={actions.platform} resume={dialog.resume} measuring={dialog.measuring} starting={dialog.starting} report={dialog.report} source={dialog.source} directory={dialog.directory} freeBytes={dialog.freeBytes} drive={dialog.drive} onClose={() => setDialog(null)} onSource={source => setDialog(current => current ? { ...current, source } : current)} onMeasure={() => void open(selected.id, dialog.resume)} onChangeDirectory={() => void changeDirectory()} onStart={() => void beginDownload(actions, dialog, setDialog, fail, starting)}/>}
    {confirming && <div className="scrim" onClick={event => { if (event.target === event.currentTarget) setConfirmId(null); }}><div className="dialog narrow" role="alertdialog" aria-modal="true"><header><h2>Delete the partial download?</h2><p>{confirming.name} · {decimalGb(confirmingJob?.done_bytes ?? 0)} GB downloaded so far will be removed. Downloading it again starts from the beginning.</p></header><footer><button className="btn" type="button" onClick={() => setConfirmId(null)}>Keep</button><button className="btn danger" type="button" onClick={() => { const id = confirming.id; setConfirmId(null); void removeDownload(actions, id, setRemoving, fail, deleting.current); }}>Delete</button></footer></div></div>}
  </>;
}
