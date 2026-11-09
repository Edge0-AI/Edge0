import { useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import { fetch } from '@tauri-apps/plugin-http';
import { buildRequest, streamChat, type Message } from './chat';
import { validateServerPort, type GenerationPhase } from './conversation';
import { Icon, type IconName } from './Icon';
import { Chat } from './pages/Chat';
import { Models } from './pages/Models';
import { Status } from './pages/Status';
import { DeleteConversation } from './Workspace';
import { Service, type Health } from './pages/Service';
import { Settings as SettingsPage } from './pages/Settings';
import { defaultSettings, generateToken, temperatures, type Settings } from './state/settings';
import { groupHistory, newConversation, editAndResend, regenerate, title, type Archive, type Conversation } from './state/history';
import { aggregateTotals, emptyTotals, observeTotals, resetTotals, type SavedTotals } from './state/totals';
import { usePolling } from './state/polling';
import { ArchiveWrites, SettingsWrites, WriteQueue } from './state/persistence';
import { waitForHealth } from './state/health';
import { deviceLabel, parseInfo, type EngineStatus, type Inspection } from './status';
import logo from '../src-tauri/icons/icon.png';
import 'streamdown/styles.css';
import './theme.css';
import './style.css';
type Page = 'chat' | 'models' | 'status' | 'service' | 'settings';
interface RunningService { url: string; pid: number; }
interface Model { id: string; context_length?: number; capabilities: { thinking: boolean; thinking_default: boolean }; }
export function App() {
  const [settings, setSettings] = useState(defaultSettings), settingsRef = useRef(defaultSettings);
  const [page, setPage] = useState<Page>('chat'), [archives, setArchives] = useState<Archive[]>([]);
  const [conversation, setConversation] = useState(() => newConversation()), conversationRef = useRef(conversation);
  const [phase, setPhase] = useState<GenerationPhase>('idle'), [draft, setDraft] = useState('');
  const [service, setService] = useState<RunningService>(), [model, setModel] = useState<Model>(), [activePath, setActivePath] = useState('');
  const [assets, setAssets] = useState<Inspection[]>([]), [status, setStatus] = useState<EngineStatus>(), [health, setHealth] = useState<Health>({});
  const [savedTotals, setSavedTotals] = useState<SavedTotals>(emptyTotals), totalsRef = useRef(savedTotals);
  const tokenRef = useRef('');
  const [token, setToken] = useState(''), [portText, setPortText] = useState('');
  const [initializing, setInitializing] = useState(true), [loading, setLoading] = useState(false), [error, setError] = useState('');
  const runtimeSettings = useRef<Settings | null>(null);
  const controller = useRef<AbortController | null>(null), serviceEpoch = useRef(0), persistence = useRef(new WriteQueue());
  const settingsWrites = useRef(new SettingsWrites<Settings>(next => persistence.current.enqueue(() => invoke<void>('save_settings', { settings: next }))));
  const archiveWrites = useRef(new ArchiveWrites<Conversation>(persistence.current, conversation => invoke<void>('save_conversation', { id: conversation.id, conversation }), id => invoke<void>('delete_conversation', { id })));
  const [deleting, setDeleting] = useState(false), deletingRef = useRef(false);
  const busy = phase !== 'idle', disabled = busy || loading || initializing || deleting;
  const report = (failure: unknown) => setError(String(failure));
  function changeConversation(next: Conversation) { conversationRef.current = next; setConversation(next); }
  function remember(next: Conversation) {
    if (!next.messages.length || !archiveWrites.current.available(next.id)) return;
    setArchives(previous => [...previous.filter(a => a.id !== next.id), { id: next.id, conversation: next, error: null }]);
    void archiveWrites.current.save(next).catch(report);
  }
  function keepTotals(next: SavedTotals) { totalsRef.current = next; setSavedTotals(next); void persistence.current.enqueue(() => invoke<void>('save_totals', { totals: next })).catch(report); }
  async function inspect(next = settingsRef.current) { const result = await invoke<Inspection[]>('inspect_assets', { settings: next }); setAssets(result); return result; }
  function newChat(thinking = model?.capabilities.thinking_default ?? false, nextSettings = settingsRef.current) { if (busy || loading || deletingRef.current) return; remember(conversationRef.current); changeConversation(newConversation(nextSettings.system_prompt, thinking && !!model?.capabilities.thinking, activePath)); setDraft(''); setPage('chat'); setError(''); }
  useEffect(() => {
    let active = true;
    Promise.all([invoke<Settings>('load_settings'), invoke<Archive[]>('list_conversations'), invoke<SavedTotals | null>('load_totals'), invoke<string | null>('read_api_token')]).then(async ([stored, history, totals, storedToken]) => {
      if (!active) return;
      const next = { ...defaultSettings, ...stored };
      setSettings(next); settingsRef.current = next; setPortText(next.port?.toString() ?? ''); setArchives(history); tokenRef.current = storedToken ?? ''; setToken(tokenRef.current);
      if (totals?.version === 1 && totals.instances && totals.baselines) { totalsRef.current = totals; setSavedTotals(totals); }
      changeConversation(newConversation(next.system_prompt));
      if (next.executable) await inspect(next);
    }).catch(report).finally(() => { if (active) setInitializing(false); });
    return () => { active = false; controller.current?.abort(); settingsWrites.current.cancel(); };
  }, []);
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => { document.documentElement.dataset.theme = settings.theme === 'system' ? media.matches ? 'dark' : 'light' : settings.theme; };
    apply(); media.addEventListener('change', apply); return () => media.removeEventListener('change', apply);
  }, [settings.theme]);
  async function readStatus() {
    if (!service) return;
    const epoch = serviceEpoch.current;
    try { const response = await fetch(service.url + '/status'); if (!response.ok) throw new Error(`Status failed (${response.status})`); const next: EngineStatus = await response.json(); if (epoch !== serviceEpoch.current) return; setStatus(next); keepTotals(observeTotals(totalsRef.current, next)); } catch (failure) { if (epoch === serviceEpoch.current) report(failure); }
  }
  usePolling(readStatus, !!service && (page === 'status' || page === 'service'));
  useEffect(() => {
    if (!service) return;
    let cancelled = false; const epoch = serviceEpoch.current;
    const timer = setInterval(() => { void invoke<{ running: boolean; detail: string }>('service_status').then(next => { if (!cancelled && epoch === serviceEpoch.current && !next.running) { controller.current?.abort(); serviceEpoch.current++; setService(undefined); setModel(undefined); setError(`The model service exited. ${next.detail}`); } }).catch(report); }, 1500);
    return () => { cancelled = true; clearInterval(timer); };
  }, [service]);
  async function stopService() { if (busy) return; serviceEpoch.current++; setLoading(true); try { await invoke('stop_service'); setService(undefined); setModel(undefined); setStatus(previous => previous ? { ...previous, current: null } : undefined); } catch (failure) { report(failure); } finally { setLoading(false); } }
  async function ensureToken() { const current = tokenRef.current || generateToken(); if (!tokenRef.current) { await invoke('write_api_token', { token: current }); tokenRef.current = current; setToken(current); } return current; }
  async function select(path: string, restore?: Conversation, nextSettings = settingsRef.current) {
    if (controller.current || busy || initializing || deletingRef.current) return;
    remember(conversationRef.current); setError(''); setLoading(true); serviceEpoch.current++; setService(undefined); setModel(undefined);
    const abort = new AbortController(); controller.current = abort;
    try {
      if (nextSettings.lan_access) await ensureToken();
      const started = await invoke<RunningService>('start_service', { settings: nextSettings, model: path }); abort.signal.throwIfAborted();
      await waitForHealth(fetch, started.url, abort.signal);
      const response = await fetch(started.url + '/v1/models', { signal: abort.signal }); if (!response.ok) throw new Error(`Model capabilities failed (${response.status})`);
      const selected: Model | undefined = (await response.json()).data?.[0]; if (!selected?.capabilities) throw new Error('The service does not provide model capabilities. Rebuild edge0-infer.');
      runtimeSettings.current = nextSettings; setActivePath(path); setModel(selected); setService(started); setStatus(undefined);
      changeConversation(restore ?? newConversation(nextSettings.system_prompt, selected.capabilities.thinking && selected.capabilities.thinking_default, path)); setDraft(''); setPage('chat');
    } catch (failure) { if (!abort.signal.aborted) report(failure); await invoke('stop_service').catch(report); }
    finally { controller.current = null; setLoading(false); }
  }
  async function generate(history: Message[]) {
    if (!service || !model || busy || loading || initializing || controller.current || deletingRef.current) return;
    const base = conversationRef.current, id = base.id;
    const started = { ...base, messages: [...history, { role: 'assistant' as const, content: '' }], updatedAt: Date.now() };
    changeConversation(started); remember(started); setDraft(''); setError(''); setPhase('generating');
    const abort = new AbortController(); controller.current = abort; let failed = false;
    try { await streamChat(fetch, service.url, buildRequest(history, base.thinking, model.id, temperatures[settingsRef.current.generation_style], base.systemPrompt), abort.signal, message => {
      if (abort.signal.aborted || conversationRef.current.id !== id) return;
      const previous = conversationRef.current; changeConversation({ ...previous, messages: [...previous.messages.slice(0, -1), { ...message, reasoningExpanded: previous.messages.at(-1)?.reasoningExpanded }] });
    }); } catch (failure) { failed = true; if (!abort.signal.aborted) report(failure); }
    finally {
      if (conversationRef.current.id === id) { const previous = conversationRef.current; const messages = previous.messages.map((message, index) => index === previous.messages.length - 1 ? { ...message, stopped: abort.signal.aborted, ...((failed || abort.signal.aborted) ? { metrics: undefined } : {}) } : message); const next = { ...previous, updatedAt: Date.now(), messages }; changeConversation(next); remember(next); }
      controller.current = null; setPhase('idle');
    }
  }
  function send(text: string) { if (text.trim()) void generate([...conversationRef.current.messages, { role: 'user', content: text.trim() }]); }
  function stop() { if (loading) { controller.current?.abort(); void invoke('stop_service').catch(report); return; } if (phase !== 'generating' || !controller.current) return; flushSync(() => setPhase('stopping')); controller.current.abort(); }
  async function copy(text: string) { try { await navigator.clipboard.writeText(text); } catch { setError('Clipboard access was denied by the WebView.'); } }
  async function saveSettings(next: Settings) { await settingsWrites.current.commit(next); }
  function updateSettings(next: Settings) {
    const previous = runtimeSettings.current; setSettings(next); settingsRef.current = next;
    settingsWrites.current.schedule(next, async () => { if (previous && (previous.executable !== next.executable || previous.device !== next.device || previous.prefill_device !== next.prefill_device || previous.port !== next.port)) await stopService(); }, report);
  }
  async function deleteArchive(id: string) {
    if (disabled || deletingRef.current) return;
    deletingRef.current = true; setDeleting(true);
    if (conversationRef.current.id === id) { changeConversation(newConversation(settingsRef.current.system_prompt, model?.capabilities.thinking_default ?? false, activePath)); setDraft(''); }
    try { await archiveWrites.current.remove(id); setArchives(previous => previous.filter(archive => archive.id !== id)); }
    catch (failure) { report(failure); }
    finally { deletingRef.current = false; setDeleting(false); }
  }
  async function pick(directory: boolean) { const chosen = await invoke<string | null>('choose_path', { directory }); return chosen?.replace(/[\\/]$/, '') || undefined; }
  async function addLocation() { try { const path = await pick(true); if (!path) return; const next = { ...settingsRef.current, assets: [...new Set([...settingsRef.current.assets, path])] }; await saveSettings(next); settingsRef.current = next; setSettings(next); await inspect(next); } catch (failure) { report(failure); } }
  async function removeLocation(path: string) { if (disabled || (service && assets.some(asset => asset.path === activePath && (asset.location ?? asset.path) === path))) return; try { const next = { ...settingsRef.current, assets: settingsRef.current.assets.filter(asset => asset !== path) }; await saveSettings(next); settingsRef.current = next; setSettings(next); setAssets(previous => previous.filter(asset => (asset.location ?? asset.path) !== path)); } catch (failure) { report(failure); } }
  async function healthCheck() { try { const info = parseInfo(await invoke<string>('engine_info', { executable: settingsRef.current.executable })); const assets = await inspect(); setHealth({ info, assets: assets.find(a => a.path === activePath) }); } catch (failure) { setHealth({ error: String(failure) }); } }
  async function network(nextToken = false) { if (disabled) return; try { if (nextToken) { const value = generateToken(); await invoke('write_api_token', { token: value }); tokenRef.current = value; setToken(value); } else if (!settingsRef.current.lan_access) await ensureToken(); const next = { ...settingsRef.current, lan_access: nextToken || !settingsRef.current.lan_access }; await saveSettings(next); settingsRef.current = next; setSettings(next); if (service && activePath) { const current = conversationRef.current; const previousPage = page; await select(activePath, current, next); setPage(previousPage); } } catch (failure) { report(failure); } }
  const loaded = !!service && !!model;
  const nav: [Page, string, IconName][] = [['chat', 'Chat', 'chat'], ['models', 'Models', 'box'], ['status', 'Status', 'pulse'], ['service', 'Service', 'server']];
  return <div className="app window"><aside className="sidebar"><div className="brand"><img className="logo" src={logo} alt="Edge0"/><div><b>Edge0 Chat</b><small>Local · on-device</small></div></div><button className="new-chat" disabled={disabled} onClick={() => newChat()}><Icon name="plus"/>New chat</button><nav className="nav">{nav.map(([target, label, icon]) => <a key={target} className={page === target ? 'on' : ''} onClick={() => setPage(target)}><Icon name={icon}/>{label}{target === 'status' && busy && <span className="badge"><span className="dot busy"/>Generating</span>}</a>)}</nav><div className="history">{Object.entries(groupHistory(archives)).filter(([, entries]) => entries.length).map(([group, entries]) => <section key={group}><h6>{group}</h6>{entries.map(archive => <div key={archive.id} className={`thread${archive.id === conversation.id ? ' on' : ''}`}><button className="thread-title" disabled={disabled || !archive.conversation} title={archive.error ?? undefined} onClick={() => { if (!archive.conversation || disabled || deletingRef.current || !archiveWrites.current.available(archive.id)) return; remember(conversationRef.current); if (archive.conversation.modelPath && (!loaded || activePath !== archive.conversation.modelPath)) { void select(archive.conversation.modelPath, archive.conversation); } else { changeConversation(archive.conversation); setPage('chat'); setDraft(''); } }}>{archive.conversation ? title(archive.conversation.messages) : 'Unable to open conversation'}</button><DeleteConversation disabled={disabled} onDelete={() => void deleteArchive(archive.id)}/></div>)}</section>)}</div><div className="side-foot">{loaded && <><span className="dot"/>{model.id} · {deviceLabel(status?.model.device ?? settings.device, status?.engine.devices)}</>}<button className="icon-btn settings-link" title="Settings" onClick={() => setPage('settings')}><Icon name="gear"/></button></div></aside><section className="main">{error && <div role="alert" className="error">{error}</div>}{loading && <div className="loading" role="status">Loading model… First load can take several minutes. <button className="btn sm" onClick={stop}>Cancel</button></div>}
    {page === 'chat' && <Chat conversation={conversation} phase={phase} loaded={loaded} model={model?.id ?? ''} device={settings.device} capacity={model?.context_length} supported={model?.capabilities.thinking ?? false} draft={draft} setDraft={setDraft} send={send} stop={stop} newThinking={thinking => changeConversation({ ...conversation, thinking })} reasoningExpanded={(index, expanded) => { const next = { ...conversationRef.current, messages: conversationRef.current.messages.map((message, i) => i === index ? { ...message, reasoningExpanded: expanded } : message) }; changeConversation(next); if (!busy) remember(next); }} edit={(index, text) => { if (!disabled) void generate(editAndResend(conversationRef.current.messages, index, text)); }} redo={() => { if (!disabled) void generate(regenerate(conversationRef.current.messages)); }} copy={text => void copy(text)} models={() => setPage('models')}/>}
    {page === 'models' && <Models assets={assets} activePath={loaded ? activePath : undefined} capacity={model?.context_length} disabled={disabled} load={path => void select(path)} unload={() => void stopService()} chat={() => setPage('chat')} refresh={() => { void inspect().catch(report); }} add={() => void addLocation()} remove={path => void removeLocation(path)}/>}
    {page === 'status' && <Status status={status} live={loaded} allTime={aggregateTotals(savedTotals)} reset={() => keepTotals(resetTotals(totalsRef.current))}/>}
    {page === 'service' && <Service status={status} running={loaded} disabled={disabled} lan={settings.lan_access} token={token} device={settings.device} health={health} restart={() => { const current = conversationRef.current; void select(activePath, current).then(() => setPage('service')); }} stop={() => void stopService()} toggle={() => void network()} newToken={() => void network(true)} copy={text => void copy(text)} check={() => void healthCheck()}/>}
    {page === 'settings' && <SettingsPage settings={settings} port={portText} disabled={disabled} update={updateSettings} setPort={text => { setPortText(text); const valid = validateServerPort(text); if (valid.valid) updateSettings({ ...settingsRef.current, port: valid.port }); }} browse={() => { void pick(false).then(path => { if (path) updateSettings({ ...settingsRef.current, executable: path }); }).catch(report); }}/>}
  </section></div>;
}
createRoot(document.getElementById('root')!).render(<App/>);
