import type { ReactNode } from 'react';
import type { AssetResult } from './chat';
import { Icon } from './Icon';
import { Markdown } from './Markdown';
export type Page = 'models' | 'chat';
export function currentPage(loaded: boolean, requested: Page): Page { return loaded ? requested : 'models'; }
export function Workspace({ loaded, page, models, chat }: { loaded: boolean; page: Page; models: ReactNode; chat: ReactNode }) {
  return currentPage(loaded, page) === 'chat' ? chat : models;
}
export function PageHeader({ chat, loaded, disabled, models, newChat, resume, refresh }: {
  chat: boolean; loaded: boolean; disabled: boolean; models: () => void; newChat: () => void; resume: () => void; refresh: () => void;
}) {
  return <header className="page-header"><div className="header-content">
    <div className="header-leading">{chat ? <button disabled={disabled} onClick={models}>Models</button> : loaded && <button disabled={disabled} onClick={resume}>Back to chat</button>}</div>
    <h1>{chat ? 'Edge0 Chat' : 'Choose a model'}</h1>
    <div className="header-trailing">{chat ? <button disabled={disabled} onClick={newChat}>New chat</button> : <button disabled={disabled} onClick={refresh}>Refresh</button>}</div>
  </div></header>;
}
export function ModelPicker({ assets, directories, disabled, initializing, activePath, load, resume, children }: {
  assets: AssetResult[]; directories: string[]; disabled: boolean; initializing: boolean; activePath?: string;
  load: (path: string) => void; resume: () => void; children: ReactNode;
}) {
  const available = assets.filter(asset => asset.selectable), unavailable = assets.filter(asset => !asset.selectable);
  return <main className="models-page"><section className="model-picker">
    {initializing && <p role="status">Preparing local settings…</p>}
    {!initializing && !available.length && <p className="asset-help">Prepare assets with the <code>tools/assets</code> scripts following <code>examples/desktop/README.md</code>, then choose their directory under Other location…; configured directories: {directories.length ? directories.join(', ') : 'none'}.</p>}
    <div className="model-list">{available.map(asset => <div className="model-row" key={asset.path} title={asset.path}>
      <span>{asset.label}</span>{asset.path === activePath ? <button disabled={disabled} onClick={resume}>Return to chat</button> : <button disabled={disabled} onClick={() => load(asset.path)}>Load</button>}
    </div>)}</div>
    {!!unavailable.length && <section className="unavailable-models"><h2>Unavailable</h2>{unavailable.map(asset => <p key={asset.path} title={asset.path}>{asset.label}: {asset.reason}</p>)}</section>}
    {children}
  </section></main>;
}
export function OtherLocation({ open = false, setOpen, children }: { open?: boolean; setOpen?: (value: boolean) => void; children: ReactNode }) {
  return <details className="other-location" open={open} onToggle={event => setOpen?.(event.currentTarget.open)}><summary>Other location…</summary>{children}</details>;
}
export function ThinkingBlock({ enabled, content, expanded = true, setExpanded }: {
  enabled: boolean; content: string; expanded?: boolean; setExpanded: (value: boolean) => void;
}) {
  if (!enabled || !content) return null;
  return <details className="reasoning-block" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
    <summary>Thinking</summary><Markdown>{content}</Markdown>
  </details>;
}
export function ThinkingSwitch({ checked, disabled, supported, locked, change }: {
  checked: boolean; disabled: boolean; supported: boolean; locked: boolean; change: (value: boolean) => void;
}) {
  return <div className={`thinking-control${disabled ? ' is-locked' : ''}`}>
    <button className="thinking-switch" type="button" role="switch" aria-label="Thinking" aria-checked={checked} disabled={disabled} onClick={() => { if (!disabled) change(!checked); }}>
      <span className={`switch-track ${checked ? 'is-on' : 'is-off'}`} aria-hidden="true"><span className="switch-thumb"/></span><span>Thinking{!supported && ' unavailable'}</span><span className="switch-state">{checked ? 'On' : 'Off'}</span>
    </button>
    {locked && <span className="thinking-note">Applies to a new chat</span>}
  </div>;
}
export function DeleteConversation({ disabled, onDelete }: { disabled: boolean; onDelete: () => void }) {
  return <button className="icon-btn delete" title="Delete conversation" aria-label="Delete conversation" disabled={disabled} onClick={onDelete}><Icon name="trash"/></button>;
}
