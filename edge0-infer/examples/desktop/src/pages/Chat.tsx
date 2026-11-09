import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { appData, assistantState, formatMetrics, type Message } from '../chat';
import { contextUsage } from '../format';
import { title, type Conversation } from '../state/history';
import type { GenerationPhase } from '../conversation';
import { Markdown } from '../Markdown';
import { Icon } from '../Icon';
import { ThinkingSwitch } from '../Workspace';
export function Chat({ conversation, phase, loaded, model, device = 'cpu', capacity, supported, draft, setDraft, send, stop, newThinking, reasoningExpanded, edit, redo, copy, models }: {
  conversation: Conversation; phase: GenerationPhase; loaded: boolean; model: string; device?: string; capacity?: number; supported: boolean; draft: string; setDraft: (text: string) => void; send: (text: string) => void; stop: () => void; newThinking: (value: boolean) => void; reasoningExpanded: (index: number, expanded: boolean) => void; edit: (index: number, text: string) => void; redo: () => void; copy: (text: string) => void; models: () => void;
}) {
  const host = (globalThis.navigator?.platform ?? '').toLowerCase().includes('mac') ? 'Mac' : 'PC';
  const deviceName = device === 'cpu' ? 'CPU' : device[0].toUpperCase() + device.slice(1);
  const messages = conversation.messages, busy = phase !== 'idle';
  const latest = [...messages].reverse().find(m => m.role === 'assistant' && m.metrics)?.metrics;
  const usage = contextUsage({ input_tokens: latest?.input_tokens, generated_tokens: latest?.generated_count, context_length: capacity });
  const list = useRef<HTMLDivElement>(null);
  const [editing, setEditing] = useState<number>(), [text, setText] = useState('');
  const last = messages.at(-1);
  useEffect(() => { list.current?.scrollTo({ top: list.current.scrollHeight, behavior: 'smooth' }); }, [messages.length, last?.content, last?.reasoning_content]);
  const renderMessage = (message: Message, index: number) => message.role === 'user' ? <div className="msg-user" key={index}><div className="msg-user-body">
    {editing === index ? <div className="edit-message"><textarea aria-label="Edit message" value={text} onChange={e => setText(e.target.value)}/><div className="inline"><button className="btn" onClick={() => setEditing(undefined)}>Cancel</button><button className="btn primary" disabled={busy || !text.trim()} onClick={() => { edit(index, text); setEditing(undefined); }}>Send</button></div></div> : <div className="bubble">{message.content}</div>}
    <div className="edit-row"><button className="icon-btn" title="Edit and resend" disabled={busy || !loaded} onClick={() => { setEditing(index); setText(message.content); }}><Icon name="edit"/></button><button className="icon-btn" title="Copy" disabled={busy} onClick={() => copy(message.content)}><Icon name="copy"/></button></div>
  </div></div> : (() => { const state = assistantState({ is_last: index === messages.length - 1, generating: busy, has_content: message.content.length > 0, shows_reasoning: conversation.thinking && !!message.reasoning_content }); return <div className="msg-ai" key={index}>
    {conversation.thinking && message.reasoning_content && <details className="reason" open={message.reasoningExpanded ?? true} onToggle={event => { const expanded = event.currentTarget.open; if (expanded !== (message.reasoningExpanded ?? true)) reasoningExpanded(index, expanded); }}><summary><Icon name="brain"/>Thinking<span className="sp"/><button type="button" className="icon-btn" title="Copy reasoning" disabled={busy} onClick={event => { event.preventDefault(); copy(message.reasoning_content!); }}><Icon name="copy"/></button><Icon name="chev"/></summary><div className="body"><Markdown disabled={busy} copy={copy}>{message.reasoning_content}</Markdown></div></details>}
    {state.pending ? <div className="pending" role="status" aria-label="Generating"><span className="dots"><i/><i/><i/></span></div> : <div className="md"><Markdown disabled={busy} copy={copy}>{message.content}</Markdown>{!state.actions && message.content && <span className="cursor"/>}</div>}
    {state.actions && <div className="ai-foot"><button className="icon-btn" title="Copy" disabled={busy} onClick={() => copy(message.content)}><Icon name="copy"/></button>{index === messages.length - 1 && <button className="icon-btn" title="Regenerate" disabled={busy || !loaded} onClick={redo}><Icon name="redo"/></button>}{message.metrics && <span className="metrics">{formatMetrics(message.metrics)}</span>}{message.stopped && <span className="stopped">Stopped</span>}</div>}
  </div>; })();
  return <><div className="topbar"><h1>{title(messages)}</h1><span className="sp"/><ThinkingSwitch checked={conversation.thinking} disabled={!loaded || busy || messages.length > 0 || !supported} supported={supported} locked={messages.length > 0} change={newThinking}/></div>
    <div className="scroll" ref={list} aria-label="Conversation">{!loaded ? <div className="empty"><p>Load a model to start a conversation.</p><button className="btn primary" onClick={models}>Choose a model</button></div> : !messages.length ? <div className="empty"><h2>What can I help with?</h2><p>{model} is loaded and running locally on {deviceName}.</p><div className="sugg">{appData.suggestions.map(s => <button key={s} onClick={() => send(s)}>{s}</button>)}</div></div> : <div className="thread-col">{messages.map(renderMessage)}</div>}</div>
    {loaded && <div className="composer-wrap"><form className="composer" onSubmit={event => { event.preventDefault(); send(draft); }}><textarea aria-label="Message" placeholder={`Message ${model}`} value={draft} disabled={phase === 'stopping'} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); send(draft); } }}/><div className="row"><span className="ctx" title="Context used by this conversation">{usage.ring_percent !== null && <span className="ring" style={{ '--p': usage.ring_percent } as CSSProperties}/>} {usage.text}</span><span className="sp"/>{phase === 'stopping' ? <button className="btn sm" type="button" disabled><span className="stop-spinner"/>Stopping…</button> : busy ? <button className="btn sm" type="button" onClick={stop}><span className="dot busy"/>Stop</button> : <button className="btn primary round" title="Send" disabled={!draft.trim()}><Icon name="arrow"/></button>}</div></form><div className="hint">Runs entirely on this {host}. Answers can be wrong.</div></div>}
  </>;
}
