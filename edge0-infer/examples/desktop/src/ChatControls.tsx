import type { GenerationPhase } from './conversation';
export function GenerationButton({ phase, canSend, stop }: { phase: GenerationPhase; canSend: boolean; stop: () => void }) {
  if (phase === 'stopping') return <button type="button" disabled><span className="stop-spinner" aria-hidden="true"/>Stopping…</button>;
  if (phase === 'generating') return <button type="button" onClick={stop}>Stop</button>;
  return <button type="submit" disabled={!canSend}>Send</button>;
}
