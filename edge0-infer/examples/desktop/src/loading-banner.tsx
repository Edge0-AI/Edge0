/** The status line while a model loads or unloads. Only a load can be cancelled. */
export function LoadingBanner({ unloading, cancel }: { unloading: boolean; cancel: () => void }) {
  if (unloading) return <div className="loading" role="status">Unloading model…</div>;
  return <div className="loading" role="status">Loading model… First load can take several minutes. <button className="btn sm" onClick={cancel}>Cancel</button></div>;
}
