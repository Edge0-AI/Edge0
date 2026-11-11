export type Window = {start: number; end: number};
export function mergeWindows(windows: Window[], gap = 0): Window[] {
  windows.sort((a,b)=>a.start-b.start);
  const result: Window[] = [];
  for (const w of windows) {
    const last = result.at(-1);
    if (last && w.start < last.end) last.end = w.end;
    else result.push(w);
  }
  return result;
}
