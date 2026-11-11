export type Window = {start: number; end: number};

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (!Number.isFinite(gap) || gap < 0) {
    throw new RangeError("gap must be a finite nonnegative number");
  }
  for (const w of windows) {
    if (!w || !Number.isFinite(w.start) || !Number.isFinite(w.end) || w.start > w.end) {
      throw new RangeError("window must have finite numeric start <= end");
    }
  }
  const sorted = [...windows].sort((a, b) => a.start - b.start || a.end - b.end);
  const result: Window[] = [];
  for (const w of sorted) {
    const last = result.at(-1);
    if (last && w.start - last.end <= gap) {
      last.end = Math.max(last.end, w.end);
    } else {
      result.push({start: w.start, end: w.end});
    }
  }
  return result;
}
