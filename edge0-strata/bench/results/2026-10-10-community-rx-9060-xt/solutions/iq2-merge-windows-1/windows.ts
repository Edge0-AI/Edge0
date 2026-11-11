export type Window = {start: number; end: number};

const isFiniteNumber = (value: number) => Number.isFinite(value);

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (!isFiniteNumber(gap) || gap < 0) throw new RangeError('gap must be finite and nonnegative');
  const items = windows.map((w) => {
    if (!isFiniteNumber(w.start) || !isFiniteNumber(w.end) || w.start > w.end) {
      throw new RangeError('window start/end must be finite with start <= end');
    }
    return {start: w.start, end: w.end};
  });
  items.sort((a, b) => a.start - b.start || a.end - b.end);
  const result: Window[] = [];
  for (const w of items) {
    const last = result.at(-1);
    if (last && w.start <= last.end + gap) {
      if (w.end > last.end) last.end = w.end;
    } else {
      result.push({start: w.start, end: w.end});
    }
  }
  return result;
}
