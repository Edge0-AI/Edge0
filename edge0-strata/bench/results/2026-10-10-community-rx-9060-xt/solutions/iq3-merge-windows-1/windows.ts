export type Window = {start: number; end: number};

function validateWindow(w: Window): void {
  if (
    typeof w?.start !== "number" ||
    typeof w?.end !== "number" ||
    !Number.isFinite(w.start) ||
    !Number.isFinite(w.end) ||
    w.start > w.end
  ) {
    throw new RangeError("window must have finite numeric start <= end");
  }
}

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (typeof gap !== "number" || !Number.isFinite(gap) || gap < 0) {
    throw new RangeError("gap must be a finite nonnegative number");
  }
  const copy = windows.map((w) => {
    validateWindow(w);
    return { start: w.start, end: w.end };
  });
  copy.sort((a, b) => a.start - b.start || a.end - b.end);
  const result: Window[] = [];
  for (const w of copy) {
    const last = result[result.length - 1];
    if (last && w.start - last.end <= gap) {
      if (w.end > last.end) last.end = w.end;
    } else {
      result.push({ start: w.start, end: w.end });
    }
  }
  return result;
}
