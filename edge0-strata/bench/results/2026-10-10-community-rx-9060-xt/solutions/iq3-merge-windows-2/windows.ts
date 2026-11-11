export type Window = {start: number; end: number};

function checkWindow(w: Window): void {
  if (w === null || typeof w !== "object") {
    throw new RangeError("window must be an object");
  }
  const {start, end} = w;
  if (typeof start !== "number" || typeof end !== "number") {
    throw new RangeError("window start/end must be numbers");
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end) {
    throw new RangeError("window must have finite start <= end");
  }
}

export function mergeWindows(windows: Window[], gap = 0): Window[] {
  if (!Array.isArray(windows)) {
    throw new RangeError("windows must be an array");
  }
  if (typeof gap !== "number" || !Number.isFinite(gap) || gap < 0) {
    throw new RangeError("gap must be a finite nonnegative number");
  }
  for (const w of windows) {
    checkWindow(w);
  }
  const sorted = [...windows].sort(
    (a, b) => a.start - b.start || a.end - b.end,
  );
  const result: Window[] = [];
  for (const w of sorted) {
    const last = result.at(-1);
    if (last && w.start - last.end <= gap) {
      if (w.end > last.end) last.end = w.end;
    } else {
      result.push({start: w.start, end: w.end});
    }
  }
  return result;
}
