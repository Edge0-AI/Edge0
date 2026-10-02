// Follow-bottom predicate: pin to the bottom on streaming increments only while the user
// is near the bottom (prevents the "scroll lock" bug that drags a scrolled-up reader back).
export const STICK_BOTTOM_THRESHOLD = 40;

export interface ScrollGeometry {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}

export function isAtBottom(el: ScrollGeometry, threshold = STICK_BOTTOM_THRESHOLD): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}
