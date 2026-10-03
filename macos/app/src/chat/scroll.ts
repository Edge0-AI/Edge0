export const STICK_BOTTOM_THRESHOLD = 40;

export interface ScrollGeometry {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}

export function isAtBottom(el: ScrollGeometry, threshold = STICK_BOTTOM_THRESHOLD): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}
