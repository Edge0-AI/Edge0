// Theme is applied via data-theme on the document; "system" is resolved in JS, not prefers-color-scheme CSS.
export type Theme = "light" | "dark" | "system";
export type FontScale = "sm" | "md" | "lg";

const THEME_KEY = "edge0.t…heme";
const FONT_KEY = "edge0.token…font";

export function resolveTheme(t: Theme): "light" | "dark" {
  if (t !== "system") return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function storedTheme(): Theme {
  const v = localStorage.getItem(THEME_KEY);
  return v === "light" || v === "dark" || v === "system" ? v : "system";
}

export function storedFont(): FontScale {
  const v = localStorage.getItem(FONT_KEY);
  return v === "sm" || v === "md" || v === "lg" ? v : "md";
}

export function applyTheme(t: Theme): void {
  localStorage.setItem(THEME_KEY, t);
  document.documentElement.dataset.theme = resolveTheme(t);
}

export function applyFont(f: FontScale): void {
  localStorage.setItem(FONT_KEY, f);
  document.documentElement.dataset.font = f;
}
