// App-layer sampling presets. Engine default 0.7 is left alone for parity; precise 0.2 is the product default.
export type GenPreset = "precise" | "balanced" | "creative";

const GEN_KEY = "edge0.gen";

export const GEN_TEMPERATURE: Record<GenPreset, number> = {
  precise: 0.2,
  balanced: 0.7,
  creative: 1.0,
};

export function storedGen(): GenPreset {
  const v = localStorage.getItem(GEN_KEY);
  return v === "balanced" || v === "creative" ? v : "precise";
}

export function setGen(p: GenPreset): void {
  localStorage.setItem(GEN_KEY, p);
}

export function currentTemperature(): number {
  return GEN_TEMPERATURE[storedGen()];
}
