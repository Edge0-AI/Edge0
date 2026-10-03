// chat/modelStatus.ts — status-light derivation. Inputs are shell snapshots only:
// the installed list + engineStore.status + loadingTier. No timers, no guessing:
// the light reflects facts alone.
import type { EngStatus } from "../api/client";

export type ModelLight = "resident" | "loading" | "installed" | "absent";

export function deriveLight(
  status: EngStatus | null,
  loadingTier: string | null,
  installed: string[],
  tier: string,
): ModelLight {
  if (!installed.includes(tier)) return "absent";
  if (status?.running && status.tier === tier) return "resident";
  if (loadingTier === tier) return "loading";
  return "installed";
}

/** Picker candidates = installed tiers (catalog/download browsing belongs to the models page). */
export function candidateModels(installed: string[]): string[] {
  return installed;
}
