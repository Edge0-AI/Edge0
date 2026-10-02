import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useState } from "react";

export interface ServiceHealth {
  daemon_version: string;
  host_app_version: string | null;
  scope: string;
  started_by: string;
  uptime_s: number;
  pid: number;
}

export interface ServicePhase {
  phase: string;
  health?: ServiceHealth;
  code?: number | null;
  detail?: string;
  base?: string;
  spawnedByUs?: boolean;
}

export interface ServiceState {
  phase: ServicePhase;
  appVersion: string;
  versionMismatch: boolean;
  spawnWasDev: boolean;
  pinned: boolean;
}

export function isLoopback(bound: string): boolean {
  return bound === "::1" || bound.startsWith("127.");
}

export function useServiceState(): [ServiceState | null, () => void] {
  const [svc, setSvc] = useState<ServiceState | null>(null);
  const refresh = useCallback(() => {
    invoke<ServiceState>("service_state")
      .then(setSvc)
      .catch(() => undefined);
  }, []);
  useEffect(() => {
    refresh();
    const poll = setInterval(refresh, 2000);
    return () => clearInterval(poll);
  }, [refresh]);
  return [svc, refresh];
}
