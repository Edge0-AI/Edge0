const DL_ERROR_KEYS: Record<string, string> = {
  "E-DL-NET": "models.err.net",
  "E-DL-SRC": "models.err.src",
  "E-DL-DISK": "models.err.disk",
  "E-MODEL-HASH": "models.err.hash",
  "E-MODEL-INVALID": "models.err.invalid",
  "E-MANIFEST-SIG": "models.err.sig",
  "E-MANIFEST-EXPIRED": "models.err.expired",
  "E-MANIFEST-MINVER": "models.err.minver",
  "E-SRV-CONFLICT": "models.err.conflict",
};

export function downloadErrKey(code: string | null): string {
  return (code && DL_ERROR_KEYS[code]) || "models.err.generic";
}

export function hasDownloadErr(code: string | null): boolean {
  return code !== null && code in DL_ERROR_KEYS;
}
