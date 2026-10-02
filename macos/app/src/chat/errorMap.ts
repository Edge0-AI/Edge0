const CHAT_ERROR_KEYS: Record<string, string> = {
  "E-SRV-NOTLOAD": "chat.err.notload",
  "E-SRV-BUSY": "chat.err.busy",
  "E-SRV-CANCEL": "chat.err.cancel",
  "E-SRV-WORKER": "chat.err.worker",
  "E-SRV-PARAM": "chat.err.param",
  "E-MODEL-MISSING": "chat.err.missing",
  "E-MEM-LOAD": "chat.err.memload",
  "E-GPU-DEVICE": "chat.err.gpu",
  "E-SRV-VERSION": "chat.err.version",
};

export function errorNextKey(code: string | null): string {
  return (code && CHAT_ERROR_KEYS[code]) || "chat.err.generic";
}

export function hasKnownError(code: string | null): boolean {
  return code !== null && code in CHAT_ERROR_KEYS;
}
