import { errorNextKey, hasKnownError } from "../chat/errorMap";
import { downloadErrKey, hasDownloadErr } from "../models/errMap";

export function requestCodeKey(code: string | null | undefined): string {
  if (!code) return "service.log.noCode";
  if (hasKnownError(code)) return errorNextKey(code);
  if (hasDownloadErr(code)) return downloadErrKey(code);
  return "chat.err.generic";
}
