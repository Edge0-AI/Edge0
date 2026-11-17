"""Owner-only files on every platform.

POSIX keeps what the code always did (``0600`` files, ``0700`` directories). NTFS has no mode bits,
so on Windows the same promise is an ACL: a protected (non-inherited) DACL with exactly one
allow entry, for the current user. This uses ``advapi32`` through ``ctypes``; there is no
``pywin32`` or ``icacls`` dependency and no output parsing that depends on the display language.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

WINDOWS = os.name == "nt"

if WINDOWS:
    import ctypes
    from ctypes import wintypes

    _advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    _kernel.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel.LocalFree.restype = ctypes.c_void_p
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    _advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
    _advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    _advapi.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                                                  ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    _advapi.SetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]
    _advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    _advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p]
    _advapi.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    _advapi.SetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _advapi.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p]
    _advapi.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.c_void_p, ctypes.c_void_p]

_SE_FILE_OBJECT = 1
_DACL = 0x4
_PROTECTED_DACL = 0x80000000
_TOKEN_QUERY = 0x8
_TOKEN_ADJUST_PRIVILEGES = 0x20
_TOKEN_ADJUST_DEFAULT = 0x80
_TOKEN_OWNER = 4
_TOKEN_DEFAULT_DACL = 6
_OWNER = 0x1
_TOKEN_USER = 1
_SDDL_REVISION_1 = 1
_sid_cache: str | None = None


def _winerr(what: str, code: int | None = None) -> OSError:
    return ctypes.WinError(code if code is not None else ctypes.get_last_error(), what)  # type: ignore[attr-defined]


def current_user_sid() -> str:
    """The string SID of the user this process runs as (Windows only)."""
    global _sid_cache
    if not WINDOWS:
        raise OSError("SIDs exist on Windows only")
    if _sid_cache:
        return _sid_cache
    token = wintypes.HANDLE()
    if not _advapi.OpenProcessToken(_kernel.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)):
        raise _winerr("OpenProcessToken")
    try:
        needed = wintypes.DWORD()
        _advapi.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not _advapi.GetTokenInformation(token, _TOKEN_USER, buffer, needed, ctypes.byref(needed)):
            raise _winerr("GetTokenInformation")
        psid = ctypes.c_void_p.from_buffer(buffer).value  # TOKEN_USER starts with the SID pointer
        text = wintypes.LPWSTR()
        if not _advapi.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            raise _winerr("ConvertSidToStringSidW")
        try:
            _sid_cache = text.value
        finally:
            _kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel.CloseHandle(token)
    assert _sid_cache
    return _sid_cache


def _same_sid(sddl_trustee: str, sid: str) -> bool:
    """True when an SDDL trustee names ``sid``. SDDL prints well-known accounts as two-letter aliases (the built-in
    Administrator, RID 500, is ``LA``), so an alias is turned into its string SID before the compare (#3302)."""
    if sddl_trustee == sid:
        return True
    if not WINDOWS:
        return False
    return _string_sid(sddl_trustee) == sid.upper()


def restrict_to_current_user(path: str | os.PathLike, *, directory: bool | None = None) -> None:
    """Make ``path`` readable and writable by the current user only.

    POSIX: ``chmod 0600`` (``0700`` for a directory). Windows: replace the DACL with one protected
    entry granting the current user full control (inherited by children when ``path`` is a
    directory) and drop every inherited entry.
    """
    path = os.fspath(path)
    if directory is None:
        directory = os.path.isdir(path)
    if not WINDOWS:
        os.chmod(path, 0o700 if directory else 0o600)
        return
    flags = "OICI" if directory else ""
    sddl = f"D:PAI(A;{flags};FA;;;{current_user_sid()})"
    descriptor = ctypes.c_void_p()
    if not _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None):
        raise _winerr("ConvertStringSecurityDescriptorToSecurityDescriptorW")
    try:
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
        if not _advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl),
                                                 ctypes.byref(defaulted)) or not present.value:
            raise _winerr("GetSecurityDescriptorDacl")
        # The explicit entry for the user's own SID is what makes the file reachable from a normal (non-elevated) session:
        # a file made by an elevated process is owned by BUILTIN\Administrators, and a rule naming only SYSTEM,
        # Administrators and OWNER RIGHTS shuts the same person out when UAC filters Administrators from their token.
        # So the owner becomes the user too, where Windows allows it; when it does not, the explicit entry still stands.
        owner = _user_psid()
        code = _advapi.SetNamedSecurityInfoW(path, _SE_FILE_OBJECT, _OWNER | _DACL | _PROTECTED_DACL,
                                             owner, None, dacl, None) if owner else 1
        if code:
            code = _advapi.SetNamedSecurityInfoW(path, _SE_FILE_OBJECT, _DACL | _PROTECTED_DACL,
                                                 None, None, dacl, None)
        if code:
            raise _winerr(f"SetNamedSecurityInfoW {path}", code)
    finally:
        _kernel.LocalFree(descriptor)


def dacl_sddl(path: str | os.PathLike) -> str:
    """The DACL of ``path`` as an SDDL string such as ``D:PAI(A;;FA;;;S-1-5-21-...)`` (Windows only)."""
    if not WINDOWS:
        raise OSError("DACLs exist on Windows only")
    descriptor = ctypes.c_void_p()
    code = _advapi.GetNamedSecurityInfoW(os.fspath(path), _SE_FILE_OBJECT, _DACL, None, None, None, None,
                                         ctypes.byref(descriptor))
    if code:
        raise _winerr(f"GetNamedSecurityInfoW {path}", code)
    try:
        text = wintypes.LPWSTR()
        if not _advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor, _SDDL_REVISION_1, _DACL, ctypes.byref(text), None):
            raise _winerr("ConvertSecurityDescriptorToStringSecurityDescriptorW")
        try:
            return text.value
        finally:
            _kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel.LocalFree(descriptor)


def is_private(path: str | os.PathLike) -> bool:
    """True when only the current user can reach ``path``.

    POSIX: no group or other permission bits. Windows: the DACL is protected (no inherited entries)
    and consists of exactly one allow entry for the current user.
    """
    if not WINDOWS:
        return not (os.lstat(path).st_mode & 0o077)
    if not volume_has_acls(path):
        return True                                   # FAT / exFAT: there are no permissions to check
    sddl = dacl_sddl(path)
    head = sddl.split("(", 1)[0]
    entries = re.findall(r"\(([^()]*)\)", sddl)
    if not head.startswith("D:") or "P" not in head[2:] or len(entries) != 1:
        return False
    fields = entries[0].split(";")
    return len(fields) >= 6 and fields[0] == "A" and _same_sid(fields[5], current_user_sid())


def is_unshared(path: str | os.PathLike) -> bool:
    """True when nobody but this user, SYSTEM and the administrators can reach ``path``.

    The relaxed form of :func:`is_private`, for files created under the user's profile where the
    inherited ACL already shuts out other accounts: any allow entry for Everyone, Users,
    Authenticated Users or another account makes it False. POSIX is the same check as
    :func:`is_private`.
    """
    if not WINDOWS:
        return is_private(path)
    if not volume_has_acls(path):
        return True                                   # FAT / exFAT: there are no permissions to check
    return not unshared_offenders(dacl_sddl(path))


def volume_has_acls(path: str | os.PathLike) -> bool:
    """False for a drive without file permissions (FAT, exFAT); True for NTFS/ReFS, and when it cannot be told (Windows only)."""
    if not WINDOWS:
        return True
    try:
        drive = os.path.splitdrive(os.path.abspath(os.fspath(path)))[0]
        if not drive:
            return True
        flags = wintypes.DWORD()
        fn = _kernel.GetVolumeInformationW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
        if not fn(drive.rstrip("\\") + "\\", None, 0, None, None, ctypes.byref(flags), None, 0):
            return True
        return bool(flags.value & 0x00000008)         # FILE_PERSISTENT_ACLS
    except (OSError, ValueError):
        return True


_ACCOUNT_NAMES = {"WD": "Everyone", "BU": "Users", "AU": "Authenticated Users", "IU": "Interactive users",
                  "NU": "Network users", "AN": "Anonymous", "DU": "Domain Users", "S-1-1-0": "Everyone",
                  "S-1-5-32-545": "Users", "S-1-5-11": "Authenticated Users", "S-1-5-4": "Interactive users"}


def _string_sid(trustee: str) -> str:
    """An SDDL trustee (a string SID or a two-letter alias such as LA, BA, SY, WD) as an upper-case string SID; the text itself
    when Windows does not know it."""
    if trustee.upper().startswith("S-1-"):
        return trustee.upper()
    pointer = ctypes.c_void_p()
    if not _advapi.ConvertStringSidToSidW(trustee, ctypes.byref(pointer)):
        return trustee.upper()
    try:
        text = wintypes.LPWSTR()
        if not _advapi.ConvertSidToStringSidW(pointer.value, ctypes.byref(text)):
            return trustee.upper()
        try:
            return text.value.upper()
        finally:
            _kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        _kernel.LocalFree(pointer)


def _is_privileged_sid(sid: str) -> bool:
    """SYSTEM, the Administrators group, the built-in Administrator account (RID 500, SDDL "LA"), OWNER RIGHTS / CREATOR OWNER / GROUP."""
    return sid in ("S-1-5-18", "S-1-5-32-544", "S-1-3-4", "S-1-3-0", "S-1-3-1") or \
        re.fullmatch(r"S-1-5-21-[0-9-]+-500", sid) is not None


def unshared_offenders(sddl: str) -> list[str]:
    """The trustees of an SDDL DACL that make the object reachable by someone besides this user, SYSTEM and the administrators:
    every allow entry that applies to the object itself, whatever form its account is written in (a string SID or an alias such
    as LA).  An inherit-only entry (flag IO) is for the children of a folder, not for the folder, and is left out."""
    me = current_user_sid().upper()
    out: list[str] = []
    for entry in re.findall(r"\(([^()]*)\)", sddl):
        fields = entry.split(";")
        if len(fields) < 6 or fields[0] not in ("A", "OA") or "IO" in re.findall("..", fields[1].upper()):
            continue
        sid = _string_sid(fields[5])
        if sid == me or _is_privileged_sid(sid) or fields[5].upper() in ("SY", "BA", "OW", "CO", "CG", "LA"):
            continue
        if fields[5] not in out:
            out.append(fields[5])
    return out


def account_name(trustee: str) -> str:
    """A plain name for an SDDL trustee (Everyone, Users, DOMAIN\\name), or the SID text when it cannot be looked up."""
    plain = _ACCOUNT_NAMES.get(trustee.upper()) or _ACCOUNT_NAMES.get(_string_sid(trustee))
    if plain or not WINDOWS:
        return plain or trustee
    pointer = ctypes.c_void_p()
    if not _advapi.ConvertStringSidToSidW(trustee, ctypes.byref(pointer)):
        return trustee
    try:
        name, domain = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
        n, d, use = wintypes.DWORD(256), wintypes.DWORD(256), wintypes.DWORD()
        lookup = _advapi.LookupAccountSidW
        lookup.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
                           wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD)]
        if lookup(None, pointer, name, ctypes.byref(n), domain, ctypes.byref(d), ctypes.byref(use)):
            return f"{domain.value}\\{name.value}" if domain.value else name.value
        return trustee
    finally:
        _kernel.LocalFree(pointer)


def who_can_reach(path: str | os.PathLike) -> list[str]:
    """Plain names of the accounts besides this user, SYSTEM and the administrators that can reach ``path`` (Windows; [] elsewhere)."""
    if not WINDOWS:
        return []
    return [account_name(t) for t in unshared_offenders(dacl_sddl(path))]


# ---------------------------------------------------------------------------------------------- access repair

def _user_psid():
    """The current user's SID as a native pointer (kept for the life of the process), or None."""
    global _psid_cache
    if not WINDOWS:
        return None
    if _psid_cache is not None:
        return _psid_cache
    pointer = ctypes.c_void_p()
    if not _advapi.ConvertStringSidToSidW(current_user_sid(), ctypes.byref(pointer)):
        return None
    _psid_cache = pointer.value
    return _psid_cache


_psid_cache = None
_keep: list = []


def private_mkdir(path: str | os.PathLike, *, parents: bool = False, exist_ok: bool = True) -> Path:
    """Create a folder only the user can reach, on every platform.

    POSIX: ``mkdir(mode=0o700)``. Windows: Python 3.13 turns ``mode=0o700`` into an ACL of SYSTEM, Administrators and OWNER
    RIGHTS (inherited by everything inside), which locks the user out when an elevated process created the folder. This
    creates the missing folders plainly and gives each one the explicit entry for the user's own SID (and the user as owner).
    """
    target = Path(path)
    if not WINDOWS:
        target.mkdir(mode=0o700, parents=parents, exist_ok=exist_ok)
        return target
    missing = []
    probe = target
    while not probe.exists() and probe != probe.parent:
        missing.append(probe)
        if not parents:
            break
        probe = probe.parent
    if not missing and not exist_ok:
        raise FileExistsError(str(target))
    for folder in reversed(missing):
        folder.mkdir(exist_ok=exist_ok)
        try:
            restrict_to_current_user(folder, directory=True)
        except OSError:
            pass            # the folder exists with its inherited rule; a later read repairs it if that is not enough
    return target


def adopt_user_defaults() -> bool:
    """Make files this process creates belong to, and be open to, the user (Windows; no effect elsewhere).

    An elevated process otherwise creates files owned by BUILTIN\\Administrators with a default rule of SYSTEM and
    Administrators only. Setting the token's default owner and default rule to the user's own SID fixes that for every file
    the process (and what it starts) creates afterwards. Returns True when both were set; never raises.
    """
    if not WINDOWS:
        return False
    try:
        sid = current_user_sid()
        psid = _user_psid()
        token = wintypes.HANDLE()
        if not psid or not _advapi.OpenProcessToken(_kernel.GetCurrentProcess(), _TOKEN_QUERY | _TOKEN_ADJUST_DEFAULT,
                                                    ctypes.byref(token)):
            return False
        try:
            class _Owner(ctypes.Structure):
                _fields_ = [("owner", ctypes.c_void_p)]

            class _DefaultDacl(ctypes.Structure):
                _fields_ = [("dacl", ctypes.c_void_p)]

            ok_owner = bool(_advapi.SetTokenInformation(token, _TOKEN_OWNER, ctypes.byref(_Owner(psid)),
                                                        ctypes.sizeof(_Owner)))
            descriptor = ctypes.c_void_p()
            # A token default DACL also applies to processes, not just files. FA is
            # FILE_ALL_ACCESS (0x1f01ff), which omits process query/set-information.
            # GA is mapped to the full rights of the object being created.
            if not _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    f"D:(A;;GA;;;{sid})(A;;GA;;;SY)(A;;GA;;;BA)", _SDDL_REVISION_1, ctypes.byref(descriptor), None):
                return False
            present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
            if not _advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl),
                                                     ctypes.byref(defaulted)) or not present.value:
                return False
            _keep.append(descriptor)           # the token keeps pointing into it
            ok_dacl = bool(_advapi.SetTokenInformation(token, _TOKEN_DEFAULT_DACL, ctypes.byref(_DefaultDacl(dacl.value)),
                                                       ctypes.sizeof(_DefaultDacl)))
            return ok_owner and ok_dacl
        finally:
            _kernel.CloseHandle(token)
    except OSError:
        return False


def _enable_privilege(name: str) -> bool:
    token = wintypes.HANDLE()
    if not _advapi.OpenProcessToken(_kernel.GetCurrentProcess(), _TOKEN_QUERY | _TOKEN_ADJUST_PRIVILEGES,
                                    ctypes.byref(token)):
        return False
    try:
        class _Luid(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", ctypes.c_long)]

        class _Privileges(ctypes.Structure):
            _fields_ = [("count", wintypes.DWORD), ("luid", _Luid), ("attributes", wintypes.DWORD)]

        luid = _Luid()
        if not _advapi.LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
            return False
        privileges = _Privileges(1, luid, 2)  # SE_PRIVILEGE_ENABLED
        ctypes.set_last_error(0)
        _advapi.AdjustTokenPrivileges(token, False, ctypes.byref(privileges), ctypes.sizeof(privileges), None, None)
        return ctypes.get_last_error() == 0
    finally:
        _kernel.CloseHandle(token)


def user_roots() -> list[Path]:
    """The folders that are the user's own per-user data: a file inside one may be repaired by that user."""
    roots: list[Path] = []
    for key in ("APPDATA", "LOCALAPPDATA", "STRATA_HOME", "STRATA_PROFILE_DIR", "STRATA_INSTALL_DIR", "STRATA_RUNTIME_DIR",
                "STRATA_STATE_DIR"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    roots.append(Path.home())
    result = []
    for root in roots:
        try:
            result.append(root.resolve())
        except OSError:
            continue
    return result


def is_users_own(path: str | os.PathLike) -> bool:
    """True when ``path`` is inside the user's own profile or data folders (never the system folders)."""
    try:
        target = Path(path).resolve()
    except OSError:
        return False
    return any(target == root or root in target.parents for root in user_roots())


def repair_access(path: str | os.PathLike, *, tree: bool = True) -> bool:
    """Give the current user back a file or folder in their own data folders that an elevated process locked them out of.

    Takes ownership and writes the explicit protected entry for the user's SID (a folder's children are repaired too).
    Needs the right to do that: an elevated session, or a user holding the take-ownership right; a normal session locked
    out by someone else's elevated process cannot do it, and this returns False (the caller shows the plain message).
    Windows only; elsewhere there is nothing to repair. Refuses anything outside the user's own folders.
    """
    if not WINDOWS or not is_users_own(path):
        return False
    _enable_privilege("SeTakeOwnershipPrivilege")
    _enable_privilege("SeRestorePrivilege")
    fixed = True
    target = Path(path)
    try:
        restrict_to_current_user(target, directory=target.is_dir())
    except OSError:
        fixed = False
    if tree and fixed and target.is_dir():
        try:
            for dirpath, dirnames, filenames in os.walk(target):
                for name in dirnames + filenames:
                    child = os.path.join(dirpath, name)
                    if os.path.islink(child):
                        continue
                    try:
                        restrict_to_current_user(child, directory=os.path.isdir(child))
                    except OSError:
                        fixed = False
        except OSError:
            fixed = False
    return fixed


def read_bytes_repairing(path: str | os.PathLike) -> bytes:
    """``Path.read_bytes`` that, on an access error in the user's own folders, repairs the access once and tries again."""
    target = Path(path)
    try:
        return target.read_bytes()
    except PermissionError:
        if not repair_access(target, tree=False):
            raise
        return target.read_bytes()


ACCESS_MESSAGE = ("Strata is not allowed to open one of its own files. Run `strata repair-access` (or press Repair in the app); "
                "these files were probably made while Strata ran as administrator, so run it as your normal user from now on.")
IN_USE_MESSAGE = ("A file Strata needs is in use by another program (usually antivirus, a backup or a sync tool). "
                  "Wait a moment and run the same command again.")


def is_sharing_violation(error: BaseException) -> bool:
    """Windows "file in use" (WinError 32 or 33), as opposed to a real refusal (WinError 5): waiting helps, repairing does not."""
    code = getattr(error, "winerror", None)
    if code in (32, 33):
        return True
    text = str(error)
    return "WinError 32" in text or "WinError 33" in text or "being used by another process" in text


_SKIP_DIRS = {"models", "node_modules", "harness", "python", "host", "node", "control", "__pycache__", ".git"}
_SKIP_SUFFIXES = (".gguf", ".part", ".bin", ".safetensors", ".zip", ".exe", ".dll")


def known_roots() -> list[Path]:
    """Strata's per-user data folders: APPDATA/Strata, LOCALAPPDATA/Strata and ~/.strata (plus the STRATA_* overrides)."""
    roots: list[Path] = []
    for key, sub in (("APPDATA", "Strata"), ("LOCALAPPDATA", "Strata")):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value) / sub)
    roots.append(Path.home() / ".strata")
    for key in ("STRATA_HOME", "STRATA_PROFILE_DIR", "STRATA_RUNTIME_DIR", "STRATA_STATE_DIR", "STRATA_INSTALL_DIR"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value))
    seen, result = set(), []
    for root in roots:
        key = str(root).lower()
        if key not in seen and root.exists():
            seen.add(key)
            result.append(root)
    return result


def repair_known_roots(*, limit: int = 20000) -> dict:
    """Walk Strata's own per-user folders and repair every entry the user cannot open (Windows).

    Meant to run when Strata starts: files and folders an elevated run created (owner Administrators, a rule that names
    only SYSTEM, Administrators and OWNER RIGHTS) are given back to the user. A normal session can only repair what its
    own token may change; an elevated start repairs everything. Never raises; returns what it did.
    """
    report = {"checked": 0, "repaired": 0, "failed": 0}
    if not WINDOWS:
        return report
    for root in known_roots():
        stack = [root]
        while stack and report["checked"] < limit:
            folder = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except PermissionError:
                report["checked"] += 1
                if repair_access(folder, tree=False):
                    report["repaired"] += 1
                    try:
                        entries = list(os.scandir(folder))
                    except OSError:
                        continue
                else:
                    report["failed"] += 1
                    continue
            except OSError:
                continue
            for entry in entries:
                report["checked"] += 1
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.lower() not in _SKIP_DIRS:
                            stack.append(Path(entry.path))
                        continue
                    if entry.name.lower().endswith(_SKIP_SUFFIXES):
                        continue
                    with open(entry.path, "rb"):
                        pass
                except PermissionError:
                    if repair_access(entry.path, tree=False):
                        report["repaired"] += 1
                    else:
                        report["failed"] += 1
                except OSError:
                    continue
    return report
