"""The engine options a person set by hand in a model's run config (strata-<model>.json).

Setup and the app write ``"args"`` from their own choices, and used to write it again on every run, so an option added by
hand (the control vector options, a --vram-reserve-mib of 0, --spec, ...) was gone after the next Load (#3216 #3215 #3270).
The run config now keeps what the person set apart from what was generated:

* ``"user_args"``: the person's options, in the same form as ``"args"`` (``["--vram-reserve-mib", "0", ...]``).  Every start
  (serve/server.py engine_args) and every setup run applies them on top of the generated ones; for the same flag the
  person's value wins.  The person can edit this list by hand; the app's settings window writes the options it shows here.
* ``"args_written"``: the whole ``"args"`` as setup or the app last wrote it.  An option in ``"args"`` that is not in this
  copy was added or changed by hand since: ``capture`` moves it into ``"user_args"``.  A config without the copy (one an
  older Strata wrote) has nothing to compare with: setup then keeps the options of the old file that its new arguments do not
  have (see ``dropped_by_setup``).

Writers that change ``"args"`` themselves (setup, the app's resource profile, a tuning run) call ``stamp`` after they have
captured, so their own changes are never mistaken for the person's.
"""

from __future__ import annotations

import re
from typing import Any

USER_KEY = "user_args"
WRITTEN_KEY = "args_written"

#: option groups the engine takes more than one of, or that make sense only together: kept whole
#: (the control vector options of the experimental speed projection and the person's own vectors, #3216)
USER_KEPT_FLAGS = frozenset({"--control-vector", "--control-vector-scaled", "--control-vector-layer-range", "--cvec-mode",
                             "--cvec-dir"})


#: options that take more than one value, and how many (``--control-vector-layer-range 4 44``): written as separate argv
#: entries, never as one ``4,44``, and removed with all of their values (#3504)
MULTI_VALUE_FLAGS = {"--control-vector-layer-range": 2}


def value_tokens(flag: str, value: Any) -> list[str]:
    """The argv entries that follow ``flag`` for ``value``: one, or the several a multi-value option takes (``4 44``, ``4,44``,
    ``4-44`` or a list)."""
    n = MULTI_VALUE_FLAGS.get(flag, 1)
    if n == 1:
        return [str(value)]
    parts = [str(v) for v in value] if isinstance(value, (list, tuple)) else re.split(r"[,;\s]+|(?<=\d)-(?=\d)", str(value).strip())
    return [t for t in parts if t]


def groups(args: Any) -> list[list[str]]:
    """Split an argument list into groups: an option and the values after it (up to the next ``--option``)."""
    out: list[list[str]] = []
    for a in args if isinstance(args, list) else []:
        a = str(a)
        if a.startswith("--") or not out:
            out.append([a])
        else:
            out[-1].append(a)
    return out


def flat(gs: list[list[str]]) -> list[str]:
    return [t for g in gs for t in g]


def flag_of(group: list[str]) -> str:
    return group[0] if group and group[0].startswith("--") else ""


def merge(args: Any, user: Any) -> list[str]:
    """``args`` with the person's options on top: every group of ``args`` whose option the person set is replaced."""
    ug = groups(user)
    mine = {flag_of(g) for g in ug if flag_of(g)}
    kept = [g for g in groups(args) if flag_of(g) not in mine]
    return flat(kept + ug)


def apply(cfg: dict) -> list[str]:
    """Puts the config's ``user_args`` over its ``args`` (in place); returns the options applied."""
    ug = groups(cfg.get(USER_KEY))
    if not ug or not isinstance(cfg.get("args"), list):
        return []
    cfg["args"] = merge(cfg["args"], flat(ug))
    return [flag_of(g) for g in ug if flag_of(g)]


def _by_flag(gs: list[list[str]]) -> dict[str, list[list[str]]]:
    d: dict[str, list[list[str]]] = {}
    for g in gs:
        d.setdefault(flag_of(g), []).append(g)
    return d


def capture(cfg: dict, skip=frozenset()) -> list[str]:
    """The person's options of a config: its ``user_args``, with the options of ``args`` that differ from ``args_written``
    (added or changed by hand since the last write) on top.  Without ``args_written`` only ``user_args`` is returned.
    ``skip``: options the caller chooses itself (setup's context, KV type, pictures, draft layer), never taken from ``args``."""
    mine = _by_flag(groups(cfg.get(USER_KEY)))
    written = cfg.get(WRITTEN_KEY)
    if isinstance(written, list) and isinstance(cfg.get("args"), list):
        cur, was = _by_flag(groups(cfg["args"])), _by_flag(groups(written))
        for flag, gs in cur.items():
            if flag and flag not in skip and was.get(flag) != gs:
                mine[flag] = gs                       # added, or its value changed, by hand
        for flag in list(mine):
            if flag in was and flag not in cur:
                del mine[flag]                        # taken out of "args" by hand: the person no longer wants it
    return flat([g for gs in mine.values() for g in gs])


def stamp(cfg: dict, user: list[str] | None = None) -> None:
    """Records ``cfg["args"]`` as written (call after the last change to it) and ``user`` as the person's options."""
    if user is not None:
        if user:
            cfg[USER_KEY] = list(user)
        else:
            cfg.pop(USER_KEY, None)
    if isinstance(cfg.get("args"), list):
        cfg[WRITTEN_KEY] = list(cfg["args"])


def dropped_by_setup(old_args: Any, new_args: Any, setup_flags: set[str] | frozenset[str], skip=frozenset()) -> list[str]:
    """For an old config with no ``args_written``: its options that the new arguments do not have and that setup never
    writes (``setup_flags``: every option setup.py names), plus the control vector options.  These were added by hand."""
    have = {flag_of(g) for g in groups(new_args)}
    out = [g for g in groups(old_args) if flag_of(g) and flag_of(g) not in have and flag_of(g) not in skip
           and (flag_of(g) not in setup_flags or flag_of(g) in USER_KEPT_FLAGS)]
    return flat(out)


def set_flag(user: list[str], flag: str, value: Any) -> list[str]:
    """``user`` with ``flag`` set to ``value`` (None: the person's option removed)."""
    rest = [g for g in groups(user) if flag_of(g) != flag]
    if value is not None:
        rest.append([flag, *value_tokens(flag, value)])
    return flat(rest)
