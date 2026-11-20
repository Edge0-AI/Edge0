#!/usr/bin/env python3
"""Check a pull request against the intake rules in CONTRIBUTING.md.

Reads the workflow event. Applies or removes the needs-info label and updates
a single comment. Exits 0 when the pull request passes.

The cap on open pull requests and the issue-link rule are skipped for
OWNER, MEMBER, and COLLABORATOR. Title, test notes, and perf evidence apply
to everyone else. Pull requests opened before APPLIES_SINCE are not checked
and do not count toward the cap.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

MAX_OPEN = 2
MARKER = "<!-- pr-hygiene -->"
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}
NEEDS_INFO = "needs-info"
# Pull requests opened before this instant keep the previous terms. They are
# not checked, labeled, or counted toward the two-open limit. This is after
# every pull request that was already open on 2026-11-19.
APPLIES_SINCE = datetime(2026, 11, 19, 8, 0, tzinfo=timezone.utc)

TITLE_RE = re.compile(
    r"^(?:\[[A-Za-z][A-Za-z0-9 _./-]{0,24}\]\s+)?"
    r"(?P<type>fix|feat|perf|docs|chore|test|tests|ci|build|refactor|revert)"
    r"(?:\([A-Za-z0-9][A-Za-z0-9._/-]{0,40}\))?"
    r":\s+\S"
)
ISSUE_RE = re.compile(
    r"\b(?:fixes|fixed|fix|closes|closed|close|resolves|resolved|resolve|related to)"
    r"\s+(?:#\d+|https://github\.com/Edge0-AI/Edge0/(?:issues|pull)/\d+)",
    re.IGNORECASE,
)
SECTION_RE = re.compile(r"^## How was it tested\?\s*$", re.MULTILINE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", text)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_grandfathered(pr: dict) -> bool:
    created = pr.get("created_at") or ""
    if not created:
        return False
    try:
        return parse_time(created) < APPLIES_SINCE
    except ValueError:
        return False


def title_type(title: str) -> str | None:
    match = TITLE_RE.search(title.strip())
    if not match:
        return None
    return match.group("type")


def tested_section(body: str) -> str | None:
    match = SECTION_RE.search(body)
    if not match:
        return None
    rest = body[match.end() :]
    nxt = re.search(r"^## ", rest, re.MULTILINE)
    chunk = rest[: nxt.start()] if nxt else rest
    return strip_comments(chunk).strip()


def evaluate(
    title: str,
    body: str,
    association: str,
    open_count: int | None,
    over_limit: bool,
) -> list[str]:
    """Return human-readable problems. An empty list means the PR passes."""
    problems: list[str] = []
    kind = title_type(title)
    maintainer = association in MAINTAINERS

    if kind is None:
        problems.append(
            "Title must look like `type(scope): summary`. "
            "type is fix, feat, perf, docs, chore, test, tests, ci, build, "
            "refactor, or revert. An `[Area]` prefix is optional. "
            "Example: `fix(serve): accept empty reasoning history`."
        )

    if not maintainer and kind != "docs" and not ISSUE_RE.search(body or ""):
        problems.append(
            "Link an issue with `Fixes #123` or `Related to #123`. "
            "For a behavior change, open the issue and wait for a maintainer "
            "to agree before writing the code. A `docs` title may omit this."
        )

    notes = tested_section(body or "")
    if notes is None:
        problems.append(
            'Add a "## How was it tested?" section, using that heading.'
        )
    elif not notes:
        problems.append(
            'Replace the template comment under "## How was it tested?" '
            "with the commands, platform, and results you actually ran."
        )

    visible = strip_comments(body or "")
    if kind == "perf" and "bench" not in visible.lower():
        problems.append(
            "A perf pull request needs a bench summary in the body: "
            "`edge0-infer bench`, or the community benchmark report "
            "(BENCH.bat / ./bench.sh) for an edge0-strata model."
        )

    if not maintainer and over_limit:
        detail = f" ({open_count} open)" if open_count is not None else ""
        problems.append(
            f"This pull request is past the limit of {MAX_OPEN} open pull requests{detail}. "
            f"The {MAX_OPEN} oldest stay in the queue. Close this one, or fold the change "
            "into an older pull request."
        )

    return problems


def github_request(method: str, url: str, token: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "edge0-pr-hygiene",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            parsed = json.loads(raw) if raw else None
            return resp.status, parsed
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> {exc.code} {detail}") from exc


def list_open_pulls(repo: str, token: str) -> list[dict] | None:
    """Every open pull request, or None when the list could not be read."""
    owner, name = repo.split("/", 1)
    pulls: list[dict] = []
    page = 1
    try:
        while True:
            url = (
                f"https://api.github.com/repos/{owner}/{name}/pulls"
                f"?state=open&per_page=100&page={page}"
            )
            _status, payload = github_request("GET", url, token)
            if not isinstance(payload, list):
                print("warning: open pull request list was not a list", file=sys.stderr)
                return None
            pulls.extend(item for item in payload if isinstance(item, dict))
            if len(payload) < 100:
                return pulls
            page += 1
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"warning: could not list open pull requests: {exc}", file=sys.stderr)
        return None


def cap_status(pr: dict, pulls: list[dict] | None) -> tuple[int | None, bool]:
    """Return (open count, whether this PR is beyond the two oldest)."""
    if pulls is None:
        return None, False
    login = (pr.get("user") or {}).get("login") or ""
    number = pr.get("number")
    if not login:
        return None, False
    mine = [
        item
        for item in pulls
        if (item.get("user") or {}).get("login") == login and not is_grandfathered(item)
    ]
    if (
        isinstance(number, int)
        and not is_grandfathered(pr)
        and not any(item.get("number") == number for item in mine)
    ):
        mine.append(pr)
    if is_grandfathered(pr):
        return len(mine), False
    mine.sort(key=lambda item: (item.get("created_at") or "", item.get("number") or 0))
    keep = {item.get("number") for item in mine[:MAX_OPEN]}
    return len(mine), number not in keep


def ensure_needs_info_label(repo: str, token: str) -> None:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/labels"
    try:
        github_request(
            "POST",
            url,
            token,
            {
                "name": NEEDS_INFO,
                "color": "d93f0b",
                "description": "Title, issue link, test notes, or PR cap still missing",
            },
        )
    except RuntimeError as exc:
        if "-> 422" not in str(exc):
            raise


def set_needs_info(repo: str, number: int, token: str, wanted: bool) -> None:
    owner, name = repo.split("/", 1)
    label = urllib.parse.quote(NEEDS_INFO)
    base = f"https://api.github.com/repos/{owner}/{name}/issues/{number}/labels"
    if wanted:
        ensure_needs_info_label(repo, token)
        github_request("POST", base, token, {"labels": [NEEDS_INFO]})
        return
    try:
        github_request("DELETE", f"{base}/{label}", token)
    except RuntimeError as exc:
        # Already absent.
        if "-> 404" not in str(exc):
            raise


def http_status(exc: BaseException) -> int | None:
    match = re.search(r"-> (\d{3})\b", str(exc))
    if not match:
        return None
    return int(match.group(1))


def comment_body(problems: list[str]) -> str:
    lines = [
        MARKER,
        "This pull request is labeled `needs-info` and is not in the review queue.",
        "The intake rules are in [CONTRIBUTING.md](https://github.com/Edge0-AI/Edge0/blob/main/CONTRIBUTING.md#pull-request-workflow).",
        "",
    ]
    lines.extend(f"- {item}" for item in problems)
    return "\n".join(lines) + "\n"


def passed_comment_body() -> str:
    return (
        f"{MARKER}\n"
        "This pull request passes the intake check.\n"
    )


def sync_comment(repo: str, number: int, token: str, problems: list[str]) -> None:
    owner, name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{name}/issues/{number}/comments"
    _status, comments = github_request("GET", f"{url}?per_page=100", token)
    existing = None
    if isinstance(comments, list):
        for comment in comments:
            body = comment.get("body") or ""
            author = (comment.get("user") or {}).get("login") or ""
            if MARKER in body and author == "github-actions[bot]":
                existing = comment
                break
    if not problems:
        clear_passed_comment(url, existing, token)
        return
    payload = {"body": comment_body(problems)}
    if existing:
        github_request("PATCH", f"{url}/{existing['id']}", token, payload)
    else:
        github_request("POST", url, token, payload)


def clear_passed_comment(comments_url: str, existing: dict | None, token: str) -> None:
    """Remove the needs-info comment once the pull request passes.

    The Actions token can edit its own comment, and GitHub often answers
    DELETE with 404 even while the comment is still there. A 404 must not
    fail the check. Fall back to rewriting the comment.
    """
    if not existing:
        return
    comment_url = f"{comments_url}/{existing['id']}"
    try:
        github_request("DELETE", comment_url, token)
        return
    except RuntimeError as exc:
        if http_status(exc) != 404:
            raise
    try:
        github_request("PATCH", comment_url, token, {"body": passed_comment_body()})
    except RuntimeError as exc:
        if http_status(exc) != 404:
            raise
        print(
            "warning: the previous hygiene comment could not be updated",
            file=sys.stderr,
        )


def problems_for(pr: dict, pulls: list[dict] | None) -> list[str]:
    count, over_limit = cap_status(pr, pulls)
    return evaluate(
        pr.get("title") or "",
        pr.get("body") or "",
        pr.get("author_association") or "",
        count,
        over_limit,
    )


def apply_result(repo: str, number: int, token: str, problems: list[str]) -> None:
    set_needs_info(repo, number, token, bool(problems))
    sync_comment(repo, number, token, problems)


def check_event(event: dict, repo: str, token: str | None, apply: bool) -> int:
    pr = event.get("pull_request")
    if not isinstance(pr, dict):
        print("event has no pull_request", file=sys.stderr)
        return 1
    number = pr.get("number")
    if is_grandfathered(pr):
        print(
            f"PR #{number} opened before {APPLIES_SINCE.isoformat()}; "
            "hygiene does not apply"
        )
        return 0
    pulls = list_open_pulls(repo, token) if token else None
    problems = problems_for(pr, pulls)
    if problems:
        print(f"PR #{number} needs info:")
        for item in problems:
            print(f"- {item}")
    else:
        print(f"PR #{number} passes hygiene")
    if apply:
        if not token or not isinstance(number, int):
            print("cannot apply labels without a token and pull request number", file=sys.stderr)
            return 1
        apply_result(repo, number, token, problems)
    return 1 if problems else 0


def sweep(repo: str, token: str, apply: bool) -> int:
    pulls = list_open_pulls(repo, token)
    if pulls is None:
        return 1
    failed = 0
    skipped = 0
    checked = 0
    for pr in pulls:
        number = pr.get("number")
        login = (pr.get("user") or {}).get("login") or "?"
        if is_grandfathered(pr):
            skipped += 1
            continue
        checked += 1
        problems = problems_for(pr, pulls)
        if problems:
            failed += 1
            print(f"#{number} {login} needs info ({len(problems)})")
            if apply and isinstance(number, int):
                apply_result(repo, number, token, problems)
        else:
            print(f"#{number} {login} passes")
            if apply and isinstance(number, int):
                apply_result(repo, number, token, [])
    print(
        f"sweep: {skipped} opened before the rules, "
        f"{failed} of {checked} later pull requests need info"
    )
    return 0


def self_test() -> int:
    ok_body = (
        "## What & why\n\nFixes #12\n\n"
        "## How was it tested?\n\nctest --preset cpu -L L0 passed on macOS.\n"
    )
    cases: list[tuple[str, str, str, int | None, int]] = [
        ("fix(serve): accept empty history", ok_body, "CONTRIBUTOR", 1, 0),
        ("fix(serve): accept empty history", ok_body, "CONTRIBUTOR", 3, 1),
        ("fix(serve): accept empty history", ok_body, "COLLABORATOR", 9, 0),
        ("docs(readme): add a Korean translation", "## How was it tested?\n\nRead the rendered page.\n", "NONE", 1, 0),
        ("feat(server): add a flag", "## How was it tested?\n\nran it\n", "NONE", 1, 1),
        ("[Engine] fix(mcp): bound calls", ok_body, "NONE", 2, 0),
        ("Feat/latent multigpu", ok_body, "NONE", 1, 1),
        ("perf(text): skip a buffer", ok_body, "NONE", 1, 1),
        (
            "perf(text): skip a buffer",
            ok_body + "\nedge0-infer bench: 20 tok/s to 24 tok/s\n",
            "NONE",
            1,
            0,
        ),
        (
            "perf(decode): faster mask",
            "Fixes #3\n## How was it tested?\n<!-- edge0-infer bench -->\n",
            "NONE",
            1,
            2,
        ),
        ("ci: run L0 on pull requests", "Related to #9\n## How was it tested?\n\nCI on this PR.\n", "NONE", 1, 0),
    ]
    # The last perf-with-only-a-comment case expects 2 problems (empty notes + bench).
    failed = 0
    for title, body, association, count, expected in cases:
        over_limit = count is not None and count > MAX_OPEN
        got = len(evaluate(title, body, association, count, over_limit))
        if got != expected:
            failed += 1
            print(f"FAIL {title!r} association={association} count={count}: got {got}, want {expected}")
            for item in evaluate(title, body, association, count, over_limit):
                print(f"  - {item}")
    author = {"login": "a"}
    pulls = [
        {"number": 1, "created_at": "2026-11-20T00:00:00Z", "user": author},
        {"number": 2, "created_at": "2026-11-21T00:00:00Z", "user": author},
        {"number": 3, "created_at": "2026-11-22T00:00:00Z", "user": author},
    ]
    cap_cases = (
        (pulls[0], (3, False)),
        (pulls[1], (3, False)),
        (pulls[2], (3, True)),
        ({"number": 4, "created_at": "2026-11-23T00:00:00Z", "user": author}, (4, True)),
        ({"number": 5, "created_at": "2026-11-24T00:00:00Z", "user": {"login": "b"}}, (1, False)),
    )
    for pr, expected_cap in cap_cases:
        got_cap = cap_status(pr, pulls)
        if got_cap != expected_cap:
            failed += 1
            print(f"FAIL cap #{pr['number']}: got {got_cap}, want {expected_cap}")
    if cap_status(pulls[0], None) != (None, False):
        failed += 1
        print("FAIL cap when the pull request list is unavailable")
    old = {"number": 9, "created_at": "2026-11-18T20:25:36Z", "user": author}
    if not is_grandfathered(old) or is_grandfathered(pulls[0]):
        failed += 1
        print("FAIL grandfather boundary")
    if cap_status(pulls[0], [old, pulls[0]]) != (1, False):
        failed += 1
        print("FAIL grandfathered pull requests must not count toward the cap")
    if http_status(RuntimeError("DELETE https://example -> 404 {\"message\": \"Not Found\"}")) != 404:
        failed += 1
        print("FAIL http status parse")
    if MARKER not in passed_comment_body():
        failed += 1
        print("FAIL passed comment lost its marker")
    if failed:
        print(f"{failed} self-test case(s) failed")
        return 1
    print(f"{len(cases)} self-test cases passed")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update the needs-info label and the hygiene comment",
    )
    parser.add_argument("--event", help="Path to a workflow event JSON file")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()

    token = os.environ.get("GITHUB_TOKEN") or None
    repo = os.environ.get("GITHUB_REPOSITORY") or ""
    if not repo:
        print("GITHUB_REPOSITORY is not set", file=sys.stderr)
        return 1
    if args.sweep:
        if not token:
            print("GITHUB_TOKEN is required for --sweep", file=sys.stderr)
            return 1
        return sweep(repo, token, args.apply)

    event_path = args.event or os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("GITHUB_EVENT_PATH is not set", file=sys.stderr)
        return 1
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)
    return check_event(event, repo, token, args.apply)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
