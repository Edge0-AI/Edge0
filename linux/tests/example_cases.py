"""Original synthetic prompts and ground truth for the Linux engine examples."""

import json


CASES = [
    {"id": "books", "title": "Books for AI coding agents", "prompt": (
        'Write a short handbook for AI coding agents titled "Read the codebase before you change it". '
        "Give a five-chapter outline, one short line per chapter, then the first approximately 150 words "
        "of chapter 1. Give practical instructions an agent can follow in an unfamiliar repository. "
        "Output only the outline and chapter excerpt.")},
    {"id": "marketing", "title": "Marketing", "prompt": (
        "Write a 60-80 word LinkedIn post announcing an open-source tool that runs a mixture-of-experts "
        "language model on a laptop CPU. Emphasize local use, privacy, and trying it out. "
        "Do not invent benchmark numbers or claim it works on every laptop. "
        "After the post, put exactly five relevant hashtags on a separate line.")},
    {"id": "reviews", "title": "Product-review tagging", "prompt": (
        "Label each product review with sentiment (positive, negative, or neutral) and topic "
        "(price, quality, shipping, or support). Return only a JSON array of eight objects, "
        "each with integer id and string sentiment and topic. Do not include explanations.\n"
        "1. The casing is sturdy and the buttons feel excellent.\n"
        "2. The handle snapped during the first use.\n"
        "3. This is a great deal for the price.\n"
        "4. It costs far too much for what you get.\n"
        "5. My package arrived two days early, which was wonderful.\n"
        "6. Delivery took three weeks longer than promised.\n"
        "7. Customer support hours are Monday through Friday, 9 to 5.\n"
        "8. Support ignored all three of my emails; very frustrating."),
     "labels": [{"sentiment": "positive", "topic": "quality"},
                {"sentiment": "negative", "topic": "quality"},
                {"sentiment": "positive", "topic": "price"},
                {"sentiment": "negative", "topic": "price"},
                {"sentiment": "positive", "topic": "shipping"},
                {"sentiment": "negative", "topic": "shipping"},
                {"sentiment": "neutral", "topic": "support"},
                {"sentiment": "negative", "topic": "support"}]},
    {"id": "commits", "title": "Coding-agent commit tagging", "prompt": (
        "Tag these commit messages by Conventional Commit type. Allowed types: feat, fix, docs, "
        "refactor, test, perf, build, ci. Return only a JSON array of eight objects, "
        "each with integer id and string type. Do not include explanations.\n"
        "1. feat(api): add a batch tagging endpoint\n"
        "2. fix(parser): handle an empty input file\n"
        "3. docs(readme): explain the Linux setup steps\n"
        "4. refactor(cache): extract eviction logic without changing behavior\n"
        "5. test(server): cover request timeout handling\n"
        "6. perf(router): avoid allocating a vector for each token\n"
        "7. build(cmake): link the POSIX threading library\n"
        "8. ci(actions): run the Linux smoke job on pull requests"),
     "labels": [{"type": value} for value in ("feat", "fix", "docs", "refactor", "test", "perf", "build", "ci")]},
]


def parse_tags(case, text):
    rows = json.loads(text)
    fields = set(case["labels"][0])
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("expected a JSON array of eight objects")
    if any(not isinstance(row, dict) or set(row) != fields | {"id"} for row in rows):
        raise ValueError("incorrect object fields")
    if any(type(row["id"]) is not int for row in rows) or {row["id"] for row in rows} != set(range(1, 9)):
        raise ValueError("expected unique integer IDs 1 through 8")
    return {row["id"]: row for row in rows}


def unwrap_fence(text):
    lines = text.strip().splitlines()
    fenced = len(lines) >= 3 and lines[0] in ("```json", "```") and lines[-1] == "```"
    return ("\n".join(lines[1:-1]) if fenced else text), fenced


def score_tags(case, text):
    fields = dict.fromkeys(case["labels"][0], 0)
    text, fenced = unwrap_fence(text)
    score = dict(valid=False, fence_removed=fenced, exact=0, total=8, fields=fields, mismatches=[])
    try:
        rows = parse_tags(case, text)
    except (ValueError, TypeError) as error:
        return dict(score, error=str(error))
    score["valid"] = not fenced
    for index, expected in enumerate(case["labels"], 1):
        matches = {key: rows[index][key] == value for key, value in expected.items()}
        for key, match in matches.items():
            fields[key] += int(match)
        score["exact"] += int(all(matches.values()))
        if not all(matches.values()):
            score["mismatches"].append(dict(id=index, expected=expected, actual=rows[index]))
    return score
