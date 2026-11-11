"""docs/TROUBLESHOOTING.md keeps every entry the 0.2.5 rewrite lost (restored for 0.2.7), and every pointer to it resolves.

setup.py and the server print "see docs/TROUBLESHOOTING.md" with an entry's name, DETAILS.md sends readers there for environment variables, and other
pages link to its headings. An entry that disappears turns those pointers into dead ends (the "Windows AMD: the driver resets" entry did, and setup's own
test failed on it).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "TROUBLESHOOTING.md"

# One distinctive phrase from the title of each entry the v0.2.4.2 page had (the page may reword the rest).
ENTRIES = [
    "couldn't find an engine for this computer", "needs a newer engine", "damaged and can't be opened", "strata doctor", "Sending logs for a bug report",
    "froze, or got very slow", "app keeps more graphics memory free", "A download stopped", "driver is too old", "Port 8080 is already in use",
    "Smart App Control", "new engine misbehaves", "git pull did not succeed", "build tools could not be installed", "does not compile",
    "Startup takes a long time", "does not match GitHub's checksum", "It is slow and the disk light", "engine stopped unexpectedly", "exit code -9",
    "exceeds the context", "slower than the tables", "came back empty", "Turing card", "--expert-cache N", "--resident-budget-gib N",
    "page-locked expert copy", "0x141", "RTX 50 card", "Pictures are refused", "KFD topology is empty", "build's list", "0xC0000005",
    "the driver resets", "Large pinned host allocations", "verify: timed out at layer N",
]


def slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    return {slug(h) for h in re.findall(r"^#+\s+(.*)$", path.read_text(encoding="utf-8"), re.M)}


class TroubleshootingPage(unittest.TestCase):
    def setUp(self):
        self.text = PAGE.read_text(encoding="utf-8")
        self.flat = re.sub(r"\s+", " ", self.text.replace("*", ""))

    def test_every_entry_of_v0_2_4_2_is_on_the_page(self):
        for phrase in ENTRIES:
            self.assertIn(re.sub(r"\s+", " ", phrase), self.flat, f"the entry about {phrase!r} is missing")

    def test_what_setup_and_the_server_point_to_is_there(self):
        setup = (ROOT / "setup.py").read_text(encoding="utf-8")
        for name in re.findall(r'docs/TROUBLESHOOTING\.md, \\"([^"\\]+)\\"', setup):
            self.assertIn(name, self.text, f"setup.py points to the entry {name!r}")
        server = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertIn("docs/TROUBLESHOOTING.md", server)
        self.assertIn("Smart App Control", self.text)        # that pointer is the blocked-executable message
        self.assertIn("--vision no", self.text)              # which names the way out for a source checkout

    def test_the_variables_that_other_pages_send_here_are_explained(self):
        details = (ROOT / "docs" / "DETAILS.md").read_text(encoding="utf-8")
        for line in details.splitlines():
            if "TROUBLESHOOTING.md" in line:
                for name in re.findall(r"STRATA_[A-Z0-9_]+", line):
                    self.assertIn(name, self.text, f"DETAILS.md sends readers here for {name}")

    def test_links_into_the_page_and_out_of_it_resolve(self):
        own = anchors(PAGE)
        for md in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md", ROOT / "SECURITY.md"]:
            for target in re.findall(r"\(((?:\.\./)?(?:docs/)?TROUBLESHOOTING\.md#[^)\s]+)\)", md.read_text(encoding="utf-8")):
                self.assertIn(target.split("#", 1)[1], own, f"{md.name} links to {target}")
        for match in re.finditer(r"\]\(([^)#\s]+)?(#[^)\s]*)?\)", self.text):
            name, anchor = match.group(1), match.group(2)
            if name and name.startswith("http"):
                continue
            target = (PAGE.parent / name).resolve() if name else PAGE
            self.assertTrue(target.exists(), f"the page links to {name}, which is not there")
            if anchor and target.suffix == ".md":
                self.assertIn(anchor[1:], anchors(target), f"the page links to {name or ''}{anchor}")


if __name__ == "__main__":
    unittest.main()
