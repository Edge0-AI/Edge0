"""Run the asset tool tests of one marker for CTest; any skipped test makes the run "not executed" (77)."""
from pathlib import Path
import sys

import pytest


class Skips:
    count = 0

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.count += 1


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('l1', 'l2'):
        print('usage: run_marked.py l1|l2')
        return 2
    skips = Skips()
    code = pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', '-m', sys.argv[1], str(Path(__file__).resolve().parent)],
                       plugins=[skips])
    if code == 0 and skips.count:
        print(f'not executed: {skips.count} skipped test(s)')
        return 77
    return int(code)


if __name__ == '__main__':
    raise SystemExit(main())
