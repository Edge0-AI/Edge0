"""A `cmake -P` script has no cmake_minimum_required, so CMake 3.x applies the OLD behaviour of CMP0057 and `IN_LIST` is not an operator
(ctest ggml_tq_overlay stopped at its first IN_LIST with CMake 3.31, #3412). Every script in cmake/ that is run with -P and uses IN_LIST
sets the policy before the first use. (Static check: the project accepts CMake 3.24 and newer, and CMake 4 sets the policy by itself.)"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

CMAKE = Path(__file__).resolve().parents[2] / "cmake"


class ScriptPolicies(unittest.TestCase):
    def test_p_scripts_that_use_in_list_set_cmp0057_first(self):
        found = 0
        for path in sorted(CMAKE.glob("*_test.cmake")):
            text = path.read_text(encoding="utf-8")
            code = re.sub(r"(?m)^\s*#.*$", "", text)
            if "IN_LIST" not in code:
                continue
            found += 1
            policy = re.search(r"cmake_policy\(\s*SET\s+CMP0057\s+NEW\s*\)", code) or re.search(r"cmake_minimum_required\(\s*VERSION\s+3\.(1[0-9]|[2-9]\d)", code)
            self.assertTrue(policy, f"{path.name} uses IN_LIST without CMP0057 NEW")
            self.assertLess(policy.start(), code.index("IN_LIST"), f"{path.name} sets the policy after its first IN_LIST")
        self.assertGreaterEqual(found, 1)


if __name__ == "__main__":
    unittest.main()
