"""Check ordered JSON against Python, including binary64 edge cases."""
import json
import math
import random
import struct
import subprocess
import sys

samples = [
    '{"z": 1, "a": [null, true, false, "\\u4e2d\\u6587", "\\n\\t\\b\\f\\r/\\\\\\""]}',
    '[-0, 0, 123456789012345678901234567890, 1.0, 0.5, 1e-05, 1e16, 1e15, -0.0]',
    '["\\ud83d\\ude00", "\\u0000\\u001f"]',
    "1e-400", "-1e-400", "5e-324", "1.7976931348623157e308",
]
rng = random.Random(17)
for _ in range(1000):
    number = struct.unpack("d", rng.randbytes(8))[0]
    if math.isfinite(number):
        samples.append(repr(number))
invalid = [
    '{"a": 1, "a": 2}', '{"a": 1, "\\u0061": 2}', "[1,]", "01", "+1",
    "1.", "1e", "NaN", "Infinity", "1e309", '"\\ud800"', '"\\udc00"',
    '{"x":}', "true false", '"\x01"',
]
result = subprocess.run([sys.argv[1]], input="\n".join(samples + invalid) + "\n",
                        text=True, encoding="utf-8", capture_output=True, check=True)
expected = [json.dumps(json.loads(s), ensure_ascii=False) for s in samples]
expected += ["INVALID"] * len(invalid)
actual = result.stdout.splitlines()
assert len(actual) == len(expected), (len(actual), len(expected))
for source, got, want in zip(samples + invalid, actual, expected):
    assert got == want, (source, got, want)
print(f"{len(expected)} ordered JSON comparisons and rejection cases passed")
