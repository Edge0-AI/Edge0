"""Compare production JNI serialization of native sample values to the shared sample."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[6]
with tempfile.TemporaryDirectory(prefix="edge0-android-status-") as temporary:
    binary = Path(temporary) / "status-mapping"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I" + str(root / "include"), "-I" + str(root / "examples/android"), str(Path(__file__).with_name("status_mapping_test.cpp")), "-o", str(binary)], check=True)
    actual = json.loads(subprocess.check_output([str(binary)], text=True))
expected = json.loads((root / "examples/shared/status-sample.json").read_text())
progress = expected["current"]
for output, source in {"phase": "phase", "inputTokens": "input_tokens", "reusedTokens": "reused_tokens", "computedTokens": "computed_tokens", "pendingTokens": "pending_tokens", "generatedTokens": "generated_tokens", "elapsedSeconds": "elapsed_seconds", "firstTokenSeconds": "first_token_seconds", "decodeElapsedSeconds": "decode_elapsed_seconds"}.items():
    assert actual["progress"][output] == progress[source], output
memory = expected["memory"]
ledger = memory["ledger"]
for output, source in {"limitBytes": "limit_bytes", "currentBytes": "current_bytes", "peakBytes": "peak_bytes", "byPurpose": "by_purpose"}.items():
    assert actual["memory"][output] == ledger[source], output
assert actual["memory"]["byDevice"] == {key: {"currentBytes": value["current_bytes"], "peakBytes": value["peak_bytes"], "memory": value["memory"]} for key, value in ledger["by_device"].items()}
assert actual["memory"]["physicalBytes"] == memory["physical"]["values"]["current_bytes"]
assert actual["memory"]["residentBytes"] == memory["resident"]["values"]["current_bytes"]
assert actual["system"] == {"totalBytes": expected["system"]["memory"]["total"]["bytes"], "availableBytes": expected["system"]["memory"]["available"]["bytes"], "thermal": expected["system"]["thermal"]["state"]}
print("PASS native status mapping: production serialization matches shared sample; dedicated and missing fields checked")
