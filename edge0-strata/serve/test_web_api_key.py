"""The web app uses an edited API key immediately, even when site storage is blocked.

Runs the page's storage and server-access code with Node, without a GPU or server:
    python -m unittest serve.test_web_api_key -v
"""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

APP = Path(__file__).with_name("web") / "app.js"


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebApiKey(unittest.TestCase):
    def run_js(self, actions, *, stored=None, blocked=False, quota=False):
        src = APP.read_text(encoding="utf-8")
        store = src[src.index("const store ="):src.index("// ------------------------------------------------------------------ toasts")]
        access = src[src.index("// ------------------------------------------------------------------ server access"):src.index("let health =")]
        js = f"""
const data = new Map(Object.entries({json.dumps(stored or {})}));
const field = {{value: ""}};
const notices = [];
const $ = () => field;
const toast = (...args) => notices.push(args);
const localStorage = {{
  getItem(k) {{
    if ({json.dumps(blocked)}) throw new Error("Storage blocked");
    return data.has(k) ? data.get(k) : null;
  }},
  setItem(k, v) {{
    if ({json.dumps(blocked or quota)}) throw new Error("Storage unavailable");
    data.set(k, v);
  }},
}};
function edit(value, event = "input") {{
  field.value = value;
  const handler = field["on" + event];
  if (handler) handler();
}}
{store}
{access}
{actions}
"""
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_input_is_used_and_saved_without_blur(self):
        got = self.run_js('''
edit("  test-key  ");
console.log(JSON.stringify({headers: headers(true), saved: data.get("strata.apikey")}));
''')
        self.assertEqual(got, {"headers": {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
                               "saved": '"test-key"'})

    def test_saved_key_is_restored_on_reload(self):
        got = self.run_js('console.log(JSON.stringify({value: field.value, headers: headers()}));',
                          stored={"strata.apikey": '"saved-key"'})
        self.assertEqual(got, {"value": "saved-key", "headers": {"Authorization": "Bearer saved-key"}})

    def test_blocked_storage_still_uses_key_in_memory(self):
        got = self.run_js('edit("test-key", "change"); console.log(JSON.stringify(headers()));', blocked=True)
        self.assertEqual(got, {"Authorization": "Bearer test-key"})

    def test_failed_write_does_not_reuse_old_key(self):
        got = self.run_js('edit("new-key", "change"); console.log(JSON.stringify(headers()));',
                          stored={"strata.apikey": '"old-key"'}, quota=True)
        self.assertEqual(got, {"Authorization": "Bearer new-key"})

    def test_clearing_key_removes_authorization(self):
        for quota in (False, True):
            with self.subTest(quota=quota):
                got = self.run_js('edit("  "); console.log(JSON.stringify(headers()));',
                                  stored={"strata.apikey": '"old-key"'}, quota=quota)
                self.assertEqual(got, {})

    def test_successful_save_notifies_only_on_change(self):
        got = self.run_js('''
edit("test-key");
const before = notices.length;
edit("test-key", "change");
console.log(JSON.stringify({before, kinds: notices.map(n => n[0])}));
''')
        self.assertEqual(got, {"before": 0, "kinds": ["success"]})

    def test_failed_save_warns_instead_of_claiming_success(self):
        got = self.run_js('edit("test-key", "change"); console.log(JSON.stringify(notices.map(n => n[0])));',
                          blocked=True)
        self.assertEqual(got, ["warn"])

    def test_server_without_key_keeps_json_header_only(self):
        got = self.run_js('console.log(JSON.stringify(headers(true)));')
        self.assertEqual(got, {"Content-Type": "application/json"})


if __name__ == "__main__":
    unittest.main()
