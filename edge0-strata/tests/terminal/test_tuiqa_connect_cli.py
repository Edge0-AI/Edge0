"""Tester report T07: `strata connect pi --check --output json` skipped the checks and said "ok": true with exit 0 even when nothing was
listening. It now runs the same two checks as the plain-words output, returns each result, and exits 1 when one failed or the server cannot be
reached."""

from __future__ import annotations

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.terminal.launch_support import Home, run_cli


class Handler(BaseHTTPRequestHandler):
    mode = "good"                                           # good | no-text | no-tool

    def log_message(self, *a) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        wants_tool = bool(body.get("tools"))
        message = {"role": "assistant", "content": "pong"}
        if Handler.mode == "no-text" and not wants_tool:
            message = {"role": "assistant", "content": None}
        if wants_tool:
            message = ({"role": "assistant", "content": "I cannot."} if Handler.mode == "no-tool" else
                       {"role": "assistant", "content": None, "tool_calls": [
                           {"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]})
        data = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ConnectCheckJson(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        Handler.mode = "good"

    def free_port(self) -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def run_json(self, port: int, *extra: str):
        code, out, err = run_cli(["connect", "pi", "--url", f"http://127.0.0.1:{port}", "--context", "128k", "--check", "--output", "json", *extra])
        return code, json.loads(out), err

    def run_human(self, port: int):
        return run_cli(["connect", "pi", "--url", f"http://127.0.0.1:{port}", "--context", "128k", "--check"])

    def test_a_working_server_passes_both_checks_with_ok_true_and_exit_zero(self):
        code, data, _ = self.run_json(self.port)
        self.assertEqual(code, 0)
        self.assertIs(data["ok"], True)
        self.assertEqual([c["name"] for c in data["checks"]], ["prompt", "tool call"])
        self.assertTrue(all(c["ok"] for c in data["checks"]))
        self.assertIn("get_weather", data["checks"][1]["detail"])
        self.assertEqual(data["checked_server"], f"http://127.0.0.1:{self.port}")
        self.assertEqual(data["vendor_tested"], "pi 0.74.2")
        self.assertIn("files", data)                          # the rendered setup is still there

    def test_a_server_that_answers_wrongly_is_false_with_exit_one_and_says_which_check(self):
        for mode, failing in (("no-text", "prompt"), ("no-tool", "tool call")):
            Handler.mode = mode
            code, data, _ = self.run_json(self.port)
            self.assertEqual(code, 1, mode)
            self.assertIs(data["ok"], False)
            bad = [c["name"] for c in data["checks"] if not c["ok"]]
            self.assertEqual(bad, [failing], mode)
            self.assertIn(failing, data["error"]["message"])
            self.assertIn("answered", data["error"]["message"])

    def test_nothing_listening_is_false_with_exit_one_and_says_it_could_not_be_reached(self):
        port = self.free_port()
        code, data, _ = self.run_json(port)
        self.assertEqual(code, 1)
        self.assertIs(data["ok"], False)
        self.assertEqual([c["ok"] for c in data["checks"]], [False, False])
        self.assertIn("could not be reached", data["error"]["message"])

    def test_json_and_human_agree_on_the_exit_code(self):
        port = self.free_port()
        self.assertEqual(self.run_json(port)[0], self.run_human(port)[0])
        Handler.mode = "no-tool"
        self.assertEqual(self.run_json(self.port)[0], self.run_human(self.port)[0])
        Handler.mode = "good"
        self.assertEqual(self.run_json(self.port)[0], self.run_human(self.port)[0], 0)

    def test_without_check_json_is_unchanged_and_asks_nothing_of_the_server(self):
        code, out, _ = run_cli(["connect", "pi", "--url", f"http://127.0.0.1:{self.free_port()}", "--output", "json"])
        data = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(data["ok"])
        self.assertNotIn("checks", data)

    def test_the_list_says_vendor_tested_and_not_a_check_of_your_server(self):
        code, out, _ = run_cli(["connect"])
        self.assertIn("Vendor tested", out)
        self.assertIn("not a check of your server", out)
        code, out, _ = run_cli(["connect", "--output", "json"])
        rows = json.loads(out)["harnesses"]
        self.assertEqual(next(r for r in rows if r["id"] == "pi")["vendor_tested"], "pi 0.74.2")


if __name__ == "__main__":
    unittest.main()
