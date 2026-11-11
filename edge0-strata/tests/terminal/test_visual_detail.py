"""Wide monitor pages preserve readable telemetry and redacted request data."""

import io
import time
import unittest

try:
    from rich.console import Console
    from strata_cli.dashboard import MonitorHistory
    from strata_cli.detail_dashboard import render_detail_dashboard
except ImportError:
    Console = None


def snapshot():
    now = time.time()
    slots = [{"slot": i, "group": i // 2, "state": "decoding" if i < 3 else "idle",
              "request_id": f"req-{i}" if i < 3 else None,
              "held_tokens": 1500 + 100 * i if i < 3 else 0,
              "tok_s": 12.0 + i if i < 3 else None} for i in range(4)]
    return {
        "sampled_at": now, "source": "test", "instance": {"id": "test", "state": "ready"},
        "model": {"id": "test-model", "loaded": True, "context": 8192},
        "work": {"active": 3, "occupied_slots": 3, "slots_effective": 4,
                 "slots_requested": 4, "groups": 2, "queued": 1, "waiting": 0,
                 "max_queue": 8, "admission_limit": 12, "admission_reserved": 4,
                 "batch_sample_age_s": 0.2, "effective_rows_window": 3.0,
                 "batch_active_rows_window": 3.0, "slots": slots,
                 "requests": [{"id": f"req-{i}", "state": "decoding", "slot": i,
                               "group": i // 2, "prompt_tokens": 400,
                               "generated": 60 + i, "queued_at": now - 2,
                               "admitted_at": now - 1.5,
                               "first_engine_token_s": 0.8,
                               "prompt_body": "SECRET_PROMPT"} for i in range(3)] +
                             [{"id": "wait-0", "state": "queued", "slot": None,
                               "prompt_tokens": 500, "queued_at": now - 1}]},
        "rates": {"aggregate_output_tok_s": 41.0, "answer_tok_s": 13.5,
                  "prefill_tok_s": 120.0},
        "topology": {"mode": "layer split", "resolved": True,
                     "selected_devices": [0, 1],
                     "stages": [{"stage": i, "device_index": i,
                                 "layer_start": 10 * i, "layer_end": 10 * i + 9}
                                for i in range(2)]},
        "devices": [{"index": i, "name": f"GPU-{i}", "util": 70 + i,
                     "mem_used": (10 + i) * 2**30, "mem_total": 24 * 2**30,
                     "temp": 60 + i, "power": 180 + i} for i in range(2)],
        "ram": {"used": 20 * 2**30, "total": 64 * 2**30},
        "recent": [{"id": "done-0", "finish": "stop", "prompt_tokens": 100,
                    "reused": 20, "output_tokens": 40, "duration_s": 3.2,
                    "first_engine_token_s": 0.7}],
        "activity": {"available": True, "last_seq": 2,
                     "events": [{"seq": 1, "at": now - 2, "request_id": "req-0",
                                 "event": "queued"},
                                {"seq": 2, "at": now - 1, "request_id": "req-0",
                                 "event": "admitted"}]},
    }


@unittest.skipIf(Console is None, "optional Rich dependency absent")
class VisualDetailTests(unittest.TestCase):
    def render(self, sample, view, width=140, height=44, **options):
        output = io.StringIO()
        history = MonitorHistory()
        history.add(sample, ("id", "test"))
        Console(file=output, width=width - 2, color_system=None).print(
            render_detail_dashboard(sample, view, width=width - 2,
                                    height=height - 2, history=history,
                                    color=False, **options))
        return output.getvalue()

    def test_all_five_pages_fit_wide_terminals_and_keep_request_text_private(self):
        sample = snapshot()
        for width, height in ((140, 44), (128, 36)):
            for view in ("serving", "requests", "engine", "hardware", "logs"):
                with self.subTest(width=width, view=view):
                    shown = self.render(sample, view, width, height)
                    self.assertIn(f"STRATA  /  {view.upper()}", shown)
                    self.assertNotIn("SECRET_PROMPT", shown)
                    self.assertLessEqual(len(shown.splitlines()), height - 2)
                    self.assertTrue(all(len(line) <= width - 2 for line in shown.splitlines()))
        self.assertIn("ROWS/WINDOW", self.render(sample, "serving"))
        self.assertIn("GPU 0", self.render(sample, "hardware"))
        self.assertIn("HOST SPAN", self.render(sample, "engine"))
        self.assertIn("req-0", self.render(sample, "logs"))

    def test_selection_detail_and_stale_readings(self):
        sample = snapshot()
        summary = self.render(sample, "requests", selected=0)
        expanded = self.render(sample, "requests", selected=0, detail=True)
        self.assertIn("Enter expands", summary)
        self.assertIn("First engine token", expanded)
        self.assertNotIn("First engine token", summary)
        stale = self.render(sample, "serving", connection_state="stale")
        self.assertIn("STALE", stale)
        self.assertNotIn("41.0 tok/s", stale)
        stale_hardware = self.render(sample, "hardware", connection_state="stale")
        self.assertNotIn("70%", stale_hardware)
        self.assertIn("--/--G", stale_hardware)

    def test_replica_batch_windows_remain_separate(self):
        sample = snapshot()
        sample["topology"]["mode"] = "replicas (configured)"
        sample["work"]["effective_rows_window"] = None
        sample["work"]["replica_batches"] = [
            {"replica": 0, "effective_rows_window": 2.0,
             "batch_active_rows_window": 1.5, "batch_sample_age_s": 0.2},
            {"replica": 1, "effective_rows_window": 3.0,
             "batch_active_rows_window": 2.0, "batch_sample_age_s": 0.3},
        ]
        shown = self.render(sample, "serving")
        self.assertIn("R0 ROWS/WIN", shown)
        self.assertIn("R1 ROWS/WIN", shown)
        self.assertIn("Replica sample windows are independent", shown)


if __name__ == "__main__":
    unittest.main()
