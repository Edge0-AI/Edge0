"""Terminal contract checks that run without model weights or a GPU."""

import asyncio
import copy
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from strata_cli.cli import main
from strata_cli.client import (ClientError, Endpoint, normalize_legacy, normalize_status_only,
                               sanitize_telemetry, server_url)
from strata_cli.render import display_width, field, render_view, safe
from serve.server import (configure_batch_telemetry_env, parse_batch_window_sample,
                          parse_stage_in_flight_sample, parse_engine_stage_line,
                          read_engine_stage_log, visible_device_order)


def four_gpu_snapshot():
    metrics = {
        "time": 1_790_000_000.0,
        "engine": {"model": "test-model", "batch_groups": 4},
        "live": {"parallel": 8, "running": 6, "queued": 5, "waiting": 2,
                 "slots": [{"slot": i, "group": i // 2, "state": "decoding" if i < 6 else "idle",
                            "prompt_tokens": 400 + i, "generated": i * 10} for i in range(8)]},
        "hardware_static": {"gpu_count": 4, "gpu_name": "aggregate"},
        "hardware": {"gpu_mem_used": 1000,
                     "gpus": [{"index": i, "name": f"GPU-{i}", "mem_used": 12 * 2**30,
                               "mem_total": 24 * 2**30, "util": 50 + i, "temp": 60 + i,
                               "power": 200 + i} for i in range(4)]},
        "requests": [{"finish": "stop", "prompt_tokens": 100, "output_tokens": 20}],
        "totals": {"requests": 1, "output_tokens": 20},
    }
    status = {"service": "strata", "model": "test-model", "loaded": True,
              "concurrency": {"serving": 8}, "started": 1_789_999_000}
    return normalize_legacy(metrics, status)


class ContractTests(unittest.TestCase):
    def test_replica_slots_and_requests_keep_identity_without_aggregate_batch_claim(self):
        sample = four_gpu_snapshot()
        sample["topology"]["mode"] = "replicas (configured)"
        sample["topology"]["replica_devices"] = [{"replica": 0, "devices": [0, 1]},
                                                  {"replica": 1, "devices": [2, 3]}]
        sample["sampled_at"] = time.time()
        sample["work"]["groups"] = 2
        sample["work"]["effective_rows_window"] = None
        for slot in sample["work"]["slots"]:
            slot["replica"] = slot["slot"] // 4
            slot["group"] = slot["replica"]
        sample["work"]["requests"] = [{"id": "abcdef012345", "state": "reading", "slot": 5,
                                        "group": 1, "replica": 1, "prompt_tokens": 100, "prompt_read": 12}]
        clean = sanitize_telemetry(sample)
        self.assertEqual(clean["work"]["slots"][5]["replica"], 1)
        self.assertEqual(clean["work"]["requests"][0]["replica"], 1)
        self.assertEqual(clean["topology"]["replica_devices"][1]["devices"], [2, 3])
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            clean, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("replicas 2", output.getvalue())
        self.assertIn("R1 G1", output.getvalue())
        self.assertRegex(output.getvalue(), r"GPU 2 R1")
        sample["work"]["replica_batches"] = [
            {"replica": 0, "batch_sample_windows": 2, "batch_sample_rows": 8,
             "batch_sample_active_rows": 4, "batch_sample_span_s": 1.0, "batch_sample_age_s": 0.5},
            {"replica": 1, "batch_sample_windows": 2, "batch_sample_rows": 6,
             "batch_sample_active_rows": 4, "batch_sample_span_s": 1.0, "batch_sample_age_s": 0.7}]
        sample["work"].update(batch_sample_windows=2, batch_sample_rows=8,
                              batch_sample_active_rows=4, batch_sample_span_s=1.0,
                              batch_sample_age_s=0.5)
        clean = sanitize_telemetry(sample)
        self.assertIsNone(clean["work"]["effective_rows_window"])
        self.assertEqual([row.get("effective_rows_window") for row in clean["work"]["replica_batches"]],
                         [4.0, 3.0])
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            clean, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("R0 4.0/2.0", output.getvalue())
        self.assertIn("R1 3.0/2.0", output.getvalue())
        self.assertIn("R1 3.0/2.0", render_view(clean, "serving", 110, 30))
        self.assertIn("Replica 1", render_view(clean, "serving", 80, 24,
                                               serving_pane="groups", selected=5, detail=True))
        self.assertIn("Replica 1", render_view(clean, "requests", 80, 24, selected=0, detail=True))
        self.assertIn("Replica R1 [configured cards 2,3", render_view(
            clean, "hardware", 80, 24, selected=2, detail=True))
        sample["topology"]["replica_stages"] = [
            {"replica": replica, "stages": [
                {"stage": stage, "layer_start": stage * 20, "layer_end": stage * 20 + 19,
                 "engine_ordinal": stage, "device_index": replica * 2 + stage}
                for stage in range(2)]} for replica in range(2)]
        sample["work"]["replica_stage_samples"] = [
            {"replica": 0, "stage_sample_elapsed_us": 1_000_000,
             "stage_in_flight_us": [800_000, 600_000], "stage_sample_age_s": 0.5},
            {"replica": 1, "stage_sample_elapsed_us": 1_000_000,
             "stage_in_flight_us": [700_000, 500_000], "stage_sample_age_s": 0.7}]
        clean = sanitize_telemetry(sample)
        self.assertTrue(clean["topology"]["replica_stages"][1]["resolved"])
        self.assertEqual(clean["quality"]["stage_in_flight"], "unavailable")
        self.assertEqual(clean["quality"]["replica_stage_in_flight"], "measured_host_span")
        self.assertEqual([row["stage_in_flight"][0]["percent"] for row in
                          clean["work"]["replica_stage_samples"]], [80.0, 70.0])
        self.assertIn("R1 Stage 1  Device 3  Layers 20-39  In-flight 50%",
                      render_view(clean, "engine", 80, 24))
        self.assertIn("Replica host span R0 S0 80% S1 60%  R1 S0 70% S1 50%", render_view(
            clean, "serving", 110, 30))
        self.assertIn("Replica R1 stage/layers 1:20-39 [engine log]", render_view(
            clean, "hardware", 80, 24, selected=3, detail=True))
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            clean, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("Host span R1 S0 70%", output.getvalue())
        self.assertIn("GPU 2 R1/S0", output.getvalue())
        old_stage = copy.deepcopy(clean)
        old_stage["work"]["replica_stage_samples"][1]["stage_sample_age_s"] = 5.1
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            old_stage, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("Host span R1 --", output.getvalue())
        self.assertNotIn("Host span R1 S0 70%", output.getvalue())
        compact_output = io.StringIO()
        Console(file=compact_output, width=80, color_system=None).print(render_dashboard(
            clean, width=78, height=24, history=MonitorHistory(), color=False))
        self.assertIn("GPU 3 R1/S1", compact_output.getvalue())
        shared_card = copy.deepcopy(clean)
        shared_card["topology"]["replica_stages"][1]["stages"][0]["device_index"] = 3
        shared_output = io.StringIO()
        Console(file=shared_output, width=80, color_system=None).print(render_dashboard(
            shared_card, width=78, height=24, history=MonitorHistory(), color=False))
        self.assertIn("GPU 3 R1 ", shared_output.getvalue())
        self.assertNotIn("GPU 3 R1/S", shared_output.getvalue())
        partial = copy.deepcopy(sample)
        partial["work"]["slots"] = partial["work"]["slots"][:2]
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sanitize_telemetry(partial), width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("replicas 2", output.getvalue())
        self.assertIn("R0 4.0/2.0", output.getvalue())
        self.assertIn("Group occupancy --", output.getvalue())
        sample["work"]["slots"][5]["replica"] = 1.5
        sample["work"]["requests"][0]["replica"] = True
        sample["work"]["replica_batches"][1]["batch_sample_rows"] = 99
        sample["work"]["replica_stage_samples"][1]["stage_in_flight_us"] = [700_000]
        sample["topology"]["replica_stages"][1]["stages"][1]["device_index"] = 9
        sample["topology"]["replica_devices"].append({"replica": 2, "devices": [3]})
        rejected = sanitize_telemetry(sample)
        self.assertIsNone(rejected["work"]["slots"][5]["replica"])
        self.assertIsNone(rejected["work"]["requests"][0]["replica"])
        self.assertIsNone(rejected["work"]["replica_batches"][1].get("effective_rows_window"))
        self.assertNotIn("stage_in_flight", rejected["work"]["replica_stage_samples"][1])
        self.assertEqual(rejected["work"]["replica_stage_samples"][0]["stage_in_flight"][0]["percent"], 80.0)
        self.assertEqual(rejected["quality"]["replica_stage_in_flight"], "measured_host_span")
        self.assertEqual(len(rejected["topology"]["replica_devices"]), 2)
        self.assertIsNone(rejected["topology"]["replica_stages"][1]["stages"][1]["device_index"])
        self.assertFalse(rejected["topology"]["replica_stages"][1]["resolved"])
        output = io.StringIO()
        Console(file=output, width=80, color_system=None).print(render_dashboard(
            rejected, width=78, height=24, history=MonitorHistory(), color=False))
        self.assertIn("GPU 3 R1 ", output.getvalue())
        self.assertNotIn("GPU 3 R1/S1", output.getvalue())

    def test_display_cells_preserve_wide_and_combining_clusters(self):
        self.assertEqual(safe("漢字", 3), "漢")
        self.assertEqual(display_width(safe("漢字", 3)), 2)
        self.assertEqual(safe("x\u0338Y", 1), "x\u0338")
        self.assertEqual(display_width(safe("x\u0338Y", 1)), 1)
        self.assertEqual(safe("🇭🇷next", 1), "")
        self.assertEqual(safe("🇭🇷next", 2), "🇭🇷")
        self.assertEqual(safe("👩‍💻next", 2), "👩‍💻")
        self.assertEqual(safe("\u202e漢", 2), "漢")
        self.assertEqual(display_width(field("漢A", 4)), 4)
        self.assertEqual(field("漢A", 4), "漢A ")
        snapshot = four_gpu_snapshot()
        snapshot["devices"][0]["name"] = "漢字GPUx\u0338"
        snapshot["work"]["requests"] = [{"id": "漢字reqx\u0338", "state": "decoding",
                                         "slot": 0, "group": 0, "generated": 3}]
        for width, height in ((110, 30), (80, 24), (60, 16), (40, 12)):
            with self.subTest(width=width):
                for name in ("hardware", "serving", "requests"):
                    view = render_view(snapshot, name, width, height)
                    self.assertTrue(all(display_width(line) <= width for line in view.splitlines()), name)

    def test_batch_telemetry_opt_in_and_explicit_disable_are_reversible(self):
        env = {"STRATA_BATCH_TELEMETRY": "1"}
        configure_batch_telemetry_env(env, {}, None)
        self.assertEqual(env["STRATA_BATCH_TELEMETRY"], "1")
        configure_batch_telemetry_env(env, {"batch_telemetry": False}, None)
        self.assertNotIn("STRATA_BATCH_TELEMETRY", env)
        configure_batch_telemetry_env(env, {"batch_telemetry": False}, True)
        self.assertEqual(env["STRATA_BATCH_TELEMETRY"], "1")
        configure_batch_telemetry_env(env, {"batch_telemetry": True}, False)
        self.assertNotIn("STRATA_BATCH_TELEMETRY", env)

    def test_batch_window_sample_distinguishes_processed_rows_from_active_slots(self):
        self.assertEqual(parse_batch_window_sample("BSTAT 3 18 9 1000\n"),
                         {"windows": 3, "processed_rows": 18, "active_rows": 9, "elapsed_ms": 1000})
        for invalid in ("BSTAT 0 2 1 100", "BSTAT 1 9 1 100", "BSTAT 1 4 5 100", "BSTAT 2 2 0 100",
                        "BSTAT 1 1 1 0", "BSTAT 1 1 1 extra", "BSTAT " + "9" * 100):
            self.assertIsNone(parse_batch_window_sample(invalid))
        sample = four_gpu_snapshot()
        sample["work"].update(batch_sample_windows=3, batch_sample_rows=18,
                              batch_sample_active_rows=9, batch_sample_span_s=1.0,
                              batch_sample_age_s=1.5,
                              effective_rows_window=999)
        sample["quality"]["batch_window_rows"] = "measured"
        clean = sanitize_telemetry(sample)
        self.assertEqual(clean["work"]["effective_rows_window"], 6.0)
        self.assertEqual(clean["work"]["batch_active_rows_window"], 3.0)
        self.assertEqual(clean["quality"]["batch_window_rows"], "measured")
        self.assertEqual(clean["quality"]["batch_window_source"], "engine_bstat")
        serving = render_view(clean, "serving", 110, 30)
        self.assertIn("6.0 processed / 3.0 active", serving)
        sample["work"]["batch_sample_rows"] = 25
        rejected = sanitize_telemetry(sample)
        self.assertIsNone(rejected["work"]["effective_rows_window"])
        self.assertEqual(rejected["quality"]["batch_window_rows"], "unavailable")

    def test_pipeline_stage_in_flight_sample_is_bounded_and_labelled_host_time(self):
        self.assertEqual(parse_stage_in_flight_sample("PSTAT 4 20000 16000 14000 12000 10000\n"),
                         {"elapsed_us": 20000, "in_flight_us": [16000, 14000, 12000, 10000]})
        for invalid in ("PSTAT 1 20000 10000", "PSTAT 4 20000 16000 14000 12000",
                        "PSTAT 4 20000 16000 14000 12000 21000", "PSTAT 2 0 0 0",
                        "PSTAT 2 100 1 -1", "PSTAT 2 100 1 1 extra", "PSTAT " + "9" * 300):
            self.assertIsNone(parse_stage_in_flight_sample(invalid))
        sample = four_gpu_snapshot()
        sample["work"].update(stage_sample_elapsed_us=1_000_000,
                              stage_in_flight_us=[800000, 700000, 600000, 500000],
                              stage_sample_age_s=0.2)
        clean = sanitize_telemetry(sample)
        self.assertEqual([row["percent"] for row in clean["work"]["stage_in_flight"]],
                         [80.0, 70.0, 60.0, 50.0])
        self.assertEqual(clean["quality"]["stage_in_flight_source"], "engine_pstat_host_span")
        self.assertIn("S3 50%", render_view(clean, "serving", 110, 30))
        self.assertIn("80/70/60/50%", render_view(clean, "engine", 40, 12))
        self.assertIn("GPU kernel occupancy unavailable", render_view(clean, "engine", 80, 24))
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard
        old_stage = copy.deepcopy(clean)
        old_stage["work"]["stage_sample_age_s"] = 5.1
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            old_stage, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("Stage host span --", output.getvalue())
        detail = render_view(clean, "serving", 80, 24, serving_pane="groups", selected=7, detail=True)
        self.assertIn("group 3", detail)
        self.assertIn("q quit", detail)
        sample["topology"] = {"mode": "layer split", "resolved": True, "source": "engine_start_log",
                              "stages": [{"stage": i, "layer_start": i * 12, "layer_end": i * 12 + 11,
                                          "engine_ordinal": i, "device_index": i} for i in range(4)]}
        mapped = sanitize_telemetry(sample)
        engine = render_view(mapped, "engine", 80, 24)
        self.assertIn("Stage 3  Engine GPU 3  Device 3  Layers 36-47  In-flight 50%", engine)
        self.assertIn("q quit", engine)
        sample["work"]["stage_in_flight_us"] = [800000, 700000, 600000]
        self.assertEqual(sanitize_telemetry(sample)["quality"]["stage_in_flight"], "unavailable")
        sample["work"]["stage_in_flight_us"] = [800000, 700000, 600000, 500000]
        sample["work"]["stage_in_flight_us"][1] = 1_000_001
        rejected = sanitize_telemetry(sample)
        self.assertEqual(rejected["work"]["stage_in_flight"], [])
        self.assertEqual(rejected["quality"]["stage_in_flight"], "unavailable")

    def test_engine_stage_log_is_bounded_and_reports_only_valid_final_ranges(self):
        line = ("strata serve: layer split: layers 0-11 (CUDA0), 12-23 (CUDA1), "
                "24-35 (CUDA2), 36-47 (CUDA3), one hand-off per window")
        self.assertEqual([row["layer_start"] for row in parse_engine_stage_line(line)], [0, 12, 24, 36])
        for bad in (line.replace("12-23", "13-23"), line.replace("CUDA3", "CUDA99"),
                    line.replace("0-11", "0" * 6000 + "-11"),
                    line.replace("one hand-off per window", "unverified")):
            self.assertIsNone(parse_engine_stage_line(bad))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.log"
            path.write_text(line + "\n", encoding="utf-8")
            offset = path.stat().st_size
            self.assertEqual(read_engine_stage_log(str(path), offset), [])
            with path.open("a", encoding="utf-8") as target:
                target.write("INFO engine=0.1.42\n" + line + "\nREADY 4096 stop\n")
            self.assertEqual(len(read_engine_stage_log(str(path), offset)), 4)
            self.assertEqual(read_engine_stage_log(str(path), offset, limit=12), [])
        cuda_env = {"CUDA_VISIBLE_DEVICES": "2,0,3,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.assertEqual(visible_device_order(cuda_env, "cuda"), [2, 0, 3, 1])
        self.assertEqual(visible_device_order({"CUDA_VISIBLE_DEVICES": "2,0,3,1"}, "cuda"), [])
        self.assertEqual(visible_device_order({"HIP_VISIBLE_DEVICES": "1,0"}, "hip"), [1, 0])
        self.assertEqual(visible_device_order({**cuda_env, "CUDA_VISIBLE_DEVICES": "2,2"}, "cuda"), [])
        self.assertEqual(visible_device_order({**cuda_env, "CUDA_VISIBLE_DEVICES": "9" * 6000}, "cuda"), [])
        self.assertEqual(visible_device_order({"CUDA_VISIBLE_DEVICES": "2,0"}, "sycl"), [])

    def test_reported_stage_ranges_render_and_invalid_remote_map_is_rejected(self):
        s = four_gpu_snapshot()
        s["topology"] = {"mode": "layer split (engine reported)", "resolved": True,
                         "source": "engine_start_log", "stages": [
                             {"stage": i, "layer_start": i * 12, "layer_end": i * 12 + 11,
                              "engine_ordinal": i, "device_index": i} for i in range(4)]}
        clean = sanitize_telemetry(s)
        self.assertTrue(clean["topology"]["resolved"])
        engine = render_view(clean, "engine", 80, 24)
        device = render_view(clean, "hardware", 80, 24, selected=3, detail=True)
        wide = render_view(clean, "serving", 110, 30)
        self.assertIn("Stage 3  Engine GPU 3  Device 3  Layers 36-47", engine)
        self.assertIn("Stage/layers 3:36-47 [engine log]", device)
        self.assertIn("GPU 3: 36-47", wide)
        s["topology"]["stages"][1]["layer_start"] = 13
        rejected = sanitize_telemetry(s)
        self.assertEqual(rejected["topology"]["stages"], [])
        self.assertFalse(rejected["topology"]["resolved"])
        self.assertEqual(rejected["quality"]["placement"], "unavailable")

    def test_remote_url_requires_tls_and_no_embedded_credentials(self):
        for url in ("http://example.com:8080", "https://key@example.com", "https://example.com/v1",
                    "https://example.com/#fragment"):
            with self.assertRaises(ValueError, msg=url):
                server_url(url)
        self.assertEqual(server_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")

    def test_parallel_fields_do_not_invent_window_metrics(self):
        s = four_gpu_snapshot()
        self.assertEqual(len(s["devices"]), 4)
        self.assertEqual(s["work"]["groups"], 4)
        self.assertEqual(s["work"]["active"], 6)
        self.assertEqual(s["work"]["occupied_slots"], 6)
        self.assertEqual(s["work"]["queued"], 5)
        self.assertIsNone(s["work"]["effective_rows_window"])
        self.assertIsNone(s["work"]["slots_requested"])
        self.assertIsNone(s["topology"]["mode"])
        serving = render_view(s, "serving", 110, 30)
        self.assertIn("GPU 3", serving)
        self.assertIn("group 3", serving)
        self.assertIn("rows/window --", serving)
        self.assertIn("Engine waiting 2", serving)
        self.assertIn("Slots occupied 6/8", serving)
        self.assertIn("Stage/layer mapping unavailable", serving)
        self.assertNotIn("aggregate", serving)
        s["work"]["requests"] = [{"id": "queued-1", "state": "queued", "queued_at": 1_790_000_000.0,
                                   "slot": None, "group": None, "prompt_tokens": 512}]
        requests = render_view(s, "serving", 80, 24, serving_pane="requests")
        groups = render_view(s, "serving", 80, 24, serving_pane="groups")
        request_view = render_view(s, "requests", 80, 24)
        self.assertIn("queued-1", requests)
        self.assertIn("group 3", groups)
        last_group = render_view(s, "serving", 80, 24, serving_pane="groups", selected=7)
        self.assertIn("slot 7", last_group)
        self.assertIn("group 3", last_group)
        self.assertIn("queued-1", request_view)
        s["work"]["requests"] = [{"id": f"r{i:02d}", "state": "queued" if i > 5 else "decoding",
                                   "queued_at": 1_790_000_000.0} for i in range(11)]
        scrolled = render_view(s, "serving", 80, 24, serving_pane="requests", selected=10)
        self.assertIn("r10", scrolled)
        self.assertIn("queued", scrolled)

    def test_occupied_slot_count_requires_complete_report(self):
        metrics = {"live": {"slots": [{"slot": i, "state": "decoding" if i < 6 else "idle"}
                                         for i in range(8)]}}
        status = {"concurrency": {"serving": 8}}
        complete = normalize_legacy(metrics, status)
        self.assertEqual(complete["work"]["occupied_slots"], 6)
        complete["work"]["occupied_slots"] = 99
        self.assertEqual(sanitize_telemetry(complete)["work"]["occupied_slots"], 6)
        metrics["live"]["slots"] = metrics["live"]["slots"][:-1]
        partial = normalize_legacy(metrics, status)
        self.assertIsNone(partial["work"]["occupied_slots"])
        partial["work"]["occupied_slots"] = 6
        self.assertIsNone(sanitize_telemetry(partial)["work"]["occupied_slots"])
        metrics["live"]["slots"].append({"slot": 0, "state": "decoding"})
        self.assertIsNone(normalize_legacy(metrics, status)["work"]["occupied_slots"])
        metrics["live"]["slots"] = [{"slot": i, "state": "decoding"} for i in range(70)]
        status["concurrency"]["serving"] = 70
        self.assertEqual(len(normalize_legacy(metrics, status)["work"]["slots"]), 64)
        self.assertIsNone(normalize_legacy(metrics, status)["work"]["occupied_slots"])

    def test_admission_pressure_is_preserved_in_monitor_contract(self):
        s = four_gpu_snapshot()
        s["work"].update(admission_reserved=8, admission_limit=8, max_queue=0,
                         admission_rejected=3, oldest_queue_s=0, client_disconnect_cancels=2)
        clean = sanitize_telemetry(s)
        self.assertEqual(clean["work"]["max_queue"], 0)
        display = render_view(clean, "serving", 110, 30)
        self.assertIn("Admission 8/8", display)
        self.assertIn("Queue cap 0", display)
        self.assertIn("Rejected 3", display)
        self.assertIn("Disconnect cancels 2", display)

    def test_wide_parallel_layout_keeps_no_slot_requests_visible(self):
        s = four_gpu_snapshot()
        s["work"]["requests"] = [
            {"id": f"r{i:02d}", "state": "decoding" if i < 8 else "admitted",
             "slot": i if i < 8 else None, "group": i // 2 if i < 8 else None}
            for i in range(12)]
        display = render_view(s, "serving", 110, 30)
        self.assertIn("r11", display)
        self.assertIn("more requests", display)

    def test_parallel_prompt_progress_and_frozen_queue_age(self):
        s = four_gpu_snapshot()
        s["work"]["requests"] = [
            {"id": "reading-1", "state": "reading", "slot": 0, "group": 0,
             "queued_at": 1_789_999_998.0, "admitted_at": 1_789_999_999.0,
             "prompt_tokens": 1000, "prompt_read": 250, "generated": 0},
            {"id": "queued-2", "state": "queued", "slot": None, "group": None,
             "queued_at": 1_789_999_995.0, "admitted_at": None,
             "prompt_tokens": 900, "prompt_read": None, "generated": None},
        ]
        for width, height, pane in ((110, 30, "devices"), (80, 24, "requests"), (40, 12, "requests")):
            shown = render_view(s, "serving", width, height, serving_pane=pane)
            self.assertIn("read 250 new", shown)
            self.assertLessEqual(len(shown.splitlines()), height)
            self.assertTrue(all(len(line) <= width for line in shown.splitlines()))
        with mock.patch("strata_cli.render.time.time", return_value=1_800_000_000.0):
            shown = render_view(s, "serving", 80, 24, serving_pane="requests")
        self.assertIn("wait 5.0 s", shown)
        self.assertIn("wait 1.0 s", shown)
        small = render_view(s, "requests", 40, 12, selected=1)
        self.assertIn("queued-2", small)
        self.assertIn("Pre-admit 5", small)
        detail = render_view(s, "serving", 80, 24, serving_pane="requests", detail=True)
        self.assertIn("REQUEST DETAIL", detail)
        self.assertIn("Prompt 1,000 tokens", detail)
        self.assertIn("First engine token --", detail)
        self.assertNotIn("prompt body", detail.lower())
        fitted = render_view(s, "serving", 78, 22, serving_pane="requests", detail=True)
        self.assertIn("First engine token --", fitted)
        self.assertIn("Sample age", fitted)
        self.assertIn("q quit", fitted)
        filtered = render_view(s, "serving", 110, 30, filter_text="queued-2")
        self.assertIn("queued-2", filtered)
        self.assertNotIn("reading-1", filtered)
        compact_detail = render_view(s, "serving", 60, 16, serving_pane="requests", selected=1, detail=True)
        self.assertIn("queued-2", compact_detail)
        self.assertIn("Wait 5.0 s", compact_detail)
        s["work"]["requests"][0]["first_engine_token_s"] = 1.234
        self.assertIn("First engine token 1.234 s", render_view(s, "serving", 80, 24,
                                                                serving_pane="requests", detail=True))
        s["recent"] = [{"finish": "stop", "first_engine_token_s": 1.234}]
        self.assertIn("First engine token 1.234 s", render_view(s, "overview", 80, 24))
        self.assertIn("First engine token 1.234 s", render_view(s, "serving", 110, 30))

    def test_small_layout_and_untrusted_text(self):
        s = four_gpu_snapshot()
        s["model"]["id"] = "\x1b[2Jfake\u202emodel"
        for width, height in ((110, 30), (80, 24), (60, 16), (40, 12)):
            output = render_view(s, "overview", width, height)
            self.assertNotIn("\x1b", output)
            self.assertNotIn("\u202e", output)
            self.assertLessEqual(len(output.splitlines()), height)
            self.assertTrue(all(len(line) <= width for line in output.splitlines()))
            if (width, height) == (40, 12):
                self.assertEqual(output.count("More: strata status --json"), 1)

    def test_recent_request_detail_scrolls_and_wide_serving_tabs_select_panes(self):
        s = four_gpu_snapshot()
        s["work"]["requests"] = [{"id": "live-1", "state": "decoding", "slot": 0, "group": 0}]
        s["recent"] = [{"id": f"done-{i:02d}", "finish": "stop", "prompt_tokens": 100 + i,
                        "reused": i, "output_tokens": 12, "duration_s": 1.25,
                        "prompt_ms": 200, "decode_ms": 400, "first_engine_token_s": 0.7}
                       for i in range(20)]
        wide_requests = render_view(s, "serving", 110, 30, serving_pane="requests")
        wide_groups = render_view(s, "serving", 110, 30, serving_pane="groups", selected=7)
        self.assertIn("Pane: Requests", wide_requests)
        self.assertIn("live-1", wide_requests)
        self.assertIn("Pane: Groups", wide_groups)
        self.assertIn("slot 7", wide_groups)
        for width, height in ((60, 16), (40, 12)):
            devices = render_view(s, "serving", width, height, serving_pane="devices")
            requests = render_view(s, "serving", width, height, serving_pane="requests")
            groups = render_view(s, "serving", width, height, serving_pane="groups", selected=7)
            self.assertIn("GPU 0", devices)
            self.assertIn("live-1", requests)
            self.assertIn("slot 7", groups)
        detail = render_view(s, "requests", 80, 24, selected=17, detail=True)
        self.assertIn("done-16", detail)
        self.assertIn("RECENT REQUEST DETAIL", detail)
        self.assertIn("First engine token 0.700 s", detail)
        self.assertIn("Sample age", detail)
        self.assertLessEqual(len(detail.splitlines()), 24)
        self.assertTrue(all(len(line) <= 80 for line in detail.splitlines()))
        small = render_view(s, "requests", 40, 12, selected=17, detail=True)
        self.assertIn("done-16", small)
        self.assertIn("Engine first 0.70 s", small)
        self.assertIn("Prompt 116", small)
        s["work"]["requests"] = []
        only_recent = render_view(s, "requests", 40, 12, selected=0, detail=True)
        self.assertIn("done-00", only_recent)
        self.assertIn("Engine first 0.70 s", only_recent)

    def test_device_and_slot_details_fit_standard_and_compact_views(self):
        s = four_gpu_snapshot()
        s["work"]["slots"][7]["request_id"] = "request-7"
        for view, pane, width, height, selected, marker in (
            ("serving", "devices", 80, 24, 3, "GPU 3"),
            ("serving", "groups", 80, 24, 7, "Slot 7"),
            ("hardware", "devices", 80, 24, 3, "GPU 3"),
            ("serving", "devices", 60, 16, 3, "GPU 3"),
            ("serving", "groups", 40, 12, 7, "slot 7"),
            ("hardware", "devices", 40, 12, 3, "GPU 3"),
        ):
            with self.subTest(view=view, pane=pane, width=width):
                display = render_view(s, view, width, height, serving_pane=pane,
                                      selected=selected, detail=True)
                self.assertIn(marker, display)
                self.assertLessEqual(len(display.splitlines()), height)
                self.assertTrue(all(len(line) <= width for line in display.splitlines()))
                self.assertIn("q quit", display)
                if width == 80:
                    self.assertIn("DETAIL  (read only)", display)
                    self.assertIn("Sample age", display)
        device = render_view(s, "serving", 80, 24, serving_pane="devices", selected=3, detail=True)
        slot = render_view(s, "serving", 80, 24, serving_pane="groups", selected=7, detail=True)
        self.assertIn("Stage/layer mapping unavailable", device)
        self.assertIn("Request request-7", slot)
        self.assertIn("not a GPU assignment", slot)

    def test_wide_serving_scrolls_to_selected_device(self):
        s = four_gpu_snapshot()
        s["devices"] = [{"index": i, "name": f"Card-{i}", "mem_used": 8 * 2**30,
                         "mem_total": 24 * 2**30, "util": i, "temp": 40 + i,
                         "power": 100 + i} for i in range(16)]
        display = render_view(s, "serving", 110, 30, selected=15)
        self.assertIn("> GPU 15", display)
        self.assertIn("GPU SENSORS  (", display)
        self.assertIn("q quit", display)
        self.assertLessEqual(len(display.splitlines()), 30)

    def test_zero_is_not_unknown(self):
        s = normalize_legacy({"live": {"queued": 0, "tok_s": 0}, "hardware": {},
                              "totals": {"requests": 0}}, {"concurrency": {"serving": 1}})
        self.assertEqual(s["work"]["queued"], 0)
        self.assertEqual(s["rates"]["answer_tok_s"], 0)
        self.assertIsNone(s["rates"]["prefill_tok_s"])

    def test_v1_snapshot_excludes_unknown_and_untrusted_content(self):
        clean = sanitize_telemetry({"schema_version": 1, "secret": "do not render",
                                    "model": {"id": "\x1b[2Jmodel", "prompt": "private"},
                                    "work": {"slots": [{"slot": 0, "state": "decoding",
                                                        "prompt": "private"}],
                                             "requests": [{"id": "req-1", "state": "queued",
                                                           "first_engine_token_s": float("nan"),
                                                           "input": "private"}]},
                                    "rates": {"answer_tok_s": float("nan")}})
        serialized = json.dumps(clean)
        self.assertNotIn("private", serialized)
        self.assertNotIn("do not render", serialized)
        self.assertNotIn("\x1b", serialized)
        self.assertIsNone(clean["rates"]["answer_tok_s"])
        self.assertIsNone(clean["work"]["requests"][0]["first_engine_token_s"])

    def test_activity_feed_is_bounded_redacted_and_visible_at_terminal_sizes(self):
        raw = four_gpu_snapshot()
        raw["activity"] = {"last_seq": 70, "events": [
            {"seq": i, "at": 1_790_000_000.0, "request_id": f"{i:012x}",
             "event": "finished", "outcome": "stop", "prompt": "private"}
            for i in range(1, 71)
        ]}
        raw["activity"]["events"][-1]["event"] = "\x1b[2Jforged"
        clean = sanitize_telemetry(raw)
        self.assertTrue(clean["activity"]["available"])
        self.assertLessEqual(len(clean["activity"]["events"]), 32)
        self.assertNotIn("private", json.dumps(clean))
        self.assertNotIn("forged", json.dumps(clean))
        self.assertEqual(clean["activity"]["events"][-1]["seq"], 69)
        for width, height in ((110, 30), (80, 24), (60, 16), (40, 12)):
            display = render_view(clean, "logs", width, height)
            self.assertIn("activity", display)
            self.assertIn("finished", display)
            self.assertNotIn("private", display)
            self.assertNotIn("\x1b", display)
            self.assertLessEqual(len(display.splitlines()), height)
            self.assertTrue(all(len(line) <= width for line in display.splitlines()))
        legacy = render_view(four_gpu_snapshot(), "logs", 80, 24)
        self.assertIn("unavailable", legacy)
        raw["activity"] = {"last_seq": [], "events": [{"event": []}]}
        self.assertFalse(sanitize_telemetry(raw)["activity"]["available"])

    def test_cli_rejects_invalid_serve_flags_before_loading_engine(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["serve", "--profile", "missing.json", "--parallel", "0"]), 2)

    def test_offline_server_rejects_remote_image_before_fetch(self):
        from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok,
                      ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja"))
        svc.offline = True
        messages = [{"role": "user", "content": [{"type": "image", "source": "https://example.com/p.png"}]}]
        with self.assertRaisesRegex(ValueError, "offline mode cannot download"):
            svc.prepare(messages, [], {})

    def test_direct_server_remote_bind_without_key_reaches_engine_load(self):
        # no early exit over the key any more: every way of saying "no key" gets past the key check
        import serve.server as server
        cases = [('{"args":[]}', []), ('{"args":[],"api_key":"none"}', []), ('{"args":[]}', ["--no-api-key"]),
                 ('{"args":[]}', ["--api-key", "none"])]
        for body, extra in cases:
            with tempfile.TemporaryDirectory() as directory:
                profile = Path(directory) / "model.json"
                profile.write_text(body, encoding="utf-8")
                err = io.StringIO()
                with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(err):
                    with self.assertRaises(SystemExit):
                        server.main(["--engine", "strata", "--config", str(profile),
                                     "--host", "0.0.0.0", "--port", "0", *extra])
                self.assertIn("tokenizer is missing", err.getvalue(), (body, extra))   # past the key check, stopped later

    def test_code_run_uses_missing_component_exit_until_shared_harness_arrives(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["code", "run", "--prompt-file", "task.md", "--policy", "policy.json",
                         "--output", "jsonl"])
        self.assertEqual(code, 5)
        self.assertEqual(json.loads(output.getvalue())["error"]["code"], 5)


try:
    from strata_cli.monitor import MonitorApp, with_aggregate_rate
except ImportError:
    MonitorApp = None


@unittest.skipIf(MonitorApp is None, "optional Textual dependency absent")
class HeadlessMonitorTests(unittest.TestCase):
    def test_live_monitor_marks_activity_gaps_after_freeze_and_resets_on_restart(self):
        from rich.console import Console

        def snapshot(instance, last, seqs):
            value = four_gpu_snapshot()
            value["sampled_at"] = time.time()
            value["instance"]["id"] = instance
            value["activity"] = {"available": True, "last_seq": last, "events": [
                {"seq": seq, "at": value["sampled_at"], "request_id": f"{seq:012x}",
                 "event": "queued", "outcome": None} for seq in seqs]}
            return value

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return self.snapshots.pop(0)

        Source.snapshots = [snapshot("a", 5, [2, 4, 5]), snapshot("b", 2, [1, 2])]

        async def exercise():
            app = MonitorApp(Source(), snapshot("a", 2, [1, 2]), interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                app.view = 5
                app.action_freeze()
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertIsNone(app.displayed_activity_continuity["gap"])
                app.action_freeze()
                await pilot.pause()
                self.assertEqual(app.displayed_activity_continuity["gap"],
                                 {"from_seq": 3, "to_seq": 3})
                shown = str(app.query_one("#content").render())
                self.assertIn("Missed activity #3-3", shown)
                app.view = 0
                app._paint()
                await pilot.pause()
                output = io.StringIO()
                Console(file=output, width=80, color_system=None).print(app.query_one("#content").content)
                self.assertIn("Missed activity #3-3", output.getvalue())
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertIsNone(app.displayed_activity_continuity["gap"])
                self.assertTrue(app.displayed_activity_continuity["restarted"])
                app.view = 5
                app._paint()
                await pilot.pause()
                self.assertIn("Serving restarted", str(app.query_one("#content").render()))
                await pilot.press("q")

        asyncio.run(exercise())

    def test_live_compact_screen_replaces_cached_work_on_disconnect(self):
        from rich.console import Console

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["model"]["id"] = "cached-model"
        sample["work"]["requests"] = [{"id": "cached-request", "state": "decoding", "slot": 0}]

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return sample

        async def exercise(width, height, view):
            app = MonitorApp(Source(), sample, interval=30)
            async with app.run_test(size=(width, height)) as pilot:
                app.view = view
                app.connection_state = "disconnected"
                app._paint()
                await pilot.pause(0.05)
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(app.query_one("#content").content)
                shown = output.getvalue()
                self.assertIn("CONNECTION LOST", shown)
                self.assertNotIn("cached-model", shown)
                self.assertNotIn("cached-request", shown)
                self.assertNotIn("GPU-0", shown)
                await pilot.press("q")

        asyncio.run(exercise(40, 12, 0))
        asyncio.run(exercise(80, 24, 1))

    def test_overview_uses_selected_cards_and_marks_missing_sensors(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["topology"]["selected_devices"] = [2, 0, 3, 1, 2, -1, 1.5, True]
        extra = {**sample["devices"][0], "index": 5, "name": "unselected GPU"}
        sample["devices"] = [extra, sample["devices"][1], sample["devices"][2], sample["devices"][0]]
        sample = sanitize_telemetry(sample)
        self.assertEqual(sample["topology"]["selected_devices"], [2, 0, 3, 1])
        history = MonitorHistory()
        complete = four_gpu_snapshot()
        history.add(complete, ("id", "selected-cards"))
        history.add(sample, ("id", "selected-cards"))
        self.assertEqual(list(history.device_util[3]), [complete["devices"][3]["util"], None])
        self.assertEqual(list(history.device_vram[3]), [50.0, None])
        self.assertNotIn(5, history.device_util)
        self.assertNotIn(5, history.device_vram)
        for width, height in ((110, 30), (80, 24)):
            with self.subTest(width=width):
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(render_dashboard(
                    sample, width=width - 2, height=height - 2, history=history,
                    color=False))
                shown = output.getvalue()
                positions = [shown.index(f"GPU {index}") for index in (2, 0, 3, 1)]
                self.assertEqual(positions, sorted(positions))
                self.assertRegex(shown, r"GPU 3[^\n]*--/--G")
                if width == 80:
                    self.assertRegex(shown, r"GPU 3[^\n]*--/--G[^\n]*--\s+--")
                    self.assertNotIn("vram", shown)
                else:
                    self.assertRegex(shown, r"GPU 2[^\n]*vram [^\s-]+")
                    self.assertRegex(shown, r"GPU 3[^\n]*vram --")
                self.assertNotIn("GPU 5", shown)
                self.assertEqual(len(shown.splitlines()), height - 1)

        duplicate = {**sample, "devices": sample["devices"] + [dict(sample["devices"][-1])]}
        history.add(duplicate, ("id", "selected-cards"))
        self.assertIsNone(history.device_util[0][-1])
        self.assertIsNone(history.device_vram[0][-1])
        output = io.StringIO()
        Console(file=output, width=80, color_system=None).print(render_dashboard(
            duplicate, width=78, height=22, history=history, color=False))
        self.assertRegex(output.getvalue(), r"GPU 0[^\n]*--/--G")
        self.assertRegex(output.getvalue(), r"GPU 0[^\n]*--/--G[^\n]*--\s+--")

    def test_monitor_history_bounds_device_indices_across_snapshots(self):
        from strata_cli.dashboard import MonitorHistory, _output_trend

        history = MonitorHistory()
        for start in (0, 16, 32):
            history.add({"devices": [{"index": index, "util": 20}
                                     for index in range(start, start + 16)],
                         "rates": {"aggregate_output_tok_s": 42.0}}, ("id", "same"))
            self.assertEqual(set(history.device_util), set(range(start, start + 16)))
            self.assertEqual(set(history.device_vram), set(range(start, start + 16)))
        history.add({"devices": [{"index": 32, "util": 40}]}, ("id", "same"))
        self.assertEqual(set(history.device_util), {32})
        self.assertEqual(set(history.device_vram), {32})
        self.assertIsNone(history.device_vram[32][-1])
        self.assertEqual(list(history.output_rate)[-1], None)
        self.assertEqual(_output_trend(history.output_rate, "ascii", False), "Output trend  --")

    def test_vram_history_is_per_card_and_missing_values_are_not_zero(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["devices"][0]["mem_used"] = 6 * 2**30
        sample["devices"][1]["mem_used"] = 18 * 2**30
        history = MonitorHistory()
        history.add(sample, ("id", "memory-samples"))
        self.assertEqual(history.device_vram[0][-1], 25.0)
        self.assertEqual(history.device_vram[1][-1], 75.0)
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertRegex(output.getvalue(), r"GPU 0[^\n]*vram :")
        self.assertRegex(output.getvalue(), r"GPU 1[^\n]*vram \*")

        sample["devices"][0]["mem_total"] = 0
        sample["devices"][1]["mem_used"] = 30 * 2**30
        history.add(sample, ("id", "memory-samples"))
        self.assertIsNone(history.device_vram[0][-1])
        self.assertIsNone(history.device_vram[1][-1])
        history.gap()
        self.assertIsNone(history.device_vram[0][-1])
        history.add(sample, ("id", "new-instance"))
        self.assertEqual(len(history.device_vram[0]), 1)
        self.assertIsNone(history.device_vram[0][-1])

        sample["devices"][0]["mem_used"] = 1e308
        sample["devices"][0]["mem_total"] = 1e308
        history.add(sample, ("id", "new-instance"))
        self.assertEqual(history.device_vram[0][-1], 100.0)
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertRegex(output.getvalue(), r"GPU 0[^\n]*vram\s+@")

    def test_overview_output_trend_is_bounded_and_resets_with_instance(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, _output_trend, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        history = MonitorHistory()
        for value in (100.0, 120.0, 110.0):
            sample["rates"]["aggregate_output_tok_s"] = value
            history.add(sample, ("id", "first"))
        for width, height in ((110, 30), (80, 24), (110, 42)):
            with self.subTest(width=width):
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(render_dashboard(
                    sample, width=width - 2, height=height - 2, history=history,
                    glyphs="ascii", color=False))
                shown = output.getvalue()
                self.assertIn("100-120 tok/s", shown)
                self.assertLessEqual(len(shown.splitlines()), height - 1)
                self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))

        stale = {**sample, "sampled_at": time.time() - 6}
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            stale, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertIn("Output trend  --", output.getvalue())
        self.assertNotIn("100-120 tok/s", output.getvalue())

        sample["rates"]["aggregate_output_tok_s"] = None
        history.add(sample, ("id", "first"))
        self.assertEqual(_output_trend(history.output_rate, "ascii", False), "Output trend  --")
        sample["rates"]["aggregate_output_tok_s"] = 90.0
        history.add(sample, ("id", "second"))
        self.assertEqual(list(history.output_rate), [90.0])
        self.assertIn("▁▁  0 tok/s", _output_trend([0.0, 0.0], "unicode", True))

    def test_batch_and_queue_histories_keep_replica_windows_separate(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        history = MonitorHistory(limit=3)
        for rows, queued in ((4.0, 1), (6.0, 5)):
            sample["work"].update(effective_rows_window=rows, batch_sample_age_s=0.5,
                                  queued=queued)
            history.add(sample, ("id", "one"))
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="ascii", color=False))
        shown = output.getvalue()
        self.assertIn("Rows trend", shown)
        self.assertIn("4.0-6.0/win", shown)
        self.assertIn("Pre-admit", shown)
        self.assertIn("1-5 req", shown)
        self.assertEqual(len(shown.splitlines()), 29)
        compact = io.StringIO()
        Console(file=compact, width=80, color_system=None).print(render_dashboard(
            sample, width=78, height=22, history=history, glyphs="ascii", color=False))
        self.assertNotIn("Rows trend", compact.getvalue())
        self.assertIn("1 Over", compact.getvalue())
        history.gap()
        self.assertIsNone(history.batch_rows[-1])
        self.assertIsNone(history.pre_admit[-1])
        self.assertIsNone(history.device_util[0][-1])
        after_gap = io.StringIO()
        Console(file=after_gap, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertNotIn("Rows trend", after_gap.getvalue())
        self.assertNotIn("Pre-admit  ", after_gap.getvalue())

        sample["work"]["batch_sample_age_s"] = 5.1
        history.add(sample, ("id", "one"))
        old_single = io.StringIO()
        Console(file=old_single, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertIsNone(history.batch_rows[-1])
        self.assertIn("Rows/win -- processed", old_single.getvalue())
        self.assertNotIn("Rows trend", old_single.getvalue())

        replica = copy.deepcopy(sample)
        replica["topology"]["mode"] = "replicas (configured)"
        replica["topology"]["replica_devices"] = [{"replica": 0, "devices": [0, 1]},
                                                    {"replica": 1, "devices": [2, 3]}]
        replica["work"]["effective_rows_window"] = None
        for first, second in ((4.0, 3.0), (5.0, 4.0)):
            replica["work"]["replica_batches"] = [
                {"replica": 0, "effective_rows_window": first, "batch_active_rows_window": 2.0,
                 "batch_sample_age_s": 0.5},
                {"replica": 1, "effective_rows_window": second, "batch_active_rows_window": 2.0,
                 "batch_sample_age_s": 0.5}]
            history.add(replica, ("id", "two"))
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            replica, width=108, height=28, history=history, glyphs="ascii", color=False))
        shown = output.getvalue()
        self.assertIn("R0 rows", shown)
        self.assertIn("4.0-5.0/win", shown)
        self.assertIn("R1 rows", shown)
        self.assertIn("3.0-4.0/win", shown)
        self.assertNotIn("Rows trend", shown)

        replica["work"]["replica_batches"][1]["batch_sample_age_s"] = 5.1
        history.add(replica, ("id", "two"))
        self.assertIsNone(history.replica_batch_rows[1][-1])
        old_sample = io.StringIO()
        Console(file=old_sample, width=110, color_system=None).print(render_dashboard(
            replica, width=108, height=28, history=history, glyphs="ascii", color=False))
        self.assertIn("R1 --/--", old_sample.getvalue())
        self.assertNotIn("R1 rows", old_sample.getvalue())

        replica["work"]["replica_batches"] = [replica["work"]["replica_batches"][0]]
        history.add(replica, ("id", "two"))
        self.assertIsNone(history.replica_batch_rows[1][-1])
        self.assertEqual(len(history.replica_batch_rows[0]), 3)
        history.add(replica, ("id", "three"))
        self.assertEqual(len(history.batch_rows), 1)
        self.assertEqual(len(history.replica_batch_rows[0]), 1)

    def test_overview_slot_phases_require_complete_slot_report(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["work"]["slots"][0]["state"] = "reading"
        self.assertIn("Occupied slots by group  G0 2/2", render_view(sample, "serving", 110, 30))
        for width, height in ((110, 30), (80, 24)):
            with self.subTest(width=width):
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(render_dashboard(
                    sample, width=width - 2, height=height - 2, history=MonitorHistory(),
                    color=False))
                shown = output.getvalue()
                self.assertIn("read 1  decode 5  idle 2", shown)
                self.assertLessEqual(len(shown.splitlines()), height - 1)
                self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))

        incomplete = dict(sample["work"])
        incomplete["slots"] = incomplete["slots"][:-1]
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            {**sample, "work": incomplete}, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertNotIn("Slot phases", output.getvalue())
        self.assertIn("Group occupancy --", output.getvalue())
        self.assertIn("Occupied slots by group -- (slot report incomplete)",
                      render_view({**sample, "work": incomplete}, "serving", 110, 30))

        duplicated = dict(sample["work"])
        duplicated["slots"] = [dict(row) for row in sample["work"]["slots"]]
        duplicated["slots"][1]["slot"] = duplicated["slots"][0]["slot"]
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            {**sample, "work": duplicated}, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertNotIn("Slot phases", output.getvalue())
        self.assertIn("Group occupancy --", output.getvalue())

        invalid_group = dict(sample["work"])
        invalid_group["slots"] = [dict(row) for row in sample["work"]["slots"]]
        invalid_group["slots"][1]["group"] = None
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            {**sample, "work": invalid_group}, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertIn("Slot phases", output.getvalue())
        self.assertIn("Group occupancy --", output.getvalue())

        invalid_group["slots"][1]["group"] = 9
        self.assertIn("Occupied slots by group -- (slot report incomplete)",
                      render_view({**sample, "work": invalid_group}, "serving", 110, 30))
        invalid_group["slots"][1]["group"] = 0
        invalid_group["slots"][1]["slot"] = 99
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            {**sample, "work": invalid_group}, width=108, height=28, history=MonitorHistory(), color=False))
        self.assertNotIn("Slot phases", output.getvalue())
        self.assertIn("Group occupancy --", output.getvalue())

    def test_connection_panel_hides_cached_work_until_recovery(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time() - 4
        sample["rates"]["answer_tok_s"] = 42.5
        sample["work"]["requests"] = [{"id": "cached-request", "state": "decoding", "slot": 0}]
        expected = {"connecting": "CONNECTING", "disconnected": "CONNECTION LOST",
                    "auth_required": "AUTHORIZATION REQUIRED", "incompatible": "INCOMPATIBLE SERVER"}
        for width, height in ((110, 30), (80, 24)):
            for state, title in expected.items():
                with self.subTest(width=width, state=state):
                    output = io.StringIO()
                    Console(file=output, width=width, color_system=None).print(render_dashboard(
                        sample, width=width - 2, height=height - 2, history=MonitorHistory(),
                        connection_state=state, color=False))
                    shown = output.getvalue()
                    self.assertIn(title, shown)
                    self.assertNotIn("cached-request", shown)
                    self.assertNotIn("42.5 tok/s", shown)
                    self.assertNotIn("GPU 0", shown)
                    self.assertLessEqual(len(shown.splitlines()), height - 1)
                    self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))

    def test_compact_connection_panel_hides_cached_work_across_views(self):
        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time() - 4
        sample["model"]["id"] = "cached-model"
        sample["rates"]["answer_tok_s"] = 42.5
        sample["work"]["requests"] = [{"id": "cached-request", "state": "decoding", "slot": 0}]
        titles = {"connecting": "CONNECTING", "disconnected": "CONNECTION LOST",
                  "auth_required": "AUTHORIZATION REQUIRED", "incompatible": "INCOMPATIBLE SERVER"}
        for width, height in ((80, 24), (60, 16), (40, 12)):
            for view in ("overview", "serving", "requests", "engine", "hardware", "logs"):
                for state, title in titles.items():
                    with self.subTest(width=width, view=view, state=state):
                        shown = render_view(sample, view, width, height, connection_state=state)
                        self.assertIn(title, shown)
                        for cached in ("cached-model", "cached-request", "42.5 tok/s", "GPU-0"):
                            self.assertNotIn(cached, shown)
                        self.assertLessEqual(len(shown.splitlines()), height)
                        self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))

    def test_overview_dashboard_keeps_parallel_work_and_four_devices_on_one_screen(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, _spark, render_dashboard
        from strata_cli.monitor import choose_glyphs

        self.assertEqual(choose_glyphs("auto", "utf-8", "xterm-256color"), "unicode")
        self.assertEqual(choose_glyphs("auto", "ascii", "xterm-256color"), "ascii")
        self.assertEqual(choose_glyphs("auto", "utf-8", "dumb"), "ascii")
        self.assertEqual(choose_glyphs("braille", "ascii", "dumb"), "braille")

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["instance"]["id"] = "fixture-instance"
        sample["rates"].update(answer_tok_s=42.5, prefill_tok_s=180.0,
                               aggregate_output_tok_s=126.0)
        sample["work"].update(effective_rows_window=5.0, batch_active_rows_window=3.0,
                              batch_sample_age_s=0.5,
                              requests=[{"id": "queued-1", "state": "queued", "slot": None,
                                         "group": None, "queued_at": sample["sampled_at"] - 2},
                                        {"id": "running-1", "state": "decoding", "slot": 0,
                                         "group": 0, "generated": 18}])
        sample["activity"] = {"available": True, "events": [
            {"seq": 1, "at": sample["sampled_at"], "request_id": "running-1", "event": "admitted"}]}
        history = MonitorHistory()
        history.add(sample, ("id", "fixture-instance"))
        for width, height in ((110, 30), (80, 24), (110, 42)):
            with self.subTest(width=width):
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(render_dashboard(
                    sample, width=width - 2, height=height - 2, history=history,
                    connection_state="connected", glyphs="ascii", color=False))
                shown = output.getvalue()
                self.assertLessEqual(len(shown.splitlines()), height - 1)
                self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))
                for expected in ("BATCHING / PARALLEL", "HARDWARE", "REQUESTS", "ACTIVITY",
                                 "5.0", "queued-1", "GPU 0", "GPU 1", "GPU 2", "GPU 3",
                                 "G0", "G3", "42.5 tok/s"):
                    self.assertIn(expected, shown)
                if width == 110:
                    self.assertIn("Stage host span", shown)
                if height == 42:
                    self.assertIn("SESSION", shown)
                    self.assertIn("ENGINE", shown)
                if width == 80:
                    self.assertIn("1 Over", shown)
                self.assertRegex(shown, r"queued-1[^\n]*wait")
                self.assertNotRegex(shown, r"queued-1[^\n]*out")

        history.add({"devices": [{"index": 0, "util": 90}]}, ("id", "restarted"))
        self.assertEqual(list(history.device_util[0]), [90])
        self.assertNotIn(1, history.device_util)
        self.assertTrue(all("\u2800" <= char <= "\u28ff" for char in _spark([25, 75, 50, 100], "braille")))
        self.assertEqual(_spark([None, None], "braille"), "--")
        self.assertEqual(len(_spark([25, 75, 50, 100], "unicode")), 4)
        sample["devices"][0]["name"] = "漢字 RTX 6000"
        sample["work"]["requests"][0]["id"] = "\x1b[2J漢字-queued\u202e"
        sample["work"]["slots"][0]["state"] = None
        output = io.StringIO()
        Console(file=output, width=80, color_system=None).print(render_dashboard(
            sample, width=78, height=22, history=history, color=False))
        self.assertIn("Group occupancy --", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())
        self.assertNotIn("\u202e", output.getvalue())
        self.assertTrue(all(display_width(line) <= 80 for line in output.getvalue().splitlines()))

        sample["work"]["requests"].append({"id": "reading-1", "state": "reading", "slot": 2,
                                               "group": 1, "prompt_read": 250, "prompt_tokens": 1000})
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="unicode", color=False))
        self.assertIn("Reading reading-1", output.getvalue())
        self.assertIn("250/1,000", output.getvalue())
        self.assertTrue(all(display_width(line) <= 110 for line in output.getvalue().splitlines()))

        sample["model"]["loaded"] = False
        output = io.StringIO()
        Console(file=output, width=110, color_system=None).print(render_dashboard(
            sample, width=108, height=28, history=history, glyphs="unicode", color=False))
        self.assertIn("SLEEPING", output.getvalue())
        self.assertNotIn("42.5 tok/s", output.getvalue())

    def test_wide_visual_overview_uses_real_rates_slots_and_request_rows(self):
        from rich.console import Console
        from strata_cli.dashboard import MonitorHistory, render_dashboard

        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["model"]["context"] = 32768
        sample["rates"].update(answer_tok_s=41.8, prefill_tok_s=188.6,
                               aggregate_output_tok_s=119.4)
        sample["work"].update(slots_effective=8, queued=2, effective_rows_window=4.0,
                              batch_active_rows_window=3.0, batch_sample_age_s=0.5,
                              requests=[{"id": "queue-a", "state": "queued", "slot": None,
                                         "prompt_tokens": 500, "queued_at": sample["sampled_at"] - 2},
                                        {"id": "decode-b", "state": "decoding", "slot": 0,
                                         "group": 0, "prompt_tokens": 200, "generated": 37,
                                         "first_engine_token_s": 0.71}])
        sample["work"]["slots"][0]["held_tokens"] = 1200
        history = MonitorHistory()
        history.add(sample, ("id", "visual-fixture"))
        history.add(sample, ("id", "visual-fixture"))
        for width, height in ((140, 44), (128, 36)):
            with self.subTest(width=width, height=height):
                output = io.StringIO()
                Console(file=output, width=width, color_system=None).print(render_dashboard(
                    sample, width=width - 2, height=height - 2, history=history,
                    connection_state="connected", glyphs="ascii", color=False))
                shown = output.getvalue()
                self.assertLessEqual(len(shown.splitlines()), height - 1)
                self.assertTrue(all(display_width(line) <= width for line in shown.splitlines()))
                for expected in ("THROUGHPUT", "PROMPT 188.6 tok/s", "119.4 tok/s",
                                 "GPUS", "GPU 0", "GPU 3", "CONTEXT / BATCHING",
                                 "MAX ACTIVE HELD 1,200", "PARALLEL SLOTS", "G0",
                                 "REQUESTS", "queue-a", "decode-b", "wait 2.0s",
                                 "1ST ENGINE"):
                    self.assertIn(expected, shown)
                self.assertNotIn("cached 0", shown)
        history.gap()
        self.assertIsNone(history.prefill_rate[-1])
        self.assertIsNone(history.answer_rate[-1])
        stale_output = io.StringIO()
        Console(file=stale_output, width=140, color_system=None).print(render_dashboard(
            sample, width=138, height=42, history=history,
            connection_state="stale", glyphs="ascii", color=False))
        self.assertIn("MAX ACTIVE HELD --", stale_output.getvalue())
        self.assertIn("PROMPT -- tok/s", stale_output.getvalue())
        self.assertNotIn("119.4 tok/s", stale_output.getvalue())
        idle = copy.deepcopy(sample)
        idle["work"].update(active=0, queued=0, waiting=0, occupied_slots=0, requests=[])
        for slot in idle["work"]["slots"]:
            slot["state"] = "idle"
        idle_output = io.StringIO()
        Console(file=idle_output, width=140, color_system=None).print(render_dashboard(
            idle, width=138, height=42, history=history,
            connection_state="connected", glyphs="ascii", color=False))
        self.assertIn("MAX ACTIVE HELD 0", idle_output.getvalue())
        self.assertIn("No active requests", idle_output.getvalue())

    def test_initial_attach_keeps_status_when_telemetry_is_unavailable(self):
        status_only = sanitize_telemetry(normalize_status_only(
            {"service": "strata", "model": "fixture", "loaded": False,
             "concurrency": {"serving": 4}}))

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return status_only

        async def exercise():
            app = MonitorApp(Source(), None, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.1)
                self.assertEqual(app.connection_state, "telemetry_unavailable")
                self.assertTrue(app.has_snapshot)
                self.assertEqual(app.displayed["model"]["id"], "fixture")
                from rich.console import Console
                output = io.StringIO()
                console = Console(file=output, width=80, color_system=None)
                console.print(app.query_one("#content").content)
                self.assertIn("Telemetry unavailable; status only", output.getvalue())
                self.assertIn("NO MODEL LOADED", output.getvalue())
                await pilot.press("q")
        asyncio.run(exercise())

    def test_stale_snapshot_hides_live_rates_without_changing_cached_values(self):
        sample = four_gpu_snapshot()
        sample["sampled_at"] = time.time()
        sample["rates"].update(answer_tok_s=12, prefill_tok_s=24, aggregate_output_tok_s=36)
        live = render_view(sample, "overview", 80, 24, connection_state="connected")
        stale = render_view(sample, "overview", 80, 24, connection_state="stale")
        serving = render_view(sample, "serving", 110, 30, connection_state="stale")
        self.assertIn("[LIVE]", live)
        self.assertIn("Answer speed  12.0 tok/s", live)
        self.assertIn("[STALE]", stale)
        self.assertIn("Answer speed  --", stale)
        self.assertIn("Prompt reading --", stale)
        self.assertIn("Total output --", serving)
        self.assertEqual(sample["rates"]["answer_tok_s"], 12)
        self.assertIn("[STALE]", render_view({**sample, "sampled_at": time.time() - 10},
                                               "overview", 80, 24, connection_state="connected"))

    def test_failed_poll_marks_stale_immediately_then_recovers(self):
        initial = four_gpu_snapshot()
        initial["sampled_at"] = time.time()
        initial["rates"]["answer_tok_s"] = 12
        initial["rates"]["output_tokens_server_counter"] = 100
        initial["instance"]["id"] = "same-instance"
        sample_time = initial["sampled_at"]

        class Source:
            root = "http://127.0.0.1:8095"
            calls = 0

            def snapshot(self):
                self.calls += 1
                if self.calls <= 3:
                    raise ClientError("temporarily unavailable", 3)
                result = four_gpu_snapshot()
                result["instance"]["id"] = "same-instance"
                result["sampled_at"] = sample_time + self.calls - 4
                result["rates"]["output_tokens_server_counter"] = 160 + 30 * (self.calls - 4)
                return result

        async def exercise():
            from rich.console import Console

            source = Source()
            app = MonitorApp(source, initial, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                for expected in ("stale", "stale", "disconnected", "connected"):
                    app._schedule_poll(force=True)
                    await pilot.pause(0.1)
                    self.assertEqual(app.connection_state, expected)
                    output = io.StringIO()
                    Console(file=output, width=80, color_system=None).print(app.query_one("#content").content)
                    if expected == "disconnected":
                        self.assertIn("CONNECTION LOST", output.getvalue())
                        self.assertNotIn("GPU 0", output.getvalue())
                    if expected in ("stale", "disconnected"):
                        self.assertIsNone(app.history.batch_rows[-1])
                        self.assertIsNone(app.history.pre_admit[-1])
                        self.assertIsNone(app.history.device_util[0][-1])
                    if expected == "connected":
                        self.assertIn("GPU 0", output.getvalue())
                        self.assertNotIn("CONNECTION LOST", output.getvalue())
                self.assertEqual(source.calls, 4)
                self.assertEqual(app.failures, 0)
                self.assertIsNone(app.latest["rates"]["aggregate_output_tok_s"])
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.latest["rates"]["aggregate_output_tok_s"], 30)
                await pilot.press("q")
        asyncio.run(exercise())

    def test_unexpected_poll_error_marks_snapshot_stale_without_leaking_details(self):
        initial = four_gpu_snapshot()
        initial["sampled_at"] = time.time()

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                raise RuntimeError("private URL and credential")

        async def exercise():
            app = MonitorApp(Source(), initial, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.connection_state, "stale")
                self.assertEqual(app.error, "Telemetry collection failed")
                self.assertIsNone(app.rate_previous)
                self.assertNotIn("private URL", str(app.query_one("#message").render()))
                await pilot.press("q")
        asyncio.run(exercise())

    def test_initial_attach_shows_connecting_and_requires_manual_retry_after_auth(self):
        class Source:
            root = "http://127.0.0.1:8095"
            calls = 0

            def snapshot(self):
                self.calls += 1
                if self.calls == 1:
                    raise ClientError("credential required", 4)
                if self.calls == 2:
                    raise ClientError("unsupported telemetry schema", 5)
                result = four_gpu_snapshot()
                result["sampled_at"] = time.time()
                return result

        async def exercise():
            source = Source()
            app = MonitorApp(source, None, interval=30)
            self.assertEqual(app.connection_state, "connecting")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.1)
                self.assertEqual(app.connection_state, "auth_required")
                self.assertEqual(app.next_poll_at, float("inf"))
                self.assertFalse(app.has_snapshot)
                app._schedule_poll()
                await pilot.pause(0.1)
                self.assertEqual(source.calls, 1)
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.connection_state, "incompatible")
                self.assertEqual(app.next_poll_at, float("inf"))
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.connection_state, "connected")
                self.assertTrue(app.has_snapshot)
                self.assertIsNone(app.selection_notice)
                await pilot.press("q")
        asyncio.run(exercise())

    def test_restart_clears_matching_request_id_after_freeze(self):
        initial = four_gpu_snapshot()
        initial["instance"]["id"] = "server-old"
        initial["work"]["requests"] = [{"id": "reused-id", "state": "decoding"}]
        initial["recent"] = []
        restarted = four_gpu_snapshot()
        restarted["instance"]["id"] = "server-new"
        restarted["work"]["requests"] = [{"id": "reused-id", "state": "queued"}]
        restarted["recent"] = []

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return restarted

        async def exercise():
            app = MonitorApp(Source(), initial, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("3", "enter", "space")
                self.assertTrue(app.detail)
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.displayed["instance"]["id"], "server-old")
                self.assertTrue(app.detail)
                await pilot.press("space")
                self.assertEqual(app.displayed["instance"]["id"], "server-new")
                self.assertFalse(app.detail)
                self.assertIsNone(app.selected_identity)
                self.assertEqual(app.selection_notice, "Server instance changed; selection cleared")
                await pilot.press("q")
        asyncio.run(exercise())

    def test_table_keys_and_device_identity_survive_reorder(self):
        initial = four_gpu_snapshot()
        initial["work"]["requests"] = [{"id": f"r{i:02d}", "state": "queued"} for i in range(20)]
        initial["recent"] = []
        reordered = four_gpu_snapshot()
        reordered["devices"].reverse()
        disappeared = four_gpu_snapshot()
        disappeared["devices"] = disappeared["devices"][:3]

        class Source:
            root = "http://127.0.0.1:8095"

            def __init__(self):
                self.values = [reordered, disappeared]

            def snapshot(self):
                return self.values.pop(0)

        async def exercise():
            app = MonitorApp(Source(), initial, interval=30, requests=20)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("3", "end")
                self.assertEqual(app.selected, 19)
                await pilot.press("pageup")
                self.assertEqual(app.selected, 7)
                await pilot.press("home", "pagedown")
                self.assertEqual(app.selected, 12)
                await pilot.press("right")
                self.assertEqual(app.view, 3)
                await pilot.press("left", "5", "end", "enter")
                self.assertEqual(app.selected_identity, ("device", "3"))
                self.assertTrue(app.detail)
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.selected, 0)
                self.assertTrue(app.detail)
                self.assertIn("GPU 3", str(app.query_one("#content").render()))
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertFalse(app.detail)
                self.assertEqual(app.selection_notice, "No longer retained; select another row")
                await pilot.press("q")
        asyncio.run(exercise())

    def test_slot_detail_follows_slot_across_snapshot_reorder(self):
        initial = four_gpu_snapshot()
        reordered = four_gpu_snapshot()
        reordered["work"]["slots"].reverse()

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return reordered

        async def exercise():
            app = MonitorApp(Source(), initial, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("2", "tab", "tab", "end", "enter")
                self.assertEqual(app.selected_identity, ("slot", "7"))
                self.assertTrue(app.detail)
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.selected, 0)
                self.assertTrue(app.detail)
                self.assertIn("Slot 7", str(app.query_one("#content").render()))
                await pilot.press("q")
        asyncio.run(exercise())

    def test_request_detail_follows_live_request_into_recent_history(self):
        value = four_gpu_snapshot()
        value["work"]["requests"] = [{"id": "finishing-1", "state": "decoding"}]
        value["recent"] = []
        finished = four_gpu_snapshot()
        finished["work"]["requests"] = []
        finished["recent"] = [{"id": "finishing-1", "finish": "stop", "output_tokens": 9}]

        class Source:
            root = "http://127.0.0.1:8095"

            def snapshot(self):
                return finished

        async def exercise():
            app = MonitorApp(Source(), value, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("3", "enter")
                self.assertEqual(app.selected_identity, ("live", "finishing-1"))
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.selected_identity, ("recent", "finishing-1"))
                self.assertTrue(app.detail)
                self.assertIn("RECENT REQUEST DETAIL", str(app.query_one("#content").render()))
                await pilot.press("q")
        asyncio.run(exercise())

    def test_recent_detail_follows_identity_and_closes_when_it_disappears(self):
        def snapshot(recent_ids):
            value = four_gpu_snapshot()
            value["work"]["requests"] = [{"id": "live-1", "state": "decoding"}]
            value["recent"] = [{"id": item, "finish": "stop", "prompt_tokens": 100,
                                "output_tokens": 2, "duration_s": 0.5,
                                "first_engine_token_s": 0.2} for item in recent_ids]
            value["sampled_at"] = time.time()
            return value

        class Source:
            root = "http://127.0.0.1:8095"

            def __init__(self):
                self.values = [snapshot(["new-1", "done-1"]), snapshot(["new-1"])]

            def snapshot(self):
                return self.values.pop(0)

        async def exercise():
            app = MonitorApp(Source(), snapshot(["done-1"]), interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("3", "down", "enter")
                await pilot.pause()
                self.assertTrue(app.detail)
                self.assertEqual(app.selected_identity, ("recent", "done-1"))
                self.assertIn("RECENT REQUEST DETAIL", str(app.query_one("#content").render()))
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertEqual(app.selected, 2)
                self.assertTrue(app.detail)
                self.assertIn("done-1", str(app.query_one("#content").render()))
                app._schedule_poll(force=True)
                await pilot.pause(0.1)
                self.assertFalse(app.detail)
                self.assertEqual(app.selection_notice, "No longer retained; select another row")
                await pilot.press("q")
        asyncio.run(exercise())

    def test_polling_has_one_in_flight_cycle_and_freeze_keeps_display(self):
        class Source:
            root = "http://127.0.0.1:8095"
            calls = 0

            def snapshot(self):
                self.calls += 1
                time.sleep(0.1)
                result = four_gpu_snapshot()
                result["sampled_at"] = time.time()
                return result

        async def exercise():
            source = Source()
            initial = four_gpu_snapshot()
            app = MonitorApp(source, initial, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                app.action_freeze()
                app._schedule_poll(force=True)
                app._schedule_poll(force=True)
                await pilot.pause(0.2)
                self.assertEqual(source.calls, 1)
                self.assertIs(app.displayed, initial)
                app.action_freeze()
                self.assertIs(app.displayed, app.latest)
                await pilot.press("q")
        asyncio.run(exercise())

    def test_aggregate_rate_uses_one_window_and_resets_on_restart(self):
        old = {"instance": {"id": "one"}, "sampled_at": 10.0,
               "rates": {"output_tokens_server_counter": 100}}
        fresh = {"instance": {"id": "one"}, "sampled_at": 12.0,
                 "rates": {"output_tokens_server_counter": 160}}
        self.assertEqual(with_aggregate_rate(old, fresh)["rates"]["aggregate_output_tok_s"], 30)
        restarted = {"instance": {"id": "two"}, "sampled_at": 14.0,
                     "rates": {"output_tokens_server_counter": 3}}
        self.assertIsNone(with_aggregate_rate(fresh, restarted)["rates"]["aggregate_output_tok_s"])
        invalid = {"instance": {"id": "one"}, "sampled_at": float("inf"),
                   "rates": {"output_tokens_server_counter": 190}}
        self.assertIsNone(with_aggregate_rate(fresh, invalid)["rates"]["aggregate_output_tok_s"])
        stale = {"instance": {"id": "one"}, "sampled_at": time.time() - 10,
                 "rates": {"output_tokens_server_counter": 100}}
        self.assertIsNone(MonitorApp(Endpoint("http://127.0.0.1:8080"), stale).rate_previous)

    def test_navigation_freeze_and_quit_at_supported_sizes(self):
        async def exercise(size):
            endpoint = Endpoint("http://127.0.0.1:18765")
            app = MonitorApp(endpoint, four_gpu_snapshot(), interval=30)
            async with app.run_test(size=size) as pilot:
                await pilot.press("2")
                await pilot.pause()
                self.assertEqual(app.view, 1)
                await pilot.press("tab")
                await pilot.pause()
                self.assertEqual(app.serving_pane, "requests")
                await pilot.press("enter")
                await pilot.pause()
                self.assertFalse(app.detail)  # fixture has no correlated request rows
                await pilot.press("space")
                await pilot.pause()
                self.assertTrue(app.frozen)
                await pilot.press("space")
                await pilot.pause()
                self.assertFalse(app.frozen)
                await pilot.press("q")
        for size in ((110, 30), (80, 24), (60, 16), (40, 12)):
            asyncio.run(exercise(size))

    def test_resize_keeps_selected_request_visible_in_tiny_serving_view(self):
        async def exercise():
            snapshot = four_gpu_snapshot()
            snapshot["work"]["requests"] = [
                {"id": "selected-1", "state": "reading", "slot": 0,
                 "group": 0, "prompt_read": 8, "prompt_tokens": 64}]
            app = MonitorApp(Endpoint("http://127.0.0.1:18765"), snapshot, interval=30)
            async with app.run_test(size=(110, 30)) as pilot:
                await pilot.press("2", "tab", "enter")
                await pilot.pause()
                self.assertTrue(app.detail)
                identity = app.selected_identity
                for width, height in ((80, 24), (60, 16), (40, 12), (110, 30)):
                    await pilot.resize_terminal(width, height)
                    content = str(app.query_one("#content").render())
                    self.assertEqual((app.screen.size.width, app.screen.size.height),
                                     (width, height))
                    self.assertIn("selected-1", content)
                    self.assertEqual(app.selected_identity, identity)
                    self.assertTrue(app.detail)
                    if (width, height) == (40, 12):
                        self.assertIn("More: strata status --json", content)
                        self.assertIn("q keeps server on", str(app.query_one("#message").render()))
                    elif (width, height) == (110, 30):
                        self.assertNotIn("More: strata status --json", content)
                await pilot.press("q")
        asyncio.run(exercise())

    def test_request_detail_key_is_read_only(self):
        async def exercise():
            snapshot = four_gpu_snapshot()
            snapshot["work"]["requests"] = [{"id": "r1", "state": "reading", "slot": 0,
                                               "group": 0, "prompt_read": 8, "prompt_tokens": 64}]
            endpoint = Endpoint("http://127.0.0.1:18765")
            app = MonitorApp(endpoint, snapshot, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("2", "tab", "enter")
                await pilot.pause()
                self.assertTrue(app.detail)
                self.assertIn("REQUEST DETAIL", str(app.query_one("#content").render()))
                await pilot.press("enter")
                await pilot.pause()
                self.assertFalse(app.detail)
                await pilot.press("q")
        asyncio.run(exercise())

    def test_filter_paste_is_bounded_clean_and_keeps_keyboard_focus(self):
        from textual.events import Paste
        from textual.widgets import Input

        async def exercise():
            snapshot = four_gpu_snapshot()
            snapshot["work"]["requests"] = [{"id": "r01", "state": "reading"},
                                               {"id": "r02", "state": "queued"}]
            app = MonitorApp(Endpoint("http://127.0.0.1:18765"), snapshot, interval=30)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.press("3", "slash")
                field = app.query_one("#filter", Input)
                self.assertIs(app.focused, field)
                field.post_message(Paste("x" * 1024 + "\u202e"))
                await pilot.pause()
                self.assertLessEqual(len(field.value), 80)
                self.assertNotIn("\u202e", field.value)
                await pilot.press("q")
                self.assertFalse(app.closing)
                field.value = ""
                field.post_message(Paste("\u202er01"))
                await pilot.pause()
                self.assertEqual(field.value, "r01")
                await pilot.press("enter")
                self.assertEqual(app.filter_text, "r01")
                self.assertIsNone(app.focused)
                shown = str(app.query_one("#content").render())
                self.assertIn("r01", shown)
                self.assertNotIn("r02", shown)
                await pilot.press("q")
                self.assertTrue(app.closing)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
