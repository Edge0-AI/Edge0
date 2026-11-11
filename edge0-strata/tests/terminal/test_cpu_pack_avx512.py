"""A CPU without AVX-512 (Ryzen 3000-5000, Intel Core 12th-14th generation) and the pack for AVX-512 CPUs: the Terminal says so
plainly (strata doctor before the start, the launcher after an engine stop), and does not blame the CPU for a pack that is simply
not the native one."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import runner as R
from strata_control import doctor

ZEN3 = {"fpu", "avx", "avx2", "fma", "bmi2"}
ZEN4 = ZEN3 | doctor.AVX512_FULL


class CpuPackIssue(unittest.TestCase):
    def pack(self, d, *names):
        p = Path(d) / "pack"
        p.mkdir()
        for n in names:
            (p / n).write_text("x", encoding="utf-8")
        return p

    def cfg(self, p):
        return {"args": ["--serve", "--pack", str(p), "--native", "x.gguf"], "cwd": str(p.parent)}

    def test_the_avx512_pack_on_an_avx2_cpu_is_named_with_the_next_step(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(doctor, "_cpu_flags", return_value=ZEN3):
            p = self.pack(d, "index.txt", "experts.bin", "dense.bin")
            issue = doctor._cpu_pack_issue(self.cfg(p), Path(d) / "run.json")
            self.assertEqual(issue["code"], "CPU_PACK_NEEDS_AVX512")
            self.assertIn("no native_experts.txt", issue["message"])
            self.assertIn("strata setup", issue["message"])
            # a half-built pack (what stopped an Arc A750 on a Ryzen 5 5600X) reads the same way
            q = Path(d) / "half"
            q.mkdir()
            (q / "index.txt").write_text("x", encoding="utf-8")
            self.assertEqual(doctor._cpu_pack_issue(self.cfg(q), Path(d) / "run.json")["code"], "CPU_PACK_NEEDS_AVX512")

    def test_nothing_is_said_when_the_cpu_has_avx512_the_pack_is_native_or_it_is_another_engine_pack(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.pack(d, "index.txt", "experts.bin")
            with mock.patch.object(doctor, "_cpu_flags", return_value=ZEN4):
                self.assertIsNone(doctor._cpu_pack_issue(self.cfg(p), Path(d) / "run.json"))
            with mock.patch.object(doctor, "_cpu_flags", return_value=ZEN3):
                (p / "native_experts.txt").write_text("x", encoding="utf-8")
                self.assertIsNone(doctor._cpu_pack_issue(self.cfg(p), Path(d) / "run.json"))
                (p / "native_experts.txt").unlink()
                (p / "ds4.txt").write_text("x", encoding="utf-8")      # a DeepSeek pack: another engine, another kernel
                self.assertIsNone(doctor._cpu_pack_issue(self.cfg(p), Path(d) / "run.json"))
                (p / "ds4.txt").unlink()
                self.assertIsNone(doctor._cpu_pack_issue({"args": ["--serve"]}, Path(d) / "run.json"))
            with mock.patch.object(doctor, "_cpu_flags", return_value=None):      # flags unreadable: no guess
                self.assertIsNone(doctor._cpu_pack_issue(self.cfg(p), Path(d) / "run.json"))

    def test_cpu_flags_reads_the_flags_line(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cpuinfo"
            f.write_text("processor\t: 0\nflags\t\t: fpu avx2 avx512f\nbogomips\t: 1\n", encoding="utf-8")
            with mock.patch.object(doctor.sys, "platform", "linux"):
                self.assertEqual(doctor._cpu_flags(str(f)), {"fpu", "avx2", "avx512f"})
                self.assertIsNone(doctor._cpu_flags(str(Path(d) / "missing")))


class EngineStopIsExplained(unittest.TestCase):
    def test_the_engine_refusal_gets_a_plain_next_step(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("strata: this CPU cannot run the expert kernel: missing AVX512F AVX512BW.\n"
                           "        The pack in --pack is the AVX-512 Q2_0 pack\n", encoding="utf-8")
            cfg = Path(d) / "run.json"
            cfg.write_text(json.dumps({"log": str(log)}), encoding="utf-8")
            text, nxt = R.explain_engine_exit({"state": "failed", "error": "server_exited", "engine_exit_code": 1}, cfg)
            self.assertIn("cannot run the expert kernel", text)
            self.assertIn("strata setup", nxt)
            self.assertIn("AVX2", nxt)


if __name__ == "__main__":
    unittest.main()
