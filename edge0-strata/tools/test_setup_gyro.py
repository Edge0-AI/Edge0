"""The Gyro entries of setup's catalog (agentionai/Qwen3.8-Flash-Next-Gyro-GGUF): sizes, pins, the VRAM + RAM memory rule,
never a default, and the MCP server's recommendation naming them only where nothing else fits.  Offline."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402


def hardware(ram, vram):
    return {"ram_gb": ram, "gpus": [{"usable": True, "vram_gb": vram, "vendor": "nvidia", "index": 0, "name": "RTX"}],
            "cpu": {"avx2": True}}


class GyroCatalog(unittest.TestCase):
    def test_the_two_sizes_and_their_numbers(self):
        self.assertEqual(setup.FAMILIES["gyro"]["file"].format(q="Gyro-S"), "Qwen3.8-Flash-Next-Gyro-S.gguf")
        for model, name, download, memory in (("Gyro-S", "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", 58.5, 27.6),
                                              ("Gyro-M", "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", 92.0, 35.0)):
            d = setup.MODELS[model]
            self.assertEqual(setup.model_file(setup.FAMILIES["gyro"], model, 1), name)
            self.assertEqual((d["download_gb"], d["ram_gb"]), (download, memory))
            size, sha = setup.GYRO_FILES[name]
            self.assertAlmostEqual(size / 1e9, download, delta=0.05)
            self.assertEqual(len(sha), 64)
            self.assertEqual(setup.model_shards(setup.FAMILIES["gyro"], model), 1)
            self.assertIs(d.get("vision"), True)
            self.assertTrue(d["mem_total"])
            self.assertEqual(d["families"], ("gyro",))

    def test_pinned_revision_and_license(self):
        repo = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
        self.assertEqual(len(setup.HF_REVISIONS[repo]), 40)
        self.assertIn(setup.HF_REVISIONS[repo], setup.FAMILIES["gyro"]["hf"])
        self.assertIn("Qwen Community License 1.0", setup.FAMILIES["gyro"]["license"])

    def test_never_the_default_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertNotIn("gyro", setup.RECOMMENDED_SIZES)

    def test_memory_is_vram_plus_ram(self):
        self.assertEqual(setup.mem_have("Gyro-S", 16.0, 12.0), 28.0)
        self.assertEqual(setup.mem_have("Q2_0", 16.0, 12.0), 16.0)
        names = ["Gyro-S", "Gyro-M"]
        self.assertEqual(setup.recommended_size("gyro", names, 24.0, 12.0), "Gyro-M")     # 36 GB
        self.assertEqual(setup.recommended_size("gyro", names, 16.0, 12.0), "Gyro-S")     # 28 GB
        self.assertEqual(setup.recommended_size("gyro", names, 8.0, 8.0), "Gyro-S")       # nothing fits: the first

    def test_mcp_table_matches(self):
        for m in ("Gyro-S", "Gyro-M"):
            for k in ("download_gb", "ram_gb", "arena_gb", "mem_total", "vision", "engine"):
                self.assertEqual(setup.MODELS[m][k], M.FALLBACK_MODELS[m][k], (m, k))
        self.assertEqual(setup.FAMILIES["gyro"]["tag"], M.FALLBACK_FAMILIES["gyro"]["tag"])

    def test_mcp_recommends_gyro_only_where_nothing_else_fits(self):
        s = M.Strata(ROOT)
        for ram, vram, family in ((64.0, 12.0, "qwen"), (48.0, 12.0, "qwen"), (32.0, 12.0, "swift"), (28.0, 8.0, "coder")):
            self.assertEqual(s.recommend(hardware(ram, vram))["family"], family, (ram, vram))
        got = s.recommend(hardware(20.0, 12.0))                  # 32 GB in total, the Coder's 32 GB of RAM missing
        self.assertEqual((got["family"], got["model"]), ("gyro", "Gyro-S"))
        got = s.recommend(hardware(28.0, 16.0))                  # 44 GB in total, 39 after the 5 GB for the system (#3425): the owner's 40 GB tier, Swift 1.5's IQ2_XS
        self.assertEqual((got["family"], got["model"]), ("swift", "IQ2_XS"))
        self.assertIn("Gyro-M", got["why"])                      # named as the other choice (NVIDIA, AMD and Intel Arc)
        intel = hardware(28.0, 16.0)
        intel["gpus"] = [{**g, "vendor": "intel"} for g in intel["gpus"]]
        self.assertIn("Gyro-M", s.recommend(intel)["why"])       # the SYCL engine has the trellis kernels (0.2.2)
        self.assertIsNone(s.recommend(hardware(12.0, 8.0))["family"])
        self.assertIsNone(s.recommend(hardware(16.0, 11.0))["family"])   # 27 GB: below Gyro-S's 27.6 + room

    def test_mcp_table_verdict_counts_vram(self):
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table(hardware(16.0, 16.0)) if f["family"] == "gyro")
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual(by["Gyro-S"]["on_this_pc"], "fits")
        self.assertEqual(by["Gyro-M"]["on_this_pc"], "tight")
        self.assertTrue(by["Gyro-S"]["images"])

    def test_mcp_table_gives_ram_plus_vram_without_the_ngram_table(self):
        """The Models page shows what a version needs of RAM + VRAM together, the n-gram table (on the SSD) not counted."""
        s = M.Strata(ROOT)
        table = s.model_table(hardware(64.0, 12.0))
        gyro = {x["model"]: x for x in next(f for f in table if f["family"] == "gyro")["sizes"]}
        self.assertEqual(gyro["Gyro-S"]["memory_gb"], 27.6)           # stated in total (mem_total), not the 58.5 GB download
        self.assertTrue(gyro["Gyro-S"]["mem_total"])
        swift = {x["model"]: x for x in next(f for f in table if f["family"] == "swift")["sizes"]}
        self.assertEqual(swift["IQ2_XS"]["memory_gb"], 36.5)          # 35.5 GB of experts + ~1 GB dense
        self.assertFalse(swift["IQ2_XS"]["mem_total"])


class GyroEngineVersionTest(unittest.TestCase):
    """Gyro's expert files are native_experts.txt v5 (Hadamard-folded). The released 0.1.45 engine reads only up to v4, so the
    minimum is 0.1.45.1, the first engine package built after the v5 reader (APR port, f47b0385f) was merged."""

    def test_gyro_needs_an_engine_that_reads_v5_files(self):
        self.assertEqual(setup.GYRO_ENGINE, (0, 1, 45, 1))
        self.assertGreater(setup.GYRO_ENGINE, (0, 1, 45))            # the released 0.1.45 is not enough
        sizes = [m for m in setup.MODELS.values() if "gyro" in str(m.get("file", "")).lower()]
        self.assertEqual(len(sizes), 2)
        self.assertTrue(all(m["engine"] == (0, 1, 45, 1) for m in sizes))

    def test_every_catalog_agrees(self):
        for m in ("Gyro-S", "Gyro-M"):
            self.assertEqual(setup.MODELS[m]["engine"], M.FALLBACK_MODELS[m]["engine"], m)

    def test_the_catalog_tells_the_app_the_minimum(self):
        table = M.Strata(ROOT).model_table(hardware(64.0, 12.0))
        gyro = {x["model"]: x for x in next(f for f in table if f["family"] == "gyro")["sizes"]}
        self.assertEqual({k: v["engine_min"] for k, v in gyro.items()}, {"Gyro-S": "0.1.45.1", "Gyro-M": "0.1.45.1"})
        qwen = {x["model"]: x for x in next(f for f in table if f["family"] == "qwen")["sizes"]}
        self.assertIsNone(qwen["Q2_0"]["engine_min"])                # a size with no minimum of its own says so

    def test_the_pack_reader_is_in_the_source_this_minimum_describes(self):
        text = (ROOT / "src" / "kernels" / "cpu" / "expert_layout.cpp").read_text(encoding="utf-8")
        self.assertIn("native_experts.txt v5 declares Hadamard-folded experts", text)


if __name__ == "__main__":
    unittest.main()
