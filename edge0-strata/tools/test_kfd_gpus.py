"""A PC with only the amdgpu driver (no rocm-smi, no setup module) still counts every AMD card, from a fake sysfs.

    python -m unittest tools.test_kfd_gpus
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_mcp as M  # noqa: E402


def node(root: Path, i: int, ver: int, simd: int, minor: int, vram: int = 0, name: str | None = None):
    n = root / "class/kfd/kfd/topology/nodes" / str(i)
    n.mkdir(parents=True)
    (n / "properties").write_text(f"cpu_cores_count {0 if simd else 12}\nsimd_count {simd}\ngfx_target_version {ver}\ndrm_render_minor {minor}\n")
    if simd:
        d = root / f"class/drm/renderD{minor}/device"
        d.mkdir(parents=True)
        (d / "mem_info_vram_total").write_text(str(vram))
        if name:
            (d / "product_name").write_text(name + "\n")


class KfdGpus(unittest.TestCase):
    def test_four_cards_and_the_cpu_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node(root, 0, 0, 0, 0)
            for k in range(4):
                node(root, k + 1, 120001, 128, 128 + k, 32 << 30, "AMD Radeon AI PRO R9700")
            gpus = M.kfd_gpus(tmp)
        self.assertEqual([g["index"] for g in gpus], [0, 1, 2, 3])
        self.assertTrue(all(g["usable"] and g["arch"] == "gfx1201" and g["vram_gb"] == 32.0 for g in gpus))

    def test_an_unlisted_architecture_is_named_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            node(Path(tmp), 0, 90010, 60, 128, 8 << 30)
            (g,) = M.kfd_gpus(tmp)
        self.assertFalse(g["usable"])
        self.assertIn("gfx90a", g["problem"])

    def test_no_amd_driver_is_an_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(M.kfd_gpus(tmp), [])


if __name__ == "__main__":
    unittest.main()
