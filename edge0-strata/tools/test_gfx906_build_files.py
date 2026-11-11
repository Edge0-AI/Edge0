"""#3172 #3271: the gfx906 (HIP wave64 compat) build stays buildable. The real check is a compile for gfx906 (done on a box with ROCm: the
`strata` target links); these cover what a text check can: the pieces of CMakeLists.txt a 0.1.44+ gfx906 build was missing."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8").replace("\r\n", "\n")


class Gfx906CMake(unittest.TestCase):
    def test_qwen35_is_not_built_for_gfx906(self):
        # q36_engine.cu calls cublasSgemm / cublasGemmStridedBatchedEx, which the wave64 compat layer does not map; main() guards every use
        # with STRATA_HAVE_QWEN35, so leaving the library out gives a gfx906 build that links
        block = re.search(r"if\(NOT STRATA_HIP_GFX906\)\n  add_library\(strata_qwen35.*?target_compile_definitions\(strata PRIVATE STRATA_HAVE_QWEN35=1\)\n  endif\(\)", TEXT, re.S)
        self.assertIsNotNone(block)
        self.assertIn("add_executable(strata-q36 ", block.group(0))
        self.assertIn("target_link_libraries(strata PRIVATE strata_qwen35)", block.group(0))

    def test_every_use_of_qwen35_in_the_engine_is_guarded(self):
        text = (ROOT / "src" / "program" / "generate.cpp").read_text(encoding="utf-8").replace("\r\n", "\n")
        for match in re.finditer(r"qwen35::|strata/qwen35/engine\.hpp", text):
            before = text[:match.start()]
            self.assertGreater(before.rfind("#if defined(STRATA_HAVE_QWEN35)"), before.rfind("#endif"), match.group(0))

    def test_the_w4a4_target_gets_the_hip_definition(self):
        self.assertRegex(TEXT, r"if\(TARGET strata_mmq_w4a4\)\n\s+target_compile_definitions\(strata_mmq_w4a4 PRIVATE GGML_USE_HIP\)")

    def test_overlay_sources_and_the_f16_stub_are_in_the_gfx906_build(self):
        tail = TEXT[TEXT.index("# ---- HIP gfx906: every CUDA source"):]
        self.assertIn("${_strata_tq_overlay_dir}/quantize.cu", tail)
        self.assertIn("mmq-instance-${_t}.cu", tail)
        self.assertIn("target_sources(strata_mmq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_f16_kq.cu)", tail)
        self.assertLess(tail.index("moe_f16_kq.cu"), tail.index("set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)"))


if __name__ == "__main__":
    unittest.main()
