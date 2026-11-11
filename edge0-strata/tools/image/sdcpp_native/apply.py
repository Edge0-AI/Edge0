"""tools/image/sdcpp_native/apply.py - put Strata's native Qwen-Image-2.1 DiT runner into a stable-diffusion.cpp
checkout at the pinned commit (tools/image/qimage.py SD_CPP_COMMIT).  Idempotent; the stock runner stays the
default: sd-cli uses the native one only with --model-args qwen_image_2_1_native=1 (serve/images.py adds it when the
config's "native" is true), and only in CUDA / HIP builds.

    python tools/image/sdcpp_native/apply.py third_party/stable-diffusion.cpp

Two new files (the kernels into ggml/src/ggml-cuda/, which ggml-cuda and ggml-hip compile by glob; the runner into
src/model/diffusion/) and two small edits (model_builders.cpp picks the runner; CMakeLists.txt defines SD_QI_NATIVE
for CUDA / HIP builds)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "// strata: native Qwen-Image-2.1 runner"

BUILDER_OLD = """            if (version == VERSION_QWEN_IMAGE_2_1) {
                result.diffusion = std::make_shared<Qwen::QwenImage21Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                             tensor_storage_map,
                                                                             "model.diffusion_model",
                                                                             weight_manager,
                                                                             sd_ctx_params->model_args);
"""
BUILDER_NEW = """            if (version == VERSION_QWEN_IMAGE_2_1) {
#ifdef SD_QI_NATIVE
                """ + MARK + """ (--model-args qwen_image_2_1_native=1)
                if (sd_ctx_params->model_args != nullptr &&
                    std::strstr(sd_ctx_params->model_args, "qwen_image_2_1_native=1") != nullptr &&
                    sd_ctx_params->diffusion_model_path != nullptr && sd_ctx_params->diffusion_model_path[0] != 0) {
                    result.diffusion = std::make_shared<Qwen::QwenImage21NativeRunner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                                       tensor_storage_map,
                                                                                       "model.diffusion_model",
                                                                                       weight_manager,
                                                                                       sd_ctx_params->model_args,
                                                                                       sd_ctx_params->diffusion_model_path);
                } else
#endif
                result.diffusion = std::make_shared<Qwen::QwenImage21Runner>(ctx.backends.runtime_backend(SDBackendModule::DIFFUSION),
                                                                             tensor_storage_map,
                                                                             "model.diffusion_model",
                                                                             weight_manager,
                                                                             sd_ctx_params->model_args);
"""
INCLUDE_OLD = '#include "model/diffusion/minit2i.hpp"\n'
INCLUDE_NEW = INCLUDE_OLD + "#ifdef SD_QI_NATIVE\n#include \"model/diffusion/qwen_image_2_1_native.hpp\"  " + MARK + "\n#endif\n"
CMAKE_OLD = "target_include_directories(${SD_LIB} PUBLIC . src include)\n"
CMAKE_NEW = ("if (SD_CUDA OR SD_HIPBLAS)  # " + MARK.lstrip("/ ") + "\n"
             "    target_compile_definitions(${SD_LIB} PRIVATE SD_QI_NATIVE)\nendif()\n" + CMAKE_OLD)


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise SystemExit(f"{path}: the expected code is not there (another stable-diffusion.cpp commit?)")
    path.write_text(text.replace(old, new), encoding="utf-8")


def apply(sd: Path) -> None:
    sd = Path(sd)
    shutil.copy2(HERE / "qi21-native.cu", sd / "ggml" / "src" / "ggml-cuda" / "qi21-native.cu")
    shutil.copy2(HERE / "qwen_image_2_1_native.hpp", sd / "src" / "model" / "diffusion" / "qwen_image_2_1_native.hpp")
    builders = sd / "src" / "pipeline" / "model_builders.cpp"
    edit(builders, INCLUDE_OLD, INCLUDE_NEW)
    edit(builders, BUILDER_OLD, BUILDER_NEW)
    edit(sd / "CMakeLists.txt", CMAKE_OLD, CMAKE_NEW)


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
    print("native Qwen-Image-2.1 runner applied")
