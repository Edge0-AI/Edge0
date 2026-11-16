// include/strata/qwen35/engine.hpp - the Qwen3.6-35B-A3B (GGUF architecture qwen35moe) engine path as a library:
// `strata` hands a qwen35moe model to it at the top of main() (the family switch), and strata-q36 is a thin wrapper
// around the same entry point.  See src/qwen35/q36_engine.cu.
#pragma once

namespace strata::qwen35 {

/// The engine's command line (strata's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --mtp N, --pool-workers, --tokens, --max-new, --no-token-graph; the
/// family's own: --expert-vram, --expert-slots, --hybrid-rows, --cpu-threads).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::qwen35
