// src/artifact/gguf_reader.cpp - the `strata-gguf` CLI.  The reader itself is header-only in
// include/strata/artifact/gguf_reader.hpp; this file is just the entry point, so it cannot drift
// from the library it exercises.
#include "strata/artifact/gguf_reader.hpp"

#ifndef STRATA_GGUF_MAIN_DISABLED
int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: gguf_reader <file.gguf> [--check]\n");
        return 2;
    }
    const bool check = (argc > 2 && std::string(argv[2]) == "--check");
    try {
        strata::GgufFile g(argv[1]);
        std::printf("%s\n", argv[1]);
        std::printf("  version %u   tensors %zu   metadata %zu   data_start %llu   size %llu\n", g.version(),
                    g.tensors().size(), g.metadata().size(), (unsigned long long)g.data_start(),
                    (unsigned long long)g.file_size());

        if (check) {
            // Independent expectations, all established by earlier phases.
            size_t n_blk = 0;
            for (const auto& t : g.tensors())
                if (t.name.rfind("blk.", 0) == 0) ++n_blk;
            std::printf("  blk.* tensors %zu   globals %zu\n", n_blk, g.tensors().size() - n_blk);
            std::printf("  Q2_0 %llu   BF16 %llu   F32 %llu\n", (unsigned long long)g.count_type("Q2_0"),
                        (unsigned long long)g.count_type("BF16"), (unsigned long long)g.count_type("F32"));

            // Every tensor's byte size must land inside the gap to the next tensor's offset, since
            // GGUF aligns every tensor. That is the same bracket test tools/verify_q2_0_geometry.py
            // uses, and it validates block_geometry() against the file rather than asserting it.
            std::vector<const strata::TensorInfo*> ordered;
            for (const auto& t : g.tensors()) ordered.push_back(&t);
            std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a->offset < b->offset; });
            size_t bad = 0, unknown = 0;
            for (size_t i = 0; i < ordered.size(); ++i) {
                const auto* t = ordered[i];
                int el = 0, by = 0;
                if (!strata::block_geometry(t->type, el, by)) {
                    ++unknown;
                    continue;
                }
                if (t->elements() % (uint64_t)el) {
                    ++bad;
                    continue;
                }
                const uint64_t nb = t->elements() / (uint64_t)el * (uint64_t)by;
                // The bracket is on the GAP to the next tensor's offset, not on that offset itself -
                // offsets are relative to data_start, so the gap is a difference. Comparing against
                // the raw offset flagged almost every tensor (213/214) while the same test in Python
                // (tools/verify_q2_0_geometry.py) passed.
                const uint64_t end = (i + 1 < ordered.size()) ? ordered[i + 1]->offset - t->offset
                                                              : g.file_size() - g.data_start() - t->offset;
                if (!(nb <= end && end < nb + 32)) ++bad;
            }
            std::printf("  geometry bracket: %zu out of range, %zu types unknown\n", bad, unknown);
            const std::string err = strata::check_architecture(g);
            std::printf("  architecture guard: %s\n", err.empty() ? "PASS" : ("FAIL - " + err).c_str());
            if (bad) return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}
#endif // STRATA_GGUF_MAIN_DISABLED
