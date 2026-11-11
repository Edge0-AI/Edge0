// src/prefill/kva_test.cpp - the KVA switch, the automatic tail and the reasons a read is turned away (no GPU needed).
#include "strata/kernels/window_start.hpp"
#include "strata/prefill/kva.hpp"
#include "strata/prefill/kva_floor.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::prefill;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
static void clear_all() {
    for (const char* k : {"STRATA_KVA", "STRATA_KVA_SPLIT", "STRATA_KVA_SUFFIX", "STRATA_KVA_MIN_PROMPT", "STRATA_KVA_PARITY",
                          "STRATA_KVA_CAPTURE", "STRATA_KVA_CAPTURE_FP32", "STRATA_KVA_ORACLE", "STRATA_KVA_ORACLE_FP32",
                          "STRATA_KVA_PROJECTOR", "STRATA_KVA_MODE", "STRATA_KVA_ONLY", "STRATA_KVA_DRAFT_FLOOR", "STRATA_KVA_SKIP", "STRATA_KVA_ANY_MODEL"})
        unset(k);
}
static bool is(const char* got, const char* want) { return got != nullptr && std::strcmp(got, want) == 0; }

int main() {
    // ---- the switch: off unless STRATA_KVA=1 (0.2.7 ships KVA opt-in; unset is the exact 0.2.6 engine)
    clear_all();
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(!c.on && !c.explicit_on);                                      // unset: off
        CHECK(c.split == 36 && c.min_prompt == 4096 && c.suffix_env == 0 && kKvaSplitDefault == 36);
        CHECK(c.mode == KvaConfig::Mode::Identity && c.only == KvaConfig::Only::All && !c.capture);
        KvaEngine eng;
        CHECK(!c.wants_cell_floor(eng, true));
        CHECK(is(c.decline(eng, 0, 30000, 0, -1, 48), "off (STRATA_KVA is not 1)"));   // a 30K cold prompt is read exactly
    }
    put("STRATA_KVA", "0");
    { const KvaConfig c = kva_config_from_env(); CHECK(!c.on && !c.explicit_on); }
    put("STRATA_KVA", "1");
    { const KvaConfig c = kva_config_from_env(); CHECK(c.on && c.explicit_on); }
    put("STRATA_KVA", "");   // an empty value is "not set"
    { const KvaConfig c = kva_config_from_env(); CHECK(!c.on && !c.explicit_on); }
    clear_all();
    put("STRATA_KVA", "1");   // everything below runs with KVA switched on

    // ---- switched on: split 36 and a 2048-token tail (the configuration measured), or STRATA_KVA_SPLIT / STRATA_KVA_SUFFIX
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.on && c.split == 36 && c.effective_suffix() == kKvaSuffixDefault && kKvaSuffixDefault == 2048);
        put("STRATA_KVA_SPLIT", "30");
        CHECK(kva_config_from_env().split == 30);
        unset("STRATA_KVA_SPLIT");
        put("STRATA_KVA_SUFFIX", "3000");
        const KvaConfig e = kva_config_from_env();
        CHECK(e.effective_suffix() == 3000);
        unset("STRATA_KVA_SUFFIX");
    }

    // ---- who is turned away, and why (the first reason that holds is the one named)
    {
        const KvaConfig c = kva_config_from_env();
        KvaEngine eng;
        bool routine = false;
        // a cold 14.6K prompt on one device: applies (14600 > 2048 + 64)
        CHECK(c.decline(eng, 0, 14600, 0, -1, 48) == nullptr);
        CHECK(c.decline(eng, 0, 14600, 0, 48, 48) == nullptr);   // le == n_layers is the whole trunk too
        // a continuation (turn 2 of a chat, a resumed prompt): routine
        CHECK(is(c.decline(eng, 9000, 14600, 0, -1, 48, false, false, &routine), "a continuation read (pos0 != 0)") && routine);
        // short prompts and a prompt the tail would swallow: routine
        CHECK(is(c.decline(eng, 0, 3000, 0, -1, 48, false, false, &routine), "the prompt is shorter than STRATA_KVA_MIN_PROMPT") && routine);
        CHECK(c.decline(eng, 0, 4096, 0, -1, 48) == nullptr);   // 4096 is the first prompt that is long enough
        put("STRATA_KVA_SUFFIX", "5000");   // a longer tail swallows a prompt of 5000 and not one of 5065
        { const KvaConfig s = kva_config_from_env();
          CHECK(is(s.decline(eng, 0, 5000, 0, -1, 48, false, false, &routine), "the suffix leaves too few positions to approximate") && routine);
          CHECK(s.decline(eng, 0, 5065, 0, -1, 48) == nullptr); }
        unset("STRATA_KVA_SUFFIX");
        // not routine: the user may want to know
        CHECK(is(c.decline(eng, 0, 14600, 24, -1, 48, false, false, &routine), "a layer split (not the whole trunk)") && !routine);
        CHECK(is(c.decline(eng, 0, 14600, 0, 24, 48), "a layer split (not the whole trunk)"));
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 48, true), "a prompt with pictures"));
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 48, false, true), "the DFlash2 drafter reads the late layers' hidden states"));
        // the engine rules the whole process out (batch slots, a layer split, a peer device)
        KvaEngine blocked = eng;
        blocked.blocked = "batch slots (--batch)";
        CHECK(is(c.decline(blocked, 0, 14600, 0, -1, 48, false, false, &routine), "batch slots (--batch)") && !routine);
        // a model whose shape KVA was not measured on (not 48 layers / 512 experts) is turned away, not warned about;
        // the routine reasons (a short prompt, a continuation) still come first
        CHECK(is(c.decline(eng, 0, 14600, 0, -1, 40, false, false, &routine, true), kKvaShapeReason) && !routine);
        CHECK(is(c.decline(eng, 0, 3000, 0, -1, 40, false, false, &routine, true), "the prompt is shorter than STRATA_KVA_MIN_PROMPT") && routine);
        CHECK(c.decline(eng, 0, 14600, 0, -1, 48, false, false, &routine, false) == nullptr);   // the measured shape applies
        put("STRATA_KVA_ANY_MODEL", "1");   // the research override allows it (prefill.cpp then warns once)
        { const KvaConfig a = kva_config_from_env(); CHECK(a.any_model && a.decline(eng, 0, 14600, 0, -1, 40, false, false, nullptr, true) == nullptr); }
        unset("STRATA_KVA_ANY_MODEL");
        CHECK(!kva_config_from_env().any_model);
        // capture wins over the rest
        put("STRATA_KVA_CAPTURE", "/tmp/kva-cap");
        { const KvaConfig o = kva_config_from_env(); CHECK(o.capture && is(o.decline(eng, 0, 14600, 0, -1, 48), "capture mode")); }
        unset("STRATA_KVA_CAPTURE");
    }
    clear_all();

    // ---- the research modes still parse
    put("STRATA_KVA_ORACLE", "/tmp/o");
    put("STRATA_KVA_ORACLE_FP32", "1");
    put("STRATA_KVA_ONLY", "qsa");
    put("STRATA_KVA_SPLIT", "30");
    put("STRATA_KVA_MIN_PROMPT", "1024");
    {
        const KvaConfig c = kva_config_from_env();
        CHECK(c.mode == KvaConfig::Mode::Oracle && c.oracle_fp32 && c.oracle_dir == "/tmp/o");
        CHECK(c.only == KvaConfig::Only::Qsa && c.split == 30 && c.min_prompt == 1024);
    }
    clear_all();

    // ---- STRATA_KVA_SKIP: the default is the state-only form; "moe" runs the attention halves and skips the MoE half
    {
        CHECK(kva_config_from_env().skip == KvaConfig::Skip::StateOnly);
        put("STRATA_KVA_SKIP", "moe");
        CHECK(kva_config_from_env().skip == KvaConfig::Skip::Moe);
        put("STRATA_KVA_SKIP", "state");
        CHECK(kva_config_from_env().skip == KvaConfig::Skip::StateOnly);
        put("STRATA_KVA_SKIP", "bogus");
        CHECK(kva_config_from_env().skip == KvaConfig::Skip::StateOnly);
        unset("STRATA_KVA_SKIP");
    }
    // ---- "not applied => identical to STRATA_KVA=0", the parts that need no GPU
    // (1) by default the draft layer is 0.2.6's own code; the cell floor is a research option, wired only when asked
    {
        KvaEngine eng;
        put("STRATA_KVA", "1");
        const KvaConfig on = kva_config_from_env();
        CHECK(on.on && !on.draft_floor && !on.wants_cell_floor(eng, true));
        put("STRATA_KVA_DRAFT_FLOOR", "1");
        const KvaConfig fl = kva_config_from_env();
        CHECK(fl.draft_floor && fl.wants_cell_floor(eng, true));
        CHECK(!fl.wants_cell_floor(eng, false));          // no draft layer: nothing to wire
        KvaEngine blocked;
        blocked.blocked = "batch slots (--batch)";
        CHECK(!fl.wants_cell_floor(blocked, true));       // ruled out for the process: 0.2.6's own attention code
        put("STRATA_KVA", "0");
        const KvaConfig off = kva_config_from_env();
        CHECK(!off.wants_cell_floor(eng, true));          // STRATA_KVA=0: untouched
        clear_all();
    }
    // (2) with a floor of 0 the window is exactly the plain window, for every length and every window
    {
        using strata::kernels::window_start;
        bool same = true;
        for (int window : {1, 2, 64, 2048, 32768, 0x7fffffff})
            for (int n_kv : {0, 1, 2, 63, 64, 65, 2047, 2048, 2049, 4096, 14000, 32768, 32769, 100000}) {
                const int plain = n_kv > window ? n_kv - window : 0;   // what window_ids always selected
                if (window_start(n_kv, window, 0) != plain) same = false;
            }
        CHECK(same);
        CHECK(window_start(14000, 32768, 11264) == 11264);   // a floor inside the window raises the start
        CHECK(window_start(14000, 2048, 11264) == 11952);    // a floor below the window changes nothing
        CHECK(window_start(14000, 32768, 20000) == 13999);   // a floor past the end keeps one cell
        CHECK(window_start(14000, 32768, 14000) == 13999);
        CHECK(window_start(1, 32768, 5) == 0);
        CHECK(window_start(0, 32768, 5) == 0);
    }
    // (3) the floor follows the session's tokens: a state no KVA read produced has floor 0
    {
        KvaFloorRegistry reg;
        std::vector<int64_t> doc;
        for (int i = 0; i < 20000; ++i) doc.push_back(1000 + (i * 7919) % 5000);
        CHECK(reg.floor_for(doc.data(), (int64_t) doc.size()) == 0);        // nothing remembered
        reg.note(doc.data(), 11264);                                         // a KVA read approximated [0, 11264)
        CHECK(reg.floor_for(doc.data(), 20000) == 11264);                    // the live session / a later turn of it
        CHECK(reg.floor_for(doc.data(), 5000) == 5000);                      // a checkpoint inside the approximated part
        std::vector<int64_t> other(doc);
        for (int i = 0; i < 20000; ++i) other[(size_t) i] += 1;             // a different prompt entirely
        CHECK(reg.floor_for(other.data(), 20000) == 0);                      // short chats, other documents: floor 0
        std::vector<int64_t> shared(doc.begin(), doc.begin() + 3000);        // a new chat sharing the first 3000 tokens
        for (int i = 3000; i < 9000; ++i) shared.push_back(900000 + i);
        CHECK(reg.floor_for(shared.data(), 9000) == 3000);
        reg.note(doc.data(), 11264);                                         // the same read again: no second entry
        CHECK(reg.size() == 1);
        KvaFloorRegistry small(2);
        for (int k = 0; k < 5; ++k) { std::vector<int64_t> d(100, k); small.note(d.data(), 100); }
        CHECK(small.size() == 2);                                            // bounded
        reg.clear();
        CHECK(reg.floor_for(doc.data(), 20000) == 0);
    }

    if (fails == 0) std::printf("kva_test: ok\n");
    return fails == 0 ? 0 : 1;
}
