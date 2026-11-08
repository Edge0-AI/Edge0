#pragma once
#include <cstdint>

#define E0_PREF_API __attribute__((visibility("default")))
struct llama_model;
struct llama_context;

// E0_PREROUTER enables advisory heads. E0_PREFETCH=0 accounts only; =1
// issues POSIX advice. Neither mode changes the model's routing decisions.
E0_PREF_API bool edge0_pref_init(llama_model * model, llama_context * ctx,
                                int pool_mib, const char * gguf_path);
E0_PREF_API void edge0_pref_on_step(llama_context * ctx, int64_t last_row);
// Called before context/model destruction, including server sleep/reload.
E0_PREF_API void edge0_pref_shutdown();
