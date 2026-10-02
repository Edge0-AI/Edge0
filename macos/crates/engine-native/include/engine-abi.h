/*
 * C ABI between the daemon and the inference engine.
 *
 * Opaque handles; extensible structs start with `size` (caller fills sizeof).
 * Compile-time E0_ABI_VERSION must match e0_abi_version() or load is refused.
 * JSON outputs use size-query then write: buf == NULL returns needed length
 * (excluding NUL); a short buf returns E0_ERR_BUFFER without truncation.
 * Only pointers, integers, and UTF-8 JSON cross the boundary.
 */
#ifndef EDGE0_ENGINE_ABI_H
#define EDGE0_ENGINE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Semantic changes bump this; appending fields is a size-compatible addition. */
#define E0_ABI_VERSION 2u

/* E0Engine binds weights and a Metal device; E0Tokenizer does not load weights. */
typedef struct E0Engine    E0Engine;
typedef struct E0Stream    E0Stream;
typedef struct E0Tokenizer E0Tokenizer;

/*
 * Negative values are errors. E0_WOULD_BLOCK / E0_NOTIFIED are for e0_next
 * poll/wait; the wait strategy (fd vs callback) is the daemon's concern.
 */
typedef enum E0Status {
  E0_OK            = 0,
  E0_NOTIFIED      = 1,  /* e0_next produced a token */
  E0_WOULD_BLOCK   = 2,  /* no token yet; not an error */
  E0_ERR_INVALID   = -1, /* bad argument, handle, or state */
  E0_ERR_BUFFER    = -2, /* output buffer too small (size-query then write) */
  E0_ERR_MODEL     = -3, /* weight / layout / hash problem */
  E0_ERR_MEM       = -4, /* load or activation out of memory */
  E0_ERR_GPU       = -5, /* Metal device or command error */
  E0_ERR_INTERNAL  = -6, /* other; details_json has a readable message */
  E0_ERR_CANCELLED = -7  /* stream was stopped by e0_cancel */
} E0Status;

typedef struct E0TokenOut {
  uint32_t size;    /* caller fills sizeof(E0TokenOut) */
  int32_t  token;   /* token id; undefined when FINISHED is set */
  uint32_t flags;
} E0TokenOut;

#define E0_TOKEN_FINISHED 0x1u /* stream ended (natural EOS or cancel) */

uint32_t e0_abi_version(void);
const char *e0_engine_version(void);

/* Engine surface: GB-scale, resident for the worker / keep_alive lifetime. */

/* config_json: {"max_layers":..., "hot_n":..., ...}; keys may be added, not redefined. */
E0Engine *e0_engine_create(const char *tier_id,
                           const char *model_dir,
                           const char *config_json);

/* prompt_tokens are tokenized ids; params_json holds sampling and stop criteria.
 * NULL on failure; details via e0_last_error. */
E0Stream *e0_generate_begin(E0Engine *h,
                            const int32_t *prompt_tokens,
                            uint32_t n_tokens,
                            const char *params_json);

/* Non-blocking next token: E0_NOTIFIED (out valid) / E0_WOULD_BLOCK / E0_ERR_*.
 * End of stream sets E0_TOKEN_FINISHED. */
E0Status e0_next(E0Engine *h, E0Stream *stream, E0TokenOut *out);

/* Cancel one generation. Idempotent. A cancelled stream ends with E0_TOKEN_FINISHED. */
E0Status e0_cancel(E0Engine *h, E0Stream *stream);

/*
 * Runtime stats as JSON (size-query then write). Key names are frozen:
 * adding a key must not change the meaning of existing keys.
 * Required keys:
 *   "route_source_by_layer" — who chose each layer's expert set:
 *        {"predicted"|"routed"|"replayed", ...}; "replayed" only when the
 *        layer's set comes from a reference-sample replay;
 *   "loaded_experts" / "cache_hits" / "cache_misses";
 *   "memory" — object with three orthogonal fields:
 *        "resident_peak_bytes"   peak process resident_size, including
 *            file-backed weight pages;
 *        "expert_cache_heap_bytes"  explicit heap bytes of the streaming
 *            expert cache (numerator of the resident-budget check);
 *        "file_backed_bytes"     mmap-backed weight pages reclaimable by the OS.
 *        Tier and doctor thresholds use
 *        committed_resident = resident_peak − file_backed;
 *   "prefix_cache" — independent of the expert-weight cache: enabled/budget_bytes,
 *        hits/misses/cached_tokens, entries/bytes/evictions, and
 *        last_cached_tokens/last_prefill_tokens/last_reason for the last request.
 */
E0Status e0_stats(E0Engine *h, char *buf, size_t buf_len, size_t *written);

/*
 * Per-request reset: drop the active stream, prerouter cross-token state
 * (previous-token exec one-hot, linear_init, stager double-buffer), request
 * KV/conv/delta, and the prefetch queue. Keep the bounded prefix_cache snapshot
 * for the current model identity. The daemon calls this before each
 * e0_generate_begin. Reset belongs on the ABI, not in policy. Snapshots from a
 * different model identity, ABI generation, or worker must not be reused.
 */
E0Status e0_turn_reset(E0Engine *h);

/* Last error as readable JSON (size-query then write). */
E0Status e0_last_error(E0Engine *h, char *buf, size_t buf_len, size_t *written);

void e0_engine_destroy(E0Engine *h);

/* Tokenizer surface: MB-scale, no weights, independent of the worker lifetime.
 * Must not take an engine handle — UI token counts (real ids, not char estimates)
 * must not pin a GB-scale worker. Reads tokenizer.json, chat_template.jinja,
 * and config.json under model_dir. */

E0Tokenizer *e0_tok_create(const char *model_dir);

/* text_json: {"text": "..."} or {"texts": [...]}; output {"tokens":[[...]]}. */
E0Status e0_tok_tokenize(E0Tokenizer *t, const char *text_json,
                         char *buf, size_t buf_len, size_t *written);

/* messages_json: OpenAI messages array (or an object with messages and template
 * variables); output is the rendered prompt text and token count. */
E0Status e0_tok_apply_template(E0Tokenizer *t, const char *messages_json,
                               char *buf, size_t buf_len, size_t *written);

/*
 * tokens_json: {"tokens": [id, ...]}; output {"text": "..."} (size-query then write).
 * Decodes a complete token sequence. Incremental streaming render is the
 * caller's job (decode the full sequence, diff against the previous text, and
 * wait for the next token when a multi-byte sequence is split). Both the real
 * engine and the replay engine must export this symbol (missing symbol = load fail).
 */
E0Status e0_tok_decode(E0Tokenizer *t, const char *tokens_json,
                       char *buf, size_t buf_len, size_t *written);

void e0_tok_destroy(E0Tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif /* EDGE0_ENGINE_ABI_H */
