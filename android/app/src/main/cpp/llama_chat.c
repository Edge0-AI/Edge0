// llama_chat.c - native llama engine shell for the Edge0 chat app.
// Session model: static history of turns; every request renders the chat template
// over the append-only transcript, tokenizes it, and lets KV prefix reuse turn the
// full render into an incremental prefill. Sampling = classic token_data_array
// chain (greedy when temperature <= 0). nativeInit carries expert-pool tier config
// (the 35B tier requires E0_MMAP_NOPREFETCH + E0_NO_REPACK), nativeReset drops KV,
// and each generate returns a structured metrics line for the Kotlin layer.
#include <jni.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <android/log.h>
#include "llama.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "E0LLAMA", __VA_ARGS__)

static struct llama_model *g_model = NULL;
static struct llama_context *g_ctx = NULL;
static struct llama_adapter_lora *g_lora = NULL;

#define MAX_MSG 128
static llama_chat_message g_hist[MAX_MSG];
static char *g_strs[MAX_MSG * 2];
static int g_nmsg = 0;

#define MAX_TOKS 8192
static llama_token g_toks[MAX_TOKS];
static llama_token *g_msg_ids[MAX_MSG];      // assistant turns: stepped token ids (sidecar)
static int g_msg_nids[MAX_MSG];
static llama_token g_msg_eog[MAX_MSG];      // end-of-turn token for each history turn
static llama_token g_out_ids[MAX_TOKS];      // ids stepped this turn (EOS excluded)
static int g_out_n = 0;
static llama_token g_out_eog = -1;
static int g_nprocessed = 0;

static volatile int g_cancel = 0;
// pool counters forward decl (defined at the file tail; the metrics window inside nativeGenerate needs them)
extern void moe_pool_stats(long long * hits, long long * misses, long long * loads, long long * evicts,
                           long long * bytes_loaded, size_t * cap_bytes, size_t * used_bytes);
extern void moe_pool_extra(long long * stall_ns, long long * slots_used, long long * slots_total,
                           long long * pf_used, long long * pf_mispredict);

// Thinking control: the built-in BAILING2 renderer emits no think tags and ignores
// jinja variables - thinking is model-native. The real switch is the template's
// "detailed thinking on/off" instruction line in the system turn (training contract).
static int g_thinking = 0;
// The on direction rewrites in place with equal or shorter length, byte-matching the
// reference jinja render; returns the corrected render length.
// Both template families align with the reference (real-template) render:
//   8B Bailing V3: off = instruction line at the end of the system turn (the built-in
//     renderer self-injects it when no system turn exists); assistant heads get the open
//     tag for on, an open+close empty pair for off.
//   35B Qwen2/CHATML: off = empty pair after the generation head only - the instruction
//     line is a foreign signal for this family (verified against the embedded template).

static int32_t bailing_flip_on(char * r, int32_t len) {   // V3: off -> on, shrink 1 byte left
    const char * needle = "detailed thinking off";
    for (int32_t i = 0; i + 21 <= len; i++) {
        if (memcmp(r + i, needle, 21) == 0) {
            r[i + 19] = 'n';
            memmove(r + i + 20, r + i + 21, (size_t) (len - i - 20));
            return len - 1;
        }
    }
    return len;
}

// Reference-template evidence: with enable_thinking=false the jinja template emits NO
// instruction line - just the empty pair after the generation head. Carrying the
// "detailed thinking off" text into a fabricated system turn is a foreign signal for
// the qwen family (it invites the model to mimic reasoning prose in the body), so
// off-mode here is a pure erase (also catches residue from user-provided system text).
static int32_t qwen_pure_erase_directive(char * r, int32_t len) {
    char needle[24];
    memcpy(needle, "detailed thin", 13);
    memcpy(needle + 13, "king off", 9);
    needle[22] = 0;
    for (;;) {
        char * f = strstr(r, needle);
        if (!f) return (int32_t) strlen(r);
        int32_t st = (int32_t) (f - r), en = st + 22;
        if (st > 0 && r[st - 1] == '\n') st--;
        else if (r[en] == '\n') en++;
        memmove(r + st, r + en, (size_t) (len - en + 1));
        len -= (en - st);
    }
}

static int g_tpl_imstart = -1;

// Bailing fallback: when a custom system turn exists the built-in renderer skips the
// instruction line entirely and the off signal is lost. The V3 contract inlines the
// line at the end of the SYSTEM turn content, before its role_end. Insert there;
// direction from g_thinking; only acts when no "detailed thinking" text exists.
static void bail_ensure_directive(char * r, int32_t * plen, int32_t cap) {
    if (g_tpl_imstart == 1) return;
    if (strstr(r, "detailed thinking")) return;   // built-in already injected it, or the user wrote their own
    char * sy = strstr(r, "<role>SYSTEM</role>");
    if (!sy) return;
    char * re = strstr(sy + 19, "<|" "role_end" "|>");
    if (!re) return;
    char frag[32];
    int fl = snprintf(frag, sizeof(frag), "\ndetailed thinking %s", g_thinking ? "on" : "off");
    if (fl <= 0 || *plen + fl + 1 > cap) return;
    memmove(re + fl, re, (size_t) (*plen - (re - r) + 1));
    memcpy(re, frag, (size_t) fl);
    *plen += fl;
}

// Product decision for the 35B family: keep the template-conformant empty-pair form
// and do not seed the continuation; sampling-side forcing measured unreliable.

/* Literal of the built-in BAILING2 default head for history without a system turn:
   segment-wise rendering would inject it between segments - strip it to keep one SYSTEM. */
static const char FAKE_SYS_HEAD[] =
    "<" "role" ">" "SYSTEM" "<" "/" "role" ">" "detailed thinking off" "<|" "role_end" "|>";
static int32_t bail_strip_fake_head(char * r, int32_t len) {
    const int fl = (int) sizeof(FAKE_SYS_HEAD) - 1;
    if (len >= fl && memcmp(r, FAKE_SYS_HEAD, (size_t) fl) == 0) {
        memmove(r, r + fl, (size_t) (len - fl + 1));
        return len - fl;
    }
    return len;
}

// Golden-path head form (instruction line alone is a half-fix against the reference
// render): after every bare assistant head emit newline + (on ? open : open+close pair).
static void bailing_think_heads(char * r, int32_t * plen, int32_t cap) {
    const char * head = "<role>ASSISTANT</role>";
    const int hl = 22;
    static const char tso[] = "\n" "<" "th" "ink" ">";
    static const char tsof[] = "\n" "<" "th" "ink" ">" "</" "th" "ink" ">";
    const char * inj = g_thinking ? tso : tsof;
    const int il = (int) strlen(inj);   /* tso 8B / tsof 16B, self-adaptive */
    int32_t pos = 0;
    for (;;) {
        char * h = strstr(r + pos, head);
        if (!h) break;
        int32_t at = (int32_t) (h - r) + hl;
        // already annotated (tag follows the newline) - skip to prevent double injection
        if (memcmp(r + at, inj, (size_t) il) == 0) { pos = at + il; continue; }
        if (*plen + il + 1 > cap) return;
        memmove(r + at + il, r + at, (size_t) (*plen - at + 1));
        memcpy(r + at, inj, (size_t) il);
        *plen += il;
        pos = at + il;
    }
}

static void qwen_inject_empty_pair(char * r, char const * head, int32_t cap) {
    size_t l = strlen(r), hl = strlen(head);
    if (l == 0 || hl == 0 || l + 32 > (size_t) cap) return;
    static const char pair[] = "<th" "ink>\n\n</th" "ink>\n\n";
    /* Training form = reference template: exactly one newline after the assistant
       head, then the empty pair. The old append style added an extra newline (a
       blank-paragraph continuation cue) and the model answered with mimic'd
       "thinking process" prose - this overlay keeps it token-identical. */
    if (l >= hl && memcmp(r + l - hl, head, hl) == 0 && !strstr(r + l - hl, "<th" "ink>")) {
        memcpy(r + (l - 1 - hl), pair, sizeof(pair) - 1);   /* keep head+newline; overlay empty pair right after it */
        r[(l - 1 - hl) + sizeof(pair) - 1] = 0;                 /* the overlay straddles the old terminator: rewrite it */
    }
}


static char g_reply_tail[8192];      // tail 8KB of the last reply (cross-check for history sync; longer = TOOLONG)
static size_t g_rlen = 0;
static llama_token_data *g_td = NULL;
static int g_reset_next = 0;

// With byte-fallback vocabularies one token may be half a UTF-8 codepoint; feeding an
// illegal continuation byte to NewStringUTF aborts under CheckJNI. Emit only complete
// codepoints; carry the partial tail into the next piece.
static char g_pend[8];
static int g_pend_len = 0;

static void emit_valid_utf8(JNIEnv *env, jobject sink, jmethodID invoke,
                            const char *buf, int32_t len) {
    char comb[16];
    int n = g_pend_len + len;
    if (n > (int)sizeof(comb)) n = (int)sizeof(comb);
    memcpy(comb, g_pend, g_pend_len);
    memcpy(comb + g_pend_len, buf, n - g_pend_len);   // clamp every memory op to the truncated n, never the raw len
    int cut = n;
    for (int k = 1; k <= 3 && k <= n; k++) {
        unsigned char b = (unsigned char)comb[n - k];
        if ((b & 0xC0) != 0x80) {
            int expect = (b < 0x80) ? 1 : (b < 0xE0) ? 2 : (b < 0xF0) ? 3 : 4;
            if (k < expect) cut = n - k;
            break;
        }
    }
    g_pend_len = n - cut;
    if (g_pend_len) memcpy(g_pend, comb + cut, g_pend_len);
    if (cut > 0) {
        char safe[16]; memcpy(safe, comb, cut); safe[cut] = 0;
        jstring js = (*env)->NewStringUTF(env, safe);
        (*env)->CallObjectMethod(env, sink, invoke, js);
        (*env)->DeleteLocalRef(env, js);
    }
}

static void flush_pend(void) { g_pend_len = 0; }

static long long now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeInit(
        JNIEnv *env, jclass cls, jstring jmodel, jstring jlora, jstring jheads,
        jint n_ctx, jint pool_mb, jint blob_mb, jint pred, jint io_threads) {
    (void) cls;
    // idempotent self-clear: model/LoRA/pool registry/history from a previous session
    // must never bleed into a new load (an 8B adapter on a 35B graph aborts on shape asserts)
    extern void moe_pool_reset_all(void);
    if (g_model || g_ctx || g_lora) {
        if (g_ctx) { llama_free(g_ctx); g_ctx = NULL; }
        if (g_lora) { llama_adapter_lora_free(g_lora); g_lora = NULL; }
        if (g_model) { llama_model_free(g_model); g_model = NULL; }
        for (int i = 0; i < g_nmsg * 2; i++) { free(g_strs[i]); g_strs[i] = NULL; }
        for (int i = 0; i < g_nmsg; i++) { free(g_msg_ids[i]); g_msg_ids[i] = NULL; g_msg_nids[i] = 0; }
        g_nmsg = 0; g_nprocessed = 0;
    }
    moe_pool_reset_all();
    const char *model = (*env)->GetStringUTFChars(env, jmodel, NULL);
    const char *lora = (*env)->GetStringUTFChars(env, jlora, NULL);
    const char *heads = (*env)->GetStringUTFChars(env, jheads, NULL);
    char err[512];
    // pool env must be set before model load (moe_pool_init reads env at register time);
    // 8B passes 0/0 = zero perturbation.
    setenv("E0_MMAP_NOPREFETCH", "1", 1);   // verified freeze-avoidance on 35B; harmless for 8B
    setenv("E0_NO_REPACK", "1", 1);
    { char b[32];
      snprintf(b, sizeof(b), "%d", pool_mb);    setenv("E0_POOL_MB", b, 1);
      snprintf(b, sizeof(b), "%d", blob_mb);    setenv("E0_BLOB_MB", b, 1);
      snprintf(b, sizeof(b), "%d", io_threads); setenv("E0_IO_THREADS", b, 1);
      setenv("E0_MEM_FLOOR_MB", "2000", 1); }
    if (heads && heads[0]) setenv("E0_HEADS", heads, 1);
    if (blob_mb > 0) setenv("E0_KEEPWARM_MB", "512", 1);   // 35B tier: turn-end keepwarm refill (moot without a pool)
    if (pred) setenv("E0_PREROUTER", "1", 1); else unsetenv("E0_PREROUTER");
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(model, mp);
    if (!g_model) {
        snprintf(err, sizeof(err), "ERR load: %s", model);
        goto ret;
    }
    if (lora && lora[0]) {
        g_lora = llama_adapter_lora_init(g_model, lora);
        if (!g_lora) { snprintf(err, sizeof(err), "ERR lora init: %s", lora); goto ret; }
    }
    {
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t) n_ctx;
        cp.n_batch = 512;
        g_ctx = llama_init_from_model(g_model, cp);
        if (!g_ctx) { snprintf(err, sizeof(err), "ERR ctx init"); goto ret; }
        if (g_lora) {
        float scale = 1.0f;   // canonical form: file alpha 32 + explicit scale 1.0
            if (llama_set_adapters_lora(g_ctx, &g_lora, 1, &scale) != 0) {
                snprintf(err, sizeof(err), "ERR lora set"); goto ret;
            }
        }
    }
    g_reset_next = 1;
    snprintf(err, sizeof(err), "OK %s pool=%d blob=%d pred=%d", llama_version(), pool_mb, blob_mb, pred);
ret:
    (*env)->ReleaseStringUTFChars(env, jmodel, model);
    (*env)->ReleaseStringUTFChars(env, jlora, lora);
    (*env)->ReleaseStringUTFChars(env, jheads, heads);
    return (*env)->NewStringUTF(env, err);
}

JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeReset(
        JNIEnv *env, jclass cls) {
    (void) cls;
    // session reset: clear history and force full KV replay (thread switch / history fork)
    for (int i = 0; i < g_nmsg * 2; i++) { free(g_strs[i]); g_strs[i] = NULL; }
    for (int i = 0; i < g_nmsg; i++) { free(g_msg_ids[i]); g_msg_ids[i] = NULL; g_msg_nids[i] = 0; }
    g_nmsg = 0;
    g_reset_next = 1;
    return (*env)->NewStringUTF(env, "OK");
}

JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeAddRole(
        JNIEnv *env, jclass cls, jstring jrole, jstring jtext) {
    (void) cls;
    const char *role = (*env)->GetStringUTFChars(env, jrole, NULL);
    const char *text = (*env)->GetStringUTFChars(env, jtext, NULL);
    int full = g_nmsg >= MAX_MSG;
    if (!full) {
        char *r = strdup(role); char *t = strdup(text);
        g_strs[g_nmsg * 2] = r; g_strs[g_nmsg * 2 + 1] = t;
        g_hist[g_nmsg].role = r; g_hist[g_nmsg].content = t;
        g_nmsg++;
    }
    (*env)->ReleaseStringUTFChars(env, jrole, role);
    (*env)->ReleaseStringUTFChars(env, jtext, text);
    return (*env)->NewStringUTF(env, full ? "FULL" : "OK");
}

// assistant turn enters history via sidecar: ids big-endian 4B per element; text render is skipped (stitched in nativeGenerate)
JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeAddRoleIds(
        JNIEnv *env, jclass cls, jstring jrole, jbyteArray jids, jint eog) {
    (void) cls;
    const char *role = (*env)->GetStringUTFChars(env, jrole, NULL);
    const int nb = jids ? (*env)->GetArrayLength(env, jids) : 0;
    int full = (g_nmsg >= MAX_MSG) || (nb % 4) != 0;
    if (!full) {
        llama_token *ids = NULL;
        if (nb > 0) {
            jbyte *bd = (*env)->GetByteArrayElements(env, jids, NULL);
            ids = (llama_token *) malloc(sizeof(llama_token) * (nb / 4));
            for (int i = 0; i < nb / 4; i++) {
                unsigned char *q = (unsigned char *) (bd + i * 4);
                ids[i] = (llama_token) ((q[0] << 24) | (q[1] << 16) | (q[2] << 8) | q[3]);
            }
            (*env)->ReleaseByteArrayElements(env, jids, bd, JNI_ABORT);
        }
        char *r = strdup(role);
        g_strs[g_nmsg * 2] = r; g_strs[g_nmsg * 2 + 1] = NULL;
        g_hist[g_nmsg].role = r; g_hist[g_nmsg].content = "";
        g_msg_ids[g_nmsg] = ids; g_msg_nids[g_nmsg] = nb / 4; g_msg_eog[g_nmsg] = (llama_token) eog;
        g_nmsg++;
    }
    (*env)->ReleaseStringUTFChars(env, jrole, role);
    return (*env)->NewStringUTF(env, full ? "FULL" : "OK");
}

// ids stepped this turn (big-endian 4B) plus the end-of-turn token
JNIEXPORT jbyteArray JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeLastAssistantIds(JNIEnv *env, jclass cls) {
    (void) cls;
    jbyteArray out = (*env)->NewByteArray(env, g_out_n * 4);
    if (out && g_out_n > 0) {
        jbyte *bd = (*env)->GetByteArrayElements(env, out, NULL);
        for (int i = 0; i < g_out_n; i++) {
            unsigned char *q = (unsigned char *) (bd + i * 4);
            q[0] = (unsigned char) (g_out_ids[i] >> 24); q[1] = (unsigned char) (g_out_ids[i] >> 16);
            q[2] = (unsigned char) (g_out_ids[i] >> 8);  q[3] = (unsigned char) g_out_ids[i];
        }
        (*env)->ReleaseByteArrayElements(env, out, bd, 0);
    }
    return out;
}
JNIEXPORT jint JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeLastEog(JNIEnv *env, jclass cls) {
    (void) env; (void) cls; return (jint) g_out_eog;
}

#include <pthread.h>
static void * kw_thread(void * arg) { (void) arg; extern void moe_pool_keepwarm(int); moe_pool_keepwarm(0); return NULL; }

JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeGenerate(
        JNIEnv *env, jclass cls, jobject sink, jint max_tokens, jfloat temp, jboolean thinking_on) {
    // Toggling thinking changes the rendered instruction line, which the UI-text
    // history comparison cannot see - trusting positions would keep dirty KV context.
    // Authoritative checkpoint: on a thinking change with live context, force a full rebuild.
    if (g_ctx && g_thinking != (thinking_on ? 1 : 0) && g_nprocessed > 0) g_reset_next = 1;
    g_thinking = thinking_on ? 1 : 0;
    char out[512];
    out[0] = 0;
    if (!g_ctx) { snprintf(out, sizeof(out), "ERR not init"); return (*env)->NewStringUTF(env, out); }
        {
        const char *t0 = llama_model_chat_template(g_model, NULL);
        g_tpl_imstart = (t0 && strstr(t0, "<|im_start|>assistant")) ? 1 : 0;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(g_model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    static char render[65536];
    const char *tmpl = llama_model_chat_template(g_model, NULL); // BAILING2 -> built-in renderer
    int has_ids = 0;
    for (int i = 0; i < g_nmsg; i++) if (g_msg_nids[i] > 0) { has_ids = 1; break; }
    int32_t n_tok = 0;
    if (!has_ids) {
        int32_t need = llama_chat_apply_template(tmpl, g_hist, (size_t) g_nmsg, true, render, sizeof(render));
        if (need <= 0 || (size_t) need >= sizeof(render)) {
            snprintf(out, sizeof(out), "ERR template need=%d", need);
            return (*env)->NewStringUTF(env, out);
        }
        if (!g_thinking) {
            if (g_tpl_imstart == 1) {
                // qwen off = pure empty pair (reference-template verified): no instruction line, no fabricated system turn
                need = qwen_pure_erase_directive(render, need);
            }
        } else {
            if (g_tpl_imstart != 1) need = bailing_flip_on(render, need);
        }
        bail_ensure_directive(render, &need, (int32_t) sizeof(render));
        if (g_tpl_imstart != 1) bailing_think_heads(render, &need, (int32_t) sizeof(render));
        qwen_inject_empty_pair(render, "<|im_start|>assistant\n", (int32_t) sizeof(render));
        n_tok = llama_tokenize(vocab, render, need, g_toks, MAX_TOKS, false, true);
        if (n_tok <= 0) { snprintf(out, sizeof(out), "ERR tokenize %d", n_tok); return (*env)->NewStringUTF(env, out); }
    } else {
        // sidecar stitching: text segments render+tokenize, id segments paste verbatim
        // (kills BPE re-encode boundary drift). Every text segment carries the generation
        // prompt; id turns = stepped body + that turn's end-of-generation token.
        int32_t w = 0; int ii = 0; int err = 0;
        while (ii < g_nmsg) {
            if (g_msg_nids[ii] > 0) {
                if (w + g_msg_nids[ii] + 1 >= MAX_TOKS - 8) { err = 2; break; }
                memcpy(&g_toks[w], g_msg_ids[ii], sizeof(llama_token) * (size_t) g_msg_nids[ii]);
                w += g_msg_nids[ii];
                if (g_msg_eog[ii] >= 0) g_toks[w++] = g_msg_eog[ii];
                ii++;
                continue;
            }
            int j = ii; while (j < g_nmsg && g_msg_nids[j] == 0) j++;
            int32_t need = llama_chat_apply_template(tmpl, &g_hist[ii], (size_t) (j - ii), true, render, sizeof(render));
            if (need <= 0 || (size_t) need >= sizeof(render)) { err = 1; break; }
            /* fake SYSTEM head stripping: the built-in renderer injects its default
               system block into every non-first segment start; strip or the prompt
               grows a second SYSTEM block and the model loses its identity */
            if (g_tpl_imstart != 1 && ii > 0) need = bail_strip_fake_head(render, need);
            if (!g_thinking) {
                if (g_tpl_imstart == 1) {
                    // qwen off = pure empty pair (reference-template verified): no instruction line, no fabricated system turn
                    need = qwen_pure_erase_directive(render, need);
                }
            } else {
                if (g_tpl_imstart != 1) need = bailing_flip_on(render, need);
            }
            // run every segment unconditionally: gating this on the thinking branch or
            // the last segment lost the instruction line from turn two onward (the
            // empty-answer bug). Families without a SYSTEM in the segment self-check to a no-op.
            if (g_tpl_imstart != 1) bail_ensure_directive(render, &need, (int32_t) sizeof(render));
            if (g_tpl_imstart != 1) bailing_think_heads(render, &need, (int32_t) sizeof(render));
            if (g_tpl_imstart == 1 && j >= g_nmsg) qwen_inject_empty_pair(render, "<|im_start>|assistant\n", (int32_t) sizeof(render));
            int32_t room = MAX_TOKS - 8 - w;
            int32_t tn = llama_tokenize(vocab, render, need, g_toks + w, room, false, true);
            if (tn <= 0) { err = 1; break; }
            w += tn; ii = j;
        }
        if (err) { snprintf(out, sizeof(out), "ERR sidecar concat %d", err); return (*env)->NewStringUTF(env, out); }
        n_tok = w;
    }
    if (n_tok >= MAX_TOKS - 8) { snprintf(out, sizeof(out), "ERR ctx too long"); return (*env)->NewStringUTF(env, out); }

    {   // render forensics: single-line dump of the tail (newlines escaped) for empty-pair / migration checks
        char dbg[512]; int di = 0;
        int rl = (int) strlen(render);
        int tl = rl > 110 ? 110 : rl;
        const char * t = render + rl - tl;
        for (int z = 0; z < tl && di < 380; z++) {
            if (t[z] == '\n') { dbg[di++] = '\\'; dbg[di++] = 'n'; }
            else dbg[di++] = t[z];
        }
        dbg[di] = 0;
        LOGI("REND tail=[%s]", dbg);
        {   /* head forensics: SYSTEM segment identity + instruction-line shape */
            char dh[260]; int dj = 0;
            for (int z = 0; z < (int) strlen(render) && z < 130 && dj < 250; z++) {
                char ch = render[z];
                if (ch == '\n') { dh[dj++] = '\\'; dh[dj++] = 'n'; }
                else dh[dj++] = ch;
            }
            dh[dj] = 0;
            LOGI("REND head=[%s]", dh);
        }
    }
    int start = g_reset_next ? 0 : g_nprocessed;
    if (start > n_tok) start = 0;
    if (g_reset_next) {
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, -1, -1);
        g_reset_next = 0;
    }

    long long ph0=0, pm0=0, pl0=0, pe0=0, pb0=0, pst0=0;
    size_t pcap0=0, pused0=0;
    moe_pool_stats(&ph0, &pm0, &pl0, &pe0, (long long*)&pb0, &pcap0, &pused0);
    { long long su,st,pu,pm; moe_pool_extra(&pst0,&su,&st,&pu,&pm); }
    jmethodID invoke = (*env)->GetMethodID(env, (*env)->GetObjectClass(env, sink),
                                           "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
    long long prefill_ms;
    {
        long long t0 = now_ms();
        // small incremental batches decode token-by-token: MoE gather cost does not
        // amortize on tiny batches (measured 7x slower than decode); <=32 new tokens
        // take the decode-like path, long batches keep the 512 window.
        const int32_t inc = n_tok - start;
        const int32_t chunk_max = (inc > 0 && inc <= 32) ? 1 : 512;
        for (int i = start; i < n_tok; i += chunk_max) {
            int chunk = (n_tok - i < chunk_max) ? n_tok - i : chunk_max;
            struct llama_batch b = llama_batch_get_one(g_toks + i, chunk);
            if (llama_decode(g_ctx, b) != 0) { snprintf(out, sizeof(out), "ERR prefill@%d", i); return (*env)->NewStringUTF(env, out); }
            if (g_cancel) { g_cancel = 0; snprintf(out, sizeof(out), "CANCELLED"); return (*env)->NewStringUTF(env, out); }
        }
        prefill_ms = now_ms() - t0;
        { long long h, m, l, e, b; long long su, stq, pu, pm; size_t cap, used; long long st1;
          moe_pool_stats(&h, &m, &l, &e, &b, &cap, &used);
          moe_pool_extra(&st1, &su, &stq, &pu, &pm);
          char dbg[256];
          snprintf(dbg, sizeof(dbg),
                   "PRE fill hit=%lld miss=%lld evict=%lld stall_ms=%lld loads=%lld",
                   h - ph0, m - pm0, e - pe0, (st1 - pst0) / 1000000, l - pl0);
          LOGI("%s", dbg); }
    }
    int reused = start;
    g_nprocessed = n_tok;

    {   // vocab width changes across models: the token buffer must realloc (8B-sized buffer fed a 35B vocab = OOB write)
        static size_t g_td_cap = 0;
        if (g_td_cap < (size_t) n_vocab) {
            free(g_td);
            g_td = (llama_token_data *) malloc(sizeof(llama_token_data) * (size_t) n_vocab);
            g_td_cap = g_td ? (size_t) n_vocab : 0;
        }
    }
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temp <= 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0xC0FFEEu));
    }
    char reply[32768]; size_t rlen = 0; reply[0] = 0;
    g_rlen = 0;
    flush_pend();
    char piece[256];
    long long t0 = now_ms();
    int gen = 0;
    long long first_ms = 0;
    g_out_n = 0; g_out_eog = -1;
    double tps = 0.0;
    llama_token next = -1;
    while (gen < max_tokens && !g_cancel) {
        const float *lg = llama_get_logits_ith(g_ctx, -1);
        if (!lg) { snprintf(out, sizeof(out), "ERR logits@%d", gen); break; }
        for (int32_t j = 0; j < n_vocab; j++) { g_td[j].id = j; g_td[j].logit = lg[j]; g_td[j].p = 1.0f; }
        llama_token_data_array cur = { .data = g_td, .size = (size_t) n_vocab, .selected = -1, .sorted = false };
        llama_sampler_apply(smpl, &cur);
        next = cur.data[cur.selected].id;
        if (llama_vocab_is_eog(vocab, next)) { g_out_eog = next; break; }
        int32_t plen = llama_token_to_piece(vocab, next, piece, sizeof(piece) - 1, 0, false);
        if (plen > 0) {
            piece[plen] = 0;
            if (rlen + plen < sizeof(reply)) { memcpy(reply + rlen, piece, plen); rlen += plen; }
            if (g_rlen + plen < sizeof(g_reply_tail)) { memcpy(g_reply_tail + g_rlen, piece, plen); g_rlen += plen; }
            else g_rlen = sizeof(g_reply_tail) + 1;
            emit_valid_utf8(env, sink, invoke, piece, plen);
        }
        if (g_out_n < MAX_TOKS) g_out_ids[g_out_n++] = next;
        gen++;
        if (gen == 1) first_ms = now_ms() - t0;
        struct llama_batch b = llama_batch_get_one(&next, 1);
        if (llama_decode(g_ctx, b) != 0) { snprintf(out, sizeof(out), "ERR decode@%d", gen); break; }
        g_nprocessed++;
        tps = gen * 1000.0 / (double) (now_ms() - t0 + 1);
    }
    flush_pend();
    while (rlen > 0 && (((unsigned char)reply[rlen - 1] & 0xC0) == 0x80)) rlen--;
    if (rlen > 0) { unsigned char b = (unsigned char)reply[rlen - 1];
        int expect = (b < 0x80) ? 1 : (b < 0xE0) ? 2 : (b < 0xF0) ? 3 : 4;
        if (expect > 1) rlen--; }
    reply[rlen] = 0;
    if (g_cancel) { g_cancel = 0;
        snprintf(out, sizeof(out), "CANCELLED gen=%d tps=%.2f prefill_ms=%lld first_ms=%lld prompt=%d reused=%d",
                 gen, tps, prefill_ms, first_ms, n_tok, reused); }
    else snprintf(out, sizeof(out), "OK gen=%d tps=%.2f prefill_ms=%lld first_ms=%lld prompt=%d reused=%d",
                  gen, tps, prefill_ms, first_ms, n_tok, reused);
    LOGI("%s", out);

    if (gen > 0 && getenv("E0_KEEPWARM_MB")) { pthread_t kt; if (pthread_create(&kt, NULL, kw_thread, NULL) == 0) pthread_detach(kt); }
    {   // assistant turn recorded with stepped ids (sidecar); the text reply stays in logs only
        char *ar = strdup("assistant");
        g_strs[g_nmsg * 2] = ar; g_strs[g_nmsg * 2 + 1] = NULL;
        if (g_nmsg < MAX_MSG) {
            llama_token *ids = NULL;
            if (g_out_n > 0) { ids = (llama_token *) malloc(sizeof(llama_token) * (size_t) g_out_n); memcpy(ids, g_out_ids, sizeof(llama_token) * (size_t) g_out_n); }
            g_hist[g_nmsg].role = ar; g_hist[g_nmsg].content = "";
            g_msg_ids[g_nmsg] = ids; g_msg_nids[g_nmsg] = ids ? g_out_n : 0; g_msg_eog[g_nmsg] = g_out_eog;
            g_nmsg++;
        }
    }
    (void) reply;
    llama_sampler_free(smpl);
    return (*env)->NewStringUTF(env, out);
}

// pool counters + process RSS export (the metrics line under each reply: hit/stall/peak-rss).
// moe_pool_* symbols live in libggml-cpu.so (linked explicitly at build time).
extern void moe_pool_stats(long long * hits, long long * misses, long long * loads, long long * evicts,
                           long long * bytes_loaded, size_t * cap_bytes, size_t * used_bytes);
extern void moe_pool_extra(long long * stall_ns, long long * slots_used, long long * slots_total,
                           long long * pf_used, long long * pf_mispredict);

JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeStats(JNIEnv *env, jclass cls) {
    (void) cls;
    long long h=0, m=0, l=0, e=0, bl=0, st=0, su=0, stot=0, pu=0, pm=0;
    size_t cap=0, used=0;
    moe_pool_stats(&h, &m, &l, &e, &bl, &cap, &used);
    moe_pool_extra(&st, &su, &stot, &pu, &pm);
    long vmhwm = 0, vmrss = 0;
    FILE * f = fopen("/proc/self/status", "r");
    if (f) {
        char ln[256];
        while (fgets(ln, sizeof(ln), f)) {
            if (!strncmp(ln, "VmHWM:", 6)) vmhwm = strtol(ln + 6, NULL, 10);
            else if (!strncmp(ln, "VmRSS:", 6)) vmrss = strtol(ln + 6, NULL, 10);
        }
        fclose(f);
    }
    char buf[512];
    snprintf(buf, sizeof(buf),
        "hits=%lld misses=%lld loads=%lld evicts=%lld flash=%lld resident=%lld cap=%lld\n" 
        "stall_ns=%lld slots=%lld/%lld pf_used=%lld pf_miss=%lld vmhwm_kb=%ld vmrss_kb=%ld",
        h, m, l, e, bl, (long long)used, (long long)cap, st, su, stot, pu, pm, vmhwm, vmrss);
    return (*env)->NewStringUTF(env, buf);
}

// tail 8KB of the last assistant reply in native form (when the UI shows equivalent text, replaying it keeps full prefix reuse; longer = TOOLONG)
JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeLastAssistant(JNIEnv *env, jclass cls) {
    (void) cls;
    if (g_rlen > sizeof(g_reply_tail)) return (*env)->NewStringUTF(env, "TOOLONG");
    char tmp[8200];
    size_t n = g_rlen; if (n > sizeof(g_reply_tail) - 1) n = sizeof(g_reply_tail) - 1;
    memcpy(tmp, g_reply_tail, n); tmp[n] = 0;
    return (*env)->NewStringUTF(env, tmp);
}

JNIEXPORT void JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeCancel(JNIEnv *env, jclass cls) {
    (void) env; (void) cls; g_cancel = 1;
}

// unload/free: real teardown. Order is law: sampler first, then adapter before model
// (freeing the model under an attached adapter crashed at 452s once) - plus all-static reset.
JNIEXPORT jstring JNICALL
Java_dev_edge0_runtime_app_runtime_LlamaNative_nativeFree(JNIEnv *env, jclass cls) {
    (void) cls;
    if (g_ctx) { llama_free(g_ctx); g_ctx = NULL; }
    if (g_lora) { llama_adapter_lora_free(g_lora); g_lora = NULL; }
    if (g_model) { llama_model_free(g_model); g_model = NULL; }
    for (int i = 0; i < g_nmsg * 2; i++) { free(g_strs[i]); g_strs[i] = NULL; }
    for (int i = 0; i < g_nmsg; i++) { free(g_msg_ids[i]); g_msg_ids[i] = NULL; g_msg_nids[i] = 0; }
    g_nmsg = 0; g_nprocessed = 0; g_reset_next = 0;
    { extern void moe_pool_reset_all(void); moe_pool_reset_all(); }
    return (*env)->NewStringUTF(env, "OK");
}
