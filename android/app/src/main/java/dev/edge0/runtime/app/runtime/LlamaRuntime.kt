// runtime/LlamaRuntime.kt - the Runtime implementation backed by llama.cpp + GGUF.
// Native side = LlamaNative / llama_chat.c; shared libs are staged into
// build-dl/llama-libs at build time (gitignored, produced by
// tools/llama/build_vendor_libs.sh from the pinned engine tree).
// Auto tiering (mirrors the validated CLI configuration): model >= 12GB takes the
//   35B tier - expert pool 6144MB + blob 3072MB + n_ctx 4096, demand-only
//   (prefetch measured net-negative on-device, so E0_PREROUTER is never set);
//   otherwise the 8B tier - fully resident, pool/blob disabled, n_ctx 8192,
//   companion LoRA auto-discovered (lora_*.gguf).
// LoRA companion applies to the 8B tier only: the 35B build bakes its LoRA in,
// and attaching an adapter on top fails initialization (verified on-device).
// History contract: native is append-only; this class keeps a mirror and does a
// prefix diff - any fork triggers a full rebuild (thread switch / message edit).
package dev.edge0.runtime.app.runtime

import dev.edge0.runtime.engine.ChatMessage
import dev.edge0.runtime.engine.GenParams
import dev.edge0.runtime.engine.Status
import dev.edge0.runtime.app.runtime.PoolMetrics
import java.io.File
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class LlamaRuntime : Runtime {
    override val state = MutableStateFlow<GenState>(GenState.Idle)
    override val lastMetrics = MutableStateFlow<RequestMetrics?>(null)

    @Volatile private var _dir: String? = null
    override val activeModelDir: String? get() = _dir

    private val slotMutex = Mutex()
    private val genExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "llama-gen").apply { isDaemon = true }
    }
    private val genCtx: CoroutineDispatcher = genExecutor.asCoroutineDispatcher()

    private val mirror = ArrayList<Pair<String, String>>()   // (role, content) kept in sync with native

    companion object {
        const val TAG = "LlamaRuntime"
        const val BIG_MODEL_BYTES = 12L shl 30   // tier threshold (midpoint of 8B=5.0GB / 35B=21.7GB)
    }

    private fun runCatchingNative(block: () -> String): String? =
        try { block() } catch (t: Throwable) {
            // never swallow native failures into a generic "so load failed" -
            // the real cause must surface in the return value
            "ERR native: ${t.javaClass.simpleName}: ${t.message?.take(160)}"
        }

    override suspend fun ensureLoaded(path: String) {
        slotMutex.withLock {
            if (_dir == path && state.value !is GenState.Error) return
            val model = resolveGguf(path)
                ?: run {
                    state.value = GenState.Error(Status.FORMAT_VERSION, "model.gguf not found: $path")
                    return
                }
            state.value = GenState.Loading(path)
            if (_dir != null) {   // model swap: unload first (never two loads resident; nativeInit self-clears as backstop)
                runCatching { LlamaNative.nativeFree() }
                _dir = null
            }
            val big = model.length() >= BIG_MODEL_BYTES
            val dir = model.parentFile
            val lora = if (big) "" else dir?.listFiles { f ->
                f.name.startsWith("lora_") && f.name.endsWith(".gguf")
            }?.firstOrNull()?.absolutePath ?: ""
            val heads = dir?.let { File(it, "e0_heads.bin") }?.takeIf { it.exists() }?.absolutePath ?: ""
            val r = withContext(Dispatchers.IO) {
                runCatchingNative {
                    LlamaNative.nativeInit(model.absolutePath, lora,
                        if (big) heads else "",          // heads ship with the 35B tier (idle unless E0_PREROUTER is enabled)
                        if (big) 4096 else 8192,
                        if (big) 6144 else 0,
                        if (big) 3072 else 0,
                        0,                               // prefetch: measured net-negative, permanently off
                        4)
                }
            }
            if (r == null || !r.startsWith("OK")) {
                state.value = GenState.Error(Status.IO, (r ?: "libedge0llama.so failed to load"))
                return
            }
            mirror.clear()
            _dir = path
            state.value = GenState.Idle
        }
    }

    /** path may be a .gguf file, or a directory containing model.gguf. */
    private fun resolveGguf(path: String): File? {
        val f = File(path)
        return when {
            f.isFile && path.endsWith(".gguf") -> f
            f.isDirectory -> File(f, "model.gguf").takeIf { it.exists() }
                ?: f.listFiles { x -> x.name.endsWith(".gguf") && !x.name.startsWith("lora_") }?.firstOrNull()
            else -> null
        }
    }

    override suspend fun unloadActive() {
        slotMutex.withLock {
            if (_dir == null) return
            withContext(Dispatchers.IO) { runCatchingNative { LlamaNative.nativeFree() } }
            mirror.clear()
            _dir = null
            if (state.value !is GenState.Error) state.value = GenState.Idle
        }
    }

    override suspend fun trim(ratio: Float): Boolean {
        // TODO: expose moe_pool trim through native; the pool currently self-sheds via E0_MEM_FLOOR_MB
        return true
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    override fun submit(threadId: String, messages: List<ChatMessage>,
                        thinkingOn: Boolean, params: GenParams): Flow<GenEvent> =
        callbackFlow {
            if (_dir == null) {
                trySend(GenEvent.Failed(Status.INVALID_ARG, "no active llama session")); close(); return@callbackFlow
            }
            if (!state.compareAndSet(GenState.Idle, GenState.Streaming(threadId))) {
                trySend(GenEvent.Failed(Status.INVALID_ARG, "generation slot busy")); close(); return@callbackFlow
            }
            launch(genCtx) {
                val splitter = ThinkSplitter(initialThink = thinkingOn)
                val tSend = System.nanoTime()
                var tFirst = 0L
                val uiText = StringBuilder()
                val sink: (String) -> Unit = { piece ->
                    if (piece.isNotEmpty()) {
                        if (tFirst == 0L) tFirst = System.nanoTime()
                        uiText.append(piece)
                        splitter.feed(piece).forEach { trySend(it.toEvent()) }
                    }
                }
                syncHistory(messages)
                val res = runCatchingNative {
                    LlamaNative.nativeGenerate(sink, params.maxNewTokens.coerceAtLeast(1),
                        params.temperature, thinkingOn)
                } ?: "ERR native"
                splitter.flush().forEach { trySend(it.toEvent()) }
                // native already appended this turn with stepped ids (sidecar); mirror keeps index parity
                if (res.startsWith("OK")) mirror.add("assistant" to uiText.toString())
                val term: GenEvent = when {
                    res.startsWith("OK") || res.startsWith("CANCELLED") -> {
                        val m = parseMetrics(res)
                        val mm = m?.copy(
                            // user-perceived TTFT measured here (native first_ms is post-prefill sampling time)
                            firstTokenMs = if (tFirst > 0) (tFirst - tSend) / 1_000_000 else m.firstTokenMs,
                            peakRssBytes = readPeakRss(),
                            pool = poolMetrics(),
                            gpu = GpuMetrics(probed = false, ok = false, active = false,
                                device = "cpu(llama-route)", why = "CPU-only (shipping configuration)"),
                        )
                        if (mm != null) lastMetrics.value = mm
                        if (res.startsWith("OK")) GenEvent.Done(mm) else GenEvent.Cancelled(mm)
                    }
                    else -> GenEvent.Failed(Status.IO, res)
                }
                trySend(term)
                if (state.value !is GenState.Error) state.value = GenState.Idle
                close()
            }
            awaitClose { }
        }

    private fun syncHistory(msgs: List<ChatMessage>) {
        // Prefix match: user turns compare verbatim (UI text feeds native losslessly);
        // assistant turns match by ROLE ONLY - the ground truth lives on the native side
        // as stepped ids (sidecar), while UI text is a lossy reassembly with think tags
        // stripped. Verbatim comparison there would always fake a fork and replay the
        // whole history every turn (once cost 60s of dead waiting). Non-contiguous hits
        // must not count toward the common prefix either.
        // Known edge: editing/regenerating history falls back to full rebuild.
        fun ok(i: Int) = i < mirror.size && mirror[i].first == msgs[i].role &&
            (msgs[i].role != "user" || mirror[i].second == msgs[i].content)
        var common = 0
        while (common < msgs.size && ok(common)) common++
        if (common == mirror.size) {
            msgs.drop(common).forEach { m ->
                val content = m.content
                LlamaNative.nativeAddRole(m.role, content)
                mirror.add(m.role to content)
            }
        } else {
            LlamaNative.nativeReset()
            mirror.clear()
            msgs.forEach { m ->
                val content = m.content
                LlamaNative.nativeAddRole(m.role, content)
                mirror.add(m.role to content)
            }
        }
    }

    internal fun parseMetrics(res: String): RequestMetrics? {
        val kv = Regex("""(\w+)=(\S+)""").findAll(res).associate { it.groupValues[1] to it.groupValues[2] }
        val gen = kv["gen"]?.toIntOrNull() ?: return null
        val tps = kv["tps"]?.toDoubleOrNull() ?: 0.0
        val pf = kv["prefill_ms"]?.toLongOrNull() ?: 0L
        val first = kv["first_ms"]?.toLongOrNull() ?: 0L
        val prompt = kv["prompt"]?.toIntOrNull() ?: 0
        val reused = kv["reused"]?.toIntOrNull() ?: 0
        return RequestMetrics(
            promptTokens = prompt,
            prefix_reused = reused > 0,
            prefix_reused_tokens = reused,
            newTokens = gen,
            prefillMs = pf,
            firstTokenMs = first,
            decodeMs = (if (gen > 0) (gen * 1000.0 / (tps + 1e-6)).toLong() else 0L) + first,
            prefillTokS = if (pf > 0) (prompt - reused) * 1000.0 / pf else 0.0,
            decodeTokS = tps,
            peakRssBytes = 0,
            chatTurns = 1,
        )
    }

    private fun poolMetrics(): PoolMetrics? {
        val raw = runCatching { LlamaNative.nativeStats() }.getOrNull() ?: return null
        if (!raw.contains("hits=")) return null
        val kv = raw.split(" ", "\n").mapNotNull {
            val i = it.indexOf('='); if (i <= 0) null else it.substring(0, i) to it.substring(i + 1)
        }.toMap()
        fun L(k: String) = (kv[k]?.toLongOrNull() ?: 0L)
        val slots = (kv["slots"] ?: "").split("/")
        return PoolMetrics(
            resident_bytes = L("resident"),
            hits = L("hits"), misses = L("misses"),
            prefetch_used = L("pf_used"), prefetch_mispredict = L("pf_miss"),
            evictions = L("evicts"), stall_ns = L("stall_ns"),
            flash_bytes = L("flash"),
            slots_used = slots.firstOrNull()?.toLongOrNull() ?: 0L,
            slots_total = slots.getOrNull(1)?.toLongOrNull() ?: 0L,
        )
    }

    private fun readPeakRss(): Long {
        val raw = runCatching { LlamaNative.nativeStats() }.getOrNull() ?: return 0
        return Regex("vmhwm_kb=(\\d+)").find(raw)?.groupValues?.get(1)?.toLongOrNull()?.times(1024) ?: 0
    }

    /** Raw native text of the last assistant turn (ground truth for history sync). */
    fun lastAssistantNative(): String? = runCatching { LlamaNative.nativeLastAssistant() }.getOrNull()

    override fun cancelActive() {
        runCatching { LlamaNative.nativeCancel() }
    }
}
