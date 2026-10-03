// LlamaRuntimeTest - app-domain smoke suite for the llama engine shell.
// Prerequisite: stage the models AFTER installing the test APK -
//   adb shell run-as dev.edge0.runtime.app cp /data/local/tmp/lgguf/edge0-8b.gguf files/models/edge0-8b.gguf
// instrumented installs wipe app data, so staging must follow installation.
// Assertions keep a conservative floor (availability over speed; the CLI benchmark
// owns ceiling checks): >= 8 tokens produced, decode > 1 t/s, non-empty text.
package dev.edge0.runtime.app

import dev.edge0.runtime.app.runtime.GenEvent
import dev.edge0.runtime.app.runtime.LlamaRuntime
import dev.edge0.runtime.engine.ChatMessage
import dev.edge0.runtime.engine.GenParams
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class LlamaRuntimeTest {
    @Test
    fun eightBChatRoundTrip() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        // files staged via run-as may carry an SELinux label the instrumentation cannot
        // read; search candidate roots instead. Production loads through the normal path.
        val want = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("model") ?: "edge0-35b.gguf"
        val model = sequenceOf(
            File(ctx.getDir("models", 0), want),
            File(ctx.filesDir, "models/$want"),
            File(ctx.cacheDir, "models/$want"),
            File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "models/$want"),
        ).firstOrNull { it.isFile }
            ?: run {
                val hit = generateSequence(ctx.dataDir) { f ->
                    f.listFiles()?.firstOrNull { it.isDirectory && !it.name.startsWith(".") }
                }.flatMap { (it.walk().firstOrNull { f2 -> f2.name == want }?.let { x -> sequenceOf(x) } ?: emptySequence()) }
                    .firstOrNull()
                hit ?: File(ctx.getDir("models", 0), want)
            }
        assertTrue("model not found anywhere: ${model.path}", model.isFile)

        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        assertTrue("load failed: ${rt.state.value}", rt.activeModelDir != null)

        val pieces = StringBuilder()
        var tps = 0.0
        rt.submit("t1", listOf(ChatMessage("user", "Introduce yourself in one sentence")),
                  true, GenParams(temperature = 0f, maxNewTokens = 32))
            .toList().forEach { ev ->
                when (ev) {
                    is GenEvent.TextPiece -> pieces.append(ev.s)
                    is GenEvent.ThinkPiece -> pieces.append(ev.s)
                    is GenEvent.Done -> tps = ev.metrics?.decodeTokS ?: 0.0
                    is GenEvent.Failed -> throw AssertionError("gen failed: ${ev.diag}")
                    else -> Unit
                }
            }
        android.util.Log.i("LlamaRT", "reply=${pieces.take(80)} tps=$tps")
        // metrics wiring assertions (pool/rss must not be N/A)
        val rm = rt.lastMetrics.value
        assertTrue("metrics missing", rm != null)
        assertTrue("pool metrics missing", rm?.pool != null)
        if (want.contains("35b")) {
            assertTrue("35B pool counters not wired", (rm!!.pool!!.hits + rm.pool!!.misses) > 0 && rm.pool!!.slots_total > 0)
        } else {
            // the 8B tier is fully resident: counters exist and read zero - that is the correct shape
            assertTrue("8B should not occupy the pool", rm!!.pool!!.slots_total == 0L)
        }
        assertTrue("peak rss N/A", (rm?.peakRssBytes ?: 0L) > 100L * 1024 * 1024)
        assertTrue("first-token timing anomaly: ${rm?.firstTokenMs}", (rm?.firstTokenMs ?: 0) > 0)
        assertTrue("empty reply", pieces.isNotBlank())
        val floor = if (want.contains("35b")) 0.5 else 1.0
        assertTrue("tps too low: $tps", tps > floor)

        // second turn exercises KV prefix reuse (history diff sync)
        val r2 = rt.submit("t2", listOf(
            ChatMessage("user", "Introduce yourself in one sentence"),
            ChatMessage("assistant", pieces.toString().trim().take(40)),
            ChatMessage("user", "a bit shorter"),
        ), true, GenParams(temperature = 0f, maxNewTokens = 24)).toList()
        assertTrue("round2 no done", r2.any { it is GenEvent.Done })
        rt.unloadActive()
    }

    // crash regression: use 8B, switch to 35B, send in a new session - stale LoRA/session
// state attaching to the new graph used to abort; bidirectional switching must survive.
    @Test
    fun switch8bThen35bSurvival() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        fun find(n: String) = sequenceOf(File(ctx.getDir("models", 0), n), File(ctx.filesDir, "models/$n"))
            .firstOrNull { it.isFile }
        val m8 = find("edge0-8b.gguf") ?: return@runBlocking
        val rt = LlamaRuntime()
        fun round(model: File, q: String) = runBlocking {
            rt.ensureLoaded(model.absolutePath)
            assertTrue("load fail ${model.name}: ${rt.state.value}", rt.activeModelDir != null)
            var n = 0
            rt.submit("sw", listOf(ChatMessage("user", q)), true,
                      GenParams(temperature = 0f, maxNewTokens = 8))
                .toList().forEach { ev ->
                    when (ev) {
                        is GenEvent.TextPiece -> n += ev.s.length
                        is GenEvent.ThinkPiece -> n += ev.s.length
                        is GenEvent.Failed -> throw AssertionError("${model.name}: ${ev.diag}")
                        else -> Unit
                    }
                }
            assertTrue("no output ${model.name}", n > 2)
        }
        round(m8, "hello")
        val m35 = find("edge0-35b.gguf")
        if (m35 != null) {
            round(m35, "hello")          // the original crash point
            round(m8, "hello")           // switch back (pool/registry reset verified both ways)
        }
        rt.unloadActive()
    }

    // real multi-turn on 35B (long answer, then thanks): log reused/prefill per round to locate forks
    @Test
    fun prefixReuseProbe() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val want = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("model") ?: "edge0-35b.gguf"
        val model = sequenceOf(File(ctx.getDir("models", 0), want), File(ctx.filesDir, "models/$want"))
            .firstOrNull { it.isFile } ?: return@runBlocking
        val syncMode = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("sync") ?: ""
        if (syncMode.isNotEmpty()) android.system.Os.setenv("E0_POOL_SYNC", "1", true)
        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        val probeThinking = (androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("thinking") ?: "off") == "on"
        val hist = mutableListOf<ChatMessage>()
        val qs = listOf("Describe three core advantages of Mixture-of-Experts models in detail.", "thanks")
        for ((i, q) in qs.withIndex()) {
            hist.add(ChatMessage("user", q))
            val sb = StringBuilder()
            rt.submit("probe$i", hist.toList(), probeThinking, GenParams(temperature = 0f, maxNewTokens = if (i == 0) 200 else 20))
                .toList().forEach { ev ->
                    when (ev) {
                        is GenEvent.TextPiece -> sb.append(ev.s)
                        is GenEvent.ThinkPiece -> sb.append("«T»").append(ev.s)
                        is GenEvent.Failed -> throw AssertionError(ev.diag)
                        else -> Unit
                    }
                }
            hist.add(ChatMessage("assistant", sb.toString()))
            if (i == 0) Thread.sleep(25000)   // simulate reading the reply: the keepwarm battleground
            android.util.Log.i("ReuseProbe", "round$i head=[${sb.toString().take(70).replace("\n", " | ")}] len=${sb.length} metrics=${rt.lastMetrics.value?.let { "prompt=${it.promptTokens} reused=${it.prefix_reused_tokens} prefill_ms=${it.prefillMs} first_ms=${it.firstTokenMs} pool=${it.pool?.let { q -> "hits=${q.hits} miss=${q.misses} evict=${q.evictions} stall_ms=${q.stall_ns / 1000000} res_MB=${q.resident_bytes / 1048576} pf=${q.prefetch_used}/${q.prefetch_mispredict}" }}" }}")
        }
        rt.unloadActive()
    }

    // thinking quality gate: same greedy prompt, 12 tokens, quadrants of {system,thinking}.
    // Verdict via logcat ThinkGate: off cells must answer directly, on cells must think.
    // Regression anchor: custom-system off leaking thinking (the ensure-position fix).
    @Test
    fun thinkSwitchQualityGate() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val want = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("model") ?: "edge0-8b.gguf"
        val model = sequenceOf(File(ctx.getDir("models", 0), want), File(ctx.filesDir, "models/$want"))
            .firstOrNull { it.isFile } ?: return@runBlocking
        val q = "Explain why the sky is blue in one short sentence."
        val sp = "You're Edge0, an on-device AI assistant."
        val heads = ArrayList<String>()
        for (sys in listOf(false, true)) for (th in listOf(false, true)) {
            val rt = LlamaRuntime()
            rt.ensureLoaded(model.absolutePath)
            val base0 = if (sys) listOf(ChatMessage("system", sp)) else emptyList()
            val turns = ArrayList<String>()
            val hist = base0.toMutableList()
            for (t in 0..1) {  // round two goes through the sidecar path (the empty-answer incident scene)
                val sb = StringBuilder()
                hist.add(ChatMessage("user", if (t == 0) q else "thanks, one more short sentence"))
                rt.submit("g$sys$th$t", hist.toList(), th,
                          GenParams(temperature = 0f, maxNewTokens = 12))
                    .toList().forEach { ev ->
                        when (ev) {
                            is GenEvent.TextPiece -> sb.append(ev.s)
                            is GenEvent.ThinkPiece -> sb.append("<T>").append(ev.s)
                            is GenEvent.Failed -> throw AssertionError(ev.diag)
                            else -> Unit
                        }
                    }
                turns.add("t" + t + "=" + sb.toString().take(44).replace("\n", " "))
                hist.add(ChatMessage("assistant", sb.toString()))
            }
            heads.add((if (sys) "sys" else "no") + "-th" + (if (th) "on" else "off") + ": " + turns.joinToString(" ~ "))
            // empty-answer anchor: no quadrant, no round may return empty
            org.junit.Assert.assertTrue("gate: empty reply in " + heads.last(),
                turns.all { it.length > 6 })
            rt.unloadActive()
        }
        android.util.Log.i("ThinkGate", heads.joinToString(" ||| "))
        org.junit.Assert.assertTrue("gate: 4 quadrants", heads.size == 4)
        org.junit.Assert.assertTrue("gate: off-no-sys empty", heads[0].length > 12)
    }

    // multi-turn identity probe: chat once (builds sidecar history), then ask who you are.
    // Single-turn was green; the incident lived in the multi-turn path.
    @Test
    fun whoAreYouSysProbe() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val model = File(ctx.filesDir, "models/edge0-8b.gguf")
        if (!model.isFile) return@runBlocking
        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        var heads_all = ""
        val hist = mutableListOf(ChatMessage("system", "You're Edge0, an on-device AI assistant."))
        for ((i, q) in listOf("What is the capital of France? One word.",
                              "Who are you? What is your name? Answer in one sentence.").withIndex()) {
            hist.add(ChatMessage("user", q))
            val sb = StringBuilder()
            rt.submit("way$i", hist.toList(), false, GenParams(temperature = 0.2f, maxNewTokens = 60))
                .toList().forEach { ev ->
                    when (ev) {
                        is GenEvent.TextPiece -> sb.append(ev.s)
                        is GenEvent.ThinkPiece -> sb.append("<T>")
                        is GenEvent.Failed -> throw AssertionError(ev.diag)
                        else -> Unit
                    }
                }
            heads_all += sb.toString(); android.util.Log.i("WayProbe", "turn" + i + " len=" + sb.length + " ans=" + sb.toString().take(90))
            hist.add(ChatMessage("assistant", sb.toString()))
        }
        rt.unloadActive()
        org.junit.Assert.assertTrue("way: identity lost in multi-turn",
            heads_all.contains("Edge0"))
    }

    // fork dissecting table: mode=ui replays the stored UI text as the assistant turn;
// mode=native replays lastAssistantNative ground truth
    @Test
    fun reuseForkBisect() = runBlocking {
        val mode = androidx.test.platform.app.InstrumentationRegistry.getArguments()
            .getString("mode") ?: "ui"
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val want = androidx.test.platform.app.InstrumentationRegistry.getArguments().getString("model") ?: "edge0-35b.gguf"
        val model = sequenceOf(
            File(ctx.getDir("models", 0), want),
            File(ctx.filesDir, "models/$want"),
        ).firstOrNull { it.isFile } ?: return@runBlocking
        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        val hist = mutableListOf<ChatMessage>()
        val qs = listOf("Introduce three advantages of Mixture-of-Experts in detail.", "thanks")
        for ((i, q) in qs.withIndex()) {
            hist.add(ChatMessage("user", q))
            val sb = StringBuilder()
            rt.submit("bs$i", hist.toList(), true, GenParams(temperature = 0f, maxNewTokens = if (i == 0) 160 else 16))
                .toList().forEach { ev ->
                    when (ev) {
                        is GenEvent.TextPiece -> sb.append(ev.s)
                        is GenEvent.ThinkPiece -> sb.append(ev.s)
                        is GenEvent.Failed -> throw AssertionError(ev.diag)
                        else -> Unit
                    }
                }
            val content = if (mode == "native") (rt.lastAssistantNative() ?: sb.toString()) else sb.toString()
            hist.add(ChatMessage("assistant", content))
            android.util.Log.i("ReuseProbe", "mode=$mode round$i uiLen=${sb.length} sentLen=${content.length} m=${rt.lastMetrics.value?.let { "prompt=${it.promptTokens} reused=${it.prefix_reused_tokens} prefill_ms=${it.prefillMs}" }}")
        }
        rt.unloadActive()
    }

    // stack-smash regression: long ASCII pieces (>16 bytes) from byte-fallback must not crash
    @Test
    fun longAsciiPieceSurvival() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val want = "edge0-8b.gguf"
        val model = sequenceOf(
            File(ctx.getDir("models", 0), want),
            File(ctx.filesDir, "models/$want"),
        ).firstOrNull { it.isFile } ?: return@runBlocking
        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        var count = 0
        rt.submit("t-ascii", listOf(
            ChatMessage("user", "Repeat exactly, no thinking: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB 123456789012345678901234567890 CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"),
        ), true, GenParams(temperature = 0f, maxNewTokens = 192)).toList()
            .forEach { ev -> when (ev) {
                is GenEvent.TextPiece -> count += ev.s.length
                is GenEvent.ThinkPiece -> count += ev.s.length
                is GenEvent.Failed -> throw AssertionError(ev.diag)
                else -> Unit
            } }
        android.util.Log.i("LlamaRT", "ascii survival chars=$count")
        assertTrue("no output", count > 20)
        rt.unloadActive()
    }

    // 35B off-mode leak measurement: the empty-pair path is live (parity with the
    // reference template); this case quantifies residual drift via thinkChars and
    // driftFirst. Assertions guard gross shape only; drift is read from LeakProbe logs.
    @Test
    fun thinkOffLeak35b() = runBlocking {
        val ctx = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().targetContext
        val want = "edge0-35b.gguf"
        val model = sequenceOf(File(ctx.getDir("models", 0), want), File(ctx.filesDir, "models/$want"))
            .firstOrNull { it.isFile } ?: return@runBlocking
        val rt = LlamaRuntime()
        rt.ensureLoaded(model.absolutePath)
        val qs = listOf("What is the capital of France? One word.",
                        "If you pick 3, 3, 8, 8 once each with +,-,*,/ to reach 24, give the expression directly.",
                        "Name one prime number larger than 90 and smaller than 100.")
        for ((i, q) in qs.withIndex()) {
            val sb = StringBuilder(); var thinkChars = 0; var textChars = 0; var firstKind = "?"
            rt.submit("leak$i", listOf(ChatMessage("user", q)), false,
                      GenParams(temperature = 0f, maxNewTokens = 48)).toList().forEach { ev ->
                when (ev) {
                    is GenEvent.TextPiece -> { if (firstKind == "?") firstKind = "T"; textChars += ev.s.length; sb.append(ev.s) }
                    is GenEvent.ThinkPiece -> { if (firstKind == "?") firstKind = "K"; thinkChars += ev.s.length; sb.append("<T>").append(ev.s) }
                    is GenEvent.Failed -> throw AssertionError(ev.diag)
                    else -> Unit
                }
            }
            android.util.Log.i("LeakProbe", "q$i first=$firstKind think=$thinkChars text=$textChars out=" +
                sb.toString().take(70).replace("\n", " "))
            org.junit.Assert.assertTrue("leak probe empty q$i", textChars + thinkChars > 3)
        }
        rt.unloadActive()
    }
}
