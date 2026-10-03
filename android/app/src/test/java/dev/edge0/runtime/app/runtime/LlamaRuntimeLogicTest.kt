// LlamaRuntimeLogicTest - pure-logic unit tests (host JVM, no device).
// Device-facing behavior is covered by the instrumented LlamaRuntimeTest suite.
package dev.edge0.runtime.app.runtime

import dev.edge0.runtime.engine.ChatMessage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LlamaRuntimeLogicTest {

    @Test fun metricsParseRoundTrip() {
        val rt = LlamaRuntime()
        val m = rt.parseMetrics(
            "OK gen=64 tps=18.19 prefill_ms=3087 first_ms=182 prompt=17 reused=12")
        checkNotNull(m)
        assertEquals(64, m.newTokens)
        assertEquals(18.19, m.decodeTokS, 1e-6)
        assertEquals(3087, m.prefillMs)
        assertEquals(17, m.promptTokens)
        assertEquals(12, m.prefix_reused_tokens)
        assertTrue(m.prefix_reused)
        // effective prefill rate = new input tokens (17-12=5) over prefill time
        assertEquals(5 * 1000.0 / 3087, m.prefillTokS, 1e-6)
    }

    @Test fun metricsParseCancelledAndGarbage() {
        val rt = LlamaRuntime()
        checkNotNull(rt.parseMetrics("CANCELLED gen=8 tps=2.00 prefill_ms=100 first_ms=10 prompt=5 reused=0"))
        assertNull(rt.parseMetrics("ERR template need=-1"))
    }

    @Test fun historyWindowSystemFirstAndLastTurnKept() {
        val long = "x".repeat(400)
        val turns = (1..30).map { i ->
            if (i % 2 == 1) StoredMessage("user", "q$i $long")
            else StoredMessage("assistant", "a$i $long")
        } + StoredMessage("user", "final question " + long)
        val out = HistoryWindow.build("You are Edge0.", turns, budgetChars = 1000)
        assertEquals("system", out.first().role)                      // system always prepended
        assertEquals("user", out.last().role)                         // current question survives
        assertTrue(out.size < turns.size + 1)                         // budget clipped history
    }

    @Test fun historyWindowDropsBlankAssistantPoisonTurns() {
        // regression sentinel (empty-answer incident): blank assistant turns without
        // genIds must never reach the model - the model would "see itself" answering empty
        val turns = listOf(
            StoredMessage("user", "hi"),
            StoredMessage("assistant", ""),
            StoredMessage("user", "who are you"),
        )
        val out = HistoryWindow.build(null, turns)
        assertTrue(out.none { it.role == "assistant" && it.content.isBlank() })
        assertEquals(2, out.size)
    }
}
