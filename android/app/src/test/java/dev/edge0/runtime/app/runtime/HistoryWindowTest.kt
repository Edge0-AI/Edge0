// HistoryWindowTest - budget trimming, ordering, assistant replay shape.
package dev.edge0.runtime.app.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HistoryWindowTest {
    private val O = ThinkSplitter.OPEN
    private val C = ThinkSplitter.CLOSE

    @Test fun tailWithinBudgetKeptInOrder() {
        val turns = listOf(
            StoredMessage("user", "q1"), StoredMessage("assistant", "a1"),
            StoredMessage("user", "q2"), StoredMessage("assistant", "a2"),
            StoredMessage("user", "q3"),
        )
        val msgs = HistoryWindow.build("sys", turns, budgetChars = 10000)
        assertEquals(listOf("system", "user", "assistant", "user", "assistant", "user"),
            msgs.map { it.role })
        assertEquals("sys", msgs[0].content)
        assertEquals("q3", msgs.last().content)
    }

    @Test fun budgetTrimsOldestNotLast() {
        val big = "W".repeat(100)
        val turns = listOf(
            StoredMessage("user", big), StoredMessage("assistant", big),
            StoredMessage("user", "lastQ"),
        )
        val msgs = HistoryWindow.build(null, turns, budgetChars = 60)
        // the last turn always survives; older turns beyond budget are trimmed
        assertEquals(1, msgs.size)
        assertEquals("lastQ", msgs[0].content)
    }

    @Test fun assistantThinkingReplayedIntoEngineTranscript() {
        // Revisited under the sidecar regime: thinking MUST enter the context -
        // KV prefix reuse requires bit-identical replay of the live tokens.
        val turns = listOf(
            StoredMessage("user", "q1"),
            StoredMessage("assistant", "body", thinking = "reason"),
            StoredMessage("user", "q2"),
        )
        val msgs = HistoryWindow.build(null, turns)
        assertEquals(O + "reason" + C + "body", msgs[1].content)
    }

    @Test fun sidecarIdsRideAlongReplay() {
        val ids = intArrayOf(5, 42, 17)
        val turns = listOf(
            StoredMessage("user", "q"),
            StoredMessage("assistant", "body", genIds = ids),
            StoredMessage("user", "q2"),
        )
        val msgs = HistoryWindow.build(null, turns)
        assertTrue(msgs[1].genIds!!.contentEquals(ids))
        assertEquals("body", msgs[1].content) // no thinking column -> no tags (ids are authoritative)
    }

    @Test fun assistantWithoutThinkingPlainBody() {
        val turns = listOf(
            StoredMessage("user", "q1"),
            StoredMessage("assistant", "plainbody"),
        )
        val msgs = HistoryWindow.build(null, turns)
        assertEquals("plainbody", msgs[1].content)
    }

    @Test fun tighterBudgetFloor() {
        assertEquals(400, HistoryWindow.tighterBudget(700))
        assertTrue(HistoryWindow.tighterBudget(300) >= 400)
    }
}
