// runtime/HistoryWindow.kt - multi-turn context assembly with a character budget.
// The budget is conservative (mixed CJK/Latin ~1.5 chars/token against a 4096-token
// context); the hard overflow gate is the engine itself (INVALID_ARG with a
// "prompt tokens=" diagnostic) and callers retry with a tighter budget.
// Assistant history re-enters the prompt wrapped in think tag pairs at render time.
package dev.edge0.runtime.app.runtime

import dev.edge0.runtime.engine.ChatMessage

/** Minimal persisted-message shape (Room entities map to this).
 *  genIds = assistant token-id sidecar: lives and dies with its row. */
data class StoredMessage(val role: String, val content: String, val thinking: String? = null,
                         val genIds: IntArray? = null)

/** Sidecar serialization: IntArray to int32-LE BLOB (Room column); empty becomes null. */
fun encodeIds(ids: IntArray?): ByteArray? {
    if (ids == null || ids.isEmpty()) return null
    val bb = java.nio.ByteBuffer.allocate(ids.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    ids.forEach { bb.putInt(it) }
    return bb.array()
}

/** Sidecar deserialization (length not a multiple of 4, or empty, falls back to text). */
fun decodeIds(b: ByteArray?): IntArray? {
    if (b == null || b.isEmpty() || b.size % 4 != 0) return null
    val bb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    return IntArray(b.size / 4) { bb.int }
}

object HistoryWindow {
    const val DEFAULT_BUDGET_CHARS = 10000

    /** turns old to new (user/assistant alternating, last is the user question); system first when non-blank. */
    fun build(systemPrompt: String?, turns: List<StoredMessage>,
              budgetChars: Int = DEFAULT_BUDGET_CHARS): List<ChatMessage> {
        val msgs = ArrayList<ChatMessage>(turns.size + 1)
        var used = 0
        if (!systemPrompt.isNullOrBlank()) {
            msgs.add(ChatMessage("system", systemPrompt))
            used += systemPrompt.length
        }
        // poison-turn filter: blank assistant turns without ids must not reach the
        // model - replaying "I answered nothing" derails every later turn. Dropped from
        // the model-facing context only; the UI record is untouched.
        val clean = turns.filterNot { it.role == "assistant" && it.content.isBlank() && it.genIds?.isEmpty() != false }
        val picked = ArrayList<StoredMessage>()
        for (t in clean.asReversed()) {
            val cost = t.content.length + (t.thinking?.length ?: 0) + 12
            // the current question is always kept, even over budget; engine overflow diagnostic triggers retry
            if (picked.isNotEmpty() && used + cost > budgetChars) break
            picked.add(t)
            used += cost
        }
        picked.reverse()
        for (t in picked) {
            // Keep the engine transcript exact while the UI still stores and
            // renders thinking separately. The native renderer handles the
            // family-specific newline/empty-pair form.
            val engineContent = if (t.role == "assistant" && !t.thinking.isNullOrEmpty()) {
                "<think>" + t.thinking + "</think>" + t.content
            } else t.content
            msgs.add(ChatMessage(t.role, engineContent, t.genIds))
        }
        return msgs
    }

    /** Tightened budget for the overflow retry (caller catches the INVALID_ARG diagnostic). */
    fun tighterBudget(budgetChars: Int): Int = (budgetChars / 2).coerceAtLeast(400)
}
