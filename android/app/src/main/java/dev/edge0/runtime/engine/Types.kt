// Shared data types of the runtime layer.
package dev.edge0.runtime.engine

enum class Status(val code: Int) {
    OK(0),
    INSUFFICIENT_STORAGE(1),
    FORMAT_VERSION(2),
    ASSET_CORRUPT(3),
    MEM_BUDGET(4),
    IO(5),
    CANCELLED(6),
    SESSION_EXISTS(7),
    INVALID_ARG(8),
    NOT_IMPLEMENTED(9),
    INTERNAL(100),
    ;
    companion object { fun of(v: Int): Status = entries.firstOrNull { it.code == v } ?: INTERNAL }
}

class EngineException(val status: Status, message: String) : Exception("[$status] $message")

data class GenParams(
    val temperature: Float = 0f,      // <= 0 selects greedy decoding (deterministic)
    val topK: Int = 0,
    val topP: Float = 1f,
    val seed: Long = 0x20260915L,
    val maxNewTokens: Int = 0,
    val repetitionPenalty: Float = 0f,    // 0 or 1.0 = off; > 0 and != 1 applies penalty
    val firstTokenGreedy: Boolean = false, // true = argmax for the first token
)

/** A chat turn. role is one of system/user/assistant; at most one system turn, first.
 *  genIds = assistant token-id sidecar: the exact token ids this assistant turn
 *  stepped through. When present the engine replays the turn from ids (the text
 *  is for the UI only), keeping the prompt byte-identical across turns. */
data class ChatMessage(val role: String, val content: String, val genIds: IntArray? = null)
