// data/Entities.kt - minimal two-table Room schema. Tagged raw text is not stored:
// thinking/body are separate columns; history reconstruction goes through
// HistoryWindow (tag pairs are re-added at render time).
package dev.edge0.runtime.app.data

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "threads", indices = [Index("updatedAt")])
data class ThreadEntity(
    @PrimaryKey val id: String,
    val title: String?,            // truncated first user message (no AI summary)
    val modelId: String,
    val createdAt: Long,
    val updatedAt: Long,
)

/** status: ok | cancelled | error | interrupted (killed process leaves interrupted) */
@Entity(
    tableName = "messages",
    foreignKeys = [ForeignKey(entity = ThreadEntity::class, parentColumns = ["id"],
                              childColumns = ["threadId"], onDelete = ForeignKey.CASCADE)],
    indices = [Index("threadId", "createdAt")],
)
data class MessageEntity(
    @PrimaryKey val id: String,
    val threadId: String,
    val role: String,              // user | assistant
    val content: String,           // tag-free body text
    val thinking: String?,         // extracted reasoning (null = no thinking this turn)
    val thinkingMs: Long?,
    val status: String,            // "streaming" while generating, overwritten at terminal state
    val modelId: String,
    val createdAt: Long,
    val promptTokens: Int?,
    val newTokens: Int?,
    val decodeTokS: Double?,
    val genTokens: ByteArray? = null,  // sidecar: token ids stepped this turn (int32 LE BLOB)
    val ttftMs: Long? = null,          // v3: perceived time-to-first-token, wall clock ms
    val prefillTokS: Double? = null,   // v3: prefill tok/s
    val memBytes: Long? = null,        // v3: expert-pool resident bytes (excludes cacheable mmap)
)

object MsgStatus {
    const val STREAMING = "streaming"
    const val OK = "ok"
    const val CANCELLED = "cancelled"
    const val ERROR = "error"
    const val INTERRUPTED = "interrupted"
}
