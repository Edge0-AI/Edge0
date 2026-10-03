// data/ChatRepository.kt - seam between UI and persistence. A fake implementation
// can be injected so UI walkthroughs run without the engine or disk.
package dev.edge0.runtime.app.data

import dev.edge0.runtime.app.runtime.StoredMessage
import dev.edge0.runtime.app.runtime.decodeIds
import java.util.UUID
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

/** Message as the UI sees it (includes persistence status). */
data class MessageView(
    val id: String,
    val role: String,
    val content: String,
    val thinking: String?,
    val thinkingMs: Long?,
    val status: String,
    val createdAt: Long,
    val newTokens: Int?,
    val decodeTokS: Double?,
    val ttftMs: Long? = null,
    val prefillTokS: Double? = null,
    val memBytes: Long? = null,
)

data class ThreadSummary(val id: String, val title: String?, val updatedAt: Long,
                         val modelId: String)

interface ChatRepository {
    fun observeThreads(): Flow<List<ThreadSummary>>
    fun observeMessages(threadId: String): Flow<List<MessageView>>
    /** Creates the thread only with the first message (no empty entries). Returns threadId. */
    suspend fun startThreadWithUserMessage(modelId: String, userText: String): String
    /** Appends a user message to an existing thread and bumps updatedAt. */
    suspend fun addUserMessage(threadId: String, userText: String)
    /** Inserts a streaming placeholder assistant message, returns its id (finalize overwrites). */
    suspend fun beginAssistant(threadId: String, modelId: String): String
    suspend fun finalizeAssistant(id: String, content: String, thinking: String?,
                                  thinkingMs: Long?, status: String,
                                  promptTokens: Int?, newTokens: Int?, decodeTokS: Double?,
                                  genTokens: ByteArray? = null,
                                  ttftMs: Long? = null, prefillTokS: Double? = null,
                                  memBytes: Long? = null)
    suspend fun renameThread(id: String, title: String)
    suspend fun deleteThread(id: String)
    /** Editing a user message truncates everything after it (incl. old answer). */
    suspend fun editUserMessage(id: String, newText: String)
    suspend fun deleteMessage(id: String)
    suspend fun markStreamingInterrupted()
    /** Thread history (old to new, without the trailing user turn the caller appends). */
    suspend fun history(threadId: String): List<StoredMessage>
    /** Full view with id/timestamps (used by regenerate to locate the last assistant row). */
    suspend fun historyWithIds(threadId: String): List<MessageView>
    /** Deletes all messages at/after the given timestamp (regenerate truncation). */
    suspend fun deleteFrom(threadId: String, fromTs: Long)
}

class RoomChatRepository(private val db: AppDatabase) : ChatRepository {
    override fun observeThreads(): Flow<List<ThreadSummary>> =
        db.threads().observeAll().map { l -> l.map { ThreadSummary(it.id, it.title, it.updatedAt, it.modelId) } }

    override fun observeMessages(threadId: String): Flow<List<MessageView>> =
        db.messages().observeFor(threadId).map { l ->
            l.map { MessageView(it.id, it.role, it.content, it.thinking, it.thinkingMs,
                                it.status, it.createdAt, it.newTokens, it.decodeTokS,
                                it.ttftMs, it.prefillTokS, it.memBytes) }
        }

    override suspend fun startThreadWithUserMessage(modelId: String, userText: String): String {
        val now = System.currentTimeMillis()
        val tid = UUID.randomUUID().toString()
        db.threads().upsert(ThreadEntity(tid, userText.take(60), modelId, now, now))
        db.messages().insert(MessageEntity(UUID.randomUUID().toString(), tid, "user",
                                           userText, null, null, MsgStatus.OK, modelId, now,
                                           null, null, null))
        return tid
    }

    override suspend fun beginAssistant(threadId: String, modelId: String): String {
        val id = UUID.randomUUID().toString()
        db.messages().insert(MessageEntity(id, threadId, "assistant", "", null, null,
                                           MsgStatus.STREAMING, modelId,
                                           System.currentTimeMillis(), null, null, null))
        return id
    }

    override suspend fun addUserMessage(threadId: String, userText: String) {
        val now = System.currentTimeMillis()
        db.messages().insert(MessageEntity(UUID.randomUUID().toString(), threadId, "user",
                                           userText, null, null, MsgStatus.OK, "", now,
                                           null, null, null))
        db.threads().touch(threadId, null, now)
    }

    override suspend fun finalizeAssistant(id: String, content: String, thinking: String?,
                                           thinkingMs: Long?, status: String,
                                           promptTokens: Int?, newTokens: Int?,
                                           decodeTokS: Double?, genTokens: ByteArray?,
                                           ttftMs: Long?, prefillTokS: Double?, memBytes: Long?) {
        db.messages().finalizeAssistant(id, content, thinking, thinkingMs, status,
                                        promptTokens, newTokens, decodeTokS, genTokens,
                                        ttftMs, prefillTokS, memBytes)
    }

    override suspend fun renameThread(id: String, title: String) =
        db.threads().touch(id, title, System.currentTimeMillis())

    override suspend fun deleteThread(id: String) = db.threads().delete(id)

    override suspend fun editUserMessage(id: String, newText: String) {
        val msg = db.messages().byId(id) ?: return
        db.messages().update(msg.copy(content = newText))
        db.messages().deleteFrom(msg.threadId, msg.createdAt + 1) // truncate everything after (incl. old answer)
        // editing the first user message renames the thread (title = first user message)
        if (db.messages().olderUserCount(msg.threadId, msg.createdAt) == 0) {
            db.threads().touch(msg.threadId, newText.take(60), System.currentTimeMillis())
        }
    }

    override suspend fun deleteMessage(id: String) = db.messages().deleteById(id)

    override suspend fun markStreamingInterrupted() = db.messages().markStreamingInterrupted()

    override suspend fun history(threadId: String): List<StoredMessage> =
        db.messages().observeFor(threadId).first()
            .map { StoredMessage(it.role, it.content, it.thinking, decodeIds(it.genTokens)) }

    override suspend fun historyWithIds(threadId: String): List<MessageView> =
        db.messages().observeFor(threadId).first()
            .map { MessageView(it.id, it.role, it.content, it.thinking, it.thinkingMs,
                               it.status, it.createdAt, it.newTokens, it.decodeTokS,
                               it.ttftMs, it.prefillTokS, it.memBytes) }

    override suspend fun deleteFrom(threadId: String, fromTs: Long) =
        db.messages().deleteFrom(threadId, fromTs)
}
