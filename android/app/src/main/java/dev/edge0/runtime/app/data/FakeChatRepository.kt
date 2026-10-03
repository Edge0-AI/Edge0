// data/FakeChatRepository.kt - in-memory implementation for UI walkthroughs (no engine, no disk).
package dev.edge0.runtime.app.data

import dev.edge0.runtime.app.runtime.StoredMessage
import dev.edge0.runtime.app.runtime.decodeIds
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.map

class FakeChatRepository : ChatRepository {
    data class Msg(val id: String, val threadId: String, val role: String, var content: String,
                   var thinking: String?, var status: String, val createdAt: Long,
                   var genTokens: ByteArray? = null)

    private val threads = MutableStateFlow<List<ThreadSummary>>(emptyList())
    private val msgs = MutableStateFlow<List<Msg>>(emptyList())
    private var seq = 0
    private fun nid() = "fx${seq++}"

    override fun observeThreads(): Flow<List<ThreadSummary>> = threads
    override fun observeMessages(threadId: String): Flow<List<MessageView>> =
        msgs.map { l -> l.filter { it.threadId == threadId }.sortedBy { it.createdAt }
            .map { MessageView(it.id, it.role, it.content, it.thinking, null, it.status,
                               it.createdAt, null, null) } }

    override suspend fun startThreadWithUserMessage(modelId: String, userText: String): String {
        val tid = nid()
        val now = System.currentTimeMillis()
        msgs.value += Msg(tid, tid, "user", userText, null, MsgStatus.OK, now)
        threads.value = listOf(ThreadSummary(tid, userText.take(60), now, modelId)) + threads.value
        return tid
    }

    override suspend fun beginAssistant(threadId: String, modelId: String): String {
        val id = nid()
        msgs.value += Msg(id, threadId, "assistant", "", null, MsgStatus.STREAMING,
                          System.currentTimeMillis())
        return id
    }

    override suspend fun addUserMessage(threadId: String, userText: String) {
        val now = System.currentTimeMillis()
        msgs.value += Msg(nid(), threadId, "user", userText, null, MsgStatus.OK, now)
        threads.value = threads.value.map {
            if (it.id == threadId) it.copy(updatedAt = now) else it
        }
    }

    override suspend fun finalizeAssistant(id: String, content: String, thinking: String?,
                                           thinkingMs: Long?, status: String,
                                           promptTokens: Int?, newTokens: Int?,
                                           decodeTokS: Double?, genTokens: ByteArray?,
                                           ttftMs: Long?, prefillTokS: Double?, memBytes: Long?) {
        msgs.value = msgs.value.map {
            if (it.id == id) it.copy(content = content, thinking = thinking, status = status,
                                     genTokens = genTokens)
            else it
        }
    }

    override suspend fun renameThread(id: String, title: String) {
        threads.value = threads.value.map { if (it.id == id) it.copy(title = title) else it }
    }

    override suspend fun deleteThread(id: String) {
        threads.value = threads.value.filter { it.id != id }
        msgs.value = msgs.value.filter { it.threadId != id }
    }

    override suspend fun editUserMessage(id: String, newText: String) {
        val target = msgs.value.firstOrNull { it.id == id } ?: return
        msgs.value = msgs.value.filter { it.createdAt < target.createdAt } +
            target.copy(content = newText)
    }

    override suspend fun deleteMessage(id: String) {
        msgs.value = msgs.value.filter { it.id != id }
    }

    override suspend fun markStreamingInterrupted() {
        msgs.value = msgs.value.map {
            if (it.status == MsgStatus.STREAMING) it.copy(status = MsgStatus.INTERRUPTED) else it
        }
    }

    override suspend fun history(threadId: String): List<StoredMessage> =
        msgs.value.filter { it.threadId == threadId && it.status != MsgStatus.STREAMING }
            .sortedBy { it.createdAt }
            .map { StoredMessage(it.role, it.content, it.thinking, decodeIds(it.genTokens)) }

    override suspend fun historyWithIds(threadId: String): List<MessageView> =
        msgs.value.filter { it.threadId == threadId }.sortedBy { it.createdAt }
            .map { MessageView(it.id, it.role, it.content, it.thinking, null, it.status,
                               it.createdAt, null, null) }

    override suspend fun deleteFrom(threadId: String, fromTs: Long) {
        msgs.value = msgs.value.filter { !(it.threadId == threadId && it.createdAt >= fromTs) }
    }
}
