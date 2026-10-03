// data/Daos.kt - thread/message DAOs (flat list ordered by updatedAt desc).
package dev.edge0.runtime.app.data

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface ThreadDao {
    @Query("SELECT * FROM threads ORDER BY updatedAt DESC")
    fun observeAll(): Flow<List<ThreadEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(thread: ThreadEntity)

    @Query("UPDATE threads SET title = COALESCE(:title, title), updatedAt = :ts WHERE id = :id")
    suspend fun touch(id: String, title: String?, ts: Long)

    @Query("DELETE FROM threads WHERE id = :id")
    suspend fun delete(id: String)
}

@Dao
interface MessageDao {
    @Query("SELECT * FROM messages WHERE threadId = :threadId ORDER BY createdAt ASC")
    fun observeFor(threadId: String): Flow<List<MessageEntity>>

    @Insert
    suspend fun insert(msg: MessageEntity)

    @Update
    suspend fun update(msg: MessageEntity)

    @Query("UPDATE messages SET content=:content, thinking=:thinking, thinkingMs=:thinkingMs, " +
           "status=:status, promptTokens=:pt, newTokens=:nt, decodeTokS=:dts, " +
           "genTokens=:genTokens, ttftMs=:ttft, prefillTokS=:pfs, memBytes=:mem WHERE id=:id")
    suspend fun finalizeAssistant(id: String, content: String, thinking: String?,
                                  thinkingMs: Long?, status: String,
                                  pt: Int?, nt: Int?, dts: Double?, genTokens: ByteArray?,
                                  ttft: Long?, pfs: Double?, mem: Long?)

    /** Cold-start sweep: rows left streaming by a killed process become interrupted. */
    @Query("UPDATE messages SET status='interrupted' WHERE status='streaming'")
    suspend fun markStreamingInterrupted()

    /** Edit/regenerate truncation: delete every message at/after the given created-at. */
    @Query("DELETE FROM messages WHERE threadId=:threadId AND createdAt >= :fromTs")
    suspend fun deleteFrom(threadId: String, fromTs: Long)

    @Query("SELECT * FROM messages WHERE id = :id")
    suspend fun byId(id: String): MessageEntity?

    @Query("SELECT COUNT(*) FROM messages WHERE threadId=:threadId AND createdAt < :ts " +
           "AND role='user'")
    suspend fun olderUserCount(threadId: String, ts: Long): Int

    @Query("DELETE FROM messages WHERE id = :id")
    suspend fun deleteById(id: String)
}
