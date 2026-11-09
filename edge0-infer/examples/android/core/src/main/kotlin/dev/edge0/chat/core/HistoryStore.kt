package dev.edge0.chat.core

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

fun jsonString(value: String): String = buildString {
    append('"'); value.forEach { c -> when (c) {
        '"' -> append("\\\""); '\\' -> append("\\\\"); '\n' -> append("\\n"); '\r' -> append("\\r"); '\t' -> append("\\t")
        else -> if (c < ' ') append("\\u" + c.code.toString(16).padStart(4, '0')) else append(c)
    } }; append('"')
}
fun archiveJson(archive: SavedConversation): String {
    fun metric(m: Metrics?) = if (m == null) "null" else """{"generatedCount":${m.generatedCount},"firstTokenSeconds":${m.firstTokenSeconds},"inputTokens":${m.inputTokens},"reusedTokens":${m.reusedTokens},"decodeTokensPerSecond":${m.decodeTokensPerSecond},"memoryPeakBytes":${m.memoryPeakBytes}}"""
    val messages = archive.messages.joinToString(",") { m -> """{"role":${jsonString(m.role)},"content":${jsonString(m.content)},"reasoning":${jsonString(m.reasoning)},"metrics":${jsonString(m.metrics)},"stopped":${m.stopped},"sourceMetrics":${metric(m.sourceMetrics)}}""" }
    return """{"id":${jsonString(archive.id)},"updatedMillis":${archive.updatedMillis},"systemPrompt":${jsonString(archive.systemPrompt)},"thinking":${archive.thinking},"messages":[$messages],"modelDirectory":${archive.modelDirectory?.let(::jsonString) ?: "null"}}"""
}
fun parseArchive(source: String): SavedConversation {
    val root = parseJson(source).obj()
    fun metrics(value: Json?): Metrics? {
        val map = (value as? Json.Obj)?.fields ?: return null
        fun n(key: String) = (map[key] as? Json.Num)?.value
        return Metrics(n("generatedCount")?.toLong(), n("firstTokenSeconds")?.toDouble(), n("inputTokens")?.toLong(), n("reusedTokens")?.toLong(), n("decodeTokensPerSecond")?.toDouble(), n("memoryPeakBytes")?.toLong())
    }
    val messages = root.getValue("messages").array().map { value ->
        val m = value.obj(); val role = m.getValue("role").string()
        require(role == "user" || role == "assistant")
        Message(role, m.getValue("content").string(), m.getValue("reasoning").string(), m.getValue("metrics").string(), (m["stopped"] as? Json.Bool)?.value ?: false, metrics(m["sourceMetrics"]))
    }
    require(messages.withIndex().all { (i, m) -> m.role == if (i % 2 == 0) "user" else "assistant" })
    return SavedConversation(root.getValue("id").string(), (root.getValue("updatedMillis") as Json.Num).value.toLong(),
        root.getValue("systemPrompt").string(), (root["thinking"] as? Json.Bool)?.value, messages, (root["modelDirectory"] as? Json.Str)?.value)
}
data class HistoryEntry(val id: String, val archive: SavedConversation?, val error: String? = null)
class ConversationFiles(private val directory: File, private val normalizeJson: (String) -> String = { it }) {
    private fun file(id: String): File { require(Regex("[A-Za-z0-9-]+").matches(id)); return File(directory, "$id.json") }
    fun read(): List<HistoryEntry> = directory.listFiles().orEmpty().filter { it.extension == "json" }.map { source ->
        runCatching { parseArchive(normalizeJson(source.readText())).also { require(it.id == source.nameWithoutExtension) } }
            .fold({ HistoryEntry(it.id, it) }, { HistoryEntry(source.nameWithoutExtension, null, "Unable to open") })
    }.sortedByDescending { it.archive?.updatedMillis ?: Long.MAX_VALUE }
    fun save(archive: SavedConversation) {
        val target = file(archive.id)
        if (target.exists() && runCatching {
            parseArchive(normalizeJson(target.readText())).also { require(it.id == archive.id) }
        }.isFailure) error("Cannot overwrite unreadable conversation")
        directory.mkdirs()
        val temporary = File.createTempFile("conversation-", ".tmp", directory)
        try {
            temporary.writeText(normalizeJson(archiveJson(archive)))
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        } finally { temporary.delete() }
    }
    fun delete(id: String) { check(!file(id).exists() || file(id).delete()) { "Cannot delete conversation" } }
}
