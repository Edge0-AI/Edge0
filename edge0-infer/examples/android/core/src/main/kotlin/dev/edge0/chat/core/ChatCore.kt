package dev.edge0.chat.core

import java.math.BigDecimal
import java.math.RoundingMode

// A small JSON reader keeps shared-data parsing independent of Android and extra libraries.
sealed interface Json {
    data class Obj(val fields: Map<String, Json>) : Json
    data class Arr(val values: List<Json>) : Json
    data class Str(val value: String) : Json
    data class Num(val value: BigDecimal) : Json
    data class Bool(val value: Boolean) : Json
    data object Null : Json
}
fun Json.obj() = (this as Json.Obj).fields
fun Json.array() = (this as Json.Arr).values
fun Json.string() = (this as Json.Str).value
fun parseJson(source: String): Json {
    class Reader {
        var at = 0
        fun whitespace() { while (at < source.length && source[at].isWhitespace()) at++ }
        fun string(): String {
            require(source[at++] == '"')
            return buildString {
                while (at < source.length) {
                    val c = source[at++]
                    if (c == '"') return@buildString
                    if (c != '\\') append(c) else {
                        val escaped = source[at++]
                        append(when (escaped) {
                            '"', '\\', '/' -> escaped
                            'b' -> '\b'; 'f' -> '\u000c'; 'n' -> '\n'; 'r' -> '\r'; 't' -> '\t'
                            'u' -> source.substring(at, at + 4).toInt(16).toChar().also { at += 4 }
                            else -> error("Invalid JSON escape")
                        })
                    }
                }
                error("Unterminated JSON string")
            }
        }
        fun value(): Json {
            whitespace()
            return when (source[at]) {
                '"' -> Json.Str(string())
                '{' -> {
                    at++; whitespace(); val fields = linkedMapOf<String, Json>()
                    if (source[at] != '}') do {
                        whitespace(); val key = string(); whitespace(); require(source[at++] == ':')
                        fields[key] = value(); whitespace()
                        if (source[at] != ',') break
                        at++
                    } while (true)
                    require(source[at++] == '}'); Json.Obj(fields)
                }
                '[' -> {
                    at++; whitespace(); val values = mutableListOf<Json>()
                    if (source[at] != ']') do {
                        values += value(); whitespace()
                        if (source[at] != ',') break
                        at++
                    } while (true)
                    require(source[at++] == ']'); Json.Arr(values)
                }
                't' -> { require(source.startsWith("true", at)); at += 4; Json.Bool(true) }
                'f' -> { require(source.startsWith("false", at)); at += 5; Json.Bool(false) }
                'n' -> { require(source.startsWith("null", at)); at += 4; Json.Null }
                else -> {
                    val start = at
                    while (at < source.length && source[at] in "0123456789.eE+-") at++
                    Json.Num(source.substring(start, at).toBigDecimal())
                }
            }
        }
    }
    val reader = Reader()
    val result = reader.value(); reader.whitespace()
    require(reader.at == source.length) { "Trailing JSON" }
    return result
}

data class AppData(val systemPrompt: String, val suggestions: List<String>) {
    companion object {
        fun parse(text: String): AppData {
            val obj = parseJson(text).obj()
            return AppData(obj.getValue("system_prompt").string(), obj.getValue("suggestions").array().map(Json::string))
        }
    }
}
data class Message(val role: String, val content: String, val reasoning: String = "", val metrics: String = "", val stopped: Boolean = false, val sourceMetrics: Metrics? = null)
data class ChatRequest(val messages: List<Message>, val thinking: Boolean, val maxTokens: Int? = null, val temperature: Double = 0.2)
class Conversation(private val data: AppData) {
    private val completed = mutableListOf<Message>()
    private var activeThinking: Boolean? = null
    fun request(user: String): List<Message> = listOf(Message("system", data.systemPrompt)) + completed + Message("user", user)
    fun prepare(user: String, thinking: Boolean, thinkingSupported: Boolean, maxTokens: Int? = null, style: GenerationStyle = GenerationStyle.Precise): ChatRequest {
        val enabled = activeThinking ?: (thinking && thinkingSupported).also { activeThinking = it }
        return ChatRequest(request(user), enabled, maxTokens, style.temperature)
    }
    fun canChangeThinking(supported: Boolean, busy: Boolean) = supported && !busy && activeThinking == null && completed.isEmpty()
    fun restore(messages: List<Message>, thinking: Boolean?) { completed.clear(); completed.addAll(messages); activeThinking = thinking }
    val lockedThinking: Boolean? get() = activeThinking
    fun complete(user: String, answer: String, reasoning: String, metrics: String = "", stopped: Boolean = false, sourceMetrics: Metrics? = null) {
        completed += Message("user", user)
        completed += Message("assistant", answer, reasoning, metrics, stopped, sourceMetrics)
    }
    fun messages(): List<Message> = completed.toList()
    fun clear() { completed.clear(); activeThinking = null }
}
data class AssetChoice(val directory: String, val modelId: String?, val selectable: Boolean, val reason: String, val totalBytes: Long? = null) {
    val displayName: String get() = modelId?.let { "$it · ${java.io.File(directory).name}" } ?: java.io.File(directory).name
}
fun assetChoice(directory: String, modelId: String?, inspectionError: String?) = AssetChoice(directory, modelId, inspectionError == null, inspectionError.orEmpty())
data class ModelChoices(val available: List<AssetChoice>, val unavailable: List<AssetChoice>)
fun modelChoices(assets: List<AssetChoice>) = ModelChoices(assets.filter { it.selectable }, assets.filterNot { it.selectable })
data class ModelGroups(val loaded: List<AssetChoice>, val available: List<AssetChoice>, val unavailable: List<AssetChoice>)
fun modelGroups(assets: List<AssetChoice>, loadedDirectory: String?) = ModelGroups(
    assets.filter { it.directory == loadedDirectory },
    assets.filter { it.selectable && it.directory != loadedDirectory },
    assets.filter { !it.selectable && it.directory != loadedDirectory },
)

enum class AppPage { chooseModel, chat }
data class ScreenState(
    val modelDirectory: String? = null,
    val chooserRequested: Boolean = false,
    val otherLocationExpanded: Boolean = false,
    val loadingModel: Boolean = false,
) {
    val hasLoadedModel: Boolean get() = modelDirectory != null
    val page: AppPage get() = if (!hasLoadedModel || chooserRequested || loadingModel) AppPage.chooseModel else AppPage.chat
    fun openModels() = copy(chooserRequested = true, otherLocationExpanded = false)
    fun backToChat() = if (hasLoadedModel && !loadingModel) copy(chooserRequested = false) else this
    fun toggleOtherLocation() = copy(otherLocationExpanded = !otherLocationExpanded)
    fun beginModelLoad() = copy(modelDirectory = null, chooserRequested = true, loadingModel = true)
    fun modelLoadFailed() = copy(modelDirectory = null, chooserRequested = true, loadingModel = false)
    fun modelLoaded(directory: String) = ScreenState(modelDirectory = directory)
}
data class ModelPickerPresentation(val choices: ModelChoices) {
    val showPushHelp: Boolean get() = choices.available.isEmpty()
}
data class ReasoningDisclosure(val expanded: Boolean = true) {
    fun toggle() = copy(expanded = !expanded)
    fun visible(reasoning: String, thinkingEnabled: Boolean) = thinkingEnabled && reasoning.isNotEmpty()
}
data class ReasoningBlocks(val states: Map<Int, ReasoningDisclosure> = emptyMap()) {
    fun forMessage(index: Int) = states[index] ?: ReasoningDisclosure()
    fun toggle(index: Int) = copy(states = states + (index to forMessage(index).toggle()))
}
enum class SwitchTone { primary, neutral }
data class ThinkingSwitchPresentation(val checked: Boolean, val enabled: Boolean) {
    val tone: SwitchTone get() = if (checked) SwitchTone.primary else SwitchTone.neutral
    val stateText: String get() = if (checked) "On" else "Off"
    val opacity: Float get() = if (enabled) 1f else 0.45f
}

enum class RequestPhase { idle, generating, stopping }
data class RequestFeedback(val phase: RequestPhase = RequestPhase.idle) {
    val active: Boolean get() = phase != RequestPhase.idle
    val stopping: Boolean get() = phase == RequestPhase.stopping
    val buttonLabel: String get() = when (phase) { RequestPhase.idle -> "Send"; RequestPhase.generating -> "Stop"; RequestPhase.stopping -> "Stopping…" }
    fun start() = RequestFeedback(RequestPhase.generating)
    fun requestStop() = if (phase == RequestPhase.generating) RequestFeedback(RequestPhase.stopping) else this
    fun finish() = RequestFeedback()
}
data class StreamFragments(val answer: String, val reasoning: String)
class StreamBuffer {
    private val answer = StringBuilder()
    private val reasoning = StringBuilder()
    private var scheduled = false
    @Synchronized fun append(kind: Int, text: String): Boolean {
        if (kind == 1) reasoning.append(text) else answer.append(text)
        val first = !scheduled
        scheduled = true
        return first
    }
    @Synchronized fun drain(): StreamFragments {
        val output = StreamFragments(answer.toString(), reasoning.toString())
        answer.clear(); reasoning.clear(); scheduled = false
        return output
    }
}

data class Metrics(
    val generatedCount: Long? = null,
    val firstTokenSeconds: Double? = null,
    val inputTokens: Long? = null,
    val reusedTokens: Long? = null,
    val decodeTokensPerSecond: Double? = null,
    val memoryPeakBytes: Long? = null,
)
fun formatMetrics(m: Metrics): String {
    fun fixed(value: Double, places: Int) = BigDecimal.valueOf(value).setScale(places, RoundingMode.HALF_UP).toPlainString()
    return buildList {
        m.generatedCount?.let { add("$it tokens") }
        m.firstTokenSeconds?.let { add("TTFT ${fixed(it, 2)} s") }
        if (m.inputTokens != null && m.reusedTokens != null && m.inputTokens > m.reusedTokens && m.reusedTokens >= 0 && m.firstTokenSeconds != null && m.firstTokenSeconds > 0)
            add("prefill ${fixed((m.inputTokens - m.reusedTokens) / m.firstTokenSeconds, 1)} tok/s")
        m.decodeTokensPerSecond?.let { add("decode ${fixed(it, 1)} tok/s") }
        m.memoryPeakBytes?.let { add("peak ${BigDecimal.valueOf(it).divide(BigDecimal(1000000), 0, RoundingMode.HALF_UP)} MB") }
    }.joinToString(" \u00b7 ")
}

fun protectMath(source: String): String {
    val out = StringBuilder()
    var i = 0
    fun escaped(at: Int): Boolean {
        var j = at - 1
        while (j >= 0 && source[j] == '\\') j--
        return (at - 1 - j) % 2 == 1
    }
    fun wrap(raw: String, block: Boolean, start: Int, end: Int): String {
        val longest = Regex("`+").findAll(raw).maxOfOrNull { it.value.length } ?: 0
        val ticks = "`".repeat(maxOf(if (block) 3 else 1, longest + 1))
        if (!block) return "$ticks$raw$ticks"
        val before = if (start > 0 && source[start - 1] != '\n') "\n" else ""
        val after = if (end < source.length && source[end] != '\n') "\n" else ""
        return "${before}${ticks}text\n$raw\n$ticks$after"
    }
    val openingFence = Regex("(?m)^ {0,3}(`{3,}|~{3,})[^\\n]*(?:\\n|$)")
    while (i < source.length) {
        val fence = openingFence.find(source, i)?.takeIf { it.range.first == i }
        if (fence != null) {
            val marker = fence.groupValues[1]
            val closingFence = Regex("(?m)^ {0,3}" + Regex.escape(marker.first().toString()) + "{${marker.length},}[ \\t]*(?:\\n|$)")
            val close = closingFence.find(source, fence.range.last + 1)
            val end = close?.range?.last?.plus(1) ?: source.length
            out.append(source.substring(i, end)); i = end; continue
        }
        if (source[i] == '`' && !escaped(i)) {
            var endTicks = i
            while (endTicks < source.length && source[endTicks] == '`') endTicks++
            val markerLength = endTicks - i
            var cursor = endTicks
            var end = source.length
            while (cursor < source.length) {
                val runStart = source.indexOf('`', cursor)
                if (runStart < 0) break
                var runEnd = runStart
                while (runEnd < source.length && source[runEnd] == '`') runEnd++
                // Inside code spans, backslashes do not escape a closing backtick run.
                if (runEnd - runStart == markerLength) { end = runEnd; break }
                cursor = runEnd
            }
            out.append(source.substring(i, end)); i = end; continue
        }
        val pair = when {
            source.startsWith("$$", i) && !escaped(i) -> "$$" to "$$"
            source.startsWith("\\(", i) && !escaped(i) -> "\\(" to "\\)"
            source.startsWith("\\[", i) && !escaped(i) -> "\\[" to "\\]"
            source[i] == '$' && !escaped(i) && !Regex("^\\$[0-9][0-9,.]*(?=\\s|$)").containsMatchIn(source.substring(i).takeWhile { it != '\n' }) -> "$" to "$"
            else -> null
        }
        if (pair != null) {
            var close = source.indexOf(pair.second, i + pair.first.length)
            while (close >= 0 && escaped(close)) close = source.indexOf(pair.second, close + pair.second.length)
            if (close >= 0) {
                val end = close + pair.second.length
                out.append(wrap(source.substring(i, end), pair.first == "$$" || pair.first == "\\[", i, end))
                i = end; continue
            }
        }
        out.append(source[i++])
    }
    return out.toString()
}
/** The answer being generated shows dots until text appears and hides its actions until generation ends. */
data class AssistantState(val pending: Boolean, val actions: Boolean)
fun assistantState(isLast: Boolean, generating: Boolean, hasContent: Boolean, showsReasoning: Boolean): AssistantState {
    val current = isLast && generating
    return AssistantState(current && !hasContent && !showsReasoning, !current)
}
