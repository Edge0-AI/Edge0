package dev.edge0.chat.core

import java.math.BigDecimal
import java.math.RoundingMode
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

private fun decimal(value: Double, places: Int) = BigDecimal.valueOf(value).setScale(places, RoundingMode.HALF_UP).toPlainString()
fun formatCount(value: Long?): String? = value?.toString()?.reversed()?.chunked(3)?.joinToString(",")?.reversed()
fun formatBytes(value: Long?): String? = value?.let {
    val divisor = if (it < 10_000_000) 1_000_000 else 1_000_000_000
    val places = if (it < 10_000_000) 0 else 2
    "${BigDecimal.valueOf(it).divide(BigDecimal(divisor), places, RoundingMode.HALF_UP)} ${if (places == 0) "MB" else "GB"}"
}
fun formatRate(value: Double?): String? = value?.let { "${decimal(it, 1)} tok/s" }
fun formatPercent(numerator: Long?, denominator: Long?): String? = if (numerator == null || denominator == null || denominator <= 0) null
else "${BigDecimal.valueOf(numerator).multiply(BigDecimal(100)).divide(BigDecimal.valueOf(denominator), 1, RoundingMode.HALF_UP)}%"
fun formatDuration(value: Double?): String? = value?.let {
    val tenths = BigDecimal.valueOf(it).setScale(1, RoundingMode.HALF_UP)
    if (tenths < BigDecimal(60)) "$tenths s" else {
        val seconds = BigDecimal.valueOf(it).setScale(0, RoundingMode.HALF_UP).toLong()
        if (seconds < 3600) "${seconds / 60} min ${seconds % 60} s" else "${seconds / 3600} h ${seconds % 3600 / 60} min"
    }
}
data class ContextUsage(val text: String, val ringPercent: Int?)
fun contextUsage(input: Long? = null, generated: Long? = null, capacity: Long? = null): ContextUsage {
    val used = (input ?: 0) + (generated ?: 0)
    return if (capacity != null && capacity > 0) ContextUsage("${formatCount(used)} / ${formatCount(capacity)} tokens",
        BigDecimal.valueOf(used).multiply(BigDecimal(100)).divide(BigDecimal.valueOf(capacity), 0, RoundingMode.HALF_UP).toInt().coerceIn(0, 100))
    else ContextUsage("${formatCount(used)} tokens", null)
}
enum class MemoryLocation { system, dedicated }
data class DeviceMemory(val currentBytes: Long, val peakBytes: Long, val memory: MemoryLocation)
data class DeviceMemorySummary(val system: String?, val dedicated: String?)
fun summarizeDeviceMemory(devices: Map<String, DeviceMemory>?): DeviceMemorySummary? {
    if (devices.isNullOrEmpty()) return null
    fun group(location: MemoryLocation) = devices.values.filter { it.memory == location }.takeIf { it.isNotEmpty() }?.sumOf { it.currentBytes }?.let(::formatBytes)
    return DeviceMemorySummary(group(MemoryLocation.system) ?: formatBytes(0), group(MemoryLocation.dedicated))
}
data class NativeProgress(
    val phase: String = "idle", val inputTokens: Long = 0, val reusedTokens: Long = 0,
    val computedTokens: Long = 0, val pendingTokens: Long = 0, val generatedTokens: Long = 0,
    val elapsedSeconds: Double? = null, val firstTokenSeconds: Double? = null,
    val decodeElapsedSeconds: Double? = null,
)
data class NativeMemory(
    val limitBytes: Long? = null, val currentBytes: Long? = null, val peakBytes: Long? = null,
    val byPurpose: Map<String, Long> = emptyMap(), val byDevice: Map<String, DeviceMemory> = emptyMap(),
    val physicalBytes: Long? = null, val residentBytes: Long? = null,
)
enum class ThermalTone { ok, warn, error }
data class ThermalDisplay(val label: String, val description: String, val tone: ThermalTone)
private val thermalLevels = listOf(
    ThermalDisplay("Normal", "No heat slowdown", ThermalTone.ok) to setOf("nominal", "none"),
    ThermalDisplay("Warm", "Little or no slowdown", ThermalTone.warn) to setOf("fair", "light"),
    ThermalDisplay("Throttled", "Running slower to cool down", ThermalTone.warn) to setOf("serious", "moderate", "severe"),
    ThermalDisplay("Too hot", "Heavily slowed; let it cool", ThermalTone.error) to setOf("critical"),
    ThermalDisplay("Shutting down", "System is about to shut parts down", ThermalTone.error) to setOf("emergency", "shutdown"),
)
fun thermalDisplay(state: String): ThermalDisplay =
    thermalLevels.firstOrNull { state in it.second }?.first ?: ThermalDisplay(state, "Unrecognized state", ThermalTone.warn)
data class NativeSystem(val totalBytes: Long? = null, val availableBytes: Long? = null, val thermal: String? = null, val appLimitBytes: Long? = null)
data class TurnStatistics(
    val inputTokens: Long = 0, val reusedTokens: Long = 0, val generatedTokens: Long = 0,
    val decodeForwards: Long? = null, val decodeSeconds: Double? = null, val firstTokenSeconds: Double? = null,
    val expertDemand: Long? = null, val expertHits: Long? = null,
)
data class Totals(
    val requests: Long = 0, val inputTokens: Long = 0, val reusedTokens: Long = 0, val generatedTokens: Long = 0,
    val decodeForwards: Long = 0, val decodeSeconds: Double = 0.0,
    val expertDemand: Long = 0, val expertHits: Long = 0,
) {
    fun add(turn: TurnStatistics) = copy(requests = requests + 1, inputTokens = inputTokens + turn.inputTokens,
        reusedTokens = reusedTokens + turn.reusedTokens, generatedTokens = generatedTokens + turn.generatedTokens,
        decodeForwards = decodeForwards + (if (turn.decodeSeconds != null) turn.decodeForwards ?: 0 else 0),
        decodeSeconds = decodeSeconds + (if (turn.decodeForwards != null) turn.decodeSeconds ?: 0.0 else 0.0),
        expertDemand = expertDemand + (turn.expertDemand ?: 0), expertHits = expertHits + (turn.expertHits ?: 0))
    val reuse: String? get() = formatPercent(reusedTokens, inputTokens)
    val hits: String? get() = formatPercent(expertHits, expertDemand)
    val averageDecode: String? get() = if (decodeSeconds > 0) formatRate(decodeForwards / decodeSeconds) else null
}
data class RequestDisplay(val phase: String, val prefill: String?, val decode: String?)
fun requestDisplay(progress: NativeProgress, last: TurnStatistics? = null): RequestDisplay {
    if (progress.phase == "prefill") return RequestDisplay("Prefilling",
        "Computing ${formatCount(progress.pendingTokens + progress.computedTokens)} tokens" + (formatDuration(progress.elapsedSeconds)?.let { " · $it" } ?: ""), null)
    if (progress.phase == "decode") {
        val prefill = listOfNotNull("${formatCount(progress.inputTokens - progress.reusedTokens)} tokens", formatDuration(progress.firstTokenSeconds),
            progress.firstTokenSeconds?.takeIf { it > 0 }?.let { formatRate((progress.inputTokens - progress.reusedTokens) / it) }).joinToString(" · ")
        val elapsed = progress.decodeElapsedSeconds
        val speed = if (elapsed != null && elapsed > 0 && progress.generatedTokens > 0) formatRate((progress.generatedTokens - 1) / elapsed) else null
        return RequestDisplay("Decoding", prefill, listOfNotNull("${formatCount(progress.generatedTokens)} tokens", speed).joinToString(" · "))
    }
    if (last == null) return RequestDisplay("Idle", null, null)
    return RequestDisplay("Idle", listOfNotNull("${formatCount(last.inputTokens - last.reusedTokens)} tokens", formatDuration(last.firstTokenSeconds),
        last.firstTokenSeconds?.takeIf { it > 0 }?.let { formatRate((last.inputTokens - last.reusedTokens) / it) }).joinToString(" · "),
        listOfNotNull("${formatCount(last.generatedTokens)} tokens", if (last.decodeSeconds != null && last.decodeSeconds > 0 && last.decodeForwards != null) formatRate(last.decodeForwards / last.decodeSeconds) else null).joinToString(" · "))
}
/** Scheduling is injected so tests can advance time without sleeping. */
class VisiblePoller(private val schedule: (Long, () -> Unit) -> (() -> Unit), private val query: (done: () -> Unit) -> Unit) {
    private var visible = false
    private var generation = 0
    private var cancel: (() -> Unit)? = null
    private var inFlight = false
    private var queriedBefore = false
    fun setVisible(value: Boolean) {
        if (value == visible) return
        visible = value; generation++; cancel?.invoke(); cancel = null
        if (value && !inFlight) enqueue(generation, if (queriedBefore) 1000 else 0)
    }
    private fun enqueue(version: Int, delay: Long) {
        cancel = schedule(delay) {
            if (visible && generation == version && !inFlight) {
                inFlight = true; queriedBefore = true
                query {
                    inFlight = false
                    if (visible) enqueue(generation, 1000)
                }
            }
        }
    }
}
enum class ThemeMode { System, Light, Dark }
enum class GenerationStyle(val temperature: Double) { Precise(0.2), Balanced(0.7), Creative(1.0) }
data class SavedConversation(
    val id: String, val updatedMillis: Long, val systemPrompt: String, val thinking: Boolean?, val messages: List<Message>, val modelDirectory: String? = null,
) {
    val title: String get() = conversationTitle(messages)
}
fun conversationTitle(messages: List<Message>, length: Int = 32): String {
    val text = messages.firstOrNull { it.role == "user" }?.content?.replace(Regex("\\s+"), " ")?.trim() ?: "New chat"
    val points = text.codePoints().toArray()
    return if (points.size <= length) text else String(points, 0, length) + "…"
}
fun historyGroup(updatedMillis: Long, today: LocalDate, zone: ZoneId): String {
    val day = Instant.ofEpochMilli(updatedMillis).atZone(zone).toLocalDate()
    return when { day >= today -> "Today"; day == today.minusDays(1) -> "Yesterday"; day >= today.minusDays(7) -> "Previous 7 days"; else -> "Earlier" }
}
fun regenerateMessages(messages: List<Message>): List<Message> {
    require(messages.lastOrNull()?.role == "assistant") { "No answer to regenerate" }
    return messages.dropLast(1)
}
fun editMessages(messages: List<Message>, index: Int, content: String): List<Message> {
    require(index in messages.indices && messages[index].role == "user" && content.isNotBlank())
    return messages.take(index) + Message("user", content)
}
fun codeBlocks(content: String): List<String> {
    val blocks = mutableListOf<String>()
    val opening = Regex("(?m)^ {0,3}(`{3,}|~{3,})[^\\n]*(?:\\n|$)")
    var cursor = 0
    while (cursor < content.length) {
        val start = opening.find(content, cursor) ?: break
        val marker = start.groupValues[1]
        val end = Regex("(?m)^ {0,3}" + Regex.escape(marker.first().toString()) + "{${marker.length},}[ \\t]*(?:\\n|$)").find(content, start.range.last + 1)
        blocks += content.substring(start.range.last + 1, end?.range?.first ?: content.length).removeSuffix("\n")
        cursor = end?.range?.last?.plus(1) ?: content.length
    }
    return blocks
}

/** The persisted turn keys make replay and a reset safe across app restarts. */
class CumulativeTotalsStore(private val read: () -> String?, private val write: (String) -> Unit) {
    private var seen: Set<String> = emptySet()
    var totals: Totals = Totals(); private set
    init {
        runCatching {
            val root = read()?.let(::parseJson)?.obj() ?: return@runCatching
            val t = root.getValue("totals").obj()
            fun long(key: String) = (t.getValue(key) as Json.Num).value.toLong()
            totals = Totals(long("requests"),long("inputTokens"),long("reusedTokens"),long("generatedTokens"),long("decodeForwards"),
                (t.getValue("decodeSeconds") as Json.Num).value.toDouble(),long("expertDemand"),long("expertHits"))
            seen = root.getValue("seen").array().map(Json::string).toSet()
        }.onFailure { totals=Totals();seen=emptySet() }
    }
    fun record(turnId: String, turn: TurnStatistics): Boolean {
        if(turnId in seen) return false
        seen=seen+turnId;totals=totals.add(turn);persist();return true
    }
    fun reset() { totals=Totals();persist() }
    private fun persist() {
        val t=totals
        write("""{"totals":{"requests":${t.requests},"inputTokens":${t.inputTokens},"reusedTokens":${t.reusedTokens},"generatedTokens":${t.generatedTokens},"decodeForwards":${t.decodeForwards},"decodeSeconds":${t.decodeSeconds},"expertDemand":${t.expertDemand},"expertHits":${t.expertHits}},"seen":[${seen.joinToString(",",transform=::jsonString)}]}""")
    }
}
