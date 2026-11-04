package dev.edge0.chat

import dev.edge0.chat.core.*
import org.json.JSONObject
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

internal data class NativeResult(val counts: LongArray, val timings: DoubleArray, val stopReason: String, val stateValid: Boolean, val error: String?) {
    fun metrics() = Metrics(generatedCount = counts[0], firstTokenSeconds = timings[0].takeIf { it >= 0 }, inputTokens = counts[1], reusedTokens = counts[3], decodeTokensPerSecond = timings[1].takeIf { it >= 0 }, memoryPeakBytes = counts[2].takeIf { it >= 0 })
    fun statistics() = TurnStatistics(counts[1], counts[3], counts[0], counts[4].takeIf { counts[7] != 0L }, timings[2].takeIf { it >= 0 }, timings[0].takeIf { it >= 0 }, counts[5].takeIf { it >= 0 }, counts[6].takeIf { it >= 0 })
}
internal fun interface NativeOutput { fun onEvent(kind: Int, bytes: ByteArray) }
internal class NativeEngine {
    companion object { init { System.loadLibrary("edge0_jni") } }
    // Hold the lock throughout every JNI call: the generation-tagged runtime value
    // does not extend its lifetime. Status and generation share a read lock;
    // lifecycle writes wait for in-flight reads before replacing or deleting handles.
    private val lifecycle = ReentrantReadWriteLock()
    private var handle = nativeCreate()
    private fun live(): Long = handle.also { check(it != 0L) { "Engine is closed" } }
    fun inspect(directory: String): AssetChoice = lifecycle.read {
        val inspection = nativeInspect(directory)
        assetChoice(directory, inspection[0].ifEmpty { null }, inspection[1].ifEmpty { null }).copy(totalBytes = inspection[2].toLongOrNull())
    }
    fun load(directory: String): BooleanArray = lifecycle.write { nativeLoad(live(), directory) }
    fun context(): Long = lifecycle.read { nativeContext(live()) }
    fun unload() = lifecycle.write { nativeUnload(live()) }
    fun prepare() = lifecycle.read { nativePrepare(live()) }
    fun generate(request: ChatRequest, output: NativeOutput): NativeResult = lifecycle.read { nativeGenerate(
        live(), request.messages.map { it.role }.toTypedArray(), request.messages.map { it.content }.toTypedArray(),
        request.messages.map { it.reasoning }.toTypedArray(), request.thinking, request.maxTokens, request.temperature, output) }
    fun cancel() = lifecycle.read { if (handle != 0L) nativeCancel(handle) }
    fun reset() = lifecycle.write { nativeReset(live()) }
    fun close() = lifecycle.write { if (handle != 0L) { nativeClose(handle); handle = 0 } }
    fun progress(): NativeProgress = lifecycle.read {
        val o = JSONObject(nativeProgress(live()))
        NativeProgress(o.optString("phase", "idle"), o.optLong("inputTokens"), o.optLong("reusedTokens"), o.optLong("computedTokens"), o.optLong("pendingTokens"), o.optLong("generatedTokens"), o.double("elapsedSeconds"), o.double("firstTokenSeconds"), o.double("decodeElapsedSeconds"))
    }
    fun snapshot(): NativeMemory = lifecycle.read {
        val o = JSONObject(nativeSnapshot(live())); val purposes = o.optJSONObject("byPurpose"); val devices = o.optJSONObject("byDevice")
        NativeMemory(o.long("limitBytes"), o.long("currentBytes"), o.long("peakBytes"),
            purposes?.keys()?.asSequence()?.associateWith { purposes.getLong(it) }.orEmpty(),
            devices?.keys()?.asSequence()?.associateWith { key -> val d = devices.getJSONObject(key); DeviceMemory(d.getLong("currentBytes"), d.getLong("peakBytes"), MemoryLocation.valueOf(d.getString("memory"))) }.orEmpty(),
            o.long("physicalBytes"), o.long("residentBytes"))
    }
    fun systemStatus(): NativeSystem = lifecycle.read {
        val o = JSONObject(nativeSystemStatus()); NativeSystem(o.long("totalBytes"), o.long("availableBytes"), o.optString("thermal").takeIf { it.isNotEmpty() }, o.long("appLimitBytes"))
    }
    private external fun nativeCreate(): Long
    private external fun nativeInspect(directory: String): Array<String>
    private external fun nativeLoad(handle: Long, directory: String): BooleanArray
    private external fun nativeContext(handle: Long): Long
    private external fun nativeUnload(handle: Long)
    private external fun nativePrepare(handle: Long)
    private external fun nativeCancel(handle: Long)
    private external fun nativeGenerate(handle: Long, roles: Array<String>, contents: Array<String>, reasoning: Array<String>, thinking: Boolean, maxTokens: Int?, temperature: Double, output: NativeOutput): NativeResult
    private external fun nativeReset(handle: Long)
    private external fun nativeClose(handle: Long)
    private external fun nativeProgress(handle: Long): String
    private external fun nativeSnapshot(handle: Long): String
    private external fun nativeSystemStatus(): String
}
private fun JSONObject.long(key: String): Long? = if (has(key) && !isNull(key)) getLong(key) else null
private fun JSONObject.double(key: String): Double? = if (has(key) && !isNull(key)) getDouble(key) else null
