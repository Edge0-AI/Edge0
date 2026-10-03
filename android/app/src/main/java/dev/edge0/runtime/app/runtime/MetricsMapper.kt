// runtime/MetricsMapper.kt - typed mapping of engine metrics (append-only schema;
// unknown fields ignored, missing fields default to the N/A display semantics).
package dev.edge0.runtime.app.runtime

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class RouteMetrics(
    val mode: String = "",        // predicted|exact (audit echo, prevents track mixing)
    val feature: String = "",     // executed|teacher|n/a
    @SerialName("dec_steps") val decSteps: Long = 0,
    @SerialName("pred_used") val predUsed: Long = 0,
    @SerialName("route_set_size") val routeSetSize: Int = 0,
    val agree: Double? = null,
)

@Serializable
data class GpuMetrics(
    val probed: Boolean = false,
    val ok: Boolean = false,
    val active: Boolean = false,  // whether this request used the GPU (prefill window)
    val policy: Int = 0,
    val device: String? = null,
    val why: String = "",         // probe conclusion echo (ok / dlopen:.. / policy-cpu ...)
    @SerialName("cache_hit") val cacheHit: Boolean = false,
)

@Serializable
data class ConfigEcho(
    @SerialName("clip_value") val clipValue: Double = 0.0, // per-layer sanitize clamp (0 = off)
    @SerialName("bundle_mode") val bundleMode: String = "",
    @SerialName("production_top_k") val productionTopK: Int = 0,
)

@Serializable
data class ParamsEcho(
    @SerialName("repetition_penalty") val repPenalty: Double = 0.0,
    @SerialName("first_token_greedy") val firstTokenGreedy: Int = 0,
)

@Serializable
data class PoolMetrics(
    val resident_bytes: Long = 0,
    val hits: Long = 0,
    val misses: Long = 0,
    val prefetch_used: Long = 0,
    val prefetch_mispredict: Long = 0,
    val evictions: Long = 0,
    val stall_ns: Long = 0,
    val flash_bytes: Long = 0,
    val slots_used: Long = 0,
    val slots_total: Long = 0,
)

@Serializable
data class RequestMetrics(
    val schema: Int = 0,
    @SerialName("prompt_tokens") val promptTokens: Int = 0,
    val prefix_reused: Boolean = false,
    val prefix_reused_tokens: Int = 0,
    @SerialName("new_tokens") val newTokens: Int = 0,
    @SerialName("prefill_ms") val prefillMs: Long = 0,
    @SerialName("first_token_ms") val firstTokenMs: Long = 0,
    @SerialName("decode_ms") val decodeMs: Long = 0,
    @SerialName("prefill_tok_s") val prefillTokS: Double = 0.0,
    @SerialName("decode_tok_s") val decodeTokS: Double = 0.0,
    @SerialName("peak_rss_bytes") val peakRssBytes: Long = 0,
    @SerialName("chat_turns") val chatTurns: Int = 0,
    @SerialName("thinking_on") val thinkingOn: Boolean = true,
    val cancelled: Boolean = false,
    val error: String = "",
    // additive echo section: missing fields on older engines fall back to N/A defaults
    val config: ConfigEcho? = null,
    val params: ParamsEcho? = null,
    val route: RouteMetrics? = null,
    val pool: PoolMetrics? = null,
    val gpu: GpuMetrics? = null,
) {
    val hitRate: Double? get() = pool?.let {
        val t = it.hits + it.misses
        if (t > 0) it.hits.toDouble() / t else null
    }
}

object MetricsMapper {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /** Parse failure / empty input returns null (UI shows N/A, never crashes). */
    fun parse(raw: String?): RequestMetrics? {
        if (raw.isNullOrBlank()) return null
        return try {
            json.decodeFromString(RequestMetrics.serializer(), raw)
        } catch (_: Exception) {
            null
        }
    }
}
