// data/SettingsStore.kt - DataStore preferences snapshot (appearance / assistant / sampling / toggles).
// Defaults follow the upstream sampling quartet; installs without a stored key
// fall back to the current defaults.
package dev.edge0.runtime.app.data

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dev.edge0.runtime.app.ui.theme.ThemeMode
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore("edge0_settings")

data class AppSettings(
    val themeMode: ThemeMode = ThemeMode.DARK,   // dark by default (does not follow system)
    val fontScale: Float = 1.0f,          // 0.85..1.3
    val systemPrompt: String = "You're Edge0, an on-device AI assistant.",
    val useSystemPrompt: Boolean = true,    // explicit toggle; empty text also means off
    val temperature: Float = 0.2f,        // precise default (control lives in Settings)
    val topK: Int = 64,
    val topP: Float = 0.95f,
    val repetitionPenalty: Float = 1.1f,  // aligned with upstream; 1.0 = off
    val maxNewTokens: Int = 0,            // 0 = app default 1024 (reasoning models need headroom)
    val seed: Long = 0x20260915L,
    val enableThinking: Boolean = false,    // master thinking toggle: drives both the
    //   model-side template instruction and UI block visibility;
    //   storage key kept as show_thinking for backward compatibility
    val metricsOverlay: Boolean = false,  // off by default
    val activeModelDir: String = "",
)

object SettingsKeys {
    val THEME = stringPreferencesKey("theme_mode")
    val FONT_SCALE = floatPreferencesKey("font_scale")
    val SYSTEM_PROMPT = stringPreferencesKey("system_prompt")
    val USE_SYS_PROMPT = booleanPreferencesKey("use_system_prompt")
    val TEMPERATURE = floatPreferencesKey("temperature")
    val TOP_K = intPreferencesKey("top_k")
    val TOP_P = floatPreferencesKey("top_p")
    val REP_PENALTY = floatPreferencesKey("repetition_penalty")
    val MAX_NEW = intPreferencesKey("max_new_tokens")
    val SEED = longPreferencesKey("seed")
    val SHOW_THINKING = booleanPreferencesKey("show_thinking")
    val OVERLAY = booleanPreferencesKey("metrics_overlay")
    val ACTIVE_MODEL = stringPreferencesKey("active_model_dir")
}

class SettingsStore(private val context: Context) {

    val flow: Flow<AppSettings> = context.dataStore.data.map { p ->
        val d = AppSettings()
        AppSettings(
            themeMode = p[SettingsKeys.THEME]?.let { runCatching { ThemeMode.valueOf(it) }.getOrNull() } ?: d.themeMode,
            fontScale = p[SettingsKeys.FONT_SCALE] ?: d.fontScale,
            systemPrompt = p[SettingsKeys.SYSTEM_PROMPT] ?: d.systemPrompt,
            useSystemPrompt = p[SettingsKeys.USE_SYS_PROMPT] ?: d.useSystemPrompt,
            temperature = p[SettingsKeys.TEMPERATURE] ?: d.temperature,
            topK = p[SettingsKeys.TOP_K] ?: d.topK,
            topP = p[SettingsKeys.TOP_P] ?: d.topP,
            repetitionPenalty = p[SettingsKeys.REP_PENALTY] ?: d.repetitionPenalty,
            maxNewTokens = p[SettingsKeys.MAX_NEW] ?: d.maxNewTokens,
            seed = p[SettingsKeys.SEED] ?: d.seed,
            enableThinking = p[SettingsKeys.SHOW_THINKING] ?: d.enableThinking,
            metricsOverlay = p[SettingsKeys.OVERLAY] ?: d.metricsOverlay,
            activeModelDir = p[SettingsKeys.ACTIVE_MODEL] ?: d.activeModelDir,
        )
    }

    suspend fun update(block: (AppSettings) -> AppSettings) {
        val current = runCatching { flow.first() }.getOrElse { AppSettings() }
        val next = block(current)
        context.dataStore.edit { p ->
            p[SettingsKeys.THEME] = next.themeMode.name
            p[SettingsKeys.FONT_SCALE] = next.fontScale
            p[SettingsKeys.SYSTEM_PROMPT] = next.systemPrompt
            p[SettingsKeys.USE_SYS_PROMPT] = next.useSystemPrompt
            p[SettingsKeys.TEMPERATURE] = next.temperature
            p[SettingsKeys.TOP_K] = next.topK
            p[SettingsKeys.TOP_P] = next.topP
            p[SettingsKeys.REP_PENALTY] = next.repetitionPenalty
            p[SettingsKeys.MAX_NEW] = next.maxNewTokens
            p[SettingsKeys.SEED] = next.seed
            p[SettingsKeys.SHOW_THINKING] = next.enableThinking
            p[SettingsKeys.OVERLAY] = next.metricsOverlay
            p[SettingsKeys.ACTIVE_MODEL] = next.activeModelDir
        }
    }
}
