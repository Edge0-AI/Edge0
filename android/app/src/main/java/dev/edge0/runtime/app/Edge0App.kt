// Edge0App.kt - process-level container (manual DI, no framework) plus:
// cold-start cleanup of orphaned streaming rows, and onTrimMemory -> engine pressure relief
// (logcat probe tag=Edge0Trim).
package dev.edge0.runtime.app

import android.app.Application
import android.content.ComponentCallbacks2
import android.content.res.Configuration
import android.util.Log
import dev.edge0.runtime.app.data.AppDatabase
import dev.edge0.runtime.app.data.ChatRepository
import dev.edge0.runtime.app.data.RoomChatRepository
import dev.edge0.runtime.app.data.SettingsStore
import dev.edge0.runtime.app.runtime.LlamaRuntime
import dev.edge0.runtime.app.runtime.Runtime
import dev.edge0.runtime.app.runtime.ThermalSource
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class AppContainer(private val app: Application) {
    private val llamaRuntime by lazy { LlamaRuntime() }
    val runtime: Runtime by lazy { llamaRuntime }
    val settings = SettingsStore(app)
    val db by lazy { AppDatabase.create(app) }
    val repository: ChatRepository by lazy { RoomChatRepository(db) }
    val thermal by lazy { ThermalSource(app) }
    /** Model import roots: internal files/models (primary - on some OEM builds
     *  shell-pushed files are invisible to the app uid, so device-internal copy is the
     *  reliable channel) plus external files/models; both roots are scanned. */
    val modelRoots: List<File?>
        get() = listOf(File(app.filesDir, "models"), app.getExternalFilesDir("models"))
}

class Edge0App : Application() {
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        // cold start: mark rows left in streaming by a killed process as interrupted (visible, not lost)
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            runCatching { container.repository.markStreamingInterrupted() }
        }
        // ComponentCallbacks2 registered directly on the Application - lifecycle observers
        // (2.8) do not carry onTrimMemory; this is the official hook
        registerComponentCallbacks(TrimBridge(container))
    }
}

/** Pressure mapping: RUNNING_LOW/BACKGROUND levels -> trim(0.5); COMPLETE -> cancel then unload. */
class TrimBridge(private val container: AppContainer) : ComponentCallbacks2 {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    override fun onConfigurationChanged(newConfig: Configuration) {}
    @Deprecated("compat") override fun onLowMemory() {}

    override fun onTrimMemory(level: Int) {
        // level bands: RUNNING_MODERATE=5/LOW=10/CRITICAL=15; UI_HIDDEN=16;
        // BACKGROUND=20, MODERATE=40, COMPLETE=80
        when {
            level == android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> {
                Log.i(TAG, "onTrimMemory level=$level action=cancel+unload")
                container.runtime.cancelActive()
                scope.launch { runCatching { container.runtime.unloadActive() } }
            }
            level in 5..19 || level in 20..79 -> {
                Log.i(TAG, "onTrimMemory level=$level action=trim(0.5)")
                scope.launch { container.runtime.trim(0.5f) }
            }
            else -> Log.i(TAG, "onTrimMemory level=$level action=none")
        }
    }

    companion object { const val TAG = "Edge0Trim" }
}
