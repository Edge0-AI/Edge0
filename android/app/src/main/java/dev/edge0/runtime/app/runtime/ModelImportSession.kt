// runtime/ModelImportSession.kt - one in-flight document import for the process.
// A 8B/35B GGUF is several gigabytes, so the copy is not owned by the Models
// screen: leaving that screen must not cancel it.
package dev.edge0.runtime.app.runtime

import android.content.Context
import android.net.Uri
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ModelImportSession(private val modelsDir: File) {
    data class Progress(val name: String, val copied: Long, val total: Long?)

    val progress = MutableStateFlow<Progress?>(null)
    val status = MutableStateFlow<String?>(null)
    val error = MutableStateFlow<String?>(null)
    val running = MutableStateFlow(false)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    fun deleteStalePartials() {
        if (!running.value) ModelImport.deleteStalePartials(modelsDir)
    }

    fun start(context: Context, uris: List<Uri>, activePath: () -> String?) {
        if (uris.isEmpty() || running.value) return
        val appContext = context.applicationContext
        running.value = true
        status.value = null
        error.value = null
        scope.launch {
            val problems = ArrayList<String>()
            var imported = 0
            try {
                for (uri in uris) {
                    try {
                        withContext(Dispatchers.IO) {
                            ModelDocumentImport.import(
                                appContext, uri, modelsDir,
                                blocked = { dest ->
                                    if (activePath() == dest.absolutePath)
                                        "Unload ${dest.name} before replacing it"
                                    else null
                                },
                            ) { name, copied, total ->
                                progress.value = Progress(name, copied, total)
                            }
                        }
                        imported++
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        problems += e.message ?: "Import failed"
                    }
                }
            } finally {
                running.value = false
                progress.value = null
            }
            status.value = when (imported) {
                0 -> null
                1 -> "Imported 1 file"
                else -> "Imported $imported files"
            }
            error.value = problems.takeIf { it.isNotEmpty() }?.joinToString("\n")
        }
    }
}
