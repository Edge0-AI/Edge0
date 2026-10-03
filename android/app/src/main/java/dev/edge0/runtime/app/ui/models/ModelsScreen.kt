// ui/models/ModelsScreen.kt - model management: scans files/models, shows
// size/integrity/load state, load|unload buttons (disabled while generating), a disk
// headroom bar (warn under 1.2x requirement) and an import-instructions card.
package dev.edge0.runtime.app.ui.models

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.edge0.runtime.app.Edge0App
import dev.edge0.runtime.app.runtime.GenState
import dev.edge0.runtime.app.runtime.ModelEntry
import dev.edge0.runtime.app.runtime.ModelInventory
import dev.edge0.runtime.app.ui.theme.LocalFontScale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private fun fmtGiB(b: Long): String = "%.2f GiB".format(b / 1073741824.0)

@Composable
fun ModelsScreen(onBack: () -> Unit) {
    val app = LocalContext.current.applicationContext as Edge0App
    val container = app.container
    val scope = rememberCoroutineScope()
    var entries by remember { mutableStateOf<List<ModelEntry>>(emptyList()) }
    var avail by remember { mutableStateOf(-1L) }
    var note by remember { mutableStateOf<String?>(null) }
    val genState by container.runtime.state.collectAsStateWithLifecycle()
    var refreshTick by remember { mutableStateOf(0) }

    LaunchedEffect(refreshTick) {
        withContext(Dispatchers.IO) {
            entries = ModelInventory.scanRoots(container.modelRoots)
            avail = ModelInventory.volumeAvailBytes(app.filesDir)
        }
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back") }
            Text("Models", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            OutlinedButton(onClick = { refreshTick++ }, modifier = Modifier.padding(end = 12.dp)) {
                Text("Refresh")
            }
        }
        // disk headroom bar (available vs required x 1.2)
        if (avail >= 0) {
            val maxReq = entries.filter { it.valid }.maxOfOrNull { it.totalBytes } ?: 0L
            val warn = maxReq > 0 && avail < maxReq * 1.2
            Surface(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)) {
                Column(Modifier.padding(12.dp)) {
                    Text("Storage: ${if (avail > 0) fmtGiB(avail) else "N/A"} free" +
                        if (maxReq > 0) " / largest model ${fmtGiB(maxReq)}, ${fmtGiB((maxReq * 1.2).toLong())} recommended" else "",
                         style = MaterialTheme.typography.bodySmall)
                    if (warn) {
                        Text("Free space below 1.2x requirement: loading may fail midway (engine reports out-of-storage)",
                             color = MaterialTheme.colorScheme.error,
                             style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        note?.let {
            Text(it, color = MaterialTheme.colorScheme.error,
                 style = MaterialTheme.typography.bodySmall,
                 modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp))
        }
        if (entries.isEmpty()) {
            Text("No models found. Import via adb to:",
                 modifier = Modifier.padding(16.dp), style = MaterialTheme.typography.bodyMedium)
            Surface(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    shape = RoundedCornerShape(10.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)) {
                Text(ModelInventory.importTemplate(app.packageName),
                     fontFamily = FontFamily.Monospace,
                     style = MaterialTheme.typography.bodySmall.copy(
                        fontSize = 11.sp * LocalFontScale.current),
                     modifier = Modifier.padding(12.dp))
            }
            Text("Tap Refresh (top right) after importing. Note: uninstalling the app wipes its data " +
                 "(Android policy); re-import assets with the command above if they disappear.",
                 style = MaterialTheme.typography.bodySmall,
                 color = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.padding(16.dp))
        }
        for (e in entries) {
            ModelRow(e,
                     active = container.runtime.activeModelDir == e.dir.absolutePath,
                     busy = genState is GenState.Streaming || genState is GenState.Cancelling,
                     loading = genState is GenState.Loading,
                     onLoad = {
                         scope.launch {
                             note = null
                             try {
                                 container.runtime.ensureLoaded(e.dir.absolutePath)
                                 container.settings.update { s -> s.copy(activeModelDir = e.dir.absolutePath) }
                             } catch (ex: Exception) {
                                 android.util.Log.e("Edge0Load", "ensureLoaded failed", ex)
                                 note = "Load failed: ${ex.message} (if the app was just updated, re-push assets to the import path)"
                             }
                         }
                     },
                     onUnload = {
                         scope.launch {
                             note = null
                             try { container.runtime.unloadActive() } catch (ex: Exception) {
                                 note = "Unload failed: ${ex.message}"
                             }
                         }
                     })
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ModelRow(e: ModelEntry, active: Boolean, busy: Boolean, loading: Boolean,
                     onLoad: () -> Unit, onUnload: () -> Unit) {
    Surface(Modifier.fillMaxWidth().padding(horizontal = 12.dp),
            shape = RoundedCornerShape(10.dp),
            color = MaterialTheme.colorScheme.surface,
            border = androidx.compose.foundation.BorderStroke(
                1.dp, if (active) MaterialTheme.colorScheme.primary
                      else MaterialTheme.colorScheme.outline)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(e.name, maxLines = 1, overflow = TextOverflow.Ellipsis,
                         style = MaterialTheme.typography.titleSmall)
                    Spacer(Modifier.size(6.dp))
                    Text("e0b", style = MaterialTheme.typography.labelSmall,
                         modifier = Modifier.padding(horizontal = 4.dp))
                }
                Text("${fmtGiB(if (e.dirBytes > 0) e.dirBytes else e.totalBytes)}" +
                     if (!e.valid) " · incomplete (${e.hint})" else "",
                     style = MaterialTheme.typography.bodySmall,
                     color = if (e.valid) MaterialTheme.colorScheme.onSurfaceVariant
                             else MaterialTheme.colorScheme.error)
                Text(if (active) "● Loaded" else if (loading) "◌ Loading..." else "○ Not loaded",
                     style = MaterialTheme.typography.bodySmall.copy(
                        fontSize = 11.sp * LocalFontScale.current),
                     color = if (active) MaterialTheme.colorScheme.primary
                             else MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (!e.valid) {
                Icon(Icons.Outlined.Warning, "Broken assets", tint = MaterialTheme.colorScheme.error)
                return@Row
            }
            if (active) {
                OutlinedButton(onClick = onUnload, enabled = !busy && !loading) { Text("Unload") }
            } else {
                Button(onClick = onLoad, enabled = !busy && !loading) {
                    if (loading) CircularProgressIndicator(Modifier.size(14.dp))
                    else Text("Load")
                }
            }
        }
    }
}
