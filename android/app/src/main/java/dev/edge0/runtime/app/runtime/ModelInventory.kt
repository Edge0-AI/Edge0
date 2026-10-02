// runtime/ModelInventory.kt - model discovery for the import/models screen.
// Scans the app's model roots (internal files/models first; external files/models
// as a secondary root - on some OEM Android builds shell-pushed files under
// Android/data/<pkg> are invisible to the app uid, so in-app copy to the
// internal dir is the primary staging channel). Only flat *.gguf files are
// entries; lora_*.gguf companions are picked up by the runtime, not listed.
package dev.edge0.runtime.app.runtime

import java.io.File

data class ModelEntry(
    val name: String,
    val dir: File,                 // the .gguf file itself
    val totalBytes: Long,          // file size (requirement yardstick)
    val dirBytes: Long,            // same, kept for the UI's size column
    val valid: Boolean,
    val hint: String?,             // reason shown when invalid
    val mode: String = "llama-gguf",
)

object ModelInventory {

    fun scanRoots(roots: List<File?>): List<ModelEntry> =
        roots.filterNotNull().flatMap { scan(it) }
             .distinctBy { it.name }
             .sortedBy { it.name }

    fun scan(root: File): List<ModelEntry> =
        (root.listFiles() ?: emptyArray()).filter {
            it.isFile && it.name.endsWith(".gguf") && !it.name.startsWith("lora_")
        }.map { f ->
            val ok = f.length() > 0
            ModelEntry(f.name.removeSuffix(".gguf"), f, f.length(), f.length(), ok,
                if (ok) null else "empty file")
        }

    /** Free space on the volume backing the primary model root (UI warning gate). */
    fun volumeAvailBytes(root: File?): Long =
        root?.let { runCatching { android.os.StatFs(it.absolutePath).availableBytes }
                     .getOrDefault(0L) } ?: 0L

    /** Import instructions shown on the models screen. */
    fun importTemplate(pkg: String): String = listOf(
        "Place model files flat under files/models/ of this app.",
        "Option 1 - in-app: use the picker to copy from device storage.",
        "Option 2 - adb (debug builds):",
        "  adb push <file> /data/local/tmp/",
        "  adb shell run-as $pkg cp /data/local/tmp/<file> files/models/<file>",
        "Expected names: edge0-8b.gguf (plus lora_edge0_8b-gguf.gguf), edge0-35b.gguf.",
    ).joinToString("\n")
}
