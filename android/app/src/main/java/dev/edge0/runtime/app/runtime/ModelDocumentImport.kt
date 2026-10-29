// runtime/ModelDocumentImport.kt - Storage Access Framework entry for Models.
// The system document picker grants a temporary read URI. No storage permission
// and no adb: the bytes are copied into the app-private models directory.
package dev.edge0.runtime.app.runtime

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File

object ModelDocumentImport {

    /**
     * Copy one picked document into [modelsDir]. [blocked] may reject a
     * destination (for example the model that is currently loaded).
     */
    suspend fun import(
        context: Context,
        uri: Uri,
        modelsDir: File,
        blocked: (File) -> String? = { null },
        onProgress: suspend (name: String, copied: Long, total: Long?) -> Unit = { _, _, _ -> },
    ): File {
        val (display, size) = readMeta(context, uri)
        val safe = ModelImport.fileName(display)
            ?: throw ModelImport.ImportException(
                "Choose a file whose name ends in .gguf" +
                    (display?.let { " ($it)" } ?: ""))
        val dest = File(modelsDir, safe)
        blocked(dest)?.let { throw ModelImport.ImportException(it) }
        val input = try {
            context.contentResolver.openInputStream(uri)
        } catch (e: SecurityException) {
            throw ModelImport.ImportException("Cannot open the selected file")
        } ?: throw ModelImport.ImportException("Cannot open the selected file")
        onProgress(safe, 0L, size)
        return ModelImport.copy(input, modelsDir, safe, size) { copied, total ->
            onProgress(safe, copied, total)
        }
    }

    private fun readMeta(context: Context, uri: Uri): Pair<String?, Long?> {
        var name: String? = null
        var size: Long? = null
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
            null, null, null,
        )?.use { c ->
            if (!c.moveToFirst()) return@use
            val ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val si = c.getColumnIndex(OpenableColumns.SIZE)
            if (ni >= 0) name = c.getString(ni)
            if (si >= 0 && !c.isNull(si)) {
                val s = c.getLong(si)
                if (s >= 0) size = s
            }
        }
        return name to size
    }
}
