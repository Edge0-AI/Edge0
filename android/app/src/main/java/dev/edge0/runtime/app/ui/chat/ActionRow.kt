// ui/chat/ActionRow.kt - per-message action row: copy / edit / delete
// (+ regenerate on the last assistant turn) and a tok/s badge; hidden while streaming.
package dev.edge0.runtime.app.ui.chat

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.edge0.runtime.app.ui.theme.LocalFontScale

@Composable
fun ActionRow(
    text: String,
    canEdit: Boolean,
    canRegenerate: Boolean,
    onEdit: () -> Unit,
    onDelete: () -> Unit,
    onRegenerate: () -> Unit,
    modifier: Modifier = Modifier,
    endAligned: Boolean = false,
) {
    val clip = LocalClipboardManager.current
    Row(modifier.padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        if (endAligned) Spacer(Modifier.weight(1f))
        IconButton(onClick = { clip.setText(AnnotatedString(text)) },
                   modifier = Modifier.width(36.dp)) {
            Icon(Icons.Outlined.ContentCopy, "Copy", tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(16.dp))
        }
        if (canEdit) IconButton(onClick = onEdit, modifier = Modifier.width(36.dp)) {
            Icon(Icons.Outlined.Edit, "Edit", tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(16.dp))
        }
        IconButton(onClick = onDelete, modifier = Modifier.width(36.dp)) {
            Icon(Icons.Outlined.Delete, "Delete", tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(16.dp))
        }
        if (canRegenerate) IconButton(onClick = onRegenerate, modifier = Modifier.width(36.dp)) {
            Icon(Icons.Outlined.Refresh, "Regenerate", tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(16.dp))
        }
        if (!endAligned) Spacer(Modifier.weight(1f))
    }
}

/** Inline stats line under each reply (Atomic style). Memory segment reports
 *  resident pool bytes only (cacheable mmap views excluded). Missing fields elide. */
@Composable
fun StatsLine(newTokens: Int?, ttftMs: Long?, prefillTokS: Double?,
              decodeTokS: Double?, memBytes: Long?,
              modifier: Modifier = Modifier) {
    val parts = ArrayList<String>(5)
    if (newTokens != null && newTokens > 0) parts.add("$newTokens tokens")
    if (ttftMs != null && ttftMs > 0) parts.add("TTFT %.1fs".format(ttftMs / 1000.0))
    if (prefillTokS != null && prefillTokS > 0.05) parts.add("prefill %.1f tok/s".format(prefillTokS))
    if (decodeTokS != null && decodeTokS > 0) parts.add("decode %.2f tok/s".format(decodeTokS))
    if (memBytes != null && memBytes > 0) parts.add("pool %.0f MB".format(memBytes / 1048576.0))
    if (parts.isEmpty()) return
    Text(parts.joinToString("  ·  "),
         style = MaterialTheme.typography.bodySmall.copy(
            fontSize = 11.5.sp * LocalFontScale.current,
            color = MaterialTheme.colorScheme.onSurfaceVariant),
         modifier = modifier.padding(top = 6.dp))
}
