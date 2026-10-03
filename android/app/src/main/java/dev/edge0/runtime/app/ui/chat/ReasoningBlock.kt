// ui/chat/ReasoningBlock.kt - collapsible reasoning card: shows
// "Thinking for Ns..." while streaming, "Thought for Ns" when done; collapsed by default.
package dev.edge0.runtime.app.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.edge0.runtime.app.ui.theme.LocalFontScale

@Composable
fun ReasoningBlock(thinking: String, active: Boolean, startedAtMs: Long?, endedAtMs: Long?,
                   nowMs: Long, modifier: Modifier = Modifier) {
    if (thinking.isEmpty() && !active) return
    var expanded by remember { mutableStateOf(false) }
    val header = if (active) {
        "Thinking for %.0fs...".format((nowMs - (startedAtMs ?: nowMs)) / 1000.0)
    } else if (startedAtMs == null && endedAtMs == null) {
        "Thought"   // legacy rows without a timing basis: qualitative label only
    } else {
        "Thought for %.0fs".format(((endedAtMs ?: nowMs) - (startedAtMs ?: endedAtMs!!)) / 1000.0)
    }
    Surface(
        modifier = modifier.fillMaxWidth().padding(vertical = 4.dp),
        shape = RoundedCornerShape(10.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
    ) {
        Column {
            Row(Modifier.fillMaxWidth().clickable { expanded = !expanded }
                .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, null,
                     modifier = Modifier.padding(end = 6.dp),
                     tint = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(header, style = MaterialTheme.typography.bodyMedium.copy(
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 13.sp * LocalFontScale.current,
                    fontStyle = FontStyle.Italic))
            }
            AnimatedVisibility(visible = expanded) {
                Text(thinking, style = MaterialTheme.typography.bodyMedium.copy(
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 13.sp * LocalFontScale.current),
                    modifier = Modifier.padding(start = 30.dp, end = 12.dp, bottom = 10.dp))
            }
        }
    }
}
