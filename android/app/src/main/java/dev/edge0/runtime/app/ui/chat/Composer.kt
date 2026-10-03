// ui/chat/Composer.kt - rounded input card plus send button, nothing else.
// Temperature/thinking live in Settings; model switching in the top bar.
// IME-safe: Enter always inserts a newline; sending is button-only.
package dev.edge0.runtime.app.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.edge0.runtime.app.ui.theme.LocalFontScale

@Composable
fun Composer(
    value: String,
    onValueChange: (String) -> Unit,
    busy: Boolean,           // streaming/cancelling shows the stop button
    enabledSend: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(modifier.fillMaxWidth().imePadding()
        .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Surface(Modifier.weight(1f),
                shape = RoundedCornerShape(26.dp),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                border = null) {
            BasicTextField(
                value = value, onValueChange = onValueChange,
                textStyle = TextStyle(
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 15.sp * LocalFontScale.current,
                    lineHeight = 21.sp * LocalFontScale.current),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                modifier = Modifier.fillMaxWidth()
                    .defaultMinSize(minHeight = 26.dp)
                    .heightIn(max = 168.dp) // ~8 lines
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                decorationBox = { inner ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (value.isEmpty()) {
                            Text("Ask anything...",
                                 color = MaterialTheme.colorScheme.onSurfaceVariant,
                                 fontSize = 15.sp * LocalFontScale.current)
                        }
                        inner()
                    }
                })
        }
        Spacer(Modifier.width(0.dp))
        val canSend = busy || enabledSend
        val bg = when {
            busy -> MaterialTheme.colorScheme.error
            canSend -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.surfaceVariant
        }
        val fg = when {
            busy -> Color.White
            canSend -> MaterialTheme.colorScheme.onPrimary
            else -> MaterialTheme.colorScheme.onSurfaceVariant
        }
        Box(Modifier.size(48.dp).background(bg, CircleShape)
            .clickable(enabled = canSend) { if (busy) onStop() else onSend() },
            contentAlignment = Alignment.Center) {
            Icon(if (busy) Icons.Filled.Stop else Icons.Filled.ArrowUpward,
                 contentDescription = if (busy) "Stop" else "Send",
                 tint = fg, modifier = Modifier.size(22.dp))
        }
    }
}
