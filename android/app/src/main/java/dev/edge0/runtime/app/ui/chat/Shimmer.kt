// ui/chat/Shimmer.kt - the Working... waiting state: gradient sweep text (opacity pulse variant).
package dev.edge0.runtime.app.ui.chat

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.text.BasicText
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.sp
import androidx.compose.material3.MaterialTheme
import dev.edge0.runtime.app.ui.theme.LocalFontScale

/** Waiting indicator before the first token (prefilling=true). */
@Composable
fun WorkingShimmer(label: String = "Working...", modifier: Modifier = Modifier) {
    val t = rememberInfiniteTransition(label = "shimmer")
    val a by t.animateFloat(0.35f, 0.9f,
        infiniteRepeatable(tween(900, easing = LinearEasing), RepeatMode.Reverse),
        label = "a")
    BasicText(
        label,
        modifier = modifier.alpha(a),
        style = TextStyle(
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 13.sp * LocalFontScale.current,
            fontStyle = FontStyle.Italic),
    )
}
