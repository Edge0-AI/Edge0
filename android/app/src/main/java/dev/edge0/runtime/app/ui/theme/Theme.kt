// ui/theme/Theme.kt - three theme modes (light/dark/system) plus a global font scale.
// LocalFontScale is the Compose equivalent of the design system's font-size-base
// multiplier; screens apply it explicitly (Material3 typography baseline is 14sp).
package dev.edge0.runtime.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// whole-app text-sm baseline (system sans fallback; custom font files out of scope for v1)
private val Edge0Typography = Typography(
    bodyLarge = TextStyle(fontFamily = FontFamily.Default, fontSize = 14.sp, lineHeight = 20.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.Default, fontSize = 14.sp, lineHeight = 20.sp),
    titleLarge = TextStyle(fontFamily = FontFamily.Default, fontSize = 18.sp,
                           fontWeight = FontWeight.Medium),
)

private val LightScheme = lightColorScheme(
    primary = LightPrimary,
    onPrimary = LightOnPrimary,
    secondary = LightSecondary,
    onSecondary = LightOnSecondary,
    background = LightBackground,
    onBackground = LightForeground,
    surface = LightBackground,
    onSurface = LightForeground,
    surfaceVariant = LightSecondary,
    outline = LightBorder,
    error = LightDestructive,
)

private val DarkScheme = darkColorScheme(
    primary = DarkPrimary,
    onPrimary = DarkOnPrimary,
    secondary = DarkSecondary,
    onSecondary = DarkOnSecondary,
    background = DarkBackground,
    onBackground = DarkForeground,
    surface = DarkBackground,
    onSurface = DarkForeground,
    surfaceVariant = DarkSecondary,
    outline = DarkBorder,
    error = DarkDestructive,
)

/** Radius ladder: md = bubble / lg = card (10dp) / 2xl = composer (24dp). */
private val Edge0Shapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(6.dp),   // user bubble
    medium = RoundedCornerShape(10.dp), // cards
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(24.dp), // composer
)

/** Font scale (0.85..1.3, driven by the settings slider, persisted in DataStore). */
val LocalFontScale = compositionLocalOf { 1.0f }

/** Theme mode: SYSTEM follows the platform setting. */
enum class ThemeMode { SYSTEM, LIGHT, DARK }

@Composable
fun Edge0Theme(
    mode: ThemeMode = ThemeMode.SYSTEM,
    fontScale: Float = 1.0f,
    content: @Composable () -> Unit,
) {
    val dark = when (mode) {
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
    }
    androidx.compose.runtime.CompositionLocalProvider(LocalFontScale provides fontScale) {
        MaterialTheme(
            colorScheme = if (dark) DarkScheme else LightScheme,
            typography = Edge0Typography,
            shapes = Edge0Shapes,
            content = content,
        )
    }
}
