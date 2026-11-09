package dev.edge0.chat.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.edge0.chat.core.ThemeMode

// Values correspond to the approved prototype's design tokens.
data class Palette(val bg: Color, val sidebar: Color, val surface: Color, val surface2: Color, val surface3: Color,
    val text: Color, val text2: Color, val text3: Color, val border: Color, val borderStrong: Color,
    val accent: Color, val accentHover: Color, val accentSoft: Color, val accentInk: Color,
    val ok: Color, val okSoft: Color, val warn: Color, val warnSoft: Color, val error: Color, val errorSoft: Color,
    val segments: List<Color>)
private val light = Palette(Color(0xfff4f4f2),Color(0xffececea),Color(0xffffffff),Color(0xfff7f7f5),Color(0xffefefec),
    Color(0xff1b1c1e),Color(0xff55585e),Color(0xff8b8e94),Color(0xffe3e3df),Color(0xffd2d2cd),
    Color(0xff2f6feb),Color(0xff2a62d1),Color(0xffe8effd),Color(0xffffffff),
    Color(0xff16834f),Color(0xffe4f3ea),Color(0xffa76400),Color(0xfffbf0dc),Color(0xffc4352f),Color(0xfffbe8e6),
    listOf(Color(0xff2f6feb),Color(0xff7b61ff),Color(0xff16a3a3),Color(0xffe8933a),Color(0xffb4b7bd),Color(0xffd9b44a)))
private val dark = Palette(Color(0xff121315),Color(0xff17181b),Color(0xff1c1d21),Color(0xff212226),Color(0xff2a2b30),
    Color(0xffececee),Color(0xffa8abb2),Color(0xff75787f),Color(0xff2c2d32),Color(0xff3a3b41),
    Color(0xff6b9bff),Color(0xff87adff),Color(0xff1f2a42),Color(0xff0f1115),
    Color(0xff4fcf8e),Color(0xff173225),Color(0xffeab155),Color(0xff3a2c14),Color(0xffff8078),Color(0xff3c1d1b),
    listOf(Color(0xff6b9bff),Color(0xffa08cff),Color(0xff3cc6c6),Color(0xfff0a75b),Color(0xff5d6067),Color(0xffc9a640)))
val LocalPalette = compositionLocalOf { light }
object Tokens {
    val vectorStroke = light.text
    val xs = 4.dp; val sm = 8.dp; val md = 12.dp; val lg = 16.dp; val xl = 20.dp; val xxl = 24.dp; val huge = 32.dp; val jumbo = 40.dp
    val radiusSmall = 6.dp; val radiusMedium = 10.dp; val radiusLarge = 14.dp; val pill = 999.dp
    val hairline = 1.dp; val header = 56.dp; val icon = 20.dp; val touch = 40.dp; val ring = 14.dp; val bar = 8.dp
    val dot = 7.dp; val dotGap = 5.dp; val dotLift = 5.dp; val dotRow = 24.dp
    val userMax = 320.dp; val drawerWidth = 320.dp; val composerMaxLines = 4; val animationMillis = 140
    val segmentInset = 2.dp; val segmentRadius = 8.dp; val segmentHorizontal = 11.dp
    val composerRadius = 24.dp; val composerAction = 32.dp; val composerInset = 6.dp; val composerMinHeight = 48.dp; val stoppingActionWidth = 120.dp
    val codeInset = 14.dp; val codeFont = 12.5.sp; val codeLineHeight = 20.sp
    val fsXs = 11.sp; val fsSm = 12.sp; val fsMd = 13.5.sp; val fsBase = 15.sp; val fsLg = 17.sp; val fsXl = 22.sp; val fs2xl = 28.sp
    val smallShape = RoundedCornerShape(radiusSmall); val mediumShape = RoundedCornerShape(radiusMedium); val largeShape = RoundedCornerShape(radiusLarge)
    val segmentShape = RoundedCornerShape(segmentRadius); val composerShape = RoundedCornerShape(composerRadius); val circleShape = RoundedCornerShape(pill)
}
@Composable fun EdgeTheme(mode: ThemeMode, content: @Composable () -> Unit) {
    val isDark = mode == ThemeMode.Dark || (mode == ThemeMode.System && isSystemInDarkTheme())
    val p = if (isDark) dark else light
    val scheme = if (isDark) darkColorScheme(primary=p.accent,onPrimary=p.accentInk,background=p.bg,onBackground=p.text,surface=p.surface,onSurface=p.text,
        surfaceVariant=p.surface2,onSurfaceVariant=p.text2,outline=p.border,error=p.error,onError=p.accentInk)
    else lightColorScheme(primary=p.accent,onPrimary=p.accentInk,background=p.bg,onBackground=p.text,surface=p.surface,onSurface=p.text,
        surfaceVariant=p.surface2,onSurfaceVariant=p.text2,outline=p.border,error=p.error,onError=p.accentInk)
    val typography = Typography(displayLarge=TextStyle(fontSize=Tokens.fs2xl),displayMedium=TextStyle(fontSize=Tokens.fsXl),displaySmall=TextStyle(fontSize=Tokens.fsLg),
        headlineLarge=TextStyle(fontSize=Tokens.fsLg),headlineSmall=TextStyle(fontSize=Tokens.fsBase),
        bodyLarge=TextStyle(fontSize=Tokens.fsBase),bodyMedium=TextStyle(fontSize=Tokens.fsMd),bodySmall=TextStyle(fontSize=Tokens.fsSm),
        titleLarge=TextStyle(fontSize=Tokens.fsXl),titleMedium=TextStyle(fontSize=Tokens.fsLg),titleSmall=TextStyle(fontSize=Tokens.fsBase),
        labelLarge=TextStyle(fontSize=Tokens.fsMd),labelMedium=TextStyle(fontSize=Tokens.fsSm),labelSmall=TextStyle(fontSize=Tokens.fsXs),headlineMedium=TextStyle(fontSize=Tokens.fs2xl))
    CompositionLocalProvider(LocalPalette provides p, LocalContentColor provides p.text) { MaterialTheme(colorScheme=scheme,typography=typography,shapes=Shapes(extraSmall=Tokens.smallShape,small=Tokens.smallShape,medium=Tokens.mediumShape,large=Tokens.largeShape,extraLarge=Tokens.largeShape),content=content) }
}
