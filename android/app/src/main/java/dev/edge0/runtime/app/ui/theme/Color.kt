// ui/theme/Color.kt - palette converted from the OKLCH design values to sRGB
// (OKLab -> linear sRGB -> gamma, per Björn Ottosson's transform). The original
// oklch() values are kept in paired comments for traceability. The palette is
// grayscale; blue appears only in active states such as download progress.
package dev.edge0.runtime.app.ui.theme

import androidx.compose.ui.graphics.Color

// ---- Light (index.css :root)----
val LightBackground = Color(0xFFFFFFFF)        // oklch(1 0 0)
val LightForeground = Color(0xFF0A0A0A)        // oklch(0.145 0 0)
val LightPrimary = Color(0xFF171717)           // oklch(0.205 0 0) send-button background
val LightOnPrimary = Color(0xFFFAFAFA)         // primary foreground, shadcn default white
val LightSecondary = Color(0xFFF5F5F5)         // oklch(0.97 0 0) user bubble background
val LightOnSecondary = Color(0xFF0A0A0A)       // secondary-foreground
val LightMutedForeground = Color(0xFF737373)   // oklch(0.556 0 0) secondary text / action row
val LightBorder = Color(0xFFE5E5E5)            // oklch(0.922 0 0)
val LightSidebar = Color(0xFFF2F2F2)           // oklch(0.96 0 0)
val LightDestructive = Color(0xFFE7000B)       // oklch(0.577 0.245 27.325)

// ---- Dark (index.css .dark)----
val DarkBackground = Color(0xFF121212)         // oklch(0.18 0 0)
val DarkForeground = Color(0xFFFAFAFA)         // oklch(0.985 0 0)
val DarkPrimary = Color(0xFFE5E5E5)            // oklch(0.922 0 0) dark primary = light gray, dark glyph on it
val DarkOnPrimary = Color(0xFF171717)
val DarkSecondary = Color(0xFF383838)          // oklch(0.34 0 0)
val DarkOnSecondary = Color(0xFFFAFAFA)
val DarkMutedForeground = Color(0xFFB7B7B7)    // oklch(0.78 0 0)
val DarkBorder = Color(0x24FFFFFF)             // oklch(1 0 0 / 14%)
val DarkSidebar = Color(0xFF1B1B1B)            // oklch(0.22 0 0)
val DarkDestructive = Color(0xFFFF6467)        // oklch(0.704 0.191 22.216)
val DarkRing = Color(0xFFA4A4A4)               // oklch(0.72 0 0)

// layout constants (design spec to dp)
object Dimens {
    const val RadiusCard = 10f    // --radius: 0.625rem ≈ 10dp
    const val RadiusComposer = 24f // rounded-3xl (composer card)
    const val RadiusBubble = 6f    // rounded-md (bubble)
    const val DrawerWidth = 288f   // Sheet 18rem (sidebar.tsx SIDEBAR_WIDTH_MOBILE)
    const val ContentMaxWidth = 768f // max-w-3xl message column (auto-centers on tablets)
}
