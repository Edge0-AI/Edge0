// ui/markdown/MarkdownText.kt - thin rendering adapter.
// Markdown is rendered by mikepenz multiplatform-markdown-renderer-m3 (tables,
// links, strikethrough included). Pinned to 0.27.0: newer releases require
// compileSdk >= 36 / Kotlin >= 2.1, while this project stays on AGP 8.7.3 (35)
// + Kotlin 2.0.21, so 0.27.0 is the last compatible line. Colors come from
// MaterialTheme, keeping the monochrome palette intact.
//
// LaTeX fallback: the renderer recognizes $$..$$ / $..$ math nodes but ships no
// math renderer, so model math output used to render as blank space. mathFallback
// degrades math to code style before parsing - raw LaTeX stays readable in the
// monochrome theme without pulling a typesetting engine into the app.
package dev.edge0.runtime.app.ui.markdown

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.mikepenz.markdown.m3.Markdown
import com.mikepenz.markdown.m3.markdownColor
import com.mikepenz.markdown.m3.markdownTypography

@Composable
fun MarkdownText(src: String, modifier: Modifier = Modifier) {
    Markdown(mathFallback(src), markdownColor(), markdownTypography(), modifier)
}

/**
 * Math degradation: the renderer recognizes math nodes but has no typesetter.
 * Fenced code blocks break when the math sits inside a list item (the fence
 * gets swallowed by the list paragraph), so every math form becomes inline
 * code with its LaTeX backslashes escaped - readable anywhere in the tree.
 */
internal fun mathFallback(src: String): String {
    if ('$' !in src && "\\(" !in src && "\\[" !in src) return src
    var s = BLOCK_DOLLAR.replace(src) { inlineMath(it.groupValues[1]) }
    s = BLOCK_BRACKET.replace(s) { inlineMath(it.groupValues[1]) }
    s = INLINE_DOLLAR.replace(s) { inlineMath(it.groupValues[1]) }
    s = INLINE_PAREN.replace(s) { inlineMath(it.groupValues[1]) }
    return s
}

private fun inlineMath(body: String): String {
    val oneLine = body.trim().replace(Regex("\\s+"), " ").replace("\\", "\\\\")
    return if (oneLine.length > 160) oneLine.take(160) else "`$oneLine`"
}

private val BLOCK_DOLLAR = Regex("""\$\$\s*([\s\S]+?)\s*\$\$""")
private val BLOCK_BRACKET = Regex("""\\\[\s*([\s\S]+?)\s*\\\]""")
private val INLINE_DOLLAR = Regex("""\$([^\s$](?:[^$\n]{0,88}[^\s$])?)\$""")
private val INLINE_PAREN = Regex("""\\\(([^$\n]{1,90}?)\\\)""")
