package dev.edge0.chat

import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import dev.edge0.chat.ui.theme.Tokens

// Outlines follow the prototype SVGs; no icon-library dependency is required.
fun appIcon(name: String): ImageVector {
    val path = when (name) {
        "Chat" -> "M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z"
        "Models" -> "M21 8l-9-5-9 5v8l9 5 9-5V8ZM3 8l9 5 9-5M12 13v8"
        "Status" -> "M3 12h4l3-8 4 16 3-8h4"
        "Settings" -> "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"
        "History" -> "M4 4h16v16H4zM9 4v16M12 8h5M12 12h5"
        "New chat" -> "M12 20H4V4h10M14 4l4-2 4 4-10 10-4 1 1-4L19 3"
        "Copy", "Copy answer", "Copy reasoning" -> "M9 9h12v12H9zM5 15H3V3h12v2"
        "Regenerate" -> "M20 7v5h-5M20 12a8 8 0 1 0-2 6"
        "Edit" -> "M16 3l5 5-12 12-6 1 1-6L16 3ZM13 6l5 5"
        "Send" -> "M12 20V4M5 11l7-7 7 7"
        "Stop" -> "M6 6h12v12H6z"
        "Delete" -> "M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"
        else -> "M6 9l6 6 6-6"
    }
    return ImageVector.Builder(name, Tokens.icon, Tokens.icon, 24f, 24f).addPath(
        pathData=addPathNodes(path), fill=null, stroke=SolidColor(Tokens.vectorStroke), strokeLineWidth=1.7f,
        strokeLineCap=StrokeCap.Round,strokeLineJoin=StrokeJoin.Round).build()
}
