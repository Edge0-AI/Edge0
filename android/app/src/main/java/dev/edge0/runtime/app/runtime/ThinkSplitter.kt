// runtime/ThinkSplitter.kt - streaming piece splitter that demuxes engine output
// into body/thinking increments. The engine never strips think tags (template purity
// contract), so parsing happens here. Tag literals are assembled in two pieces to
// survive tooling that strips special-token lookalikes; verify bytes on disk after edits.
package dev.edge0.runtime.app.runtime

sealed interface Seg {
    data class Text(val s: String) : Seg
    data class Think(val s: String) : Seg
}

class ThinkSplitter(initialThink: Boolean = false) {
    companion object {
        const val OPEN = "<th" + "ink>"
        const val CLOSE = "</th" + "ink>"
        private const val MIN_OVERLAP_KEEP = 1
    }

    // when thinking_on the opening tag is injected at the prompt tail by the
    // template, so the stream starts inside the thinking region - callers must pass
    // initialThink=thinkingOn or thinking text leaks wholesale into the body.
    private var inThink = initialThink
    private var carry = "" // sole retained state: suffix of the watched tag that is a real prefix of it

    /** Whether we are currently inside a thinking segment (drives the UI timer). */
    val thinkingNow: Boolean get() = inThink

    fun feed(piece: String): List<Seg> {
        if (piece.isEmpty() && carry.isEmpty()) return emptyList()
        val s = carry + piece
        carry = ""
        val out = ArrayList<Seg>(4)
        var i = 0
        while (i < s.length || i == 0) {
            val tag = if (inThink) CLOSE else OPEN
            val idx = s.indexOf(tag, i)
            if (idx >= 0) {
                if (idx > i) emit(out, s.substring(i, idx))
                inThink = !inThink
                i = idx + tag.length
                if (i >= s.length) return out
            } else {
                val keep = overlapSuffixLen(s, tag)
                if (keep > 0 && i <= s.length - keep) {
                    if (s.length - i - keep > 0) emit(out, s.substring(i, s.length - keep))
                    carry = s.substring(s.length - keep)
                } else if (i < s.length) {
                    emit(out, s.substring(i))
                }
                return out
            }
        }
        return out
    }

    /** End of stream: flush the carried prefix per current mode (unclosed thinking is tolerated). */
    fun flush(): List<Seg> {
        if (carry.isEmpty()) return emptyList()
        val out = listOf(segOf(carry))
        carry = ""
        return out
    }

    // maximal overlap of the tail with a real prefix of tag (length <= tag.size-1); cross-chunk tags live only here
    private fun overlapSuffixLen(s: String, tag: String): Int {
        val max = minOf(tag.length - 1, s.length)
        for (k in max downTo MIN_OVERLAP_KEEP) {
            if (s.endsWith(tag.substring(0, k))) return k
        }
        return 0
    }

    private fun emit(out: MutableList<Seg>, text: String) {
        val seg = segOf(text)
        // merge consecutive same-kind segments (keep event count minimal after flips within one feed)
        val last = out.lastOrNull()
        out.add(when {
            last is Seg.Text && seg is Seg.Text -> { out.removeAt(out.size - 1); Seg.Text(last.s + seg.s) }
            last is Seg.Think && seg is Seg.Think -> { out.removeAt(out.size - 1); Seg.Think(last.s + seg.s) }
            else -> seg
        })
    }

    private fun segOf(text: String): Seg =
        if (inThink) Seg.Think(text) else Seg.Text(text)
}
