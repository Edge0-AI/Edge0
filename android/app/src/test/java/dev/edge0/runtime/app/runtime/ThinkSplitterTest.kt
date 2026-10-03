// ThinkSplitterTest - incremental tag-pair splitting table (cross-chunk tag cases).
// Tag literals are always assembled from constants (tooling strips special-looking
// literals; the rule applies to tests too).
// Delivery semantics: same-kind segments merge within one feed; across feeds the
// UI aggregates.
package dev.edge0.runtime.app.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class ThinkSplitterTest {
    private val O = ThinkSplitter.OPEN
    private val C = ThinkSplitter.CLOSE

    private fun run(pieces: List<String>): List<Seg> {
        val sp = ThinkSplitter()
        val out = ArrayList<Seg>()
        for (p in pieces) out.addAll(sp.feed(p))
        out.addAll(sp.flush())
        return foldAdjacent(out)
    }

    /** Fold adjacent same-kind segments (normalize the cross-feed freedom). */
    private fun foldAdjacent(segs: List<Seg>): List<Seg> {
        val out = ArrayList<Seg>()
        for (s in segs) {
            val last = out.lastOrNull()
            if (last != null && last::class == s::class) {
                out[out.size - 1] = when (s) {
                    is Seg.Text -> Seg.Text((last as Seg.Text).s + s.s)
                    is Seg.Think -> Seg.Think((last as Seg.Think).s + s.s)
                }
            } else out.add(s)
        }
        return out
    }

    @Test fun promptInjectedOpenStartsInThink() {
        // real-device scenario: the OPEN tag lives in the prompt, the stream starts inside thinking
        val sp = ThinkSplitter(initialThink = true)
        val out = ArrayList<Seg>()
        out.addAll(sp.feed("plan text"))
        out.addAll(sp.feed("$C" + "answer"))
        out.addAll(sp.flush())
        assertEquals(listOf<Seg>(Seg.Think("plan text"), Seg.Text("answer")), foldAdjacent(out))
    }

    @Test fun wholePairInOneFeed() {
        assertEquals(listOf<Seg>(Seg.Think("abc"), Seg.Text("def")),
            run(listOf("$O" + "abc$C" + "def")))
    }

    @Test fun openTagSplitAcrossPieces() {
        // open tag split across three chunks + close tag across two (folded result must be whole)
        assertEquals(listOf<Seg>(Seg.Text("hi "), Seg.Think("thinkingbody"), Seg.Text("tail")),
            run(listOf("hi <", "th", "ink>thinking", "body</t", "h", "ink>tail")))
    }

    @Test fun plainTextNoTags() {
        assertEquals(listOf<Seg>(Seg.Text("purebody.")), run(listOf("pure", "body.")))
    }

    @Test fun fakePrefixNotTag() {
        // body containing a fake "<th" prefix not followed by the rest -> stays body
        assertEquals(listOf<Seg>(Seg.Text("a <th b")), run(listOf("a <th", " b")))
    }

    @Test fun nestedOpenInsideThinkIsPlainThink() {
        assertEquals(listOf<Seg>(Seg.Think("  again $O same again"), Seg.Text("post")),
            run(listOf("$O  again $O same again$C" + "post")))
    }

    @Test fun unclosedToEosIsAllThink() {
        assertEquals(listOf<Seg>(Seg.Think("never closed")), run(listOf("$O" + "never closed")))
    }

    @Test fun emptyPairEmitsNoThinkSegment() {
        assertEquals(listOf<Seg>(Seg.Text("x")), run(listOf("$O$C" + "x")))
    }

    @Test fun trailingFakeOpenFlushesAsText() {
        // EOS with a half-tag carried: flush delivers under the current mode (TEXT)
        assertEquals(listOf<Seg>(Seg.Text("before<th")), run(listOf("before<th")))
    }

    @Test fun thinkThenMoreThinkBlocks() {
        assertEquals(
            listOf<Seg>(Seg.Think("a"), Seg.Text("b"), Seg.Think("c"), Seg.Text("d")),
            run(listOf("$O" + "a$C" + "b$O" + "c$C" + "d")))
    }

    @Test fun charByCharFeedMatchesOneShot() {
        val sp1 = ThinkSplitter()
        val src = "$O" + "multi step" + "thinking$C" + "reply $O" + "again$C" + "end"
        val oneShot = foldAdjacent(sp1.feed(src) + sp1.flush())
        val sp2 = ThinkSplitter()
        val per = ArrayList<Seg>()
        for (ch in src) { per.addAll(sp2.feed(ch.toString())) }
        per.addAll(sp2.flush())
        assertEquals(oneShot, foldAdjacent(per))
    }

    @Test fun thinkingNowTracksMode() {
        val sp = ThinkSplitter()
        sp.feed("$O" + "x")
        assertEquals(true, sp.thinkingNow)
        sp.feed("y$C" + "z")
        assertEquals(false, sp.thinkingNow)
    }
}
