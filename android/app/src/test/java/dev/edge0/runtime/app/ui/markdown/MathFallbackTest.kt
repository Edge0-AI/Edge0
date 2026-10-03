// MathFallbackTest - LaTeX degradation preprocessing for the markdown adapter.
// Guards the blank-math-render incident: $$..$$ math nodes were accepted by the
// renderer but drawn as empty space; mathFallback makes them visible as inline
// code (fenced blocks break inside list items, which is where models put math).
package dev.edge0.runtime.app.ui.markdown

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MathFallbackTest {
    private val D = "\$" // a dollar sign without triggering string interpolation

    @Test fun blockDollarBecomesInlineCode() {
        val src = "1. Divide 8 by 3:\n\n" + D + D + "\\frac{8}{3}" + D + D + "\n\ndone"
        val out = mathFallback(src)
        assertTrue("math content visible", "frac" in out)
        assertTrue("math delimiters gone", D !in out)
        assertTrue("no fence noise", "```" !in out)
    }

    @Test fun inlineDollarBecomesCode() {
        val out = mathFallback("where " + D + "x^2 + y" + D + " holds")
        assertTrue("backtick-wrapped", out.contains("`x^2 + y`"))
    }

    @Test fun moneyTextUntouched() {
        val src = "costs " + D + "5 and " + D + "6 total"
        assertEquals(src, mathFallback(src))
    }

    @Test fun plainTextPassesThrough() {
        assertEquals("no math here at all", mathFallback("no math here at all"))
    }

    @Test fun parenAndBracketForms() {
        assertTrue("`a+b`" in mathFallback("do \\(a+b\\) now"))
        val br = mathFallback("\\[ \\frac{1}{2} \\]")
        assertTrue("bracket body visible", "frac" in br)
        assertTrue("delimiters gone", "\\[" !in br)
    }

    @Test fun multilineBlockCollapsesToOneLine() {
        val src = D + D + "\n\\sum_{i=1}^{n} i\n" + D + D
        val out = mathFallback(src)
        assertTrue("body present", "sum_{i=1}^{n} i" in out)
        assertTrue("single line", !out.trim().contains("\n"))
    }
}
