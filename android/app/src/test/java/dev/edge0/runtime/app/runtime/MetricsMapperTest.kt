// MetricsMapperTest - real-device schema1 sample capture plus tolerance checks.
package dev.edge0.runtime.app.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MetricsMapperTest {

    private fun resource(): String =
        javaClass.classLoader!!.getResourceAsStream("metrics_real.json")!!
            .readBytes().toString(Charsets.UTF_8)

    @Test fun realDeviceSampleMaps() {
        val m = MetricsMapper.parse(resource())
        assertNotNull(m)
        m!!
        assertEquals(1, m.schema)
        assertEquals(23, m.promptTokens)
        assertEquals(false, m.prefix_reused)
        assertEquals(0, m.prefix_reused_tokens)
        assertEquals(8, m.newTokens)
        assertEquals(1, m.chatTurns)
        assertTrue(m.thinkingOn)
        assertEquals(1_366_167_552L, m.peakRssBytes)
        assertTrue(m.decodeTokS > 4.0 && m.decodeTokS < 4.4)
        assertEquals("predicted", m.route?.mode)
        assertEquals("executed", m.route?.feature)
        assertNull(m.route?.agree) // JSON null → Kotlin null
        assertEquals(635L, m.pool?.hits)
        assertEquals(2095L, m.pool?.misses)
        assertTrue(m.gpu?.active == true)
        assertTrue(m.gpu?.device?.contains("Adreno") == true)
        assertEquals("ok", m.gpu?.why)        // backend-audit row data source
        assertTrue(m.gpu?.cacheHit == true)   // kernel cache hit (CLI-domain precedent)
        val hr = m.hitRate!!
        assertTrue(hr > 0.22 && hr < 0.24) // 635/2730
    }

    @Test fun missingFieldsDefaultNotThrow() {
        val m = MetricsMapper.parse("""{"schema": 1}""")
        assertNotNull(m)
        assertEquals(0, m!!.newTokens)
        assertEquals(false, m.prefix_reused)
        assertEquals(0, m.prefix_reused_tokens)
        assertNull(m.route)
        assertNull(m.hitRate)
    }

    @Test fun unknownFieldsIgnored() {
        val m = MetricsMapper.parse("""{"schema":1,"brand_new_field":{"a":[1,2]},"decode_tok_s":3.0}""")
        assertNotNull(m)
        assertEquals(3.0, m!!.decodeTokS, 1e-9)
    }

    @Test fun corruptInputReturnsNull() {
        assertNull(MetricsMapper.parse("not json at all"))
        assertNull(MetricsMapper.parse(""))
        assertNull(MetricsMapper.parse(null))
    }
}
