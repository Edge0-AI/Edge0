// runtime/ThermalSource.kt - dual-source thermal telemetry:
// PowerManager.currentThermalStatus (API 29+ enum) plus /sys/class/thermal
// nodes (non-root readability verified per node; unreadable ones degrade to N/A).
package dev.edge0.runtime.app.runtime

import android.content.Context
import android.os.Build
import android.os.PowerManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File

data class ThermalReading(
    val statusLevel: Int?,         // PowerManager level (null when unavailable)
    val statusName: String?,
    val topCpuGpuMilli: Int?,      // max across compute-thermal zones (cpu/gpuss/ddr/qfprom)
    val tsEpochMs: Long,
)

class ThermalSource(context: Context) {
    private val power = context.getSystemService(PowerManager::class.java)

    fun read(): ThermalReading {
        var lvl: Int? = null
        var name: String? = null
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                lvl = power.currentThermalStatus
                name = when (lvl) {
                    PowerManager.THERMAL_STATUS_NONE -> "NONE"
                    PowerManager.THERMAL_STATUS_LIGHT -> "LIGHT"
                    PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE"
                    PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
                    PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
                    PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
                    PowerManager.THERMAL_STATUS_SHUTDOWN -> "SHUTDOWN"
                    else -> "S$lvl"
                }
            }
        } catch (_: Exception) { /* degrade to N/A */ }
        var max = -1
        try {
            val zones = File("/sys/class/thermal").listFiles() ?: emptyArray()
            for (z in zones) {
                if (!z.name.startsWith("thermal_zone")) continue
                val type = try { File(z, "type").readText().trim() } catch (_: Exception) { continue }
                if (!(type.contains("cpu", true) || type.contains("gpuss", true) ||
                        type.contains("ddr", true) || type.contains("qfprom", true))) continue
                val milli = try { File(z, "temp").readText().trim().toIntOrNull() } catch (_: Exception) { null }
                if (milli != null && milli > max) max = milli
            }
        } catch (_: Exception) { /* unreadable: null */ }
        return ThermalReading(lvl, name, if (max >= 0) max else null, System.currentTimeMillis())
    }

    /** 1 Hz sampling flow (collected only while the overlay is visible; unsubscribes when hidden). */
    fun flow(): Flow<ThermalReading> = flow {
        while (true) {
            emit(read())
            delay(1000)
        }
    }.flowOn(Dispatchers.IO)
}
