// MainActivity.kt - single-Activity host (edge-to-edge; Compose-drawn; theme follows settings).
package dev.edge0.runtime.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.edge0.runtime.app.data.AppSettings
import dev.edge0.runtime.app.ui.AppRoot
import dev.edge0.runtime.app.ui.theme.Edge0Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        val container = (application as Edge0App).container
        setContent {
            val settings by container.settings.flow
                .collectAsStateWithLifecycle(initialValue = AppSettings())
            Edge0Theme(mode = settings.themeMode, fontScale = settings.fontScale) {
                AppRoot()
            }
        }
    }
}
