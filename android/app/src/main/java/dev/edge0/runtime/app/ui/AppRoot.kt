// ui/AppRoot.kt - DrawerHost + NavHost wiring.
// Side sheet (288dp): New Chat / thread list (updatedAt desc, busy pulse dot,
// long-press delete menu) / footer with model management and settings.
package dev.edge0.runtime.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.outlined.Close
import androidx.compose.ui.draw.clip
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.edge0.runtime.app.Edge0App
import dev.edge0.runtime.app.data.AppSettings
import dev.edge0.runtime.app.data.ThreadSummary
import dev.edge0.runtime.app.runtime.GenState
import dev.edge0.runtime.app.ui.chat.ChatScreen
import dev.edge0.runtime.app.ui.chat.ChatViewModel
import dev.edge0.runtime.app.ui.models.ModelsScreen
import dev.edge0.runtime.app.ui.settings.SettingsScreen
import kotlinx.coroutines.launch

object Routes {
    const val CHAT = "chat"
    const val MODELS = "models"
    const val SETTINGS = "settings"
}

@Composable
fun AppRoot() {
    val app = LocalContext.current.applicationContext as Edge0App
    val container = app.container
    val factory = remember {
        object : ViewModelProvider.Factory {
            override fun <T : androidx.lifecycle.ViewModel> create(
                modelClass: Class<T>): T =
                ChatViewModel(null, container.repository, container.runtime,
                              container.settings) as T
        }
    }
    val vm: ChatViewModel = viewModel(factory = factory)
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val nav = rememberNavController()
    val scope = rememberCoroutineScope()
    val threads by container.repository.observeThreads()
        .collectAsStateWithLifecycle(initialValue = emptyList())
    val genState by vm.genState.collectAsStateWithLifecycle()
    val settings by container.settings.flow
        .collectAsStateWithLifecycle(initialValue = AppSettings())

    val closeDrawer: () -> Unit = { scope.launch { drawerState.close() } }

    ModalNavigationDrawer(drawerState = drawerState, drawerContent = {
        ModalDrawerSheet(modifier = Modifier.width(288.dp),
                         drawerContainerColor = MaterialTheme.colorScheme.background) {
            Column(Modifier.fillMaxSize()) {
                // title row: name left, X closes the drawer (thread/model entries live in the top bar)
                Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp,
                                                    top = 28.dp, bottom = 14.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Text("Edge0 Chat", style = MaterialTheme.typography.titleLarge,
                         modifier = Modifier.weight(1f))
                    IconButton(onClick = { closeDrawer() }) {
                        Icon(Icons.Outlined.Close, "Close drawer",
                             tint = MaterialTheme.colorScheme.onSurface)
                    }
                }
                Box(Modifier.fillMaxWidth().height(1.dp)
                    .background(MaterialTheme.colorScheme.outline))
                Text("History", style = MaterialTheme.typography.labelLarge,
                     color = MaterialTheme.colorScheme.onSurfaceVariant,
                     modifier = Modifier.padding(start = 20.dp, top = 16.dp, bottom = 6.dp))
                LazyColumn(Modifier.weight(1f),
                           contentPadding = PaddingValues(horizontal = 10.dp)) {
                    items(threads, key = { it.id }) { t ->
                        ThreadItem(
                            t = t,
                            selected = vm.currentThreadId() == t.id,
                            busyHere = genState is GenState.Streaming &&
                                vm.currentThreadId() == t.id,
                            onOpen = { vm.openThread(t.id); nav.navigate(Routes.CHAT)
                                      closeDrawer() },
                            onDelete = { vm.deleteThread(t.id) },
                        )
                    }
                }
                NavigationDrawerItem(label = { Text("Settings") },
                    icon = { Icon(Icons.Outlined.Settings, null) }, selected = false,
                    onClick = { nav.navigate(Routes.SETTINGS); closeDrawer() })
                Spacer(Modifier.height(8.dp))
            }
        }
    }) {
        Scaffold { insets ->
            NavHost(navController = nav, startDestination = Routes.CHAT,
                    modifier = Modifier.padding(insets)) {
                composable(Routes.CHAT) {
                    Box(Modifier.fillMaxSize()) {
                        ChatScreen(
                            vm = vm,
                            modelLabel = settings.activeModelDir.substringAfterLast('/'),
                            onOpenDrawer = { scope.launch { drawerState.open() } },
                            onOpenModels = { nav.navigate(Routes.MODELS) },
                            onNewChat = { vm.newChat(); nav.navigate(Routes.CHAT) },
                        )
                    }
                }
                composable(Routes.MODELS) {
                    ModelsScreen(onBack = { nav.popBackStack(Routes.CHAT, false) })
                }
                composable(Routes.SETTINGS) {
                    SettingsScreen(onBack = { nav.popBackStack(Routes.CHAT, false) })
                }
            }
        }
    }
}

@Composable
private fun ThreadItem(t: ThreadSummary, selected: Boolean, busyHere: Boolean,
                       onOpen: () -> Unit, onDelete: () -> Unit) {
    // row = rounded card (selected gets surfaceVariant backing); trailing X deletes
    // immediately without confirmation; rename was retired by product decision.
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(if (selected) MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f)
                        else androidx.compose.ui.graphics.Color.Transparent)
            .clickable(onClick = onOpen)
            .padding(start = 14.dp, end = 4.dp, top = 13.dp, bottom = 13.dp),
        verticalAlignment = Alignment.CenterVertically) {
        if (busyHere) {
            val tr = rememberInfiniteTransition(label = "pulse")
            val a by tr.animateFloat(0.25f, 1f,
                infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "p")
            Box(Modifier.size(6.dp).background(
                MaterialTheme.colorScheme.primary, CircleShape).alpha(a))
            Spacer(Modifier.width(8.dp))
        }
        Text(t.title ?: "New chat", maxLines = 1, overflow = TextOverflow.Ellipsis,
             style = MaterialTheme.typography.bodyMedium,
             modifier = Modifier.weight(1f))
        IconButton(onClick = onDelete, modifier = Modifier.size(36.dp)) {
            Icon(Icons.Outlined.Close, "Delete chat",
                 tint = MaterialTheme.colorScheme.onSurfaceVariant,
                 modifier = Modifier.size(17.dp))
        }
    }
}

@Composable
private fun PlaceholderScreen(name: String) {
    Column(Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center) {
        Text(name, style = MaterialTheme.typography.bodyLarge)
        Text("S4-Scaffold-OK", style = MaterialTheme.typography.bodyLarge,
             color = MaterialTheme.colorScheme.error)
    }
}
