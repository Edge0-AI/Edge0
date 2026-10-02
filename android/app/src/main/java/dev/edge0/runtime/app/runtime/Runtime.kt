// runtime/Runtime.kt - the runtime facade contract consumed by ViewModel/screens.
package dev.edge0.runtime.app.runtime

import dev.edge0.runtime.engine.ChatMessage
import dev.edge0.runtime.engine.GenParams
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow

interface Runtime {
    val state: MutableStateFlow<GenState>
    val lastMetrics: MutableStateFlow<RequestMetrics?>
    val activeModelDir: String?
    suspend fun ensureLoaded(path: String)
    suspend fun unloadActive()
    suspend fun trim(ratio: Float): Boolean
    fun submit(threadId: String, messages: List<ChatMessage>,
               thinkingOn: Boolean, params: GenParams): Flow<GenEvent>
    fun cancelActive()
}
