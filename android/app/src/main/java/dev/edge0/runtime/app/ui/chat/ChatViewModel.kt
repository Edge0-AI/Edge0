// ui/chat/ChatViewModel.kt - chat page orchestration: send -> load -> collect
// generation events -> persist terminal state. Engine cancellation goes exclusively
// through runtime.cancelActive (no fake interruption in the VM). The engine
// overflow diagnostic (INVALID_ARG "prompt tokens=") triggers one retry with a
// tighter budget. Both Done and Cancelled finalize to storage (a cancelled bubble
// keeps its partial content).
package dev.edge0.runtime.app.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.edge0.runtime.app.data.AppSettings
import dev.edge0.runtime.app.data.ChatRepository
import dev.edge0.runtime.app.data.MessageView
import dev.edge0.runtime.app.data.MsgStatus
import dev.edge0.runtime.app.data.SettingsStore
import dev.edge0.runtime.app.runtime.Runtime
import dev.edge0.runtime.app.runtime.GenEvent
import dev.edge0.runtime.app.runtime.GenState
import dev.edge0.runtime.app.runtime.encodeIds
import dev.edge0.runtime.app.runtime.HistoryWindow
import dev.edge0.runtime.app.runtime.RequestMetrics
import dev.edge0.runtime.engine.GenParams
import dev.edge0.runtime.engine.Status
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class StreamingDraft(
    val text: String = "",
    val thinking: String = "",
    val prefilling: Boolean = true,   // before the first event: show the Working... shimmer
    val thinkingActive: Boolean = false,
    val thinkStartedMs: Long? = null,
    val thinkEndedMs: Long? = null,
)

class ChatViewModel(
    initialThreadId: String?,
    private val repo: ChatRepository,
    private val runtime: Runtime,
    private val settingsStore: SettingsStore,
) : ViewModel() {

    val draft = MutableStateFlow("")
    val streaming = MutableStateFlow<StreamingDraft?>(null)
    val banner = MutableStateFlow<String?>(null)
    val genState: StateFlow<GenState> = runtime.state
    val lastMetrics: StateFlow<RequestMetrics?> = runtime.lastMetrics

    private val activeThreadId = MutableStateFlow(initialThreadId)
    fun currentThreadId(): String? = activeThreadId.value

    /** Message stream of the current thread (re-follows on thread switch). */
    @OptIn(ExperimentalCoroutinesApi::class)
    val messages: StateFlow<List<MessageView>> = activeThreadId
        .flatMapLatest { tid -> if (tid == null) flowOf(emptyList()) else repo.observeMessages(tid) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun newChat() {
        activeThreadId.value = null
        draft.value = ""
        banner.value = null
    }

    fun openThread(id: String) {
        if (runtime.state.value is GenState.Streaming) return // no thread switch mid-generation (single slot)
        activeThreadId.value = id
        banner.value = null
    }

    fun sendMessage() {
        val userText = draft.value.trim()
        if (userText.isEmpty()) return
        val st = runtime.state.value
        if (st is GenState.Streaming || st is GenState.Cancelling || streaming.value != null) return
        draft.value = ""
        viewModelScope.launch {
            val settings = runCatching { settingsStore.flow.first() }.getOrNull() ?: return@launch
            val existing = activeThreadId.value
            val tid = existing ?: repo.startThreadWithUserMessage(
                runtime.activeModelDir ?: settings.activeModelDir, userText).also {
                activeThreadId.value = it
            }
            if (existing != null) repo.addUserMessage(tid, userText)

            if (runtime.activeModelDir == null) {
                if (settings.activeModelDir.isBlank()) {
                    banner.value = "No model loaded: import one into the app (adb push) via Models, then load it"
                    return@launch
                }
                try {
                    runtime.ensureLoaded(settings.activeModelDir)
                } catch (e: Exception) {
                    banner.value = "Load failed: ${e.message}"
                    return@launch
                }
            }
            runGenerate(tid, settings, budget = HistoryWindow.DEFAULT_BUDGET_CHARS,
                        allowRetry = true)
        }
    }

    /** One generation round (with the overflow-trim retry). The last history entry is the new user turn. */
    private suspend fun runGenerate(tid: String, settings: AppSettings,
                                    budget: Int, allowRetry: Boolean) {
        val history = repo.history(tid)
        val msgs = HistoryWindow.build(
            settings.systemPrompt.takeIf { settings.useSystemPrompt }?.ifBlank { null },
            history, budget)
        val mid = repo.beginAssistant(tid, runtime.activeModelDir ?: "")
        streaming.value = StreamingDraft()
        val text = StringBuilder()
        val think = StringBuilder()
        var thinkStart = 0L
        var thinkMs: Long? = null
        var terminal: GenEvent? = null
        val params = settings.toGenParams()
        runtime.submit(tid, msgs, settings.enableThinking, params).collect { ev ->
            when (ev) {
                is GenEvent.ThinkPiece -> {
                    if (think.isEmpty()) thinkStart = System.currentTimeMillis()
                    think.append(ev.s)
                    streaming.value = (streaming.value ?: StreamingDraft()).copy(
                        prefilling = false, thinkingActive = true,
                        thinking = think.toString(), thinkStartedMs = thinkStart)
                }
                is GenEvent.TextPiece -> {
                    if (streaming.value?.thinkingActive == true && thinkMs == null) {
                        thinkMs = System.currentTimeMillis() - thinkStart
                    }
                    text.append(ev.s)
                    streaming.value = (streaming.value ?: StreamingDraft()).copy(
                        prefilling = false, thinkingActive = false,
                        text = text.toString(), thinking = think.toString(),
                        thinkEndedMs = thinkMs)
                }
                else -> terminal = ev
            }
        }
        streaming.value = null
        val metrics = (terminal as? GenEvent.Done)?.metrics
            ?: (terminal as? GenEvent.Cancelled)?.metrics
        // timing fallback: stopped/cancelled turns never flip THINK to TEXT -
        // freeze the clock at the terminal moment. Unclosed thinking is kept whole in
        // the thinking region (same semantics as the upstream split helper).
        var thinkMsFinal = thinkMs
        if (thinkMsFinal == null && think.isNotEmpty() && thinkStart > 0) {
            thinkMsFinal = System.currentTimeMillis() - thinkStart
        }
        val status = when (terminal) {
            is GenEvent.Done -> MsgStatus.OK
            is GenEvent.Cancelled -> MsgStatus.CANCELLED
            else -> MsgStatus.ERROR
        }
        // the sidecar persists only on successful turns (null on cancel/failure)
        repo.finalizeAssistant(mid, text.toString(), think.toString().ifBlank { null },
                               thinkMsFinal, status, metrics?.promptTokens, metrics?.newTokens,
                               metrics?.decodeTokS,
                               (terminal as? GenEvent.Done)?.genIds?.takeIf { it.isNotEmpty() }
                                   ?.let { encodeIds(it) },
                               ttftMs = metrics?.firstTokenMs?.takeIf { it > 0 },
                               prefillTokS = metrics?.prefillTokS?.takeIf { it > 0.05 },
                               // report resident pool usage only (cacheable mmap views excluded)
                               memBytes = metrics?.pool?.resident_bytes?.takeIf { it > 0 })
        val fail = terminal as? GenEvent.Failed
        if (fail != null) {
            if (allowRetry && fail.status == Status.INVALID_ARG &&
                fail.diag.contains("prompt tokens=")) {
                runGenerate(tid, settings,
                            HistoryWindow.tighterBudget(budget), allowRetry = false)
            } else {
                banner.value = "Generation failed[${fail.status}]: ${fail.diag}"
            }
        }
    }

    fun cancel() { runtime.cancelActive() }

    fun regenerate() {
        val st = runtime.state.value
        if (st !is GenState.Idle || streaming.value != null) return
        viewModelScope.launch {
            val tid = activeThreadId.value ?: return@launch
            val rows = repo.historyWithIds(tid)
            val lastAssistant = rows.lastOrNull { it.role == "assistant" } ?: return@launch
            repo.deleteFrom(tid, lastAssistant.createdAt)
            val settings = settingsStore.flow.first()
            runGenerate(tid, settings, HistoryWindow.DEFAULT_BUDGET_CHARS, allowRetry = true)
        }
    }

    fun deleteMessage(id: String) = viewModelScope.launch { repo.deleteMessage(id) }

    fun editUserMessage(id: String, newText: String) = viewModelScope.launch {
        repo.editUserMessage(id, newText)
    }

    fun renameThread(id: String, title: String) =
        viewModelScope.launch { repo.renameThread(id, title) }

    fun deleteThread(id: String) = viewModelScope.launch {
        repo.deleteThread(id)
        if (activeThreadId.value == id) newChat()
    }
}

private fun AppSettings.toGenParams() =
    GenParams(temperature = temperature, topK = topK, topP = topP, seed = seed,
              // reasoning models need more than the 256-token engine default (observed truncation on a "hello" turn)
              maxNewTokens = if (maxNewTokens > 0) maxNewTokens else 1024,
              // repetition penalty follows the settings panel; first-token greedy is fixed upstream behavior
              repetitionPenalty = repetitionPenalty,
              firstTokenGreedy = true)
