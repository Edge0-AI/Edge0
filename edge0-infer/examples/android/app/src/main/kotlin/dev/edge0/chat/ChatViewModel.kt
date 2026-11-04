package dev.edge0.chat

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import dev.edge0.chat.core.*
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors

enum class MobilePage { Chat, Models, Status, Settings }
class ChatViewModel(application: Application) : AndroidViewModel(application) {
    val appData = AppData.parse(application.assets.open("app.json").bufferedReader().use { it.readText() })
    val assetsRoot = File(application.filesDir, "assets").apply { mkdirs() }
    private val preferences = application.getSharedPreferences("chat", 0)
    private val cumulativeStore = CumulativeTotalsStore({ preferences.getString("cumulative", null) }, { preferences.edit().putString("cumulative", JSONObject(it).toString()).apply() })
    private val files = ConversationFiles(File(application.filesDir, "conversations")) { JSONObject(it).toString() }
    private var systemPrompt = appData.systemPrompt
    private var history = Conversation(appData)
    private var conversationId = UUID.randomUUID().toString()
    private val engine = NativeEngine()
    private val worker = Executors.newSingleThreadExecutor()
    private val observer = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val stream = StreamBuffer()
    @Volatile private var closed = false
    private var foreground = false
    var page by mutableStateOf(MobilePage.Chat); private set
    var historyOpen by mutableStateOf(false)
    var historyEntries by mutableStateOf<List<HistoryEntry>>(emptyList()); private set
    var theme by mutableStateOf(runCatching { ThemeMode.valueOf(preferences.getString("theme", "System")!!) }.getOrDefault(ThemeMode.System)); private set
    var style by mutableStateOf(runCatching { GenerationStyle.valueOf(preferences.getString("style", "Precise")!!) }.getOrDefault(GenerationStyle.Precise)); private set
    var customSystemPrompt by mutableStateOf(preferences.getString("systemPrompt", "").orEmpty()); private set
    var cumulativeView by mutableStateOf(false); private set
    var startupTotals by mutableStateOf(Totals()); private set
    var cumulativeTotals by mutableStateOf(cumulativeStore.totals); private set
    val totals: Totals get() = if (cumulativeView) cumulativeTotals else startupTotals
    var progress by mutableStateOf(NativeProgress()); private set
    var memory by mutableStateOf(NativeMemory()); private set
    var system by mutableStateOf(NativeSystem()); private set
    var lastTurn by mutableStateOf<TurnStatistics?>(null); private set
    var contextCapacity by mutableStateOf<Long?>(null); private set
    var lastMetrics by mutableStateOf<Metrics?>(null); private set
    val context: ContextUsage get() = contextUsage(lastMetrics?.inputTokens, lastMetrics?.generatedCount, contextCapacity)
    var models by mutableStateOf<List<AssetChoice>>(emptyList()); private set
    var navigation by mutableStateOf(ScreenState()); private set
    val selected: String get() = navigation.modelDirectory.orEmpty()
    val title: String get() = conversationTitle(messages)
    var reasoningBlocks by mutableStateOf(ReasoningBlocks()); private set
    fun toggleReasoning(index: Int) { reasoningBlocks = reasoningBlocks.toggle(index) }
    var messages by mutableStateOf<List<Message>>(emptyList()); private set
    var busy by mutableStateOf(false); private set
    var feedback by mutableStateOf(RequestFeedback()); private set
    val generating: Boolean get() = feedback.active
    val stopping: Boolean get() = feedback.stopping
    var ready by mutableStateOf(false); private set
    var thinkingSupported by mutableStateOf(false); private set
    var thinking by mutableStateOf(false); private set
    val canChangeThinking: Boolean get() = history.canChangeThinking(thinkingSupported, busy)
    var status by mutableStateOf(""); private set
    var draft by mutableStateOf("")
    var pathDraft by mutableStateOf("")
    internal var lastStopReason = ""; private set
    internal var lastStateValid = true; private set
    internal var lastTemperature = 0.0; private set
    internal var lastGeneratedCount = 0L; private set
    internal var statusQueries = 0; private set
    private var answer = ""
    private var reasoning = ""
    private var activeUser = ""
    private val poller = VisiblePoller(schedule = { delay, action ->
        val task = Runnable(action); main.postDelayed(task, delay); { main.removeCallbacks(task) }
    }, query = { done ->
        observer.execute {
            val result = runCatching { Triple(engine.progress(), engine.snapshot(), engine.systemStatus()) }
            ui {
                if (page == MobilePage.Status && foreground) result.onSuccess { (p, m, s) -> progress = p; memory = m; system = s; statusQueries++ }
                done()
            }
        }
    })
    private val flushStream = Runnable {
        val fragments = stream.drain()
        if (!closed && (fragments.answer.isNotEmpty() || fragments.reasoning.isNotEmpty())) {
            answer += fragments.answer; reasoning += fragments.reasoning
            messages = history.messages() + Message("user", activeUser) + Message("assistant", answer, reasoning)
        }
    }
    init {
        worker.execute { val entries = files.read(); ui { historyEntries = entries } }
        refreshModels()
    }
    private fun ui(action: () -> Unit) { main.post { if (!closed) action() } }
    fun setForeground(value: Boolean) { foreground = value; poller.setVisible(value && page == MobilePage.Status) }
    fun selectPage(value: MobilePage) { page = value; poller.setVisible(foreground && page == MobilePage.Status) }
    fun openModels() { navigation = navigation.openModels(); selectPage(MobilePage.Models) }
    fun backToChat() { navigation = navigation.backToChat(); selectPage(MobilePage.Chat) }
    fun toggleOtherLocation() { navigation = navigation.toggleOtherLocation() }
    fun toggleThinking(value: Boolean) { if (canChangeThinking) thinking = value }
    fun updateTheme(value: ThemeMode) { theme = value; preferences.edit().putString("theme", value.name).apply() }
    fun updateStyle(value: GenerationStyle) { style = value; preferences.edit().putString("style", value.name).apply() }
    fun setSystemPrompt(value: String) { customSystemPrompt = value; preferences.edit().putString("systemPrompt", value).apply() }
    fun setCumulative(value: Boolean) { cumulativeView = value }
    fun resetCumulative() { cumulativeStore.reset(); cumulativeTotals = cumulativeStore.totals }
    private fun persistConversation() {
        if (messages.isEmpty()) return
        val archive = SavedConversation(conversationId, System.currentTimeMillis(), systemPrompt, history.lockedThinking, history.messages(), selected.takeIf { it.isNotEmpty() })
        worker.execute {
            val result = runCatching { files.save(archive); files.read() }
            ui { result.onSuccess { historyEntries = it }.onFailure { status = it.message ?: "Cannot save conversation" } }
        }
    }
    fun switchConversation(id: String) {
        if (busy) return
        val archive = historyEntries.firstOrNull { it.id == id }?.archive ?: return
        fun restore() {
            systemPrompt = archive.systemPrompt; history = Conversation(appData.copy(systemPrompt = systemPrompt))
            history.restore(archive.messages, archive.thinking); conversationId = archive.id; messages = history.messages()
            thinking = archive.thinking ?: thinking; lastMetrics = messages.lastOrNull { it.role == "assistant" }?.sourceMetrics
            reasoningBlocks = ReasoningBlocks(); historyOpen = false; selectPage(MobilePage.Chat)
        }
        val directory = archive.modelDirectory
        if (directory != null && directory != selected) {
            if (models.none { it.directory == directory && it.selectable }) { status = "This conversation's model is unavailable."; return }
            load(directory, ::restore)
        } else restore()
    }

    fun deleteConversation(id: String) {
        if (busy) return
        worker.execute {
            val result = runCatching { files.delete(id); files.read() }
            ui { result.onSuccess { historyEntries = it; if (id == conversationId) newConversation() }.onFailure { status = "Cannot delete conversation" } }
        }
    }
    fun refreshModels() {
        if (busy) return
        busy = true
        val configured = preferences.getStringSet("directories", emptySet()).orEmpty().toSet()
        worker.execute {
            val choices = (assetsRoot.listFiles().orEmpty().filter { it.isDirectory }.map { it.absolutePath } + configured).distinct().sorted().map(engine::inspect)
            ui { models = choices; busy = false }
        }
    }
    fun addDirectory() {
        val directory = pathDraft.trim(); if (directory.isEmpty() || busy) return
        val configured = preferences.getStringSet("directories", emptySet()).orEmpty().toMutableSet(); configured += directory
        preferences.edit().putStringSet("directories", configured).apply(); pathDraft = ""; refreshModels()
    }
    private fun clearConversation() {
        systemPrompt = customSystemPrompt.ifBlank { appData.systemPrompt }; history = Conversation(appData.copy(systemPrompt = systemPrompt))
        conversationId = UUID.randomUUID().toString(); messages = emptyList(); lastMetrics = null; reasoningBlocks = ReasoningBlocks()
    }
    fun load(directory: String, onLoaded: (() -> Unit)? = null) {
        if (busy) return
        busy = true; ready = false; status = "Loading model…"; contextCapacity = null
        navigation = navigation.beginModelLoad(); clearConversation()
        worker.execute {
            runCatching { engine.load(directory) to engine.context() }.onSuccess { (capabilities, context) -> ui {
                navigation = navigation.modelLoaded(directory); contextCapacity = context; ready = true; busy = false
                thinkingSupported = capabilities[0]; thinking = capabilities[0] && capabilities[1]; status = "Ready"; selectPage(MobilePage.Chat); onLoaded?.invoke()
            } }.onFailure { error -> ui { navigation = navigation.modelLoadFailed(); busy = false; status = error.message ?: "Model load failed"; selectPage(MobilePage.Models) } }
        }
    }
    fun unload() {
        if (busy || selected.isEmpty()) return
        busy = true
        worker.execute { runCatching { engine.unload() }.onSuccess { ui {
            navigation = ScreenState(); ready = false; busy = false; contextCapacity = null; thinkingSupported = false; status = ""; clearConversation()
        } }.onFailure { ui { busy = false; status = it.message ?: "Unload failed" } } }
    }
    fun regenerate() {
        if (busy || messages.lastOrNull()?.role != "assistant") return
        val prefix = regenerateMessages(messages); val user = prefix.lastOrNull()?.takeIf { it.role == "user" }?.content ?: return
        history.restore(prefix.dropLast(1), history.lockedThinking); messages = history.messages(); send(user)
    }
    fun editAndSend(index: Int, text: String) {
        if (busy || text.isBlank()) return
        val prefix = editMessages(messages, index, text)
        history.restore(prefix.dropLast(1), history.lockedThinking); messages = history.messages(); send(text)
    }
    fun send(text: String = draft, maxTokens: Int? = null) {
        val user = text.trim(); if (user.isEmpty() || busy || !ready) return
        try { engine.prepare() } catch (error: Exception) { status = error.message ?: "Cannot prepare generation"; return }
        val turnId = UUID.randomUUID().toString()
        val request = history.prepare(user, thinking, thinkingSupported, maxTokens, style)
        stream.drain(); main.removeCallbacks(flushStream)
        busy = true; feedback = feedback.start(); draft = ""; activeUser = user; answer = ""; reasoning = ""; status = "Generating…"
        messages = history.messages() + Message("user", user) + Message("assistant", "")
        worker.execute {
            runCatching { engine.generate(request, NativeOutput { kind, bytes -> if (stream.append(kind, bytes.toString(Charsets.UTF_8))) main.postDelayed(flushStream, 80) }) }
                .onSuccess { result -> ui {
                    main.removeCallbacks(flushStream); flushStream.run()
                    val source = result.metrics(); val metrics = formatMetrics(source)
                    history.complete(user, answer, reasoning, metrics, stopped = result.stopReason == "cancelled", sourceMetrics = source)
                    messages = history.messages(); lastMetrics = source; lastTurn = result.statistics()
                    startupTotals = startupTotals.add(result.statistics()); cumulativeStore.record(turnId, result.statistics()); cumulativeTotals = cumulativeStore.totals; persistConversation()
                    feedback = feedback.finish(); busy = false; ready = result.stateValid
                    lastTemperature = result.timings[3]; lastStopReason = result.stopReason; lastStateValid = result.stateValid; lastGeneratedCount = result.counts[0]
                    status = result.error ?: if (result.stateValid) "Ready" else "Start a new chat to continue."
                    Log.i("Edge0Chat", "metrics=$metrics")
                } }.onFailure { error -> ui {
                    main.removeCallbacks(flushStream); flushStream.run(); history.complete(user, answer, reasoning)
                    messages = history.messages(); persistConversation(); feedback = feedback.finish(); busy = false; ready = false; status = error.message ?: "Generation failed"
                } }
        }
    }
    fun stop() { if (generating && !stopping) { feedback = feedback.requestStop(); status = "Stopping…"; engine.cancel() } }
    fun newConversation() {
        if (busy) return
        if (selected.isEmpty()) { clearConversation(); return }
        busy = true
        worker.execute { runCatching { engine.reset() }.onSuccess { ui {
            clearConversation(); ready = true; busy = false; status = "Ready"; selectPage(MobilePage.Chat)
        } }.onFailure { error -> ui { busy = false; status = error.message ?: "Reset failed" } } }
    }
    override fun onCleared() {
        poller.setVisible(false); closed = true; main.removeCallbacks(flushStream); engine.cancel()
        observer.shutdown()
        // close's write lock waits for the observer's last read on this background worker.
        worker.execute { runCatching { engine.close() }; worker.shutdown() }
    }
}
