package dev.edge0.chat.core

import java.io.File
import org.junit.Assert.*
import org.junit.Test

class ChatCoreTest {
    private val shared = File(System.getProperty("shared.dir"))
    private fun read(name: String) = File(shared, name).readText()
    @Test fun sharedAppAndHistory() {
        val data = AppData.parse(read("app.json"))
        val source = parseJson(read("app.json")).obj()
        assertEquals(source.getValue("system_prompt").string(), data.systemPrompt)
        assertEquals(source.getValue("suggestions").array().map(Json::string), data.suggestions)
        assertEquals(4, data.suggestions.size)
        val history = Conversation(data)
        assertEquals(listOf(Message("system", data.systemPrompt), Message("user", "first")), history.request("first"))
        history.complete("first", "answer", "reasoning")
        val second = history.request("second")
        assertEquals(Message("assistant", "answer", "reasoning"), second[2])
        assertEquals(4, second.size)
        history.clear()
        assertEquals(listOf(Message("system", data.systemPrompt), Message("user", "new")), history.request("new"))
    }
    @Test fun sharedMetrics() {
        for (entry in parseJson(read("metrics-cases.json")).array()) {
            val obj = entry.obj(); val input = obj.getValue("input").obj()
            fun number(key: String) = (input[key] as? Json.Num)?.value
            val metrics = Metrics(generatedCount = number("generated_count")?.toLong(), firstTokenSeconds = number("first_token_seconds")?.toDouble(),
                inputTokens = number("input_tokens")?.toLong(), reusedTokens = number("reused_tokens")?.toLong(), decodeTokensPerSecond = number("decode_tokens_per_second")?.toDouble(), memoryPeakBytes = number("memory_peak_bytes")?.toLong())
            assertEquals(obj.getValue("name").string(), obj.getValue("expected").string(), formatMetrics(metrics))
        }
    }
    @Test fun sharedMath() {
        for (entry in parseJson(read("math-cases.json")).array()) {
            val obj = entry.obj()
            assertEquals(obj.getValue("name").string(), obj.getValue("expected").string(), protectMath(obj.getValue("input").string()))
        }
    }
    @Test fun inlineCodeClosesOnlyWithEqualBacktickRuns() {
        assertEquals("`a ``` \$x\$` and `\$y_i * z_i\$`", protectMath("`a ``` \$x\$` and \$y_i * z_i\$"))
        assertEquals("``a ` \$x\$`` and `\$y\$`", protectMath("``a ` \$x\$`` and \$y\$"))
        assertEquals("`a \\` and `\$y\$`", protectMath("`a \\` and \$y\$"))
    }
    @Test fun thinkingIsFixedUntilNewConversation() {
        val history = Conversation(AppData.parse(read("app.json")))
        assertTrue(history.canChangeThinking(true, false))
        assertFalse(history.canChangeThinking(true, true))
        assertFalse(history.prepare("off", false, true).thinking)
        assertFalse(history.canChangeThinking(true, false))
        history.complete("off", "answer", "")
        assertFalse(history.prepare("cannot switch", true, true).thinking)
        history.clear()
        assertTrue(history.canChangeThinking(true, false))
        assertTrue(history.prepare("new on", true, true).thinking)
        history.clear()
        assertFalse(history.prepare("unsupported", true, false).thinking)
    }
    @Test fun assetInspectionMapping() {
        val available = assetChoice("assets/35b", "edge0_35b", null)
        assertTrue(available.selectable)
        assertEquals("edge0_35b · 35b", available.displayName)
        val unavailable = assetChoice("assets/missing", null, "Missing assets.tsv")
        assertFalse(unavailable.selectable)
        assertEquals("missing", unavailable.displayName)
        assertEquals("Missing assets.tsv", unavailable.reason)
        val choices = modelChoices(listOf(available, unavailable))
        assertEquals(listOf(available), choices.available)
        assertEquals(listOf(unavailable), choices.unavailable)
        val other=assetChoice("assets/other","edge0_35b",null)
        assertEquals(ModelGroups(listOf(available),listOf(other),listOf(unavailable)),modelGroups(listOf(available,other,unavailable),available.directory))
        assertEquals(ModelGroups(emptyList(),listOf(available,other),listOf(unavailable)),modelGroups(listOf(available,other,unavailable),null))
        assertEquals(ModelGroups(listOf(unavailable),listOf(available),emptyList()),modelGroups(listOf(available,unavailable),unavailable.directory))
    }
    @Test fun stoppingFeedbackAndMarker() {
        val generating = RequestFeedback().start()
        assertTrue(generating.active)
        assertEquals("Stop", generating.buttonLabel)
        val stopping = generating.requestStop()
        assertTrue(stopping.stopping)
        assertEquals("Stopping…", stopping.buttonLabel)
        assertEquals(stopping, stopping.requestStop())
        assertEquals("Send", stopping.finish().buttonLabel)
        assertFalse(stopping.finish().active)
        val history = Conversation(AppData.parse(read("app.json")))
        history.complete("user", "partial", "", "16 tokens", stopped = true)
        assertTrue(history.messages().last().stopped)
        assertEquals("16 tokens", history.messages().last().metrics)
        assertFalse(history.messages().last().metrics.contains("state_valid"))
    }
    @Test fun coalescedStreamKeepsAllFragments() {
        val buffer = StreamBuffer()
        assertTrue(buffer.append(0, "first"))
        assertFalse(buffer.append(0, " second"))
        assertFalse(buffer.append(1, "thought"))
        assertEquals(StreamFragments("first second", "thought"), buffer.drain())
        assertEquals(StreamFragments("", ""), buffer.drain())
        assertTrue(buffer.append(0, "next"))
        assertEquals(StreamFragments("next", ""), buffer.drain())
    }
    @Test fun actualScreenStateFollowsModelLifecycle() {
        val initial = ScreenState()
        assertEquals(AppPage.chooseModel, initial.page)
        assertFalse(initial.hasLoadedModel)
        assertFalse(initial.otherLocationExpanded)
        assertTrue(initial.toggleOtherLocation().otherLocationExpanded)
        val loading = initial.beginModelLoad()
        assertTrue(loading.loadingModel)
        assertEquals(AppPage.chooseModel, loading.page)
        assertEquals(AppPage.chooseModel, loading.modelLoadFailed().page)
        val loaded = loading.modelLoaded("assets/35b")
        assertEquals(AppPage.chat, loaded.page)
        assertTrue(loaded.hasLoadedModel)
        val chooser = loaded.openModels()
        assertEquals(AppPage.chooseModel, chooser.page)
        assertEquals("assets/35b", chooser.modelDirectory)
        assertFalse(chooser.otherLocationExpanded)
        assertEquals(loaded, chooser.backToChat())
        val failedReload = loaded.beginModelLoad().modelLoadFailed()
        assertFalse(failedReload.hasLoadedModel)
        assertNull(failedReload.modelDirectory)
        assertEquals(AppPage.chooseModel, failedReload.backToChat().page)
    }
    @Test fun actualPickerOnlyExplainsMissingAvailableModels() {
        val valid = assetChoice("assets/35b", "edge0_35b", null)
        val invalid = assetChoice("assets/missing", null, "Missing metadata")
        assertFalse(ModelPickerPresentation(modelChoices(listOf(valid, invalid))).showPushHelp)
        assertTrue(ModelPickerPresentation(modelChoices(listOf(invalid))).showPushHelp)
        assertTrue(ModelPickerPresentation(modelChoices(emptyList())).showPushHelp)
    }
    @Test fun actualReasoningDisclosureIsPresentationOnlyAndSurvivesStreaming() {
        val history = Conversation(AppData.parse(read("app.json")))
        history.complete("question", "answer", "first thought")
        val before = history.request("next")
        val expanded = ReasoningBlocks()
        assertTrue(expanded.forMessage(1).expanded)
        assertTrue(expanded.forMessage(1).visible("first thought", thinkingEnabled = true))
        assertFalse(expanded.forMessage(1).visible("", thinkingEnabled = true))
        assertFalse(expanded.forMessage(1).visible("unexpected nonempty reasoning", thinkingEnabled = false))
        val collapsed = expanded.toggle(1)
        assertFalse(collapsed.forMessage(1).expanded)
        assertTrue(collapsed.forMessage(1).visible("first thought plus another fragment", thinkingEnabled = true))
        assertFalse(collapsed.forMessage(1).expanded)
        assertEquals(before, history.request("next"))
        assertTrue(collapsed.toggle(1).forMessage(1).expanded)
        assertTrue(collapsed.forMessage(3).expanded)
    }
    @Test fun actualSwitchPresentationRetainsCheckedToneWhenLocked() {
        val on = ThinkingSwitchPresentation(checked = true, enabled = false)
        val off = ThinkingSwitchPresentation(checked = false, enabled = false)
        assertEquals("On", on.stateText)
        assertEquals("Off", off.stateText)
        assertEquals("On", on.copy(enabled = true).stateText)
        assertEquals("Off", off.copy(enabled = true).stateText)
        assertEquals(SwitchTone.primary, on.tone)
        assertEquals(SwitchTone.neutral, off.tone)
        assertEquals(0.45f, on.opacity, 0f)
        assertEquals(on.opacity, off.opacity, 0f)
        assertEquals(1f, on.copy(enabled = true).opacity, 0f)
        assertTrue(on.checked)
        assertFalse(off.checked)
    }
    @Test fun actualSendRequestOmitsLimitUnlessTestExplicitlyOverridesIt() {
        val history = Conversation(AppData.parse(read("app.json")))
        val normalRequest = history.prepare("normal send", false, true)
        assertNull(normalRequest.maxTokens)
        assertFalse(normalRequest.thinking)
        val boundedTestRequest = history.prepare("short automatic test", false, true, maxTokens = 16)
        assertEquals(16, boundedTestRequest.maxTokens)
        assertNull(history.prepare("normal send after a bounded test", false, true).maxTokens)
    }
    @Test fun jsonEscapesAndInvalidInput() {
        assertEquals("a\n\"b", parseJson("\"a\\n\\\"b\"").string())
        assertThrows(IllegalArgumentException::class.java) { parseJson("{} trailing") }
    }
}
