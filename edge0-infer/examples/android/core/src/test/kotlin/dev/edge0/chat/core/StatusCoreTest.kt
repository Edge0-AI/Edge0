package dev.edge0.chat.core

import java.io.File
import java.nio.file.Files
import java.time.LocalDate
import java.time.ZoneId
import org.junit.Assert.*
import org.junit.Test

class StatusCoreTest {
    private val shared = File(System.getProperty("shared.dir"))
    private fun source(name: String) = parseJson(File(shared,name).readText())
    @Test fun sharedStatusVectors() {
        for (entry in source("status-cases.json").array()) {
            val o=entry.obj();val input=o.getValue("input").obj()
            fun n(key: String)=(input[key] as? Json.Num)?.value
            val actual: Json = when(o.getValue("kind").string()) {
                "device_memory" -> {
                    val devices=(input["by_device"] as? Json.Obj)?.fields?.mapValues {(_,value)->
                        val d=value.obj();DeviceMemory((d.getValue("current_bytes") as Json.Num).value.toLong(),(d.getValue("peak_bytes") as Json.Num).value.toLong(),MemoryLocation.valueOf(d.getValue("memory").string()))
                    }
                    summarizeDeviceMemory(devices)?.let {Json.Obj(mapOf("system" to (it.system?.let(Json::Str)?:Json.Null),"dedicated" to (it.dedicated?.let(Json::Str)?:Json.Null)))}?:Json.Null
                }
                else -> {
                    val value=when(o.getValue("kind").string()) {
                        "bytes"->formatBytes(n("value")?.toLong());"count"->formatCount(n("value")?.toLong());"rate"->formatRate(n("value")?.toDouble())
                        "percent"->formatPercent(n("numerator")?.toLong(),n("denominator")?.toLong());"duration"->formatDuration(n("value")?.toDouble());else->error("Unknown kind")
                    };value?.let(Json::Str)?:Json.Null
                }
            }
            assertEquals(o.getValue("name").string(),o.getValue("expected"),actual)
        }
    }
    @Test fun sharedAssistantStateVectors() {
        source("assistant-state-cases.json").array().forEach {entry->
            val o=entry.obj();val i=o.getValue("input").obj();val e=o.getValue("expected").obj();val name=o.getValue("name").string()
            fun b(m: Map<String, Json>,k:String)=(m.getValue(k) as Json.Bool).value
            val result=assistantState(b(i,"is_last"),b(i,"generating"),b(i,"has_content"),b(i,"shows_reasoning"))
            assertEquals(name,b(e,"pending"),result.pending);assertEquals(name,b(e,"actions"),result.actions)
        }
    }
    @Test fun sharedThermalVectors() {
        source("thermal-cases.json").array().forEach {entry->
            val o=entry.obj();val expected=o.getValue("expected").obj();val name=o.getValue("name").string()
            val result=thermalDisplay(o.getValue("input").obj().getValue("state").string())
            assertEquals(name,expected.getValue("label").string(),result.label)
            assertEquals(name,expected.getValue("description").string(),result.description)
            assertEquals(name,expected.getValue("tone").string(),result.tone.name)
        }
    }
    @Test fun sharedContextVectors() {
        source("context-cases.json").array().forEach {entry->
            val o=entry.obj();val input=o.getValue("input").obj();fun n(key:String)=(input[key] as? Json.Num)?.value?.toLong()
            val result=contextUsage(n("input_tokens"),n("generated_tokens"),n("context_length"));val expected=o.getValue("expected").obj()
            assertEquals(o.getValue("name").string(),expected.getValue("text").string(),result.text)
            assertEquals((expected["ring_percent"] as? Json.Num)?.value?.toInt(),result.ringPercent)
        }
    }
    @Test fun nativeSampleMatchesSharedStatusValues() {
        // These are native query structures, not an HTTP parser. Equality with the
        // checked-in real response prevents drifting onto different sample fields.
        val progress=NativeProgress("decode",13,12,1,0,3,0.0061735,0.002878875,0.003294625)
        val memory=NativeMemory(17179869184,46837024,48252608,mapOf("resident_weights" to 20945152,"experts" to 491520,"session_state" to 13328384,"workspace" to 4231136,"transfer" to 2176,"staging" to 0,"metadata" to 7838656),mapOf("CPU" to DeviceMemory(38998368,39993312,MemoryLocation.system),"host" to DeviceMemory(7838656,8259296,MemoryLocation.system)),169476720,199081984)
        val system=NativeSystem(17179869184,4924522496,"nominal",null)
        val root=source("status-sample.json").obj();val current=root.getValue("current").obj();val sampleMemory=root.getValue("memory").obj();val ledger=sampleMemory.getValue("ledger").obj()
        fun n(map:Map<String,Json>,key:String)=(map.getValue(key) as Json.Num).value
        assertEquals(current.getValue("phase").string(),progress.phase)
        assertEquals(n(current,"input_tokens").toLong(),progress.inputTokens);assertEquals(n(current,"reused_tokens").toLong(),progress.reusedTokens)
        assertEquals(n(current,"computed_tokens").toLong(),progress.computedTokens);assertEquals(n(current,"pending_tokens").toLong(),progress.pendingTokens);assertEquals(n(current,"generated_tokens").toLong(),progress.generatedTokens)
        assertEquals(n(current,"elapsed_seconds").toDouble(),progress.elapsedSeconds!!,0.0);assertEquals(n(current,"first_token_seconds").toDouble(),progress.firstTokenSeconds!!,0.0)
        assertEquals(n(current,"decode_elapsed_seconds").toDouble(),progress.decodeElapsedSeconds!!,0.0)
        assertEquals(n(ledger,"limit_bytes").toLong(),memory.limitBytes);assertEquals(n(ledger,"current_bytes").toLong(),memory.currentBytes);assertEquals(n(ledger,"peak_bytes").toLong(),memory.peakBytes)
        assertEquals(ledger.getValue("by_purpose").obj().mapValues {(_,v)->(v as Json.Num).value.toLong()},memory.byPurpose)
        ledger.getValue("by_device").obj().forEach { (name,v)->val d=v.obj();assertEquals(DeviceMemory(n(d,"current_bytes").toLong(),n(d,"peak_bytes").toLong(),MemoryLocation.valueOf(d.getValue("memory").string())),memory.byDevice.getValue(name)) }
        assertEquals(n(sampleMemory.getValue("physical").obj().getValue("values").obj(),"current_bytes").toLong(),memory.physicalBytes)
        assertEquals(n(sampleMemory.getValue("resident").obj().getValue("values").obj(),"current_bytes").toLong(),memory.residentBytes)
        val s=root.getValue("system").obj();val sm=s.getValue("memory").obj()
        assertEquals(n(sm.getValue("total").obj(),"bytes").toLong(),system.totalBytes);assertEquals(n(sm.getValue("available").obj(),"bytes").toLong(),system.availableBytes)
        assertEquals(s.getValue("thermal").obj().getValue("state").string(),system.thermal);assertNull(system.appLimitBytes)
        assertEquals(DeviceMemorySummary("0.05 GB",null),summarizeDeviceMemory(memory.byDevice))
        assertEquals("0.17 GB",formatBytes(memory.physicalBytes))
        val display=requestDisplay(progress)
        assertEquals(RequestDisplay("Decoding","1 tokens · 0.0 s · 347.4 tok/s","3 tokens · 607.0 tok/s"),display)
        val last=TurnStatistics(13,0,8,7,0.01239325,0.006230292,365,95)
        val totals=Totals().add(last)
        assertEquals("564.8 tok/s",totals.averageDecode);assertEquals("26.0%",totals.hits);assertEquals("0.0%",totals.reuse)
        assertEquals("Computing 9 tokens · 2.3 s",requestDisplay(NativeProgress("prefill",13,4,5,4,0,2.25)).prefill)
    }
    @Test fun cumulativeUsesSumsAndCanReset() {
        val totals=Totals().add(TurnStatistics(100,20,10,10,1.0,expertDemand=10,expertHits=8)).add(TurnStatistics(300,220,20,30,3.0,expertDemand=30,expertHits=12))
        assertEquals(2L,totals.requests);assertEquals("60.0%",totals.reuse);assertEquals("50.0%",totals.hits);assertEquals("10.0 tok/s",totals.averageDecode)
        val dir=Files.createTempDirectory("edge0-totals").toFile()
        try {
            val file=File(dir,"totals.json")
            fun store()=CumulativeTotalsStore({file.takeIf {it.exists()}?.readText()},{file.writeText(it)})
            val first=store();val turn=TurnStatistics(100,20,10,10,1.0)
            assertTrue(first.record("round-a",turn));assertFalse(first.record("round-a",turn))
            val restored=store();assertEquals(first.totals,restored.totals)
            assertFalse(restored.record("round-a",turn));assertEquals(1L,restored.totals.requests)
            restored.reset();val afterReset=store();assertEquals(Totals(),afterReset.totals)
            assertFalse(afterReset.record("round-a",turn));assertEquals(Totals(),store().totals)
            assertTrue(afterReset.record("round-b",turn));assertEquals(1L,store().totals.requests)
        } finally {dir.deleteRecursively()}
        assertNull(Totals().averageDecode);assertNull(Totals().hits)
        assertEquals(DeviceMemorySummary("0 MB","0 MB"),summarizeDeviceMemory(mapOf("gpu" to DeviceMemory(0,0,MemoryLocation.dedicated))))
    }
    @Test fun pollingWaitsForCompletionAndVisibilityWithoutOverlap() {
        data class Task(val at:Long,val run:()->Unit,var cancelled:Boolean=false)
        var now=0L;val tasks=mutableListOf<Task>();var count=0;var done:(()->Unit)?=null
        val poller=VisiblePoller({delay,action->val t=Task(now+delay,action);tasks+=t;{t.cancelled=true}},{finish->count++;done=finish})
        fun advance(to:Long) {while(true) {val next=tasks.filter {!it.cancelled&&it.at<=to}.minByOrNull {it.at}?:break;tasks.remove(next);now=next.at;next.run()};now=to}
        poller.setVisible(true);advance(0);assertEquals(1,count)
        poller.setVisible(false);poller.setVisible(true);advance(5000);assertEquals(1,count)
        done!!();advance(5999);assertEquals(1,count);advance(6000);assertEquals(2,count)
        checkNotNull(done)();poller.setVisible(false);advance(10000);assertEquals(2,count)
        poller.setVisible(true);advance(10999);assertEquals(2,count);advance(11000);assertEquals(3,count)
        checkNotNull(done)();poller.setVisible(false);poller.setVisible(true);advance(11999);assertEquals(3,count);advance(12000);assertEquals(4,count)
    }
    @Test fun historyPersistenceCorruptionAndMessageActions() {
        val dir=Files.createTempDirectory("edge0-history-test").toFile()
        try {
            val messages=listOf(Message("user","one"),Message("assistant","answer","thought","8 tokens",sourceMetrics=Metrics(generatedCount=8,inputTokens=13)),Message("user","two"),Message("assistant","second"))
            val archive=SavedConversation("first",123,"custom",true,messages,"assets/model-a")
            val files=ConversationFiles(dir);files.save(archive);assertEquals(archive,files.read().single().archive)
            val corrupt=File(dir,"broken.json");corrupt.writeText("broken input")
            assertEquals(2,files.read().size);assertNotNull(files.read().first {it.id=="broken"}.error)
            assertThrows(IllegalStateException::class.java) {files.save(archive.copy(id="broken"))};assertEquals("broken input",corrupt.readText())
            val mismatched=File(dir,"mismatched.json");mismatched.writeText(archiveJson(archive.copy(id="another")))
            val original=mismatched.readText()
            assertNotNull(files.read().first {it.id=="mismatched"}.error)
            assertThrows(IllegalStateException::class.java) {files.save(archive.copy(id="mismatched"))}
            assertEquals(original,mismatched.readText())
            val restored=Conversation(AppData(archive.systemPrompt,emptyList()));restored.restore(archive.messages,archive.thinking)
            assertEquals(listOf(Message("system","custom"))+messages+Message("user","next"),restored.request("next"))
            assertFalse(restored.canChangeThinking(true,false));assertTrue(restored.prepare("next",false,true).thinking)
            assertEquals(messages.dropLast(1),regenerateMessages(messages));assertEquals(listOf(Message("user","edited")),editMessages(messages,0,"edited"))
            assertThrows(IllegalArgumentException::class.java) {editMessages(messages,1,"bad")}
            assertThrows(IllegalArgumentException::class.java) {regenerateMessages(emptyList())}
            files.delete("first");assertNull(files.read().firstOrNull {it.id=="first"})
            assertEquals(listOf("print(1)","hello"),codeBlocks("```python\nprint(1)\n```\n~~~text\nhello\n~~~"))
            assertEquals(listOf("print(1)"),codeBlocks("```python\nprint(1)\n````\n"))
            assertEquals(listOf("unfinished"),codeBlocks("~~~text\nunfinished"))
        } finally {dir.deleteRecursively()}
    }
    @Test fun groupingTitleAndSettingsBoundaries() {
        val today=LocalDate.of(2026,11,3);val zone=ZoneId.of("Asia/Singapore")
        fun group(days:Long)=historyGroup(today.minusDays(days).atStartOfDay(zone).toInstant().toEpochMilli(),today,zone)
        assertEquals("Today",group(0));assertEquals("Yesterday",group(1));assertEquals("Previous 7 days",group(2));assertEquals("Previous 7 days",group(7));assertEquals("Earlier",group(8))
        assertEquals("😀".repeat(32)+"…",conversationTitle(listOf(Message("user","😀".repeat(33)))))
        assertEquals("one two",conversationTitle(listOf(Message("user"," one\n two "))))
        assertEquals(0.2,GenerationStyle.Precise.temperature,0.0);assertEquals(0.7,GenerationStyle.Balanced.temperature,0.0);assertEquals(1.0,GenerationStyle.Creative.temperature,0.0)
        val configured=Conversation(AppData("default",emptyList()))
        assertEquals(0.2,configured.prepare("default",false,true).temperature,0.0)
        GenerationStyle.entries.forEach { style -> assertEquals(style.temperature,configured.prepare("request",false,true,style=style).temperature,0.0) }
        val old=Conversation(AppData("old",emptyList()));old.complete("question","reply","")
        val newer=Conversation(AppData("new",emptyList()));assertEquals("old",old.request("continue").first().content);assertEquals("new",newer.request("start").first().content)
    }
}
