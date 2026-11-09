package dev.edge0.chat

import android.app.Activity
import android.app.Instrumentation
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import android.view.MotionEvent
import dev.edge0.chat.core.GenerationStyle
import dev.edge0.chat.core.ThemeMode
import java.io.File

/** Debug-only checks exercise the production ViewModel, JNI and rendered activity. */
class DeviceChecks : Instrumentation() {
    private lateinit var arguments: Bundle
    override fun onCreate(arguments: Bundle?) { this.arguments=arguments?:Bundle();start() }
    override fun onStart() {
        try {
            val activity=startActivitySync(Intent(targetContext,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
            waitForIdleSync();val vm=activity.model
            fun onUi(action:()->Unit)=runOnMainSync(action)
            fun await(label:String,condition:()->Boolean) {
                val deadline=SystemClock.elapsedRealtime()+600_000
                while(true) {var good=false;onUi {good=condition()};if(good)break;check(SystemClock.elapsedRealtime()<deadline) {"Timed out: $label"};SystemClock.sleep(40)}
            }
            fun verify(label:String,condition:()->Boolean) {var good=false;onUi {good=condition()};check(good) {"Failed: $label"};Log.i("Edge0Checks","PASS $label")}
            fun capture(name:String) {
                waitForIdleSync();SystemClock.sleep(200)
                val bitmap=checkNotNull(uiAutomation.takeScreenshot());File(targetContext.cacheDir,name).outputStream().use {bitmap.compress(Bitmap.CompressFormat.PNG,100,it)};bitmap.recycle()
                Log.i("Edge0Checks","screenshot=$name")
            }
            fun node(label:String,parent:AccessibilityNodeInfo?=uiAutomation.rootInActiveWindow):AccessibilityNodeInfo? {
                if(parent==null)return null
                if(parent.text?.toString()==label||parent.contentDescription?.toString()==label)return parent
                for(i in 0 until parent.childCount) node(label,parent.getChild(i))?.let {return it}
                return null
            }
            fun scrollNode(parent:AccessibilityNodeInfo?=uiAutomation.rootInActiveWindow):AccessibilityNodeInfo? {
                if(parent==null)return null
                if(parent.isScrollable)return parent
                for(i in 0 until parent.childCount)scrollNode(parent.getChild(i))?.let {return it}
                return null
            }
            fun click(label:String,settle:Boolean=true) {
                val deadline=SystemClock.elapsedRealtime()+5000;var found:AccessibilityNodeInfo?=null
                while(found==null&&SystemClock.elapsedRealtime()<deadline) {
                    found=node(label)
                    if(found==null) {
                        if(label in listOf("Hide","Show","Copy reasoning"))scrollNode()?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)
                        if(label.startsWith("Copy code")||label=="Copy answer")scrollNode()?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
                        SystemClock.sleep(80)
                    }
                }
                val identified=checkNotNull(found) {"Control not found: $label"}
                var target:AccessibilityNodeInfo?=identified
                while(target!=null&&!target.isClickable)target=target.parent
                if(target!=null) {
                    target.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_SHOW_ON_SCREEN.id);check(target.performAction(AccessibilityNodeInfo.ACTION_CLICK))
                } else {
                    val bounds=Rect();identified.getBoundsInScreen(bounds);check(!bounds.isEmpty)
                    val time=SystemClock.uptimeMillis()
                    val down=MotionEvent.obtain(time,time,MotionEvent.ACTION_DOWN,bounds.exactCenterX(),bounds.exactCenterY(),0)
                    val up=MotionEvent.obtain(time,time+1,MotionEvent.ACTION_UP,bounds.exactCenterX(),bounds.exactCenterY(),0)
                    check(uiAutomation.injectInputEvent(down,true));check(uiAutomation.injectInputEvent(up,true));down.recycle();up.recycle()
                }
                if(settle)waitForIdleSync()
            }
            fun send(prompt:String,tokens:Int=24) {
                onUi {vm.send(prompt,tokens)};await("generation") {!vm.busy}
                verify("native reply and metrics") {vm.ready&&vm.messages.lastOrNull()?.sourceMetrics!=null&&vm.messages.last().metrics.isNotEmpty()}
                verify("temperature reaches native configuration") {vm.lastTemperature==vm.style.temperature}
            }
            fun screenshots() {
                for(theme in listOf(ThemeMode.Light,ThemeMode.Dark)) {
                    onUi {vm.updateTheme(theme)}
                    for(page in MobilePage.entries) {onUi {vm.selectPage(page)};SystemClock.sleep(if(page==MobilePage.Status)1200 else 250);capture("android-${page.name.lowercase()}-${theme.name.lowercase()}.png")}
                }
                onUi {vm.updateTheme(ThemeMode.System);vm.selectPage(MobilePage.Chat)}
            }
            await("asset inspection") {!vm.busy}
            val mode=arguments.getString("mode")?:"status-pages"
            if(mode=="static-screens") {screenshots();finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS static screenshots")});return}
            if(mode=="asset-switch") {
                val badName=checkNotNull(arguments.getString("invalid")) {"Missing focused invalid-asset name"}
                val badDirectory=vm.assetsRoot.resolve(badName).absolutePath
                val bad=checkNotNull(vm.models.firstOrNull {it.directory==badDirectory})
                verify("focused invalid asset is unavailable with reason") {!bad.selectable&&bad.reason.isNotBlank()}
                onUi {vm.selectPage(MobilePage.Models)}
                val reasonDeadline=SystemClock.elapsedRealtime()+5000
                while(node(bad.reason)==null&&SystemClock.elapsedRealtime()<reasonDeadline) {
                    scrollNode()?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD);SystemClock.sleep(80)
                }
                check(node(bad.reason)!=null&&node("Unavailable")!=null) {"Invalid asset reason is not visible under Unavailable"}
                capture("android-b-unavailable.png")
                Log.i("Edge0Checks","PASS actual Models unavailable reason rendered without crash")
                val base=vm.assetsRoot.resolve("35b").absolutePath
                val tiny=vm.assetsRoot.resolve("tiny-a").absolutePath
                verify("focused complete models are inspected") {vm.models.any {it.directory==base&&it.selectable}&&vm.models.any {it.directory==tiny&&it.selectable}}
                onUi {vm.load(base)};await("initial 35B load") {!vm.busy}
                verify("initial 35B is ready") {vm.ready&&vm.selected==base&&(vm.contextCapacity?:0)>0}
                for((index,directory) in listOf(tiny,base).withIndex()) {
                    val label=if(index==0) "tiny" else "35b-return"
                    onUi {vm.load(directory)};await("focused model switch") {!vm.busy}
                    verify("focused switch $label clears chat") {vm.ready&&vm.selected==directory&&vm.messages.isEmpty()&&vm.context.text.startsWith("0 /")}
                    onUi {vm.newConversation()};await("focused new chat") {!vm.busy}
                    verify("focused new chat $label is usable") {vm.ready&&vm.messages.isEmpty()&&vm.context.text.startsWith("0 /")}
                    onUi {vm.toggleThinking(false)};send("Hello.",16)
                    verify("focused native result $label remains valid") {vm.lastStateValid&&vm.lastGeneratedCount>0}
                    onUi {vm.newConversation()};await("focused post-result reset") {!vm.busy}
                    verify("focused post-result reset $label clears usage") {vm.messages.isEmpty()&&vm.context.text.startsWith("0 /")}
                    var reads=0
                    onUi {reads=vm.statusQueries;vm.selectPage(MobilePage.Status)}
                    await("focused status query") {vm.statusQueries>reads&&vm.memory.currentBytes!=null&&vm.system.totalBytes!=null}
                    verify("focused status $label reads native idle data") {vm.progress.phase=="idle"&&(vm.memory.currentBytes?:0)>0&&(vm.system.totalBytes?:0)>0}
                    capture("android-b-status-$label.png")
                    onUi {vm.selectPage(MobilePage.Models)};capture("android-b-loaded-$label.png")
                }
                onUi {vm.unload()};await("focused final unload") {!vm.busy}
                verify("focused finish unloads model") {!vm.ready&&vm.selected.isEmpty()}
                onUi {vm.selectPage(MobilePage.Chat)}
                finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS asset-switch focused checks")});return
            }
            val saved=targetContext.getSharedPreferences("device-checks",0)
            if(mode=="restore") {
                val id=saved.getString("historyId",null)
                verify("history survives process restart") {vm.historyEntries.any {it.id==id&&it.archive?.messages?.lastOrNull()?.metrics?.isNotEmpty()==true}}
                verify("cumulative survives process restart") {vm.cumulativeTotals.requests==saved.getLong("requests",0)}
            }
            val name=arguments.getString("model")?:"35b"
            val directory=vm.assetsRoot.resolve(name).absolutePath
            verify("existing private assets inspected") {vm.models.any {it.directory==directory&&it.selectable&&it.totalBytes!=null}}
            if(mode=="presentation") {
                verify("initial chooser without stale model") {!vm.navigation.hasLoadedModel&&!vm.navigation.otherLocationExpanded}
                onUi {vm.toggleOtherLocation()};verify("manual location state expands") {vm.navigation.otherLocationExpanded};onUi {vm.toggleOtherLocation()}
            }
            onUi {vm.load(directory)};await("model load") {!vm.busy};verify("native model load and context") {vm.ready&&(vm.contextCapacity?:0)>0}
            if(mode=="d-visual"||mode=="d-continue"||mode=="d-stop") {
                if(mode!="d-stop") {
                val id=checkNotNull(saved.getString("historyId",null))
                onUi {vm.switchConversation(id)}
                val expected=vm.messages.filter {it.role=="assistant"}.flatMap {dev.edge0.chat.core.codeBlocks(it.content)}
                check(expected.isNotEmpty()) {"Missing persisted real code block"}
                if(mode=="d-visual") {
                    click("Copy code");SystemClock.sleep(100)
                    val copied=targetContext.getSystemService(ClipboardManager::class.java).primaryClip?.getItemAt(0)?.text?.toString()
                    check(copied in expected) {"Code header clipboard mismatch"};Log.i("Edge0Checks","PASS actual code header copy")
                }
                onUi {vm.selectPage(MobilePage.Settings)}
                if(mode=="d-visual") {click("Balanced · 0.7");verify("segmented style selection") {vm.style==GenerationStyle.Balanced}}
                click("Precise · 0.2");verify("segmented style restored") {vm.style==GenerationStyle.Precise}
                click("Dark");verify("segmented dark selection") {vm.theme==ThemeMode.Dark}
                click("Light");verify("segmented light selection") {vm.theme==ThemeMode.Light}
                for(theme in listOf(ThemeMode.Light,ThemeMode.Dark)) {
                    onUi {vm.updateTheme(theme);vm.selectPage(MobilePage.Chat)};capture("android-d-chat-${theme.name.lowercase()}.png")
                    onUi {vm.selectPage(MobilePage.Settings)};capture("android-d-settings-${theme.name.lowercase()}.png")
                }
                onUi {vm.updateTheme(ThemeMode.System);vm.selectPage(MobilePage.Chat);vm.draft="This long draft keeps the composer action visible. ".repeat(24)}
                waitForIdleSync();SystemClock.sleep(200)
                val fieldBounds=Rect();val actionBounds=Rect()
                checkNotNull(node("Message")).getBoundsInScreen(fieldBounds);checkNotNull(node("Send")).getBoundsInScreen(actionBounds)
                check(fieldBounds.right<=actionBounds.left) {"Long draft overlaps the composer action"}
                capture("android-d-composer-long.png");Log.i("Edge0Checks","PASS long draft reserves action space")
                }
                onUi {vm.updateTheme(ThemeMode.System);vm.selectPage(MobilePage.Chat);vm.newConversation()};await("composer regression new chat") {!vm.busy}
                onUi {vm.toggleThinking(false);vm.draft="Write a long numbered list about computers."}
                click("Send");await("inner composer streamed output") {vm.messages.lastOrNull()?.content?.isNotEmpty()==true}
                verify("inner composer send starts native generation") {vm.generating&&vm.messages.first().content=="Write a long numbered list about computers."}
                capture("android-d-composer-active.png")
                click("Stop",settle=false)
                verify("inner composer immediately enters stopping") {vm.stopping&&vm.feedback.buttonLabel=="Stopping…"}
                val stoppingControl=node("Stopping…")
                if(stoppingControl!=null) {check(!stoppingControl.isEnabled);Log.i("Edge0Checks","PASS rendered Stopping control is disabled")}
                await("inner composer cancellation") {!vm.busy}
                verify("inner composer stop preserves valid session") {vm.lastStopReason=="cancelled"&&vm.lastStateValid&&vm.messages.last().stopped}
                onUi {vm.unload()};await("D final unload") {!vm.busy};verify("D leaves model unloaded") {!vm.ready&&vm.selected.isEmpty()}
                finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS D visual focused checks")});return
            }
            if(mode=="repaint-screens") {
                val id=checkNotNull(saved.getString("historyId",null))
                onUi {vm.switchConversation(id);vm.setCumulative(true)}
                screenshots()
                onUi {vm.unload()};await("unload after screenshots") {!vm.busy}
                verify("screenshots leave model unloaded") {!vm.ready&&vm.selected.isEmpty()}
                finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS repaint screenshots without generation")});return
            }
            if(mode=="restore") {
                val id=checkNotNull(saved.getString("historyId",null));onUi {vm.switchConversation(id)}
                val before=vm.messages.size;send("Reply with one word: hello.",16);verify("restored conversation continues") {vm.messages.size==before+2}
                screenshots();finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS restart and screenshots")});return
            }
            if(mode=="engine-default") {
                onUi {vm.toggleThinking(false);vm.send("Write an 850-word beginner guide to growing tomatoes in containers. Finish with END OF GUIDE.")}
                await("engine default generation") {!vm.busy};verify("normal request has engine generation limit") {vm.lastGeneratedCount>512&&vm.lastStopReason!="length"&&vm.lastStateValid};screenshots()
                onUi {vm.newConversation()};await("new thinking chat") {!vm.busy};onUi {vm.toggleThinking(true)};send("What is 2 + 2? Keep the answer brief.",64)
                verify("locked on text") {vm.thinking&&!vm.canChangeThinking};capture("android-engine-default-thinking.png")
                finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS engine-default checks")});return
            }
            // Compatibility modes preserve the original native/behavior checks;
            // approved bottom tabs replace the old Models/Back header controls.
            if(mode in setOf("tiny","full","presentation","revision","compare")) {
                if(mode=="compare") {
                    onUi {vm.toggleThinking(false)};send("Write a concise paragraph explaining how a rainbow forms.",96)
                } else if(mode=="presentation") {
                    verify("successful load enters chat") {vm.page==MobilePage.Chat&&vm.ready};capture("android-presentation-loaded.png")
                    onUi {vm.load(vm.assetsRoot.resolve("missing-not-created").absolutePath)};await("failed reload") {!vm.busy}
                    verify("failed reload has no stale model") {vm.selected.isEmpty()&&!vm.ready&&vm.page==MobilePage.Models}
                    onUi {vm.load(directory)};await("reload") {!vm.busy};onUi {vm.toggleThinking(true)};send("What is 2 + 2? Keep the answer brief.",64)
                    verify("reasoning initially expanded") {vm.messages.last().reasoning.isNotEmpty()&&vm.reasoningBlocks.forMessage(1).expanded&&!vm.canChangeThinking&&vm.thinking}
                    val before=vm.messages
                    click("Hide");verify("real hide changes presentation only") {!vm.reasoningBlocks.forMessage(1).expanded&&vm.messages==before}
                    click("Models");verify("models preserves conversation") {vm.page==MobilePage.Models&&vm.messages==before}
                    click("Chat");verify("chat preserves collapsed reasoning") {vm.page==MobilePage.Chat&&!vm.reasoningBlocks.forMessage(1).expanded&&vm.messages==before}
                    click("Show");verify("real show expands") {vm.reasoningBlocks.forMessage(1).expanded}
                    onUi {vm.newConversation()};await("new off chat") {!vm.busy};onUi {vm.toggleThinking(false)};send("Hello",16)
                    verify("thinking off has no block") {vm.messages.last().reasoning.isEmpty()&&!vm.thinking&&!vm.canChangeThinking};capture("android-presentation-off.png")
                } else if(mode=="revision") {
                    verify("new chat thinking unlocked") {vm.canChangeThinking};onUi {vm.toggleThinking(false)};send("Hello",16)
                    verify("history thinking locked") {!vm.canChangeThinking};onUi {vm.toggleThinking(true)};verify("locked thinking unchanged") {!vm.thinking}
                    onUi {vm.send("Write a long numbered list about computers.",512)};await("partial output") {vm.messages.last().content.isNotEmpty()};onUi {vm.stop()}
                    verify("immediate stopping feedback") {vm.stopping&&vm.feedback.buttonLabel=="Stopping…"};await("stopped") {!vm.busy}
                    verify("stopped marker and native state") {vm.messages.last().stopped&&vm.lastStopReason=="cancelled"&&vm.lastStateValid&&vm.status=="Ready"}
                    onUi {vm.newConversation()};await("new chat") {!vm.busy};verify("thinking unlocked again") {vm.canChangeThinking}
                } else {
                    verify("invalid assets disabled") {vm.models.any {!it.selectable&&it.reason.isNotEmpty()}}
                    onUi {vm.toggleThinking(false)};send(if(mode=="tiny") "Hello" else "Reply in one short sentence: what is an on-device language model?")
                    verify("thinking off") {vm.messages.last().reasoning.isEmpty()}
                    if(mode=="tiny") {
                        val second=vm.assetsRoot.resolve(if(name=="tiny-a") "tiny-b" else "tiny-a").absolutePath
                        onUi {vm.load(second)};await("switch fixture") {!vm.busy};verify("model switch") {vm.ready&&vm.selected==second&&vm.messages.isEmpty()};send("Hello after switching models",16)
                    } else {
                        if(vm.thinkingSupported) {
                            onUi {vm.newConversation()};await("thinking chat") {!vm.busy};onUi {vm.toggleThinking(true)};send("What is 13 times 17? Explain briefly.",160)
                            verify("thinking on") {vm.messages.last().reasoning.isNotEmpty()};onUi {vm.newConversation()};await("off chat") {!vm.busy};onUi {vm.toggleThinking(false)}
                        }
                        for(prompt in listOf("Remember the phrase cobalt meadow.","What phrase did I ask you to remember?","Give one use for it as a story title.","Now suggest a second title.","Which of those titles is shorter?"))send(prompt,48)
                        verify("five continued turns") {vm.messages.size==10}
                    }
                    onUi {vm.send("Write a very long numbered list of interesting facts about computers.",4096)}
                    await("streaming partial") {vm.messages.lastOrNull()?.let {it.content.isNotEmpty()||it.reasoning.isNotEmpty()}==true};onUi {vm.stop()};await("cancel") {!vm.busy}
                    verify("cancelled state valid") {vm.lastStopReason=="cancelled"&&vm.lastStateValid&&vm.ready&&vm.messages.last().stopped};send("Reply with a short greeting.",16)
                    val count=vm.messages.size
                    targetContext.startActivity(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));SystemClock.sleep(400)
                    targetContext.startActivity(Intent(targetContext,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT));waitForIdleSync()
                    verify("background conversation retained") {vm.messages.size==count};onUi {vm.newConversation()};await("reset") {!vm.busy}
                    verify("new conversation empty") {vm.messages.isEmpty()&&vm.ready&&vm.appData.suggestions.size==4}
                    send(if(mode=="tiny") "New chat" else "What did I say before this new conversation?",48)
                    if(mode=="full")send("Show a Markdown heading, a list, bold and italic, a link, fenced Python code, inline formula $ x_i * y $, block formula $$ x^2+y^2 $$, and USD 5. Be concise.",256)
                }
                finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS $mode compatibility checks")});return
            }
            if(mode!="finish-status") {
            if(mode!="continue-status") {
            onUi {vm.toggleThinking(false)}
            send("Remember the phrase cobalt meadow. Acknowledge briefly.",20)
            val firstCount=vm.messages.size
            for(prompt in listOf("What phrase did I ask you to remember?","Give one short story title using that phrase.","Give a second short title.","Which title is shorter?","Repeat the phrase in two words."))send(prompt,20)
            verify("five continued turns") {vm.messages.size==firstCount+10&&(vm.lastMetrics?.reusedTokens?:0)>0}
            val previous=vm.messages.toList()
            click("Copy answer")
            SystemClock.sleep(100)
            val clip=targetContext.getSystemService(ClipboardManager::class.java).primaryClip?.getItemAt(0)?.text?.toString()
            check(clip!=null&&vm.messages.any {it.role=="assistant"&&it.content==clip}) {"Answer clipboard mismatch"};Log.i("Edge0Checks","PASS actual answer copy")
            onUi {vm.regenerate()};await("regenerate") {!vm.busy};verify("regenerate preserves earlier history") {vm.messages.dropLast(1)==previous.dropLast(1)}
            onUi {vm.editAndSend(0,"Remember the phrase silver fern. Acknowledge briefly.")};await("edit resend") {!vm.busy};verify("edit deletes following messages") {vm.messages.size==2&&vm.messages.first().content.contains("silver fern")}
            onUi {vm.send("Write a long numbered list of facts about computers.",256)}
            await("streaming output") {vm.messages.lastOrNull()?.content?.isNotEmpty()==true}
            verify("stream visible before completion") {vm.generating}
            var liveQueries=0
            onUi {liveQueries=vm.statusQueries;vm.selectPage(MobilePage.Status)}
            await("native status during generation") {vm.statusQueries>liveQueries&&vm.progress.phase!="idle"}
            verify("concurrent status keeps generation active") {vm.generating&&vm.memory.currentBytes!=null}
            capture("android-status-decoding.png")
            onUi {vm.selectPage(MobilePage.Chat);vm.stop()};verify("immediate stopping feedback") {vm.stopping&&vm.feedback.buttonLabel=="Stopping…"}
            await("cancellation") {!vm.busy};verify("cancelled native state remains usable") {vm.lastStopReason=="cancelled"&&vm.lastStateValid&&vm.messages.last().stopped}
            send("Reply with a short greeting.",16)
            onUi {vm.selectPage(MobilePage.Status)};SystemClock.sleep(1400)
            verify("native status read") {vm.statusQueries>0&&vm.memory.currentBytes!=null&&vm.system.totalBytes!=null}
            var queries=0;onUi {vm.selectPage(MobilePage.Chat);queries=vm.statusQueries};SystemClock.sleep(1400)
            verify("off-page polling stops") {vm.statusQueries==queries}
            for(style in GenerationStyle.entries) {onUi {vm.updateStyle(style)};send("Reply with the word hello.",8)}
            onUi {vm.updateStyle(GenerationStyle.Precise);vm.newConversation()};await("new conversation") {!vm.busy}
            verify("new chat resets usage and unlocks thinking") {vm.messages.isEmpty()&&vm.canChangeThinking&&vm.context.text.startsWith("0 /")&&vm.appData.suggestions.size==4}
            onUi {vm.toggleThinking(true)};send("What is two plus two? Answer briefly.",80)
            } else {
                val reasoningArchive=checkNotNull(vm.historyEntries.firstOrNull {it.archive?.messages?.lastOrNull()?.reasoning?.isNotEmpty()==true}) {"No saved reasoning conversation"}
                onUi {vm.switchConversation(reasoningArchive.id)}
            }
            verify("thinking has separate expanded block") {vm.messages.last().reasoning.isNotEmpty()&&vm.reasoningBlocks.forMessage(1).expanded&&!vm.canChangeThinking&&vm.thinking}
            click("Hide");verify("real disclosure hides") {!vm.reasoningBlocks.forMessage(1).expanded};capture("android-reasoning-collapsed.png");click("Show");verify("real disclosure expands") {vm.reasoningBlocks.forMessage(1).expanded}
            click("Copy reasoning");SystemClock.sleep(100)
            val reasoningClip=targetContext.getSystemService(ClipboardManager::class.java).primaryClip?.getItemAt(0)?.text?.toString()
            check(reasoningClip==vm.messages.last().reasoning);Log.i("Edge0Checks","PASS actual reasoning copy")
            onUi {vm.newConversation()};await("new non-thinking conversation") {!vm.busy};onUi {vm.toggleThinking(false)}
            send("Show a short Markdown heading, one bold word, a fenced Python code block with print(1), inline formula $ x+y $, block formula $$ x^2 $$, and USD 5. Keep it brief.",140)
            } else {
                val markdownArchive=checkNotNull(vm.historyEntries.firstOrNull {it.archive?.messages?.lastOrNull()?.content?.let {content -> dev.edge0.chat.core.codeBlocks(content)}?.isNotEmpty()==true}) {"No saved Markdown conversation"}
                onUi {vm.switchConversation(markdownArchive.id)}
            }
            check(dev.edge0.chat.core.codeBlocks(vm.messages.last().content).isNotEmpty()) {"Model did not produce the requested fenced code"}
            click("Copy code");SystemClock.sleep(100)
            val code=targetContext.getSystemService(ClipboardManager::class.java).primaryClip?.getItemAt(0)?.text?.toString()
            check(dev.edge0.chat.core.codeBlocks(vm.messages.last().content).contains(code));Log.i("Edge0Checks","PASS actual code copy")
            verify("thinking off remains locked") {!vm.thinking&&!vm.canChangeThinking&&vm.messages.last().reasoning.isEmpty()}
            await("conversation persisted") {vm.historyEntries.any {it.archive?.messages==vm.messages}}
            val id=vm.historyEntries.first {it.archive?.messages==vm.messages}.id
            onUi {vm.historyOpen=true};SystemClock.sleep(200);capture("android-history.png");onUi {vm.historyOpen=false}
            var before=0;onUi {before=vm.messages.size}
            targetContext.startActivity(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));SystemClock.sleep(300)
            targetContext.startActivity(Intent(targetContext,MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT));waitForIdleSync()
            verify("background conversation retained") {vm.messages.size==before}
            onUi {vm.newConversation()};await("second saved chat") {!vm.busy};onUi {vm.setSystemPrompt("Reply in brief English sentences.");vm.newConversation()};await("custom system prompt") {!vm.busy}
            send("Hello.",8);await("second archive persisted") {vm.historyEntries.any {it.archive?.messages==vm.messages}}
            val newId=vm.historyEntries.first {it.archive?.messages==vm.messages}.id
            onUi {vm.switchConversation(id)};verify("history switch retains metrics") {vm.messages.last().metrics.isNotEmpty()}
            onUi {vm.deleteConversation(newId)};await("history delete") {vm.historyEntries.none {it.id==newId}}
            onUi {vm.setSystemPrompt("")}
            saved.edit().putString("historyId",id).putLong("requests",vm.cumulativeTotals.requests).commit()
            screenshots()
            onUi {vm.selectPage(MobilePage.Models);vm.unload()};await("unload") {!vm.busy};verify("model unload") {!vm.ready&&vm.selected.isEmpty()}
            onUi {vm.load(directory)};await("model reload") {!vm.busy};verify("model reload") {vm.ready}
            onUi {vm.newConversation()};await("suggestion chat") {!vm.busy};onUi {vm.toggleThinking(false)}
            val suggestion=vm.appData.suggestions.first()
            click(suggestion)
            await("suggestion streamed") {vm.messages.lastOrNull()?.content?.isNotEmpty()==true}
            verify("actual shared suggestion sends") {vm.messages.first().content==suggestion&&vm.generating}
            click("Stop");await("suggestion stopped") {!vm.busy}
            verify("actual Stop control cancels native request") {vm.lastStopReason=="cancelled"&&vm.lastStateValid&&vm.messages.last().stopped}
            onUi {vm.newConversation()};await("remaining Markdown chat") {!vm.busy}
            send("Reply exactly with this one Markdown bullet and nothing else: - *Italic* [Example](https://example.com).",128)
            verify("real bullet italic and link output") {Regex("(?m)^[-*+]\\s").containsMatchIn(vm.messages.last().content)&&vm.messages.last().content.contains("*Italic*")&&vm.messages.last().content.contains("[Example](https://example.com)")}
            capture("android-markdown-extra.png")
            saved.edit().putLong("requests",vm.cumulativeTotals.requests).commit()
            Log.i("Edge0Checks","SKIP second valid model and pre-existing invalid assets: unavailable")
            finish(Activity.RESULT_OK,Bundle().apply {putString("result","PASS status-pages device checks")})
        } catch(error:Throwable) {Log.e("Edge0Checks","FAIL device checks",error);finish(Activity.RESULT_CANCELED,Bundle().apply {putString("error",error.toString())})}
    }
}
