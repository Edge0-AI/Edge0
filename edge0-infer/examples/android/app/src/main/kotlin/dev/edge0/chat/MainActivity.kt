package dev.edge0.chat

import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.ViewModelProvider
import com.mikepenz.markdown.m3.Markdown
import com.mikepenz.markdown.m3.markdownColor
import com.mikepenz.markdown.model.markdownAnimations
import com.mikepenz.markdown.model.rememberMarkdownState
import com.mikepenz.markdown.compose.components.markdownComponents
import com.mikepenz.markdown.compose.elements.MarkdownCodeFence
import com.mikepenz.markdown.compose.elements.MarkdownCodeBlock
import dev.edge0.chat.core.*
import dev.edge0.chat.ui.theme.*
import java.time.LocalDate
import java.time.ZoneId

class MainActivity : ComponentActivity() {
    internal lateinit var model: ChatViewModel; private set
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState); enableEdgeToEdge(); model = ViewModelProvider(this)[ChatViewModel::class.java]
        setContent {
            val dark = model.theme==ThemeMode.Dark || (model.theme==ThemeMode.System && isSystemInDarkTheme())
            SideEffect {
                val style = if(dark) SystemBarStyle.dark(android.graphics.Color.TRANSPARENT) else SystemBarStyle.light(android.graphics.Color.TRANSPARENT,android.graphics.Color.TRANSPARENT)
                enableEdgeToEdge(statusBarStyle=style,navigationBarStyle=style)
            }
            EdgeTheme(model.theme) { AppScreen(model) }
        }
    }
    override fun onStart() { super.onStart(); if (::model.isInitialized) model.setForeground(true) }
    override fun onStop() { if (::model.isInitialized) model.setForeground(false); super.onStop() }
}
@Composable private fun OutlineIcon(name: String) { Icon(remember(name) { appIcon(name) }, name, Modifier.size(Tokens.icon)) }
@Composable private fun SmallIconButton(name: String, enabled: Boolean = true, onClick: () -> Unit) {
    IconButton(onClick, enabled=enabled, modifier=Modifier.size(Tokens.touch)) { OutlineIcon(name) }
}
@Composable private fun StableMarkdown(content: String, gray: Boolean = false, actionsEnabled: Boolean = true) {
    val p = LocalPalette.current
    val protected = remember(content) { protectMath(content) }
    val state = rememberMarkdownState(content=protected,retainState=true)
    val originalCodes = remember(content) { codeBlocks(content) }
    val components = remember(content,actionsEnabled) { markdownComponents(
        codeFence = { model ->
            MarkdownCodeFence(model.content,model.node,model.typography.code) { code,language,style ->
                val raw = codeBlocks(model.content.substring(model.node.startOffset,model.node.endOffset)).firstOrNull() ?: code
                val protectedFormula = language=="text" && raw !in originalCodes && (raw.trimStart().startsWith("$$") || raw.trimStart().startsWith("\\["))
                MessageCodeBlock(code,raw,language,style,actionsEnabled,!protectedFormula)
            }
        },
        codeBlock = { model -> MarkdownCodeBlock(model.content,model.node,model.typography.code) { code,language,style -> MessageCodeBlock(code,code,language,style,actionsEnabled,true) } },
    ) }
    Markdown(markdownState=state,modifier=Modifier.fillMaxWidth(),colors=markdownColor(text=if(gray) p.text2 else p.text),components=components,animations=markdownAnimations(animateTextSize={this}))
}
@Composable private fun MessageCodeBlock(code: String,copyText: String,language: String?,style: TextStyle,actionsEnabled: Boolean,showHeader: Boolean) {
    val p=LocalPalette.current;val context=LocalContext.current
    Column(Modifier.fillMaxWidth().clip(Tokens.mediumShape).border(Tokens.hairline,p.border,Tokens.mediumShape).background(p.surface)) {
        if(showHeader) {
            Row(Modifier.fillMaxWidth().background(p.surface2).padding(start=Tokens.codeInset,end=Tokens.sm),verticalAlignment=Alignment.CenterVertically) {
                Text(language?.takeIf {it.isNotBlank()} ?: "text",color=p.text3,style=MaterialTheme.typography.labelSmall.copy(fontFamily=FontFamily.Monospace),modifier=Modifier.weight(1f))
                IconButton(onClick={context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("Edge0",copyText))},enabled=actionsEnabled,modifier=Modifier.size(Tokens.composerAction)) {
                    Icon(remember {appIcon("Copy")},"Copy code",Modifier.size(Tokens.lg),tint=p.text3)
                }
            }
            HorizontalDivider(thickness=Tokens.hairline,color=p.border)
        }
        SelectionContainer {
            Text(if(showHeader) code else copyText,style=style.copy(color=p.text,fontFamily=FontFamily.Monospace,fontSize=Tokens.codeFont,lineHeight=Tokens.codeLineHeight),softWrap=false,
                modifier=Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal=Tokens.codeInset,vertical=Tokens.md))
        }
    }
}
@Composable private fun AppScreen(vm: ChatViewModel) {
    val p=LocalPalette.current
    val drawer = rememberDrawerState(DrawerValue.Closed)
    LaunchedEffect(vm.historyOpen) { if(vm.historyOpen) drawer.open() else drawer.close() }
    LaunchedEffect(drawer.currentValue) { if(drawer.currentValue==DrawerValue.Closed) vm.historyOpen=false }
    ModalNavigationDrawer(drawerState=drawer,gesturesEnabled=vm.historyOpen,drawerContent={
        ModalDrawerSheet(modifier=Modifier.width(Tokens.drawerWidth),drawerContainerColor=p.sidebar) {
            HistoryDrawer(vm)
        }
    }) {
        Column(Modifier.fillMaxSize().background(p.bg).windowInsetsPadding(WindowInsets.safeDrawing)) {
            Box(Modifier.weight(1f)) {
                when(vm.page) {
                    MobilePage.Chat -> ChatPage(vm)
                    MobilePage.Models -> ModelsPage(vm)
                    MobilePage.Status -> StatusPage(vm)
                    MobilePage.Settings -> SettingsPage(vm)
                }
            }
            Row(Modifier.fillMaxWidth().background(p.surface).border(Tokens.hairline,p.border).padding(vertical=Tokens.sm),horizontalArrangement=Arrangement.SpaceAround) {
                MobilePage.entries.forEach { page ->
                    val selected=vm.page==page
                    Column(Modifier.weight(1f).clickable { vm.selectPage(page) }.padding(Tokens.xs),horizontalAlignment=Alignment.CenterHorizontally) {
                        CompositionLocalProvider(LocalContentColor provides if(selected) p.accent else p.text3) {
                            OutlineIcon(page.name); Spacer(Modifier.height(Tokens.xs)); Text(page.name,style=MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
    }
}
@Composable private fun HistoryDrawer(vm: ChatViewModel) {
    val p=LocalPalette.current
    Column(Modifier.fillMaxHeight().padding(Tokens.lg)) {
        TextButton(onClick={vm.newConversation();vm.historyOpen=false},enabled=!vm.busy) { OutlineIcon("New chat"); Spacer(Modifier.width(Tokens.sm));Text("New chat") }
        val zone=ZoneId.systemDefault(); val today=LocalDate.now(zone)
        val groups=vm.historyEntries.groupBy { it.archive?.let { a -> historyGroup(a.updatedMillis,today,zone) } ?: "Unable to open" }
        LazyColumn(verticalArrangement=Arrangement.spacedBy(Tokens.xs)) {
            listOf("Today","Yesterday","Previous 7 days","Earlier","Unable to open").forEach { group ->
                val entries=groups[group].orEmpty()
                if(entries.isNotEmpty()) {
                    item { Text(group,color=p.text3,style=MaterialTheme.typography.labelSmall,modifier=Modifier.padding(top=Tokens.xl,bottom=Tokens.sm)) }
                    items(entries,key={it.id}) { entry ->
                        Row(Modifier.fillMaxWidth().clickable(enabled=entry.archive!=null&&!vm.busy) {vm.switchConversation(entry.id)},verticalAlignment=Alignment.CenterVertically) {
                            Text(entry.archive?.title ?: "Unable to open",modifier=Modifier.weight(1f),maxLines=1,overflow=TextOverflow.Ellipsis,style=MaterialTheme.typography.bodyMedium)
                            SmallIconButton("Delete",!vm.busy) {vm.deleteConversation(entry.id)}
                        }
                    }
                }
            }
        }
    }
}
@Composable private fun ChatPage(vm: ChatViewModel) {
    val p=LocalPalette.current
    Column(Modifier.fillMaxSize().background(p.surface)) {
        Row(Modifier.fillMaxWidth().padding(horizontal=Tokens.sm).height(Tokens.header),verticalAlignment=Alignment.CenterVertically) {
            SmallIconButton("History",!vm.busy) {vm.historyOpen=true}
            Column(Modifier.weight(1f).padding(horizontal=Tokens.sm)) {
                Text(vm.title,style=MaterialTheme.typography.titleSmall,fontWeight=FontWeight.SemiBold,maxLines=1,overflow=TextOverflow.Ellipsis)
                vm.models.firstOrNull {it.directory==vm.selected}?.let {Text("${it.modelId} · CPU",style=MaterialTheme.typography.labelSmall,color=p.text3)}
            }
            SmallIconButton("New chat",!vm.busy,vm::newConversation)
        }
        if(vm.selected.isEmpty()) {
            Column(Modifier.fillMaxSize().padding(Tokens.xxl),verticalArrangement=Arrangement.Center,horizontalAlignment=Alignment.CenterHorizontally) {
                Text("Load a model to start chatting",style=MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(Tokens.lg)); Button(onClick=vm::openModels) {Text("Choose a model")}
            }
            return
        }
        Row(Modifier.fillMaxWidth().border(Tokens.hairline,p.border).padding(horizontal=Tokens.lg,vertical=Tokens.xs),verticalAlignment=Alignment.CenterVertically) {
            vm.context.ringPercent?.let {CircularProgressIndicator(progress={it/100f},modifier=Modifier.size(Tokens.ring),color=p.accent,trackColor=p.border,strokeWidth=Tokens.hairline);Spacer(Modifier.width(Tokens.sm))}
            Text(vm.context.text,color=p.text3,style=MaterialTheme.typography.labelSmall,modifier=Modifier.weight(1f))
            Text("Thinking",style=MaterialTheme.typography.labelSmall)
            Switch(checked=vm.thinking,onCheckedChange=vm::toggleThinking,enabled=vm.canChangeThinking,modifier=Modifier.alpha(if(vm.canChangeThinking) 1f else 0.45f),
                colors=SwitchDefaults.colors(disabledCheckedTrackColor=p.accent,disabledCheckedThumbColor=p.accentInk,disabledUncheckedTrackColor=p.surface3,disabledUncheckedThumbColor=p.text3))
            Text(if(vm.thinking) "On" else "Off",style=MaterialTheme.typography.labelSmall)
        }
        if(vm.messages.isNotEmpty()) Text("Applies to a new chat",color=p.text3,style=MaterialTheme.typography.labelSmall,modifier=Modifier.align(Alignment.End).padding(horizontal=Tokens.lg))
        val scroll=androidx.compose.foundation.lazy.rememberLazyListState()
        LaunchedEffect(vm.messages) {if(vm.messages.isNotEmpty()) scroll.scrollToItem(vm.messages.size)}
        LazyColumn(Modifier.weight(1f).padding(horizontal=Tokens.lg),state=scroll,verticalArrangement=Arrangement.spacedBy(Tokens.md)) {
            if(vm.messages.isEmpty()) items(vm.appData.suggestions,key={it}) {suggestion -> OutlinedButton(onClick={vm.send(suggestion)},enabled=vm.ready&&!vm.busy,modifier=Modifier.fillMaxWidth()) {Text(suggestion)} }
            itemsIndexed(vm.messages,key={index,_->"message-$index"}) {index,message->MessageContent(vm,index,message)}
            item {Spacer(Modifier.height(Tokens.xs))}
        }
        if(vm.status.isNotEmpty()&&vm.status!="Ready"&&vm.status!="Generating…") Text(vm.status,color=if(vm.stopping) p.text2 else p.error,style=MaterialTheme.typography.labelSmall,modifier=Modifier.padding(horizontal=Tokens.lg))
        ChatComposer(vm)
    }
}
@Composable private fun ChatComposer(vm: ChatViewModel) {
    val p=LocalPalette.current
    val width=if(vm.stopping) Tokens.stoppingActionWidth else Tokens.composerAction
    val enabled=if(vm.generating) !vm.stopping else vm.ready&&!vm.busy&&vm.draft.isNotBlank()
    Box(Modifier.fillMaxWidth().imePadding().padding(Tokens.md)) {
        Box(Modifier.fillMaxWidth().heightIn(min=Tokens.composerMinHeight).clip(Tokens.composerShape).background(p.surface).border(Tokens.hairline,p.borderStrong,Tokens.composerShape)
            .padding(start=Tokens.lg,end=Tokens.composerInset,top=Tokens.sm,bottom=Tokens.composerInset)) {
            BasicTextField(value=vm.draft,onValueChange={vm.draft=it},enabled=vm.ready&&!vm.busy,maxLines=Tokens.composerMaxLines,
                textStyle=MaterialTheme.typography.bodyLarge.copy(color=p.text),cursorBrush=SolidColor(p.accent),
                modifier=Modifier.fillMaxWidth().align(Alignment.TopStart).padding(end=width+Tokens.sm,top=Tokens.xs,bottom=Tokens.xs).semantics {contentDescription="Message"},
                decorationBox={inner->Box {if(vm.draft.isEmpty()) Text("Message",color=p.text3,style=MaterialTheme.typography.bodyLarge);inner()}})
            Box(Modifier.align(Alignment.BottomEnd).width(width).height(Tokens.composerAction).clip(Tokens.circleShape)
                .background(if(enabled) p.accent else p.surface3).clickable(enabled=enabled,role=Role.Button) {if(vm.generating) vm.stop() else vm.send()}
                .semantics {contentDescription=if(vm.stopping) "Stopping…" else if(vm.generating) "Stop" else "Send"},contentAlignment=Alignment.Center) {
                CompositionLocalProvider(LocalContentColor provides if(enabled) p.accentInk else p.text3) {
                    if(vm.stopping) Row(verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(Tokens.xs)) {
                        CircularProgressIndicator(Modifier.size(Tokens.lg),strokeWidth=Tokens.hairline,color=p.text3)
                        Text("Stopping…",style=MaterialTheme.typography.labelSmall)
                    } else Icon(remember(vm.generating) {appIcon(if(vm.generating) "Stop" else "Send")},contentDescription=null,modifier=Modifier.size(Tokens.icon))
                }
            }
        }
    }
}
@Composable private fun MessageContent(vm: ChatViewModel,index: Int,message: Message) {
    val p=LocalPalette.current; val context=LocalContext.current
    fun copy(value: String) { (context.getSystemService(ClipboardManager::class.java)).setPrimaryClip(ClipData.newPlainText("Edge0",value)) }
    var editing by remember { mutableStateOf(false) }; var edit by remember(message.content) { mutableStateOf(message.content) }
    Column(Modifier.fillMaxWidth(),horizontalAlignment=if(message.role=="user") Alignment.End else Alignment.Start) {
        if(message.role=="user") {
            Text(message.content,modifier=Modifier.widthIn(max=Tokens.userMax).background(p.surface3,Tokens.largeShape).padding(Tokens.md))
            SmallIconButton("Edit",!vm.busy) {editing=true}
        } else {
            val disclosure=vm.reasoningBlocks.forMessage(index)
            if(disclosure.visible(message.reasoning,vm.thinking)) {
                Column(Modifier.fillMaxWidth().background(p.surface2,Tokens.mediumShape).border(Tokens.hairline,p.border,Tokens.mediumShape).padding(Tokens.md)) {
                    Row(verticalAlignment=Alignment.CenterVertically) {Text("Thinking",color=p.text2,style=MaterialTheme.typography.bodySmall,modifier=Modifier.weight(1f));TextButton(onClick={vm.toggleReasoning(index)}) {Text(if(disclosure.expanded) "Hide" else "Show",color=p.text2)}; SmallIconButton("Copy reasoning",!vm.busy) {copy(message.reasoning)} }
                    if(disclosure.expanded) StableMarkdown(message.reasoning,true,!vm.busy)
                }
                Spacer(Modifier.height(Tokens.sm))
            }
            val state=assistantState(index==vm.messages.lastIndex,vm.generating,message.content.isNotEmpty(),disclosure.visible(message.reasoning,vm.thinking))
            if(state.pending) PendingDots() else StableMarkdown(message.content,actionsEnabled=!vm.busy)
            if(state.actions) Row(verticalAlignment=Alignment.CenterVertically) {
                SmallIconButton("Copy answer",!vm.busy) {copy(message.content)}
                if(index==vm.messages.lastIndex) SmallIconButton("Regenerate",!vm.busy,vm::regenerate)
                Text(message.metrics,color=p.text3,style=MaterialTheme.typography.labelSmall)
            }
            if(message.stopped) Text("Stopped",color=p.text3,style=MaterialTheme.typography.labelSmall)
        }
    }
    if(editing) AlertDialog(onDismissRequest={editing=false},title={Text("Edit message")},text={OutlinedTextField(edit,{edit=it},maxLines=Tokens.composerMaxLines)},
        confirmButton={TextButton(onClick={editing=false;vm.editAndSend(index,edit)},enabled=edit.isNotBlank()) {Text("Send")}},dismissButton={TextButton(onClick={editing=false}) {Text("Cancel")}})
}
@Composable private fun PageColumn(title: String,content: @Composable ColumnScope.()->Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(Tokens.lg),verticalArrangement=Arrangement.spacedBy(Tokens.md)) {
        Text(title,style=MaterialTheme.typography.headlineMedium,fontWeight=FontWeight.SemiBold,modifier=Modifier.padding(vertical=Tokens.sm));content()
    }
}
@Composable private fun SurfaceCard(content: @Composable ColumnScope.()->Unit) {
    val p=LocalPalette.current
    Column(Modifier.fillMaxWidth().background(p.surface,Tokens.largeShape).border(Tokens.hairline,p.border,Tokens.largeShape).padding(Tokens.lg),verticalArrangement=Arrangement.spacedBy(Tokens.sm),content=content)
}
@Composable private fun SectionLabel(text: String) {Text(text,color=LocalPalette.current.text3,style=MaterialTheme.typography.labelSmall,modifier=Modifier.padding(top=Tokens.sm))}
@Composable private fun ModelsPage(vm: ChatViewModel) {
    val p=LocalPalette.current
    PageColumn("Models") {
        Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically) {Text("Local models",style=MaterialTheme.typography.titleMedium,modifier=Modifier.weight(1f));TextButton(onClick=vm::refreshModels,enabled=!vm.busy) {Text("Refresh")}}
        val groups=modelGroups(vm.models,vm.selected)
        val loaded=groups.loaded
        val available=groups.available
        val unavailable=groups.unavailable
        listOf("Loaded" to loaded,"Available" to available,"Unavailable" to unavailable).forEach { (group,models) ->
            SectionLabel(group)
            models.forEach {model->SurfaceCard {
                Text(model.modelId?:java.io.File(model.directory).name,style=MaterialTheme.typography.titleSmall,fontWeight=FontWeight.SemiBold)
                Text(java.io.File(model.directory).name,color=p.text2,style=MaterialTheme.typography.bodySmall)
                model.totalBytes?.let {Text(formatBytes(it).orEmpty(),color=p.text3,style=MaterialTheme.typography.bodySmall)}
                if(group=="Loaded") {
                    vm.contextCapacity?.let {Text("${formatCount(it)} context tokens",color=p.text3,style=MaterialTheme.typography.bodySmall)}
                    Row(horizontalArrangement=Arrangement.spacedBy(Tokens.sm)) {OutlinedButton(onClick=vm::unload,enabled=!vm.busy) {Text("Unload")};Button(onClick=vm::backToChat) {Text("Open chat")}}
                } else if(model.selectable) Button(onClick={vm.load(model.directory)},enabled=!vm.busy) {Text("Load")}
                else Text(model.reason,color=p.error,style=MaterialTheme.typography.bodySmall)
            } }
        }
        if(available.isEmpty()&&loaded.isEmpty()&&!vm.busy) Text("Push model assets as described in the Android README.",color=p.text2)
        if(vm.busy) Row(verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(Tokens.sm)) {CircularProgressIndicator(Modifier.size(Tokens.icon));Text(if(vm.navigation.loadingModel) "Loading model…" else "Checking models…",style=MaterialTheme.typography.bodySmall)}
        else if(vm.status.isNotEmpty()&&vm.status!="Ready") Text(vm.status,color=p.error,style=MaterialTheme.typography.bodySmall)
    }
}
@Composable private fun KeyValue(label: String,value: String) {Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(Tokens.md)) {Text(label,color=LocalPalette.current.text3,style=MaterialTheme.typography.bodySmall,modifier=Modifier.weight(1f));Text(value,style=MaterialTheme.typography.bodySmall)}}
@Composable private fun StatusPage(vm: ChatViewModel) {
    val p=LocalPalette.current; val request=requestDisplay(vm.progress,vm.lastTurn)
    PageColumn("Status") {
        if(vm.progress.phase!="idle"||vm.lastTurn!=null) SurfaceCard {
            Row(verticalAlignment=Alignment.CenterVertically) {Text(if(request.phase=="Idle") "Last request" else "Current request",style=MaterialTheme.typography.titleSmall,fontWeight=FontWeight.SemiBold,modifier=Modifier.weight(1f));Text(request.phase,color=if(request.phase=="Idle") p.text3 else p.accent,style=MaterialTheme.typography.labelSmall)}
            request.prefill?.let {KeyValue("Prefill",it)};request.decode?.let {KeyValue("Decode",it)}
        }
        SurfaceCard {
            Row {Text("Memory",style=MaterialTheme.typography.titleSmall,fontWeight=FontWeight.SemiBold,modifier=Modifier.weight(1f));formatBytes(vm.memory.physicalBytes?:vm.memory.residentBytes)?.let {Text(it,color=p.text2,style=MaterialTheme.typography.bodySmall)}}
            vm.memory.currentBytes?.let { current ->
                val limit=vm.memory.limitBytes?.takeIf {it>0} ?: current.takeIf {it>0}
                if(limit!=null) Row(Modifier.fillMaxWidth().height(Tokens.bar).background(p.surface3,Tokens.smallShape)) {
                    val categories=listOf("resident_weights","experts","session_state","workspace","staging","other")
                    categories.forEachIndexed { index,key ->
                        val bytes=if(key=="other") (vm.memory.byPurpose["metadata"]?:0)+(vm.memory.byPurpose["transfer"]?:0) else vm.memory.byPurpose[key]?:0
                        if(bytes>0) Box(Modifier.weight((bytes.toDouble()/limit).toFloat()).fillMaxHeight().background(p.segments[index]))
                    }
                    if(limit>current) Spacer(Modifier.weight(((limit-current).toDouble()/limit).toFloat()))
                }
                val labels=listOf("Weights","Experts","Session state","Workspace","Staging","Other")
                labels.chunked(3).forEachIndexed { row,group ->Row(horizontalArrangement=Arrangement.spacedBy(Tokens.md)) {
                    group.forEachIndexed {column,label ->Row(verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(Tokens.xs)) {Box(Modifier.size(Tokens.sm).background(p.segments[row*3+column],Tokens.smallShape));Text(label,style=MaterialTheme.typography.labelSmall,color=p.text2)} }
                } }
            }
        }
        Row(horizontalArrangement=Arrangement.spacedBy(Tokens.xs),verticalAlignment=Alignment.CenterVertically) {
            TextButton(onClick={vm.setCumulative(false)}) {Text("Since start",color=if(!vm.cumulativeView) p.accent else p.text3,style=MaterialTheme.typography.bodySmall)}
            TextButton(onClick={vm.setCumulative(true)}) {Text("All time",color=if(vm.cumulativeView) p.accent else p.text3,style=MaterialTheme.typography.bodySmall)}
            Spacer(Modifier.weight(1f));if(vm.cumulativeView) TextButton(onClick=vm::resetCumulative) {Text("Reset all-time",style=MaterialTheme.typography.bodySmall)}
        }
        Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(Tokens.md)) {
            StatCard("Reused",vm.totals.reuse,"of input tokens",Modifier.weight(1f));StatCard("Expert slot hits",vm.totals.hits,"expert uses",Modifier.weight(1f))
        }
        Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(Tokens.md)) {
            StatCard("Avg decode",vm.totals.averageDecode,null,Modifier.weight(1f));val heat=vm.system.thermal?.let(::thermalDisplay);StatCard("Thermal",heat?.label,heat?.description,Modifier.weight(1f),heat?.let {when(it.tone){ThermalTone.ok->p.ok;ThermalTone.warn->p.warn;ThermalTone.error->p.error}})
        }
    }
}
/** Three dots bouncing in turn while the answer has no text yet. */
@Composable private fun PendingDots() {
    val p=LocalPalette.current
    val cycle by rememberInfiniteTransition(label="pending").animateFloat(0f,1f,infiniteRepeatable(tween(1200,easing=LinearEasing)),label="cycle")
    Row(Modifier.height(Tokens.dotRow).semantics {contentDescription="Generating"},verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.spacedBy(Tokens.dotGap)) {
        repeat(3) { index ->
            val phase=((cycle-index*0.125f)%1f+1f)%1f
            val lift=if(phase<0.5f) kotlin.math.sin(phase*2f*Math.PI.toFloat()) else 0f
            Box(Modifier.size(Tokens.dot).graphicsLayer {translationY=-Tokens.dotLift.toPx()*lift;alpha=0.45f+0.55f*lift}.background(p.text3,Tokens.circleShape))
        }
    }
}
@Composable private fun StatCard(label: String,value: String?,sub: String?,modifier: Modifier,valueColor: Color?=null) {
    if(value==null) return
    val p=LocalPalette.current
    Column(modifier.background(p.surface,Tokens.largeShape).border(Tokens.hairline,p.border,Tokens.largeShape).padding(Tokens.lg),verticalArrangement=Arrangement.spacedBy(Tokens.sm)) {
        Text(label,color=p.text3,style=MaterialTheme.typography.bodySmall)
        value?.let {Text(it,style=if(label=="Avg decode"||label=="Thermal") MaterialTheme.typography.titleMedium else MaterialTheme.typography.headlineMedium,fontWeight=FontWeight.SemiBold,color=valueColor?:p.text)}
        if(value!=null&&sub!=null) Text(sub,color=p.text3,style=MaterialTheme.typography.labelSmall)
    }
}
@Composable private fun SettingsPage(vm: ChatViewModel) {
    val p=LocalPalette.current
    PageColumn("Settings") {
        SectionLabel("Appearance")
        SurfaceCard {Text("Theme",style=MaterialTheme.typography.bodySmall);SegmentedOptions(ThemeMode.entries.map {it.name},ThemeMode.entries.indexOf(vm.theme)) {vm.updateTheme(ThemeMode.entries[it])}}
        SectionLabel("Conversation")
        SurfaceCard {
            Text("Generation style",style=MaterialTheme.typography.bodySmall)
            SegmentedOptions(GenerationStyle.entries.map {"${it.name} · ${it.temperature}"},GenerationStyle.entries.indexOf(vm.style),Tokens.xs) {vm.updateStyle(GenerationStyle.entries[it])}
            Text("Sets the temperature for every request. Every other sampling setting follows the model.",color=p.text3,style=MaterialTheme.typography.bodySmall)
            Spacer(Modifier.height(Tokens.sm));Text("System prompt",style=MaterialTheme.typography.bodySmall)
            OutlinedTextField(vm.customSystemPrompt,vm::setSystemPrompt,placeholder={Text(vm.appData.systemPrompt)},modifier=Modifier.fillMaxWidth(),minLines=Tokens.composerMaxLines)
            Text("Leave empty to use the default shown above. Applies to new chats.",color=p.text3,style=MaterialTheme.typography.bodySmall)
        }
    }
}
@Composable private fun SegmentedOptions(options: List<String>,selected: Int,horizontalPadding: Dp=Tokens.segmentHorizontal,onSelect: (Int)->Unit) {
    val p=LocalPalette.current
    Row(Modifier.fillMaxWidth().clip(Tokens.mediumShape).background(p.surface3).padding(Tokens.segmentInset).selectableGroup(),horizontalArrangement=Arrangement.spacedBy(Tokens.segmentInset)) {
        options.forEachIndexed {index,label->
            val chosen=index==selected
            Box(Modifier.weight(1f).clip(Tokens.segmentShape).background(if(chosen) p.surface else p.surface3)
                .selectable(selected=chosen,onClick={onSelect(index)},role=Role.RadioButton)
                .padding(horizontal=horizontalPadding,vertical=Tokens.xs),contentAlignment=Alignment.Center) {
                Text(label,color=if(chosen) p.text else p.text2,style=MaterialTheme.typography.bodySmall,fontWeight=if(chosen) FontWeight.SemiBold else FontWeight.Normal,maxLines=1)
            }
        }
    }
}
