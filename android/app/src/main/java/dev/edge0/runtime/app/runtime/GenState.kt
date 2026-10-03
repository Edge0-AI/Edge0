// runtime/GenState.kt - generation state machine and events (typed, single-slot).
package dev.edge0.runtime.app.runtime

import dev.edge0.runtime.engine.Status

sealed interface GenState {
    data object Idle : GenState
    data class Loading(val modelDir: String) : GenState
    data class Streaming(val threadId: String) : GenState
    data object Cancelling : GenState // settling gate: cancel sent, old job not yet home (new submits refused)
    data class Error(val status: Status, val diag: String) : GenState
}

sealed interface GenEvent {
    data class ThinkPiece(val s: String) : GenEvent
    data class TextPiece(val s: String) : GenEvent
    // genIds: token ids actually stepped this turn (sidecar; absent on cancel/failure)
    data class Done(val metrics: RequestMetrics?, val genIds: IntArray = IntArray(0)) : GenEvent
    data class Cancelled(val metrics: RequestMetrics?) : GenEvent
    data class Failed(val status: Status, val diag: String) : GenEvent
}

fun Seg.toEvent(): GenEvent = when (this) {
    is Seg.Text -> GenEvent.TextPiece(s)
    is Seg.Think -> GenEvent.ThinkPiece(s)
}
