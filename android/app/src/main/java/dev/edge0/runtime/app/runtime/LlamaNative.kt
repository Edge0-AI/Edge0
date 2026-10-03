// LlamaNative.kt - JNI declaration surface for llama_chat.c.
// external fun symbols bind hard to the package name
// (dev_edge0_runtime_app_runtime_LlamaNative_*): moving packages requires
// mirroring the rename on the C side. Top-level object members keep stable
// exported names (no @JvmName needed).
package dev.edge0.runtime.app.runtime

object LlamaNative {
    init {
        // packaged via APK jniLibs (five shared libs from tools/llama/build_vendor_libs.sh);
// DT_NEEDED resolves libllama -> libggml* automatically
        System.loadLibrary("edge0llama")
    }

    external fun nativeInit(model: String, lora: String, heads: String,
                            nCtx: Int, poolMb: Int, blobMb: Int, pred: Int, ioThreads: Int): String
    external fun nativeReset(): String
    external fun nativeAddRole(role: String, text: String): String
    external fun nativeGenerate(sink: (String) -> Unit, maxTokens: Int, temp: Float,
                           thinkingOn: Boolean): String
    external fun nativeCancel()
    external fun nativeStats(): String
    external fun nativeLastAssistant(): String?
    external fun nativeFree(): String
}
