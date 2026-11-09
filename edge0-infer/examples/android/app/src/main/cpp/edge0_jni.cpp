#include <jni.h>
#include <android/log.h>
#include <edge0/assets.h>
#include <edge0/diagnostics.h>
#include <edge0/generation.h>
#include <mutex>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "status_mapping.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Engine {
    std::optional<edge0::RuntimeReference> runtime;
    edge0::ModelReference model;
    edge0::SessionReference session;
    std::mutex cancel_mutex;
    std::optional<edge0::CancelToken> cancel;
};
Engine& engine(jlong handle) { return *reinterpret_cast<Engine*>(handle); }
template<class T> T take(const edge0::Result<T>& value) {
    if (!value) throw std::runtime_error(value.error().message);
    return value.value();
}
std::string utf8(JNIEnv* env, jstring input) {
    if (!input) return {};
    const jchar* chars = env->GetStringChars(input, nullptr);
    const auto length = env->GetStringLength(input);
    std::string result;
    for (jsize i = 0; i < length; ++i) {
        unsigned code = chars[i];
        if (code >= 0xd800 && code <= 0xdbff && i + 1 < length && chars[i+1] >= 0xdc00 && chars[i+1] <= 0xdfff)
            code = 0x10000 + ((code - 0xd800) << 10) + (chars[++i] - 0xdc00);
        else if (code >= 0xd800 && code <= 0xdfff) code = 0xfffd;
        if (code < 0x80) result += static_cast<char>(code);
        else if (code < 0x800) { result += static_cast<char>(0xc0 | (code >> 6)); result += static_cast<char>(0x80 | (code & 63)); }
        else if (code < 0x10000) {
            result += static_cast<char>(0xe0 | (code >> 12)); result += static_cast<char>(0x80 | ((code >> 6) & 63)); result += static_cast<char>(0x80 | (code & 63));
        } else {
            result += static_cast<char>(0xf0 | (code >> 18)); result += static_cast<char>(0x80 | ((code >> 12) & 63));
            result += static_cast<char>(0x80 | ((code >> 6) & 63)); result += static_cast<char>(0x80 | (code & 63));
        }
    }
    env->ReleaseStringChars(input, chars);
    return result;
}
jbyteArray bytes(JNIEnv* env, const std::string& value) {
    auto array = env->NewByteArray(static_cast<jsize>(value.size()));
    env->SetByteArrayRegion(array, 0, static_cast<jsize>(value.size()), reinterpret_cast<const jbyte*>(value.data()));
    return array;
}
jstring string(JNIEnv* env, const std::string& value) {
    auto cls = env->FindClass("java/lang/String");
    auto constructor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    auto array = bytes(env, value);
    auto charset = env->NewStringUTF("UTF-8");
    auto result = static_cast<jstring>(env->NewObject(cls, constructor, array, charset));
    env->DeleteLocalRef(array); env->DeleteLocalRef(charset); env->DeleteLocalRef(cls);
    return result;
}
void fail(JNIEnv* env, const std::exception& error) {
    auto cls = env->FindClass("java/lang/IllegalStateException");
    auto constructor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;)V");
    auto message = string(env, error.what());
    auto throwable = static_cast<jthrowable>(env->NewObject(cls, constructor, message));
    env->Throw(throwable); env->DeleteLocalRef(message); env->DeleteLocalRef(throwable); env->DeleteLocalRef(cls);
}
void unload(Engine& value) {
    if (value.session) { take(edge0::release_session(value.session)); value.session.reset(); }
    if (value.model) { take(edge0::unload_model(value.model)); value.model.reset(); }
}
const char* stop_name(edge0::StopReason reason) {
    switch (reason) {
        case edge0::StopReason::end_of_generation: return "end_of_generation";
        case edge0::StopReason::length: return "length";
        case edge0::StopReason::capacity: return "capacity";
        case edge0::StopReason::cancelled: return "cancelled";
        case edge0::StopReason::failed: return "failed";
        case edge0::StopReason::stop_string: return "stop_string";
        case edge0::StopReason::tool_calls: return "tool_calls";
    }
    return "failed";
}
}
extern "C" JNIEXPORT jlong JNICALL Java_dev_edge0_chat_NativeEngine_nativeCreate(JNIEnv*, jobject) {
    return reinterpret_cast<jlong>(new Engine());
}
extern "C" JNIEXPORT jobjectArray JNICALL Java_dev_edge0_chat_NativeEngine_nativeInspect(JNIEnv* env, jobject, jstring directory) {
    std::string model, error, total;
    try {
        auto result = edge0::inspect_assets({utf8(env, directory), false});
        if (result) { model = result.value().architecture; std::uint64_t size = 0; for (const auto& file : result.value().files) size += file.bytes; total = std::to_string(size); }
        else error = result.error().message;
    } catch (const std::exception& exception) { error = exception.what(); }
    auto cls = env->FindClass("java/lang/String");
    auto result = env->NewObjectArray(3, cls, nullptr);
    auto identifier = string(env, model); auto reason = string(env, error);
    auto size = string(env, total); env->SetObjectArrayElement(result, 2, size); env->DeleteLocalRef(size);
    env->SetObjectArrayElement(result, 0, identifier); env->SetObjectArrayElement(result, 1, reason);
    env->DeleteLocalRef(identifier); env->DeleteLocalRef(reason); env->DeleteLocalRef(cls);
    return result;
}
extern "C" JNIEXPORT jbooleanArray JNICALL Java_dev_edge0_chat_NativeEngine_nativeLoad(JNIEnv* env, jobject, jlong handle, jstring directory) {
    try {
        auto& value = engine(handle);
        unload(value);
        if (!value.runtime) {
            edge0::RuntimeOptions options;
            const auto capabilities = edge0::query_capabilities();
            __android_log_print(ANDROID_LOG_INFO, "Edge0Chat", "engine version=%s source_commit=%s source_modified=%s ggml=%s",
                capabilities.engine_version.c_str(), capabilities.source_commit.c_str(),
                capabilities.source_modified == edge0::TriState::yes ? "yes" : capabilities.source_modified == edge0::TriState::no ? "no" : "unknown",
                capabilities.ggml_version.c_str());
            for (const auto& device : capabilities.devices) {
                if (device.backend == "cpu") { options.devices = {device.name}; break; }
            }
            if (options.devices.empty()) throw std::runtime_error("No CPU device available");
            options.log_receiver = [](edge0::LogLevel, const std::string& message) { __android_log_write(ANDROID_LOG_INFO, "Edge0Engine", message.c_str()); };
            value.runtime = take(edge0::open_runtime(options));
        }
        edge0::LoadOverrides overrides; overrides.device = value.runtime->available_devices().front();
        auto configuration = take(edge0::resolve_load_configuration(*value.runtime, utf8(env, directory), overrides));
        value.model = take(edge0::load_model(*value.runtime, configuration));
        auto description = take(edge0::query_model(value.model));
        value.session = take(edge0::create_session(value.model));
        const jboolean capabilities[] = {static_cast<jboolean>(description.thinking_enabled_supported), static_cast<jboolean>(description.thinking_default)};
        auto result = env->NewBooleanArray(2); env->SetBooleanArrayRegion(result, 0, 2, capabilities);
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL Java_dev_edge0_chat_NativeEngine_nativePrepare(JNIEnv* env, jobject, jlong handle) {
    try {
        auto& value = engine(handle);
        auto token = take(edge0::create_cancel_token(*value.runtime));
        std::lock_guard<std::mutex> lock(value.cancel_mutex);
        value.cancel = std::move(token);
    } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT void JNICALL Java_dev_edge0_chat_NativeEngine_nativeCancel(JNIEnv*, jobject, jlong handle) {
    auto& value = engine(handle);
    std::lock_guard<std::mutex> lock(value.cancel_mutex);
    if (value.cancel) edge0::request_cancel(*value.cancel);
}
extern "C" JNIEXPORT jobject JNICALL Java_dev_edge0_chat_NativeEngine_nativeGenerate(JNIEnv* env, jobject, jlong handle,
    jobjectArray roles, jobjectArray contents, jobjectArray reasoning, jboolean thinking, jobject max_tokens, jdouble temperature, jobject callback) {
    try {
        auto& value = engine(handle);
        edge0::CancelToken token;
        { std::lock_guard<std::mutex> lock(value.cancel_mutex); if (!value.cancel) throw std::runtime_error("Missing cancellation token"); token = *value.cancel; }
        const auto count = env->GetArrayLength(roles);
        if (count != env->GetArrayLength(contents) || count != env->GetArrayLength(reasoning)) throw std::runtime_error("Message array lengths differ");
        std::vector<edge0::TextMessage> messages;
        for (jsize i = 0; i < count; ++i) {
            auto r = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
            auto c = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
            auto t = static_cast<jstring>(env->GetObjectArrayElement(reasoning, i));
            edge0::TextMessage message{utf8(env, r), utf8(env, c)};
            if (message.role == "assistant") message.reasoning_content = utf8(env, t);
            messages.push_back(std::move(message));
            env->DeleteLocalRef(r); env->DeleteLocalRef(c); env->DeleteLocalRef(t);
        }
        edge0::GenerationOverrides overrides; overrides.thinking = thinking; overrides.temperature = temperature;
        if (max_tokens != nullptr) {
            auto integer_class = env->GetObjectClass(max_tokens);
            auto int_value = env->GetMethodID(integer_class, "intValue", "()I");
            overrides.max_tokens = env->CallIntMethod(max_tokens, int_value);
            env->DeleteLocalRef(integer_class);
        }
        auto configuration = take(edge0::resolve_generation_configuration(value.model, overrides));
        const auto source_entry = configuration.sources.find("max_tokens");
        const char* source = "missing";
        if (source_entry != configuration.sources.end()) {
            switch (source_entry->second) {
                case edge0::ConfigurationSource::platform: source = "platform"; break;
                case edge0::ConfigurationSource::model: source = "model"; break;
                case edge0::ConfigurationSource::application: source = "application"; break;
                case edge0::ConfigurationSource::request: source = "request"; break;
            }
        }
        const auto override_text = overrides.max_tokens ? std::to_string(*overrides.max_tokens) : "unset";
        __android_log_print(ANDROID_LOG_INFO, "Edge0Chat", "generation_config max_tokens_override=%s max_tokens_source=%s resolved_max_tokens=%lld thinking=%s", override_text.c_str(), source, static_cast<long long>(configuration.max_tokens), thinking ? "on" : "off");
        auto callback_class = env->GetObjectClass(callback);
        auto method = env->GetMethodID(callback_class, "onEvent", "(I[B)V");
        auto result = take(edge0::generate(value.session, messages, configuration, {}, [&](const edge0::OutputEvent& event) -> std::optional<edge0::Error> {
            if (event.kind == edge0::OutputKind::answer || event.kind == edge0::OutputKind::reasoning) {
                auto data = bytes(env, event.text);
                env->CallVoidMethod(callback, method, event.kind == edge0::OutputKind::reasoning ? 1 : 0, data);
                env->DeleteLocalRef(data);
                if (env->ExceptionCheck()) { env->ExceptionClear(); return edge0::Error{edge0::ErrorCategory::invalid_argument, "Output callback failed"}; }
            }
            return std::nullopt;
        }, &token));
        env->DeleteLocalRef(callback_class);
        jlong peak = -1;
        auto memory = edge0::memory_snapshot(*value.runtime);
        if (memory) {
            const auto& snapshot = memory.value();
            if (snapshot.physical.values && snapshot.physical.values->peak_bytes) peak = static_cast<jlong>(*snapshot.physical.values->peak_bytes);
            else if (snapshot.rss.values && snapshot.rss.values->peak_bytes) peak = static_cast<jlong>(*snapshot.rss.values->peak_bytes);
        }
        const jlong counters[] = {static_cast<jlong>(result.generated_count), static_cast<jlong>(result.input.size()), peak, static_cast<jlong>(result.reused_tokens), static_cast<jlong>(result.request_summary.decode_forwards.value_or(0)),
            result.weight_activity.applicable ? static_cast<jlong>(result.weight_activity.prefill.demand + result.weight_activity.decode.demand) : -1,
            result.weight_activity.applicable ? static_cast<jlong>(result.weight_activity.prefill.hits + result.weight_activity.decode.hits) : -1,
            result.request_summary.decode_forwards ? 1 : 0};
        const jdouble times[] = {result.request_summary.first_token_seconds.value_or(-1), result.request_summary.decode_tokens_per_second.value_or(-1), result.request_summary.decode_seconds.value_or(-1), result.configuration.temperature};
        auto counts = env->NewLongArray(8); env->SetLongArrayRegion(counts, 0, 8, counters);
        auto timings = env->NewDoubleArray(4); env->SetDoubleArrayRegion(timings, 0, 4, times);
        auto cls = env->FindClass("dev/edge0/chat/NativeResult");
        auto constructor = env->GetMethodID(cls, "<init>", "([J[DLjava/lang/String;ZLjava/lang/String;)V");
        auto stop = string(env, stop_name(result.stop_reason));
        auto error = result.error ? string(env, result.error->message) : nullptr;
        auto object = env->NewObject(cls, constructor, counts, timings, stop, static_cast<jboolean>(result.state_valid), error);
        __android_log_print(ANDROID_LOG_INFO, "Edge0Chat", "stop_reason=%s state_valid=%s generated_count=%zu input_tokens=%zu reused_tokens=%zu", stop_name(result.stop_reason), result.state_valid ? "true" : "false", result.generated_count, result.input.size(), result.reused_tokens);
        env->DeleteLocalRef(counts); env->DeleteLocalRef(timings); env->DeleteLocalRef(stop); if (error) env->DeleteLocalRef(error); env->DeleteLocalRef(cls);
        { std::lock_guard<std::mutex> lock(value.cancel_mutex); value.cancel.reset(); }
        return object;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL Java_dev_edge0_chat_NativeEngine_nativeReset(JNIEnv* env, jobject, jlong handle) {
    try { take(edge0::reset_conversation(engine(handle).session)); }
    catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT void JNICALL Java_dev_edge0_chat_NativeEngine_nativeClose(JNIEnv* env, jobject, jlong handle) {
    try {
        auto* value = &engine(handle);
        unload(*value);
        if (value->runtime) take(edge0::finish_runtime_use(*value->runtime));
        delete value;
    } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT jlong JNICALL Java_dev_edge0_chat_NativeEngine_nativeContext(JNIEnv* env, jobject, jlong handle) {
    try { return static_cast<jlong>(take(edge0::query_model(engine(handle).model)).context); }
    catch (const std::exception& error) { fail(env, error); return 0; }
}
extern "C" JNIEXPORT void JNICALL Java_dev_edge0_chat_NativeEngine_nativeUnload(JNIEnv* env, jobject, jlong handle) {
    try { unload(engine(handle)); } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT jstring JNICALL Java_dev_edge0_chat_NativeEngine_nativeProgress(JNIEnv* env, jobject, jlong handle) {
    try {
        auto& value = engine(handle);
        const auto progress = value.runtime ? take(edge0::query_generation_progress(*value.runtime)) : edge0::GenerationProgress{};
        return string(env, android_status::progress_json(progress, std::chrono::steady_clock::now()));
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_dev_edge0_chat_NativeEngine_nativeSnapshot(JNIEnv* env, jobject, jlong handle) {
    try {
        auto& value = engine(handle);
        return string(env, value.runtime ? android_status::memory_json(take(edge0::memory_snapshot(*value.runtime))) : "{}");
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_dev_edge0_chat_NativeEngine_nativeSystemStatus(JNIEnv* env, jobject) {
    try { return string(env, android_status::system_json(edge0::query_system_status())); }
    catch (const std::exception& error) { fail(env, error); return nullptr; }
}
