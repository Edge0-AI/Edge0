#import "Edge0Bridge.h"
#include "GenerationOptions.h"
#include "StatusMapping.h"
#include "edge0/assets.h"
#include "edge0/diagnostics.h"
#include "edge0/generation.h"
#include <mutex>
#include <optional>

namespace {
NSString *string(const std::string &value) {
    return [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding] ?: @"";
}
NSDictionary *failure(const edge0::Error &error) { return @{@"error": string(error.message)}; }
std::string native(NSString *value) { return value ? std::string(value.UTF8String) : std::string(); }
}
@implementation Edge0Bridge {
    std::optional<edge0::RuntimeReference> _runtime;
    edge0::ModelReference _model;
    edge0::SessionReference _session;
    std::mutex _cancelMutex;
    std::mutex _statusMutex;
    std::optional<edge0::CancelToken> _cancel;
    bool _cancelRequested;
}
- (NSArray<NSDictionary<NSString *, id> *> *)modelsAt:(NSString *)root {
    NSMutableArray *rows = [NSMutableArray array];
    NSArray *names = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil];
    for (NSString *name in [names sortedArrayUsingSelector:@selector(localizedStandardCompare:)]) {
        NSString *path = [root stringByAppendingPathComponent:name];
        BOOL directory = NO;
        if (![[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&directory] || !directory) continue;
        edge0::InspectOptions options; options.directory = native(path);
        auto inspected = edge0::inspect_assets(options);
        NSMutableDictionary *row = [@{@"path": path, @"name": name} mutableCopy];
        if (!inspected) row[@"error"] = string(inspected.error().message);
        else {
            row[@"model_identifier"] = string(inspected.value().adapter);
            std::uint64_t total = 0;
            for (const auto &file : inspected.value().files) total += file.bytes;
            row[@"total_bytes"] = @(total);
        }
        [rows addObject:row];
    }
    return rows;
}
- (NSDictionary<NSString *, id> *)load:(NSString *)path {
    NSDictionary *closed = [self close];
    if (closed[@"error"]) return closed;
    auto capabilities = edge0::query_capabilities();
    std::string device;
    for (const auto &candidate : capabilities.devices) if (candidate.backend == "metal") { device = candidate.name; break; }
    if (device.empty()) return @{@"error": @"No Metal device is available"};
    edge0::RuntimeOptions runtimeOptions; runtimeOptions.devices = {device};
    auto opened = edge0::open_runtime(runtimeOptions);
    if (!opened) return failure(opened.error());
    { std::lock_guard<std::mutex> lock(_statusMutex); _runtime = opened.value(); }
    edge0::LoadOverrides overrides; overrides.device = device;
    auto loading = edge0::resolve_load_configuration(*_runtime, native(path), overrides);
    if (!loading) { auto error = failure(loading.error()); [self close]; return error; }
    auto loaded = edge0::load_model(*_runtime, loading.value());
    if (!loaded) { auto error = failure(loaded.error()); [self close]; return error; }
    _model = loaded.value();
    auto described = edge0::query_model(_model);
    if (!described) { auto error = failure(described.error()); [self close]; return error; }
    auto session = edge0::create_session(_model);
    if (!session) { auto error = failure(session.error()); [self close]; return error; }
    _session = session.value();
    NSLog(@"example chat: model loaded; context=%lld engine=%@ source_commit=%@ source_modified=%@", (long long)described.value().context, string(capabilities.engine_version), string(capabilities.source_commit), capabilities.source_modified == edge0::TriState::yes ? @"yes" : capabilities.source_modified == edge0::TriState::no ? @"no" : @"unknown");
    return @{@"thinking_supported": @(described.value().thinking_enabled_supported), @"thinking_default": @(described.value().thinking_default), @"context": @(described.value().context)};
}
- (NSDictionary<NSString *, id> *)generate:(NSArray<NSDictionary<NSString *, NSString *> *> *)messages thinking:(BOOL)thinking temperature:(double)temperature maxTokens:(NSNumber *)maxTokens tokenEvents:(BOOL)tokenEvents receive:(void (^)(NSString *, NSString *))receive {
    if (!_runtime || !_session) return @{@"error": @"Load a model first"};
    std::vector<edge0::TextMessage> history;
    for (NSDictionary *message in messages) {
        edge0::TextMessage item; item.role = native(message[@"role"]); item.content = native(message[@"content"]);
        NSString *reasoning = message[@"reasoning_content"];
        if (reasoning.length) item.reasoning_content = native(reasoning);
        history.push_back(std::move(item));
    }
    auto overrides = example_chat::generation_overrides((bool)thinking,
        maxTokens ? std::optional<std::int64_t>{maxTokens.longLongValue} : std::nullopt, (bool)tokenEvents, temperature);
    auto configuration = edge0::resolve_generation_configuration(_model, overrides);
    if (!configuration) return failure(configuration.error());
    auto token = edge0::create_cancel_token(*_runtime);
    if (!token) return failure(token.error());
    { std::lock_guard<std::mutex> lock(_cancelMutex); _cancel = token.value(); if (_cancelRequested) edge0::request_cancel(*_cancel); }
    auto generated = edge0::generate(_session, history, configuration.value(), {}, [receive](const edge0::OutputEvent &event) -> std::optional<edge0::Error> {
        @autoreleasepool {
            if (event.kind == edge0::OutputKind::answer) receive(@"answer", string(event.text));
            if (event.kind == edge0::OutputKind::reasoning) receive(@"reasoning", string(event.text));
            if (event.kind == edge0::OutputKind::token_logprob) receive(@"token", @"");
        }
        return std::nullopt;
    }, &token.value());
    { std::lock_guard<std::mutex> lock(_cancelMutex); _cancel.reset(); }
    if (!generated) return failure(generated.error());
    const auto &result = generated.value();
    NSMutableDictionary *metrics = [@{@"generated_count": @(result.generated_count), @"input_tokens": @(result.input.size()), @"reused_tokens": @(result.reused_tokens)} mutableCopy];
    if (result.request_summary.first_token_seconds) metrics[@"first_token_seconds"] = @(*result.request_summary.first_token_seconds);
    if (result.request_summary.decode_tokens_per_second) metrics[@"decode_tokens_per_second"] = @(*result.request_summary.decode_tokens_per_second);
    auto memory = edge0::memory_snapshot(*_runtime);
    if (memory) {
        const auto &metric = memory.value().physical.values && memory.value().physical.values->peak_bytes ? memory.value().physical : memory.value().rss;
        if (metric.values && metric.values->peak_bytes) metrics[@"memory_peak_bytes"] = @(*metric.values->peak_bytes);
    }
    NSString *stop;
    switch (result.stop_reason) {
        case edge0::StopReason::end_of_generation: stop = @"end_of_generation"; break;
        case edge0::StopReason::length: stop = @"length"; break;
        case edge0::StopReason::capacity: stop = @"capacity"; break;
        case edge0::StopReason::cancelled: stop = @"cancelled"; break;
        case edge0::StopReason::failed: stop = @"failed"; break;
        case edge0::StopReason::stop_string: stop = @"stop_string"; break;
        case edge0::StopReason::tool_calls: stop = @"tool_calls"; break;
    }
    NSLog(@"example chat: stop=%@ state_valid=%@ generated=%zu input=%zu reused=%zu metrics=%@", stop, result.state_valid ? @"true" : @"false", result.generated_count, result.input.size(), result.reused_tokens, metrics);
    NSMutableDictionary *response = [@{@"text": string(result.text), @"reasoning": string(result.reasoning), @"metrics": metrics, @"stop": stop, @"state_valid": @(result.state_valid)} mutableCopy];
    if (result.error) response[@"error"] = string(result.error->message);
    NSMutableDictionary *summary = [@{@"input_tokens": @(result.input.size()), @"reused_tokens": @(result.reused_tokens), @"computed_tokens": @(result.computed_tokens), @"generated_tokens": @(result.generated_count), @"total_seconds": @(result.request_summary.total_seconds), @"reuse_method": @(static_cast<int>(result.reuse_method)), @"checkpoint_restored": @(result.checkpoint_restored)} mutableCopy];
    const char *reuseNames[] = {"first", "keep", "truncate", "restore", "cold"};
    summary[@"reuse_method"] = [NSString stringWithUTF8String:reuseNames[static_cast<int>(result.reuse_method)]];
    if (result.request_summary.first_token_seconds) summary[@"first_token_seconds"] = @(*result.request_summary.first_token_seconds);
    if (result.request_summary.decode_seconds) summary[@"decode_seconds"] = @(*result.request_summary.decode_seconds);
    if (result.request_summary.decode_forwards) summary[@"decode_forwards"] = @(*result.request_summary.decode_forwards);
    if (result.weight_activity.applicable) {
        auto experts = result.weight_activity.prefill; experts += result.weight_activity.decode;
        summary[@"experts"] = @{@"demand": @(experts.demand), @"hits": @(experts.hits), @"loads": @(experts.loads), @"bytes": @(experts.bytes)};
    }
    response[@"summary"] = summary;
    return response;
}
- (NSDictionary<NSString *, id> *)progress {
    std::lock_guard<std::mutex> lock(_statusMutex);
    if (!_runtime) return @{};
    auto result = edge0::query_generation_progress(*_runtime);
    if (!result) return failure(result.error());
    return example_chat::progress_status(result.value(), std::chrono::steady_clock::now());
}
- (NSDictionary<NSString *, id> *)snapshot {
    std::lock_guard<std::mutex> lock(_statusMutex);
    if (!_runtime) return @{};
    auto result = edge0::memory_snapshot(*_runtime);
    if (!result) return failure(result.error());
    return example_chat::snapshot_status(result.value());
}
- (NSDictionary<NSString *, id> *)systemStatus {
    return example_chat::system_status(edge0::query_system_status());
}
- (NSDictionary<NSString *, id> *)reset {
    if (!_session) return @{};
    auto result = edge0::reset_conversation(_session);
    return result ? @{} : failure(result.error());
}
- (void)prepareGeneration { std::lock_guard<std::mutex> lock(_cancelMutex); _cancelRequested = false; }
- (void)cancel { std::lock_guard<std::mutex> lock(_cancelMutex); _cancelRequested = true; if (_cancel) edge0::request_cancel(*_cancel); }
- (NSDictionary<NSString *, id> *)close {
    // The owner serializes close behind generate. Cancellation alone is not a join.
    [self cancel];
    if (_session) {
        auto released = edge0::release_session(_session);
        if (!released) return failure(released.error());
        _session.reset();
    }
    if (_model) {
        auto unloaded = edge0::unload_model(_model);
        if (!unloaded) return failure(unloaded.error());
        _model.reset();
    }
    std::lock_guard<std::mutex> statusLock(_statusMutex);
    if (_runtime) {
        auto finished = edge0::finish_runtime_use(*_runtime);
        if (!finished) return failure(finished.error());
        _runtime.reset();
    }
    return @{};
}
- (void)dealloc { [self close]; }
@end
