#pragma once
#import <Foundation/Foundation.h>
#include "edge0/diagnostics.h"

namespace example_chat {
inline NSString *status_string(const std::string &value) { return [[NSString alloc] initWithBytes:value.data() length:value.size() encoding:NSUTF8StringEncoding] ?: @""; }
inline NSDictionary *progress_status(const edge0::GenerationProgress &p, std::chrono::steady_clock::time_point now) {
    if (p.phase == edge0::GenerationPhase::idle) return @{};
    NSMutableDictionary *row = [@{@"phase": p.phase == edge0::GenerationPhase::prefill ? @"prefill" : @"decode", @"input_tokens": @(p.input_tokens), @"reused_tokens": @(p.reused_tokens), @"computed_tokens": @(p.computed_tokens), @"pending_tokens": @(p.pending_tokens), @"generated_tokens": @(p.generated_tokens), @"elapsed_seconds": @(std::chrono::duration<double>(now - p.started).count())} mutableCopy];
    if (p.first_token) {
        const double first = std::chrono::duration<double>(*p.first_token - p.started).count();
        const double decode = std::chrono::duration<double>(now - *p.first_token).count();
        row[@"first_token_seconds"] = @(first); row[@"decode_elapsed_seconds"] = @(decode);
        if (first > 0) row[@"prefill_tokens_per_second"] = @((p.input_tokens - p.reused_tokens) / first);
        if (decode > 0 && p.generated_tokens > 0) row[@"decode_tokens_per_second"] = @((p.generated_tokens - 1) / decode);
    }
    return row;
}
inline NSDictionary *snapshot_status(const edge0::MemorySnapshot &snapshot) {
    const auto &ledger = snapshot.ledger;
    NSMutableDictionary *purposes = [NSMutableDictionary dictionary], *devices = [NSMutableDictionary dictionary];
    const std::pair<edge0::MemoryPurpose, const char *> names[] = {
        {edge0::MemoryPurpose::resident_weights, "resident_weights"}, {edge0::MemoryPurpose::experts, "experts"}, {edge0::MemoryPurpose::session_state, "session_state"}, {edge0::MemoryPurpose::workspace, "workspace"}, {edge0::MemoryPurpose::transfer, "transfer"}, {edge0::MemoryPurpose::staging, "staging"}, {edge0::MemoryPurpose::metadata, "metadata"}};
    for (const auto &[purpose, name] : names) { auto found = ledger.by_purpose.find(purpose); if (found != ledger.by_purpose.end()) purposes[[NSString stringWithUTF8String:name]] = @(found->second); }
    for (const auto &[name, device] : ledger.by_device) devices[status_string(name)] = @{@"current_bytes": @(device.current_bytes), @"peak_bytes": @(device.peak_bytes), @"memory": device.memory == edge0::MemoryLocation::system ? @"system" : @"dedicated"};
    auto metric = [](const edge0::MemoryMetric &value) -> NSDictionary * {
        NSMutableDictionary *row = [@{@"reason": status_string(value.reason), @"source": status_string(value.source)} mutableCopy];
        if (value.values) {
            NSMutableDictionary *values = [NSMutableDictionary dictionary];
            if (value.values->current_bytes) values[@"current_bytes"] = @(*value.values->current_bytes);
            if (value.values->peak_bytes) values[@"peak_bytes"] = @(*value.values->peak_bytes);
            row[@"values"] = values;
        }
        return row;
    };
    return @{@"ledger": @{@"limit_bytes": @(ledger.limit_bytes), @"current_bytes": @(ledger.current_bytes), @"peak_bytes": @(ledger.peak_bytes), @"by_purpose": purposes, @"by_device": devices}, @"physical": metric(snapshot.physical), @"resident": metric(snapshot.rss)};
}
inline NSDictionary *system_status(const edge0::SystemStatus &status) {
    auto value = [](const edge0::SystemValue &item) -> NSMutableDictionary * {
        NSMutableDictionary *row = [@{@"source": status_string(item.source), @"reason": status_string(item.reason)} mutableCopy];
        if (item.bytes) row[@"bytes"] = @(*item.bytes);
        return row;
    };
    auto limit = value(status.app_memory_limit);
    if (status.app_memory_limit.current_physical_bytes) limit[@"current_physical_bytes"] = @(*status.app_memory_limit.current_physical_bytes);
    NSMutableDictionary *thermal = [@{@"source": status_string(status.thermal.source), @"reason": status_string(status.thermal.reason)} mutableCopy];
    if (status.thermal.state) thermal[@"state"] = status_string(*status.thermal.state);
    return @{@"memory": @{@"total": value(status.memory.total), @"available": value(status.memory.available)}, @"thermal": thermal, @"app_memory_limit": limit};
}
}
