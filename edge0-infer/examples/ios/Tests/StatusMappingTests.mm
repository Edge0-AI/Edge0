#include "StatusMapping.h"
#include <cassert>

int main(int argc, char **argv) {
    @autoreleasepool {
        assert(argc == 2);
        NSDictionary *sample = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:nil];
        edge0::GenerationProgress p;
        p.phase = edge0::GenerationPhase::decode; p.input_tokens = 13; p.reused_tokens = 12; p.computed_tokens = 1; p.pending_tokens = 0; p.generated_tokens = 3;
        p.started = std::chrono::steady_clock::time_point(std::chrono::seconds(1000));
        p.first_token = p.started + std::chrono::nanoseconds(2878875);
        auto row = example_chat::progress_status(p, p.started + std::chrono::nanoseconds(6173500));
        for (NSString *key in @[@"input_tokens", @"reused_tokens", @"computed_tokens", @"pending_tokens", @"generated_tokens", @"phase"]) assert([row[key] isEqual:sample[@"current"][key]]);
        assert(std::abs([row[@"first_token_seconds"] doubleValue] - 0.002878875) < 1e-12);
        assert(std::abs([row[@"decode_tokens_per_second"] doubleValue] - 607.0493607011421) < 1e-9);
        p.phase = edge0::GenerationPhase::idle; assert(example_chat::progress_status(p, p.started).count == 0);
        edge0::MemorySnapshot memory;
        memory.ledger.limit_bytes = 17179869184; memory.ledger.current_bytes = 46837024; memory.ledger.peak_bytes = 48252608;
        memory.ledger.by_purpose = {{edge0::MemoryPurpose::resident_weights,20945152}, {edge0::MemoryPurpose::experts,491520}, {edge0::MemoryPurpose::session_state,13328384}, {edge0::MemoryPurpose::workspace,4231136}, {edge0::MemoryPurpose::transfer,2176}, {edge0::MemoryPurpose::staging,0}, {edge0::MemoryPurpose::metadata,7838656}};
        memory.ledger.by_device = {{"CPU", {38998368,39993312,edge0::MemoryLocation::system}}, {"host", {7838656,8259296,edge0::MemoryLocation::system}}};
        memory.rss.values = edge0::MemoryValues{199081984,220856320}; memory.rss.source = "task_info.resident_size/ru_maxrss";
        memory.physical.values = edge0::MemoryValues{169476720,215188032}; memory.physical.source = "phys_footprint";
        NSDictionary *mapped = example_chat::snapshot_status(memory);
        NSDictionary *expected = sample[@"memory"];
        assert([mapped[@"ledger"][@"by_device"] isEqual:expected[@"ledger"][@"by_device"]]);
        assert([mapped[@"ledger"][@"by_purpose"] isEqual:expected[@"ledger"][@"by_purpose"]]);
        assert([mapped[@"physical"][@"values"][@"current_bytes"] isEqual:expected[@"physical"][@"values"][@"current_bytes"]]);
        memory.ledger.by_device["Vulkan0"] = {11804000000,11804000000,edge0::MemoryLocation::dedicated};
        assert([example_chat::snapshot_status(memory)[@"ledger"][@"by_device"][@"Vulkan0"][@"memory"] isEqual:@"dedicated"]);
        edge0::SystemStatus system;
        system.memory.total = {17179869184,"sysctl.hw.memsize",""}; system.memory.available = {4924522496,"host_statistics64.free_count+inactive_count+purgeable_count+speculative_count",""};
        system.thermal = {"nominal","NSProcessInfo.thermalState",""}; system.app_memory_limit.reason = "App memory limit is only available on iOS";
        NSDictionary *mappedSystem = example_chat::system_status(system);
        assert([mappedSystem[@"memory"] isEqual:sample[@"system"][@"memory"]]); assert([mappedSystem[@"thermal"] isEqual:sample[@"system"][@"thermal"]]);
        assert(!mappedSystem[@"app_memory_limit"][@"bytes"]);
        system.app_memory_limit.bytes = 3290000000; system.app_memory_limit.current_physical_bytes = 2800000000;
        assert([example_chat::system_status(system)[@"app_memory_limit"][@"bytes"] unsignedLongLongValue] >= [example_chat::system_status(system)[@"app_memory_limit"][@"current_physical_bytes"] unsignedLongLongValue]);
    }
}
