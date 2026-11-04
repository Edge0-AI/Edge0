#include "app/src/main/cpp/status_mapping.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace std::chrono;
    edge0::GenerationProgress progress;
    progress.phase=edge0::GenerationPhase::decode;
    progress.input_tokens=13;progress.reused_tokens=12;progress.computed_tokens=1;progress.pending_tokens=0;progress.generated_tokens=3;
    const auto started=steady_clock::time_point{}+seconds(1000);
    progress.started=started;progress.first_token=started+nanoseconds(2878875);
    const auto encoded=android_status::progress_json(progress,started+nanoseconds(6173500));
    assert(encoded.find("\"phase\":\"decode\"")!=std::string::npos);
    assert(encoded.find("\"inputTokens\":13")!=std::string::npos);
    std::cout<<"{\"progress\":"<<encoded;
    edge0::MemorySnapshot memory;
    memory.ledger.limit_bytes=17179869184;memory.ledger.current_bytes=46837024;memory.ledger.peak_bytes=48252608;
    memory.ledger.by_purpose={{edge0::MemoryPurpose::resident_weights,20945152},{edge0::MemoryPurpose::experts,491520},{edge0::MemoryPurpose::session_state,13328384},{edge0::MemoryPurpose::workspace,4231136},{edge0::MemoryPurpose::transfer,2176},{edge0::MemoryPurpose::staging,0},{edge0::MemoryPurpose::metadata,7838656}};
    memory.ledger.by_device={{"CPU",{38998368,39993312,edge0::MemoryLocation::system}},{"host",{7838656,8259296,edge0::MemoryLocation::system}}};
    memory.physical.values=edge0::MemoryValues{169476720,215188032};memory.rss.values=edge0::MemoryValues{199081984,220856320};
    std::cout<<",\"memory\":"<<android_status::memory_json(memory);
    edge0::SystemStatus system;
    system.memory.total.bytes=17179869184;system.memory.available.bytes=4924522496;system.thermal.state="nominal";
    std::cout<<",\"system\":"<<android_status::system_json(system)<<"}\n";
    memory.ledger.by_device={{"GPU",{11804000000,11804000000,edge0::MemoryLocation::dedicated}},{"host",{90000000,90000000,edge0::MemoryLocation::system}}};
    assert(android_status::memory_json(memory).find("\"memory\":\"dedicated\"")!=std::string::npos);
    progress.phase=edge0::GenerationPhase::idle;
    assert(android_status::progress_json(progress,started).find("elapsedSeconds")==std::string::npos);
    system.memory.total.bytes.reset();system.thermal.state.reset();
    assert(android_status::system_json(system).find("totalBytes")==std::string::npos);
    assert(android_status::system_json(system).find("thermal")==std::string::npos);
}
