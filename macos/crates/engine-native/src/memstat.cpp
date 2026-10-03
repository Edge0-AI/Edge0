#include "memstat.hpp"

#include <mach/mach.h>

namespace e0n {

MemSample mem_sample() {
  MemSample s;
  // resident_size includes file-backed pages; phys_footprint is dirty-page
  // residency (Activity Monitor "Memory"). This SDK has no ri_resident_size_peak.
  task_vm_info_data_t vi{};
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                               reinterpret_cast<task_info_t>(&vi), &cnt);
  if (kr == KERN_SUCCESS) {
    s.resident_peak_bytes = vi.resident_size_peak;
    s.resident_bytes = vi.resident_size;
    s.footprint_bytes = vi.phys_footprint;
  }
  return s;
}

}  // namespace e0n
