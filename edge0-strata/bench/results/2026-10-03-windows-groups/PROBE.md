# Windows processor-group and pool-affinity probe

Captured on 2026-10-03 on the Windows 11 Pro build 10.0.26300 machine used for the RX 6900 XT benchmark. The probe ran in a separate 64-bit Python 3.12.10 process. It made no changes to Strata, model settings, servers, or other processes. Only short-lived threads owned by the probe were given temporary affinities, which the helper restored before thread exit.

## Observed topology

- Windows reports 2 active processor groups, 64 active logical processors in each, 128 total, and 64 physical-core records.
- Each group contains 32 physical cores, each with two logical processors. The first logical processor of each core is group-relative IDs 0, 2, 4, ... 62 in both groups.
- `RelationNumaNodeEx` reports one NUMA node (node 0), whose group masks include all 64 processors in group 0 and all 64 in group 1. The two processor groups are therefore not two NUMA nodes on this machine.
- The probe process/thread primary group was 1, although the sampled current processor was in group 0. This is consistent with Windows 11's default affinities spanning groups while retaining a primary group.

## Legacy affinity test

On a new probe thread inheriting primary group 1, `SetThreadAffinityMask(thread, 1)` succeeded. `GetThreadGroupAffinity` then reported primary group 1 with mask bit 0, and `GetCurrentProcessorNumberEx` reported group 1, processor 0. The request represented group 0, processor 0 in the flattened topology list; it did not select that group-0 processor. A second new thread explicitly assigned to group 1 and then given the same legacy mask correctly selected group 1, processor 0.

The upstream Windows pool code at 99f3dbd enumerates each physical core as `group * 64 + bit`, then pins with `SetThreadAffinityMask(..., 1 << (core & 63))` (`src/kernels/cpu/pool.cpp`). The modulo folds away the group. Worker threads are created by the same constructor thread, with no group-aware thread assignment. On this host, group-0 and group-1 physical-core representatives use the same local IDs (0, 2, ... 62); so in a fixed primary group, their legacy affinity masks alias. After reserving the first core, the default 63-worker list contains 31 group-0 and 32 group-1 representatives but only 32 distinct group-local masks. This demonstrates the group-loss/duplicate-pinning defect in the affinity path. It is a plausible explanation for the measured slowdown, but this probe did not run inference and does not independently establish the benchmark's performance causality.

## Reproduce

Run the standalone helper from any 64-bit Windows Python:

```powershell
python windows_processor_groups_affinity_probe.py
```

It writes `probe-results.json` next to itself. That JSON contains all physical-core group masks, NUMA node masks, active counts, test-thread before/after group affinity, returned legacy masks, current processor readings, OS/Python version, and the safety note.

The helper reads processor/core records with `GetLogicalProcessorInformationEx(RelationProcessorCore)` and full NUMA group masks with `GetLogicalProcessorInformationEx(RelationNumaNodeEx)`. It tests legacy affinity only on its own threads.

## Windows documentation

- [Processor Groups](https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups): Windows 11 process/thread affinities span groups by default; the primary group remains relevant to legacy affinity APIs; logical processor numbers are group-relative.
- [SetThreadAffinityMask](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask): the mask applies to the thread's current primary group on Windows 11, and its bits identify processors in that group.
- [GetLogicalProcessorInformationEx](https://learn.microsoft.com/en-us/windows/desktop/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex): `RelationNumaNodeEx` returns NUMA affinity across groups; `RelationNumaNode` provides only the primary group's mask.
- [GetActiveProcessorCount](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getactiveprocessorcount): the all-groups value reports total active processors.
