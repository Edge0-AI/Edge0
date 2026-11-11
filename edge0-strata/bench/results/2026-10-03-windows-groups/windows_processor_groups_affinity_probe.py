"""Read-only Windows processor group / NUMA / affinity probe.
Runs only in this helper process; affinity is changed only on its own temporary test threads.
"""
import ctypes as C
import json
import os
import platform
import sys
import struct
import threading
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
k32 = C.WinDLL("kernel32", use_last_error=True)
DWORD = C.c_uint32
WORD = C.c_uint16
BYTE = C.c_ubyte
KAFFINITY = C.c_size_t
ALL_PROCESSOR_GROUPS = 0xFFFF

class GROUP_AFFINITY(C.Structure):
    _fields_ = [("Mask", KAFFINITY), ("Group", WORD), ("Reserved", WORD * 3)]

class PROCESSOR_NUMBER(C.Structure):
    _fields_ = [("Group", WORD), ("Number", BYTE), ("Reserved", BYTE)]

# Declare only APIs used by the probe; all processor IDs are group-relative.
k32.GetActiveProcessorGroupCount.restype = WORD
k32.GetActiveProcessorCount.argtypes = [WORD]
k32.GetActiveProcessorCount.restype = DWORD
k32.GetCurrentThread.restype = C.c_void_p
k32.GetCurrentProcessorNumberEx.argtypes = [C.POINTER(PROCESSOR_NUMBER)]
k32.SetThreadAffinityMask.argtypes = [C.c_void_p, KAFFINITY]
k32.SetThreadAffinityMask.restype = KAFFINITY
k32.GetThreadGroupAffinity.argtypes = [C.c_void_p, C.POINTER(GROUP_AFFINITY)]
k32.GetThreadGroupAffinity.restype = C.c_int
k32.SetThreadGroupAffinity.argtypes = [C.c_void_p, C.POINTER(GROUP_AFFINITY), C.POINTER(GROUP_AFFINITY)]
k32.SetThreadGroupAffinity.restype = C.c_int
k32.GetLogicalProcessorInformationEx.argtypes = [DWORD, C.c_void_p, C.POINTER(DWORD)]
k32.GetLogicalProcessorInformationEx.restype = C.c_int


def api_error(name):
    raise OSError(C.get_last_error(), f"{name} failed")


def masks(records_relation):
    needed = DWORD(0)
    C.set_last_error(0)
    ok = k32.GetLogicalProcessorInformationEx(records_relation, None, C.byref(needed))
    if ok or needed.value == 0:
        api_error("GetLogicalProcessorInformationEx(size query)")
    buf = C.create_string_buffer(needed.value)
    if not k32.GetLogicalProcessorInformationEx(records_relation, C.cast(buf, C.c_void_p), C.byref(needed)):
        api_error("GetLogicalProcessorInformationEx(data query)")
    records = []
    pos = 0
    while pos < needed.value:
        relationship, size = struct.unpack_from("=II", buf.raw, pos)
        if size < 8 or pos + size > needed.value:
            raise RuntimeError(f"invalid relation record at offset {pos}: size={size}, length={needed.value}")
        # RelationNumaNodeEx is requested with 6 but, on this Windows build,
        # the returned record's Relationship field is RelationNumaNode (1).
        if relationship == records_relation or (records_relation == 6 and relationship == 1):
            records.append((pos, size))
        pos += size
    if pos != needed.value:
        raise RuntimeError(f"record sequence ended at {pos}, expected {needed.value}")
    return buf, records


def parse_affinities(buf, offset, count):
    out = []
    for i in range(count):
        mask, group = struct.unpack_from("=QH", buf.raw, offset + i * C.sizeof(GROUP_AFFINITY))
        processors = [bit for bit in range(64) if mask & (1 << bit)]
        out.append({"group": group, "mask_hex": f"0x{mask:016X}", "processors": processors})
    return out


def parse_cores():
    buf, records = masks(0)  # RelationProcessorCore
    cores = []
    for ordinal, (pos, size) in enumerate(records):
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX union starts at byte 8;
        # PROCESSOR_RELATIONSHIP.GroupCount is byte 22 in that union, first GROUP_AFFINITY byte 24.
        count = struct.unpack_from("=H", buf.raw, pos + 30)[0]
        groups = parse_affinities(buf, pos + 32, count)
        cores.append({"core_record": ordinal, "groups": groups})
    return cores


def parse_numa():
    buf, records = masks(6)  # RelationNumaNodeEx; full masks span processor groups
    nodes = []
    for pos, size in records:
        node = struct.unpack_from("=I", buf.raw, pos + 8)[0]
        count = struct.unpack_from("=H", buf.raw, pos + 30)[0]
        groups = parse_affinities(buf, pos + 32, count)
        nodes.append({"node": node, "groups": groups, "record_size": size})
    return nodes


def affinity_snapshot(handle):
    ga = GROUP_AFFINITY()
    if not k32.GetThreadGroupAffinity(handle, C.byref(ga)):
        api_error("GetThreadGroupAffinity")
    pn = PROCESSOR_NUMBER()
    k32.GetCurrentProcessorNumberEx(C.byref(pn))
    return {"group": int(ga.Group), "mask_hex": f"0x{ga.Mask:016X}",
            "mask_processors": [i for i in range(64) if ga.Mask & (1 << i)],
            "current_processor": {"group": int(pn.Group), "number": int(pn.Number)}}


def legacy_test(target_group, target_processor, move_group_first):
    done = threading.Event()
    result = {}
    def body():
        handle = k32.GetCurrentThread()
        original = GROUP_AFFINITY()
        moved = False
        previous_mask = 0
        try:
            if move_group_first:
                ga = GROUP_AFFINITY()
                ga.Mask = KAFFINITY(1 << target_processor).value
                ga.Group = target_group
                if not k32.SetThreadGroupAffinity(handle, C.byref(ga), C.byref(original)):
                    api_error("SetThreadGroupAffinity(test setup)")
                moved = True
            result["before"] = affinity_snapshot(handle)
            requested = KAFFINITY(1 << target_processor).value
            C.set_last_error(0)
            previous_mask = k32.SetThreadAffinityMask(handle, requested)
            result["set_thread_affinity_mask"] = {"requested_hex": f"0x{requested:016X}",
                "succeeded": bool(previous_mask), "returned_previous_hex": f"0x{previous_mask:016X}",
                "error": None if previous_mask else C.get_last_error()}
            if previous_mask:
                # Give the scheduler time to honor the single-processor mask before sampling current CPU.
                end = time.perf_counter() + 0.02
                while time.perf_counter() < end:
                    pass
            result["after"] = affinity_snapshot(handle)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Restore only this probe thread's temporary affinity before it exits.
            if previous_mask:
                k32.SetThreadAffinityMask(handle, previous_mask)
            if moved:
                k32.SetThreadGroupAffinity(handle, C.byref(original), None)
            done.set()
    t = threading.Thread(target=body, name=f"affinity-probe-g{target_group}-p{target_processor}")
    t.start()
    done.wait(5)
    t.join(5)
    if t.is_alive():
        raise RuntimeError("affinity test thread did not finish")
    result["expected_processor"] = {"group": target_group, "number": target_processor}
    return result


def main():
    group_count = int(k32.GetActiveProcessorGroupCount())
    group_counts = [int(k32.GetActiveProcessorCount(i)) for i in range(group_count)]
    cores = parse_cores()
    numa = parse_numa()
    by_group = {g: [] for g in range(group_count)}
    for core in cores:
        for entry in core["groups"]:
            by_group.setdefault(entry["group"], []).append(entry["processors"])
    summaries = {}
    for group, core_lists in by_group.items():
        summaries[str(group)] = {
            "physical_core_records": len(core_lists),
            "logical_processors_in_core_records": sum(len(x) for x in core_lists),
            "representative_physical_core_logical_ids": [x[0] for x in core_lists],
            "first_4_core_smt_masks": core_lists[:4],
            "last_2_core_smt_masks": core_lists[-2:],
        }
    tests = []
    # The first test requests a group-0 flat CPU ID on a newly created thread,
    # which inherits this process's primary group. The second sets group 1
    # explicitly, showing that the legacy mask is group-relative once the group
    # is selected. The default engine path has no SetThreadGroupAffinity call.
    reps = {g: (core_lists[0][0] if core_lists and core_lists[0] else 0)
            for g, core_lists in by_group.items()}
    if 0 in reps:
        tests.append({"case": "legacy_mask_for_flattened_group0_core_on_inherited_primary_group",
                      "target_group": 0, "target_processor": reps[0],
                      "move_group_first": False, "result": legacy_test(0, reps[0], False)})
    if 1 in reps:
        tests.append({"case": "group1_explicit_group_affinity_then_legacy_mask", "target_group": 1,
                      "target_processor": reps[1], "move_group_first": True,
                      "result": legacy_test(1, reps[1], True)})
    result = {
        "probe": "windows_processor_groups_affinity_probe.py",
        "safety": "Separate helper process. Affinity modified only on its own short-lived Python test threads; no Strata/model/server/process affinity was queried or changed.",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "os": {"platform": platform.platform(), "version": platform.version(), "python": sys.version,
              "pointer_bits": C.sizeof(C.c_void_p) * 8, "processor_count": os.cpu_count()},
        "helper_main_thread_affinity": affinity_snapshot(k32.GetCurrentThread()),
        "groups": {"active_group_count": group_count, "active_processors_per_group": group_counts,
                   "total_active_processors_api": int(k32.GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)),
                   "physical_core_records": len(cores), "cores_by_group": summaries},
        "numa_nodes_relation_numa_node_ex": numa,
        "legacy_set_thread_affinity_tests": tests,
        "source_notes": {
            "core_mask_parsing": "SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX RelationProcessorCore; all GROUP_AFFINITY records read.",
            "numa_mask_parsing": "RelationNumaNodeEx (relationship 6); variable GroupCount and full per-group masks read.",
            "affinity_scope": "SetThreadGroupAffinity and SetThreadAffinityMask applied only to test threads created by this script. GetThreadGroupAffinity and GetCurrentProcessorNumberEx read back their placement.",
        },
    }
    (OUT / "probe-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
