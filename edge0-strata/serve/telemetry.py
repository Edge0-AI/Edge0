"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.  On macOS (Metal backend): IOKit's
  IOAccelerator registry - load and VRAM; the temperature and power sensors need root (`powermetrics`), and an Apple
  Silicon GPU sits on the die, so there is no PCIe link.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import re
import subprocess
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}


    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out


def metal_working_set():
    """The Apple GPU's memory budget in bytes: MTLCreateSystemDefaultDevice's recommendedMaxWorkingSetSize, through
    the Metal framework and objc_msgSend (no PyObjC) - the same property metal/src/mtl_rt.mm reads for the engine
    (its "GPU working set" log line).  None off macOS, or when it cannot be read: a selector the device does not
    answer would raise an Objective-C exception no ctypes call can catch, so the class is asked first."""
    if sys.platform != "darwin":
        return None
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.object_getClass.argtypes = [ctypes.c_void_p]
        objc.object_getClass.restype = ctypes.c_void_p
        objc.class_respondsToSelector.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.class_respondsToSelector.restype = ctypes.c_ubyte
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = ctypes.c_ulonglong
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return None
        sel = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
        if not objc.class_respondsToSelector(objc.object_getClass(dev), sel):
            return None
        return int(objc.objc_msgSend(ctypes.c_void_p(dev), ctypes.c_void_p(sel))) or None
    except (OSError, AttributeError, ValueError):
        return None


class _Metal:
    """macOS: an Apple GPU's readings from the IOKit registry (`ioreg -l -w0 -r -c IOAccelerator`, 22 ms), with
    _Nvml's interface: the load and the memory are the driver's PerformanceStatistics of the index'th accelerator
    node - Device Utilization % and the in-use system memory in bytes.  The total is the device's Metal working set,
    the budget one app gets of the unified memory.  No temperature or power reading exists without root
    (`powermetrics`), and an Apple Silicon GPU sits on the die, so the PCIe cards have nothing to show."""
    note = ("no GPU temperature or power reading without root (powermetrics), and an Apple Silicon GPU sits on the "
            "die, so there is no PCIe link to show; the engine's own figures are in its log")
    cmd = ("ioreg", "-l", "-w0", "-r", "-c", "IOAccelerator")

    def __init__(self, index=0, run=None):
        self.index = index
        self._run = run or self._ioreg
        self.nodes = len(self._blocks())
        self._name = None
        b = self._block()
        if b:
            m = re.search(r'"model" = "([^"]+)"', b)
            self._name = m.group(1) if m else None

    @staticmethod
    def _ioreg():
        try:
            return subprocess.run(list(_Metal.cmd), capture_output=True, text=True, timeout=10).stdout or ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def _blocks(self):
        """The accelerator nodes, in the registry's order: every node that carries a PerformanceStatistics."""
        try:
            out = self._run() or ""
        except OSError:                     # ioreg missing, or a caller's runner that raises
            return []
        return [b for b in re.split(r"^\s*\+-o ", out, flags=re.M) if '"PerformanceStatistics"' in b]

    def _block(self):
        b = self._blocks()
        return b[self.index] if 0 <= self.index < len(b) else None

    def ok(self):
        return self.index < self.nodes

    def name(self):
        return self._name

    def read(self):
        b = self._block()
        if b is None:
            return {}

        def num(key):
            m = re.search(r'"%s"=([0-9.]+)' % re.escape(key), b)
            return float(m.group(1)) if m else None

        out = {}
        u = num("Device Utilization %")
        if u is None:                      # older chips report only the renderer and the tiler
            r, t = num("Renderer Utilization %"), num("Tiler Utilization %")
            u = max(v for v in (r, t) if v is not None) if (r is not None or t is not None) else None
        out["util"] = int(u) if u is not None else None
        used = num("In use system memory")
        out["mem_used"] = int(used) if used is not None else None
        out["mem_total"] = metal_working_set()
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend (#301), or IOKit's
    IOAccelerator registry on macOS (Metal backend).  `amd` comes first: no Mac runs the HIP backend, and the
    tests' AMD trees are read on a Mac too."""
    if amd:
        return _Amd(index)
    return _Metal(index) if sys.platform == "darwin" else _Nvml(index)


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters(vendor=None) -> list:
    """#1587: the display adapters Windows has drivers for, from the registry's display class (no subprocess, no vendor
    library): [{"name", "provider", "driver", "vram_bytes"}] in the class's order.  `vendor`: "amd", "nvidia" or
    "intel" keeps that vendor's cards.  [] off Windows or when nothing can be read."""
    if os.name != "nt":
        return []
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            k = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name, k=k):
            try:
                return winreg.QueryValueEx(k, name)[0]
            except OSError:
                return None
        name = val("DriverDesc")
        if not isinstance(name, str) or not name.strip():
            continue
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            mem = val("HardwareInformation.MemorySize")
        if isinstance(mem, (bytes, bytearray)):
            mem = int.from_bytes(bytes(mem[:8]), "little") if mem else None
        out.append({"name": name.strip(), "provider": str(val("ProviderName") or ""), "driver": val("DriverVersion"),
                    "vram_bytes": mem if isinstance(mem, int) and mem > 0 else None})
    if vendor:
        keys = {"amd": ("advanced micro devices", "amd", "radeon"), "nvidia": ("nvidia",), "intel": ("intel",)}[vendor]
        out = [a for a in out if any(s in (a["provider"] + " " + a["name"]).lower() for s in keys)]
    return out


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too
    (512 MiB on a Ryzen 5 7600): beside a card with 2 GiB or more it is left out (an APU alone keeps it)."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    return [c for c in cards if (c["vram_bytes"] or 0) >= 2 << 30] or cards


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_topology(root: str = "/sys/devices/system"):
    """(sockets, NUMA nodes) of this computer: how many CPU packages it has and how many memory nodes (None where it cannot be read).
    Linux reads /sys, Windows asks GetLogicalProcessorInformationEx / GetNumaHighestNodeNumber, macOS asks sysctl (Apple silicon and every Mac
    sold so far: one package, no NUMA)."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            needed = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(3, None, ctypes.byref(needed))          # RelationProcessorPackage
            buf = ctypes.create_string_buffer(needed.value)
            sockets = None
            if needed.value and kernel32.GetLogicalProcessorInformationEx(3, buf, ctypes.byref(needed)):
                sockets, offset = 0, 0
                while offset + 8 <= needed.value:
                    relationship = int.from_bytes(buf.raw[offset:offset + 4], "little")
                    size = int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
                    if size <= 0:
                        break
                    sockets += 1 if relationship == 3 else 0
                    offset += size
            high = ctypes.c_ulong(0)
            numa = high.value + 1 if kernel32.GetNumaHighestNodeNumber(ctypes.byref(high)) else None
            return (sockets or None), numa
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.packages"], capture_output=True, text=True, timeout=5).stdout.strip()
            return (int(out) if out.isdigit() else 1), None
        import glob
        packages = set()
        for f in glob.glob(os.path.join(root, "cpu", "cpu[0-9]*", "topology", "physical_package_id")):
            try:
                packages.add(open(f, encoding="utf-8").read().strip())
            except OSError:
                continue
        nodes = len(glob.glob(os.path.join(root, "node", "node[0-9]*")))
        return (len(packages) or None), (nodes or None)
    except Exception:                                        # noqa: BLE001 - a missing figure is shown as missing, never guessed
        return None, None


def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        gpu_name = " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None
        win_amd = None
        if gpu_name is None and amd and os.name == "nt":
            # #1587: no readings for an AMD card on Windows yet, but its name (and VRAM size) are in the registry, so the
            # dashboard and strata_bench do not call the run "CPU only".  The class order is taken as HIP's order.
            cards = windows_amd_cards()
            win_amd = [cards[i] for i in idx if 0 <= i < len(cards)] or cards[:1]
            gpu_name = " + ".join(c["name"] for c in win_amd) if win_amd else None
        # the macOS reader knows the device's working set (the VRAM budget of the unified memory); None elsewhere
        vram_total = None
        if win_amd is None and isinstance(self.gpu, _Metal) and self.gpu.ok():
            ws = metal_working_set()
            vram_total = ws >> 20 if ws else None
        self.static = {
            "gpu_name": gpu_name,
            "gpu_vram_total_mib": (sum(c["vram_bytes"] for c in win_amd) >> 20
                                   if win_amd and all(c["vram_bytes"] for c in win_amd) else vram_total),
            "gpu_count": len(self.gpus),
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            "gpu_note": ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu "
                         "driver); the engine's own VRAM figures are in its log" if amd and not self.gpu.ok()
                         else getattr(self.gpu, "note", None)),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "sockets": _cpu_topology()[0],
            "numa_nodes": _cpu_topology()[1],
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "name": reader.name(), "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for (i, r), (_, reader) in zip(reads, self.gpus)]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        """One sample a second, for as long as the server lives.

        A failing sample must not end the thread: on a machine where the GPU/psutil source is not readable
        (a CI runner, a container without NVML) the first exception used to kill the sampler outright, so
        `GET /metrics` silently stopped updating and `test_responses.NoLeakedSampler` saw "the server did
        not start its hardware sampler" - the thread had started and died between two readings.
        """
        complained = False
        while not self._stop.is_set():
            try:
                s = self.sample()
            except Exception as e:                    # noqa: BLE001 - a monitor must outlive its source
                if not complained:                    # once, not once a second
                    print(f"[strata] metrics: a sample failed ({e}); the sampler keeps trying", flush=True)
                    complained = True
                self._stop.wait(1.0)
                continue
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
