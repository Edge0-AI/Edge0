"""Measurement hosts: the only place that runs the measured command and reads system state.

Every measurement script talks to a host object with four operations:

* ``run(argv)``       run the measured command (a list of strings, never a shell line);
* ``preflight()``     check the conditions a measurement needs and raise ``PreflightError`` otherwise;
* ``snapshot(model)`` return the system state before or after a process;
* ``device()``        return the device identity for the report.

Snapshot and device values are either the platform's own value or ``{"available": false, "reason": ...}``.
A missing value is never replaced by zero or by an estimate, and platform-specific names and sources are
kept (for example ``vm_stat`` text on macOS) instead of being renamed to another platform's terms.
"""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Platforms that exist in the tool but are provided by later work. Selecting one is refused before
# any measured command runs.
PLANNED = {}


class PreflightError(RuntimeError):
    """A measurement precondition is not met."""


class UnsupportedPlatform(ValueError):
    """The requested platform has no host implementation yet."""


class BackgroundLoadError(PreflightError):
    """A process outside the system directories is using the CPU; the measurement would be disturbed."""


# Executables under these prefixes are the operating system's own; they are recorded but never block.
SYSTEM_PREFIXES = ('/System/', '/usr/', '/bin/', '/sbin/', '/Library/Apple/')
# The Windows counterpart: executables under the Windows directory are the operating system's own.
# The default system drive is assumed; a process started from elsewhere counts as foreground load.
WINDOWS_SYSTEM_PREFIX = 'c:\\windows\\'
LOAD_THRESHOLD = 15.0


def _json_records(text):
    """Parse ``ConvertTo-Json`` output, which is a bare object for one row and an array for several,
    into a list of records."""
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


class CommandResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def unavailable(reason):
    return {'available': False, 'reason': str(reason)}


def is_unavailable(value):
    return isinstance(value, dict) and value.get('available') is False


def local_run(argv):
    """Run a command on this machine without a shell."""
    result = subprocess.run(argv, capture_output=True, text=True)
    return CommandResult(result.returncode, result.stdout, result.stderr)


def page_cache_residency(path):
    """Resident bytes of a file in the page cache, via mincore (POSIX)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as view:
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        vec = (ctypes.c_ubyte * pages)()
        if libc.mincore(address, size, vec):
            raise OSError(ctypes.get_errno(), 'mincore failed')
        resident = sum(bool(v & 1) for v in vec)
    return dict(bytes=size, page_bytes=mmap.PAGESIZE, resident_pages=resident, resident_bytes=resident * mmap.PAGESIZE)


class MacOSHost:
    """macOS on the machine that runs the measurement script."""
    name = 'macos'
    backend = 'metal'

    def unsupported_configuration(self, name):
        return None

    def __init__(self, execute=local_run, residency=page_cache_residency, clock=time.time):
        self._execute, self._residency, self._clock = execute, residency, clock

    def run(self, argv):
        return self._execute(list(argv))

    def _capture(self, argv):
        """Command output, or raise RuntimeError with the failure reason."""
        try:
            result = self._execute(list(argv))
        except Exception as error:  # a missing executable is a failed probe, not a crash
            raise RuntimeError(f'{argv[0]} could not run: {error}')
        if result.returncode:
            raise RuntimeError(f'{argv[0]} failed with exit code {result.returncode}: {result.stderr.strip()}')
        return result.stdout.strip()

    def _optional(self, argv):
        try:
            return self._capture(argv)
        except RuntimeError as error:
            return unavailable(error)

    def preflight(self):
        try:
            power = self._capture(['pmset', '-g', 'batt'])
        except RuntimeError as error:
            raise PreflightError(f'Cannot verify AC power: {error}')
        if "'AC Power'" not in power:
            raise PreflightError('Measurement requires AC power throughout the comparison')

    def background_load(self, threshold=LOAD_THRESHOLD):
        """Processes above ``threshold`` percent CPU, split into ``other`` (outside the system directories)
        and ``system``. Only executable names are returned, never paths or arguments. This process and its
        parent (the launcher of the measurement script) are not counted."""
        text = self._capture(['ps', '-A', '-o', 'pid=,%cpu=,comm='])
        skip = {os.getpid(), os.getppid()}
        other, system, quiet = [], [], []
        for line in text.splitlines():
            try:
                pid, cpu, path = line.split(None, 2)
                pid, cpu = int(pid), float(cpu)
            except ValueError:
                raise RuntimeError(f'unparsable ps output: {line!r}')
            if pid in skip:
                continue
            entry = dict(name=Path(path.strip()).name, cpu_percent=cpu)
            if not path.startswith(SYSTEM_PREFIXES):
                quiet.append(entry)
            if cpu <= threshold:
                continue
            (system if path.startswith(SYSTEM_PREFIXES) else other).append(entry)
        key = lambda e: -e['cpu_percent']
        return dict(threshold=threshold, other=sorted(other, key=key), system=sorted(system, key=key),
                    non_system_top=sorted(quiet, key=key)[:10])

    def check_background_load(self):
        """Raise BackgroundLoadError naming the offending processes; returns the load record otherwise."""
        try:
            load = self.background_load()
        except RuntimeError as error:
            raise BackgroundLoadError(f'background load could not be checked: {error}')
        if load['other']:
            names = ', '.join(f"{e['name']} ({e['cpu_percent']:.0f}%)" for e in load['other'])
            raise BackgroundLoadError(f'background load: {names} above {load["threshold"]:g}% CPU')
        return load

    def snapshot(self, model):
        """System state, including the page-cache residency of the model asset files."""
        shot = dict(platform='macos', time=self._clock())
        power = self._optional(['pmset', '-g', 'batt'])
        shot['power'] = power if is_unavailable(power) else dict(ac_power="'AC Power'" in power, raw=power)
        shot['thermal'] = self._optional(['pmset', '-g', 'therm'])
        memory = self._optional(['vm_stat'])
        shot['memory'] = memory if is_unavailable(memory) else dict(source='vm_stat', raw=memory)
        shot['load'] = self._optional(['sysctl', 'vm.loadavg'])
        shot['top_processes'] = self._top_processes()
        try:
            shot['background_load'] = self.background_load()
        except RuntimeError as error:
            shot['background_load'] = unavailable(error)
        try:
            shot['residency'] = {name: self._residency(Path(model) / name) for name in ['base.gguf']}
            if shot['residency']['base.gguf'] is None:
                shot['residency']['base.gguf'] = unavailable('file not found')
        except (OSError, ValueError) as error:
            shot['residency'] = unavailable(error)
        return shot

    def _top_processes(self):
        # Darwin -c requests executable names, never command arguments.
        text = self._optional(['ps', '-A', '-c', '-o', '%cpu=,comm='])
        if is_unavailable(text):
            return text
        processes = []
        try:
            for line in text.splitlines():
                cpu, executable = line.split(None, 1)
                processes.append(dict(name=Path(executable).name, cpu_percent=float(cpu)))
        except ValueError as error:
            return unavailable(f'unparsable ps output: {error}')
        processes.sort(key=lambda process: process['cpu_percent'], reverse=True)
        return processes[:5]

    def device(self):
        info = dict(platform='macos', model=self._optional(['sysctl', '-n', 'hw.model']),
                    chip=self._optional(['sysctl', '-n', 'machdep.cpu.brand_string']),
                    os_version=self._optional(['sw_vers', '-productVersion']))
        memory = self._optional(['sysctl', '-n', 'hw.memsize'])
        if is_unavailable(memory):
            info['memory_bytes'] = memory
        else:
            try:
                info['memory_bytes'] = int(memory)
            except ValueError:
                info['memory_bytes'] = unavailable(f'unparsable hw.memsize: {memory!r}')
        return info


def default_platform():
    return {'darwin': 'macos', 'win32': 'windows', 'linux': 'linux'}.get(sys.platform, sys.platform)


def select(platform=None, **options):
    """Host for ``platform`` (default: this machine). Unimplemented platforms raise UnsupportedPlatform."""
    platform = platform or default_platform()
    if platform == 'ios':
        return IOSHost(**options)
    if platform == 'android':
        return AndroidHost(**options)
    if platform == 'macos':
        return MacOSHost(**options)
    if platform == 'windows':
        return WindowsHost(**options)
    if platform in PLANNED:
        raise UnsupportedPlatform(f'Platform {platform} is not implemented here; it is provided by {PLANNED[platform]}')
    raise UnsupportedPlatform(f'Platform {platform} has no measurement host')


class AndroidHost:
    """CPU measurements through the same transport as the platform suite."""
    name = 'android'
    backend = 'cpu'

    def __init__(self, transport=None, execute=local_run, clock=time.time):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
        from android import Android
        self.transport = transport or Android()
        self._execute, self._clock = execute, clock

    def unsupported_configuration(self, name):
        return 'Android does not support direct expert reads' if name == 'direct' else None

    def run(self, argv):
        try:
            binary = Path(argv[0])
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            if self.transport.digest(self.transport.root + '/edge0-infer') != digest:
                raise RuntimeError('Installed Android binary differs from --cli; push it explicitly')
        except (OSError, RuntimeError) as error:
            return CommandResult(1, '', str(error))
        wrapper = Path(__file__).resolve().parents[1] / 'platform/android-cli'
        return self._execute([str(wrapper), *map(str, argv[1:])])

    def _capture(self, *argv):
        result = self.transport.shell(*argv)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    def _optional(self, *argv):
        try:
            text = self._capture(*argv)
            return text if text else unavailable('Empty device response: ' + ' '.join(argv))
        except (OSError, RuntimeError) as error:
            return unavailable(error)

    def preflight(self):
        try:
            self.transport.reachable()
            power = self._capture('dumpsys', 'battery')
            awake = self._capture('dumpsys', 'power')
        except (OSError, RuntimeError) as error:
            raise PreflightError(str(error)) from error
        if not any(f'{kind} powered: true' in power for kind in ('AC', 'USB')):
            raise PreflightError('Android measurement requires AC or USB power')
        if 'mWakefulness=Awake' not in awake:
            raise PreflightError('Android measurement requires the screen to be awake')

    def _top_processes(self):
        text = self._optional('ps', '-A', '-o', 'NAME')
        if is_unavailable(text):
            return text
        return [dict(name=Path(line.strip()).name) for line in text.splitlines()[1:] if line.strip()]

    def check_background_load(self):
        return dict(load=self._optional('cat', '/proc/loadavg'), top_processes=self._top_processes())

    def snapshot(self, model):
        frequency = self._optional('sh', '-c',
            'for p in /sys/devices/system/cpu/cpufreq/policy*; do '
            'echo "$p"; cat "$p/scaling_max_freq" "$p/cpuinfo_max_freq" || exit 1; done')
        cpufreq = frequency
        if not is_unavailable(frequency):
            lines = frequency.splitlines()
            try:
                if len(lines) % 3:
                    raise ValueError('Incomplete frequency record')
                cpufreq = [dict(policy=Path(lines[i]).name, scaling_max_freq_khz=int(lines[i+1]),
                                cpuinfo_max_freq_khz=int(lines[i+2])) for i in range(0, len(lines), 3)]
            except ValueError as error:
                cpufreq = unavailable(error)
        memory = self._optional('cat', '/proc/meminfo')
        return dict(platform=self.name, time=self._clock(), power=self._optional('dumpsys', 'battery'),
                    thermal=self._optional('dumpsys', 'thermalservice'), cpufreq=cpufreq,
                    memory=memory if is_unavailable(memory) else dict(source='meminfo', raw=memory),
                    load=self._optional('cat', '/proc/loadavg'), top_processes=self._top_processes(),
                    residency=unavailable('Android does not provide host mincore access to remote model files'),
                    background_load=self.check_background_load())

    def device(self):
        memory = self._optional('cat', '/proc/meminfo')
        try:
            total = next(line.split()[1] for line in memory.splitlines() if line.startswith('MemTotal:'))
            memory_bytes = int(total) * 1024
        except (AttributeError, StopIteration, ValueError):
            memory_bytes = unavailable('MemTotal unavailable or malformed')
        return dict(platform=self.name, model=self._optional('getprop', 'ro.product.model'),
                    chip=self._optional('getprop', 'ro.soc.model'), memory_bytes=memory_bytes,
                    os_version=self._optional('getprop', 'ro.build.version.release'),
                    sdk=self._optional('getprop', 'ro.build.version.sdk'))


def _ps(command):
    """A non-interactive PowerShell invocation carrying one command string as a single argument."""
    return ['powershell', '-NoProfile', '-NonInteractive', '-Command', command]


class WindowsHost:
    """Windows on the machine that runs the measurement script. CPU backend only (no direct reads).

    Thermal sensors, a load average and page-cache residency have no user-space query on Windows, so
    those snapshot items report ``{"available": false, "reason": ...}`` rather than a stand-in value.
    """
    name = 'windows'
    backend = 'cpu'

    # Battery status 2 means "on AC"; a desktop reports no battery device at all.
    POWER = _ps('Get-CimInstance Win32_Battery | Select-Object -ExpandProperty BatteryStatus')
    MEMORY = _ps('Get-CimInstance Win32_OperatingSystem | '
                 'Select-Object FreePhysicalMemory,TotalVisibleMemorySize | ConvertTo-Json')
    TOP = _ps("Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | "
              "Where-Object { $_.Name -notin '_Total','Idle' } | "
              'Sort-Object PercentProcessorTime -Descending | '
              'Select-Object -First 5 Name,PercentProcessorTime | ConvertTo-Json')
    # Per-process CPU percent joined with the executable path, so the system directory can be told apart.
    LOAD = _ps('$path=@{}; foreach ($p in Get-CimInstance Win32_Process) { $path[[int]$p.ProcessId]=$p.ExecutablePath }; '
               "Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | "
               "Where-Object { $_.Name -notin '_Total','Idle' } | "
               'ForEach-Object { [pscustomobject]@{ pid=[int]$_.IDProcess; path=$path[[int]$_.IDProcess]; '
               'cpu_percent=[int]$_.PercentProcessorTime } } | ConvertTo-Json')
    MODEL = _ps('Get-CimInstance Win32_ComputerSystem | '
                'Select-Object Model,TotalPhysicalMemory | ConvertTo-Json')
    CHIP = _ps('Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name')
    # Win32_Processor.CurrentClockSpeed usually repeats the nominal clock; the processor-performance
    # counter (percent of nominal, above 100 under turbo) gives the effective frequency.
    FREQUENCY = _ps('$p = Get-CimInstance Win32_Processor | Select-Object -First 1; '
                    "$c = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter \"Name='_Total'\"; "
                    '[pscustomobject]@{ nominal_mhz=[int]$p.MaxClockSpeed; current_mhz=[int]$p.CurrentClockSpeed; '
                    'performance_percent=$(if ($c) { [int]$c.PercentProcessorPerformance } else { $null }) } | ConvertTo-Json')
    OSVER = _ps('Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Version')

    def __init__(self, execute=local_run, clock=time.time):
        self._execute, self._clock = execute, clock

    def unsupported_configuration(self, name):
        return 'Windows does not support direct expert reads' if name == 'direct' else None

    def run(self, argv):
        return self._execute(list(argv))

    def _capture(self, argv):
        """Command output, or raise RuntimeError with the failure reason."""
        try:
            result = self._execute(list(argv))
        except Exception as error:  # a missing executable is a failed probe, not a crash
            raise RuntimeError(f'{argv[0]} could not run: {error}')
        if result.returncode:
            raise RuntimeError(f'{argv[0]} failed with exit code {result.returncode}: {result.stderr.strip()}')
        return result.stdout.strip()

    def _optional(self, argv):
        try:
            return self._capture(argv)
        except RuntimeError as error:
            return unavailable(error)

    @staticmethod
    def _on_ac(status):
        lines = {line.strip() for line in status.splitlines() if line.strip()}
        return (not lines) or '2' in lines  # no battery device, or a battery reporting AC

    def preflight(self):
        try:
            status = self._capture(self.POWER)
        except RuntimeError as error:
            raise PreflightError(f'Cannot verify power source: {error}')
        if not self._on_ac(status):
            raise PreflightError('Windows measurement requires AC power (battery not on AC)')

    def background_load(self, threshold=LOAD_THRESHOLD):
        """Processes above ``threshold`` percent CPU, split into ``other`` (outside the Windows directory)
        and ``system``. Only executable names are returned, never full paths. This process and its parent
        are not counted."""
        text = self._capture(self.LOAD)
        try:
            records = _json_records(text)
        except ValueError as error:
            raise RuntimeError(f'unparsable process counters: {error}')
        skip = {os.getpid(), os.getppid()}
        other, system, quiet = [], [], []
        for record in records:
            try:
                pid, cpu = int(record['pid']), float(record['cpu_percent'])
                path = record.get('path') or ''
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f'unparsable process record {record!r}: {error}')
            if pid in skip:
                continue
            # A protected OS process (Registry, Secure System, a sealed service) exposes no executable
            # path; it is the system's own, so it is recorded but never blocks, like paths under Windows.
            is_system = (not path) or path.lower().startswith(WINDOWS_SYSTEM_PREFIX)
            entry = dict(name=PureWindowsPath(path).name if path else '(unknown)', cpu_percent=cpu)
            if not is_system:
                quiet.append(entry)
            if cpu <= threshold:
                continue
            (system if is_system else other).append(entry)
        key = lambda entry: -entry['cpu_percent']
        return dict(threshold=threshold, other=sorted(other, key=key), system=sorted(system, key=key),
                    non_system_top=sorted(quiet, key=key)[:10])

    def check_background_load(self):
        """Raise BackgroundLoadError naming the offending processes; returns the load record otherwise."""
        try:
            load = self.background_load()
        except RuntimeError as error:
            raise BackgroundLoadError(f'background load could not be checked: {error}')
        if load['other']:
            names = ', '.join(f"{e['name']} ({e['cpu_percent']:.0f}%)" for e in load['other'])
            raise BackgroundLoadError(f'background load: {names} above {load["threshold"]:g}% CPU')
        return load

    def _top_processes(self):
        text = self._optional(self.TOP)
        if is_unavailable(text):
            return text
        try:
            return [dict(name=str(record['Name']), cpu_percent=float(record['PercentProcessorTime']))
                    for record in _json_records(text)]
        except (ValueError, KeyError, TypeError) as error:
            return unavailable(f'unparsable process counters: {error}')

    def _frequency(self):
        text = self._optional(self.FREQUENCY)
        if is_unavailable(text):
            return text
        try:
            record = json.loads(text)
            nominal, current = int(record['nominal_mhz']), int(record['current_mhz'])
            percent = record.get('performance_percent')
        except (ValueError, KeyError, TypeError) as error:
            return unavailable(f'unparsable processor frequency: {error}')
        result = dict(nominal_mhz=nominal, reported_current_mhz=current)
        if percent is None:
            result['effective_mhz'] = unavailable('processor performance counter not available')
        else:
            result['performance_percent'] = int(percent)
            result['effective_mhz'] = round(nominal * int(percent) / 100)
        return result

    def snapshot(self, model):
        shot = dict(platform=self.name, time=self._clock())
        power = self._optional(self.POWER)
        shot['power'] = power if is_unavailable(power) else dict(ac_power=self._on_ac(power),
                                                                 raw=power or 'no battery device')
        shot['thermal'] = unavailable('Windows exposes no thermal sensor without vendor tooling')
        shot['cpufreq'] = self._frequency()
        memory = self._optional(self.MEMORY)
        shot['memory'] = memory if is_unavailable(memory) else dict(source='Win32_OperatingSystem', raw=memory)
        shot['load'] = unavailable('Windows has no load-average metric')
        shot['top_processes'] = self._top_processes()
        try:
            shot['background_load'] = self.background_load()
        except RuntimeError as error:
            shot['background_load'] = unavailable(error)
        shot['residency'] = unavailable('Windows provides no page-cache residency query for model files')
        return shot

    def device(self):
        info = dict(platform=self.name, chip=self._optional(self.CHIP), os_version=self._optional(self.OSVER))
        model = self._optional(self.MODEL)
        if is_unavailable(model):
            info['model'] = info['memory_bytes'] = model
            return info
        try:
            record = json.loads(model)
            info['model'] = str(record['Model'])
            info['memory_bytes'] = int(record['TotalPhysicalMemory'])
        except (ValueError, KeyError, TypeError) as error:
            info['model'] = info['memory_bytes'] = unavailable(f'unparsable Win32_ComputerSystem: {error}')
        return info


class IOSHost:
    """Metal measurements inside the App; snapshots come from the measured process."""
    name = 'ios'
    backend = 'metal'

    def __init__(self, transport=None, runner=None, clock=time.time):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'platform'))
        from ios import IOS
        from ios_cli import execute
        self.transport = transport or IOS()
        self._runner = runner or execute
        self._clock = clock
        self._before = None
        self._after = None
        self._details = {}
        self._power = unavailable('devicectl does not expose battery or charging state')

    def unsupported_configuration(self, name):
        return None

    def preflight(self):
        try:
            self._details = self.transport.reachable()
        except RuntimeError as error:
            raise PreflightError(str(error)) from error
        self._before = None; self._after = None

    def _snapshot(self, probe):
        absent = lambda field: probe.get(field, unavailable('Measured process did not report ' + field))
        return dict(platform='ios', timestamp=probe.get('timestamp', self._clock()), power=self._power,
                    thermal=absent('thermal_state'), process_available_memory=absent('os_proc_available_memory'),
                    memory=dict(physical_memory=absent('physical_memory'), source='NSProcessInfo.physicalMemory'),
                    cpufreq=unavailable('iOS does not expose CPU frequency to this App'),
                    temperature=unavailable('iOS does not expose numeric temperature to this App'),
                    residency=unavailable('iOS does not expose asset page residency to this App'))

    def snapshot(self, model):
        if self._after is not None:
            return self._after
        self._before = self._snapshot({})
        return self._before

    def run(self, argv):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp) / 'probe.json'
            try:
                code, stdout, stderr = self._runner(self.transport, list(map(str, argv[1:])), b'', probe)
                if code >= 128:
                    raise SystemExit(f'iOS App terminated by signal {code - 128}; stop and report without retry')
                data = json.loads(probe.read_text()) if probe.exists() else {}
                if self._before is not None:
                    self._before.update(self._snapshot(data.get('before', {})))
                self._after = self._snapshot(data.get('after', {}))
                return CommandResult(code, stdout.decode('utf-8'), stderr.decode('utf-8'))
            except (OSError, ValueError, RuntimeError) as error:
                self._after = self._snapshot({})
                return CommandResult(1, '', str(error))

    def device(self):
        details = self._details or self.transport.reachable()
        hardware = details.get('hardwareProperties', {})
        properties = details.get('deviceProperties', {})
        return dict(platform='ios', model=hardware.get('productType', unavailable('Model unavailable')),
                    os_version=properties.get('osVersionNumber', unavailable('OS version unavailable')),
                    physical_memory=unavailable('devicectl does not report physical RAM; see process snapshots'))
