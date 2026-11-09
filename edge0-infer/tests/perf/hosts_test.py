"""Measurement host: macOS snapshot parsing with recorded command output, preflight and platform selection."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / 'fixtures' / 'perf'
spec = importlib.util.spec_from_file_location('hosts', HERE / 'hosts.py')
hosts = importlib.util.module_from_spec(spec)
sys.modules['hosts'] = hosts
spec.loader.exec_module(hosts)


def fixture(name):
    return (FIXTURES / name).read_text()


LOAD_QUIET = (' 100  3.0 /Applications/Example.app/Contents/MacOS/Example\n'
              ' 101 40.0 /System/Library/PrivateFrameworks/Example.framework/WindowServer\n'
              ' 102  9.0 /usr/libexec/mdworker_shared\n')


def load_listing(*rows):
    return ''.join(f' {pid} {cpu:5.1f} {path}\n' for pid, cpu, path in rows)


class FakeExecutor:
    """Maps command prefixes to recorded output; a value of None makes the command fail."""

    def __init__(self, outputs, battery='pmset_batt_ac.txt'):
        self.calls = []
        self.outputs = {
            ('pmset', '-g', 'batt'): fixture(battery), ('pmset', '-g', 'therm'): fixture('pmset_therm.txt'),
            ('vm_stat',): fixture('vm_stat.txt'), ('sysctl', 'vm.loadavg'): fixture('sysctl_loadavg.txt'),
            ('ps', '-A', '-c', '-o', '%cpu=,comm='): fixture('ps_cpu.txt'),
            ('sysctl', '-n', 'hw.model'): fixture('hw_model.txt'),
            ('sysctl', '-n', 'machdep.cpu.brand_string'): fixture('cpu_brand.txt'),
            ('sysctl', '-n', 'hw.memsize'): fixture('hw_memsize.txt'), ('sw_vers', '-productVersion'): fixture('sw_vers.txt'),
            ('ps', '-A', '-o', 'pid=,%cpu=,comm='): LOAD_QUIET}
        self.outputs.update(outputs)

    def __call__(self, argv):
        self.calls.append(list(argv))
        output = self.outputs.get(tuple(argv), 'MISSING')
        if output is None:
            return hosts.CommandResult(1, '', 'simulated failure')
        return hosts.CommandResult(0, output, '')


def host(outputs=None, **options):
    executor = FakeExecutor(outputs or {}, **options)
    return hosts.MacOSHost(execute=executor, residency=lambda path: dict(resident_bytes=4096), clock=lambda: 1.5), executor


class MacOSHostTest(unittest.TestCase):
    def test_snapshot_values(self):
        snapshot = host()[0].snapshot(Path('model'))
        self.assertEqual(snapshot['platform'], 'macos')
        self.assertTrue(snapshot['power']['ac_power'])
        self.assertEqual(snapshot['memory']['source'], 'vm_stat')
        self.assertIn('Pages free', snapshot['memory']['raw'])
        self.assertIn('vm.loadavg', snapshot['load'])
        self.assertEqual(snapshot['residency'], {'base.gguf': dict(resident_bytes=4096)})

    def test_top_processes_keep_only_executable_names_and_cpu(self):
        top = host()[0].snapshot(Path('model'))['top_processes']
        self.assertEqual(len(top), 5)
        self.assertEqual(top[:2], [dict(name='worker', cpu_percent=7.0), dict(name='Example Helper', cpu_percent=2.5)])
        for process in top:
            self.assertEqual(set(process), {'name', 'cpu_percent'})
            self.assertNotIn('/', process['name'])
        self.assertNotIn('/', json.dumps(top))

    def test_failed_command_makes_only_that_item_unavailable(self):
        snapshot = host({('pmset', '-g', 'therm'): None})[0].snapshot(Path('model'))
        self.assertFalse(snapshot['thermal']['available'])
        self.assertIn('pmset', snapshot['thermal']['reason'])
        self.assertTrue(hosts.is_unavailable(snapshot['thermal']))
        self.assertIn('Pages free', snapshot['memory']['raw'])
        self.assertTrue(snapshot['power']['ac_power'])

    def test_missing_file_residency_is_unavailable_not_zero(self):
        def missing(path):
            raise FileNotFoundError(str(path))
        subject = hosts.MacOSHost(execute=FakeExecutor({}), residency=missing)
        value = subject.snapshot(Path('model'))['residency']
        self.assertTrue(hosts.is_unavailable(value))

    def test_preflight_requires_ac_power(self):
        host()[0].preflight()
        with self.assertRaises(hosts.PreflightError):
            host(battery='pmset_batt_battery.txt')[0].preflight()
        with self.assertRaises(hosts.PreflightError):
            host({('pmset', '-g', 'batt'): None})[0].preflight()

    def test_device(self):
        device = host()[0].device()
        self.assertEqual((device['model'], device['chip'], device['memory_bytes'], device['os_version']),
                         ('Mac17,3', 'Apple M5', 17179869184, '26.5.1'))
        partial = host({('sysctl', '-n', 'hw.memsize'): None})[0].device()
        self.assertTrue(hosts.is_unavailable(partial['memory_bytes']))
        self.assertEqual(partial['model'], 'Mac17,3')

    def test_run_passes_argument_list_through(self):
        subject, executor = host({('echo', 'a b'): 'ok'})
        self.assertEqual(subject.run(['echo', 'a b']).stdout, 'ok')
        self.assertEqual(executor.calls[-1], ['echo', 'a b'])


class BackgroundLoadTest(unittest.TestCase):
    KEY = ('ps', '-A', '-o', 'pid=,%cpu=,comm=')

    def check(self, *rows):
        return host({self.KEY: load_listing(*rows)})[0].check_background_load()

    def test_non_system_process_above_threshold_fails_with_its_name_only(self):
        with self.assertRaises(hosts.BackgroundLoadError) as raised:
            self.check((200, 16.0, '/Applications/Lark.app/Contents/Helpers/Lark Helper (Iron)'))
        message = str(raised.exception)
        self.assertIn('Lark Helper (Iron)', message)
        self.assertNotIn('/Applications', message)
        self.assertIsInstance(raised.exception, hosts.PreflightError)

    def test_exactly_at_threshold_passes(self):
        self.assertEqual(self.check((200, 15.0, '/Applications/Example.app/Example'))['other'], [])

    def test_system_process_is_recorded_but_does_not_block(self):
        load = self.check((201, 40.0, '/System/Library/CoreServices/WindowServer'), (202, 30.0, '/usr/sbin/coreaudiod'),
                          (203, 20.0, '/Library/Apple/System/Library/Example'), (204, 20.0, '/bin/example'), (205, 20.0, '/sbin/example'))
        self.assertEqual(load['other'], [])
        self.assertEqual([e['name'] for e in load['system']], ['WindowServer', 'coreaudiod', 'Example', 'example', 'example'])
        self.assertNotIn('/', json.dumps(load))

    def test_lookalike_directories_are_not_system(self):
        with self.assertRaises(hosts.BackgroundLoadError):
            self.check((206, 50.0, '/Users/someone/usr/bin/tool'))
        with self.assertRaises(hosts.BackgroundLoadError):
            self.check((207, 50.0, '/System2/tool'))

    def test_own_process_and_parent_are_not_counted(self):
        load = self.check((os.getpid(), 90.0, '/opt/python/bin/python'), (os.getppid(), 80.0, '/opt/uv/bin/uv'))
        self.assertEqual(load['other'], [])

    def test_unreadable_process_list_refuses(self):
        with self.assertRaises(hosts.BackgroundLoadError):
            host({self.KEY: None})[0].check_background_load()

    def test_snapshot_records_names_only(self):
        shot = host({self.KEY: load_listing((300, 50.0, '/System/Library/WindowServer'), (301, 20.0, '/Applications/X.app/X Helper'))})[0].snapshot(Path('model'))
        self.assertEqual(shot['background_load']['system'], [dict(name='WindowServer', cpu_percent=50.0)])
        self.assertEqual(shot['background_load']['other'], [dict(name='X Helper', cpu_percent=20.0)])
        self.assertNotIn('/', json.dumps(shot['background_load']))


class SelectionTest(unittest.TestCase):
    def test_unimplemented_platforms_are_refused_with_provider(self):
        with self.assertRaises(hosts.UnsupportedPlatform) as raised:
            hosts.select('unregistered')
        self.assertIn('has no measurement host', str(raised.exception))

    def test_macos_is_available(self):
        self.assertEqual(hosts.select('macos').name, 'macos')

    def test_windows_is_available(self):
        self.assertEqual(hosts.select('windows').name, 'windows')

    def test_command_line_exits_two_for_unimplemented_platform_without_running_anything(self):
        marker = HERE / '.never-created'
        result = subprocess.run([sys.executable, '-I', str(HERE / 'abba.py'), '--platform', 'unregistered', '--a', json.dumps(['touch', str(marker)]),
                                 '--b', '[]', '--out', 'unused', '--raw', 'unused'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('has no measurement host', result.stderr)
        self.assertFalse(marker.exists())



class AndroidHostTest(unittest.TestCase):
    class Transport:
        root = 'device-root'
        def digest(self, path):
            import hashlib
            return hashlib.sha256(b'binary').hexdigest()
        def __init__(self):
            self.offline = False
            self.data = {
                ('dumpsys', 'battery'): 'AC powered: false\nUSB powered: true',
                ('dumpsys', 'power'): 'mWakefulness=Awake',
                ('dumpsys', 'thermalservice'): 'Thermal Status: 0\nTemperature: 30',
                ('cat', '/proc/meminfo'): 'MemTotal: 16384 kB',
                ('cat', '/proc/loadavg'): '1 2 3 1/4 5',
                ('getprop', 'ro.product.model'): 'PKR110',
                ('getprop', 'ro.soc.model'): 'SM8750',
                ('getprop', 'ro.build.version.release'): '16',
                ('getprop', 'ro.build.version.sdk'): '36',
                ('ps', '-A', '-o', 'NAME'): 'NAME\n/system/bin/app',
            }
        def reachable(self):
            if self.offline:
                raise RuntimeError('offline')
        def shell(self, *args):
            if args[:2] == ('sh', '-c') and not self.offline:
                return hosts.CommandResult(0, '/sys/devices/system/cpu/cpufreq/policy0\n3000\n4000', '')
            if args not in self.data:
                raise RuntimeError('unavailable: ' + str(args))
            return hosts.CommandResult(0, self.data[args], '')

    def test_preflight_and_snapshot(self):
        transport = self.Transport()
        host = hosts.AndroidHost(transport=transport)
        host.preflight()
        shot = host.snapshot('ignored')
        self.assertEqual(shot['cpufreq'][0]['scaling_max_freq_khz'], 3000)
        self.assertEqual(shot['cpufreq'][0]['cpuinfo_max_freq_khz'], 4000)
        self.assertEqual(shot['memory']['source'], 'meminfo')
        self.assertFalse(shot['residency']['available'])
        self.assertTrue(shot['residency']['reason'])
        self.assertEqual(shot['top_processes'], [dict(name='app')])
        self.assertEqual(host.device()['model'], 'PKR110')
        self.assertEqual(host.device()['memory_bytes'], 16384*1024)
        for key, value in [(('dumpsys', 'battery'), 'USB powered: false'), (('dumpsys', 'power'), 'mWakefulness=Asleep')]:
            saved = transport.data[key]; transport.data[key] = value
            with self.assertRaises(hosts.PreflightError): host.preflight()
            transport.data[key] = saved
        transport.offline = True
        with self.assertRaises(hosts.PreflightError): host.preflight()
        transport.data.clear()
        shot = host.snapshot('ignored')
        for key in ('power', 'thermal', 'cpufreq', 'memory', 'load', 'top_processes', 'residency'):
            self.assertFalse(shot[key]['available'], key)
            self.assertTrue(shot[key]['reason'], key)

    def test_run_reuses_wrapper_and_direct_is_excluded(self):
        calls = []
        host = hosts.AndroidHost(transport=self.Transport(), execute=lambda args: calls.append(args))
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'binary'; binary.write_bytes(b'binary')
            host.run([str(binary), 'bench', '--model', 'model'])
            binary.write_bytes(b'changed')
            result = host.run([str(binary), 'bench'])
            self.assertEqual(result.returncode, 1)
            self.assertIn('binary differs', result.stderr)
        self.assertTrue(calls[0][0].endswith('android-cli'))
        self.assertEqual(calls[0][1:], ['bench', '--model', 'model'])
        self.assertIsNotNone(host.unsupported_configuration('direct'))
        self.assertIsNone(host.unsupported_configuration('actual'))

class WindowsFakeExecutor:
    """Maps the WindowsHost command lists to recorded output; a value of None makes the command fail."""

    def __init__(self, outputs):
        self.calls = []
        W = hosts.WindowsHost
        self.outputs = {
            tuple(W.POWER): '',  # a desktop reports no battery device, which is AC power
            tuple(W.MEMORY): json.dumps({'FreePhysicalMemory': 4000000, 'TotalVisibleMemorySize': 16000000}),
            tuple(W.TOP): json.dumps([{'Name': 'edge0-infer', 'PercentProcessorTime': 80},
                                      {'Name': 'explorer', 'PercentProcessorTime': 5}]),
            tuple(W.LOAD): json.dumps([{'pid': 4321, 'path': 'C:\\Windows\\System32\\MsMpEng.exe', 'cpu_percent': 40},
                                       {'pid': 4322, 'path': 'C:\\Users\\me\\game.exe', 'cpu_percent': 3}]),
            tuple(W.MODEL): json.dumps({'Model': 'OMEN 25L', 'TotalPhysicalMemory': 17043451904}),
            tuple(W.CHIP): '12th Gen Intel(R) Core(TM) i5-12400',
            tuple(W.FREQUENCY): json.dumps({'nominal_mhz': 2500, 'current_mhz': 2500, 'performance_percent': 168}),
            tuple(W.OSVER): '10.0.19045'}
        self.outputs.update(outputs)

    def __call__(self, argv):
        self.calls.append(list(argv))
        output = self.outputs.get(tuple(argv), 'MISSING')
        if output is None:
            return hosts.CommandResult(1, '', 'simulated failure')
        return hosts.CommandResult(0, output, '')


def win_host(outputs=None):
    executor = WindowsFakeExecutor(outputs or {})
    return hosts.WindowsHost(execute=executor, clock=lambda: 2.5), executor


class WindowsHostTest(unittest.TestCase):
    def test_snapshot_values(self):
        shot = win_host()[0].snapshot(Path('model'))
        self.assertEqual(shot['platform'], 'windows')
        self.assertEqual(shot['time'], 2.5)
        self.assertTrue(shot['power']['ac_power'])
        self.assertEqual(shot['memory']['source'], 'Win32_OperatingSystem')
        self.assertIn('FreePhysicalMemory', shot['memory']['raw'])

    def test_items_without_a_windows_query_are_unavailable_with_a_reason(self):
        shot = win_host()[0].snapshot(Path('model'))
        for key in ('thermal', 'load', 'residency'):
            self.assertTrue(hosts.is_unavailable(shot[key]), key)
            self.assertTrue(shot[key]['reason'], key)

    def test_failed_command_makes_only_that_item_unavailable(self):
        shot = win_host({tuple(hosts.WindowsHost.MEMORY): None})[0].snapshot(Path('model'))
        self.assertTrue(hosts.is_unavailable(shot['memory']))
        self.assertIn('powershell', shot['memory']['reason'])
        self.assertTrue(shot['power']['ac_power'])

    def test_cpu_frequency_uses_the_performance_counter(self):
        freq = win_host()[0].snapshot(Path('model'))['cpufreq']
        self.assertEqual(freq, dict(nominal_mhz=2500, reported_current_mhz=2500, performance_percent=168, effective_mhz=4200))

    def test_cpu_frequency_without_counter_or_query_is_unavailable_with_a_reason(self):
        key = tuple(hosts.WindowsHost.FREQUENCY)
        freq = win_host({key: json.dumps({'nominal_mhz': 2500, 'current_mhz': 2500, 'performance_percent': None})})[0] \
            .snapshot(Path('model'))['cpufreq']
        self.assertEqual(freq['nominal_mhz'], 2500)
        self.assertTrue(hosts.is_unavailable(freq['effective_mhz']) and freq['effective_mhz']['reason'])
        for output in (None, 'not json', json.dumps({'current_mhz': 2500})):
            with self.subTest(output=output):
                freq = win_host({key: output})[0].snapshot(Path('model'))['cpufreq']
                self.assertTrue(hosts.is_unavailable(freq) and freq['reason'])

    def test_top_processes_keep_only_name_and_cpu(self):
        top = win_host()[0].snapshot(Path('model'))['top_processes']
        self.assertEqual(top, [dict(name='edge0-infer', cpu_percent=80.0), dict(name='explorer', cpu_percent=5.0)])
        self.assertNotIn('\\', json.dumps(top))

    def test_device(self):
        device = win_host()[0].device()
        self.assertEqual((device['model'], device['chip'], device['memory_bytes'], device['os_version']),
                         ('OMEN 25L', '12th Gen Intel(R) Core(TM) i5-12400', 17043451904, '10.0.19045'))
        partial = win_host({tuple(hosts.WindowsHost.MODEL): None})[0].device()
        self.assertTrue(hosts.is_unavailable(partial['memory_bytes']))
        self.assertTrue(hosts.is_unavailable(partial['model']))
        self.assertEqual(partial['chip'], '12th Gen Intel(R) Core(TM) i5-12400')

    def test_preflight_requires_ac_power(self):
        win_host()[0].preflight()  # a desktop without a battery is on AC
        with self.assertRaises(hosts.PreflightError):
            win_host({tuple(hosts.WindowsHost.POWER): '1'})[0].preflight()  # discharging on battery
        with self.assertRaises(hosts.PreflightError):
            win_host({tuple(hosts.WindowsHost.POWER): None})[0].preflight()  # power source unreadable

    def test_run_passes_argument_list_through(self):
        subject, executor = win_host({('edge0-infer', 'bench'): 'ok'})
        self.assertEqual(subject.run(['edge0-infer', 'bench']).stdout, 'ok')
        self.assertEqual(executor.calls[-1], ['edge0-infer', 'bench'])

    def test_direct_configuration_is_unsupported(self):
        self.assertIsNotNone(win_host()[0].unsupported_configuration('direct'))
        self.assertIsNone(win_host()[0].unsupported_configuration('actual'))


class WindowsBackgroundLoadTest(unittest.TestCase):
    KEY = tuple(hosts.WindowsHost.LOAD)

    def host(self, records):
        return win_host({self.KEY: json.dumps(records)})[0]

    def test_non_system_process_above_threshold_fails_with_its_name_only(self):
        with self.assertRaises(hosts.BackgroundLoadError) as raised:
            self.host([{'pid': 500, 'path': 'C:\\Users\\me\\Games\\Lark.exe', 'cpu_percent': 60}]).check_background_load()
        message = str(raised.exception)
        self.assertIn('Lark.exe', message)
        self.assertNotIn('Users', message)
        self.assertNotIn('\\', message)
        self.assertIsInstance(raised.exception, hosts.PreflightError)

    def test_exactly_at_threshold_passes(self):
        load = self.host([{'pid': 700, 'path': 'C:\\Users\\me\\app.exe', 'cpu_percent': 15}]).check_background_load()
        self.assertEqual(load['other'], [])

    def test_system_process_is_recorded_but_does_not_block(self):
        load = self.host([{'pid': 600, 'path': 'C:\\Windows\\System32\\MsMpEng.exe', 'cpu_percent': 55},
                          {'pid': 601, 'path': 'C:\\WINDOWS\\explorer.exe', 'cpu_percent': 30}]).check_background_load()
        self.assertEqual(load['other'], [])
        self.assertEqual([e['name'] for e in load['system']], ['MsMpEng.exe', 'explorer.exe'])
        self.assertNotIn('\\', json.dumps(load))

    def test_own_process_and_parent_are_not_counted(self):
        load = self.host([{'pid': os.getpid(), 'path': 'C:\\Users\\me\\python.exe', 'cpu_percent': 90},
                          {'pid': os.getppid(), 'path': 'C:\\Users\\me\\uv.exe', 'cpu_percent': 80}]).check_background_load()
        self.assertEqual(load['other'], [])

    def test_unreadable_process_list_refuses(self):
        with self.assertRaises(hosts.BackgroundLoadError):
            win_host({self.KEY: None})[0].check_background_load()

    def test_snapshot_records_names_only(self):
        shot = self.host([{'pid': 800, 'path': 'C:\\Windows\\svc.exe', 'cpu_percent': 50},
                          {'pid': 801, 'path': 'C:\\Apps\\X.exe', 'cpu_percent': 20}]).snapshot(Path('model'))
        self.assertEqual(shot['background_load']['system'], [dict(name='svc.exe', cpu_percent=50.0)])
        self.assertEqual(shot['background_load']['other'], [dict(name='X.exe', cpu_percent=20.0)])
        self.assertNotIn('\\', json.dumps(shot['background_load']))


class IOSHostTest(unittest.TestCase):
    def test_probe_and_identity(self):
        class Transport:
            def reachable(self):
                return dict(hardwareProperties=dict(productType='SyntheticPhone', udid='private'),
                            deviceProperties=dict(osVersionNumber='1', name='private'))
        def runner(device, argv, stdin, probe):
            probe.write_text(json.dumps(dict(before=dict(thermal_state='nominal', os_proc_available_memory=123, physical_memory=456),
                                             after=dict(thermal_state='fair', os_proc_available_memory=100, physical_memory=456))))
            return 0, b'output', b'error'
        subject=hosts.IOSHost(transport=Transport(),runner=runner)
        subject.preflight(); before=subject.snapshot(Path('model'))
        result=subject.run(['cli','bench'])
        self.assertEqual(result.returncode,0)
        self.assertEqual(before['process_available_memory'],123)
        self.assertEqual(subject.snapshot(Path('model'))['thermal'],'fair')
        self.assertTrue(hosts.is_unavailable(before['cpufreq']))
        self.assertNotIn('private',json.dumps(subject.device()))

    def test_connection_rejection(self):
        for reason in ('unreachable','requires a data cable','locked'):
            class Transport:
                def reachable(self): raise RuntimeError(reason)
            with self.assertRaises(hosts.PreflightError): hosts.IOSHost(transport=Transport()).preflight()

    def test_fake_devicectl_preflight(self):
        sys.path.insert(0, str(HERE.parent / 'platform'))
        from ios import IOS
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, MOE_IOS_DEVICE='synthetic-device', MOE_IOS_TEAM='synthetic-team',
                       MOE_DEVICECTL=str(HERE.parent / 'platform/fake_devicectl.py'), FAKE_IOS_ROOT=temp)
            hosts.IOSHost(transport=IOS(environ=env)).preflight()
            for key, value in [('FAKE_IOS_UNREACHABLE','1'), ('FAKE_IOS_TRANSPORT','wifi'), ('FAKE_IOS_LOCKED','1')]:
                with self.assertRaises(hosts.PreflightError):
                    hosts.IOSHost(transport=IOS(environ=dict(env, **{key:value}))).preflight()

    def test_missing_probe(self):
        subject=hosts.IOSHost(transport=object(),runner=lambda *args:(1,b'',b'failed'))
        before=subject.snapshot(Path('model')); subject.run(['cli','bench'])
        self.assertTrue(hosts.is_unavailable(before['thermal']))
        self.assertTrue(hosts.is_unavailable(subject.snapshot(Path('model'))['thermal']))


if __name__ == '__main__':
    unittest.main()
