"""Check the platform system fields through the existing info command."""
import argparse
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', required=True)
    parser.add_argument('--platform', required=True, choices=('android','ios','windows'))
    args = parser.parse_args()
    result = subprocess.run([args.cli,'info'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    fields = {}
    for line in result.stdout.splitlines():
        if line.startswith('system.'):
            key, value = line.split(': ',1)
            parts = shlex.split(value)
            fields[key] = (parts[0],dict(item.split('=',1) for item in parts[1:]))
    def number(name):
        value, meta = fields['system.'+name]
        assert value != 'unavailable' and meta['source'] and not meta['reason'], (name,fields)
        return int(value)
    total = number('memory.total_bytes'); available = number('memory.available_bytes')
    assert total > 0 and 0 <= available <= total
    state, meta = fields['system.thermal']
    if args.platform == 'ios':
        assert state in ('nominal','fair','serious','critical')
        assert number('app_memory_limit') >= number('app_memory_limit.current_physical_bytes') > 0
    elif args.platform == 'android':
        assert state in ('none','light','moderate','severe','critical','emergency','shutdown') or state == 'unavailable' and meta['reason']
    else:
        assert state == 'unavailable' and meta['reason']
    print(result.stdout, end='')
    print('PASS system memory, thermal source and availability')


if __name__ == '__main__':
    main()
