"""Compare every incremental return value against Python codecs, including finish."""
import argparse
import codecs
import json
from pathlib import Path
import random
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('executable')
    parser.add_argument('golden', type=Path)
    args = parser.parse_args()
    subprocess.run([args.executable], check=True)
    process = subprocess.Popen(
        [args.executable, '--decode-steps'], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True)
    cases = 0

    def check(chunks):
        nonlocal cases
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        expected = [decoder.decode(chunk).encode().hex() for chunk in chunks]
        expected.append(decoder.decode(b'', final=True).encode().hex())
        process.stdin.write(';'.join(chunk.hex() for chunk in chunks) + '\n')
        process.stdin.flush()
        response = process.stdout.readline()
        if not response:
            raise AssertionError('native decoder terminated early')
        actual = response.rstrip('\n').split(':')
        if actual != expected:
            raise AssertionError(json.dumps({
                'chunks': [chunk.hex() for chunk in chunks],
                'native_steps': actual, 'python_steps': expected}))
        cases += 1

    try:
        # The independent acceptance counterexamples, with and without more input.
        for chunks in [[b'\xed', b'\xba'], [b'\xed', b'\xba', b'H'],
                       [b'\xed\xba', b''], [b'\xed', b'', b'\xba', b'\x80']]:
            check(chunks)
        # Every two-byte prefix, delivered both together and byte by byte.
        for first in range(256):
            for second in range(256):
                check([bytes([first]), bytes([second])])
                check([bytes([first, second])])
        rng = random.Random(98412)
        for _ in range(1500):
            raw = bytes(rng.randrange(256) for _ in range(rng.randrange(30)))
            check([bytes([byte]) for byte in raw] or [b''])
        for row in json.loads(args.golden.read_bytes()):
            raw = bytes.fromhex(row['input'])
            for _ in range(1000):
                chunks = []
                offset = 0
                while offset < len(raw):
                    size = rng.randrange(1, 8)
                    chunks.append(raw[offset:offset + size])
                    offset += size
                check(chunks or [b''])
    finally:
        process.stdin.close()
        process.stdout.close()
        returncode = process.wait()
    assert returncode == 0, returncode
    print(f'Python per-call decode oracle: {cases} partitions passed, including finish')


if __name__ == '__main__':
    main()
