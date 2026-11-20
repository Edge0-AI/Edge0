"""How much of each file is in the page cache (mincore), without reading the file.

Usage: file_cache.py FILE [FILE ...] [--json]

Maps each file read-only, asks mincore which pages are resident and prints the share, e.g.
"experts.bin  40.08 GB  cached 45.90%". With --json, one JSON object with the same numbers.
Standard library only (ctypes), macOS or Linux.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
MAP_FAILED = ctypes.c_void_p(-1).value
WINDOW = 1 << 30


def cached_share(path):
    """(bytes, cached bytes) of the file."""
    size = os.path.getsize(path)
    page = mmap.PAGESIZE
    cached = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while offset < size:
            length = min(WINDOW, size - offset)
            address = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, offset)
            if address in (None, MAP_FAILED):
                raise OSError(ctypes.get_errno(), f'mmap {path}')
            try:
                pages = (length + page - 1) // page
                vector = ctypes.create_string_buffer(pages)
                if libc.mincore(address, length, vector) != 0:
                    raise OSError(ctypes.get_errno(), f'mincore {path}')
                resident = sum(1 for b in vector.raw if b & 1)
                cached += min(length, resident * page)
            finally:
                libc.munmap(address, length)
            offset += length
    finally:
        os.close(fd)
    return size, cached


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = {}
    for path in args.files:
        size, cached = cached_share(path)
        result[path] = {'bytes': size, 'cached_bytes': cached, 'cached_percent': round(100.0 * cached / max(1, size), 2)}
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        for path, r in result.items():
            print(f'{path}  {r["bytes"] / 1e9:.2f} GB  cached {r["cached_percent"]:.2f}%')


if __name__ == '__main__':
    main()
