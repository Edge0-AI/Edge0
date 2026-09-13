"""Local reproducible coding-conversation benchmark; JSON output.

Run once normally, then with --restart against the same cache directory.
Restart means a new Python/model process, not a cold OS filesystem cache.
"""
import argparse
import json
import resource
import threading
import time
from pathlib import Path

import psutil
from edge0 import AutoEngine
from edge0.backends import core
from edge0.conversation import CacheConfig
from edge0.conversation.store import Radix
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession


def fixture():
    code = '\n\n'.join(
        f'def normalize_path_{i}(path: str) -> str:\n'
        f'    """Normalize a path for source module {i}."""\n'
        '    parts = [part for part in path.split("/") if part and part != "."]\n'
        '    return "/".join(parts)'
        for i in range(48))
    return [ChatMessage('user', 'Review this Python module. Explain two concrete improvements.\n```python\n' + code + '\n```')]


def measure(engine, name, messages):
    process = psutil.Process()
    rss = [process.memory_info().rss]
    stopped = threading.Event()
    def watch():
        while not stopped.wait(0.02):
            rss.append(process.memory_info().rss)
    thread = threading.Thread(target=watch)
    thread.start()
    core.reset_peak_memory()
    before = resource.getrusage(resource.RUSAGE_SELF)
    disk_before = psutil.disk_io_counters()
    start = time.perf_counter()
    first = []
    def token(_):
        if not first:
            first.append(time.perf_counter() - start)
    try:
        tokens, meta = ChatSession(engine, ChatRequest(engine.name, messages,
            temperature=0, max_tokens=8)).run(on_token=token)
    finally:
        stopped.set()
        thread.join()
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    disk_after = psutil.disk_io_counters()
    result = dict(case=name, output_tokens=tokens, ttft_s=first[0] if first else None, wall_s=wall,
        post_first_token_effective_tok_s=(len(tokens) - 1) / (wall - first[0]) if first and len(tokens) > 1 else None,
        tokenization_s=engine.last_tokenization_s,
        **engine.last_generation_metrics,
        peak_mlx_bytes=core.get_peak_memory(), peak_rss_bytes=max(rss),
        io_read_blocks=after.ru_inblock - before.ru_inblock,
        io_write_blocks=after.ru_oublock - before.ru_oublock,
        system_disk_read_bytes=disk_after.read_bytes - disk_before.read_bytes,
        system_disk_write_bytes=disk_after.write_bytes - disk_before.write_bytes,
        **{k: v for k, v in meta.items() if k != "wall_s"})
    print(json.dumps(result), flush=True)
    return tokens


def lookup_benchmark():
    for count in (1000, 10000, 100000):
        tree = Radix()
        prefix = list(range(128))
        for i in range(count):
            tree.insert(prefix + [i, i + 1], i)
        started = time.perf_counter()
        for i in range(10000):
            assert tree.longest(prefix + [i % count, i % count + 1, -1]) == i % count
        print(json.dumps(dict(case='radix_lookup', checkpoints=count,
                              lookup_us=(time.perf_counter() - started) * 100)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir')
    parser.add_argument('--cache-dir')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--lookup-only', action='store_true')
    args = parser.parse_args()
    if args.lookup_only:
        lookup_benchmark()
        return
    engine = AutoEngine.from_pretrained(args.model_dir, name='edge0-8b',
        conversation_cache=CacheConfig(args.cache_dir))
    try:
        messages = fixture()
        if args.restart:
            measure(engine, 'process_restart', messages)
            return
        cache = engine.conversation_cache
        cache.clear()
        engine.conversation_cache = None
        baseline = measure(engine, 'disabled', messages)
        engine.conversation_cache = cache
        tokens = measure(engine, 'first_write', messages)
        repeated = measure(engine, 'repeat', messages)
        assert baseline == tokens == repeated, 'repeated prompt continuation changed'
        appended = messages + [ChatMessage('assistant', engine._tok.decode(tokens)),
                               ChatMessage('user', 'Add type annotations and unit tests for the first function.')]
        measure(engine, 'appended_turn', appended)
        branch = messages + [ChatMessage('assistant', engine._tok.decode(tokens)),
                             ChatMessage('user', 'Instead, explain how to handle parent directory components.')]
        measure(engine, 'branch', branch)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
