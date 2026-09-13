"""Locked, transactional, content-addressed checkpoint storage.

The index holds token edges only. Payloads are opened only for the selected
boundary; no inactive tensors are retained. SQLite is the publication point.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_default(value):
    from enum import Enum
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported namespace value: {type(value)}")


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=_json_default).encode()


@dataclass(frozen=True)
class CacheConfig:
    directory: str = ''
    budget_bytes: int = 20 * 1024**3
    interval: int = 2048
    token_budget_bytes: int = 16 * 1024**2

    def __post_init__(self):
        if self.budget_bytes < 0 or self.token_budget_bytes < 0 or self.interval < 1:
            raise ValueError('cache budgets must be nonnegative and interval positive')


class Radix:
    def __init__(self):
        self.children = {}
        self.value = None

    def insert(self, tokens, value):
        node = self
        tokens = tuple(tokens)
        while tokens:
            item = node.children.get(tokens[0])
            if item is None:
                child = Radix()
                child.value = value
                node.children[tokens[0]] = (tokens, child)
                return
            edge, child = item
            n = 0
            while n < min(len(edge), len(tokens)) and edge[n] == tokens[n]:
                n += 1
            if n < len(edge):
                split = Radix()
                split.children[edge[n]] = (edge[n:], child)
                node.children[tokens[0]] = (edge[:n], split)
                child = split
            node, tokens = child, tokens[n:]
        node.value = value

    def remove(self, tokens):
        node, pos, parents = self, 0, []
        while pos < len(tokens):
            item = node.children.get(tokens[pos])
            if item is None:
                return
            edge, child = item
            if tuple(tokens[pos:pos + len(edge)]) != edge:
                return
            parents.append((node, tokens[pos]))
            pos += len(edge)
            node = child
        node.value = None
        for parent, key in reversed(parents):
            edge, child = parent.children[key]
            if child.value is None and not child.children:
                del parent.children[key]
            elif child.value is None and len(child.children) == 1:
                tail, grandchild = next(iter(child.children.values()))
                parent.children[key] = (edge + tail, grandchild)

    def longest(self, tokens):
        node, pos, best = self, 0, None
        while pos < len(tokens):
            item = node.children.get(tokens[pos])
            if item is None:
                break
            edge, child = item
            if tuple(tokens[pos:pos + len(edge)]) != edge:
                break
            pos += len(edge)
            node = child
            if node.value is not None:
                best = node.value
        return best


class _Oversized(ValueError):
    pass


def _managed_payload(path):
    return bool(re.fullmatch(r'[0-9a-f]{64}\.safetensors|[0-9a-f]{32}\.tmp(?:\.safetensors)?', path.name))


class CheckpointStore:
    def __init__(self, config: CacheConfig, namespace: str, *, maintenance=True):
        self.config, self.namespace = config, namespace
        self.maintenance = maintenance
        self.root = Path(config.directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._version = None
        self.index = Radix()
        self.metrics = {}
        with self.locked(enforce=maintenance) as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY, ns TEXT, tokens BLOB, manifest BLOB, used REAL);
                CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, size INTEGER);
                CREATE TABLE IF NOT EXISTS refs (checkpoint TEXT, object TEXT,
                    PRIMARY KEY(checkpoint, object));
                CREATE TABLE IF NOT EXISTS tokenizations (
                    id TEXT PRIMARY KEY, value BLOB, used REAL);
                CREATE TABLE IF NOT EXISTS fingerprints (
                    path TEXT PRIMARY KEY, identity TEXT, hash TEXT);
                CREATE TABLE IF NOT EXISTS revision (value INTEGER);
                INSERT INTO revision SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM revision);
            ''')
            for table in ('checkpoints', 'tokenizations'):
                if 'checksum' not in {row[1] for row in db.execute(f'PRAGMA table_info({table})')}:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN checksum TEXT')
            if maintenance:
                self._cleanup(db)
                self._evict(db)

    @contextmanager
    def locked(self, *, enforce=False):
        with self._mutex, open(self.root / 'cache.lock', 'a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(self.root / 'metadata.sqlite')
            try:
                with db:
                    yield db
                if self.maintenance and enforce:
                    self._bound_directory(db)
            finally:
                db.close()
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _bound_directory(self, db):
        # Include allocated SQLite pages, not just live tensor bytes. The
        # separate tokenization allowance and a fixed 64 KiB schema reserve
        # keep even tiny test budgets usable. Compact only under pressure.
        limit = self.config.budget_bytes + self.config.token_budget_bytes + 65536
        def physical():
            return sum(p.stat().st_size for p in self.root.iterdir()
                       if p.is_file() and (_managed_payload(p) or p.name == 'metadata.sqlite'))
        if physical() <= limit:
            return
        db.execute('VACUUM')
        while physical() > limit:
            row = db.execute('SELECT id FROM checkpoints ORDER BY used LIMIT 1').fetchone()
            with db:
                if row:
                    self._delete(db, row[0])
                    self.metrics['evictions'] = self.metrics.get('evictions', 0) + 1
                    self._cleanup(db)
                else:
                    db.execute('DELETE FROM tokenizations')
                    db.execute('DELETE FROM fingerprints')
            db.execute('VACUUM')
            if not row:
                break

    def _refresh(self, db):
        version = db.execute('SELECT value FROM revision').fetchone()[0]
        if version != self._version:
            self.index = Radix()
            for key, tokens in db.execute('SELECT id,tokens FROM checkpoints WHERE ns=?', (self.namespace,)):
                try:
                    self.index.insert(json.loads(tokens), key)
                except (ValueError, TypeError):
                    self._delete(db, key)
            self._version = version

    def _delete(self, db, key):
        row = db.execute('SELECT ns,tokens FROM checkpoints WHERE id=?', (key,)).fetchone()
        if row and row[0] == self.namespace:
            try:
                self.index.remove(json.loads(row[1]))
            except (ValueError, TypeError):
                pass
        db.execute('DELETE FROM checkpoints WHERE id=?', (key,))
        db.execute('DELETE FROM refs WHERE checkpoint=?', (key,))
        db.execute('UPDATE revision SET value=value+1')

    def _cleanup(self, db):
        db.execute('DELETE FROM objects WHERE id NOT IN (SELECT object FROM refs)')
        live = {r[0] for r in db.execute('SELECT id FROM objects')}
        for path in self.root.glob('*.safetensors'):
            if _managed_payload(path) and path.stem not in live:
                path.unlink(missing_ok=True)
        for path in self.root.glob('*.tmp'):
            if _managed_payload(path):
                path.unlink(missing_ok=True)

    def _evict(self, db):
        while self._bytes(db) > self.config.budget_bytes:
            row = db.execute('SELECT id FROM checkpoints ORDER BY used LIMIT 1').fetchone()
            if row is None:
                break
            self._delete(db, row[0])
            db.execute('DELETE FROM objects WHERE id NOT IN (SELECT object FROM refs)')
            self.metrics['evictions'] = self.metrics.get('evictions', 0) + 1
        self._cleanup(db)

    def _bytes(self, db):
        return db.execute('SELECT coalesce(sum(size),0) FROM objects').fetchone()[0]

    def restore(self, tokens, reader):
        started = time.perf_counter()
        with self.locked() as db:
            self._refresh(db)
            key = self.index.longest(tokens)
            self.metrics['lookup_s'] = time.perf_counter() - started
            if key is None:
                return None
            row = db.execute('SELECT tokens,manifest,checksum FROM checkpoints WHERE id=?', (key,)).fetchone()
            try:
                if digest(row[0] + row[1]) != row[2]:
                    raise ValueError('metadata checksum mismatch')
                saved, manifest = json.loads(row[0]), json.loads(row[1])
                if saved != list(tokens[:len(saved)]):
                    raise ValueError('token mismatch')
                paths = {}
                started = time.perf_counter()
                for obj, in db.execute('SELECT object FROM refs WHERE checkpoint=?', (key,)):
                    path = self.root / (obj + '.safetensors')
                    if file_hash(path) != obj:
                        raise ValueError('checksum mismatch')
                    paths[obj] = path
                state = reader(manifest, paths)
                db.execute('UPDATE checkpoints SET used=? WHERE id=?', (time.time(), key))
                self.metrics['restore_s'] = time.perf_counter() - started
                self.metrics['reused_tokens'] = len(saved)
                return saved, state
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                self._delete(db, key)
                self._cleanup(db)
                self.metrics['corrupt_entries'] = self.metrics.get('corrupt_entries', 0) + 1
                return None

    def publish(self, tokens, writer):
        if not tokens:
            return
        started = time.perf_counter()
        key = digest(packed([self.namespace, tokens]))
        with self.locked(enforce=True) as db:
            self._refresh(db)
            if db.execute('SELECT 1 FROM checkpoints WHERE id=?', (key,)).fetchone():
                return
            objects = {}
            def put(save):
                path = self.root / (uuid.uuid4().hex + '.tmp.safetensors')
                save(path)
                with open(path, 'rb') as f:
                    os.fsync(f.fileno())
                obj = file_hash(path)
                size = path.stat().st_size
                dest = self.root / (obj + '.safetensors')
                if dest.exists() and file_hash(dest) == obj:
                    path.unlink()
                else:
                    os.replace(path, dest)
                objects[obj] = size
                if sum(objects.values()) > self.config.budget_bytes:
                    raise _Oversized('checkpoint exceeds disk budget')
                return obj
            try:
                manifest = writer(put)
                if sum(objects.values()) > self.config.budget_bytes:
                    self.metrics['oversized'] = self.metrics.get('oversized', 0) + 1
                    return
                fd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                encoded_tokens, encoded_manifest = packed(tokens), packed(manifest)
                db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?)',
                           (key, self.namespace, encoded_tokens, encoded_manifest, time.time(),
                            digest(encoded_tokens + encoded_manifest)))
                for obj, size in objects.items():
                    db.execute('INSERT OR IGNORE INTO objects VALUES (?,?)', (obj, size))
                    db.execute('INSERT INTO refs VALUES (?,?)', (key, obj))
                db.execute('UPDATE revision SET value=value+1')
                self.index.insert(tokens, key)
                self._evict(db)
                self._version = db.execute('SELECT value FROM revision').fetchone()[0]
                self.metrics['writes'] = self.metrics.get('writes', 0) + 1
                self.metrics['write_s'] = self.metrics.get('write_s', 0) + time.perf_counter() - started
                self.metrics['stored_bytes'] = self._bytes(db)
            except _Oversized:
                self.metrics['oversized'] = self.metrics.get('oversized', 0) + 1
            finally:
                self._cleanup(db)

    def tokenize(self, identity, encode):
        key = digest(packed([self.namespace, identity]))
        started = time.perf_counter()
        with self.locked(enforce=True) as db:
            row = db.execute('SELECT value,checksum FROM tokenizations WHERE id=?', (key,)).fetchone()
            if row and digest(row[0]) != row[1]:
                db.execute('DELETE FROM tokenizations WHERE id=?', (key,))
                row = None
            if row:
                ids = json.loads(row[0])
                db.execute('UPDATE tokenizations SET used=? WHERE id=?', (time.time(), key))
            else:
                ids = list(encode())
                value = packed(ids)
                if len(value) <= self.config.token_budget_bytes:
                    db.execute('INSERT OR REPLACE INTO tokenizations VALUES (?,?,?,?)', (key, value, time.time(), digest(value)))
            while db.execute('SELECT coalesce(sum(length(value)),0) FROM tokenizations').fetchone()[0] > self.config.token_budget_bytes:
                db.execute('DELETE FROM tokenizations WHERE id=(SELECT id FROM tokenizations ORDER BY used LIMIT 1)')
        self.metrics['tokenization_s'] = time.perf_counter() - started
        return ids

    def inspect(self):
        with self.locked() as db:
            return dict(checkpoints=db.execute('SELECT count(*) FROM checkpoints').fetchone()[0],
                        stored_bytes=self._bytes(db), budget_bytes=self.config.budget_bytes,
                        metadata_bytes=(self.root / 'metadata.sqlite').stat().st_size,
                        tokenization_bytes=db.execute('SELECT coalesce(sum(length(value)),0) FROM tokenizations').fetchone()[0])

    def clear(self):
        with self.locked() as db:
            db.execute('DELETE FROM checkpoints')
            db.execute('DELETE FROM refs')
            db.execute('DELETE FROM tokenizations')
            db.execute('UPDATE revision SET value=value+1')
            self._cleanup(db)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def engine_namespace(engine, store):
    """Hash artifacts once; reuse hashes only when full stat identity matches."""
    from dataclasses import asdict
    from importlib.metadata import version
    paths = set(p for p in Path(engine.dir).rglob('*') if p.is_file()
                and store.root not in p.resolve().parents
                and p.suffix in {'.json', '.jinja', '.safetensors', '.model', '.txt'})
    for value in (engine.cfg.lora, getattr(engine.cfg.prerouter, 'weights_file', '')):
        if value:
            paths.add(Path(value))
    hashes = []
    with store.locked(enforce=True) as db:
        for path in sorted(paths):
            path = path.resolve()
            st = path.stat()
            identity = packed([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]).decode()
            row = db.execute('SELECT identity,hash FROM fingerprints WHERE path=?', (str(path),)).fetchone()
            h = row[1] if row and row[0] == identity else file_hash(path)
            db.execute('INSERT OR REPLACE INTO fingerprints VALUES (?,?,?)', (str(path), identity, h))
            hashes.append([str(path), h])
    cfg = asdict(engine.cfg)
    cfg.pop('conversation_cache', None)
    # Include implementation bytes, templates, custom tokenizer identity and all
    # family environment switches conservatively (performance switches included).
    sources = [(str(p.relative_to(Path(__file__).parents[1])), file_hash(p))
               for p in sorted(Path(__file__).parents[1].rglob('*.py'))]
    tokenizer = engine._tok
    backend_tokenizer = getattr(tokenizer, 'backend_tokenizer', None)
    tokenizer_hash = None
    if backend_tokenizer is not None and hasattr(backend_tokenizer, 'to_str'):
        tokenizer_hash = digest(backend_tokenizer.to_str().encode())
    elif tokenizer is not None and hasattr(tokenizer, 'get_vocab'):
        tokenizer_hash = digest(packed(tokenizer.get_vocab()))
    return digest(packed(dict(format=2, tokenizer_hash=tokenizer_hash, artifacts=hashes, config=cfg, interval=store.config.interval, sources=sources,
        tokenizer=repr(type(engine._tok)), template=getattr(engine._tok, 'chat_template', None),
        versions={p: version(p) for p in ('mlx', 'mlx-lm', 'tokenizers')},
        environment={k: v for k, v in os.environ.items() if k.startswith(('LING_', 'QWEN_', 'EDGE0_', 'MLX_'))})))
