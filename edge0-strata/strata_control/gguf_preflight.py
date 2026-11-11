"""Bounded, constant-memory read of the GGUF metadata needed for import preview.

This reads tensor directory entries and checks their declared byte ranges, but it
does not read tensor payloads or establish provenance.
"""

from __future__ import annotations

import os
import stat
import struct
from pathlib import Path

from strata_control.profiles import ProfileError
from tools.gguf_reader import BLOCK_GEOMETRY, GGML_TYPES


MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_METADATA_KEYS = 4096
MAX_STRING_BYTES = 1024 * 1024
MAX_ARRAY_ITEMS = 500_000
MAX_TENSORS = 100_000
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}
CAPTURE_STRINGS = {"general.architecture", "tokenizer.ggml.model", "clip.projector_type"}
PRESENCE_KEYS = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}


class _Reader:
    def __init__(self, stream, size: int):
        self.stream, self.size, self.used = stream, size, 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.used + count > min(self.size, MAX_HEADER_BYTES):
            raise ProfileError("GGUF metadata exceeds the bounded inspection limit", 5)
        data = self.stream.read(count)
        if len(data) != count:
            raise ProfileError("GGUF metadata is truncated", 5)
        self.used += count
        return data

    def skip(self, count: int) -> None:
        if count < 0 or self.used + count > min(self.size, MAX_HEADER_BYTES):
            raise ProfileError("GGUF metadata exceeds the bounded inspection limit", 5)
        self.stream.seek(count, os.SEEK_CUR)
        self.used += count

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self, *, capture: bool = False) -> str | None:
        size = self.u64()
        if size > MAX_STRING_BYTES:
            raise ProfileError("GGUF metadata string is too large to inspect", 5)
        if not capture:
            self.skip(size)
            return None
        try:
            return self.take(size).decode("utf-8")
        except UnicodeError:
            raise ProfileError("GGUF metadata has invalid UTF-8", 5) from None

    def value(self, kind: int, *, capture: bool = False) -> object:
        if kind == 8:
            return self.string(capture=capture)
        if kind == 9:
            element_type, count = self.u32(), self.u64()
            if count > MAX_ARRAY_ITEMS:
                raise ProfileError("GGUF metadata array exceeds the inspection limit", 5)
            if element_type == 8:
                for _ in range(count):
                    self.string()
            elif element_type in SCALAR_BYTES:
                self.skip(count * SCALAR_BYTES[element_type])
            else:
                raise ProfileError("GGUF metadata has an unsupported array type", 5)
            return element_type, count
        if kind not in SCALAR_BYTES:
            raise ProfileError("GGUF metadata has an unsupported value type", 5)
        if capture and kind in (4, 10):
            return struct.unpack("<I" if kind == 4 else "<Q", self.take(SCALAR_BYTES[kind]))[0]
        self.skip(SCALAR_BYTES[kind])
        return None


def real_model_file(path: Path) -> Path:
    """The regular file a model path stands for. A model file the person picked can be a link (a Hugging Face cache snapshot, an
    Ollama blob, a shortcut to a bigger drive): it is followed here, once, and the target is what gets opened. Strata's own state files
    (run configs, profiles, runtime records) keep the strict no-link rule in their readers."""
    try:
        target = Path(os.path.realpath(path))
        info = target.stat()
    except OSError:
        if os.path.islink(path):
            raise ProfileError(f"{Path(path).name} is a link to a file that is not there", 5) from None
        raise ProfileError(f"{Path(path).name} is missing", 5) from None
    if not stat.S_ISREG(info.st_mode):
        raise ProfileError("GGUF shard must be a regular file", 5)
    return target


def inspect_metadata(path: Path) -> dict:
    """Return only bounded compatibility signals from one regular GGUF v3 shard (a link to one is followed)."""
    try:
        path = real_model_file(Path(path))
        descriptor = os.open(path, os.O_RDONLY)
        with os.fdopen(descriptor, "rb") as stream:
            details = os.fstat(stream.fileno())
            if not stat.S_ISREG(details.st_mode):
                raise ProfileError("GGUF shard must be a regular file", 5)
            reader = _Reader(stream, details.st_size)
            if reader.take(4) != b"GGUF":
                raise ProfileError("file does not begin with a GGUF header", 5)
            version, tensors, keys = reader.u32(), reader.u64(), reader.u64()
            if version != 3:
                raise ProfileError("Strata metadata preflight requires GGUF v3", 5)
            # tensors == 0 is a real file: a split set made with llama.cpp's gguf-split (Unsloth's) keeps only the metadata in its first shard
            if not 0 <= tensors <= MAX_TENSORS or keys > MAX_METADATA_KEYS:
                raise ProfileError("GGUF tensor or metadata count exceeds the inspection limit", 5)
            found: dict[str, object] = {}
            for _ in range(keys):
                key = reader.string(capture=True)
                if key is None or len(key) > 256:
                    raise ProfileError("GGUF metadata key is invalid or too long", 5)
                value_type = reader.u32()
                if key in CAPTURE_STRINGS and value_type != 8:
                    raise ProfileError("GGUF architecture or tokenizer model is not a string", 5)
                value = reader.value(value_type, capture=key in CAPTURE_STRINGS or
                                     key == "general.alignment")
                if key in CAPTURE_STRINGS:
                    found[key] = value
                elif key in PRESENCE_KEYS:
                    found[key] = value_type == 9 and isinstance(value, tuple) and \
                        value[0] == 8 and value[1] > 0
                elif key == "general.alignment":
                    found[key] = value
            alignment = found.get("general.alignment", 32)
            if type(alignment) is not int or not 1 <= alignment <= 4096 or \
                    alignment & (alignment - 1):
                raise ProfileError("GGUF tensor alignment is invalid", 5)
            tensor_types: dict[str, int] = {}
            expert_types: dict[str, int] = {}
            required_payload_bytes = 0
            extents: list[tuple[int, int]] = []
            for _ in range(tensors):
                name = reader.string(capture=True)
                if not name or len(name) > 256:
                    raise ProfileError("GGUF tensor name is invalid or too long", 5)
                dimensions = reader.u32()
                if not 1 <= dimensions <= 8:
                    raise ProfileError("GGUF tensor dimension count is invalid", 5)
                elements = 1
                for _ in range(dimensions):
                    size = reader.u64()
                    if size == 0 or size > 1_000_000_000 or elements > (1 << 63) // size:
                        raise ProfileError("GGUF tensor shape is invalid", 5)
                    elements *= size
                type_id, offset = reader.u32(), reader.u64()
                type_name = GGML_TYPES.get(type_id)
                geometry = BLOCK_GEOMETRY.get(type_name) if type_name else None
                if geometry is None or elements % geometry[0]:
                    raise ProfileError("GGUF tensor has an unsupported encoding or shape", 5)
                if offset % alignment:
                    raise ProfileError("GGUF tensor offset is not aligned", 5)
                needed = offset + elements // geometry[0] * geometry[1]
                required_payload_bytes = max(required_payload_bytes, needed)
                extents.append((offset, needed))
                tensor_types[type_name] = tensor_types.get(type_name, 0) + 1
                if "_exps" in name or ".experts." in name:
                    expert_types[type_name] = expert_types.get(type_name, 0) + 1
            data_start = (reader.used + alignment - 1) // alignment * alignment
            extents.sort()
            if any(next_start < end for (_, end), (next_start, _) in zip(extents, extents[1:])):
                raise ProfileError("GGUF tensor byte ranges overlap", 5)
            if tensors and data_start + required_payload_bytes > details.st_size:
                raise ProfileError("GGUF tensor payload is shorter than its directory requires", 5)
            if tensors == 1 and details.st_size - (data_start + required_payload_bytes) >= 64:
                raise ProfileError("single-tensor GGUF shard has unexplained trailing bytes", 5)
            return {"architecture": found.get("general.architecture"),
                    "projector_type": found.get("clip.projector_type"),
                    "tokenizer_model": found.get("tokenizer.ggml.model"),
                    "tokenizer_tokens_present": found.get("tokenizer.ggml.tokens") is True,
                    "tokenizer_merges_present": found.get("tokenizer.ggml.merges") is True,
                    "tensor_count_reported": tensors, "tensor_count_scanned": tensors,
                    "tensor_types": dict(sorted(tensor_types.items())),
                    "expert_tensor_types": dict(sorted(expert_types.items())),
                    "header_bytes_scanned": reader.used,
                    "payload_bytes_required": required_payload_bytes,
                    "payload_extent_valid": True}
    except OSError:
        raise ProfileError("GGUF metadata could not be read", 5) from None
