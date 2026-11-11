"""pack_gguf must never take an Engram file's metadata for the model's: ISTA-DASLab's final Engram says general.architecture 'deepseek41-engram'
(the first release's said 'engram').  The fixture is the real header of DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (its 416 bytes: metadata and the tensor
directory, no data)."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dsv41" / "engram-final-header.gguf"


def tiny_gguf(path: Path, arch: str | None, extra: dict | None = None, tensors: tuple = ()) -> Path:
    """A GGUF with only metadata (string and uint32 values) and tensor infos (an IQ4_NL matrix each, no data); arch None leaves out general.architecture."""
    kv = {**({} if arch is None else {"general.architecture": arch}), **(extra or {})}
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, v in kv.items():
        out += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            out += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    for name in tensors:
        out += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", 2) + struct.pack("<2Q", 32, 4) + struct.pack("<IQ", 20, 0)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out))
    return path


class EngramHeader(unittest.TestCase):
    def test_the_real_final_engram_header_is_an_engram_file(self):
        g = PG.Gguf(FIXTURE)
        self.assertEqual(g.kv["general.architecture"], "deepseek41-engram")
        self.assertTrue(PG.is_engram_file(g))
        self.assertEqual(len(g.tensors), 2)
        self.assertEqual(g.data_start, 416)

    def test_it_is_never_the_group_the_model_metadata_comes_from(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shard = PG.Gguf(tiny_gguf(d / "model-00001-of-00004.gguf", "deepseek41", {"split.count": 4}))
            engram = PG.Gguf(FIXTURE)
            for order in ([engram, shard], [shard, engram]):
                kv = PG.model_kv(order)
                self.assertIsNotNone(kv)
                self.assertEqual(kv["general.architecture"], "deepseek41")
            self.assertIsNone(PG.model_kv([engram]))                    # an Engram alone is not a model

    def test_the_first_releases_engram_is_one_too(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "e.gguf", "engram"))))
            self.assertFalse(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "m.gguf", "deepseek41"))))


CONVERTED = ROOT / "tests" / "fixtures" / "dsv41" / "engram-converted-27oct-header.gguf"
SPLIT = {"split.count": 6, "split.no": 5}
PLE = ("per_layer_token_embd.weight",)


class SixShardSplit(unittest.TestCase):
    """ISTA-DASLab's six-shard sets (llama.cpp b11870 and newer): split.count is 6, five shards are the set's own and the sixth is the Engram file,
    handed over with --engram instead of in --shards.  CONVERTED is the real header (416 bytes) of the Engram at their 2026-10-27 revision: no
    general.architecture, split.no 5 of 6, one tensor, per_layer_token_embd.weight."""

    def model_shards(self, d: Path, n: int = 5, count: int = 6) -> list:
        return [tiny_gguf(d / ("m-%05d-of-%05d.gguf" % (i, count)), "deepseek41", {"split.count": count, "split.no": i - 1}) for i in range(1, n + 1)]

    def test_the_real_converted_header_is_what_the_check_expects(self):
        g = PG.Gguf(CONVERTED)
        self.assertIsNone(g.kv.get("general.architecture"))
        self.assertFalse(PG.is_engram_file(g))                                          # why the check cannot use the architecture
        self.assertEqual((g.kv["split.no"], g.kv["split.count"]), (5, 6))
        self.assertEqual([t[0] for t in g.tensors], ["per_layer_token_embd.weight"])

    def test_the_engram_counts_as_the_sixth_shard(self):
        with tempfile.TemporaryDirectory() as d:
            shards = self.model_shards(Path(d))
            self.assertEqual(PG.engram_is_last_shard(str(CONVERTED), 6, shards), 1)
            self.assertEqual(PG.engram_is_last_shard(None, 6, shards), 0)                # no --engram: the five are five of six

    def test_the_pinned_four_shard_releases_engram_counts_too_it_has_the_same_tables(self):
        with tempfile.TemporaryDirectory() as d:
            shards = self.model_shards(Path(d))
            g = PG.Gguf(FIXTURE)
            self.assertTrue(PG.is_engram_file(g))                                       # general.architecture deepseek41-engram
            self.assertEqual(PG.engram_is_last_shard(str(FIXTURE), 6, shards), 1)       # six-shard shards with the Engram of the pinned four-shard release
            self.assertEqual(PG.engram_is_last_shard(str(FIXTURE), 6, self.model_shards(Path(d), 4)), 0)    # still five of six when a shard is missing
            self.assertEqual(PG.engram_is_last_shard(str(FIXTURE), 4, self.model_shards(Path(d), 4)), 0)    # the four-shard sets do not need it counted

    def test_it_never_counts_when_a_shard_is_missing_the_split_differs_or_the_path_is_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(PG.engram_is_last_shard(str(CONVERTED), 6, self.model_shards(d, 4)), 0)
            self.assertEqual(PG.engram_is_last_shard(str(CONVERTED), 4, self.model_shards(d, 3, 4)), 0)    # another split's file
            self.assertEqual(PG.engram_is_last_shard(str(d / "gone.gguf"), 6, self.model_shards(d)), 0)
            self.assertEqual(PG.engram_is_last_shard(str(d), 6, self.model_shards(d)), 0)                # a folder (the earlier layouts' --engram)

    def test_one_file_is_never_counted_twice(self):
        with tempfile.TemporaryDirectory() as d:
            four = self.model_shards(Path(d), 4)
            self.assertEqual(PG.engram_is_last_shard(str(CONVERTED), 6, four + [CONVERTED]), 0)         # four shards + the Engram in --shards, and --engram too
            self.assertEqual(PG.engram_is_last_shard(str(four[0]), 6, four + [four[0]]), 0)

    def test_a_file_that_is_not_the_engram_never_counts(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shards = self.model_shards(d)
            for name, arch, extra, tensors in (("no split keys", None, {}, PLE),
                                               ("another number", None, {"split.count": 6, "split.no": 2}, PLE),
                                               ("another tensor", None, SPLIT, ("blk.0.attn_norm.weight",)),
                                               ("two tensors", None, SPLIT, PLE + ("blk.0.attn_norm.weight",)),
                                               ("no tensor", "deepseek41", SPLIT, ())):
                f = tiny_gguf(d / "x.gguf", arch, extra, tensors)
                self.assertEqual(PG.engram_is_last_shard(str(f), 6, shards), 0, name)
            self.assertEqual(PG.engram_is_last_shard(str(tiny_gguf(d / "ok.gguf", None, SPLIT, PLE)), 6, shards), 1)      # the same file as a real one

    def run_main(self, d: Path, engram: Path):
        shards = self.model_shards(d)
        argv = ["pack_gguf.py", "--model", str(d), "--shards", *[p.name for p in shards], "--engram", str(engram), "--out", str(d / "pack")]
        with unittest.mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as cm:
                PG.main()
        return str(cm.exception.code)

    def test_main_says_how_many_shards_are_here_when_the_sixth_is_not_the_engram(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wrong = tiny_gguf(d / "not-the-engram.gguf", "deepseek41", SPLIT, ("blk.0.attn_norm.weight",))
            self.assertIn("5 of the 6 shards are here", self.run_main(d, wrong))

    def test_main_passes_the_split_check_with_the_real_engram_header(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertNotIn("shards are here", self.run_main(d, CONVERTED))          # (it stops later: the header alone has no tensor data)


class LegacyModelMetadata(unittest.TestCase):
    def test_legacy_engram_names_write_the_same_hash_table(self):
        common = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.max_ngram_size": 2,
            "deepseek41.engram.head_count": 1,
            "deepseek41.engram.token_map": [0, 1, 2],
            "deepseek41.engram.multipliers": [3, 5],
            "deepseek41.engram.primes": [7],
            "deepseek41.engram.offsets": [0],
        }
        legacy = {**common, "deepseek41.engram.layer_ids": [1],
                  "deepseek41.engram.key_length": 256, "deepseek41.engram.pad_id": 2}
        current = {**common, "deepseek41.engram.layers": [1],
                   "deepseek41.engram.head_dim": 256, "deepseek41.engram.pad_token_id": 2,
                   "deepseek41.engram.key_length": 999, "deepseek41.engram.pad_id": 999}
        PG.int_tensors_to_engram([], legacy)
        PG.int_tensors_to_engram([], current)
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "legacy.bin", Path(d) / "current.bin"
            PG.write_engram(a, legacy, 3)
            PG.write_engram(b, current, 3)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(struct.unpack("<4s6i", a.read_bytes()[:28]),
                             (b"DSE1", 3, 1, 2, 1, 256, 2))

    def test_source_config_restores_omitted_sparse_geometry(self):
        kv = {"general.architecture": "deepseek41", "deepseek41.block_count": 40,
              "deepseek41.embedding_length": 5120, "deepseek41.expert_count": 384}
        cfg = {"text_config": {"num_hidden_layers": 40, "hidden_size": 5120,
               "n_routed_experts": 384, "kv_source_layer_ids": [2, 8, 14, 20],
               "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
               "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048,
               "candidate_block_size": 8}}
        PG.apply_source_config(kv, cfg)
        geometry = PG.geometry(kv, 129280)
        self.assertEqual(geometry["kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(geometry["index_source_layer_ids"], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(geometry["candidate_source_layer_id"], 20)
        self.assertEqual(geometry["candidate_topk_blocks"], 2048)
        self.assertEqual(geometry["candidate_block_size"], 8)
        cfg["text_config"]["candidate_block_size"] = 16
        with self.assertRaisesRegex(SystemExit, "candidate_block_size"):
            PG.apply_source_config(kv, cfg)


if __name__ == "__main__":
    unittest.main()
