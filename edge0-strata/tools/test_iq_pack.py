"""Focused pack compatibility tests; run .venv/bin/python -m unittest discover -s tools -p test_iq_pack.py."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

from _paths import add_gguf_py
import iq_pack                         # needs numpy + gguf_reader only, not gguf-py: import it unguarded

try:
    add_gguf_py()                       # llama.cpp's gguf-py: setup.py clones it; gguf_py() exits when absent
    from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
except SystemExit:
    GGUF_PY = None
else:
    GGUF_PY = "found"


def require_gguf_py(test):
    """setUpClass for the classes that build GGUF fixtures: skip (not fail) on a checkout without llama.cpp."""
    @classmethod
    def setUpClass(cls):
        if GGUF_PY is None:
            raise unittest.SkipTest("llama.cpp's gguf-py not found: run setup.py (it clones llama.cpp into "
                                    "third_party/) or set STRATA_GGUF_PY")
    return setUpClass


def write_gguf(path, tensors, split=None, arch=True):
    """`split` = (no, count, tensors): llama.cpp's gguf-split keys (0-based split.no)."""
    writer = GGUFWriter(path, "qwen4exp")
    if not arch:                                   # a later shard of a split: no model metadata
        writer.kv_data[0].pop("general.architecture", None)
    if split is not None:
        writer.add_uint16("split.no", split[0])
        writer.add_uint16("split.count", split[1])
        writer.add_int32("split.tensors.count", split[2])
    for name, values, kind in tensors:
        writer.add_tensor(name, quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class CompatibilityTests(unittest.TestCase):
    setUpClass = require_gguf_py(unittest.TestCase)
    def test_bf16_halfway_rounds_to_even(self):
        values = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32)
        got = np.frombuffer(iq_pack.bf16_bytes(values.view(np.uint8), "F32"), dtype=np.uint16)
        np.testing.assert_array_equal(got, [0x3F80, 0x3F82, 0xBF80, 0xBF82])

    def test_nonfinite_conversion_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            iq_pack.bf16_bytes(np.array([np.nan], dtype=np.float32).view(np.uint8), "F32")

    def test_projection_conversion_and_native_bytes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
            router = np.full((2, 128), 0.10001, dtype=np.float32)
            names = ["blk.0.hc_attn_down.weight", "output_hc_up.weight",
                     "blk.0.ssm_alpha.weight", "blk.3.indexer.q_proj.weight", "blk.1.ple_value.weight"]
            write_gguf(source, [(n, values, Q.Q8_0) for n in names] + [
                ("blk.1.ple_key.weight", values, Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", router, Q.F32),
                ("blk.0.attn_qkv.weight", values, Q.Q8_0),
                ("per_layer_token_embd.weight", values, Q.Q8_0),
            ])
            before = source.read_bytes()
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model, True), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            # A scalar oracle independent of the BF16 encoder: choose the nearest BF16 value,
            # using the even representation at exact ties.
            decoded = quants.dequantize(quants.quantize(values, Q.Q8_0), Q.Q8_0).ravel()
            def rounded(x):
                bits = int(np.float32(x).view(np.uint32))
                return (bits >> 16) + ((bits & 65535) > 32768 or
                                      ((bits & 65535) == 32768 and (bits >> 16) & 1))
            expected = np.array([rounded(x) for x in decoded], dtype=np.uint16)
            for name in names:
                row = rows[name]
                self.assertEqual(row[2], "4")
                offset, size = int(row[3]), int(row[4])
                np.testing.assert_array_equal(np.frombuffer(dense[offset:offset + size], dtype=np.uint16), expected)
            for name in ["blk.0.attn_qkv.weight", "blk.1.ple_key.weight"]:   # a Q8_0 PLE key stays native
                native = rows[name]
                self.assertEqual(native[2], "0")
                self.assertEqual(native[4], "0")
                self.assertEqual(native[9], "8")
            self.assertNotIn("per_layer_token_embd.weight", rows)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(json.loads((root / "compat-bf16.json").read_text())["tensors"]), 6)
            conv = json.loads((root / "conversions.json").read_text())
            self.assertEqual(sorted(r["name"] for r in conv["tensors"]), sorted(names + ["blk.0.ffn_gate_inp.weight"]))

    def test_default_still_refuses_inexact_f32_router(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.ffn_gate_inp.weight", np.full((2, 32), 0.10001, np.float32), Q.F32)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)

    def test_existing_bf16_pack_remains_byte_identical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            values = np.linspace(-2, 2, 256, dtype=np.float32).reshape(2, 128)
            write_gguf(source, [("blk.0.hc_attn_down.weight", values, Q.BF16)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 0)
            self.assertEqual((root / "dense.bin").read_bytes(), quants.quantize(values, Q.BF16).tobytes())

    def test_quantized_control_weights_need_explicit_conversion(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = root / "model.gguf"
            write_gguf(source, [("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0)])
            (root / "dense.bin").write_bytes(b"previous pack")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 1)
            self.assertEqual((root / "dense.bin").read_bytes(), b"previous pack")

    def test_q2_ple_and_large_tensors_stay_native(self):
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q2_0"))
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q8_0"))
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "IQ3_XXS"))
        self.assertFalse(iq_pack.needs_bf16("blk.1.ple_key.weight", "IQ4_XS"))
        self.assertTrue(iq_pack.needs_bf16("blk.1.ple_key.weight", "Q6_K"))
        for name in ["blk.0.ffn_gate_exps.weight", "token_embd.weight", "output.weight",
                     "per_layer_token_embd.weight", "blk.0.attn_qkv.weight"]:
            self.assertFalse(iq_pack.needs_bf16(name, "IQ3_XXS"))

    def test_iq_ple_keys_are_indexed_without_conversion(self):
        for kind, block_bytes in [(Q.IQ3_XXS, 98), (Q.IQ4_XS, 136)]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                root = Path(tmp)
                source = root / "model.gguf"
                writer = GGUFWriter(source, "qwen4exp")
                writer.add_tensor("blk.1.ple_key.weight", np.zeros((2, block_bytes), dtype=np.uint8),
                                  raw_dtype=kind)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                before = source.read_bytes()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(iq_pack.index_standalone(source, root, iq_pack.Model(source)), 0)
                _, rows = iq_pack.read_index(root / "index.txt")
                row = rows["blk.1.ple_key.weight"]
                self.assertEqual(row[2], "0")
                self.assertEqual((root / "dense.bin").read_bytes(), b"")
                self.assertEqual(json.loads((root / "conversions.json").read_text())["tensors"], [])
                self.assertEqual(source.read_bytes(), before)

    def test_experts_bin_is_kept_only_with_this_ggufs_blobs(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()

            def model(name, scale):
                path = root / name
                write_gguf(path, [("blk.%d.hc_attn_down.weight" % l, np.ones((2, 32), np.float32), Q.BF16)
                                  for l in range(3)] +
                                 [("blk.%d.ffn_gate_inp.weight" % l, np.ones((4, 32), np.float32), Q.BF16)
                                  for l in range(3)] +
                                 [("blk.%d.ffn_%s_exps.weight" % (l, r),
                                   scale * np.arange(4 * 2 * 32, dtype=np.float32).reshape(4, 2, 32) / 256, Q.Q8_0)
                                  for l in range(3) for r in ("gate", "up", "down")])
                return path

            def pack(path):
                log = io.StringIO()
                argv = ["iq_pack.py", "--gguf", str(path), "--out", str(out), "--experts-bin"]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(log):
                    self.assertEqual(iq_pack.main(), 0)
                return log.getvalue()

            first = model("a.gguf", 1)
            pack(first)
            written = (out / "experts.bin").read_bytes()
            self.assertIn("not rewritten", pack(first))
            self.assertEqual((out / "experts.bin").read_bytes(), written)
            # the same geometry with other weights under the same file name: the size and the sidecar (shard names
            # and sizes, native_experts.txt) match, the contents do not
            second = model("a.gguf", -1)
            self.assertIn("not this GGUF's expert blobs", pack(second))
            rewritten = (out / "experts.bin").read_bytes()
            self.assertEqual(len(rewritten), len(written))
            self.assertNotEqual(rewritten, written)
            self.assertIn("not rewritten", pack(second))

    def test_snapshot_symlinks_keep_split_discovery(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snapshot"
            snap.mkdir()
            first = snap / "model-00001-of-00002.gguf"
            second = snap / "model-00002-of-00002.gguf"
            write_gguf(root / "blob1", [
                ("blk.0.hc_attn_down.weight", np.ones((2, 32), np.float32), Q.Q8_0),
                ("blk.0.ffn_gate_inp.weight", np.ones((512, 32), np.float32), Q.BF16),
            ], split=(0, 2, 5))
            write_gguf(root / "blob2", [(f"blk.0.ffn_{r}_exps.weight", np.ones((512, 2, 32), np.float32), Q.Q8_0)
                                       for r in ("gate", "up", "down")], split=(1, 2, 5), arch=False)
            try:
                first.symlink_to(root / "blob1")
            except OSError:                              # Windows without Developer Mode or admin rights
                self.skipTest("symlinks are not available here")
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards"):
                iq_pack.Model(first)
            second.symlink_to(root / "blob2")
            out = root / "pack"
            (out / "tokenizer").mkdir(parents=True)
            for name in ["vocab.json", "chat_template.jinja"]:
                (out / "tokenizer" / name).touch()
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--compat-bf16"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(iq_pack.main(), 0)
            expert_index = (out / "native_experts.txt").read_text()
            self.assertIn(second.name, expert_index)
            self.assertIn("n_expert 512,", expert_index)
            (root / "blob2").write_bytes((root / "blob2").read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(first)



def bits(u32):
    return np.array(u32, dtype=np.uint32).view(np.float32)


def pack_one(root, tensors, *flags):
    """A one-file model of `tensors` packed into `root` (index only): (rc, printed, rows, dense, conversions)."""
    source = root / "model.gguf"
    write_gguf(source, tensors)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = iq_pack.index_standalone(source, root, iq_pack.Model(source), *flags)
    if rc:
        return rc, buf.getvalue(), None, None, None
    _, rows = iq_pack.read_index(root / "index.txt")
    conv = json.loads((root / "conversions.json").read_text())
    return rc, buf.getvalue(), rows, (root / "dense.bin").read_bytes(), conv


def stored(rows, dense, name, dtype):
    row = rows[name]
    return np.frombuffer(dense[int(row[3]):int(row[3]) + int(row[4])], dtype=dtype)


class ConversionTests(unittest.TestCase):
    """CS2: FORM decides the engine's form; exact F32->BF16 by default, rounding only with --compat-bf16, the PLE
    conv F32 -> F16 by the loader (index kind 3), non-finite values refused, every conversion in conversions.json."""

    setUpClass = require_gguf_py(unittest.TestCase)

    def test_exact_f32_to_bf16_by_default(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            x = bits(np.arange(64, dtype=np.uint32).reshape(2, 32) << 16 | 0x3F000000)   # BF16 values held as F32
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.0.hc_attn_inject.weight", x, Q.F32),
                                                              ("blk.0.ssm_beta.weight", x, Q.F32),
                                                              ("blk.0.ssm_norm.weight", x, Q.F32)])
            self.assertEqual(rc, 0, log)
            for name in ["blk.0.hc_attn_inject.weight", "blk.0.ssm_beta.weight"]:
                self.assertEqual(rows[name][2], "4")
                np.testing.assert_array_equal(stored(rows, dense, name, "<u2"), (x.view(np.uint32) >> 16).ravel())
            self.assertEqual(rows["blk.0.ssm_norm.weight"][2], "2")               # F32 form: as stored
            recs = {r["name"]: r for r in conv["tensors"]}
            self.assertEqual(set(recs), {"blk.0.hc_attn_inject.weight", "blk.0.ssm_beta.weight"})
            r = recs["blk.0.ssm_beta.weight"]
            self.assertEqual((r["src_type"], r["dst_type"], r["exact"], r["max_abs_err"], r["shard"]),
                             ("F32", "BF16", True, 0.0, "model.gguf"))
            self.assertEqual(r["source_sha256"], hashlib.sha256(x.tobytes()).hexdigest())
            self.assertEqual(conv["schema"], 1)
            self.assertEqual(conv["source_shards"], [{"name": "model.gguf",
                                                      "size": (Path(tmp) / "model.gguf").stat().st_size}])

    def test_inexact_f32_rounds_only_with_compat(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32) + np.float32(1e-4)
            rc, log, *_ = pack_one(Path(tmp), [("blk.0.hc_ffn_inject.weight", x, Q.F32)])
            self.assertEqual(rc, 1)
            self.assertIn("not BF16", log)
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.0.hc_ffn_inject.weight", x, Q.F32)], True)
            self.assertEqual(rc, 0, log)
            got = stored(rows, dense, "blk.0.hc_ffn_inject.weight", "<u2")
            np.testing.assert_array_equal(got, np.frombuffer(iq_pack.bf16_bytes(x.view(np.uint8).ravel(), "F32"),
                                                             dtype="<u2"))
            r = conv["tensors"][0]
            back = (got.astype(np.uint32) << 16).view(np.float32)
            self.assertFalse(r["exact"])
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(back - x.ravel()))))
            self.assertLessEqual(r["max_abs_err"], 2.0 ** -8)
            self.assertIn("--compat-bf16", r["method"])

    def test_non_finite_is_refused(self):
        for bad in (0x7FC00000, 0x7F800000, 0xFF800000):          # NaN and +-Inf: their low 16 bits are zero too
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                x = bits([[0x3F800000] * 31 + [bad]] * 2)
                (Path(tmp) / "dense.bin").write_bytes(b"previous pack")
                for flags in ((), (True,)):
                    rc, log, *_ = pack_one(Path(tmp), [("blk.0.ssm_alpha.weight", x, Q.F32)], *flags)
                    self.assertEqual(rc, 1, (hex(bad), flags))
                    self.assertIn("non-finite", log)
                self.assertEqual((Path(tmp) / "dense.bin").read_bytes(), b"previous pack")
                self.assertFalse((Path(tmp) / "dense.bin.tmp").exists())
                rc, log, *_ = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", x, Q.F32)])
                self.assertEqual(rc, 1)
                self.assertIn("non-finite", log)

    def test_ple_conv1d_f32_is_narrowed_by_the_loader(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            conv1d = np.random.default_rng(3).normal(0, 0.0062, (32, 4)).astype(np.float32)
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d, Q.F32)])
            self.assertEqual(rc, 0, log)                                          # no --compat-bf16 needed
            row = rows["blk.1.ple_conv1d.weight"]
            self.assertEqual((row[2], row[4], row[6]), ("3", str(conv1d.nbytes), str(conv1d.nbytes // 2)))
            np.testing.assert_array_equal(stored(rows, dense, "blk.1.ple_conv1d.weight", "<f4"), conv1d.ravel())
            r = conv["tensors"][0]
            self.assertEqual((r["src_type"], r["dst_type"]), ("F32", "F16"))
            self.assertIn("index kind 3", r["method"])
            h = conv1d.astype(np.float16).astype(np.float32)
            self.assertEqual(r["exact"], bool(np.array_equal(h, conv1d)))
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(h - conv1d))))
            self.assertIn("narrowed to F16 by the loader", log)
            # an F16 conv (the GSQ-RCO files) stays as it is: index kind 5, nothing recorded
            rc, log, rows, dense, conv = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d, Q.F16)])
            self.assertEqual((rows["blk.1.ple_conv1d.weight"][2], conv["tensors"]), ("5", []))
            # beyond F16's range: refused, not saturated
            rc, log, *_ = pack_one(Path(tmp), [("blk.1.ple_conv1d.weight", conv1d * 1e7, Q.F32)])
            self.assertEqual(rc, 1)
            self.assertIn("beyond F16", log)

    def test_q8_0_projections_need_compat_and_are_recorded(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            v = np.random.default_rng(4).normal(0, 0.05, (4, 64)).astype(np.float32)
            tensors = [("blk.1.ple_value.weight", v, Q.Q8_0), ("blk.2.hc_attn_up.weight", v, Q.Q8_0),
                       ("blk.1.ple_key.weight", v, Q.Q8_0)]
            rc, log, *_ = pack_one(Path(tmp), tensors)
            self.assertEqual(rc, 1)
            self.assertIn("use --compat-bf16", log)
            self.assertNotIn("ple_key", log)                                      # the Q8_0 key stays native
            rc, log, rows, dense, conv = pack_one(Path(tmp), tensors, True)
            self.assertEqual(rc, 0, log)
            self.assertEqual(rows["blk.1.ple_key.weight"][2], "0")
            recs = {r["name"]: r for r in conv["tensors"]}
            self.assertEqual(set(recs), {"blk.1.ple_value.weight", "blk.2.hc_attn_up.weight"})
            deq = quants.dequantize(quants.quantize(v, Q.Q8_0), Q.Q8_0).ravel()
            got = (stored(rows, dense, "blk.1.ple_value.weight", "<u2").astype(np.uint32) << 16).view(np.float32)
            r = recs["blk.1.ple_value.weight"]
            self.assertEqual((r["src_type"], r["dst_type"]), ("Q8_0", "BF16"))
            self.assertAlmostEqual(r["max_abs_err"], float(np.max(np.abs(got - deq))))
            self.assertEqual(conv["compat_bf16"], True)


# ---- split artifacts: Unsloth's UD-Q4_K_XL in miniature.  Shard 1 holds the metadata and no tensor; shard 2 the
# routers, layer 0 and layer 1's down; shard 3 layer 1's gate and up (a shard boundary inside a layer, like its
# layer 11); shard 4 one float tensor.  4 experts of 32 x 32, Q8_0 and F32.
N_EXPERT = 4


def expert(seed, shape):
    return np.random.default_rng(seed).standard_normal((N_EXPERT, *shape)).astype(np.float32)


def split_model(root, name="model", split_layer=True, dup=False, split_no=None, gate1=None, seed=0):
    """Writes the 4 shards; returns their paths."""
    gate1 = Q.Q8_0 if gate1 is None else gate1      # not a default arg: the module imports without gguf-py
    gu, dn = (32, 32), (32, 32)
    router = [(f"blk.{l}.ffn_gate_inp.weight", np.full((N_EXPERT, 32), 0.5, np.float32), Q.F32) for l in (0, 1)]
    l0 = [(f"blk.0.ffn_{r}_exps.weight", expert(seed + 10 + i, gu if r != "down" else dn), Q.Q8_0)
          for i, r in enumerate(("gate", "up", "down"))]
    l1_gu = [("blk.1.ffn_gate_exps.weight", expert(seed + 20, gu), gate1),
             ("blk.1.ffn_up_exps.weight", expert(seed + 21, gu), Q.Q8_0)]
    l1_d = [("blk.1.ffn_down_exps.weight", expert(seed + 22, dn), Q.F32)]
    shards = [[], router + l0 + (l1_d if split_layer else []), l1_gu + ([] if split_layer else l1_d),
              [("output_norm.weight", np.ones(32, np.float32), Q.F32)]]
    if dup:
        shards[3].append(l1_gu[1])
    total = sum(len(s) for s in shards)
    paths = []
    for i, tensors in enumerate(shards):
        p = root / f"{name}-{i + 1:05d}-of-00004.gguf"
        no = split_no.get(i, i) if split_no else i
        write_gguf(p, tensors, split=(no, 4, total), arch=i == 0)
        paths.append(p)
    return paths


def run_pack(first, out, *extra):
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for n in ["vocab.json", "chat_template.jinja"]:
        (out / "tokenizer" / n).touch()
    buf = io.StringIO()
    with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), *extra]):
        with contextlib.redirect_stdout(buf):
            rc = iq_pack.main()
    return rc, buf.getvalue()


def expected_experts(model):
    """experts.bin from first principles: per layer, per expert, gate | up | down slices."""
    parts = []
    for l in (0, 1):
        ts = [model.bytes(f"blk.{l}.ffn_{r}_exps.weight").reshape(N_EXPERT, -1) for r in ("gate", "up", "down")]
        parts.append(np.concatenate(ts, axis=1).tobytes())
    return b"".join(parts)


class SplitArtifactTests(unittest.TestCase):
    setUpClass = require_gguf_py(unittest.TestCase)
    def test_four_shards_and_a_layer_split_per_role(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            rc, log = run_pack(paths[0], root / "pack", "--experts-bin")
            self.assertEqual(rc, 0, log)
            text = (root / "pack" / "native_experts.txt").read_text()
            head, line0, line1 = text.splitlines()
            self.assertTrue(head.startswith("# strata native experts v4:"), head)
            self.assertIn("(n_expert 4,", head)
            s2, s3 = paths[1].name, paths[2].name
            self.assertEqual(line0.split()[8:], [s2])
            self.assertEqual(line1.split()[8:], [f"{s3},{s3},{s2}"])
            # every role's offset is absolute in ITS shard: shard 3's data_start for gate/up, shard 2's for down
            m = iq_pack.Model(paths[0])
            for line, l in ((line0, 0), (line1, 1)):
                f = line.split()
                for r, role in enumerate(("gate", "up", "down")):
                    g, t, _, _ = m.where[f"blk.{l}.ffn_{role}_exps.weight"]
                    self.assertEqual(int(f[5 + r]), g.data_start + t.offset)
            self.assertNotEqual(m.where["blk.1.ffn_gate_exps.weight"][0].data_start,
                                m.where["blk.1.ffn_down_exps.weight"][0].data_start)
            self.assertEqual((root / "pack" / "experts.bin").read_bytes(), expected_experts(m))
            side = json.loads((root / "pack" / "experts.bin.src.json").read_text())
            self.assertEqual([s["name"] for s in side["shards"]], [p.name for p in paths])
            self.assertFalse(list((root / "pack").glob("*.tmp")))
            # the index: routers as BF16, nothing of the experts
            _, rows = iq_pack.read_index(root / "pack" / "index.txt")
            self.assertNotIn("blk.1.ffn_gate_exps.weight", rows)
            self.assertEqual(rows["blk.1.ffn_gate_inp.weight"][2], "4")

    def test_unsplit_layers_keep_the_v3_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, split_layer=False)
            rc, log = run_pack(paths[0], root / "pack")
            self.assertEqual(rc, 0, log)
            head, line0, line1 = (root / "pack" / "native_experts.txt").read_text().splitlines()
            self.assertTrue(head.startswith("# strata native experts v3: layer gu_type d_type offset blob_bytes "
                                            "gate_off up_off down_off [shard] (n_expert 4, total "), head)
            self.assertEqual(line1.split()[8:], [paths[2].name])

    def test_missing_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[3].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing model shards.*00004-of-00004"):
                iq_pack.Model(paths[0])

    def test_duplicate_tensor_across_shards(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), dup=True)
            with self.assertRaisesRegex(ValueError, "duplicate tensor blk.1.ffn_up_exps.weight"):
                iq_pack.Model(paths[0])

    def test_shard_of_another_split(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp), split_no={2: 1})
            with self.assertRaisesRegex(ValueError, "does not declare itself shard 3 of 4"):
                iq_pack.Model(paths[0])

    def test_truncated_shard(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            paths = split_model(Path(tmp))
            paths[2].write_bytes(paths[2].read_bytes()[:-64])
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                iq_pack.Model(paths[0])

    def test_wrong_type_or_shape_leaves_the_pack_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root, gate1=Q.F32)            # layer 1: gate F32, up Q8_0
            pack = root / "pack"
            pack.mkdir()
            (pack / "native_experts.txt").write_text("previous pack")
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("layer 1: gate and up differ in type", log)
            self.assertEqual((pack / "native_experts.txt").read_text(), "previous pack")
            self.assertFalse((pack / "dense.bin").exists())

    def test_interrupted_pack(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("stopped part-way")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack, "--experts-bin")
            self.assertFalse((pack / "experts.bin").exists())
            self.assertFalse((pack / "experts.bin.src.json").exists())
            # the index stops part-way too: the old pack's files stay as they were
            before = {n: (pack / n).read_bytes() for n in ("index.txt", "dense.bin", "native_experts.txt")}
            with patch.object(iq_pack, "write_index", side_effect=KeyboardInterrupt("stopped")):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack)
            self.assertEqual(before, {n: (pack / n).read_bytes() for n in before})
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), expected_experts(iq_pack.Model(paths[0])))

    def test_interrupted_pack_continues_at_the_next_layer(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("killed")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack, "--experts-bin")
            want = expected_experts(iq_pack.Model(paths[0]))
            lines = [ln.split() for ln in (pack / "native_experts.txt").read_text().splitlines() if ln and not ln.startswith("#")]
            one_layer = int(lines[1][3])                                                  # where layer 1 starts
            self.assertFalse((pack / "experts.bin").exists())
            self.assertEqual((pack / "experts.bin.tmp").stat().st_size, one_layer)       # layer 0 is on disk
            self.assertEqual(json.loads((pack / "experts.bin.tmp.json").read_text())["layers"], 1)
            # a partial write after the recorded layer (the kill came mid-write) is cut off
            with open(pack / "experts.bin.tmp", "ab") as f:
                f.write(b"garbage")
            written = []
            real_layer = iq_pack.np.concatenate

            def counting(parts, axis):
                written.append(1)
                return real_layer(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", counting):
                rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertIn("keeping its first 1 of 2 layers and continuing with layer 1", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), want)
            self.assertEqual(len(written), 2)                  # one layer re-read to check the kept one, one layer written
            self.assertFalse((pack / "experts.bin.tmp").exists())
            self.assertFalse((pack / "experts.bin.tmp.json").exists())
            self.assertEqual(json.loads((pack / "experts.bin.src.json").read_text())["bytes"], len(want))

    def test_interrupted_pack_of_another_model_is_not_continued(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            other = split_model(root / "a", name="other", seed=100)
            paths = split_model(root / "b")
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("killed")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(other[0], pack, "--experts-bin")
            rc, log = run_pack(paths[0], pack, "--experts-bin")          # the same geometry, other weights
            self.assertEqual(rc, 0, log)
            self.assertIn("cannot be continued: starting it again", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), expected_experts(iq_pack.Model(paths[0])))

    def test_a_kept_layer_that_no_longer_matches_is_not_trusted(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            paths = split_model(root)
            pack = root / "pack"
            real = np.concatenate
            calls = []

            def dies_on_layer_1(parts, axis):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt("killed")
                return real(parts, axis=axis)
            with patch.object(iq_pack.np, "concatenate", dies_on_layer_1):
                with self.assertRaises(KeyboardInterrupt):
                    run_pack(paths[0], pack, "--experts-bin")
            data = bytearray((pack / "experts.bin.tmp").read_bytes())
            data[0] ^= 0xFF                                               # the disk lost a bit
            (pack / "experts.bin.tmp").write_bytes(bytes(data))
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertIn("cannot be continued", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), expected_experts(iq_pack.Model(paths[0])))

    def test_stale_same_size_experts_bin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            other = split_model(root / "a", name="other", seed=100)
            paths = split_model(root / "b")
            pack = root / "pack"
            rc, log = run_pack(other[0], pack, "--experts-bin")    # another model, the same geometry
            self.assertEqual(rc, 0, log)
            stale = (pack / "experts.bin").read_bytes()
            want = expected_experts(iq_pack.Model(paths[0]))
            self.assertEqual(len(stale), len(want))
            self.assertNotEqual(stale, want)
            # without --experts-bin the engine would read it: refused, nothing rewritten
            rc, log = run_pack(paths[0], pack)
            self.assertEqual(rc, 1)
            self.assertIn("another source", log)
            self.assertEqual((pack / "experts.bin").read_bytes(), stale)
            # with it: rewritten, not reused by its size
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual(rc, 0, log)
            self.assertEqual((pack / "experts.bin").read_bytes(), want)
            # and now reused
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertIn("not rewritten", log)
            # an experts.bin without its sidecar (an older packer) is rewritten too
            (pack / "experts.bin.src.json").unlink()
            (pack / "experts.bin").write_bytes(stale)
            rc, log = run_pack(paths[0], pack, "--experts-bin")
            self.assertEqual((pack / "experts.bin").read_bytes(), want)


class HadamardTests(unittest.TestCase):
    """prism.hadamard (Agention's APR files) -> hadamard.txt; anything the engine cannot rotate is refused."""
    N_EMBD, N_FF = 256, 128

    def meta(self, **kw):
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for l in range(2) for r in ("gate", "up", "down")]
        m = {"prism.hadamard.version": 1, "prism.hadamard.block_size": 128,
             "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
             "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
             "prism.hadamard.weight_names": names,
             "prism.hadamard.sign_widths": [self.N_FF, self.N_EMBD],
             "prism.hadamard.sign_values": [1, -1] * (self.N_FF // 2) + [-1, 1] * (self.N_EMBD // 2)}
        m.update(kw)
        return m

    def layout(self, gu=145, d=146):
        return [(l, gu, d, 0, 0, None) for l in range(2)]

    def test_no_metadata(self):
        self.assertIsNone(iq_pack.hadamard_spec({}, self.layout(), self.N_EMBD, self.N_FF))

    def test_explicit_signs_written(self):
        text = iq_pack.hadamard_spec(self.meta(), self.layout(), self.N_EMBD, self.N_FF)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("# strata hadamard v1"))
        self.assertEqual(lines[1], "block 128")
        self.assertEqual(lines[2].split()[:4], ["signs", str(self.N_FF), "1", "-1"])
        self.assertEqual(len(lines[3].split()), 2 + self.N_EMBD)
        self.assertEqual(lines[4:], ["layer 0 1 1 1", "layer 1 1 1 1"])

    def test_refusals(self):
        bad = {
            "transform": self.meta(**{"prism.hadamard.transform": "dct"}),
            "dense weight": self.meta(**{"prism.hadamard.weight_names": ["blk.0.attn_q.weight"]}),
            "inverse": self.meta(**{"prism.hadamard.inverse_weight_names": ["token_embd.weight"]}),
            "gate only": self.meta(**{"prism.hadamard.weight_names": ["blk.0.ffn_gate_exps.weight"]}),
            "missing width": self.meta(**{"prism.hadamard.sign_widths": [self.N_EMBD],
                                          "prism.hadamard.sign_values": [1] * self.N_EMBD}),
            "bad sign": self.meta(**{"prism.hadamard.sign_values": [2] * (self.N_FF + self.N_EMBD)}),
            "block": self.meta(**{"prism.hadamard.block_size": 96}),
        }
        for what, m in bad.items():
            with self.subTest(what):
                got = iq_pack.hadamard_spec(m, self.layout(), self.N_EMBD, self.N_FF)
                self.assertIsInstance(got, str)
                self.assertFalse(got.startswith("# strata hadamard"))
        got = iq_pack.hadamard_spec(self.meta(), self.layout(gu=42), self.N_EMBD, self.N_FF)
        self.assertIn("Q2_0", got)


class NativeExpertsV5Tests(unittest.TestCase):
    """The v5 mark (Hadamard-folded experts) rewrites only the tables the packer writes, v3 and v4."""
    setUpClass = require_gguf_py(unittest.TestCase)   # test_packer_refuses_before_writing builds GGUF fixtures
    ROWS = "0 145 146 0 100 1 2 3\n1 145 146 100 100 4 5 6\n"

    def table(self, version, tail="(n_expert 2, total 200)"):
        return "# strata native experts v%s: layer gu_type d_type offset ... %s\n" % (version, tail) + self.ROWS

    def test_v3_and_v4_become_v5(self):
        for v in ("3", "4"):
            with self.subTest(v):
                got = iq_pack.native_experts_v5(self.table(v))
                head, rest = got.split("\n", 1)
                self.assertTrue(head.startswith("# strata native experts v5: layer gu_type"))
                self.assertTrue(head.endswith("(n_expert 2, total 200) - v5: Hadamard-folded experts, see hadamard.txt"))
                self.assertEqual(rest, self.ROWS)

    def test_other_versions_are_refused_by_name(self):
        for v in ("2", "5", "6", "10"):
            with self.subTest(v):
                got = iq_pack.native_experts_v5(self.table(v))
                self.assertFalse(got.startswith("# strata native experts"))
                self.assertIn("v%s" % v, got)
                self.assertIn("v3 and v4", got)
        got = iq_pack.native_experts_v5("# something else\n" + self.ROWS)
        self.assertIn("without a version header", got)

    def test_packer_refuses_before_writing(self):
        real = iq_pack.expert_layout

        def v6_layout(model, src):
            layout, text, n_expert, offset = real(model, src)
            return layout, text.replace("native experts v3:", "native experts v6:", 1), n_expert, offset

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = split_model(root, split_layer=False)
            out = root / "pack"
            with patch.object(iq_pack, "expert_layout", v6_layout), \
                    patch.object(iq_pack, "hadamard_spec", return_value="# strata hadamard v1\nblock 32\n"):
                rc, log = run_pack(paths[0], out)
            self.assertEqual(rc, 1)
            self.assertIn("native_experts.txt v6", log)
            self.assertIn("only v3 and v4", log)
            self.assertFalse((out / "native_experts.txt").exists())
            self.assertFalse((out / "hadamard.txt").exists())


if __name__ == "__main__":
    unittest.main()
