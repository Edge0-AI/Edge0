"""Tests for windows/tools/gguf_tool.py.

Standard library plus pytest only. Models are synthetic GGUF files written by this
module, so nothing is downloaded. The registry check against the vendored engine runs
only when vendor/llama.cpp is present (it is gitignored, so CI skips it).
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import gguf_tool as gt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
VENDOR_LLAMA = REPO / "vendor" / "llama.cpp"
GIB = gt.GIB

# gguf value type codes
T_U32, T_F32, T_BOOL, T_STR, T_ARR, T_U64 = 4, 6, 7, 8, 9, 10


# ---------------------------------------------------------------- GGUF writer


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(vtype: int, value) -> bytes:
    if vtype == T_U32:
        return struct.pack("<I", value)
    if vtype == T_F32:
        return struct.pack("<f", value)
    if vtype == T_BOOL:
        return struct.pack("<?", value)
    if vtype == T_STR:
        return _s(value)
    if vtype == T_U64:
        return struct.pack("<Q", value)
    if vtype == T_ARR:
        etype, items = value
        out = struct.pack("<I", etype) + struct.pack("<Q", len(items))
        for item in items:
            out += _value(etype, item)
        return out
    raise AssertionError(f"unsupported test value type {vtype}")


def write_gguf(path: Path, kv, tensors, *, version: int = 3, alignment: int = 32) -> Path:
    """Write a GGUF file. kv: [(key, vtype, value)]; tensors: [(name, dims, ggml_type, nbytes)].

    Tensor sizes should be multiples of 64 so the offset-based size calculation is exact.
    """
    head = bytearray(b"GGUF")
    head += struct.pack("<I", version)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", len(kv))
    for key, vtype, value in kv:
        head += _s(key) + struct.pack("<I", vtype) + _value(vtype, value)
    offsets = []
    off = 0
    for _, _, _, nbytes in tensors:
        off = -(-off // alignment) * alignment
        offsets.append(off)
        off += nbytes
    for (name, dims, ttype, _), offset in zip(tensors, offsets):
        head += _s(name) + struct.pack("<I", len(dims))
        head += b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<I", ttype) + struct.pack("<Q", offset)
    while len(head) % alignment:
        head += b"\0"
    path.write_bytes(bytes(head) + bytes(off))
    return path


def moe_kv(arch: str, *, layers: int = 4, experts: int = 8, used: int = 2, chat: bool = True):
    kv = [
        ("general.architecture", T_STR, arch),
        ("general.name", T_STR, f"{arch}-test"),
        (f"{arch}.block_count", T_U32, layers),
        (f"{arch}.expert_count", T_U32, experts),
        (f"{arch}.expert_used_count", T_U32, used),
        (f"{arch}.attention.head_count", T_U32, 8),
        (f"{arch}.attention.head_count_kv", T_U32, 2),
        (f"{arch}.attention.key_length", T_U32, 128),
        (f"{arch}.attention.value_length", T_U32, 128),
        (f"{arch}.embedding_length", T_U32, 1024),
        (f"{arch}.context_length", T_U32, 32768),
    ]
    if chat:
        kv.append(("tokenizer.chat_template", T_STR, "{{ messages }}"))
    return kv


def moe_tensors(layers: int = 4, expert_type: int = 12):
    """Dense: 2048 (embd) + 2048 (output) + 1536/layer. Experts: 12288/layer."""
    t = [("token_embd.weight", [1024, 1], 0, 2048), ("output.weight", [1024, 1], 0, 2048)]
    for i in range(layers):
        t.append((f"blk.{i}.attn_q.weight", [1024, 1], 0, 1024))
        t.append((f"blk.{i}.ffn_gate_inp.weight", [8, 1], 0, 512))
        t.append((f"blk.{i}.ffn_gate_exps.weight", [1024, 8], expert_type, 4096))
        t.append((f"blk.{i}.ffn_up_exps.weight", [1024, 8], expert_type, 4096))
        t.append((f"blk.{i}.ffn_down_exps.weight", [1024, 8], expert_type, 4096))
    return t


@pytest.fixture
def moe_file(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "qwen3-moe-test-Q4_K.gguf", moe_kv("qwen3moe"), moe_tensors())


# ---------------------------------------------------------------- parsing


def test_header_round_trip(moe_file: Path):
    h = gt.read_header(str(moe_file))
    assert h["version"] == 3
    assert h["kv"]["general.architecture"] == "qwen3moe"
    assert h["kv"]["qwen3moe.expert_count"] == 8
    assert len(h["tensors"]) == 2 + 4 * 5
    assert h["data_start"] % 32 == 0


def test_tensor_sizes_match_written_sizes(moe_file: Path):
    h = gt.read_header(str(moe_file))
    sizes = gt.tensor_byte_sizes(h)
    written = [n for _, _, _, n in moe_tensors()]
    assert sizes == written


def test_version_2_is_accepted(tmp_path: Path):
    p = write_gguf(tmp_path / "v2.gguf", moe_kv("llama", chat=False), [("token_embd.weight", [64], 0, 256)], version=2)
    assert gt.read_header(str(p))["version"] == 2


def test_custom_alignment_is_honoured(tmp_path: Path):
    kv = [("general.architecture", T_STR, "llama"), ("general.alignment", T_U32, 64)]
    p = write_gguf(tmp_path / "align.gguf", kv, [("w", [64], 0, 256)], alignment=64)
    h = gt.read_header(str(p))
    assert h["data_start"] % 64 == 0


def test_non_gguf_file_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(gt.GGUFError, match="not a GGUF"):
        gt.read_header(str(bad))


def test_unsupported_version_is_rejected(tmp_path: Path):
    p = write_gguf(tmp_path / "v9.gguf", moe_kv("llama"), [("w", [64], 0, 64)], version=9)
    with pytest.raises(gt.GGUFError, match="unsupported GGUF version"):
        gt.read_header(str(p))


def test_truncated_header_is_rejected(tmp_path: Path, moe_file: Path):
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(moe_file.read_bytes()[:40])
    with pytest.raises(gt.GGUFError):
        gt.read_header(str(cut))


# ---------------------------------------------------------------- shards


def test_split_model_reads_all_shards(tmp_path: Path):
    names = [f"big-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    for i, name in enumerate(names):
        kv = moe_kv("qwen3moe") if i == 0 else []
        write_gguf(tmp_path / name, kv, [(f"blk.{i}.ffn_gate_exps.weight", [64], 12, 64)])
    paths = gt.shard_paths(str(tmp_path / names[1]))
    assert [Path(p).name for p in paths] == names
    a = gt.analyze(str(tmp_path / names[0]))
    assert a["architecture"] == "qwen3moe"
    assert a["tensor_count"] == 3
    assert a["shards"] == names
    assert "split model: 3 shards" in a["warnings"]


def test_missing_shard_is_reported(tmp_path: Path):
    write_gguf(tmp_path / "big-00001-of-00003.gguf", moe_kv("llama"), [("w", [64], 0, 64)])
    write_gguf(tmp_path / "big-00003-of-00003.gguf", [], [("w2", [64], 0, 64)])
    with pytest.raises(gt.GGUFError, match="missing GGUF shards: big-00002-of-00003.gguf"):
        gt.shard_paths(str(tmp_path / "big-00001-of-00003.gguf"))


def test_is_first_shard():
    assert gt.is_first_shard("m-00001-of-00004.gguf")
    assert not gt.is_first_shard("m-00002-of-00004.gguf")
    assert gt.is_first_shard("plain.gguf")


# ---------------------------------------------------------------- expert classification


@pytest.mark.parametrize(
    "name,expected",
    [
        ("blk.0.ffn_gate_exps.weight", True),
        ("blk.12.ffn_up_exps.weight", True),
        ("blk.3.ffn_down_exps.weight", True),
        ("blk.3.ffn_gate_up_exps.weight", True),
        ("blk.3.ffn_up_chexps.weight", True),
        ("blk.3.ffn_gate_shexp.weight", False),
        ("blk.3.ffn_gate_inp.weight", False),
        ("blk.3.attn_q.weight", False),
    ],
)
def test_expert_regex_matches_engine_override(name, expected):
    assert bool(gt.EXPERT_RE.search(name)) is expected


def test_analysis_counts_experts_per_layer(moe_file: Path):
    a = gt.analyze(str(moe_file))
    assert a["is_moe"] is True
    assert a["layers"] == 4
    assert a["experts"] == 8
    assert a["expert_bytes"] == 4 * 3 * 4096
    assert a["_expert_by_layer"] == [3 * 4096] * 4
    assert a["quant"] == "Q4_K"
    assert a["has_chat_template"] is True
    assert a["family"]["id"] == "qwen3-moe"


def test_dense_model_is_not_moe(tmp_path: Path):
    p = write_gguf(tmp_path / "dense.gguf", moe_kv("llama", experts=0), [("blk.0.ffn_up.weight", [64], 0, 64)])
    a = gt.analyze(str(p))
    assert a["is_moe"] is False
    assert a["family"] is None
    assert "dense model: expert offload does not apply" in a["warnings"]


def test_mixtral_needs_experts(tmp_path: Path):
    dense = write_gguf(tmp_path / "llama-dense.gguf", moe_kv("llama", experts=0), [("w", [64], 0, 64)])
    mixtral = write_gguf(
        tmp_path / "mixtral.gguf",
        moe_kv("llama"),
        [("blk.0.ffn_up_exps.weight", [64, 8], 12, 64)],
    )
    assert gt.analyze(str(dense))["family"] is None
    assert gt.analyze(str(mixtral))["family"]["id"] == "mixtral"


def test_unknown_moe_arch_warns(tmp_path: Path):
    p = write_gguf(tmp_path / "novel.gguf", moe_kv("made-up-moe"), [("blk.0.ffn_up_exps.weight", [64], 12, 64)])
    a = gt.analyze(str(p))
    assert a["family"] is None
    assert any("not listed" in w for w in a["warnings"])


# ---------------------------------------------------------------- family registry


REG = gt.load_registry()


@pytest.mark.parametrize(
    "arch,expert_count,family_id",
    [
        ("qwen3moe", 128, "qwen3-moe"),
        ("qwen35moe", 256, "qwen35-moe"),
        ("qwen3next", 512, "qwen3-next"),
        ("qwen3vlmoe", 128, "qwen3-vl-moe"),
        ("qwen2moe", 60, "qwen2-moe"),
        ("gpt-oss", 32, "gpt-oss"),
        ("laguna", 256, "laguna"),
        ("glm4moe", 128, "glm4-moe"),
        ("glm-dsa", 256, "glm5-dsa"),
        ("deepseek2", 256, "deepseek-v2-v3"),
        ("llama4", 16, "llama4"),
        ("minimax-m2", 256, "minimax-m2"),
        ("granitemoe", 40, "granite-moe"),
        ("bailingmoe3", 256, "ling-bailing-moe"),
    ],
)
def test_family_lookup(arch, expert_count, family_id):
    fam = gt.match_family(arch, expert_count, REG)
    assert fam is not None and fam["id"] == family_id


def test_requested_families_are_all_registered():
    ids = {f["id"] for f in REG["families"]}
    for required in ("qwen3-moe", "qwen35-moe", "gpt-oss", "laguna", "glm4-moe", "glm5-dsa"):
        assert required in ids


def test_registry_is_well_formed():
    seen_ids = set()
    seen_archs = {}
    for fam in REG["families"]:
        assert fam["id"] not in seen_ids, f"duplicate family id {fam['id']}"
        seen_ids.add(fam["id"])
        assert fam["label"] and fam["archs"], fam["id"]
        for arch in fam["archs"]:
            assert arch not in seen_archs, f"arch {arch} listed twice ({seen_archs.get(arch)}, {fam['id']})"
            seen_archs[arch] = fam["id"]
        for ex in fam["examples"]:
            assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", ex), ex


def test_arch_regex_reads_engine_table_format():
    snippet = '        { LLM_ARCH_LLAMA,            "llama"      },\n        { LLM_ARCH_GLM_DSA, "glm-dsa" },\n'
    assert gt.ARCH_RE.findall(snippet) == ["llama", "glm-dsa"]


@pytest.mark.skipif(not (VENDOR_LLAMA / "src" / "llama-arch.cpp").is_file(), reason="vendor/llama.cpp not present")
def test_every_family_arch_is_registered_at_pin():
    res = gt.registry_check(str(VENDOR_LLAMA))
    assert res["ok"], f"archs missing from the pinned engine: {res['missing']}"


# ---------------------------------------------------------------- KV estimate


def _meta(kv: dict, arch: str = "x"):
    return gt._meta_getter(kv, arch)


def test_kv_scalar_heads():
    kv = {"x.block_count": 4, "x.attention.head_count_kv": 2, "x.attention.key_length": 128, "x.attention.value_length": 128}
    per_tok, note = gt.kv_bytes_per_token(_meta(kv), 4)
    assert per_tok == 2 * 2 * 4 * (128 + 128) and note is None


def test_kv_per_layer_heads_skip_recurrent_layers():
    kv = {"x.attention.head_count_kv": [0, 2, 0, 2], "x.attention.key_length": 128, "x.attention.value_length": 128}
    per_tok, _ = gt.kv_bytes_per_token(_meta(kv), 4)
    assert per_tok == 2 * 4 * (128 + 128)


def test_kv_mla_uses_latent_size():
    kv = {"x.attention.kv_lora_rank": 512, "x.rope.dimension_count": 64}
    per_tok, note = gt.kv_bytes_per_token(_meta(kv), 61)
    assert per_tok == 2 * (512 + 64) * 61
    assert "MLA" in note


def test_kv_unknown_is_reported():
    per_tok, note = gt.kv_bytes_per_token(_meta({}), 4)
    assert per_tok == 0 and "unknown" in note


# ---------------------------------------------------------------- planning (pure arithmetic)


def _analysis(*, moe=True, layers=4, dense=1.0, per_layer=1.0, kv_tok=0):
    per = [int(per_layer * GIB)] * layers
    experts = sum(per)
    dense_b = int(dense * GIB)
    return {
        "is_moe": moe,
        "layers": layers,
        "tensor_bytes": dense_b + experts,
        "expert_bytes": experts,
        "_expert_by_layer": per,
        "kv_bytes_per_token": kv_tok,
        "kv_note": None,
    }


def test_plan_all_experts_fit_on_gpu():
    plan = gt.plan_offload(_analysis(), vram_gb=6, ram_gb=None, ctx=8192, headroom_gb=0.5)
    assert plan["fits"] and plan["mode"] == "gpu"
    assert plan["args"] == ["-ngl", "99"]


def test_plan_picks_smallest_ncmoe():
    # dense 1 GiB + 4 layers x 1 GiB experts; budget 4.5 GiB -> need 3 GiB of experts on GPU -> N=1
    plan = gt.plan_offload(_analysis(), vram_gb=4.5, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert plan["mode"] == "cpu-experts"
    assert plan["n_cpu_moe"] == 1
    assert plan["args"] == ["-ngl", "99", "--n-cpu-moe", "1"]
    assert plan["cpu_gib_est"] == 1.0 and plan["gpu_gib_est"] == 4.0


def test_plan_deeper_offload():
    plan = gt.plan_offload(_analysis(), vram_gb=2.5, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert plan["n_cpu_moe"] == 3
    assert plan["args"] == ["-ngl", "99", "--n-cpu-moe", "3"]


def test_plan_falls_back_to_cpu_moe_flag():
    plan = gt.plan_offload(_analysis(), vram_gb=1.5, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert plan["fits"] and plan["cpu_moe"] is True
    assert plan["args"] == ["-ngl", "99", "--cpu-moe"]


def test_plan_reports_does_not_fit():
    plan = gt.plan_offload(_analysis(), vram_gb=0.5, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert plan["fits"] is False and plan["mode"] == "does-not-fit"
    assert any("lower --ctx" in n for n in plan["notes"])


def test_plan_kv_cache_shrinks_budget():
    # 1 MiB of KV per token at 1024 tokens = 1 GiB, which pushes the plan deeper
    a = _analysis(kv_tok=GIB // 1024)
    with_kv = gt.plan_offload(a, vram_gb=4.5, ram_gb=None, ctx=1024, headroom_gb=0.0)
    without = gt.plan_offload(_analysis(), vram_gb=4.5, ram_gb=None, ctx=1024, headroom_gb=0.0)
    assert with_kv["n_cpu_moe"] > without["n_cpu_moe"]


def test_plan_warns_when_cpu_experts_exceed_ram():
    plan = gt.plan_offload(_analysis(), vram_gb=2.5, ram_gb=2, ctx=8192, headroom_gb=0.0)
    assert any("expect paging" in n for n in plan["notes"])


def test_plan_dense_model_partial_offload():
    a = {"is_moe": False, "layers": 3, "tensor_bytes": 3 * GIB, "expert_bytes": 0, "_expert_by_layer": [0, 0, 0],
         "kv_bytes_per_token": 0, "kv_note": None}
    assert gt.plan_offload(a, vram_gb=4, ram_gb=None, ctx=8192, headroom_gb=0.0)["args"] == ["-ngl", "99"]
    part = gt.plan_offload(a, vram_gb=2, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert part["args"] == ["-ngl", "2"] and part["fits"] is True
    none = gt.plan_offload(a, vram_gb=0.5, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert none["fits"] is False


def test_plan_unknown_layer_count():
    a = _analysis()
    a["layers"] = 0
    plan = gt.plan_offload(a, vram_gb=8, ram_gb=None, ctx=8192, headroom_gb=0.0)
    assert plan["fits"] is None and plan["mode"] == "unknown"


# ---------------------------------------------------------------- CLI and scan


def _run_cli(*args):
    proc = subprocess.run([sys.executable, str(TOOLS / "gguf_tool.py"), *args], capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout)


def test_cli_inspect_with_plan(moe_file: Path):
    code, out = _run_cli("inspect", str(moe_file), "--vram-gb", "0.0001", "--ram-gb", "32", "--ctx", "1024")
    assert code == 0 and out["ok"] is True
    assert out["architecture"] == "qwen3moe"
    assert out["family"]["id"] == "qwen3-moe"
    assert out["plan"]["mode"] == "does-not-fit"
    assert out["plan"]["fits"] is False


def test_cli_inspect_without_plan(moe_file: Path):
    code, out = _run_cli("inspect", str(moe_file))
    assert code == 0 and "plan" not in out
    assert "_expert_by_layer" not in out


def test_cli_error_is_json(tmp_path: Path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"ZZZZ")
    code, out = _run_cli("inspect", str(bad))
    assert code == 1 and out["ok"] is False and "not a GGUF" in out["error"]


def test_scan_lists_models_and_skips_later_shards(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    write_gguf(tmp_path / "a" / "qwen3-moe.gguf", moe_kv("qwen3moe"), moe_tensors())
    write_gguf(tmp_path / "b" / "split-00001-of-00002.gguf", moe_kv("gpt-oss"), [("blk.0.ffn_up_exps.weight", [64], 39, 64)])
    write_gguf(tmp_path / "b" / "split-00002-of-00002.gguf", [], [("blk.1.ffn_up_exps.weight", [64], 39, 64)])
    (tmp_path / "broken.gguf").write_bytes(b"junk")
    rows = gt.scan_dir(str(tmp_path))
    names = sorted(Path(r["path"]).name for r in rows)
    assert names == ["broken.gguf", "qwen3-moe.gguf", "split-00001-of-00002.gguf"]
    by_name = {Path(r["path"]).name: r for r in rows}
    assert by_name["qwen3-moe.gguf"]["family"] == "qwen3-moe"
    assert by_name["split-00001-of-00002.gguf"]["family"] == "gpt-oss"
    assert by_name["split-00001-of-00002.gguf"]["quant"] == "MXFP4"
    assert by_name["broken.gguf"]["ok"] is False


def test_registry_check_flags_missing_arch(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "llama-arch.cpp").write_text('{ LLM_ARCH_LLAMA, "llama" },\n', encoding="utf-8")
    res = gt.registry_check(str(tmp_path))
    assert res["ok"] is False and "qwen3moe" in res["missing"]
    code, out = _run_cli("registry-check", "--llama-src", str(tmp_path))
    assert code == 1 and out["ok"] is False
