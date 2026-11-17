"""tools/nvfp4_convert.py - a ModelOpt NVFP4 Qwen3.8-Flash-Next checkpoint -> a GGUF Strata can pack.

    python tools/nvfp4_convert.py --model <ModelOpt NVFP4 dir> --outfile <model.gguf>

This is llama.cpp's own converter (third_party/llama.cpp at the commit setup.py pins), run with the type policy
Strata's engine needs. The converter already repacks ModelOpt NVFP4 experts into ggml's block_nvfp4 without loss
(tools/nvfp4_verify.py proves it bit for bit) and writes each expert's weight_scale_2 as a separate F32 tensor
`blk.N.ffn_{gate,up,down}_exps.scale`, one value per expert. What it does not know is Strata's layout:

  * the small projections the residual, router, GDN, QSA and PLE kernels read as BF16 (iq_pack.BF16_PROJECTIONS)
    must be BF16 - the converter would otherwise write routers and indexer projections as F32, which iq_pack
    refuses, and the rest as the --outtype;
  * everything the engine serves natively (attention, ssm_out, shared experts, output) goes to Q8_0: the native
    GEMV has no BF16 path, and Q8_0 is the closest to the checkpoint's BF16 it has;
  * token_embd goes to Q8_0 too. The abliteration edited embed_tokens, so another model's table cannot be reused,
    and gguf-py cannot write the IQ types the native embedding reads today - the engine learns Q8_0 instead;
  * the 51.2B-parameter PLE n-gram table is left out. The engine only reads it as IQ4_NL (ngram.cpp), and
    dequantizing the checkpoint's FP8 copy would need ~205 GB of RAM. The abliteration did not touch this table,
    so the engine is pointed at ISTA-DASLab's IQ4_NL shard 2 with --ple-gguf. The PLE hash constants and the
    row width still come from this checkpoint and are written into the metadata.

The MTP head is dropped here as llama.cpp does; tools/mtp_pack.py packs it separately.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ENGINE = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ENGINE / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))
sys.path.insert(0, str(LLAMA))

import gguf  # noqa: E402
from conversion import qwen4exp as Q4  # noqa: E402

T = gguf.GGMLQuantizationType

# Strata reads these as BF16 (tools/iq_pack.py BF16_PROJECTIONS + BF16_OUTPUT, and the routers).
BF16_SUFFIXES = (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "ssm_alpha.weight", "ssm_beta.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
    "ple_value.weight", "ple_key.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
    "output_hc_down.weight", "output_hc_up.weight",
)
F16_SUFFIXES = ("ple_conv1d.weight",)
Q8_NAMES = {"token_embd.weight", "output.weight"}

dropped_ple: list[str] = []


def install_policy() -> None:
    cls = Q4.Qwen4ExpTextModel
    base_force = cls.tensor_force_quant

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims >= 2:
            if new_name.endswith(BF16_SUFFIXES):
                return T.BF16
            if new_name.endswith(F16_SUFFIXES):
                return T.F16
            if new_name in Q8_NAMES:
                return T.Q8_0
        return base_force(self, name, new_name, bid, n_dims)

    def place_ple_shard(self, data_torch, name):
        # Record the shard so the parts check in prepare_tensors still holds, and the row width for the
        # metadata; write nothing - the table comes from the IQ4_NL PLE GGUF at run time.
        idx = int(name.rsplit("shard_", 1)[1].split(".", 1)[0])
        self._ple_shards[idx] = name
        self._ple_row_dim = int(data_torch.shape[-1])
        dropped_ple.append(name)
        return []

    cls.tensor_force_quant = tensor_force_quant
    cls._place_ple_shard = place_ple_shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the ModelOpt NVFP4 checkpoint directory")
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    model = pathlib.Path(a.model)
    out = pathlib.Path(a.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    install_policy()
    import convert_hf_to_gguf as C  # noqa: E402  (after the policy is installed on the class)

    argv = [str(model), "--outtype", "q8_0", "--outfile", str(out)]
    if a.dry_run:
        argv.append("--dry-run")
    sys.argv = ["convert_hf_to_gguf.py"] + argv
    t0 = time.time()
    rc = C.main()
    took = time.time() - t0

    try:   # a release bundle ships the converter's files without git: its COMMIT file names the commit
        commit = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    if not commit and (LLAMA / "COMMIT").exists():
        commit = (LLAMA / "COMMIT").read_text(encoding="utf-8").strip()
    manifest = {
        "source": str(model),
        "llama_cpp_commit": commit,
        "policy": {"bf16": list(BF16_SUFFIXES), "f16": list(F16_SUFFIXES), "q8_0": sorted(Q8_NAMES),
                   "default_2d": "Q8_0", "experts": "NVFP4 (lossless repack) + per-expert F32 .scale"},
        "ple_table": {"written": False, "shards_dropped": len(dropped_ple),
                      "use": "--ple-gguf with an IQ4_NL per_layer_token_embd (ISTA-DASLab shard 2)"},
        "mtp": "dropped here; pack with tools/mtp_pack.py",
        "seconds": round(took, 1),
    }
    if not a.dry_run:
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("nvfp4_convert: done in %.0f s, PLE shards dropped: %d" % (took, len(dropped_ple)))
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
