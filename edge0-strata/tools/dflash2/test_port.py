import sys, importlib.util, json, torch
sys.path.insert(0, r"C:\dflash2-work")
import dflash2_ref as R
torch.manual_seed(0)
# the reference (z-lab) model on transformers 4.57
spec = importlib.util.spec_from_file_location("ghm", r"C:\Users\AI-Server\Desktop\Strata\phaseA-tests\ornith_tmp\gh_model.py")
ghm = importlib.util.module_from_spec(spec); spec.loader.exec_module(ghm)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
cfgd = json.load(open(r"C:\dflash2-work\drafter_config.json"))
c = Qwen3Config(**{k: v for k, v in cfgd.items() if k in ("hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim","vocab_size","rms_norm_eps","max_position_embeddings","attention_bias","sliding_window","layer_types","use_sliding_window","max_window_layers")})
c.rope_theta = cfgd["rope_parameters"]["rope_theta"]
c.dflash_config = cfgd["dflash_config"]
c.is_causal = False
c.num_target_layers = 48
m = ghm.DFlash2DraftModel(c)
from safetensors.torch import load_file
sd = load_file(r"C:\dflash2-work\drafter.safetensors")
sd = {(k + ".weight" if k.startswith("candidate_selector.") and k.endswith("codebook") else k): v for k, v in sd.items()}
miss = m.load_state_dict(sd, strict=False); print("missing", miss.missing_keys[:5], "unexpected", miss.unexpected_keys[:5])
m = m.float().eval()
# my port with a tiny random embed/head
V = 248320; H = 2560
embed = torch.randn(V, H) * 0.02
head = torch.randn(V, H) * 0.02
d = R.Drafter(r"C:\dflash2-work\drafter.safetensors", r"C:\dflash2-work\drafter_config.json", embed, head)
n = 300; b = 8; s = 250
feats = torch.randn(n, 5 * H) * 2.0
d.set_context(feats)
anchor = 1234
mine_path, _ = d.draft(s, anchor, b)
ids = torch.tensor([[anchor] + [d.mask_id] * (b - 1)])
noise = embed[ids]
pos = torch.arange(0, s + b)[None]       # context 0..s-1 then block
with torch.no_grad():
    out = m(position_ids=pos, noise_embedding=noise, target_hidden=feats[None, :s], past_key_values=None, use_cache=False)
print("shape", out.shape, "mine", d.last_all.shape)
err = (out[0] - d.last_all).abs().max().item()
print("max abs err all rows", err, "ref scale", out.abs().mean().item())
# the selector path against the reference
logits = (out[:, 1:] @ head.T)
class Tgt: pass
with torch.no_grad():
    path, idx, q = m.candidate_selector.select(out[:, 1:], logits, torch.tensor([anchor]), 0.0)
print("ref path ", path[0].tolist())
print("mine path", mine_path)
