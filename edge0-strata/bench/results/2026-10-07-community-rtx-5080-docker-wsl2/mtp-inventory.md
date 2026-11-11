# MTP block in the BF16 checkpoint

31 tensors, 5.214 GB, in 28 shards.

| tensor | dtype | shape | MB |
|---|---|---|---:|
| `mtp.fc_embedding.weight` | BF16 | 2560x2560 | 13.1 |
| `mtp.fc_hidden.weight` | BF16 | 2560x2560 | 13.1 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | BF16 | 10240 | 0.0 |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | BF16 | 320x10240 | 6.6 |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | BF16 | 10240x320 | 6.6 |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | 0.1 |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | BF16 | 10240 | 0.0 |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | 6.6 |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | 6.6 |
| `mtp.layers.0.mlp.experts.down_proj` | BF16 | 512x2560x640 | 1677.7 |
| `mtp.layers.0.mlp.experts.gate_up_proj` | BF16 | 512x1280x2560 | 3355.4 |
| `mtp.layers.0.mlp.gate.weight` | BF16 | 512x2560 | 2.6 |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | BF16 | 2560x640 | 3.3 |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | BF16 | 640x2560 | 3.3 |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | BF16 | 640x2560 | 3.3 |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | BF16 | 1x2560 | 0.0 |
| `mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight` | BF16 | 4x10240 | 0.1 |
| `mtp.layers.0.mlp_hyper_connection.hc_norm.weight` | BF16 | 10240 | 0.0 |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | BF16 | 320x10240 | 6.6 |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | BF16 | 10240x320 | 6.6 |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | BF16 | 640x2560 | 3.3 |
| `mtp.layers.0.self_attn.indexer.k_layernorm.weight` | BF16 | 128 | 0.0 |
| `mtp.layers.0.self_attn.indexer.q_layernorm.weight` | BF16 | 128 | 0.0 |
| `mtp.layers.0.self_attn.k_norm.weight` | BF16 | 256 | 0.0 |
| `mtp.layers.0.self_attn.k_proj.weight` | BF16 | 512x2560 | 2.6 |
| `mtp.layers.0.self_attn.o_proj.weight` | BF16 | 2560x6144 | 31.5 |
| `mtp.layers.0.self_attn.q_norm.weight` | BF16 | 256 | 0.0 |
| `mtp.layers.0.self_attn.q_proj.weight` | BF16 | 12288x2560 | 62.9 |
| `mtp.layers.0.self_attn.v_proj.weight` | BF16 | 512x2560 | 2.6 |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | 2560 | 0.0 |
| `mtp.pre_fc_norm_hidden.weight` | BF16 | 10240 | 0.0 |
