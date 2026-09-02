#!/usr/bin/env python3
"""
Reference oracle for SD3/3.5's timestep+pooled-embedding conditioning vector
(MMDiTModel.ComputeTimeAndPooledEmbedding), the SINGLE input every joint block's AdaLN modulation
depends on -- a bug here would explain total-noise output better than a localized per-block bug.

Faithfully reimplements examples/diffusers/src/diffusers/models/embeddings.py's
CombinedTimestepTextProjEmbeddings.forward: Timesteps(num_channels=256, flip_sin_to_cos=True,
downscale_freq_shift=0) -> TimestepEmbedding MLP (SiLU-gated 2-layer) added to a
PixArtAlphaTextProjection MLP (same SiLU-gated 2-layer shape) over the pooled text embedding.

Real Timesteps math (get_timestep_embedding, downscale_freq_shift=0, flip_sin_to_cos=True):
    half = 128
    freq[i] = exp(-log(10000) * i / half)          for i in [0, half)
    arg = timestep * freq
    emb = concat([sin(arg), cos(arg)])              # [256], NOT flipped yet
    emb = concat([emb[half:], emb[:half]])          # flip_sin_to_cos -> [cos(arg), sin(arg)]
This matches MMDiTModel.cs's sinEmb[i]=cos(arg), sinEmb[half+i]=sin(arg) exactly -- read and
confirmed against embeddings.py before writing this, per CLAUDE.md rule 8.

Usage:
    python scripts/sd3_timestep_embed_ref.py <path-to-sd3.5-medium-DiT.gguf> [out_dir]

Writes (default out_dir = tests/fixtures/sd3_timestep_embed):
    conditioning.f32   raw float32 [hiddenSize], for a fixed timestep=500.0 and a synthetic
                        deterministic pooled_projection vector (NOT real CLIP output -- this
                        isolates the embedding MLP math only, not the text encoders).
    meta.json           timestep, hiddenSize, admInChannels, pooled_projection (so the C# test
                         can reconstruct the identical input deterministically).
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

def main():
    dit_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/sd3_timestep_embed"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(dit_path)
    tmap = {t.name: t for t in r.tensors}

    def raw_data(name):
        t = tmap[name]
        tt = t.tensor_type.name
        if tt == "F32":
            return np.asarray(t.data, dtype=np.float32)
        if tt == "F16":
            return np.asarray(t.data, dtype=np.float16).astype(np.float32)
        if tt == "BF16":
            bits = np.frombuffer(t.data.tobytes(), dtype=np.uint16).astype(np.uint32)
            return (bits << 16).view(np.float32).reshape(t.data.shape)
        from gguf import quants
        return np.asarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32)

    t0_w, t0_b = raw_data("t_embedder.mlp.0.weight"), raw_data("t_embedder.mlp.0.bias")
    t2_w, t2_b = raw_data("t_embedder.mlp.2.weight"), raw_data("t_embedder.mlp.2.bias")
    y0_w, y0_b = raw_data("y_embedder.mlp.0.weight"), raw_data("y_embedder.mlp.0.bias")
    y2_w, y2_b = raw_data("y_embedder.mlp.2.weight"), raw_data("y_embedder.mlp.2.bias")

    hidden_size = t0_w.shape[0]
    adm_in_channels = y0_w.shape[1]
    assert t2_w.shape == (hidden_size, hidden_size), t2_w.shape
    assert y2_w.shape == (hidden_size, hidden_size), y2_w.shape

    timestep = 500.0
    # Deterministic synthetic pooled projection (not real CLIP output -- isolates MLP math only).
    pooled = (np.sin(np.arange(adm_in_channels, dtype=np.float32) * 0.017) * 0.5).astype(np.float32)

    def silu(x):
        return x / (1.0 + np.exp(-x))

    half = 128
    freq = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    arg = timestep * freq
    sin_emb = np.concatenate([np.cos(arg), np.sin(arg)]).astype(np.float32)  # flip_sin_to_cos

    t_emb = silu(sin_emb @ t0_w.T + t0_b) @ t2_w.T + t2_b
    y_emb = silu(pooled @ y0_w.T + y0_b) @ y2_w.T + y2_b
    conditioning = (t_emb + y_emb).astype(np.float32)

    conditioning.tofile(os.path.join(out_dir, "conditioning.f32"))
    meta = dict(timestep=timestep, hiddenSize=int(hidden_size), admInChannels=int(adm_in_channels),
                pooled=pooled.tolist(),
                stats=[float(conditioning.mean()), float(conditioning.std()),
                       float(conditioning.min()), float(conditioning.max())])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(json.dumps({k: v for k, v in meta.items() if k != "pooled"}, indent=2))
    print(f"\nWrote conditioning.f32 [{hidden_size}] to {out_dir}")

if __name__ == "__main__":
    main()
