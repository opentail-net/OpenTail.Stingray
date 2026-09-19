#!/usr/bin/env python3
"""
Reference oracle for FLUX.1's conditioning vector `vec` (FluxDiT.ComputeVec), the single input
every double/single block's AdaLN modulation derives from -- same rationale as
scripts/sd3_timestep_embed_ref.py, this project's precedent for isolating this exact class of bug
via direct GGUF weight reads + a hand-reimplemented reference formula (no full diffusers pipeline
load required; the DiT's own `time_in`/`vector_in` MLP weights are read straight out of the real
checkpoint).

Faithfully reimplements FLUX.1's real vec construction, confirmed against
examples/diffusers/src/diffusers/models/transformers/transformer_flux.py's `FluxTransformer2DModel.
forward` and `embeddings.py`'s `CombinedTimestepGuidanceTextProjEmbeddings`/
`CombinedTimestepTextProjEmbeddings`:

    tEmb = get_timestep_embedding(t*1000, 256, flip_sin_to_cos=True, downscale_freq_shift=0)
    tProj = time_in.out_layer(silu(time_in.in_layer(tEmb)))          # MLPEmbedder, [3072]
    vProj = vector_in.out_layer(silu(vector_in.in_layer(pooled)))    # MLPEmbedder, [3072]
    vec = tProj + vProj   (+ guidance_in(...) for FLUX.1-dev only -- schnell has no guidance_in)

`pooled` here is a synthetic deterministic vector (NOT real CLIP-L output), same deliberate scope
limit as sd3_timestep_embed_ref.py -- this isolates the embedding-MLP math only, not the text
encoder. `t` is a fixed timestep in [0,1] (flow-matching convention), pre-scaled by 1000 exactly as
FluxDiT.TimestepEmbedding does internally.

Usage:
    python scripts/flux1_vec_embed_ref.py <path-to-flux1-schnell-Q4_K_S.gguf> [out_dir]

Writes (default out_dir = tests/fixtures/flux1_vec_embed):
    vec.f32     raw float32 [hiddenSize=3072], the real reference `vec`
    meta.json   timestep, hiddenSize, vecDim, pooled (so the C# test can reconstruct the identical
                input deterministically)
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader


def main():
    dit_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/flux1_vec_embed"
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

    prefix = "model.diffusion_model."
    # ComfyUI-GGUF-style converters sometimes strip this prefix -- match FluxDiT.cs's own fallback.
    if f"{prefix}time_in.in_layer.weight" not in tmap:
        prefix = ""

    t0_w = raw_data(f"{prefix}time_in.in_layer.weight")
    t0_b = raw_data(f"{prefix}time_in.in_layer.bias")
    t2_w = raw_data(f"{prefix}time_in.out_layer.weight")
    t2_b = raw_data(f"{prefix}time_in.out_layer.bias")
    v0_w = raw_data(f"{prefix}vector_in.in_layer.weight")
    v0_b = raw_data(f"{prefix}vector_in.in_layer.bias")
    v2_w = raw_data(f"{prefix}vector_in.out_layer.weight")
    v2_b = raw_data(f"{prefix}vector_in.out_layer.bias")

    hidden_size = t2_w.shape[0]
    vec_dim = v0_w.shape[1]

    timestep = 0.5  # fixed, mid-trajectory flow-matching t in [0,1]

    # get_timestep_embedding(t*1000, 256, flip_sin_to_cos=True, downscale_freq_shift=0):
    # real diffusers divides by half_dim (shift=0), NOT half_dim-1 -- matches FluxDiT.cs's own
    # 2026-09-11 fix (docs/056 Round 7). emb layout after the flip is [cos, sin], not [sin, cos].
    half = 128
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = (timestep * 1000.0) * freqs
    t_emb = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)  # [256]

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def mlp(x, w0, b0, w2, b2):
        h = w0 @ x + b0
        h = silu(h)
        return w2 @ h + b2

    t_proj = mlp(t_emb, t0_w, t0_b, t2_w, t2_b)

    # Synthetic deterministic pooled vector, same convention as sd3_timestep_embed_ref.py.
    rng = np.random.RandomState(42)
    pooled = rng.uniform(-1.0, 1.0, size=vec_dim).astype(np.float32)

    v_proj = mlp(pooled, v0_w, b0=v0_b, w2=v2_w, b2=v2_b)

    vec = (t_proj + v_proj).astype(np.float32)

    vec.tofile(os.path.join(out_dir, "vec.f32"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "timestep": timestep,
            "hiddenSize": int(hidden_size),
            "vecDim": int(vec_dim),
            "pooled": pooled.tolist(),
        }, f)

    print(f"Wrote {out_dir}/vec.f32 ({hidden_size} floats) and meta.json")


if __name__ == "__main__":
    main()
