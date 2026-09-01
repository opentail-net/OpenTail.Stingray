#!/usr/bin/env python3
"""
Reference oracle for Zhipu AI GLM-4.6V's ViT + 4-section M-RoPE + Conv2D patch-merger projector
(clip.vision.projector_type = "glm4v").

Faithfully reimplements tools/mtmd/models/glm4v.cpp's build() (122 lines, read in full) plus the
real GGML_ROPE_TYPE_VISION math traced through ggml-cpu/ops.cpp's ggml_mrope_cache_init +
rotate_pairs (see Glm4VisionEncoder.cs's ApplyMrope doc comment for the derivation: with
n_dims=head_dim/2 and 4 equal sections, only 2 of the 4 position channels are ever selected in
practice -- first quarter of [0,head_dim/2) rotates by row/py, second quarter by col/px, each
paired with its +head_dim/2 partner, covering the FULL head_dim). Same pattern as
scripts/llava_ref.py / scripts/pixtral_ref.py: real numpy port, reading the same local mmproj GGUF
the C# encoder reads, checked against the C# encoder's actual output.

Real tensor names confirmed via `list-tensors` before writing this (several differ from what a
"standard" mmproj naming convention would suggest -- see Glm4VisionEncoder.cs's doc comments for
the exact bugs this exposed and fixed):
  - ViT QKV is FUSED: v.blk.N.attn_qkv.weight (out=3*embd), no separate q/k/v tensors, no biases
    anywhere in the ViT blocks (qkv/attn_out/ffn all bias-free in this checkpoint).
  - Dual patch embed: v.patch_embd.weight + v.patch_embd.weight.1, summed.
  - Patch merger is a REAL strided Conv2D (mm.patch_merger.weight [cout=4096,cin=1536,kh=2,kw=2]
    via gguf-py's native (out,in,kh,kw) reshape) + mm.patch_merger.bias, not a plain concat.
  - FC projector tensor is mm.model.fc.weight/.bias (NOT mm.fc.* or mm.0.*).
  - Projector tail: mm.post_norm (plain LayerNorm, eps=1e-5, distinct from the ViT's own RMS eps)
    -> gelu_erf (erf-based, NOT tanh-approx) -> gated SiLU FFN (mm.gate/mm.up/mm.down).

Forward (per glm4v.cpp::build):
    dual conv2d patch embed (sum) -> + patch_bias -> RMSNorm(norm_embd)
        -> + learned position_embd (raw, AFTER norm_embd -- real build_vit adds it before any layer)
    24x transformer block:
        ln1 (RMSNorm) -> fused QKV -> split -> 4-section M-RoPE -> MHA -> attn_out -> residual
        ln2 (RMSNorm) -> SwiGLU (gate/up/down, no biases) -> residual
    post_ln (RMSNorm, v.post_ln.weight)
    patch merger: real strided Conv2D (2x2, 1536->4096) + bias
        -> FC (mm.model.fc, 4096->4096) -> post_norm (LayerNorm) -> gelu_erf
        -> gated SiLU FFN (mm.gate/mm.up/mm.down, 4096->10944->4096)
  -> [n_tokens, 4096]

Test image is generated at the checkpoint's own native image_size (336x336, patch=14 -> 24x24=576
patches) so the learned position_embd (stored for exactly a 576-position grid) applies directly,
with no bicubic resize -- matching the current C# encoder's scope, which does not implement a
resize path either.

Usage:
    python scripts/glm4v_ref.py models/mmproj-glm-4.6v-q4.gguf [out_dir]

Writes (default out_dir = tests/fixtures/glm4v):
    input_chw.f32    raw float32, shape [3,H,W] (synthetic preprocessed image, values 0..1)
    output.f32       raw float32, shape [n_tokens,4096]
    meta.json        shapes, dims, per-step stats
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 14
IMG = 336  # native image_size for this checkpoint -- matches its learned position_embd grid (24x24=576)

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "models/mmproj-glm-4.6v-q4.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/glm4v"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(mmproj)
    tmap = {t.name: t for t in r.tensors}
    meta = {f.name: f for f in r.fields.values()}

    def meta_float(name, default):
        if name in meta:
            v = meta[name].parts[meta[name].data[0]]
            return float(v[0])
        return default

    rope_theta = meta_float("clip.vision.rope.freq_base", 10000.0)

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
        if tt in ("Q4_1", "Q4_0", "Q8_0", "Q4_K", "Q6_K", "Q5_K", "Q3_K", "Q2_K"):
            # dequantize via gguf-py's own numpy dequant helpers (ggml_quants) for whichever
            # k-quant/legacy type appears -- same approach used implicitly by t.data for F-types.
            from gguf import quants
            deq = quants.dequantize(t.data, t.tensor_type)
            return np.asarray(deq, dtype=np.float32)
        raise RuntimeError(f"unhandled dtype {tt} for {name}")

    def vec_opt(name, dim):
        return raw_data(name) if name in tmap else np.zeros(dim, dtype=np.float32)

    n_embd = 1536
    n_heads = 12
    head_dim = n_embd // n_heads
    n_layers = 24
    proj_dim = 4096
    mm_ffn_dim = 10944
    eps = 1e-5

    pe0_w = raw_data("v.patch_embd.weight")
    assert pe0_w.shape == (n_embd, 3, PATCH, PATCH), pe0_w.shape
    pe1_w = raw_data("v.patch_embd.weight.1")
    assert pe1_w.shape == (n_embd, 3, PATCH, PATCH), pe1_w.shape
    patch_bias = vec_opt("v.patch_bias", n_embd)
    norm_embd_w = raw_data("v.norm_embd.weight")
    pos_embd = raw_data("v.position_embd.weight")  # (576, n_embd) native PyTorch [rows, embd]
    post_ln_w = raw_data("v.post_ln.weight")

    patch_merger_w = raw_data("mm.patch_merger.weight")  # (cout=4096, cin=1536, kh=2, kw=2)
    assert patch_merger_w.shape == (proj_dim, n_embd, 2, 2), patch_merger_w.shape
    patch_merger_b = raw_data("mm.patch_merger.bias")
    fc_w = raw_data("mm.model.fc.weight")
    fc_b = vec_opt("mm.model.fc.bias", proj_dim)
    assert fc_w.shape == (proj_dim, proj_dim), fc_w.shape
    post_norm_w = raw_data("mm.post_norm.weight")
    post_norm_b = raw_data("mm.post_norm.bias")
    gate_w = raw_data("mm.gate.weight")
    up_w = raw_data("mm.up.weight")
    down_w = raw_data("mm.down.weight")
    assert gate_w.shape == (mm_ffn_dim, proj_dim), gate_w.shape
    assert down_w.shape == (proj_dim, mm_ffn_dim), down_w.shape

    layers = []
    for l in range(n_layers):
        p = f"v.blk.{l}"
        qkv_w = raw_data(f"{p}.attn_qkv.weight")
        assert qkv_w.shape == (3 * n_embd, n_embd), qkv_w.shape
        gate_lw = raw_data(f"{p}.ffn_gate.weight")
        down_lw = raw_data(f"{p}.ffn_down.weight")
        layers.append(dict(
            ln1_w=raw_data(f"{p}.ln1.weight"),
            qkv_w=qkv_w,
            o_w=raw_data(f"{p}.attn_out.weight"),
            ln2_w=raw_data(f"{p}.ln2.weight"),
            gate_w=gate_lw,
            up_w=raw_data(f"{p}.ffn_up.weight"),
            down_w=down_lw,
            ffn_dim=gate_lw.shape[0],
        ))

    def rmsnorm(x, w, eps=eps):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def layernorm(x, w, b, eps=eps):
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def gelu_erf(x):
        from scipy.special import erf
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    half = head_dim // 2
    quarter = half // 2

    def mrope(q, k, px_arr, py_arr):
        # q,k: (n_tok, n_heads, head_dim). See module doc comment / Glm4VisionEncoder.ApplyMrope
        # for the derivation: ic in [0,quarter) rotates by py, ic in [quarter,half) rotates by px,
        # each paired with partner ic+half -- covering the FULL head_dim.
        d = np.arange(half)
        freq = rope_theta ** (-4.0 * d / head_dim)  # (half,)
        pos = np.where(d < quarter, py_arr[:, None], px_arr[:, None]).astype(np.float32)  # (n_tok,half)
        angle = pos * freq[None, :]
        cos_t, sin_t = np.cos(angle), np.sin(angle)  # (n_tok, half)

        def apply(x):
            x0 = x[:, :, :half]
            x1 = x[:, :, half:]
            c = cos_t[:, None, :]
            s = sin_t[:, None, :]
            new0 = x0 * c - x1 * s
            new1 = x0 * s + x1 * c
            return np.concatenate([new0, new1], axis=-1)

        return apply(q), apply(k)

    def mha(x, blk, px_arr, py_arr):
        n = x.shape[0]
        qkv = x @ blk["qkv_w"].T  # (n, 3*embd)
        q = qkv[:, :n_embd].reshape(n, n_heads, head_dim)
        k = qkv[:, n_embd:2 * n_embd].reshape(n, n_heads, head_dim)
        v = qkv[:, 2 * n_embd:].reshape(n, n_heads, head_dim)
        q, k = mrope(q, k, px_arr, py_arr)
        scale = 1.0 / np.sqrt(head_dim)
        out = np.empty((n, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            scores = (q[:, h] @ k[:, h].T) * scale
            scores = scores - scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p = p / p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h]
        return out.reshape(n, n_embd) @ blk["o_w"].T

    # ---- synthetic preprocessed image: CHW [3,336,336], values 0..1, deterministic ----
    H = W = IMG
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()

    gx = gy = IMG // PATCH
    n_patches = gx * gy
    assert n_patches == pos_embd.shape[0], (n_patches, pos_embd.shape)

    patches = np.empty((n_patches, n_embd), dtype=np.float32)
    px_arr = np.empty(n_patches, dtype=np.float32)
    py_arr = np.empty(n_patches, dtype=np.float32)
    for py in range(gy):
        for pxi in range(gx):
            p = py * gx + pxi
            block = img[:, py*PATCH:(py+1)*PATCH, pxi*PATCH:(pxi+1)*PATCH]
            v0 = np.tensordot(pe0_w, block, axes=([1, 2, 3], [0, 1, 2]))
            v1 = np.tensordot(pe1_w, block, axes=([1, 2, 3], [0, 1, 2]))
            patches[p] = v0 + v1 + patch_bias
            px_arr[p] = pxi
            py_arr[p] = py

    x = rmsnorm(patches, norm_embd_w)
    x = x + pos_embd

    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("after_patch_embd", x)

    for li, blk in enumerate(layers):
        normed = rmsnorm(x, blk["ln1_w"])
        attn = mha(normed, blk, px_arr, py_arr)
        x = x + attn

        normed2 = rmsnorm(x, blk["ln2_w"])
        gate = silu(normed2 @ blk["gate_w"].T)
        up = normed2 @ blk["up_w"].T
        ff = (gate * up) @ blk["down_w"].T
        x = x + ff
        if li == n_layers - 1:
            rec("after_last_block", x)

    x = rmsnorm(x, post_ln_w)
    rec("after_post_ln", x)

    # ---- real strided Conv2D patch merger (2x2, embd -> proj_dim) ----
    scale = 2
    downX, downY = gx // scale, gy // scale
    n_tokens = downX * downY
    grid = x.reshape(gy, gx, n_embd)  # [row, col, embd]
    merged = np.empty((n_tokens, proj_dim), dtype=np.float32)
    for dy0 in range(downY):
        for dx0 in range(downX):
            t = dy0 * downX + dx0
            block = grid[dy0*scale:(dy0+1)*scale, dx0*scale:(dx0+1)*scale, :]  # [2,2,embd]
            # patch_merger_w: (cout, cin, kh, kw); block: (kh, kw, cin) -> reorder to (cin,kh,kw)
            block_ckw = np.transpose(block, (2, 0, 1))
            merged[t] = np.tensordot(patch_merger_w, block_ckw, axes=([1, 2, 3], [0, 1, 2])) + patch_merger_b
    rec("after_patch_merger", merged)

    h = merged @ fc_w.T + fc_b
    h = layernorm(h, post_norm_w, post_norm_b)
    h = gelu_erf(h)
    rec("after_fc_gelu", h)

    gate = silu(h @ gate_w.T)
    up = h @ up_w.T
    out = (gate * up) @ down_w.T
    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta_out = dict(patch=PATCH, H=H, W=W, gx=gx, gy=gy, n_tokens=n_tokens, n_embd=proj_dim,
                     rope_theta=rope_theta, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(json.dumps(meta_out, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_tokens},{proj_dim}] to {out_dir}")

if __name__ == "__main__":
    main()
