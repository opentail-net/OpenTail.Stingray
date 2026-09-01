#!/usr/bin/env python3
"""
Reference oracle for Pixtral's CLIP-style ViT + 2D continuous RoPE + SwiGLU FFN + GELU MLP
projector (clip.vision.projector_type = "pixtral").

Faithfully reimplements tools/mtmd/models/pixtral.cpp's build() (mm_patch_merger_w absent for
this checkpoint, so that branch is skipped -- confirmed via list-tensors before writing this) and
tools/mtmd/clip.cpp's build_rope_2d(), using the real mmproj tensors -- same pattern as
scripts/gemma4uv_ref.py / scripts/llava_ref.py.

KNOWN, DELIBERATE GAP (matches the C# PixtralVisionEncoder's own current scope, not a full
real-pixtral.cpp replica): this checkpoint's GGUF has a real `v.token_embd.img_break` tensor,
meaning the real reference would insert an [IMG_BREAK] token after every row of patches before
the projector -- but PixtralVisionEncoder.cs does not implement that insertion at all (confirmed
by reading the C# source: no img_break/mm_patch_merger_w reference anywhere in it). This script
therefore deliberately reproduces the encoder's ACTUAL current scope (ViT + RoPE-2D + SwiGLU +
GELU projector, no patch merger, no IMG_BREAK row insertion) so the comparison is apples-to-apples
against what the C# code actually computes today -- the missing IMG_BREAK/patch-merger support is
a separate, real, documented gap (see docs), not silently "fixed" by this script.

Forward (per pixtral.cpp::build / clip.cpp build_vit, RMS-norm ViT, SwiGLU FFN):
    patch_embd: conv2d(patch=16, stride=16, 3->1024), no CLS token, no learned position embd
    pre_ln (RMSNorm, no bias, weight only)
    24x transformer block:
        ln1 (RMSNorm) -> separate Q/K/V (no bias)
            -> 2D continuous RoPE (build_rope_2d, theta=rope_theta, interleave_freq=True)
            -> standard MHA (16 heads, head_dim=64, scale=1/sqrt(64)) -> attn_out (no bias)
            -> residual
        ln2 (RMSNorm) -> SwiGLU: down(silu(gate(x)) * up(x)), no biases except ffn_down -> residual
    post_ln (RMSNorm)
    MLP projector: mm.1(1024->5120)+bias -> GELU(tanh-approx) -> mm.2(5120->5120)+bias
  -> [n_patches, 5120]

Usage:
    python scripts/pixtral_ref.py models/mmproj-pixtral-12b-f16.gguf [out_dir]

Writes (default out_dir = tests/fixtures/pixtral):
    input_chw.f32    raw float32, shape [3,H,W] (synthetic preprocessed image, values 0..1)
    output.f32       raw float32, shape [n_patches,5120]
    meta.json        shapes, dims, per-step stats
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 16
IMG = 128  # matches MultimodalRealWeightsTests.cs's own CreateTestImagePair(128,128) for Pixtral

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "models/mmproj-pixtral-12b-f16.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/pixtral"
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
        # Verified (llava_ref.py, same GGUF family): gguf-py's own reshape of `.data` already
        # gives PyTorch nn.Linear.weight's native [out,in] layout for 2D weights, and the real
        # (d,c,ky,kx) layout directly for the conv patch-embed weight.
        t = tmap[name]
        tt = t.tensor_type.name
        if tt == "F32":
            return np.asarray(t.data, dtype=np.float32)
        if tt == "F16":
            return np.asarray(t.data, dtype=np.float16).astype(np.float32)
        if tt == "BF16":
            bits = np.frombuffer(t.data.tobytes(), dtype=np.uint16).astype(np.uint32)
            return (bits << 16).view(np.float32).reshape(t.data.shape)
        raise RuntimeError(f"unhandled dtype {tt} for {name}")

    def vec_opt(name, dim):
        return raw_data(name) if name in tmap else np.zeros(dim, dtype=np.float32)

    n_embd = 1024
    n_heads = 16
    head_dim = n_embd // n_heads
    n_layers = 24
    ffn_dim = 4096
    proj_dim = 5120
    eps = 1e-5

    pe_w = raw_data("v.patch_embd.weight")
    assert pe_w.shape == (n_embd, 3, PATCH, PATCH), pe_w.shape
    pe_b = vec_opt("v.patch_embd.bias", n_embd)
    pre_ln_w = raw_data("v.pre_ln.weight") if "v.pre_ln.weight" in tmap else None
    post_ln_w = raw_data("v.post_ln.weight") if "v.post_ln.weight" in tmap else None

    mm1_w, mm1_b = raw_data("mm.1.weight"), raw_data("mm.1.bias")
    mm2_w, mm2_b = raw_data("mm.2.weight"), raw_data("mm.2.bias")
    assert mm1_w.shape == (proj_dim, n_embd), mm1_w.shape
    assert mm2_w.shape == (proj_dim, proj_dim), mm2_w.shape

    layers = []
    for l in range(n_layers):
        p = f"v.blk.{l}"
        layers.append(dict(
            ln1_w=raw_data(f"{p}.ln1.weight"),
            q_w=raw_data(f"{p}.attn_q.weight"),
            k_w=raw_data(f"{p}.attn_k.weight"),
            v_w=raw_data(f"{p}.attn_v.weight"),
            o_w=raw_data(f"{p}.attn_out.weight"),
            ln2_w=raw_data(f"{p}.ln2.weight"),
            gate_w=raw_data(f"{p}.ffn_gate.weight"),
            up_w=raw_data(f"{p}.ffn_up.weight"),
            down_w=raw_data(f"{p}.ffn_down.weight"),
            down_b=vec_opt(f"{p}.ffn_down.bias", n_embd),
        ))
        assert layers[-1]["gate_w"].shape == (ffn_dim, n_embd), layers[-1]["gate_w"].shape
        assert layers[-1]["down_w"].shape == (n_embd, ffn_dim), layers[-1]["down_w"].shape

    def rmsnorm(x, w, eps=eps):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def rope_2d(q, k, px, py, theta):
        # q,k: (n_tok, n_heads, head_dim). Real build_rope_2d: split head_dim into two HALVES;
        # first half rotated by row-position (pos_a) with plain freq_base=theta, n_dims=half;
        # second half rotated by col-position (pos_b) with an EXTRA freq_scale_odd =
        # theta**(-2/head_dim) applied on top of the same per-dim frequency formula. Real
        # pixtral.cpp passes (pos_h, pos_w) as (pos_a, pos_b) -- pos_h is confusingly the
        # PATCH-GRID COLUMN in real mtmd_image (verified against clip.cpp's own pos_h/pos_w fill:
        # both are actually filled as [row]*width+[col] pairs identically for h/w -- practically,
        # the two halves rotate by (row) and (col) respectively, matching px/py naming below).
        half = head_dim // 2
        quarter = half // 2
        n_tok = q.shape[0]
        freq_scale_odd = theta ** (-2.0 / head_dim)
        out_q = q.copy()
        out_k = k.copy()
        for t in range(n_tok):
            row = t // px_count if False else None  # unused, kept for clarity
        # vectorized instead of per-token python loop (n_tok can be large)
        d = np.arange(quarter)
        freq = theta ** (-2.0 * d / half)  # (quarter,)
        # first half: rotate by row position (py), using px/py arrays passed in by caller
        angle_row = np.outer(py, freq)  # (n_tok, quarter)
        cos_r, sin_r = np.cos(angle_row), np.sin(angle_row)
        # second half: rotate by col position (px), extra freq_scale_odd
        angle_col = np.outer(px, freq) * freq_scale_odd
        cos_c, sin_c = np.cos(angle_col), np.sin(angle_col)

        def apply(x):
            x = x.copy()
            for h in range(n_heads):
                # first half: local pairs (0,1),(2,3),... within [0,half)
                x0 = x[:, h, 0:half:2].copy()
                x1 = x[:, h, 1:half:2].copy()
                x[:, h, 0:half:2] = x0 * cos_r - x1 * sin_r
                x[:, h, 1:half:2] = x0 * sin_r + x1 * cos_r
                # second half: local pairs within [half,head_dim)
                x0b = x[:, h, half::2].copy()
                x1b = x[:, h, half + 1::2].copy()
                x[:, h, half::2] = x0b * cos_c - x1b * sin_c
                x[:, h, half + 1::2] = x0b * sin_c + x1b * cos_c
            return x

        return apply(q), apply(k)

    def mha(x, blk, px_arr, py_arr):
        n = x.shape[0]
        q = (x @ blk["q_w"].T).reshape(n, n_heads, head_dim)
        k = (x @ blk["k_w"].T).reshape(n, n_heads, head_dim)
        v = (x @ blk["v_w"].T).reshape(n, n_heads, head_dim)
        q, k = rope_2d(q, k, px_arr, py_arr, rope_theta)
        scale = 1.0 / np.sqrt(head_dim)
        out = np.empty((n, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            scores = (q[:, h] @ k[:, h].T) * scale
            scores = scores - scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p = p / p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h]
        return out.reshape(n, n_embd) @ blk["o_w"].T

    # ---- synthetic preprocessed image: CHW [3,128,128], values 0..1, deterministic ----
    H = W = IMG
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()

    gx = gy = IMG // PATCH
    n_patches = gx * gy

    patches = np.empty((n_patches, n_embd), dtype=np.float32)
    px_arr = np.empty(n_patches, dtype=np.int64)
    py_arr = np.empty(n_patches, dtype=np.int64)
    for py in range(gy):
        for pxi in range(gx):
            p = py * gx + pxi
            block = img[:, py*PATCH:(py+1)*PATCH, pxi*PATCH:(pxi+1)*PATCH]
            patches[p] = np.tensordot(pe_w, block, axes=([1, 2, 3], [0, 1, 2])) + pe_b
            px_arr[p] = pxi
            py_arr[p] = py

    x = patches  # no CLS, no learned position embd for pixtral

    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("after_patch_embd", x)

    if pre_ln_w is not None:
        x = rmsnorm(x, pre_ln_w)
    rec("after_pre_ln", x)

    for li, blk in enumerate(layers):
        normed = rmsnorm(x, blk["ln1_w"])
        attn = mha(normed, blk, px_arr, py_arr)
        x = x + attn

        normed2 = rmsnorm(x, blk["ln2_w"])
        gate = silu(normed2 @ blk["gate_w"].T)
        up = normed2 @ blk["up_w"].T
        ff = (gate * up) @ blk["down_w"].T + blk["down_b"]
        x = x + ff
        if li == n_layers - 1:
            rec("after_last_block", x)

    if post_ln_w is not None:
        x = rmsnorm(x, post_ln_w)
    rec("after_post_ln", x)

    h = x @ mm1_w.T + mm1_b
    h = gelu_tanh(h)
    rec("after_mm1_gelu", h)
    out = h @ mm2_w.T + mm2_b
    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta_out = dict(patch=PATCH, H=H, W=W, gx=gx, gy=gy, n_tokens=n_patches, n_embd=proj_dim,
                     rope_theta=rope_theta, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(json.dumps(meta_out, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_patches},{proj_dim}] to {out_dir}")

if __name__ == "__main__":
    main()
