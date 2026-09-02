#!/usr/bin/env python3
"""
Reference oracle for Tencent HunyuanVL (HunyuanOCR)'s plain-LayerNorm ViT + strided-Conv2D
perceiver projector + image-wrap-token sequence (clip.projector_type = "hunyuanvl").

Faithfully reimplements tools/mtmd/models/hunyuanvl.cpp's build() (63 lines, read in full) plus
the real PROJECTOR_TYPE_HUNYUANVL position-embedding bilinear-resize branch in clip.cpp
(~lines 4809-4869) -- see docs/059-hunyuanvl-implementation-plan.md for the full real-reference
citations this was derived from. Same pattern as scripts/exaone4_ref.py / scripts/glm4v_ref.py:
real numpy port, reading the same local mmproj GGUF the C# encoder reads.

Real tensor names/shapes confirmed via `list-tensors`/`list-metadata` before writing this:
  - ViT: plain (not fused) attn_q/k/v/out with bias; plain LayerNorm ln1/ln2 (NOT RMSNorm), both
    with bias; non-gated GELU FFN (ffn_up/ffn_down only, both with bias -- no ffn_gate tensor
    exists in this checkpoint). No RoPE at all. No separate v.pre_ln/v.post_ln tensors exist.
  - patch_size=16, embedding_length=1152, block_count=27, head_count=16 (head_dim=72).
  - v.position_embd.weight is a native 128x128 grid ([n_pos=16384, embd=1152] once dequantized via
    gguf-py's native (out,in) 2D reshape) that must be bilinearly resized (pixel-center convention,
    align_corners=False) to the real (patchesX,patchesY) grid on every forward pass.
  - mm.pre_norm: RMSNorm (not LayerNorm), applied to the ViT's raw output before the projector.
  - mm.0: a REAL strided Conv2D (kernel=stride=n_merge=2), raw shape [2,2,1152,2304] -> via
    gguf-py's native (cout,cin,kh,kw) reshape, (2304,1152,2,2) -- channel OUTER, spatial (dy,dx)
    INNER per output channel. GELU (tanh-approx, matching VisionOps.GeluScalar / ggml_gelu, NOT
    erf) after.
  - mm.2: a 1x1 Conv2D, raw shape [1,1,2304,4608] -> (4608,2304,1,1), mathematically a plain
    per-position Linear once squeezed.
  - v.image_newline ([4608]) is inserted after every row of the merged (outX,outY) token grid
    BEFORE the final projection -- real token count is (outX+1)*outY, not outX*outY.
  - mm.model.fc ([4608,1024] Q8_0) projects to LLM hidden size, AFTER newline insertion.
  - mm.image_begin/mm.image_end ([1024] each) wrap the WHOLE projected sequence (prepend/append),
    and ONLY THEN is mm.post_norm (RMSNorm) applied -- to the whole wrapped sequence including the
    begin/end markers. Real final token count: (outX+1)*outY + 2.

Forward (per hunyuanvl.cpp::build + clip.cpp's position-fill branch):
    patch embed (single conv2d, patch=16, with bias)
        -> + bilinearly-resized position_embd (128x128 grid -> (patchesX,patchesY))
    27x transformer block (plain LayerNorm ViT, no RoPE):
        ln1 (LayerNorm, bias) -> q/k/v (separate, bias) -> MHA (bias-free softmax) -> attn_out (bias)
            -> residual
        ln2 (LayerNorm, bias) -> GELU-tanh FFN (up/down, bias, non-gated) -> residual
    mm.pre_norm (RMSNorm)
    mm.0: real strided Conv2D (2x2, 1152->2304) + bias -> GELU-tanh
    mm.2: real 1x1 Conv2D (2304->4608) + bias  [== plain Linear]
    insert v.image_newline after every row -> (outX+1)*outY tokens
    mm.model.fc: 4608->1024 + bias
    wrap with mm.image_begin (prepend) / mm.image_end (append) -> (outX+1)*outY + 2 tokens
    mm.post_norm (RMSNorm, applied to the whole wrapped sequence)
  -> [n_tokens, 1024]

Test image: a non-native size deliberately chosen so patchesX != patchesY != 128 -- this actually
exercises the bilinear position-embedding resize path (a native 2048x2048 image would coincide
exactly with the stored 128x128 grid and never exercise the resize math at all). 160x128 pixels,
patch=16 -> patchesX=10, patchesY=8 -> merged 5x4 grid -> (5+1)*4+2 = 26 real output tokens.

Usage:
    python scripts/hunyuanvl_ref.py models/mmproj-hunyuanocr-q8_0.gguf [out_dir]

Writes (default out_dir = tests/fixtures/hunyuanvl):
    input_chw.f32, output.f32, meta.json
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 16
IMG_W = 160
IMG_H = 128
MERGE = 2

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "models/mmproj-hunyuanocr-q8_0.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/hunyuanvl"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(mmproj)
    tmap = {t.name: t for t in r.tensors}
    meta = {f.name: f for f in r.fields.values()}

    def meta_float(name, default):
        if name in meta:
            v = meta[name].parts[meta[name].data[0]]
            return float(v[0])
        return default

    eps = meta_float("clip.vision.attention.layer_norm_epsilon", 1e-5)

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

    n_embd = 1152
    n_heads = 16
    head_dim = n_embd // n_heads
    n_layers = 27
    proj_dim = 1024
    mm0_out = 2304
    mm2_out = 4608

    patch_w = raw_data("v.patch_embd.weight")
    assert patch_w.shape == (n_embd, 3, PATCH, PATCH), patch_w.shape
    patch_b = raw_data("v.patch_embd.bias")

    pos_embd = raw_data("v.position_embd.weight")
    n_grid = int(round(np.sqrt(pos_embd.shape[0])))
    assert pos_embd.shape == (n_grid * n_grid, n_embd), pos_embd.shape

    pre_norm_w = raw_data("mm.pre_norm.weight")

    mm0_w = raw_data("mm.0.weight")
    assert mm0_w.shape == (mm0_out, n_embd, MERGE, MERGE), mm0_w.shape
    mm0_b = raw_data("mm.0.bias")

    mm2_w = raw_data("mm.2.weight").reshape(mm2_out, mm0_out)
    mm2_b = raw_data("mm.2.bias")

    image_newline = raw_data("v.image_newline")
    fc_w = raw_data("mm.model.fc.weight")
    assert fc_w.shape == (proj_dim, mm2_out), fc_w.shape
    fc_b = raw_data("mm.model.fc.bias")
    image_begin = raw_data("mm.image_begin")
    image_end = raw_data("mm.image_end")
    post_norm_w = raw_data("mm.post_norm.weight")

    layers = []
    for l in range(n_layers):
        p = f"v.blk.{l}"
        layers.append(dict(
            ln1_w=raw_data(f"{p}.ln1.weight"), ln1_b=raw_data(f"{p}.ln1.bias"),
            q_w=raw_data(f"{p}.attn_q.weight"), q_b=raw_data(f"{p}.attn_q.bias"),
            k_w=raw_data(f"{p}.attn_k.weight"), k_b=raw_data(f"{p}.attn_k.bias"),
            v_w=raw_data(f"{p}.attn_v.weight"), v_b=raw_data(f"{p}.attn_v.bias"),
            o_w=raw_data(f"{p}.attn_out.weight"), o_b=raw_data(f"{p}.attn_out.bias"),
            ln2_w=raw_data(f"{p}.ln2.weight"), ln2_b=raw_data(f"{p}.ln2.bias"),
            up_w=raw_data(f"{p}.ffn_up.weight"), up_b=raw_data(f"{p}.ffn_up.bias"),
            down_w=raw_data(f"{p}.ffn_down.weight"), down_b=raw_data(f"{p}.ffn_down.bias"),
        ))

    def layernorm(x, w, b, eps=eps):
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    def rmsnorm(x, w, eps=eps):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def mha(x, blk):
        n = x.shape[0]
        q = (x @ blk["q_w"].T + blk["q_b"]).reshape(n, n_heads, head_dim)
        k = (x @ blk["k_w"].T + blk["k_b"]).reshape(n, n_heads, head_dim)
        v = (x @ blk["v_w"].T + blk["v_b"]).reshape(n, n_heads, head_dim)
        scale = 1.0 / np.sqrt(head_dim)
        out = np.empty((n, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            scores = (q[:, h] @ k[:, h].T) * scale
            scores = scores - scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p = p / p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h]
        return out.reshape(n, n_heads * head_dim) @ blk["o_w"].T + blk["o_b"]

    # ---- synthetic preprocessed image: CHW [3,H,W], values 0..1, deterministic ----
    H, W = IMG_H, IMG_W
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()

    gx, gy = W // PATCH, H // PATCH
    n_patches = gx * gy

    patches = np.empty((n_patches, n_embd), dtype=np.float32)
    for py in range(gy):
        for pxi in range(gx):
            p = py * gx + pxi
            block = img[:, py*PATCH:(py+1)*PATCH, pxi*PATCH:(pxi+1)*PATCH]
            patches[p] = np.tensordot(patch_w, block, axes=([1, 2, 3], [0, 1, 2])) + patch_b

    # ---- bilinear-resized position embedding add (pixel-center, align_corners=False) ----
    grid = pos_embd.reshape(n_grid, n_grid, n_embd)
    sx = (gx + 0.1) / n_grid
    sy = (gy + 0.1) / n_grid
    for y in range(gy):
        fy = (y + 0.5) / sy - 0.5
        y0 = int(np.clip(np.floor(fy), 0, n_grid - 1))
        y1 = int(np.clip(y0 + 1, 0, n_grid - 1))
        wy1 = float(np.clip(fy - y0, 0.0, 1.0))
        wy0 = 1.0 - wy1
        for xi in range(gx):
            fx = (xi + 0.5) / sx - 0.5
            x0 = int(np.clip(np.floor(fx), 0, n_grid - 1))
            x1 = int(np.clip(x0 + 1, 0, n_grid - 1))
            wx1 = float(np.clip(fx - x0, 0.0, 1.0))
            wx0 = 1.0 - wx1
            interp = (wy0 * wx0 * grid[y0, x0] + wy0 * wx1 * grid[y0, x1] +
                      wy1 * wx0 * grid[y1, x0] + wy1 * wx1 * grid[y1, x1])
            patches[y * gx + xi] += interp

    x = patches
    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("after_pos_embd", x)

    for li, blk in enumerate(layers):
        normed = layernorm(x, blk["ln1_w"], blk["ln1_b"])
        x = x + mha(normed, blk)

        normed2 = layernorm(x, blk["ln2_w"], blk["ln2_b"])
        up = normed2 @ blk["up_w"].T + blk["up_b"]
        ff = gelu_tanh(up) @ blk["down_w"].T + blk["down_b"]
        x = x + ff
        if li == n_layers - 1:
            rec("after_last_block", x)

    x = rmsnorm(x, pre_norm_w)
    rec("after_pre_norm", x)

    downX, downY = gx // MERGE, gy // MERGE
    grid2 = x.reshape(gy, gx, n_embd)
    merged = np.empty((downY * downX, mm0_out), dtype=np.float32)
    for dy0 in range(downY):
        for dx0 in range(downX):
            t = dy0 * downX + dx0
            for o in range(mm0_out):
                s = mm0_b[o]
                for dy in range(MERGE):
                    for dx in range(MERGE):
                        srcY, srcX = dy0 * MERGE + dy, dx0 * MERGE + dx
                        s += np.dot(grid2[srcY, srcX], mm0_w[o, :, dy, dx])
                merged[t, o] = s
    merged = gelu_tanh(merged)
    rec("after_mm0", merged)

    mm2out = merged @ mm2_w.T + mm2_b
    rec("after_mm2", mm2out)

    # ---- insert image_newline after every row ----
    row_tokens = downX + 1
    withnl = np.empty((row_tokens * downY, mm2_out), dtype=np.float32)
    grid3 = mm2out.reshape(downY, downX, mm2_out)
    for row in range(downY):
        withnl[row * row_tokens:row * row_tokens + downX] = grid3[row]
        withnl[row * row_tokens + downX] = image_newline
    rec("after_newline", withnl)

    projected = withnl @ fc_w.T + fc_b
    rec("after_fc", projected)

    n_tokens = projected.shape[0] + 2
    out = np.empty((n_tokens, proj_dim), dtype=np.float32)
    out[0] = image_begin
    out[1:-1] = projected
    out[-1] = image_end

    out = rmsnorm(out, post_norm_w)
    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta_out = dict(patch=PATCH, H=H, W=W, n_tokens=n_tokens, n_embd=proj_dim, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(json.dumps(meta_out, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_tokens},{proj_dim}] to {out_dir}")

if __name__ == "__main__":
    main()
