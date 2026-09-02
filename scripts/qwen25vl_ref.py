#!/usr/bin/env python3
"""
Reference oracle for Qwen2.5-VL's ViT + windowed 4-section M-RoPE + 2x2 spatial-merge MLP
projector (clip.projector_type = "qwen2.5vl_merger").

Faithfully reimplements tools/mtmd/models/qwen2vl.cpp's build() (shared by Qwen2.5-VL and
EXAONE 4.5) plus the real windowing/mask construction in clip.cpp's shared
PROJECTOR_TYPE_QWEN25VL/EXAONE4_5/YOUTUVL switch case, and the same GGML_ROPE_TYPE_VISION math
already derived for GLM4V/Qwen2VL (see Glm4VisionEncoder.ApplyMrope's doc comment). Same pattern
as scripts/glm4v_ref.py / pixtral_ref.py: real numpy port, reading the same local mmproj GGUF the
C# encoder reads.

Windowing is implemented here by computing each token's real window id directly from its spatial
(row,col) merge-tile position rather than by reordering into contiguous blocks -- mathematically
identical to the real reference's reorder-then-mask approach (see
VisionOps.AttentionGqaWindowed's doc comment for why: layer il gets FULL attention only when
(il+1) % n_wa_pattern == 0; every other layer only attends within its own
gridWindow x gridWindow merge-tile window).

Real tensor names confirmed via list-tensors before writing this: dual patch embed
(v.patch_embd.weight + v.patch_embd.weight.1, summed -- fetched but never applied in the prior
C# encoder, now fixed), separate v.blk.N.attn_q/k/v.weight+bias (not fused), v.post_ln.weight
(no bias), mm.0 (5120->5120, GELU-tanh) + mm.2 (5120->3584) projector with biases on both.

Forward (per qwen2vl.cpp::build, RMSNorm ViT since projector_type != "qwen2vl"):
    dual conv2d patch embed (sum) + patch_bias
    32x transformer block:
        ln1 (RMSNorm) -> separate Q/K/V (with bias)
            -> 4-section M-RoPE (windowed variant: same math, window membership only affects mask)
            -> windowed/full MHA (16 heads, head_dim=80) -> attn_out (with bias) -> residual
        ln2 (RMSNorm) -> SwiGLU (gate/up/down, all with bias) -> residual
    post_ln (RMSNorm, no bias)
    2x2 spatial merge (concat 4 adjacent patches -> 4*embd) -> mm.0 (GELU-tanh)+bias -> mm.2+bias
  -> [n_tokens, 3584]

Test image: 224x224 (16x16=256 raw patches, patch=14) so window_size=112 (real default, unset in
this checkpoint's metadata) produces a real 2x2 grid of windows (gridWindow=4 merge-tiles/side,
mergeCols=8 -> 2 windows/side) -- large enough to meaningfully exercise masking, small enough to
stay fast.

Usage:
    python scripts/qwen25vl_ref.py models/mmproj-qwen2.5-vl-7b-f16.gguf [out_dir]

Writes (default out_dir = tests/fixtures/qwen25vl):
    input_chw.f32, output.f32, meta.json
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 14
IMG = 224
MERGE = 2
DEFAULT_WINDOW_SIZE = 112

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "models/mmproj-qwen2.5-vl-7b-f16.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/qwen25vl"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(mmproj)
    tmap = {t.name: t for t in r.tensors}
    meta = {f.name: f for f in r.fields.values()}

    def meta_int(name, default):
        if name in meta:
            v = meta[name].parts[meta[name].data[0]]
            return int(v[0])
        return default

    n_wa_pattern = meta_int("clip.vision.n_wa_pattern", 0)
    window_size = meta_int("clip.vision.window_size", DEFAULT_WINDOW_SIZE)

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

    def vec_opt(name, dim):
        return raw_data(name) if name in tmap else np.zeros(dim, dtype=np.float32)

    n_embd = 1280
    n_heads = 16
    head_dim = n_embd // n_heads
    n_layers = 32
    proj_dim = 3584
    merged_dim = n_embd * 4
    eps = 1e-6

    pe0_w = raw_data("v.patch_embd.weight")
    assert pe0_w.shape == (n_embd, 3, PATCH, PATCH), pe0_w.shape
    pe1_w = raw_data("v.patch_embd.weight.1")
    patch_bias = vec_opt("v.patch_bias", n_embd)
    post_ln_w = raw_data("v.post_ln.weight")

    mm0_w, mm0_b = raw_data("mm.0.weight"), raw_data("mm.0.bias")
    mm2_w, mm2_b = raw_data("mm.2.weight"), raw_data("mm.2.bias")
    assert mm0_w.shape == (merged_dim, merged_dim), mm0_w.shape
    assert mm2_w.shape == (proj_dim, merged_dim), mm2_w.shape

    layers = []
    for l in range(n_layers):
        p = f"v.blk.{l}"
        gate_w = raw_data(f"{p}.ffn_gate.weight")
        layers.append(dict(
            ln1_w=raw_data(f"{p}.ln1.weight"),
            q_w=raw_data(f"{p}.attn_q.weight"), q_b=raw_data(f"{p}.attn_q.bias"),
            k_w=raw_data(f"{p}.attn_k.weight"), k_b=raw_data(f"{p}.attn_k.bias"),
            v_w=raw_data(f"{p}.attn_v.weight"), v_b=raw_data(f"{p}.attn_v.bias"),
            o_w=raw_data(f"{p}.attn_out.weight"), o_b=raw_data(f"{p}.attn_out.bias"),
            ln2_w=raw_data(f"{p}.ln2.weight"),
            gate_w=gate_w, gate_b=raw_data(f"{p}.ffn_gate.bias"),
            up_w=raw_data(f"{p}.ffn_up.weight"), up_b=raw_data(f"{p}.ffn_up.bias"),
            down_w=raw_data(f"{p}.ffn_down.weight"), down_b=raw_data(f"{p}.ffn_down.bias"),
            ffn_dim=gate_w.shape[0],
        ))

    def rmsnorm(x, w, eps=eps):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def silu(x):
        return x / (1.0 + np.exp(-x))

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    half = head_dim // 2
    quarter = half // 2

    def mrope(q, k, px_arr, py_arr):
        d = np.arange(half)
        freq = 10000.0 ** (-4.0 * d / head_dim)
        pos = np.where(d < quarter, py_arr[:, None], px_arr[:, None]).astype(np.float32)
        angle = pos * freq[None, :]
        cos_t, sin_t = np.cos(angle), np.sin(angle)

        def apply(x):
            x0 = x[:, :, :half]
            x1 = x[:, :, half:]
            c = cos_t[:, None, :]
            s = sin_t[:, None, :]
            new0 = x0 * c - x1 * s
            new1 = x0 * s + x1 * c
            return np.concatenate([new0, new1], axis=-1)

        return apply(q), apply(k)

    def mha(x, blk, px_arr, py_arr, window_id):
        n = x.shape[0]
        q = (x @ blk["q_w"].T + blk["q_b"]).reshape(n, n_heads, head_dim)
        k = (x @ blk["k_w"].T + blk["k_b"]).reshape(n, n_heads, head_dim)
        v = (x @ blk["v_w"].T + blk["v_b"]).reshape(n, n_heads, head_dim)
        q, k = mrope(q, k, px_arr, py_arr)
        scale = 1.0 / np.sqrt(head_dim)
        out = np.empty((n, n_heads, head_dim), dtype=np.float32)
        mask = None
        if window_id is not None:
            mask = np.where(window_id[:, None] == window_id[None, :], 0.0, -np.inf).astype(np.float32)
        for h in range(n_heads):
            scores = (q[:, h] @ k[:, h].T) * scale
            if mask is not None:
                scores = scores + mask
            scores = scores - scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p = p / p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h]
        return out.reshape(n, n_embd) @ blk["o_w"].T + blk["o_b"]

    # ---- synthetic preprocessed image: CHW [3,224,224], values 0..1, deterministic ----
    H = W = IMG
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()

    gx = gy = IMG // PATCH
    n_patches = gx * gy

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

    # ---- window id per token (real spatial merge-tile window, gridWindow merge-tiles/side) ----
    use_window = n_wa_pattern > 0
    window_id = None
    if use_window:
        grid_window = max(1, window_size // PATCH // MERGE)
        merge_cols = max(1, gx // MERGE)
        window_cols = (merge_cols + grid_window - 1) // grid_window
        window_id = np.empty(n_patches, dtype=np.int64)
        for py in range(gy):
            wr = (py // MERGE) // grid_window
            for pxi in range(gx):
                wc = (pxi // MERGE) // grid_window
                window_id[py * gx + pxi] = wr * window_cols + wc

    x = patches
    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("after_patch_embd", x)

    for li, blk in enumerate(layers):
        full_attn = (not use_window) or ((li + 1) % n_wa_pattern == 0)
        wid = None if full_attn else window_id

        normed = rmsnorm(x, blk["ln1_w"])
        attn = mha(normed, blk, px_arr, py_arr, wid)
        x = x + attn

        normed2 = rmsnorm(x, blk["ln2_w"])
        gate = silu(normed2 @ blk["gate_w"].T + blk["gate_b"])
        up = normed2 @ blk["up_w"].T + blk["up_b"]
        ff = (gate * up) @ blk["down_w"].T + blk["down_b"]
        x = x + ff
        if li == n_layers - 1:
            rec("after_last_block", x)

    x = rmsnorm(x, post_ln_w)
    rec("after_post_ln", x)

    # ---- 2x2 spatial merge: concat 4 adjacent raw patches -> merged_dim ----
    downX, downY = gx // MERGE, gy // MERGE
    n_tokens = downX * downY
    grid = x.reshape(gy, gx, n_embd)
    merged = np.empty((n_tokens, merged_dim), dtype=np.float32)
    for my in range(downY):
        for mx in range(downX):
            t = my * downX + mx
            sub = 0
            for dy in range(MERGE):
                for dx in range(MERGE):
                    merged[t, sub*n_embd:(sub+1)*n_embd] = grid[my*MERGE+dy, mx*MERGE+dx]
                    sub += 1
    rec("after_merge", merged)

    h = gelu_tanh(merged @ mm0_w.T + mm0_b)
    rec("after_mm0_gelu", h)
    out = h @ mm2_w.T + mm2_b
    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta_out = dict(patch=PATCH, H=H, W=W, gx=downX, gy=downY, n_tokens=n_tokens, n_embd=proj_dim,
                     n_wa_pattern=n_wa_pattern, window_size=window_size, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(json.dumps(meta_out, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_tokens},{proj_dim}] to {out_dir}")

if __name__ == "__main__":
    main()
