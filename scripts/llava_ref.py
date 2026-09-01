#!/usr/bin/env python3
"""
Reference oracle for LLaVA-1.5's CLIP ViT-L/14-336 encoder + 2-layer GELU MLP projector
(clip.vision.projector_type = "mlp").

Faithfully reimplements tools/mtmd/models/llava.cpp's build() (proj_type == PROJECTOR_TYPE_MLP
branch) from llama.cpp using the real mmproj tensors, so the C# LlavaVisionEncoder can be
parity-checked against it -- same pattern as scripts/gemma4uv_ref.py.

Forward (per llava.cpp::build, standard CLIP ViT with select_layer=-2 already baked into the
GGUF's block_count=23, so ALL blocks present are used and there is genuinely no post_ln tensor
in this checkpoint -- confirmed via list-metadata/list-tensors before writing this, not assumed):
    patch_embd: conv2d(patch=14, stride=14, 3->1024), weight ne=[14,14,3,1024] (kx,ky,c,d
                fastest-to-slowest -- GGUF's real conv2d layout, confirmed against the C#
                encoder's own weight-index arithmetic, which already matches this)
    prepend class_embd -> 577 tokens total (1 CLS + 24*24 patches)
    + position_embd (per-token learned, ne=[1024,577])
    pre_ln (LayerNorm, eps=1e-5)
    23x transformer block:
        ln1 -> separate Q/K/V (attn_q/attn_k/attn_v, NOT fused qkv in this checkpoint)
            -> standard MHA (16 heads, head_dim=64, scale=1/sqrt(64)) -> attn_out -> residual
        ln2 -> ffn_up(1024->4096) -> QuickGELU (clip.use_gelu=false) -> ffn_down(4096->1024)
            -> residual
    (no post_ln -- absent from this checkpoint, confirmed)
    strip CLS token -> 576 patch tokens
    MLP projector: mm.0(1024->4096)+bias -> GELU(tanh-approx) -> mm.2(4096->4096)+bias
  -> [576, 4096]

Usage:
    python scripts/llava_ref.py models/mmproj-llava-v1.5-7b-f16.gguf [out_dir]

Writes (default out_dir = tests/fixtures/llava):
    input_chw.f32    raw float32, shape [3,336,336] (synthetic preprocessed image, values 0..1)
    output.f32       raw float32, shape [576,4096] (projector soft tokens)
    meta.json        shapes, dims, per-step stats
"""
import sys, os, json
import numpy as np
from gguf.gguf_reader import GGUFReader

PATCH = 14
IMG = 336
LN_EPS = 1e-5

def main():
    mmproj = sys.argv[1] if len(sys.argv) > 1 else "models/mmproj-llava-v1.5-7b-f16.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/llava"
    os.makedirs(out_dir, exist_ok=True)

    r = GGUFReader(mmproj)
    tmap = {t.name: t for t in r.tensors}

    def raw_data(name):
        """Real tensor data in gguf-py's own natural (already-correctly-shaped) layout --
        ReaderTensor.data is reshaped by gguf-py itself using the reversed dims, i.e. it is
        ALREADY a valid, correctly-ordered numpy array (verified empirically against known
        in/out dims below -- do not re-derive orientation from the raw `ne`/`shape` field,
        which reports dims in a DIFFERENT order than `.data`'s actual numpy shape and led to a
        wrong assumption on the first attempt at this script)."""
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

    def mat_pytorch(name):
        """EVERY 2D weight tensor in this checkpoint (both mm.* and v.blk.*) is stored in
        PyTorch nn.Linear.weight's native [out,in] layout, unmodified by either converter path
        -- verified empirically via data.shape against mm.0 (must be out=4096,in=1024) AND,
        decisively, via bias tensor SIZES for ffn_up/ffn_down (ffn_up.bias is 1024-wide,
        ffn_down.bias is 4096-wide -- the exact OPPOSITE of what the names suggest: in this
        checkpoint "ffn_down" is functionally the FIRST FFN linear (embd->ffn_intermediate) and
        "ffn_up" is the SECOND (ffn_intermediate->embd), a naming swap specific to this
        checkpoint's clip.cpp export, not a general llama.cpp convention -- an earlier attempt
        at this script assumed a split GGML-ready/[in,out] convention for v.blk.* tensors based
        on a wrong guess at which name meant which direction; the bias sizes settle it).
        Returns (out,in); caller does y = x @ W.T."""
        return raw_data(name)

    def vec(name):
        return raw_data(name)

    def vec_opt(name, dim):
        return raw_data(name) if name in tmap else np.zeros(dim, dtype=np.float32)

    n_embd = 1024
    n_heads = 16
    head_dim = n_embd // n_heads
    n_layers = 23
    ffn_dim = 4096
    proj_dim = 4096

    # patch_embd conv weight: gguf-py's own reshape gives natural shape (d,c,ky,kx) directly
    # (verified: raw_data(...).shape == (1024,3,14,14) == (n_embd,3,PATCH,PATCH))
    pe_w = raw_data("v.patch_embd.weight")
    assert pe_w.shape == (n_embd, 3, PATCH, PATCH), pe_w.shape
    pe_b = vec_opt("v.patch_embd.bias", n_embd)
    cls_embd = vec("v.class_embd")
    # position_embd: natural shape (577, 1024) == (n_pos, n_embd), used as-is (embedding table)
    pos_embd = vec("v.position_embd.weight")
    assert pos_embd.shape == (577, n_embd), pos_embd.shape
    pre_ln_w, pre_ln_b = vec("v.pre_ln.weight"), vec("v.pre_ln.bias")

    # mm.* (llava's own projector): PyTorch-native [out,in] -- y = x @ W.T
    mm0_w, mm0_b = mat_pytorch("mm.0.weight"), vec("mm.0.bias")
    mm2_w, mm2_b = mat_pytorch("mm.2.weight"), vec("mm.2.bias")
    assert mm0_w.shape == (4096, n_embd), mm0_w.shape
    assert mm2_w.shape == (4096, 4096), mm2_w.shape

    # v.blk.* (CLIP ViT): also PyTorch-native [out,in] -- y = x @ W.T. NOTE the ffn tensor-name
    # swap documented on mat_pytorch above: "ffn1" below reads the GGUF's "ffn_down" tensor
    # (the REAL first/expanding linear, embd->ffn_intermediate) and "ffn2" reads "ffn_up" (the
    # REAL second/contracting linear, ffn_intermediate->embd) -- named by FUNCTION here, not by
    # the GGUF's own (misleading, for this checkpoint) tensor names.
    layers = []
    for l in range(n_layers):
        p = f"v.blk.{l}"
        layers.append(dict(
            ln1_w=vec(f"{p}.ln1.weight"), ln1_b=vec(f"{p}.ln1.bias"),
            q_w=mat_pytorch(f"{p}.attn_q.weight"), q_b=vec(f"{p}.attn_q.bias"),
            k_w=mat_pytorch(f"{p}.attn_k.weight"), k_b=vec(f"{p}.attn_k.bias"),
            v_w=mat_pytorch(f"{p}.attn_v.weight"), v_b=vec(f"{p}.attn_v.bias"),
            o_w=mat_pytorch(f"{p}.attn_out.weight"), o_b=vec(f"{p}.attn_out.bias"),
            ln2_w=vec(f"{p}.ln2.weight"), ln2_b=vec(f"{p}.ln2.bias"),
            ffn1_w=mat_pytorch(f"{p}.ffn_down.weight"), ffn1_b=vec(f"{p}.ffn_down.bias"),
            ffn2_w=mat_pytorch(f"{p}.ffn_up.weight"), ffn2_b=vec(f"{p}.ffn_up.bias"),
        ))
        assert layers[-1]["q_w"].shape == (n_embd, n_embd), layers[-1]["q_w"].shape
        assert layers[-1]["ffn1_w"].shape == (ffn_dim, n_embd), layers[-1]["ffn1_w"].shape
        assert layers[-1]["ffn2_w"].shape == (n_embd, ffn_dim), layers[-1]["ffn2_w"].shape
        assert layers[-1]["ffn1_b"].shape == (ffn_dim,), layers[-1]["ffn1_b"].shape
        assert layers[-1]["ffn2_b"].shape == (n_embd,), layers[-1]["ffn2_b"].shape

    def layernorm(x, w, b, eps=LN_EPS):
        m = x.mean(-1, keepdims=True)
        d = x - m
        v = (d * d).mean(-1, keepdims=True)
        return d / np.sqrt(v + eps) * w + b

    def quick_gelu(x):
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def mha(x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b):
        # q_w/k_w/v_w/o_w are PyTorch-native [out,in] -- x @ W.T.
        n = x.shape[0]
        q = (x @ q_w.T + q_b).reshape(n, n_heads, head_dim)
        k = (x @ k_w.T + k_b).reshape(n, n_heads, head_dim)
        v = (x @ v_w.T + v_b).reshape(n, n_heads, head_dim)
        scale = 1.0 / np.sqrt(head_dim)
        out = np.empty((n, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            scores = (q[:, h] @ k[:, h].T) * scale  # (n,n)
            scores = scores - scores.max(-1, keepdims=True)
            p = np.exp(scores)
            p = p / p.sum(-1, keepdims=True)
            out[:, h] = p @ v[:, h]
        out = out.reshape(n, n_embd)
        return out @ o_w.T + o_b

    # ---- synthetic preprocessed image: CHW [3,336,336], values 0..1, deterministic ----
    H = W = IMG
    c_idx = np.arange(3).reshape(3, 1, 1)
    y_idx = np.arange(H).reshape(1, H, 1)
    x_idx = np.arange(W).reshape(1, 1, W)
    img = ((np.sin(x_idx * 0.05 + c_idx) * np.cos(y_idx * 0.04 + c_idx) + 1.0) * 0.5).astype(np.float32)
    img = np.broadcast_to(img, (3, H, W)).copy()

    gx = gy = IMG // PATCH
    n_patches = gx * gy

    # ---- patch embed (conv2d as im2col + matmul) ----
    patches = np.empty((n_patches, n_embd), dtype=np.float32)
    for py in range(gy):
        for px in range(gx):
            p = py * gx + px
            block = img[:, py*PATCH:(py+1)*PATCH, px*PATCH:(px+1)*PATCH]  # (3,14,14) c,ky,kx
            # pe_w is (d, c, ky, kx); sum over c,ky,kx
            patches[p] = np.tensordot(pe_w, block, axes=([1, 2, 3], [0, 1, 2])) + pe_b

    x = np.concatenate([cls_embd[None, :], patches], axis=0)  # (577,1024)
    x = x + pos_embd  # position_embd covers all 577 tokens (CLS + patches)

    stats = {}
    def rec(tag, a): stats[tag] = [float(a.mean()), float(a.std()), float(a.min()), float(a.max())]
    rec("after_patch_and_pos", x)

    x = layernorm(x, pre_ln_w, pre_ln_b)
    rec("after_pre_ln", x)

    for li, blk in enumerate(layers):
        normed = layernorm(x, blk["ln1_w"], blk["ln1_b"])
        attn = mha(normed, blk["q_w"], blk["q_b"], blk["k_w"], blk["k_b"],
                   blk["v_w"], blk["v_b"], blk["o_w"], blk["o_b"])
        x = x + attn

        normed2 = layernorm(x, blk["ln2_w"], blk["ln2_b"])
        ff = normed2 @ blk["ffn1_w"].T + blk["ffn1_b"]
        ff = quick_gelu(ff)
        ff = ff @ blk["ffn2_w"].T + blk["ffn2_b"]
        x = x + ff
        if li == n_layers - 1:
            rec("after_last_block", x)

    # strip CLS, project
    patch_tokens = x[1:]  # (576,1024)
    rec("patch_tokens", patch_tokens)

    h = patch_tokens @ mm0_w.T + mm0_b
    h = gelu_tanh(h)
    rec("after_mm0_gelu", h)
    out = h @ mm2_w.T + mm2_b
    rec("output", out)

    img.tofile(os.path.join(out_dir, "input_chw.f32"))
    out.astype(np.float32).tofile(os.path.join(out_dir, "output.f32"))
    meta = dict(patch=PATCH, H=H, W=W, gx=gx, gy=gy, n_tokens=n_patches, n_embd=proj_dim,
                ln_eps=LN_EPS, stats=stats)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nWrote input_chw.f32 [3,{H},{W}], output.f32 [{n_patches},{proj_dim}] to {out_dir}")

if __name__ == "__main__":
    main()
