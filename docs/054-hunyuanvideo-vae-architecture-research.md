# 054 — HunyuanVideo real VAE architecture (research, ready to implement)

Research pass (subagent-assisted, user-authorized exception to the standing no-subagents rule for
this one investigation), following up on [the HunyuanVideo audit](diffusion-samples/README.md)
that found the DiT structurally sound but blocked on a real VAE decoder. This doc is the
implementation spec for that decoder — the next step is a direct C# port, same process as
`WanVaeDecoder3D.cs`.

## Critical: two different HunyuanVideo VAEs exist — use the diffusers one, not the GGML one

`examples/stable-diffusion.cpp/src/model/vae/hunyuan_vae.hpp` is for **HunyuanVideo 1.5** (a
different, newer model — confirmed via `examples/stable-diffusion.cpp/docs/hunyuan_video.md`),
with a different VAE (5-stage `block_out_channels={128,256,512,1024,1024}`, `z_channels=32`,
`spatial_compression_ratio=16`, pixel-shuffle up/downsampling, reuses Wan 2.2's RMSNorm — it's
architecturally a Wan-style VAE wearing a Hunyuan name). **Do not use it as a reference for the
checkpoint already downloaded in this repo.**

The checkpoint actually downloaded (`models/hunyuanvideo/hunyuan_video_720_cfgdistill_fp8_e4m3fn.
safetensors`) is the **original HunyuanVideo** (720p, cfg-distilled DiT). Its real VAE is
`AutoencoderKLHunyuanVideo` in `examples/diffusers/src/diffusers/models/autoencoders/
autoencoder_kl_hunyuan_video.py` — 4-stage `block_out_channels=(128,256,512,512)`,
`latent_channels=16`, GroupNorm (not RMSNorm), nearest-neighbor+conv upsampling (not
pixel-shuffle), `scaling_factor=0.476986`. **No GGML/C++ reference exists for this original
variant anywhere in this repo** — the diffusers Python source is the sole reference. No local VAE
safetensors file has been downloaded yet either (only the DiT).

## Top-level module (`AutoencoderKLHunyuanVideo`)

Config: `in_channels=3`, `out_channels=3`, `latent_channels=16`,
`block_out_channels=(128,256,512,512)`, `layers_per_block=2`, `norm_num_groups=32`,
`act_fn="silu"`, `scaling_factor=0.476986`, `spatial_compression_ratio=8`,
`temporal_compression_ratio=4`, `mid_block_add_attention=True`.

For decode-only: `self.decoder` + `self.post_quant_conv` (`nn.Conv3d(16,16,kernel_size=1)`, plain
conv, NOT causal-padded).

## Latent scaling (real formula, from `pipeline_hunyuan_video.py`)

```python
latents = latents / self.vae.config.scaling_factor   # z = latents / 0.476986
video = self.vae.decode(latents)
```

Single global scalar divisor — **no per-channel `latents_mean`/`latents_std`** (unlike Wan).

## `HunyuanVideoDecoder3D` structure

```
conv_in:       CausalConv3d(16 -> 512, k=3, stride=1)
mid_block:     resnet(512->512) -> [attention -> resnet(512->512)] x1
up_blocks[0]:  3x resnet(512->512), spatial upsample x2, no temporal upsample
up_blocks[1]:  3x resnet(512->512), spatial upsample x2, temporal upsample x2
up_blocks[2]:  3x resnet(512->256, then 256->256 x2), spatial upsample x2, temporal upsample x2
up_blocks[3]:  3x resnet(256->128, then 128->128 x2), no upsample (final block)
conv_norm_out: GroupNorm(32, 128, eps=1e-6)
conv_act:      SiLU
conv_out:      CausalConv3d(128 -> 3, k=3)
```

`layers_per_block=2` but each up-block actually gets `layers_per_block + 1 = 3` resnets. Total
spatial upsample 2³=8 (matches `spatial_compression_ratio`), total temporal upsample 2²=4
(matches `temporal_compression_ratio`) — confirms the per-stage table above.

## `HunyuanVideoCausalConv3d` (causal padding mechanics)

```python
time_causal_padding = (k_w//2, k_w//2, k_h//2, k_h//2, k_t-1, 0)  # (W,W,H,H,T_left,T_right)
x = F.pad(x, time_causal_padding, mode="replicate")   # REPLICATE, not zero-pad
return conv(x)  # padding=0 on the nn.Conv3d itself
```

Spatial (H,W) padding is symmetric `k//2` each side; temporal padding is fully causal
(`k_t - 1` replicated frames prepended, zero trailing) — same causal-video-VAE idiom as Wan, but
**replicate padding, not zero padding**, and this is a straightforward whole-clip conv (no
per-frame recurrent cache state in this reference, unlike the GGML *1.5* variant's streaming
cache — a real, deliberate simplification versus Wan's per-frame threading, not a gap).

## `HunyuanVideoResnetBlockCausal3D`

```
norm1 = GroupNorm(32, in_ch, eps=1e-6)
conv1 = CausalConv3d(in_ch, out_ch, k=3)
norm2 = GroupNorm(32, out_ch, eps=1e-6)
conv2 = CausalConv3d(out_ch, out_ch, k=3)
conv_shortcut = CausalConv3d(in_ch, out_ch, k=1) -- only when in_ch != out_ch

forward: h = conv1(silu(norm1(x))); h = conv2(silu(norm2(h)));
         residual = conv_shortcut(x) if in!=out else x;
         return h + residual
```

## `HunyuanVideoMidBlock3D` (real attention structure)

`resnet0(512->512) -> attn(heads=1, dim_head=512, causal temporal mask) -> resnet1(512->512)`.
Attention is diffusers' generic `Attention` module with `residual_connection=True` (skip built
in), `norm_num_groups=32` (GroupNorm pre-norm inside the Attention module itself),
`upcast_softmax=True`, `_from_deprecated_attn_block=True` (Linear Q/K/V/out, not conv). Applied
over the FULL flattened `(T*H*W)` sequence with a **causal mask across time** (frame i can only
attend to frames <= i, computed via `prepare_causal_attention_mask`). Reshape:
`[b,c,t,h,w] -permute-> [b, t*h*w, c] -attn-> back to [b,c,t,h,w]`.

## `HunyuanVideoUpsampleCausal3D` (real upsample mechanism — nearest, not pixel-shuffle)

```python
first_frame, other_frames = x.split((1, T-1), dim=2)
first_frame = interpolate(first_frame, scale_factor=upsample_factor[1:], mode="nearest")  # SPATIAL ONLY
other_frames = interpolate(other_frames, scale_factor=upsample_factor, mode="nearest")     # full (T,H,W)
x = cat([first_frame, other_frames], dim=2) if T>1 else first_frame
return CausalConv3d(x)  # k=3, stride=1, same causal padding as above
```

Critical detail: the first frame is ALWAYS spatially-only upsampled (never temporally) since
there's no earlier frame to interpolate from — preserves causality. `upsample_factor` is
`(factor_T, factor_H, factor_W)` = `(2,2,2)` when both temporal+spatial, `(1,2,2)` when
spatial-only, per the per-stage table above.

## Real tensor naming (inferred from diffusers attribute names, NOT yet verified against a real
checkpoint — no VAE safetensors downloaded locally yet)

```
decoder.conv_in.conv.weight/.bias
decoder.mid_block.resnets.{0,1}.norm1/conv1.conv/norm2/conv2.conv.weight/.bias
decoder.mid_block.attentions.0.group_norm.weight/.bias
decoder.mid_block.attentions.0.to_q/to_k/to_v.weight/.bias
decoder.mid_block.attentions.0.to_out.0.weight/.bias   (Sequential[Linear, Dropout])
decoder.up_blocks.{0..3}.resnets.{0,1,2}.norm1/conv1.conv/norm2/conv2.conv.weight/.bias
decoder.up_blocks.{0..3}.resnets.0.conv_shortcut.conv.weight/.bias  (only where in!=out: up_blocks.2/.3's resnets.0)
decoder.up_blocks.{0,1,2}.upsamplers.0.conv.conv.weight/.bias   (up_blocks.3 has none)
decoder.conv_norm_out.weight/.bias
decoder.conv_out.conv.weight/.bias
post_quant_conv.weight/.bias   (plain Conv3d, NOT causal-wrapped)
```

Note the double `.conv.conv.` nesting for anything wrapping `HunyuanVideoCausalConv3d` (which
itself has an inner `self.conv` attribute) — same pattern as Wan's `CausalConv3d` wrapping.
**Verify these exact strings against the real checkpoint once a VAE safetensors file is
downloaded** — this naming is inferred from Python module structure, not confirmed byte-for-byte.

## Tiling — skippable for a first port

The VAE supports spatial/temporal tiled decode for memory-constrained large clips
(`tile_sample_min_num_frames=16`, blend-overlap logic similar to Wan's). Not required for
correctness on a modest clip — `use_tiling=False` by default, temporal tiling only triggers past
~4 latent frames. Port the direct whole-clip path first, same as Wan's simple case was validated
before any tiling was considered.

## Real differences from Wan's already-ported VAE (this is NOT a copy-paste)

| Aspect | Wan 2.1 | HunyuanVideo (original) |
|---|---|---|
| Normalization | RMSNorm (`.gamma`) | **GroupNorm(32, eps=1e-6)** |
| Upsample | (see WanVaeDecoder3D.cs) | **Nearest-interpolate + causal conv**, first-frame spatial-only |
| Attention | single-head, no causal mask | **Single-head (heads=1, dim_head=512), causal temporal mask**, residual built into the Attention module |
| Latent scale | per-channel mean/std (16 floats each) | **single scalar 0.476986** |
| Causal pad | zero-pad | **replicate-pad** |
| Channel stages | 384→384→384→192→96 | 512→512→512→256→128 |

## Not yet resolved

- Real state_dict key strings unverified (no local VAE checkpoint to check against).
- Whether the released checkpoint wraps decoder weights under a `vae.` prefix, or ships standalone.
- Output value range / clamping handled in the pipeline's `VideoProcessor`, not the VAE class itself — check separately if pursuing a bit-exact port.
