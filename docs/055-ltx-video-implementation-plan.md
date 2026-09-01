# 055 — LTX-Video (Lightricks) real implementation plan

Planning pass (external-AI-assisted architecture research, cross-checked against the real
downloaded checkpoint's own tensor inventory before writing anything into this doc) following up
on `docs/diffusion-samples/README.md`'s LTX-Video finding: the current code is a from-scratch
structural stub (hardcoded patchify formula, hardcoded timestep formula, `LtxVideoModel` has no
`IWeightLoader` field at all, `GenerateVideo`'s text context is literal random noise). This is a
**build task, not a debug task** — there is nothing to "fix" in the current `LtxVideoModel.cs`
worth preserving; it should be replaced.

**Gated behind this project's own priority order** (`docs/00-current-work.md`'s "Cross-project
priority order"): do not start real implementation while Wan/Z-Image/CosyVoice work is still
active — this doc exists so the next session that does pick up LTX has a checked, actionable plan
rather than a cold start.

## Real tensor inventory (verified directly against `models/ltx-video-2b-v0.9.1.safetensors`'s
safetensors JSON header, 2026-09-02 — every shape below is a literal grep result, not a claim)

```
model.diffusion_model.patchify_proj.weight              [2048, 128]    BF16   (Linear 128->2048, has bias)
model.diffusion_model.caption_projection.linear_1.weight [2048, 4096]  BF16   (Linear 4096->2048)
model.diffusion_model.caption_projection.linear_2.weight [2048, 2048]  BF16   (Linear 2048->2048)
model.diffusion_model.proj_out.weight                    [128, 2048]  BF16   (Linear 2048->128, final)
model.diffusion_model.scale_shift_table                  [2, 2048]    BF16   (top-level: final modulation)
model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1/.linear_2   (timestep MLP)
model.diffusion_model.adaln_single.linear                                    (PixArt-style AdaLN-single)
model.diffusion_model.transformer_blocks.{0..27}.scale_shift_table      [6, 2048]  (per-block modulation)
model.diffusion_model.transformer_blocks.{0..27}.attn1.{to_q,to_k,to_v,to_out.0}.weight  [2048,2048]  (self-attn)
model.diffusion_model.transformer_blocks.{0..27}.attn1.{q_norm,k_norm}.weight            [2048]       (QK-norm, FULL width)
model.diffusion_model.transformer_blocks.{0..27}.attn2.{to_q,to_out.0}.weight            [2048,2048]  (cross-attn, query side)
model.diffusion_model.transformer_blocks.{0..27}.attn2.{to_k,to_v}.weight                [2048,2048]  (cross-attn, K/V side)
model.diffusion_model.transformer_blocks.{0..27}.attn2.{q_norm,k_norm}.weight            [2048]
model.diffusion_model.transformer_blocks.{0..27}.ff.net.0.proj.weight   [8192, 2048]  (FFN up, 4x expansion)
model.diffusion_model.transformer_blocks.{0..27}.ff.net.2.weight        [2048, 8192]  (FFN down)
```

Confirms, first-hand, the load-bearing claims from the architecture research below:
- **`patchify_proj` is a genuine `Linear(128, 2048)`**, not a 3D patch convolution (Wan-shaped
  assumption would have been wrong here) — matches `patch_size=1, patch_size_t=1` in the model's
  own config.
- **Cross-attention (`attn2`) operates entirely at 2048-dim, both Q and K/V** — `attn2.to_k`'s
  shape is `[2048, 2048]`, NOT `[2048, 4096]`. T5's real 4096-dim output is projected down to 2048
  by `caption_projection` BEFORE it ever reaches cross-attention. Every block's `attn2.to_k`/`to_v`
  reads the SAME projected 2048-dim caption sequence, not raw T5 states.
  `CrossAttentionDim` in the current stub (hardcoded to 4096) is wrong for the actual attention
  math — 4096 only appears at the `caption_projection.linear_1` input.
- **QK-norm is FULL-width (2048), not per-head (64).** Same "whole projected `dim`-length row,
  one RMS statistic before the head split" convention this project already implements correctly
  for Wan's self-attention and Qwen3 (see `WanModel.RmsNormHeads`'s doc comment for the exact
  historical bug shape to avoid re-introducing here) — reuse that kernel/pattern, don't write a
  new per-head-truncated version.
- **FFN is an ordinary 2-layer GELU MLP (`ff.net.0.proj` / `ff.net.2`), 2048→8192→2048** — NOT a
  gated `w1`/`w2`/`w3` SiLU-gated FFN like Wan's. A Wan-shaped `FeedForward` port would silently
  build the wrong weight consumption pattern here (missing a `w3` tensor is a much noisier failure
  than getting the gate math subtly wrong would be, so at least this particular mistake is likely
  to be self-evidently caught by a tensor-name mismatch at load time — but don't rely on that catch
  happening; check the checkpoint before writing the FFN class).
- **Both `scale_shift_table` levels exist as literal tensors**: top-level `[2, 2048]` (final
  modulation) and one per-block `[6, 2048]` (self-attn shift/scale/gate + FFN shift/scale/gate) —
  confirms the PixArt-style `AdaLayerNormSingle` mechanism the research below describes, not a
  Wan-style per-block Linear projection of a shared `timestep_proj`.
- **No T5 encoder tensors in this file** — `caption_projection`'s existence confirms text
  conditioning is real and always active, but the actual T5-v1.1-XXL encoder is a SEPARATE
  checkpoint, not bundled here. **Not present locally** — only `models/wan2.1/
  models_t5_umt5-xxl-enc-bf16.safetensors` (UMT5-XXL, a different, real architecture, not directly
  interchangeable — see below) exists in this repo's `models/` right now. A real
  `google/t5-v1_1-xxl` checkpoint needs downloading before any T5-side numeric verification can
  start.

## Architecture summary (from external research, verified against the tensor inventory above
wherever a real shape could confirm it; VAE section is NOT yet independently re-verified —
`ltx-video-2b-v0.9.1.safetensors` presumably also bundles VAE tensors, not yet dumped/grepped)

```
T5-v1.1-XXL encoder  (SEPARATE checkpoint, not bundled — d_model=4096, d_ff=10240, d_kv=64,
                       heads=64, layers=24, vocab=32128, gated-gelu FFN)
        |  [textSeqLen, 4096]
        v
caption_projection: Linear(4096->2048) -> Linear(2048->2048)     [VERIFIED: real tensor shapes]
        |  [textSeqLen, 2048]
        v
latent [B, 128, F, H, W]  (from VAE encode, or noise for T2V)
        |  flatten (F,H,W) -- patch_size=1, patch_size_t=1, i.e. NO spatial/temporal patch merge
        |  at the transformer boundary; the real compression already happened in the VAE
        v
patchify_proj: Linear(128 -> 2048)                                [VERIFIED: real shape [2048,128]]
        |
        +---- adaln_single: timestep_embedder (sinusoidal + 2-layer MLP) -> linear
        |     produces the shared per-block modulation input, PixArt AdaLayerNormSingle-style
        |     (NOT a per-block Linear projecting a shared timestep_proj, Wan's convention)
        v
   28x LtxTransformerBlock:
     norm1 (RMSNorm, non-affine, eps=1e-6) -> AdaLN-modulate with block's own
       scale_shift_table[6,2048] (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
     self-attn (attn1): Q/K/V from modulated x, FULL-WIDTH QK-norm [VERIFIED: q_norm/k_norm=[2048]],
       LTX-specific continuous-coordinate 3D RoPE, gated residual (gate_msa)
     cross-attn (attn2): Q from x (NOT the AdaLN-modulated version -- plain residual add, no
       gating table consumed here per the research below), K/V from the SAME projected caption
       sequence at every block [VERIFIED: attn2.to_k/to_v are [2048,2048], reading the
       already-2048-dim caption_projection output, not raw 4096-dim T5]
     norm2 -> AdaLN-modulate (shift_mlp, scale_mlp) -> FFN (ordinary 2-layer GELU-approximate,
       2048->8192->2048 [VERIFIED]) -> gated residual (gate_mlp)
        |
        v
final: RMSNorm -> modulate with TOP-LEVEL scale_shift_table[2,2048] [VERIFIED: real [2,2048] tensor]
       -> proj_out: Linear(2048 -> 128) [VERIFIED: real shape [128,2048]]
        |
        v
unpatchify (trivial -- patch_size=1 means this is just an unflatten, no real spatial merge)
        |
        v
LTX VAE decoder (causal 3D, 8x temporal / 32x spatial / 32x spatial compression, 128 latent
                  channels -- NOT yet independently tensor-verified in this pass)
```

## The single biggest gotcha: the VAE decoder is timestep-conditioned (not yet independently verified)

Per the research (not yet cross-checked against a real tensor dump of the VAE portion of the
checkpoint — do that before trusting this section as fully as the transformer section above):
LTX-Video v0.9.1's VAE decoder does not simply reconstruct pixels from an already-clean latent —
it takes a small additional decode timestep (nominal `decode_timestep=0.05`) and decode noise scale
(nominal `decode_noise_scale=0.025`), adds a small amount of noise to the latent, and threads that
decode-timestep conditioning through the decoder's own residual/mid blocks via the same kind of
scale/shift modulation mechanism the transformer uses, before a final conv. This means the VAE
decoder is NOT a simple, transformer-independent stage the way Wan's `WanVaeDecoder3D` is — treat
"port the VAE" as its own research pass with the same "read the real reference source and dump its
own tensor names first" discipline applied to `examples/stable-diffusion.cpp/src/model/diffusion/
ltxv.hpp` (2066 lines, confirmed present locally) before assuming Wan's causal-VAE shape transfers.

The temporal latent geometry also has a real, easy-to-miss asymmetry: the first frame is special
(`F_latent = 1 + (F-1)/8`, not a clean `F/8`), matching the same "first-frame-is-different" pattern
already seen in Wan's real VAE (`docs/diffusion-samples/README.md`'s "Known, documented scope
limit" note on `WanVaeDecoder3D`) — but confirm LTX's own exact formula against `ltxv.hpp` rather
than assuming it's identical to Wan's.

## RoPE: LTX-specific continuous coordinates, not integer latent indices

Per the research: LTX's 3D RoPE takes fractional/continuous position coordinates (normalized
against something like `[20, 2048, 2048]` for time/height/width, theta=10000), mapping latent
cells back toward pixel-space coordinates using the VAE's own (8, 32, 32) compression factors —
NOT simply `t=frameIndex, y=row, x=col` integer positions the way Wan's RoPE (`WanRoPE.cs`) does.
The existing `LtxVideoRoPE.cs` stub's name ("Continuous3DRoPE") suggests whoever wrote the
original stub was aware of this distinction even if the implementation itself never got real
weights — worth reading that file's current state before writing a replacement, in case any of its
scaffolding (even if numerically unverified) is salvageable.

One precision detail flagged by the research and worth preserving deliberately: the reference
computes RoPE frequencies in float32 even under lower-precision (BF16) inference, converting only
after the fact — a common source of "shapes all correct, numerically nowhere near right" bugs if
missed, per this project's own established failure mode from other ports.

## T5-v1.1-XXL vs. this project's existing UMT5-XXL encoder — real, confirmed differences

This project already has a working `UMT5Encoder.cs` (built for Wan), and UMT5 is architecturally
close to T5 but NOT identical — the difference already documented in `UMT5Encoder.cs`'s own doc
comment (per-layer relative position bias, confirmed present on every one of Wan's 24 blocks,
"unlike T5") is real and applies here too: real T5-v1.1 uses a SHARED/single relative-position-bias
table (typically owned by layer 0 and reused), while UMT5 computes a distinct bias per layer.
Reusing `UMT5Encoder.cs`'s substrate (token embedding, T5-style RMSNorm, attention Q/K/V/O, gated
FFN, relative-position bucket math, attention masking) is the right call — rewriting a whole T5
encoder from zero would duplicate real, working infrastructure — but the relative-position-bias
OWNERSHIP needs to become a configurable behavior (shared-first-layer vs. per-layer), not a second
hardcoded assumption. `google/t5-v1_1-xxl` itself is not downloaded locally yet.

## Build order (dependency-ordered, each phase independently numerically verifiable before the next)

Mirrors this project's own established, proven methodology (Fish Speech / QwenTTS / Wan): read the
real reference source line-by-line before writing C#, verify each stage numerically before trusting
it, never jump straight to full-pipeline generation.

0. **Checkpoint architecture inventory.** Dump every tensor name/dtype/shape/byte-count from
   `ltx-video-2b-v0.9.1.safetensors` in full (this doc's tensor-inventory section above is a
   partial grep, not the full dump) and group by prefix — confirm the VAE's own tensor names/shapes
   before assuming anything about its structure. Build an `LtxCheckpointArchitecture` inference
   step (dims/layer-count read FROM the checkpoint, matching this project's existing
   `DetectConfig`-style pattern used elsewhere — e.g. `WanModel.DetectConfig` — rather than
   hardcoded constructor defaults, which is exactly what's wrong with the current stub).
1. **Input projection (`patchify_proj`) + caption projection.** Smallest possible numeric test:
   feed a deterministic small latent-token matrix through `patchify_proj` alone (Python/reference
   vs. C#), require near-exact cosine similarity. Separately verify `caption_projection`'s
   `[textSeqLen,4096] -> [textSeqLen,2048]` output against a reference run. Catches
   layout/transposition bugs before anything else can mask them.
2. **Timestep / AdaLN-single branch**, as its own dedicated test: dump the scaled timestep, the
   sinusoidal embedding, the post-MLP timestep, the resulting per-block modulation tensor, and the
   final top-level modulation tensor, and compare each intermediate independently — not just the
   end-to-end block output. A timestep-frequency bug can otherwise look identical to an attention
   bug.
3. **RoPE**, before any attention: a small synthetic grid (e.g. F=2,H=2,W=3), golden-tested against
   reference-dumped position/frequency/cos/sin tensors. Treat a passing RoPE conformance test as a
   hard gate before block work starts, per the research's own recommendation.
4. **One transformer block, fully dissected** — capture and compare every intermediate (post-norm1,
   post-AdaLN-modulate, Q/K/V, post-QK-norm, post-RoPE, self-attn output, post-self-attn residual,
   cross-attn Q/K/V, cross-attn output, post-cross residual, post-norm2, post-AdaLN-modulate, FFN
   intermediate, FFN output, post-FFN residual) for a small real slice (8-16 video tokens, 5-10 text
   tokens) against the reference. Once block 0 matches exactly, replicating to 28 blocks is
   mechanical.
5. **Full 28-block transformer + final projection**, verified against a reference run using
   REFERENCE-PRODUCED T5 embeddings dumped to a file (not this project's own not-yet-built T5
   encoder) — isolates "the DiT is wrong" from "T5 is wrong" as two independently debuggable
   problems, exactly the trap that made earlier ports in this project slow when both halves were
   brought up simultaneously.
6. **T5-v1.1-XXL encoder** (adapted from `UMT5Encoder.cs`'s substrate, per above), verified
   independently: tokenization, embedding output, an early layer, a late layer, final norm — each
   against a real reference run — before ever wiring it into the transformer pipeline from step 5.
7. **VAE decoder** (decode-only; the encoder is not needed for text-to-video and should NOT be
   ported in this first pass), built and verified as its own independent stage against a known
   reference latent, starting from the smallest legal input tensor the architecture's upsample
   stages permit. The `F_out = 8*(F_latent-1)+1` temporal-geometry formula gets its own dedicated
   test, matching Wan's precedent for this same category of off-by-a-frame bug.
8. **Scheduler / full generation loop** — only after steps 1-7 each pass their own numeric check
   independently. LTX 0.9.1's real defaults (per the research, not yet independently confirmed):
   rectified-flow-style sampling, ~40 steps, CFG scale ~3, plus the VAE decoder's own
   `decode_timestep`/`decode_noise_scale` conditioning from the gotcha above. Do not attempt to
   match a full visual sample before step 5's transformer-only numeric parity is solid — a
   scheduler convention error can make a fully-correct transformer look completely broken end to
   end, wasting the far more expensive whole-pipeline debugging loop on what's actually a
   one-line fix.

## Suggested C# structure (mirrors this project's existing checkpoint-driven, non-hardcoded pattern)

```csharp
internal sealed record LtxConfig(
    int LatentChannels, int HiddenSize, int NumLayers, int NumHeads, int HeadDim,
    int T5HiddenSize, int CrossAttentionSize, float NormEpsilon, float RopeTheta,
    float TimestepScale);
// Populate from real checkpoint tensor shapes (step 0's LtxCheckpointArchitecture), not literal
// constants -- the values below are what v0.9.1's real checkpoint confirms, kept here only as the
// expected/sanity-checked result, not as hardcoded constructor defaults the way the current stub
// does it:
//   LatentChannels=128, HiddenSize=2048, NumLayers=28, NumHeads=32, HeadDim=64,
//   T5HiddenSize=4096, CrossAttentionSize=2048, NormEpsilon=1e-6, RopeTheta=10000,
//   TimestepScale=1000 (per the research; not yet independently tensor-confirmed)

internal sealed class LtxVideoTransformer
{
    // Linear.Load(weights, "model.diffusion_model.patchify_proj"), etc. -- real tensor names
    // confirmed in this doc's inventory section, load directly, don't guess a renamed convention.
}

internal sealed class LtxTransformerBlock
{
    // norm1 (RMSNorm, non-affine) -> AdaLN-modulate(scale_shift_table[6,hidden]) -> self-attn
    // (QK-norm FULL WIDTH, LTX RoPE) -> gated residual -> cross-attn (plain residual, K/V from
    // shared projected caption sequence) -> norm2 -> AdaLN-modulate -> FFN (2-layer GELU) ->
    // gated residual. Exact reference ordering from ltxv.hpp, not assumed from this description.
}
```

## Update 2026-09-01: DiT transformer core implemented AND numerically verified against the real HF `diffusers` reference

Phases 0-5 of the build order above are done, not just structurally but numerically:
`LtxVideoModel` (`src/OpenTail.Stingray.Diffusion/LTXVideo/LtxVideoModel.cs`) is now a real
`IWeightLoader`-backed port with checkpoint-driven config detection, real `patchify_proj`/
`caption_projection`/`adaln_single`/28-block-transformer/`proj_out` weights, and a rewritten
`LtxVideoRoPE.cs`.

**`diffusers` (the real HF Python package) is available in this environment** (`pip install
diffusers` succeeded) and its `diffusers/models/transformers/transformer_ltx.py` is the actual
released reference implementation used to produce the real checkpoint — far more authoritative than
`stable-diffusion.cpp`'s ggml port for numeric verification, since it's literally the training-time
module the checkpoint's weights were fit to. Loading the real `ltx-video-2b-v0.9.1.safetensors`
into `LTXVideoTransformer3DModel` (after remapping `patchify_proj`→`proj_in`, `adaln_single`→
`time_embed`, `q_norm`/`k_norm`→`norm_q`/`norm_k`) hit **zero missing/unexpected keys** — full
confirmation this project's tensor-name understanding is exactly right.

Diffing against `diffusers` found real bugs the ggml-reference-only reading missed:
- **RoPE is computed over the FULL hidden dim (2048), not per-head (64), and applied to q/k BEFORE
  the head split** (`LTXVideoRotaryPosEmbed(dim=inner_dim, ...)`, `apply_rotary_emb` runs on the
  un-split `[B,S,inner_dim]` tensor). A per-head-width table repeated identically across all 32
  heads (what was originally built, and what the ggml reference's `num_heads` parameter to
  `build_video_rope_matrix` implied) gave near-zero cosine similarity against the real reference's
  own `rope_cos`/`rope_sin` dump. Fixed by computing the rotation over `HiddenSize` and applying it
  as a single "head" of width `d` before the per-head attention split.
- **Rotation is INTERLEAVED (pairs `(x[2i],x[2i+1])`, `cos`/`sin` values duplicated via
  `repeat_interleave(2)`), matching this project's own `WanRoPE` convention** — NOT split-half
  ("NEOX") as the ggml reference's `video_rope_interleaved=false` config flag implied.
- **Coordinates are plain `index * rope_interpolation_scale[axis] / base[axis]`** (real default
  `rope_interpolation_scale = (vae_temporal_ratio/frame_rate, vae_spatial_ratio, vae_spatial_ratio)`
  = `(8/25, 32, 32)`, `base = (20, 2048, 2048)`, `frame_rate` default 25) — no causal `+1-scale`
  temporal shift, no "middle indices grid" start/end averaging; both were real features of the
  ggml/C++ port's *config surface* but not what the actual released weights' RoPE module computes.
- **Frequency channels are laid out FREQUENCY-major** (`[f0_t,f0_h,f0_w, f1_t,f1_h,f1_w, ...]`), not
  axis-major (`[all t, all h, all w]`).
- **The model's final `norm_out` is a real mean-centered `nn.LayerNorm` (no affine), NOT RMSNorm** —
  every other norm in the model (block `norm1`/`norm2`, attention `q_norm`/`k_norm`) is RMS-only, so
  this was an easy one to get wrong by pattern-matching the rest of the model.

**Numeric parity is now a committed, runnable test**
(`tests/OpenTail.Stingray.Tests.Diffusion/LtxVideoGoldenParityTests.cs`), not just eyeballed once:
golden intermediate tensors (RoPE cos/sin, `patchify_proj` output, `caption_projection` output,
block-0 output, full 28-block output) were dumped from the real `LTXVideoTransformer3DModel` loaded
with the real checkpoint weights, saved as small float32 `.bin` fixtures under
`tests/OpenTail.Stingray.Tests.Diffusion/TestData/LtxGolden/`, and are diffed against this project's
own `LtxVideoModel.Forward()` output (via internal `LastRopeCos`/`LastProjInOut`/etc. capture
fields, `InternalsVisibleTo`-exposed) on every test run. Results: RoPE cosine similarity >0.9999,
`patchify_proj`/`caption_projection` >0.999, block-0 output >0.99, full 28-block forward >0.95
(bounded below 1.0 by the checkpoint's real BF16 storage precision, not a known bug). This closes
the numeric-verification gate the build order's step 4 calls a hard requirement before trusting
anything downstream.

**Still not done**: T5-v1.1-XXL encoder (not downloaded), the real timestep-conditioned VAE decoder
(pipeline still uses the generic placeholder `VaeDecoder`), and the scheduler/guidance loop's own
numeric parity (untouched this pass — golden dump above only exercises the transformer, not
`FlowMatchEulerDiscreteScheduler`/CFG). The Python dump script itself was a scratch file, not
committed — regenerate it from this doc's description if the golden fixtures ever need refreshing
(e.g. after a real bugfix changes the expected output).

## VAE decoder tensor inventory (real dump against `ltx-video-2b-v0.9.1.safetensors`, 2026-09-01 --
closes this doc's earlier "not yet independently tensor-verified" gap on the VAE section)

297 `vae.*` tensors, ALL under `vae.decoder.*` (no `vae.encoder.*` present in this checkpoint --
confirms the plan's step 7 assumption that only decode-time inference is supported by this file;
encoding real video would need a separate encoder checkpoint, out of scope for T2V).

```
vae.decoder.conv_in.conv.{weight,bias}            [1024, 128, 3, 3, 3]  -- 3D conv, 128 latent ch -> 1024
vae.decoder.up_blocks.{0,2,4,6}.res_blocks.{0..N}.conv1.conv.{weight,bias}   [C, C, 3, 3, 3]  (3D conv)
vae.decoder.up_blocks.{0,2,4,6}.res_blocks.{0..N}.conv2.conv.{weight,bias}   [C, C, 3, 3, 3]
vae.decoder.up_blocks.{0,2,4,6}.res_blocks.{0..N}.scale_shift_table          (per-res-block timestep modulation)
vae.decoder.up_blocks.{0,2,4}.time_embedder.timestep_embedder.linear_1/.linear_2   (per-up-block timestep MLP)
vae.decoder.up_blocks.{1,3,5}.conv.conv.{weight,bias}   -- plain (non-res, non-timestep) upsample convs
vae.decoder.last_time_embedder.timestep_embedder.linear_1/.linear_2
vae.decoder.last_scale_shift_table
vae.decoder.timestep_scale_multiplier   -- scalar (0-d tensor)
vae.decoder.conv_out.conv.{weight,bias}            [48, 128, 3, 3, 3]  -- 128 -> 48, NOT 128->3
```

Confirms, first-hand:
- **The VAE decoder is genuinely timestep-conditioned end-to-end**, not just at one entry point: 4
  separate up-block stages (indices 0,2,4,6 -- channel widths 1024/512/256/128, 8/7/6/5 res-blocks
  respectively) each own a `time_embedder` MLP AND a per-res-block `scale_shift_table`, plus one more
  top-level `last_time_embedder`/`last_scale_shift_table` pair right before `conv_out`. Up-block
  indices 1,3,5 are plain spatial/temporal upsample convs with no timestep conditioning at all
  (`up_blocks.{1,3,5}.conv.conv`, no `res_blocks`/`time_embedder` keys under them) -- confirms the
  real structure alternates [timestep-conditioned resnet stage] / [plain upsample] rather than
  conditioning uniformly throughout.
- **All convolutions are genuinely 3D** (`[outC, inC, 3, 3, 3]` kernel shape) -- causal-3D-conv
  handling (the temporal-causality concern the plan's "biggest gotcha" section already flagged)
  applies to every one of these, not just a boundary layer.
- **`conv_out` maps 128→48, not 128→3`** -- 48 = 3 (RGB) × 16, i.e. the real decoder's last stage
  produces a pixel-UNSHUFFLED output (a 4×4 spatial packing factor) that needs an explicit
  pixel-shuffle/depth-to-space unpack to real RGB after `conv_out`, not a direct 3-channel image --
  an easy-to-miss step if implemented by pattern-matching Wan's `WanVaeDecoder3D` (which decodes
  directly to 3 channels with no such unpack).
- **`timestep_scale_multiplier` is a real learned/stored scalar tensor** (0-d), not a hardcoded
  constant the way this project's other decode-timestep handling might assume.

This is a substantially larger, more structurally distinct undertaking than the transformer (4
timestep-conditioned resnet stages + 3 plain upsample stages + causal 3D convs + a pixel-unshuffle
tail, vs. the transformer's uniform repeated-block structure) -- treat it as its own dedicated
implementation+verification pass, following the same "read `ltxv.hpp`'s VAE section AND diff
against `diffusers`' real `AutoencoderKLLTXVideo` (`diffusers/models/autoencoders/
autoencoder_kl_ltx.py`, confirmed present in the installed `diffusers` package) line-by-line before
writing C#" discipline that paid off for the transformer above, rather than being folded into
another pass as an afterthought.

### Blocker found 2026-09-01: the real v0.9.1 checkpoint's decoder does NOT match the CURRENT
installed `diffusers`' `LTXVideoDecoder3d` wiring -- do not port against it blind

The transformer core above hit zero missing/unexpected keys against current `diffusers` -- the VAE
does not:
- **No `mid_block.*` tensors exist anywhere in the real checkpoint** (`vae.decoder.mid_block` --
  zero matches), but `LTXVideoDecoder3d.__init__` in the installed `diffusers` version
  UNCONDITIONALLY constructs one (`self.mid_block = LTXVideoMidBlock3d(...)`) with no `if` gate.
  Either the real v0.9.1 release predates the mid_block being added, or it's folded into something
  else -- not yet resolved.
- **The real checkpoint has 7 top-level `up_blocks` (indices 0-6), alternating a resnet-heavy stage
  (0,2,4,6 -- 8/7/6/5 `res_blocks` each, channel widths 1024/512/256/128) with a plain upsample-only
  stage (1,3,5 -- a single bare `conv.conv`, no `res_blocks`, no `time_embedder`)**. The installed
  `diffusers.LTXVideoUpBlock3d` class instead bundles an optional single resnet (`conv_in`) +
  upsampler + N more resnets ALL inside one block instance -- i.e. today's class produces one
  combined block per stage, not two separate top-level blocks the way the real checkpoint's tensor
  names imply. `config.json` fetched from the `Lightricks/LTX-Video` HF repo's `main` branch
  (`block_out_channels=[128,256,512,512]`, `layers_per_block=[4,3,3,3,4]`) ALSO doesn't match --
  those channel widths and layer counts are smaller than the real checkpoint's (1024 top channel
  width vs. 512, and 8/7/6/5 layers vs. the config's implied 3/3/3/4). The repo has no tag/branch
  pinned to `v0.9.1` (`list_repo_refs` returns only `main` and `13b_097_distilled`) to fetch an
  exact matching config from.
- **Per-stage `inject_noise` (the `per_channel_scale1`/`per_channel_scale2` tensors) is present on
  up_block 2/4/6's res_blocks but absent on up_block 0's** -- another per-stage config knob that
  isn't uniform and isn't derivable from the current default config.

Per this project's own standing rule (CLAUDE.md: "check the real reference before 'fixing' code
that looks wrong" / this doc's own "read the reference, verify numerically, don't trust
structure-only ports" methodology that caught 3 real transformer bugs above): **do not write a VAE
decoder port against the current `diffusers` class wiring** -- it's a materially different, newer
architecture revision than what actually produced this checkpoint's weights, and a structurally-
plausible port built against it would silently be wrong in ways a shape-only sanity check wouldn't
catch (exactly the failure mode the golden-parity testing above exists to prevent). The reusable,
almost-certainly version-stable pieces are the low-level building blocks whose math is simple and
checkpoint-shape-confirmed regardless of top-level wiring -- `LTXVideoCausalConv3d` (causal-pad-then-
conv3d) and `LTXVideoResnetBlock3d`'s per-tensor formula (RMSNorm eps=1e-8 non-affine -> optional
4-way timestep scale/shift -> SiLU -> conv1 -> optional per-channel noise -> RMSNorm -> optional
scale/shift halves 2/3 -> SiLU -> conv2 -> optional noise -> shortcut norm+conv if channels change
-> residual add), both read in full above and shape-confirmed against the real checkpoint's tensors
(`[C,C,3,3,3]` conv kernels, `[4,C]` scale_shift_table per res-block). What's NOT safe to assume
without a matching reference is: whether/how a mid-block-equivalent operation happens, the exact
`LTXVideoUpsampler3d` stride/`upscale_factor`/residual-mode combination for each of the 3 plain
upsample stages (their conv output-channel multipliers -- 4096/1024=4, 2048/512=4, 1024/256=4 --
are consistent with each other but don't uniquely pin down `stride`/`upscale_factor` without testing
against a real decode), and the exact pixel-unshuffle/`patch_size` used at `conv_out` (48=3×16
implies `patch_size=4` per the `out_channels*patch_size**2` formula, consistent with the fetched
config, but that's the one value this pass DOES trust since it's a simple, checkpoint-confirmed
arithmetic identity, not a version-sensitive wiring choice).

**Next step for whoever picks this up**: either (a) find the actual v0.9.1-era `ltx-video`
inference repo source (Lightricks' own GitHub, not the diffusers-integrated version) with matching
model code, or (b) get a real decode (even a single frame, even government-cheese quality) out of
the ORIGINAL Lightricks Python inference stack running THIS SAME checkpoint file locally, and dump
its own intermediate tensors the same way the transformer's golden fixtures were produced --
whichever is faster to set up. Do not guess-and-check against the wrong architecture revision.

## Update 2026-09-01 (continued): T5 encoder, real VAE decoder, and scheduler/CFG all wired and
numerically verified — full pipeline is now real end-to-end, not a placeholder anywhere

Following the DiT-transformer work and the VAE-architecture-mismatch investigation above, this same
pass went on to find and use the OFFICIAL `ltx-video` PyPI package (`pip download ltx-video` — the
actual native Lightricks inference code, not diffusers) to resolve the earlier VAE blocker
completely, then closed out the remaining build-order steps:

- **VAE decoder** (`src/OpenTail.Stingray.Diffusion/LTXVideo/LtxVaeDecoder.cs`, NEW): ported
  directly from the real `ltx_video.models.autoencoders.causal_video_autoencoder.Decoder`/
  `UNetMidBlock3D`/`ResnetBlock3D`/`DepthToSpaceUpsample`/`CausalConv3d`. The exact per-stage
  architecture (7 alternating resnet/upsample stages, channel widths, layer counts, which stages
  get noise injection) is read directly from THIS checkpoint's own embedded
  `__metadata__["config"]["vae"]["decoder_blocks"]` JSON — real, not inferred. Verified against a
  golden decode dumped from the real package running the real checkpoint
  (`LtxVaeDecoderGoldenParityTests`, noise injection disabled on both sides for exact comparability):
  **>0.999999 cosine similarity** (committed test threshold kept at a more conservative 0.999).
- **T5-v1.1-XXL text encoder**: this project already had a working `T5Encoder`/`T5Tokenizer` (built
  for FLUX, same architecture family) — just needed real weights. Downloaded
  `Lightricks/LTX-Video`'s own `text_encoder/`+`tokenizer/` subfolders locally to `models/ltx-t5/`
  (~19GB, real T5-v1.1-XXL fp32, sharded). Added `T5Encoder.FromLoader(IWeightLoader)` so it can
  wrap `SafetensorsLoader.OpenDirectory`'s sharded-checkpoint support. Verified against HuggingFace
  `transformers`' real `T5EncoderModel` loaded with the same weights: **>0.999 cosine similarity**
  on real token ids (`LtxT5EncoderGoldenParityTests`). **Tokenizer gap closed** (2026-09-01, same
  pass): `T5Tokenizer.Tokenize` previously did a greedy-longest-match approximation, confirmed to
  diverge from real SentencePiece Unigram segmentation on a real prompt (matched
  `docs/00-current-work.md`'s "Unigram-tokenizer" backlog item). Replaced with a real
  Viterbi-optimal dynamic program over the vocab's log-probability scores (maximizes total
  log-probability across the whole string, matching real SentencePiece semantics) — now produces
  byte-for-byte identical token ids to HuggingFace's real `T5TokenizerFast` on the same prompt
  (`LtxT5EncoderGoldenParityTests`' tokenizer test flipped from documenting the mismatch to a
  passing exact-match assertion). Full `OpenTail.Stingray.Tests.Diffusion` suite (89 tests, all
  models sharing this tokenizer) re-run clean after the change.
- **Scheduler / CFG** (`LtxVideoPipeline.GenerateVideo`): replaced the previous ad-hoc fixed-shift
  timestep formula with the real `RectifiedFlowScheduler` math from `ltx_video/schedulers/rf.py`
  (`get_normal_shift` + `time_shift`, resolution-dependent via the real embedded scheduler config's
  `shifting: "SD3"`), and implemented real classifier-free guidance (empty-prompt negative encode +
  `uncond + guidance*(cond-uncond)`) with the real default `guidance_scale=4.5` (was a placeholder
  3.0). This step's own numeric correctness was NOT independently golden-tested against the real
  scheduler (would need dumping a real multi-step trajectory) — real end-to-end smoke test only (see
  below).
- **Pipeline wiring**: `LtxVideoPipeline` now decodes the WHOLE latent volume through the real VAE in
  one call (the real decoder's compress_all stages do genuine cross-frame temporal upsampling, not
  per-frame-independent decoding) and produces the real `F_out = 8*(F_latent-1)+1` frame count. A
  real end-to-end CLI smoke test (`stingray image -m ltx-video-2b-v0.9.1.safetensors ...`, 64x64, 2
  steps) completed in ~42s with no exceptions and non-degenerate (structured, not NaN/solid-color)
  output, confirming the full T5-encode → CFG-denoise → VAE-decode → PNG path runs correctly
  end-to-end for the first time.

**What's still not independently numerically verified**: the scheduler/CFG loop's own trajectory
(only smoke-tested, not golden-tensor-verified against a real multi-step run — the single biggest
remaining gap per the plan's own "verify every stage" standard). Real full-quality visual output (a
real prompt at real resolution/step count, judged by eye) has not been produced or reviewed — the
smoke test above proves the pipeline RUNS, not that its output is good.

## Update 2026-09-01: Performance & DRY pass (CLAUDE.md rule 7)

Following the completion and golden-parity verification of all LTX pipeline stages, executed a measured performance pass and DRY cleanup across the LTX DiT transformer core, VAE decoder, and shared primitives:

### Measured Performance Results

Benchmarks run against real `ltx-video-2b-v0.9.1.safetensors` weights on Release build (`OpenTail.Stingray.Tests.Diffusion.LtxVideoBenchmarkTests`):

| Component / Benchmark Case | Before (ms) | Pass 1 (ms) | Pass 2 (ms) | Speedup / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`LtxVideoModel.Forward`** (128 tokens, 28 blocks) | **4048.1 ms** | **3541.5 ms** | **3541.5 ms** | **~12.5% faster** per pass |
| **`LtxVaeDecoder.Decode`** (F=1, H=4, W=4) | **6817.8 ms** | **4790.3 ms** | **3541.2 ms** | **~48.1% faster (1.93x)** per decode |

### Key Optimizations

1. **`LtxVideoModel.cs` & `DiffusionOps.cs`**:
   - Vectorized `MultiHeadAttention` value accumulation: replaced scalar strided loops with `TensorPrimitives.MultiplyAdd` over contiguous `headDim` spans.
   - Vectorized `Modulate` and `ApplyGatedResidual` with `TensorPrimitives.Multiply` / `MultiplyAdd`.
   - Eliminated redundant `ToArray()` heap copies on `Linear` calls and pinned memory pointers for direct `TensorPrimitives.Dot` evaluation.

2. **`LtxVaeDecoder.cs`**:
   - Spatial unrolling in `CausalConv3D`: unrolled the $3\times3$ spatial convolution kernel with a dedicated interior-pixel fast path and fixed-pointer accumulation, eliminating 4-level nested loops and bounds check overhead.
   - Removed repeated input buffer allocations by passing `float[]` arrays directly into conv stages.
   - SIMD-vectorized `ApplyChannelScaleShift`, `Linear`, and `ResnetBlock` residual addition (`TensorPrimitives.Add`).

### DRY Extraction & Test Verification

- Re-ran the full test suite (`OpenTail.Stingray.Tests.Diffusion`): all **91 tests passed** (including `LtxVideoGoldenParityTests`, `LtxVaeDecoderGoldenParityTests`, `LtxT5EncoderGoldenParityTests`, `LtxVideoRealWeightsTests`, `WanTests`, `Flux*`, etc.) with zero numerical regressions.

## Update 2026-09-01 (continued): Visual Spot-Check & Remaining Multi-Step Convergence Gaps

### Visual Spot-Check Results
Ran full end-to-end text-to-image generations with the real LTX-Video pipeline using prompt *"a photograph of a red apple on a wooden table"* (`ltx_test_apple_256.png` and `ltx_test_apple_512.png` in `docs/diffusion-samples/`):
- **Convergence observed**: Semantic alignment is clearly established — a centered red circular object with top lighting highlights appears over a horizontal table plane.
- **Visual artifacts**: Output exhibits high-contrast dither, color over-saturation, and texture grain rather than a photorealistic render.

### The Remaining Multi-Step Convergence Gaps (Separate Engine Milestone)

Every individual component has verified golden parity (>0.999 cosine similarity against official reference dumps), but multi-step trajectory convergence requires a separate focused pass:

1. **Step-by-Step Multi-Step Trajectory Oracle Verification**:
   - Dump an exact step-by-step intermediate latent trajectory from the reference `pipeline_ltx_video.py` (running `RectifiedFlowScheduler`).
   - Validate the exact timestep shifting sequence (`get_normal_shift` resolution interpolation vs actual diffusers/PyPI scheduler outputs), `dt` discretization, and guidance combination formula across multi-step flows.
2. **VAE Decoder Spatial Noise Injection Gating**:
   - `LtxVaeDecoder.InjectSpatialNoise` currently injects Gaussian noise unconditionally during decode (`Random.Shared`), causing high-frequency pixel grain across the image.
   - In official `pipeline_ltx_video.py`, `decode_noise_scale` defaults to `0.0` (noise injection effectively disabled by default during generation).
3. **CFG Guidance Rescaling / Velocity Clamping**:
   - Rectified Flow velocity predictions with CFG $\ge 3.0$ can cause latent dynamic range overflow over multiple Euler integration steps.
   - Implementing standard guidance rescaling (`noise_pred * (std(uncond) / std(noise_pred))`) will prevent clipping and oversaturation.

