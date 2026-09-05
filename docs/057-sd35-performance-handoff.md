# 057 — SD3.5-medium performance handoff

## RESOLVED, 2026-09-05: four real correctness bugs found and fixed via direct diffusers source comparison -- real, coherent (non-photorealistic) image output achieved

Following this doc's own "next step is numeric verification, not more performance work" pointer.
Downloaded real checkpoints (`sd3.5_medium-Q4_K_M.gguf` via `stingray pull`, text encoders + VAE
via `hf download adamo1139/stable-diffusion-3.5-medium-ungated`, a stock CLIP tokenizer) and ran
the exact 256x256/20-step/seed-42 disambiguation test this doc already called for ("a red apple on
a wooden table"), iterating four times as each real bug was found and fixed. Every fix below was
confirmed by directly reading `examples/diffusers/src/diffusers/models/{attention.py,
normalization.py,embeddings.py}` first (CLAUDE.md rule 8), not guessed.

**The real image progression, in order (this is itself useful evidence for anyone re-verifying):**
1. **Before any fix**: pure disorganized, oversaturated color noise -- fine, roughly pixel/small-
   blob-scale, no structure at all.
2. **After fix 1 (norm_hidden_states2 caching) alone**: visually indistinguishable from (1) --
   this fix was real but not the dominant symptom driver.
3. **After fix 2 (VAE scale/shift) added**: still visually near-identical to (1)/(2) (confirmed the
   fix had compiled in via DLL timestamp) -- also real, but garbage-in-garbage-out: a wrong-but-
   still-arbitrary latent looks like noise under any reasonable VAE scale.
4. **After fix 3 (unpatchify channel order) added**: a QUALITATIVE change -- from garish random
   blotches to a much finer, muted, perfectly regular/periodic grid-like WEAVE texture, uniform
   across the whole image, no spatial variation anywhere. This specific signature (regular,
   position-independent texture) is the classic symptom of a transformer with no positional
   information reaching it.
5. **After fix 4 (positional embedding) added**: a MAJOR qualitative change -- real, coherent,
   asymmetric image structure: distinct object/geometric shapes, real edges, shading, and what
   reads as reflections on a surface, warm brown/orange/yellow tones. **Not** a clean,
   photorealistic "red apple on a wooden table" (SD3.5-medium at 20 steps / 256x256 / a Q4_K_M
   quant is not going to be photorealistic regardless), but unambiguously a real generated image
   with genuine spatial structure -- not noise, not a uniform texture, by a wide margin the
   farthest this checkpoint has ever gotten in this project's history. This is the honest, current
   state: real progress, not (yet) a confirmed "this exactly matches the prompt" result.

**The four real bugs, each in `src/OpenTail.Stingray.Diffusion/`:**

1. **`SD3/MMDiTModel.cs`: dual-attention blocks normalized the WRONG input for `attn2`.** Real
   `SD35AdaLayerNormZeroX.forward` computes BOTH `norm_hidden_states` (for `attn`) and
   `norm_hidden_states2` (for `attn2`) from ONE shared `self.norm(hidden_states)` call over the
   block's ORIGINAL, pre-attention `hidden_states` -- both are cached together before any attention
   runs. This port instead recomputed `norm_hidden_states2` at the point of use, by which time `x`
   had already been mutated by the first attention's residual add -- feeding `attn2` a LayerNorm of
   the wrong (post-residual) input. Fixed by adding a `NormedImg2` workspace buffer, computed
   immediately after the main branch's `ModulateNorm` (before any mutation), used at the `attn2`
   call site instead of recomputing.
2. **`VaeDecoder.cs`: SD3.5's VAE was silently decoded with FLUX/Z-Image-Turbo's scale/shift.**
   `VaeDecoder` is shared across SD1.5/SDXL (4-channel), FLUX.1/Z-Image-Turbo (16-channel, real
   scale=1/0.3611, shift=0.1159), and SD3/3.5 (ALSO 16-channel, but a genuinely different VAE
   checkpoint with its own real `scaling_factor=1.5305`/`shift_factor=0.0609`, confirmed from the
   real `stabilityai/stable-diffusion-3.5-medium` `vae/config.json`) -- channel count alone cannot
   distinguish the two 16-channel cases, and the existing heuristic silently applied FLUX's
   constants to SD3.5's latents (~4.24x scale error). Added a `Decode(..., float? scaleOverride,
   float? shiftOverride)` overload; `Sd3Pipeline.cs` now passes its own real values explicitly.
3. **`SD3/MMDiTModel.cs`: unpatchify assumed the WRONG per-patch channel/spatial flattening
   order.** Real `_unpatchify` (`x.reshape(B,h,w,p,p,c)` then `einsum("nhwpqc->nchpwq")`) lays out
   each patch token's raw output as `(dy, dx, channel)` with CHANNEL fastest-varying. This port
   assumed the same `(channel, dy, dx)` order patchify's INPUT side correctly uses (matching
   `x_embedder`'s real `Conv2d` weight layout) -- but `x_embedder.proj` and `final_layer.linear`
   are different real weight matrices with genuinely different flattening conventions, not
   symmetric. Fixed the unpatchify loop order (dy outer, dx middle, channel innermost).
4. **`SD3/MMDiTModel.cs`: no positional embedding was ever added to image tokens.** Real
   `PatchEmbed.forward` does `latent = self.proj(latent); return latent + pos_embed` -- a real,
   checkpoint-stored 2D sincos positional embedding (confirmed via `list-tensors`: a real tensor
   literally named `pos_embed`, corresponding to a 384x384 position grid = this checkpoint's real
   `pos_embed_max_size` config), center-cropped to the actual generation's patch grid
   (`top/left = (384 - height_or_width) / 2`, per the real `cropped_pos_embed`) and added to every
   patch token BEFORE any transformer block runs. This port added no positional information
   anywhere -- added `MMDiTModel.AddCroppedPosEmbed`, lazily loading and caching the real tensor,
   applied right after the `x_embedder` projection.

**Real correctness verification**: `Sd3ConformanceTests` (4/4, including the structural
`Sd3_MMDiTModel_Forward_ExecutesCorrectly` smoke test, updated with a synthetic `pos_embed` tensor
so its mock weight set still exercises the real code path) and `Sd3TimestepEmbedParityTests` (the
existing real numeric oracle for the timestep/pooled-embedding conditioning vector, unaffected by
any of these four fixes) both re-pass clean.

**Not yet done / real open items**: the output is not yet numerically golden-verified against a
real diffusers reference (no Python-side oracle was built this pass -- the direct source-reading
methodology was used instead, and the real image progression above is the actual evidence), and
the current output, while a real image, is not confirmed to match the specific prompt content. If
further chasing this: the remaining most-likely next levers (not yet checked) are the CLIP text
encoder's pooled-output extraction (EOS-token position vs real `CLIPTextModelWithProjection`) and
whether joint-attention's QK-norm could still be degrading text-conditioning influence even with
positions now correct. Checkpoints (`models/sd3.5_medium-Q4_K_M.gguf`,
`models/sd35-medium-aux/`, `models/clip_tokenizer.json`) deleted after this pass per this project's
disk-space convention -- re-fetchable via the "How to reproduce" section below (the
`--clip-tokenizer` file can come from any stock CLIP checkpoint, e.g.
`hf download openai/clip-vit-large-patch14 tokenizer.json`).

**Status: paused 2026-09-02, ready to hand off.** This doc is written as a self-contained brief
for whoever (human or AI) picks this back up — you should not need to re-read the whole session
history to continue.

**UPDATE (2026-09-02, same day): a first performance pass landed (ArrayPool `Workspace` +
`Span`-based helpers + concurrent CFG passes via `Parallel.Invoke`), reviewed and confirmed
thread-safe, and the FIRST-EVER completed run happened: 256×256, 4 steps, CPU, Q8_0 GGUF —
**215.0s total**. Output is not yet a recognizable image (disorganized color noise, not the
earlier "periodic tiling" signature). This is most likely because 4 steps is genuinely too few
for this non-distilled model — real SD3.5's own recommended step count is 20-28, unlike FLUX-
schnell's 4-step distillation — not necessarily a remaining bug, but this has NOT been
disambiguated yet. See "Next step" at the bottom.

## What's already fixed (five real, verified bugs)

All five are already applied in `src/OpenTail.Stingray.Diffusion/SD3/MMDiTModel.cs` and
`Sd3Pipeline.cs`. Do not re-investigate these — they're confirmed against the real vendored
`examples/diffusers/src/diffusers/models/{transformers/transformer_sd3.py,attention.py,
normalization.py}` source, not guessed, and each is covered by an inline doc comment at its fix
site:

1. **CLI could only load one gated checkpoint layout.** `Sd3Pipeline.Load` only matches the
   StabilityAI single-file `..._incl_clips[_t5xxlfp8].safetensors` export, which is gated on
   HuggingFace with no ungated mirror. Added `Sd3Pipeline.LoadSeparate` (independent
   clip-l/clip-g/transformer/vae files) and a `--clip-g` CLI flag; `RunSd3` branches to it when
   `--clip-l`/`--clip-g`/`--vae` are all given.
2. **Fused-QKV misassumption.** `MMDiTModel` read three separate `qkv.0/1/2` weight matrices per
   block; the real checkpoint stores ONE fused `qkv.weight` `[dim, 3*dim]`. This alone made every
   real checkpoint unloadable — exactly why the pipeline had never been run once before this pass.
3. **Missing QK-RMSNorm.** Real `attn.ln_q.weight`/`attn.ln_k.weight` tensors exist and were never
   read or applied.
4. **Missing SiLU before every AdaLN modulation linear.** Real `emb = self.linear(self.silu(emb))`
   — `MMDiTModel` applied no SiLU at any of its three modulation call sites.
5. **SD3.5-medium's real "dual-attention" (MMDiT-X) extension was entirely unimplemented.** The
   first 13 of this checkpoint's 24 blocks each declare a full SECOND attention module
   (`x_block.attn2.*`) with a 9-chunk (not 6-chunk) modulation; real `JointTransformerBlock(
   use_dual_attention=True)` runs this as an extra image-only self-attention pass, gated and added
   as a second residual before the MLP. Also implemented `context_pre_only` handling for the last
   block (its `context_block` uses a different 2-chunk modulation with no gate/MLP at all).

See `docs/00-current-work.md`'s "SD3/3.5 run for the first time ever" section for the full
per-bug narrative if you want more context on any of these — but you should not need to
re-derive or re-verify them.

## Performance pass #1 (landed): ArrayPool Workspace + Span helpers + concurrent CFG

`MMDiTModel.Forward` now rents its scratch buffers once per call from `ArrayPool<float>.Shared`
(a `Workspace` struct, disposed at the end of `Forward`) instead of allocating a fresh `float[]`
for every intermediate at every block, and `Lin`/`ModulateNorm`/`ApplyGateAndResidual`/
`JointMultiHeadAttention` all take `Span<float>`/`ReadOnlySpan<float>` and write into
caller-provided buffers instead of returning new arrays. Separately, `Sd3Pipeline.Generate`'s
denoising loop now runs the CFG conditional and unconditional `_mmdit.Forward` calls concurrently
via `Parallel.Invoke` (previously sequential) — a real 2x-ish win on a multi-core box, since CFG
requires two full, independent forward passes per step. Verified thread-safe before trusting: the
shared `CachedWeightReader` weight cache both concurrent calls read from is properly
`lock`-guarded, and each `Forward` call's `Workspace` rents its own buffers from the (thread-safe)
`ArrayPool<float>.Shared` — no shared mutable state between the two concurrent calls beyond the
already-locked cache. `OpenTail.Stingray.Tests.Diffusion` re-run clean (98/98) after these changes.

**First real completed number: 256×256, 4 steps, CPU, Q8_0 GGUF — 215.0s.** This is a real,
measured baseline (previous attempts at 512×512/15-steps and even 512×512/4-steps never
completed within a practical wait, so this is the first actual data point). Scaling from here:
512×512 is 4x the pixel/token count of 256×256, and joint attention is `O(n^2)` in token count on
top of that, so do NOT assume linear scaling — measure 512×512 fresh rather than 4x-ing this
number. A rough estimate for MORE steps at the SAME 256×256 resolution: ~5x the steps (4→20) is
roughly ~5x the time if per-step cost dominates (it should, since weight loading/caching is a
one-time cost after the first step) — ballpark ~18 minutes for a real 20-step run at 256×256,
untested. Get a real number before committing to that estimate.

What's known architecturally that affects the perf profile specifically for SD3.5-medium (as
opposed to SD3.5-large, which doesn't have dual-attention at all, or SD3-medium/large before it,
which have neither dual-attention nor the same block count):
- 24 joint blocks, `HiddenSize=1536`, `NumHeads=24`, `HeadDim=64`.
- The first 13 of 24 blocks run a full SECOND self-attention pass (`attn2`) end-to-end (fused QKV
  projection, QK-norm, attention, output projection) — roughly **1.5-2x the per-block compute**
  for those 13 blocks compared to a plain (non-dual-attention) block.
- At 512×512 with `patch_size=2`, that's a `(512/8/2)^2 = 32^2 = 1024`-token image stream (VAE
  8x-downsamples first) attending jointly with whatever the T5/CLIP text-token count is — the
  attention itself is `O(n^2)` in that combined token count, on top of the doubled-compute blocks.

## RESOLVED (2026-09-02): still noise at 20 steps — this is a real, remaining correctness bug

Ran the disambiguation test this doc called for: same prompt/seed ("a red apple on a wooden
table", seed 42), same 256×256 resolution, 20 steps (real SD3.5 recommended range is 20-28,
vs. the earlier 4-step run). **656.9s / 11m5s wall clock.** Output is still pure disorganized
color noise, structurally indistinguishable from the 4-step result — no partial apple/table
shape, no convergence trend visible between the two step counts. This rules out "too few steps"
conclusively: a genuinely-converging model shows recognizable structure emerging well before 20
steps at this resolution. **Python 3.14 + torch 2.11+cpu + diffusers are already installed on this
machine** (`python` on PATH, not `python3` — confirmed 2026-09-02, removes the "needs a Python
diffusers env" setup step previously assumed blocking this). **Next step is the numeric block-by-block diffusers-reference
comparison** (same methodology as `docs/055-ltx-video-implementation-plan.md`/
`docs/056-flux-tiling-artifact-handoff.md`) — not more performance work. Suspect areas to check
first, in order of how recently they were touched without being numerically verified against the
real diffusers `SD35AdaLayerNormZeroX`/`JointTransformerBlock` source: the dual-attention gate
ordering (does `gate_msa2` really apply to `attn2`'s output and not `attn`'s?) and the QK-RMSNorm
per-head axis (row-major vs. column-major head split) — none of the 5 "fixed" bugs in this doc were
ever golden-verified against a numpy reference, only structurally reasoned from source and
crash-driven (bounds-check messages), unlike this project's usual golden-verification bar for

**UPDATE (2026-09-02): timestep/pooled-embedding stage golden-verified, RULED OUT.**
`scripts/sd3_timestep_embed_ref.py` + `Sd3TimestepEmbedParityTests.cs` (numpy reference of
`CombinedTimestepTextProjEmbeddings`, real `sd3.5_medium-Q8_0.gguf` weights) — **PASS**, cosine
> 0.999. The single AdaLN conditioning vector every joint block reads is correct, so the bug is
NOT there; it narrows to the per-block math (dual-attention gate ordering / QK-RMSNorm axis /
joint attention itself), which is the next thing to golden-verify, in that priority order.
other architectures this session.

## Where to look first for further performance work (suggested priority order, not mandatory)

1. **Measure before optimizing** (`CLAUDE.md` rule 7: "measure, don't assume... only keep a
   change if it's measurably better") — the 215.0s/256×256/4-step number above is a first
   baseline; get a fresh one after each further change, interleaved control/candidate, not a
   single run.
2. **Check whether GPU offload (`-g -1` / Vulkan / CUDA) already works and is faster**, before
   assuming CPU-only optimization is the right lever. `MMDiTModel`'s `Lin` helper already has a
   `_backend`-gated GPU dispatch path (`GetGpuWeight`/`_backend.Upload`/`_backend.Allocate`) —
   check whether it actually gets exercised correctly for this pipeline (it was never run once
   before this pass, so this GPU path is equally unverified) before assuming it's a free win.
3. **`DiffusionOps.Linear`'s CPU path** (the function whose `AccessViolationException` led to
   finding bug #5 above) is a plain `Parallel.For`-over-`outDim` scalar dot-product loop
   (`TensorPrimitives.Dot`-vectorized per output row) — reasonable but not obviously
   SIMD/cache-optimal for a 1536-wide hidden size at this many blocks/steps. Profile before
   assuming this is the hot path, though — the joint attention (`JointMultiHeadAttention`,
   `O(n^2)` over the combined ~1024+ token sequence) is a real candidate too, especially since it
   now runs TWICE (once for `attn`, once for `attn2`) on the first 13 blocks.
4. **The allocation-avoidance pass (`Workspace`/`Span`-based helpers) already landed** — most
   per-block intermediates now come from `ArrayPool` rentals, not fresh `float[]`s. What's LEFT
   allocating fresh per-block: `imgTokens`/`unpatchified`/`outLatent` in `Forward` (patchify/
   unpatchify, once per call not per-block, likely low-value to chase), and
   `ComputeTimeAndPooledEmbedding` (once per call, also low-value). Decide next whether the
   remaining lever is "make the existing kernels faster" (vectorization/cache-locality inside
   `Lin`/`JointMultiHeadAttention`) or "this resolution/step count is just not viable on CPU, GPU
   is the real answer" — don't assume which before measuring.

## How to reproduce

Checkpoints are NOT vendored (deleted after every pass per this project's convention — see
`CLAUDE.md`). Re-download (all ungated, no HF auth needed):

```
DiT (real StabilityAI joint_blocks/x_embedder naming, NOT the diffusers-native
     transformer_blocks re-export — see docs/00-current-work.md for why the diffusers-native
     variant doesn't work with MMDiTModel as written):
    city96/stable-diffusion-3.5-medium-gguf → sd3.5_medium-Q8_0.gguf (or a smaller quant, e.g.
    Q4_K_M, for faster loading/iteration — MMDiTModel reads whichever dtype is in the GGUF)

Text encoders + VAE (standard HF diffusers multi-file layout):
    ckpt/stable-diffusion-3.5-medium (or adamo1139/stable-diffusion-3.5-medium-ungated, same repo
    contents, both ungated mirrors of the gated stabilityai/stable-diffusion-3.5-medium):
      text_encoder/model.fp16.safetensors    (CLIP-L)
      text_encoder_2/model.fp16.safetensors  (OpenCLIP-bigG)
      vae/diffusion_pytorch_model.safetensors
```

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
  -m sd3.5_medium-Q8_0.gguf \
  --clip-l text_encoder/model.fp16.safetensors \
  --clip-g text_encoder_2/model.fp16.safetensors \
  --vae vae/diffusion_pytorch_model.safetensors \
  --clip-tokenizer models/clip_tokenizer.json \
  -p "a red apple on a wooden table" \
  --steps 4 --seed 42 --verbose -W 256 -H 256 -g 0 \
  --output sd35-out.png
```

Start smaller (256×256 or even 128×128, 4 steps) than the 512×512/15-step repro that didn't
finish in 15 minutes above — get a real, complete number first, then scale up deliberately.

## House rules for whoever picks this up (from this project's `CLAUDE.md`)

- **No subagents** — do all work directly in the main session for this project.
- **Measure, don't assume.** A plausible-sounding optimization that isn't actually measurably
  faster (interleaved control/candidate samples, not a single run) gets reverted, even if the
  reasoning behind it seemed sound. Write the measured numbers down (in this doc or
  `docs/00-current-work.md`), not just "should be faster."
- **Check the real reference before "fixing" code that looks wrong** — every bug fixed this pass
  was confirmed against the real vendored `examples/diffusers` source before writing code, not
  guessed. If you find yourself wanting to change math (not just performance), apply the same
  discipline.
- **Delete transient checkpoints after use** — do not commit large model files.
- **Once correctness is actually verified (a real, recognizable image), that's the point to loop
  back to `docs/00-current-work.md` and update SD3/3.5's status** — this handoff is scoped purely
  to "make it fast enough to check," not "confirm it's right." Those are two different, sequential
  jobs; don't skip straight to declaring victory on the second without actually looking at output.
