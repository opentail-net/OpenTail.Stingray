# 057 — SD3.5-medium performance handoff

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
steps at this resolution. **Next step is the numeric block-by-block diffusers-reference
comparison** (same methodology as `docs/055-ltx-video-implementation-plan.md`/
`docs/056-flux-tiling-artifact-handoff.md`) — not more performance work. Suspect areas to check
first, in order of how recently they were touched without being numerically verified against the
real diffusers `SD35AdaLayerNormZeroX`/`JointTransformerBlock` source: the dual-attention gate
ordering (does `gate_msa2` really apply to `attn2`'s output and not `attn`'s?), the QK-RMSNorm
per-head axis (row-major vs. column-major head split), and the timestep/pooled-embedding
computation feeding the AdaLN conditioning vector — none of the 5 "fixed" bugs in this doc were
ever golden-verified against a numpy reference, only structurally reasoned from source and
crash-driven (bounds-check messages), unlike this project's usual golden-verification bar for
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
