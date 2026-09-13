# LTX-Video GPU Residency Plan (2026-09-13)

## Read this warning before doing any GPU work here

Unlike FLUX (`docs/069`), Wan (`docs/072`/`docs/073`), Z-Image-Turbo (`docs/075`), and
SD3.5-medium (`docs/076`) — all of which have a **real, verified-correct CPU baseline** to build
GPU residency on top of and check GPU output against — **LTX-Video's own correctness is not yet
solid.** Per `PerformanceLeague.md`'s LTX-Video row: "only 1 of 6 total runs across this checkpoint
has ever produced a coherent image" — the rest, including repeat runs at the same explicit seed,
produced garbled visual noise, and the cause is explicitly **not root-caused** as of that entry.
`LtxVideoModel.cs`'s own class doc comment also states it is "NOT yet wired to a real T5-v1.1-XXL
encoder... or the VAE decoder" — both deferred, per the original implementation plan
(`docs/055-ltx-video-implementation-plan.md`).

**Building GPU residency on top of an unresolved correctness bug means every future GPU-vs-CPU
parity check is comparing against a baseline that might itself be wrong** — this doubles the
debugging surface (is a mismatch a GPU bug, or is it the CPU path's own known-unreliable output?)
exactly the situation this project's own `CLAUDE.md` rule 7 warns against ("Performance pass...
once a model's port is complete... do a performance pass"). **Recommended: treat the correctness
investigation as a prerequisite, not a parallel task** — either do it first, or at minimum flag
very clearly in any GPU-residency PR that correctness here was already shaky before the GPU work
started, so a future regression report doesn't waste time blaming the wrong change.

If the correctness bug gets fixed (or ruled unrelated to the DiT itself — e.g. traced to VAE/T5
wiring, which are explicitly not yet real per the class doc comment above) before this doc is
picked up, the rest of this plan is still accurate and can proceed normally.

## Current state — confirmed less mature than the other four models

`src/OpenTail.Stingray.Diffusion/LTXVideo/LtxVideoModel.cs` (441 lines) has **zero references to
`IComputeBackend`, `Gpu`, or `Vulkan` anywhere** — confirmed by grep. This is a step earlier than
FLUX/Wan/Z-Image/SD3.5 were before their own residency work: those all had at least the
`MatQ`-style per-matmul GPU dispatch path already wired (upload/`Sgemm`/download/free per call);
LTX-Video has no GPU code at all yet, CPU-only from the ground up.

## Real architecture — confirmed from the file directly

- **`headDim=64` by default** (`DetectConfig`'s fallback `heads=32, headDim=64` for
  `hidden=2048`, confirmed real via `InferAttentionLayout` — checked against the actual checkpoint,
  not hardcoded-and-assumed). Same as SD3.5-medium (`docs/076`): use `MultiHeadAttentionTiled`
  (headDim=64), NOT `MultiHeadAttentionTiled128` — the two share no fused kernel this session's
  FLUX/Wan/Z-Image work already tuned.
- **Continuous 3D RoPE, confirmed INTERLEAVED pairing** (`LtxVideoRoPE.ComputeContinuous3DRoPE`'s
  own doc comment: "interleaved layout: index 2i and 2i+1 share one rotation angle") — the SAME
  convention FLUX's `Flux2DRoPE` GPU kernel already implements. Unlike Z-Image (`docs/075`, where
  this needed independent verification) this one's convention is already documented in the source
  file itself, so `Flux2DRoPE` is very likely directly reusable via a Wan-style compact
  frequency-table builder (mirror `WanRoPE.Compute3DRoPECompact`'s approach: build LTX-Video's own
  per-axis frequency bands in the compact `[tokens, headDim/2]` one-value-per-pair layout
  `Flux2DRoPE` expects) — **but confirm the actual per-axis dimension split** (LTX-Video is video,
  so likely frame/height/width like Wan, not FLUX's 2-axis image split) directly against
  `ComputeContinuous3DRoPE`'s implementation before assuming Wan's specific 44/42/42-style numbers
  apply — they won't; get LTX-Video's own real axis-dim split from the source.
- **`BasicTransformerBlock` with cross-attention, gated self/cross-attention, and an AdaLN-single
  timestep branch** (per the class doc comment, referencing the real `stable-diffusion.cpp`
  `ltxv.hpp` structure directly) — this has a real cross-attention sub-layer (against T5-XXL
  caption embeddings) in addition to self-attention, structurally closer to Wan's
  self-attn+cross-attn block shape (`docs/072`) than to FLUX's single joint-attention blocks.
  `CrossAttentionAdaln`/`SelfAttentionGated`/`CrossAttentionGated` are real, checkpoint-detected
  booleans (not assumed) — read `LtxVideoModel.cs`'s actual block-forward method to get the exact
  modulation/gating wiring right per these flags before porting; do not assume they're always true
  or always match another model's convention.
- **Intermediate-tensor capture fields already exist** (`LastProjInOut`, `LastCaptionProjOut`,
  `LastEmbeddedTimestep`, `LastTimestepProj`, `LastRopeCos`, `LastRopeSin`, `LastBlock0Out`) —
  populated unconditionally by the CPU `Forward()` for `LtxVideoGoldenParityTests`. **A GPU port's
  own parity test can piggyback on these same captured intermediates** rather than needing to
  invent new checkpoints — genuinely useful, already-built infrastructure for exactly this kind of
  verification work.

## Recommended approach

Given LTX-Video starts one step earlier than the other four models (no GPU code at all, not even
`MatQ`), consider doing this in two explicit phases rather than jumping straight to full
residency:

1. **Phase 0 (prerequisite, see warning above)**: confirm or fix LTX-Video's CPU correctness bug,
   or get explicit sign-off that GPU residency work should proceed anyway (e.g. if the bug is
   confirmed to live in the VAE/T5 wiring, not the DiT this doc covers).
2. **Phase 1**: add basic GPU dispatch (`_backend` field + constructor parameter, `MatQ`-style
   per-matmul upload/`Sgemm`/download/free — the SAME starting shape FLUX/Wan/Z-Image/SD3.5 all
   had before their own residency work) — gets a working, correctness-checkable GPU path with
   minimal new surface area, useful as an intermediate correctness checkpoint even if slow.
3. **Phase 2**: real residency — `LtxVideoGpuWeights`/`LtxVideoGpuWorkspace` (mirror
   `FluxGpuWeights.cs`/`WanGpuWeights.cs` directly), a real `ForwardGpu`/`BasicTransformerBlockGpu`
   wiring self-attention + cross-attention (against precomputed, GPU-resident caption K/V, the same
   invariant-caching pattern Wan's `PrecomputeCrossKvCacheGpu` already established) + gated
   residuals + FFN through GPU-resident weights and workspace buffers, batched one
   `BeginBatch()`/`EndBatch()` per block.
4. **Verify correctness with a real, cheap parity test first, using the existing captured
   intermediates** (`LastBlock0Out` etc.) as the comparison points, or the same
   `FluxGpuVsCpuForwardBisectDebugTest.cs`-style small-synthetic-scale approach if those fields
   don't cover what's needed.
5. **Real end-to-end re-verification is unusually important here** given the known correctness
   fragility — don't just check the GPU path matches the CPU path numerically; re-run the same
   visual-inspection discipline `PerformanceLeague.md`'s own LTX-Video row already established
   (view the actual output image, don't just trust a non-crash) before claiming anything works.
6. **Update `PerformanceLeague.md`** with real, measured before/after numbers AND an honest
   correctness note — if the underlying CPU correctness issue is still unresolved when this is
   picked up, say so explicitly in the same entry rather than silently reporting only a timing
   number next to a possibly-still-broken image.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code.**
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration**, not full end-to-end runs, until confident.

## Success criterion

Given the correctness caveat above, success here is **not just a speed number** — it's a real,
visually-confirmed-coherent output on both CPU and GPU paths, with the GPU path measured honestly
against CPU (win or not). Do not report a "GPU residency win" for this model without first
addressing whether the underlying image-coherence bug is still present.
