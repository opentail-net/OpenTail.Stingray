# Z-Image-Turbo S3-DiT GPU Residency Plan (2026-09-13)

## Context — read `docs/069` (FLUX) and `docs/072`/`docs/073` (Wan) first

Both FLUX's `FluxDiT` and Wan's `WanModel` have now been through this exact transformation: from
a per-matmul CPU/GPU ping-pong (`MatQ`-style: upload activation, dispatch one GEMM, download
result, free buffers, every single call) to real GPU residency (persistent VRAM-resident weights
+ workspace, a real per-block GPU forward pass). Both are real, proven, committed work
(`53839a3`, `0132aa7`, `d877353`) — this doc applies the same playbook to
`src/OpenTail.Stingray.Diffusion/ZImageDiT.cs`, which is **currently in the exact "before" state**
those two were in: confirmed by grep, its `_backend` field is only ever used via the old
`MatQ`-style upload/dispatch/download/free pattern (see `ZImageDiT.cs` lines ~388-546 for the
`Fp8`/`Bf16`/`Fp16` GPU-dequant branches, all following that shape) — no `ForwardGpu`, no
`ZImageGpuWeights`/`ZImageGpuWorkspace` exist yet.

## Why this is a good next target

- **Same `headDim=128`** as both FLUX and Wan (`ZImageParams.HeadDim => Dim / NHeads`, confirmed
  128) — `MultiHeadAttentionTiled128` (the fused attention kernel already tuned and verified twice
  this session) is directly reusable, no new attention kernel needed.
  `docs/069`/`docs/073`'s SGEMM work is likewise shape-agnostic infrastructure (`Sgemm`), already
  proven on two different models' matrix shapes.
- Z-Image-Turbo already got a **real CPU optimization pass** (context caching, FlashAttention —
  see `PerformanceLeague.md`'s "Z-Image-Turbo (Optimized CPU DiT + Context Caching +
  FlashAttention)" row, ~1.5× CPU speedup, 100% numerical parity verified) — so the CPU reference
  to check GPU output against is itself already fast and already trustworthy, unlike some other
  candidates where the CPU baseline itself might need scrutiny first.

## Real architecture — confirmed from `ZImageDiT.cs`'s own class doc comment, don't guess

- **30 single-stream layers + 2 context_refiner + 2 noise_refiner blocks**, `dim=3840`
  (`ZImageParams`), RMSNorm (no bias, `_p.NormEps`), gated-GELU-family FFN (confirm exact
  activation before assuming — check `ApplyBlock`'s FFN section directly, this doc didn't fully
  trace it).
- **3-axis RoPE** (`ZImageRoPE.cs`, `_dims = [32, 48, 48]` half-dims per axis, i.e. actual per-axis
  half-dims `[16, 24, 24]`, summing to 64 = `headDim/2`) — structurally the same *shape* of
  3-axis split FLUX (16/56/56) and Wan (44/42/42, in pairs 22/21/21) both use, but **the exact
  pairing convention (interleaved vs split-half) has NOT been checked in this doc and MUST be
  verified against `ZImageRoPE.cs`'s own rotation-application code before assuming FLUX's
  `Flux2DRoPE` GPU kernel is reusable.** This is not a formality — Wan's own RoPE turned out to use
  a genuinely different (interleaved) convention than the pre-existing `RoPE3D` GPU kernel's
  split-half assumption, and reusing the wrong one would have silently reintroduced a real,
  previously-fixed bug (see `docs/072`'s own account of this exact near-miss). **Read
  `ZImageRoPE.cs`'s rotation-application method directly, confirm interleaved-pair vs split-half
  against it, and only then decide whether `Flux2DRoPE` (interleaved) or a split-half kernel is the
  correct reuse** — do not assume either way.
- **AdaLN modulation uses a DIFFERENT convention than FLUX/Wan** — confirmed by reading
  `ApplyBlock` directly (`ZImageDiT.cs` lines ~190-260): gate values are `tanh`-activated
  (`TensorPrimitives.Tanh(gateMsa...)`) before use, NOT the raw linear-projection gate FLUX/Wan
  both use directly; and there is **no shift term at all**, only a scale (`1 + scaleMsa`,
  pre-incremented on host) — FLUX/Wan's AdaLN-Zero convention has both shift and scale. **The
  existing `AdaLNModulate`/`ScaleGateAdd` GPU kernels assume FLUX/Wan's exact convention and are
  NOT a direct drop-in as-is.** Recommended low-risk fix: keep computing the small
  `(scale, gate)` vectors on CPU exactly as `ApplyBlock` already does today (cheap, `dim`-sized,
  same pattern FLUX/Wan already use for their own small per-block modulation vectors), including
  the `tanh`/`+1` step, then upload the *already-transformed* result and call the existing
  `AdaLNModulate` (with `shiftOffset` pointing at a zero-filled buffer — `WanGpuWorkspace.Zeros`-
  style, or a fresh one) and `ScaleGateAdd` unmodified. This avoids writing a new shader entirely
  — same trick used to keep FLUX's/Wan's own residency work low-risk.
- **Two DIFFERENT block types feed into shared logic**: `context_refiner`/`noise_refiner` blocks
  run on text-only/image-only token subsets respectively with `modulated=false` (uses a
  `scale=gate=1` shortcut, no adaLN at all — see the `_onesCache` branch in `ApplyBlock`), while
  the main 30 `layers` run on the concatenated `[txt|img]` sequence with `modulated=true`. Both
  call the same `ApplyBlock` — a GPU port should preserve this shared-method structure (one
  `ApplyBlockGpu`, called with different `modulated`/token-range arguments) rather than duplicating
  it, mirroring how `FluxDiT.DoubleBlockGpu`/`SingleBlockGpu` stayed as two distinct methods only
  where the actual block *shape* differs, not where only modulation-on/off differs.
- **Invariant caching already exists on the CPU path** (`_cachedTxtEmbeds`/`_cachedRefinedTxtHid`,
  `_cachedImgFreqs`/`_cachedTxtFreqs`/`_cachedCombinedFreqs`) — port this same caching strategy to
  the GPU workspace (precompute once per prompt/resolution, reuse across denoising steps), the same
  invariant-caching pattern FLUX's `PrecomputeTxtGpu` and Wan's `PrecomputeCrossKvCacheGpu` already
  established for their own text-conditioning paths.

## Recommended approach

1. **RoPE pairing convention — CONFIRMED 2026-09-14, resolved**: directly read `ZImageRoPE.Apply`
   (the actual rotation-application code, not just the class doc comment) — it operates on
   `qk[j*2+0]`/`qk[j*2+1]` (adjacent-pair indices), i.e. the same INTERLEAVED convention FLUX and
   Wan both use, NOT split-half. **`Flux2DRoPE`'s GPU kernel is directly reusable**, via a compact
   per-token `[nTokens, headDim/2]` frequency table matching `WanRoPE.Compute3DRoPECompact`'s own
   pattern (write an analogous `ZImageRoPE.BuildFreqsCompact`, axis boundaries at half-dims
   [16, 24, 24] summing to 64 = headDim/2, same shape of fix already proven for Wan's [22,21,21]).
   This was the one genuinely open question in this doc — now closed, de-risking the rest of this
   plan; the remaining steps (GPU weights/workspace classes, `ApplyBlockGpu`, the tanh-gated
   AdaLN-without-shift quirk, a real parity test) are a substantial, multi-file undertaking not
   attempted in this pass — starting it without enough remaining room to properly verify each piece
   (per this whole session's own "green test doesn't mean real" lesson) would risk introducing an
   unverified new bug rather than real progress. Left for the next iteration with this blocker
   cleared.
2. **`ZImageGpuWeights` class** (new file): mirror `FluxGpuWeights.cs`/`WanGpuWeights.cs` — upload
   all block weights (context_refiner ×2, noise_refiner ×2, layers ×30, each with attention
   Q/K/V/O + norms + FFN + adaLN-modulation weights where present) once, resident in VRAM, same
   `UploadWeight` FP16-if-available pattern both prior classes already use.
3. **`ZImageGpuWorkspace` class** (new file): preallocated activation/normed/QKV/attention-output/
   FFN buffers sized for the real token counts (image patches + text tokens), plus the precomputed
   RoPE frequency tables and cached text-refiner output, following `FluxGpuWorkspace`/
   `WanGpuWorkspace`'s exact structure.
4. **`ApplyBlockGpu`** (new method on `ZImageDiT`, mirroring `ApplyBlock`'s existing structure):
   RMSNorm (reuse the existing GPU RMSNorm/AdaLNModulate primitives — Z-Image's norm has no
   affine weight multiply beyond RMSNorm itself per the CPU code, confirm this before assuming),
   scale-multiply, Q/K/V via `Sgemm`, RoPE (kernel choice per step 1), attention via
   `MultiHeadAttentionTiled128`, O projection, gated residual (`ScaleGateAdd` after the CPU-side
   tanh/scale prep described above), then the FFN sub-block analogously.
5. **Batch per-block**, not per-op or per-step — `BeginBatch()`/`EndBatch()` around each
   `ApplyBlockGpu` call, the same granularity choice FLUX settled on (too-fine pays per-dispatch
   sync cost; too-coarse risks starving the OS GPU scheduler on this shared iGPU) and Wan's own
   kernel-tuning work built on directly.
6. **Verify correctness with a real, cheap parity test first** — mirror
   `FluxGpuVsCpuForwardBisectDebugTest.cs`/`WanGpuParityTests.cs`'s pattern exactly: real weights,
   a small-but-real token count, compare the GPU forward pass against the existing (already
   parity-verified) CPU `Forward`/block methods for identical input, with a real numeric assertion
   (measure the actual FP16-GPU-vs-FP32-CPU gap, don't assume a specific tolerance number transfers
   from the other two models without checking).
7. **Real end-to-end re-verification**: Z-Image-Turbo already has a real, working CPU repro
   (`docs/00-current-work.md`'s 2026-09-12 fix session, `docs/diffusion-samples/
   z-image-turbo_red-apple-on-white-table_CPU-256x256-4steps_FIXED-2026-09-12.png` per
   `PerformanceLeague.md`) — use the same prompt/resolution/seed for the GPU-path re-verification,
   confirm the output still looks correct (visually compare against that known-good reference), not
   just that the run completes without error.
8. **Update `PerformanceLeague.md`** with real, measured before/after numbers — following the same
   honesty bar every prior doc this session established: if the first real GPU-resident version is
   slower than CPU (which happened for Wan before its own kernel-tuning pass, `docs/072`/`docs/073`
   — a real, expected, non-alarming outcome for freshly-wired-but-untuned GPU code), report that
   plainly rather than assume a win.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code** — the CPU `Forward`/`ApplyBlock` path stays.
- **Ask before committing** — recent work in adjacent areas was committed with explicit
  authorization each time; confirm current expectations rather than assume standing permission.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use small synthetic-scale tests for iteration**, not full end-to-end runs, until confident.

## Success criterion

Real, measured GPU-vs-CPU per-block (or per-forward) time for Z-Image-Turbo's S3-DiT, reported
honestly regardless of which way it comes out — correctness (a real parity assertion, plus a real
visual re-check against the known-good CPU sample) matters more than a specific speed number on
the first pass. Update `PerformanceLeague.md` only with what was actually measured.

## 2026-09-14 addendum — precise block-level scoping (real, from direct code reading, not implemented)

Read `ApplyBlock` (`ZImageDiT.cs` lines ~190-275) in full to correct/complete this doc's earlier
summary of the modulation convention, and to make the actual implementation lower-risk next time
(same discipline as the SD3.5 doc's own similar treatment) rather than rush a partial port now.

This is a real **"sandwich norm"** structure, distinct from every other model's own GPU-residency
work this session (FLUX/Wan/F5/CosyVoice3 are all pre-norm-only): **each sub-block (attention,
FFN) applies RMSNorm BOTH before AND after the actual computation.** Pre-norm+scale on the input
matches `AdaLNModulate`'s existing `isRmsNorm=true` path (with a zero-filled shift tensor
substituted, since Z-Image genuinely has no shift term at all — a safe reuse, not a new kernel).
But there is a SECOND, separate RMSNorm on the branch's raw output BEFORE the gated residual add
(`normW2`/`normW4` in the real code) — no existing GPU kernel fuses "RMSNorm the branch output,
then gate*result+x" in one call. Compose it from two already-proven ops instead: `RmsNormBatched`
on the branch output, then `ScaleGateAdd`'s plain 3-tensor overload
(`ScaleGateAdd(x, proj, gate, nTokens, dim)`), passing the real per-block `gateMsa`/`gateMlp`
vector as its own dedicated `[dim]` tensor — not the chunk-offset overload the other models use,
since Z-Image's gate isn't part of one combined modulation buffer at the point this needs it.

**The `tanh` + `1+scale` transform has no existing GPU op** (`TensorPrimitives.Tanh` is called
directly on the CPU in the real code). Handle it exactly like F5's own per-block modulation
Sgemm result (`F5DiTBlock.ForwardGpu`'s established precedent): download the real
`4*dim=15360`-float `mod` Sgemm output, apply `tanh`+`1+scale` on the CPU (negligible cost at this
size), re-upload the four `[dim]` sub-vectors (`scaleMsa`, `gateMsa`, `scaleMlp`, `gateMlp`) as
their own small GPU tensors. A real, bounded, small host round-trip per block — not eliminable
without a new dedicated shader (a legitimate future kernel-tuning target once correctness lands,
mirroring F5's own later RoPE-elimination follow-up, docs/080).

**Recommended block-level GPU op sequence** (per block, `modulated=true` case): compute
scaleMsa/gateMsa/scaleMlp/gateMlp (Sgemm + CPU round-trip, above) -> `AdaLNModulate(normed, x,
zeroShift, scaleMsa, nTok, dim, isRmsNorm:true)` -> QKV `Sgemm` -> RoPE (compact table at 16/24/24
half-dims, per this doc's own already-resolved interleaved-pairing finding) ->
`MultiHeadAttentionTiled128` (headDim=128, confirmed reusable) -> O-proj `Sgemm` ->
`RmsNormBatched` (branch output, `normW2`) -> `ScaleGateAdd(x, normedAttnOut, gateMsaTensor, nTok,
dim)` -> repeat the same pre/post-norm pattern for the FFN. **Real flag**: Z-Image's FFN is
`w2(silu(w1(x))*w3(x))` (SiLU-gated), NOT the tanh-GELU `VisionGeluInPlace` F5/CosyVoice3 use —
these activations are NOT interchangeable; check for an existing SiLU-in-place GPU kernel before
assuming any existing gated-FFN helper is a safe substitute.

**Not yet implemented** — `modulated=false` (the context_refiner/noise_refiner blocks, using the
cheap all-ones scale/gate shortcut) still needs its own pass through this same sequence with the
CPU round-trip skipped entirely (no real per-block modulation to compute there), plus the final
layer's own projection. Real next step: implement `ZImageGpuWeights`/`ZImageGpuWorkspace` +
`ApplyBlockGpu` per this scoping, verify with a real single-block parity test (same bar every
other model on this session's checklist has been held to) before trusting any of it.

## 2026-09-14, same day — `layers.N` block implemented, but genuinely UNVERIFIED (real blocker, not a skipped step)

Implemented the 30 main `layers.N` blocks per the scoping above: `ZImageGpuWeights.cs` (new file,
real per-block weight upload including the fp16-if-available `UploadWeight` pattern every other
model's own class uses), `ZImageGpuWorkspace.cs` (new file, including the zero-shift tensor trick
for `AdaLNModulate` reuse), and `ZImageDiT.ApplyBlockGpu` (new method, composed entirely from
already-proven kernels per the scoping above: `AdaLNModulate` with a zero shift, `FluxUnpackQkv`
for the fused-QKV split, `RmsNormBatched` reinterpreted as `[nTok*nHeads, headDim]` rows for the
real per-head QK-norm scope, `Flux2DRoPE`, `MultiHeadAttentionTiled` (auto-selects the headDim=128
fused kernel), `ScaleGateAdd`'s plain 3-tensor overload for the post-norm gated residual, and the
newly-interface-exposed `SiLuMul` for the SwiGLU FFN). Also added `IComputeBackend.SiLuMul` to the
core interface (`VulkanBackend` already had a real, working implementation — `docs/…` note: this
was dead/unreachable through the generic interface until now, a real, separate small fix).
Builds clean.

**Could NOT verify with REAL checkpoint weights — the actual DiT file is not present on this
machine.** `models/z-image-turbo/` only contains `tokenizer/` and `vae/` subdirectories; the DiT
weights file (`z_image_turbo-Q4_0.gguf`, referenced by the existing `ZImageRealWeightsTests.cs`)
does not exist locally, and — same as `docs/076`'s own SD3.5 finding — the C: drive is at ~453MB
free, not enough to download it. **Verified instead with a real synthetic-weight parity test**
(`ZImageGpuParityTests.cs`, new file) — the exact same primary verification method this session's
Wan/T5/UMT5 GPU-residency work relied on as ITS first correctness check (structurally real shapes
— headDim=128 kept at the real value since it drives kernel selection, dim/nHeads scaled down for
test speed — random-but-realistic weight magnitudes, real math). Added `ZImageDiT.
ApplyBlockForTest` (public test-only wrapper around the private `ApplyBlock`, mirroring every
other model's own "ForTest" accessor this session). **Result: maxDiff=0.0285 vs meanAbs=0.496
(~5.7% relative) — within normal fp16-GPU-vs-fp32-CPU precision expectations, a real pass, not
assumed.** This gives real confidence the implementation is structurally correct even without real
checkpoint weights — but a REAL-weights parity test (and a real visual end-to-end check) should
still be run by whoever has the checkpoint available (or frees disk space to download it) before
this is trusted for production use; synthetic-weight parity catches structural/formula bugs but
can't catch a checkpoint-specific tensor-naming mismatch the way F5TTS's own Q8_0-precision bug
(only found once, docs/080, via a real-weights test) would need real weights to surface.
