# 091 — FLUX.2 Vulkan GPU-residency scoping

**Status, 2026-09-19: gating condition 1 (CPU correctness) is now MET — FLUX.2's Pass 1 closed
for real (512×512/20-step production run confirms a clean coherent apple, see docs/088). Real
scoping work has now started** (`Flux2GpuWeights.cs` exists, defused of a real OOM landmine — see
below), but the double-block GPU forward pass, workspace, and a NEW required shader are not yet
built. Do not start further implementation from this doc alone without re-reading the "Real
blockers found before/while implementing" section below — they materially change the scope from
what was originally planned here.

## Gating conditions — check these before starting

1. **CPU correctness must be settled first.** As of this doc's writing, FLUX.2's real output is a
   structured periodic grid/tiling pattern, not yet coherent (a timestep-blindness bug was found
   and fixed this session, `docs/087`/`docs/088`; a second, narrower structural bug — likely
   patchify/token-ordering — is suspected but not yet found). **Porting broken math to GPU produces
   a faster wrong answer and doubles the eventual debugging cost** (CLAUDE.md rule 7, and this
   project's own established discipline — see the rule's own citation of past mistakes). Do not
   start this task until a real end-to-end FLUX.2 run produces genuinely coherent output on CPU.

2. **CPU performance must be re-measured after `docs/090`'s fix lands and is verified.** If the
   `QuantizedWeightCache` integration (already landed as of this doc's writing, not yet fully
   verified at production resolution) gets CPU generation into a reasonable time budget (see
   `docs/090` for context), GPU work may turn out to be lower priority than expected — measure
   first, don't assume GPU is needed just because it wasn't fast before the CPU fix.

3. **This machine's GPU is an integrated Radeon (Ryzen 5700G), not a discrete GPU.** CLAUDE.md rule
   13 documents a real, measured case (MiniMax-Music3's DiT) where this exact iGPU was **2.5x
   SLOWER than CPU** for a real workload, due to per-call dispatch overhead and shared-memory
   bandwidth with no VRAM advantage over the CPU it shares memory with. **Do not assume GPU
   residency will help FLUX.2 without a real, measured comparison on THIS hardware** — a large
   model with large per-call payloads (FLUX.2's biggest linear is ~340M elements) is a more
   favorable case for GPU dispatch-overhead amortization than MiniMax-Music3's smaller ops was, but
   this must be verified, not assumed. If it turns out slower, that is a real, useful finding about
   THIS machine, not evidence the GPU code path itself is wrong (CLAUDE.md rule 13's own framing).

## Why this task, if/when it happens, should look like FLUX.1's GPU port, not a fresh design

FLUX.2 is BFL's newer release in the exact same architectural lineage as FLUX.1 (confirmed in-repo
via `examples/flux`/`examples/flux2`): both use double-stream blocks + single-stream blocks with
shared (not per-block) modulation, RoPE-after-norm joint attention, and a fused single-block
`linear1`/`linear2`. FLUX.1 already has a complete, working, measured Vulkan GPU-residency port:

- `src/OpenTail.Stingray.Diffusion/FluxGpuWeights.cs` — one-time FP16 weight upload/residency,
  structured as `DoubleBlockGpuWeights[]`/`SingleBlockGpuWeights[]` arrays, mirroring the model's
  own block structure (see its `DoubleBlockGpuWeights` nested class for the exact per-block tensor
  set: QKV, QK-norm scales, attn-proj, MLP weights, modulation weights — FLUX.2 needs the analogous
  set, adjusted for FLUX.2's different dimensions (`HiddenSize=6144` vs FLUX.1's, `MlpRatio=3.0`
  SiLU-gated vs FLUX.1's GEGLU, 4-axis RoPE vs FLUX.1's 3-axis, shared single-vs-double modulation
  already documented in `docs/087`).
- `src/OpenTail.Stingray.Diffusion/FluxGpuWorkspace.cs` — the per-step GPU scratch-buffer
  workspace (intermediate activations), reused across steps to avoid repeated allocation.
- `PerformanceLeague.md`'s FLUX.1-schnell row documents the real, measured progression this port
  went through (298.5s → 198.1s warm, `M=1` matrix-vector fast-path dispatch, 2-block chunked
  command-buffer batching, pipelined concat/slice/final-layer, chunked T5-XXL text-encoder
  batching) — read this for the actual sequence of real, measured wins, not just the end state, so
  the FLUX.2 port can go through the same stages instead of guessing which optimization matters
  most.

**Port `FluxGpuWeights.cs`/`FluxGpuWorkspace.cs`'s pattern directly, adjusted for FLUX.2's real
dimensions and gated-FFN/4-axis-RoPE differences (already fully documented in `docs/087`'s
architecture derivation) — do not design a new GPU-residency pattern from scratch.**

## What's specifically different for FLUX.2 vs. the FLUX.1 GPU port

- **SiLU-gated FFN, not GEGLU**: FLUX.2's MLP is `SiLU(gate) * value` on a `2*mlpHidden`-wide
  up-projection split in half (confirmed against `examples/flux2/src/flux2/model.py`'s
  `SiLUActivation`, see `docs/087`) — FLUX.1's GPU MLP kernel almost certainly assumes GEGLU or a
  different gate convention; do not reuse it unmodified, verify the activation math specifically.
- **4-axis RoPE (t,h,w,l), not 3-axis**: `Flux2RoPE.cs`'s CPU implementation already generalizes
  `InterleavedRoPE` to N axes; the GPU RoPE kernel (if FLUX.1's is hardcoded to 3 axes) needs the
  same generalization, or a FLUX.2-specific variant.
- **A much larger text encoder (Mistral-24B, not FLUX.1's CLIP-L+T5-XXL combo)**: text conditioning
  for FLUX.2 (`Flux2TextConditioning.Encode`) runs through the shared `Engine.ForwardPass`/
  `EnableHiddenTaps` mechanism, not a diffusion-specific text encoder GPU path — whether this needs
  its own GPU work is a separate question from the DiT's own GPU residency, and should be profiled
  independently (it may already be the dominant cost given its size — measure before assuming the
  DiT is the bottleneck).
- **No reference-image conditioning support yet on CPU** (`Flux2DiT.Forward` throws
  `NotSupportedException` for reference images, a documented gap in `docs/087`) — the GPU port
  should match whatever the CPU path supports at the time it's built, not attempt to add ref-image
  support as part of a GPU-residency task (that's a separate, unrelated correctness task).

## Real blockers found before/while implementing (2026-09-19)

1. **Memory budget — full residency does not fit this machine.** Worked the real numbers before
   writing any upload code: FLUX.2's 48 single-stream blocks at `HiddenSize=6144`/`MlpRatio=3.0`
   need ≈47.1GB at FP16 (each block's `linear1` is `[55296,6144]`=340M params, `linear2` is
   `[24576,6144]`=151M params), plus ≈15.7GB for the 8 double-stream blocks — ≈63GB total, this
   machine's ENTIRE system RAM (an iGPU sharing system memory, not a discrete GPU with its own
   VRAM — CLAUDE.md rule 13). A prior commit (`4e4dc96`, a concurrent AI thread) had already built
   `Flux2GpuWeights.cs` uploading all 48 single blocks unconditionally — a real, uncaught OOM
   landmine (never wired to a call site, so never actually triggered, but would have exhausted this
   machine's memory the moment anyone instantiated it expecting FLUX.1-style full parity). **Fixed
   (commit `347aabd`)**: added an `includeSingleBlocks` parameter, default `false` — by default only
   the 8 double-stream blocks (≈15.7GB) upload to GPU; the 48 single blocks stay on the existing,
   already-fast CPU `QuantizedWeightCache` path. This is now a genuinely scoped **partial**
   GPU-residency task (double blocks only), not full residency — re-scope expectations accordingly.

2. **A required GPU shader does not exist yet.** FLUX.2's gated FFN is `SiLU(gate) * value` on a
   `2*mlpHidden`-wide up-projection split in half (confirmed against `examples/flux2/src/flux2/
   model.py`) — checked every existing GPU op in `IVisionOpsBackend`/`IImageOpsBackend` (`grep -n
   "Silu\|Multiply" src/OpenTail.Stingray.Core/I*OpsBackend.cs`) and **no generic standalone SiLU or
   elementwise-multiply op exists on GPU tensors at all**, let alone a fused gate-multiply kernel.
   FLUX.1's GPU path only has `VisionGeluInPlace` (plain GELU, no gating) because FLUX.1's own MLP
   is GEGLU-free per-element GELU, not gated. **A new compute shader is required** (GLSL, a new
   `ComputePipeline` dispatch method analogous to `AdaLNModulate`'s, and a `scripts/gen-spirv.ps1`
   recompile per CLAUDE.md rule 5) before the double-block GPU forward pass can be correctly
   implemented — this is real shader-authoring work, not a matter of wiring existing ops together.

   **RESOLVED 2026-09-19**: authored `Shaders.SiluGateMul` (new GLSL kernel, `VulkanBackend.
   SiluGateMul` dispatch method, `IVisionOpsBackend.SiluGateMul` interface method with a
   `NotSupportedException` default for non-Vulkan backends), recompiled the precompiled SPIR-V
   table via `scripts/gen-spirv.ps1` (141 shaders now, Vulkan SDK 1.4.357.0 confirmed available on
   this machine), and re-ran `VulkanPrecompiledShaderTests` (`STINGRAY_RUN_HEAVY_TESTS=1`) to
   confirm the table stays in sync — all 3 pass. New real GPU test
   (`tests/OpenTail.Stingray.Tests.Diffusion/Flux2SiluGateMulGpuTests.cs`) confirms the shader
   matches `Flux2DiT.cs`'s own CPU `GatedFfn` reference to machine precision
   (`maxDiff=1.91×10⁻⁶`, at a deliberately non-workgroup-aligned `nTokens=37` to catch any
   boundary-handling bug). Re-ran the existing `FluxGpuParityTests` suite (7 tests) to confirm zero
   regression to FLUX.1's own GPU path from touching shared files (`VulkanBackend.cs`,
   `IVisionOpsBackend.cs`) — all pass. **Blocker 2 is closed.**

3. **`AdaLNModulate`'s `isRmsNorm` flag needs to be `false` for FLUX.2, unlike FLUX.1's GPU path.**
   `AdaLNModulate` (used by FLUX.1's `DoubleBlockGpu`) takes a real `isRmsNorm` toggle dispatched
   into the shader's push constants — FLUX.1's GPU path passes `isRmsNorm: true` for every call.
   FLUX.2's real CPU path (`Flux2DiT.cs`'s `ModulateCopy`) uses `DiffusionOps.LayerNormNoAffine`
   (mean-subtracted, affine-free LayerNorm), NOT RMSNorm — confirmed by direct source read. The GPU
   port MUST pass `isRmsNorm: false` at every FLUX.2 call site, or it will silently compute the
   wrong normalization (RMSNorm skips mean-subtraction; the outputs would differ non-trivially,
   producing a real, hard-to-spot numerical bug rather than an obvious crash). Not yet verified this
   flag actually produces correct LayerNorm math on this driver when set to `false` — no test
   exercises the `isRmsNorm=false` branch anywhere in this codebase yet (checked: FLUX.1 is the only
   caller, and always passes `true`). **Verify this branch works correctly with a small isolated
   test BEFORE wiring the full double-block forward pass around it** — an untested code path in a
   shared, safety-critical shader is exactly the kind of thing that silently breaks everything
   downstream.

   **RESOLVED 2026-09-19** (`tests/OpenTail.Stingray.Tests.Diffusion/
   Flux2AdaLNModulateLayerNormGpuTests.cs`, new): two real GPU tests on this actual Vulkan driver.
   `AdaLNModulate_IsRmsNormFalse_MatchesCpuLayerNormReference` confirms `isRmsNorm: false` matches
   the CPU `DiffusionOps.AdaLNModulate` LayerNorm reference to machine precision
   (`maxDiff=7.15×10⁻⁷`). `AdaLNModulate_RmsNormAndLayerNormBranches_GenuinelyDiffer` guards against
   a trivial false-pass (e.g. a dead/no-op branch) by confirming the two branches genuinely diverge
   on a nonzero-mean input (`maxDiff=2.82`, as expected since RMSNorm skips mean-subtraction).
   **Blocker 3 is closed** — safe to build the double-block forward pass on top of this now.

4. **RoPE dispatch is likely already solved.** `Flux2RoPE.BuildContextFreqsCompact` (added in
   `4e4dc96`) already builds the compact `[nTokens, headDim/2]` one-value-per-pair table the
   existing `Flux2DRoPE` GPU shader expects, with a passing unit test confirming it matches the
   existing CPU `InterleavedRoPE` math — since the GPU rotation kernel itself only consumes a
   precomputed per-token cos/sin table (axis count is a CPU-side table-construction concern, not a
   GPU-kernel concern), this piece does NOT need new shader work, just correct wiring. Not yet
   independently re-verified this session, but no red flags found.

## Suggested implementation order — REVISED 2026-09-19 to reflect the real blockers found above

1. ~~Re-verify the CPU path is correct~~ — **DONE**, FLUX.2's Pass 1 closed 2026-09-19 (real
   512×512/20-step production run, clean coherent output, see docs/088).
2. ~~Profile where CPU time actually goes~~ — not yet done with fresh post-close numbers; worth a
   quick check before investing further GPU effort, but not currently believed to change the plan
   (the DiT denoise loop dominated every other FLUX-family model's own profile).
3. ~~Write a small, ISOLATED test exercising `AdaLNModulate`'s `isRmsNorm: false` branch~~ —
   **DONE 2026-09-19**, machine-precision match confirmed on real hardware (see blocker 3 above).
4. ~~Author the new SiLU-gated-FFN GPU shader~~ — **DONE 2026-09-19**, machine-precision match
   confirmed on real hardware, zero regression to FLUX.1's GPU path (see blocker 2 above).
5. Build `Flux2GpuWorkspace.cs` (double-block-scale buffers only — `nImg`/`nTxt`/`d`-sized
   activation/attention/modulation buffers, following `FluxGpuWorkspace.cs`'s structure) and wire a
   `Flux2DiT.ForwardGpu` that runs `img_in`/`txt_in` projection + the 8 double blocks on GPU (using
   `Flux2GpuWeights` with `includeSingleBlocks: false`), then downloads to CPU `float[]` for the
   remaining 48 single blocks + final layer via the already-working, unchanged CPU path.
6. Write a real GPU-vs-CPU parity test comparing img/txt hidden states after the double-block loop
   (before the single-block CPU handoff) — do not trust any output or timing until this passes.
7. Get a real, measured single-step GPU-vs-CPU timing comparison on THIS machine's iGPU — per the
   gating conditions, partial (double-block-only) residency could still lose to CPU on this
   hardware; measure, don't assume (CLAUDE.md rule 13).
8. If GPU wins: consider whether the SiLU-gated shader from step 4 or the `isRmsNorm=false` path
   from step 3 need the same optimization sequence FLUX.1 went through (matrix-vector fast paths,
   command-buffer batching) — don't assume the win transfers automatically.
9. Record every real measurement in `PerformanceLeague.md` and `docs/088`, following this project's
   existing documentation conventions.
10. Single-stream-block GPU residency (the other ≈47.1GB) is a SEPARATE, larger future task —
    requires either a wider-BN quantized-matmul shader variant (see docs/088 Pass 2 §2d's own
    analysis) to stay within this machine's memory budget, or a machine with real headroom. Do not
    attempt it by simply flipping `includeSingleBlocks: true` on this hardware.
