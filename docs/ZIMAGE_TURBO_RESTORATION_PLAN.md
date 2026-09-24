# Z-Image-Turbo Restoration & Anti-Conflation Plan

**Document Path**: `docs/ZIMAGE_TURBO_RESTORATION_PLAN.md`  
**Date**: September 24, 2026  
**Status**: Ready for Verification / External Review

---

## 1. Executive Summary & Objective

The objective of this task is to bring **Z-Image-Turbo** in `OpenTail.Stingray` back to life so that it produces clean, coherent, artifact-free images (matching reference images `z-image-turbo_red-apple-on-white-table_CPU-256x256-4steps_FIXED-2026-09-12.png` and `zimage-vulkan-fixed.png`), completely eliminating the 16×16 quilted mosaic pebble noise regression.

### Primary Constraints
1. **Zero Regression**: Under no circumstances may any changes regress other model families in Stingray (`FLUX.2`, `SD3.5`, `Wan 2.1`, `AceStep`, `ControlNet`, `TinyVAE`, `StableAudio`).
2. **Strict Architectural Separation (No Conflation)**: Z-Image must never borrow, call, or share kernels, caches, or state from Wan, FLUX, SD3, or other models. Model codebases must remain strictly decoupled.
3. **Maintain Memory Optimizations**: Keep working set memory reduction (~3.86 GB down from 6.7 GB via prompt-encoder unloading after embedding).
4. **Fast Iteration**: Use the Vulkan GPU resident test (~183 seconds per 256×256 run) for rapid verification instead of the ~31-minute CPU loop.

---

## 1a. Verification Outcome (2026-09-24) — ACTUAL root cause

Running Step 1 with only the fixes in §4 applied **still produced the mosaic** (Vulkan 256×256:
latent `std=0.4845`, range `[-2.04, 2.01]`). The fixes in §4 only touch CPU-only code (the CPU
`SelfAttention` and the CPU branch of `MatQ`), and GPU residency is disabled (`ZImageGpuResidencyRealScaleBugFound=false`),
so the Vulkan per-op path never ran any of them.

**Real root cause**: `DiffusionOps.SinusoidalTimestepEmbedding` gained a `flipSinToCos` parameter after
`235fab2` (added for Wan/LTX/Qwen Image/HunyuanVideo/FLUX.2, which all pass `flipSinToCos: true`).
The new `false` default switched the output from the old `[cos, sin]` to `[sin, cos]`. `ZImageDiT.TimestepEmbed`
was the only caller that didn't pass the flag, so its timestep conditioning was silently scrambled and every
adaLN modulation came out wrong. This affects the CPU and GPU paths equally.

**Fix**: `ZImageDiT.cs` now passes `flipSinToCos: true`, matching Z-Image's `cat([cos, sin])` and the old behavior.
This change is local to Z-Image; no shared code changed.

**Result (Vulkan per-op path, 256×256, 4 steps, seed 42, real Q4_0 weights, 160.5s)**: latent `std=1.4481`,
range `[-3.98, 4.30]` (healthy band). The image shows a clean red apple with a stem and specular highlights on a
white table against a dark background, with no grid or pebble artifacts. The file is `docs/diffusion-samples/zimage_vulkan_restored.png` (local-only).

---

## 2. Archaeological Root Cause Analysis

> **Caveat (2026-09-24)**: A and B below are the plan's original hypotheses. **Neither was shown to cause
> the mosaic**; §1a has the real cause. (A) **Disproven and reverted**: a 2026-09-24 benchmark on
> Z-Image's real attention shapes (30 heads × 128) found `WanAttention.TiledMultiHeadAttention` matches the
> per-head loop to max |diff| ~1e-6. Median time at 320 tokens was 13.3ms vs 50.7ms; at 1056 tokens,
> 124ms vs 195ms; at 4128 tokens, 12.0s vs 8.9s, but the per-head loop needs a ~2 GB score buffer there.
> `ZImageDiT` uses `WanAttention` again, and §3 rule 1's attention-kernel ban no longer applies.
> (B) `SimdKernels.MatMulBatched` already defaults to `allowQ8: false`, so this simply restores `235fab2`'s
> exact CPU matmul path; the measured CPU time (226s) is in line with 2026-09-12's 194–232s.

### A. The Conflation Regression (Commit `ffa2681`)
On Sunday, Sep 13, 2026, commit `ffa2681` (`feat(diffusion): accelerate CPU DiT paths with FlashAttention, context caching, and full numerical parity`) attempted to apply optimizations across Wan 2.1, HunyuanVideo, SD3, and Z-Image concurrently. In that commit:
1. **Replaced Z-Image Attention with Wan Attention**: `ZImageDiT.cs` had its native, mathematically verified SIMD dot-product attention loop deleted and replaced with `Wan.WanAttention.TiledMultiHeadAttention(q, k, v, attn.AsSpan(), nTok, nTok, nHeads, headDim)`. This introduced memory transpose and tile stride mismatches that broke Z-Image's per-head attention math.
2. **Stale Text Context Caching**: Introduced `_cachedRefinedTxtHid` and `_cachedTxtEmbeds` caching in `ZImageDiT.cs`, bypassing the per-step embedding/context refinement pass.

### B. Int8-Activation Quantization Corrupting AdaLN Vectors
In `src/OpenTail.Stingray.Diffusion/QuantizedWeightCache.cs` (lines 60–73), `Linear` defaults to `allowQ8: true`. When activations have wide dynamic range (such as diffusion timestep embeddings, adaLN scale/gate modulations, and sandwich RMSNorm outputs), int8 activation quantization introduces catastrophic truncation error. In Z-Image, this caused the modulation vector to destabilize, producing the characteristic 16×16 grid/pebble pattern.

### C. The 64×64 Fallacy vs 256×256
As documented in `docs/diffusion-samples/README.md`:
- At 64×64, the latent grid is only 8×8 tokens.
- The FLUX/Z-Image VAE decoder has an effective receptive field of 6–8 latent pixels around borders.
- Consequently, in a 64×64 direct decode, 100% of the image is submerged in edge-boundary decay, yielding dark brown/green mud even when DiT latents are numerically correct.
- `zimage-vulkan-fixed.png` (which depicts a clean apple with sub-pixel specular highlights and a stem) was generated at 256×256 / 512×512 and downscaled, **not** generated from an unscaled 64×64 latent.
- **Verification Rule**: All visual verification must be executed at **256×256** (32×32 latent), where the center of the image is completely free of boundary decay.

### D. Numerical Latent Invariants
- **Healthy clean apple (Reference)**: `std ≈ 1.34`, `range ∈ [-3.5, 4.5]`.
- **Corrupted mosaic noise**: `std ≈ 2.23`, `range ∈ [-10.4, 9.4]`.

---

## 3. Strict Architectural Boundary Rules (Anti-Conflation Policy)

To ensure this regression never happens again, the following rules are enforced:

1. **Zero Cross-Model Kernel Sharing**:
   - `WanAttention` is reserved strictly for `Wan 2.1`.
   - `ZImageDiT` must maintain its own dedicated, isolated SIMD attention kernel.
   - Do not call `Wan` or `Flux` kernels from `ZImage`, and do not call `ZImage` kernels from other pipelines.
2. **Explicit Disabling of Int8-Activation Quantization**:
   - CPU `MatQ` calls in `ZImageDiT.cs` must explicitly pass `allowQ8: false` when calling `SimdKernels.MatMulBatched`.
3. **No State Caching Across Pipeline Steps**:
   - Text conditioning and context refiners must execute cleanly per step unless mathematically invariant and proven against ground truth.

---

## 4. Work Completed to Date

The following fixes have been applied and tested for clean compilation (0 errors, 0 warnings):

1. **`src/OpenTail.Stingray.Diffusion/ZImageDiT.cs`**:
   - Eradicated all calls to `Wan.WanAttention`.
   - Restored the verified per-head SIMD dot-product attention loop with per-head Q/K RMSNorm and RoPE.
   - Removed `_cachedRefinedTxtHid` text context cache; `txtHid` is now freshly embedded and refined via `context_refiner` each step.
   - Replaced `_quantizedCache.Linear` in the CPU path of `MatQ` with direct `SimdKernels.MatMulBatched(..., allowQ8: false)` to eliminate activation quantization noise.
   - Restored test hooks (`ApplyBlockForTest`, `ApplyBlockGpu`, `OnMainBlockOutputCpu`, `OnMainBlockOutputGpu`, `RunMainLayersCpuForTest`, `RunMainLayersGpuForTest`).
   - Added explicit architectural isolation docstrings to the class header and `SelfAttention` method.

2. **`src/OpenTail.Stingray.Diffusion/Wan/WanAttention.cs`**:
   - Added explicit architectural boundary docstring forbidding external models from calling `WanAttention`.

3. **`src/OpenTail.Stingray.Diffusion/ZImagePipeline.cs`**:
   - Verified flow matching scheduler direction (`sign: 1`).
   - Retained text encoder memory unloading (`_encoder.Dispose(); GC.Collect(); TrimWorkingSet();`), preserving working set memory at ~3.86 GB.

4. **`tests/OpenTail.Stingray.Tests.Diffusion/ZImageGpuResidencyEndToEndTests.cs`**:
   - Configured `ZImagePipeline_Gpu_256x256_4Steps_RealApple` to output directly to `docs/diffusion-samples/zimage_vulkan_restored.png` and log pre-VAE latent metrics (`std`, `min`, `max`).

---

## 5. Next Steps for Verification & Hand-off

Any developer or AI reviewing this task should follow this exact sequence:

### Step 1: Run Fast 256×256 GPU Verification (~3 minutes)
Run the GPU residency test:
```powershell
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *ZImagePipeline_Gpu_256x256_4Steps_RealApple*
```
**Acceptance Criteria**:
- Test passes within ~185 seconds.
- Latent metrics logged: `std ≈ 1.34` (healthy), NOT `std ≈ 2.2` (mosaic noise).
- Output image saved to: `docs/diffusion-samples/zimage_vulkan_restored.png`.

### Step 2: Visual Inspection
Inspect `docs/diffusion-samples/zimage_vulkan_restored.png` and compare against:
- `docs/diffusion-samples/z-image-turbo_red-apple-on-white-table_CPU-256x256-4steps_FIXED-2026-09-12.png`
- `docs/diffusion-samples/zimage-vulkan-fixed.png`
**Acceptance Criteria**:
- Crisp red apple on a clean white surface.
- Zero 16×16 tiled grid or pebble noise artifacts.

### Step 3: Run CPU Verification
Run the CPU 256×256 test:
```powershell
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *ZImagePipeline_Cpu_256x256_4Steps_RealApple*
```
**Acceptance Criteria**:
- Output matches the GPU output within normal floating-point tolerance.

### Step 4: Regression Test Other Models
Verify that no other models in the repository were affected:
```powershell
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *Flux2*
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *Sd3*
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *Wan*
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj --filter-method *AceStep*
```
**Acceptance Criteria**:
- 100% passing tests across all other diffusion models.

### Step 5: Clean Up & Documentation
- Revert or finalize any diagnostic hooks added to test files.
- Document results in `PerformanceLeague.md` under Z-Image-Turbo benchmarks.
