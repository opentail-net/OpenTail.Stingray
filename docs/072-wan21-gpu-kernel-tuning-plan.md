# Wan2.1 GPU Kernel Tuning Plan (2026-09-13)

## Context — what's already real, read `docs/069`/`docs/071` for the FLUX playbook first

As of commit `53839a3`, Wan2.1's Vulkan GPU-resident DiT path is **genuinely correct for the first
time** — a real bug was found and fixed where `WanModel.ForwardGpu` uploaded/downloaded only the
patch-embed and head projections to/from GPU, while the entire 30-layer transformer block loop
silently ran on plain CPU code (`TransformerBlock`, a host `float[]`/`WanWorkspace`), even though
`WanGpuWeights` already had every layer's weights genuinely resident in VRAM. `WanGpuParityTests`
was consequently comparing the CPU path against itself and always "passed" regardless of whether
GPU compute worked at all.

**This is now fixed.** `WanModel.TransformerBlockGpu` (new, `src/OpenTail.Stingray.Diffusion/Wan/
WanModel.cs`) wires every op — Q/K/V/O projections, self-attn RMS-norm, 3D-RoPE, self-attention,
cross-attention against the precomputed GPU-resident KV cache, gated FFN — through
`WanGpuWeights`/`WanGpuWorkspace`'s real device buffers, batched one Vulkan command buffer per
block. `WanGpuParityTests` is now a real, meaningful assertion and passes (GPU matches CPU to
`<1e-2` absolute tolerance). Full narrative: `PerformanceLeague.md`'s Wan2.1 section, and this
session's own history in the chat transcript that produced this doc.

## The problem, in one line

**Now that Wan2.1's GPU path does real compute, it's honestly slower than CPU, not faster.**
Real measured numbers (1-block synthetic microbench, `WanModelSyntheticMicrobenchTests`,
`numTokens=2048`, `dim=1536`, `numHeads=12`, `ffnDim=8960` — i.e. real Wan2.1-T2V-1.3B shapes):

| | CPU (`WanModel.Forward`, AVX2) | GPU (`WanModel.ForwardGpu`, Vulkan, real compute) |
|---|---|---|
| Per-block time | **1146.0ms** | **1873.2ms** |
| Ratio | — | **~1.6× slower than CPU** |

This is the exact same starting position FLUX's own GPU-resident path was in before its kernel
optimization pass (`docs/069`) — real, correct GPU compute, just not yet tuned. No kernel-level
work has been done for Wan's specific matrix shapes (`dim=1536`, `ffnDim=8960`, `numTokens=2048`) —
FLUX's SGEMM/attention kernel improvements (64×128 tiled SGEMM, 32×16 vec4 attention) were tuned
and verified against FLUX's own shapes (`d=3072`, `nSeq=1280`), and while Wan's GEMMs reuse the
exact same shared kernels, "shared kernel, different shape" is not a given win — it needs
measuring, the same discipline `docs/069` already used for FLUX.

## What was already checked before writing this doc (don't re-derive)

- **Batching granularity was already ruled out as the dominant cost for Wan**, unlike FLUX: an
  unbatched version (individual `Sgemm`/`AdaLNModulate`/etc. dispatches, no `BeginBatch`/`EndBatch`)
  measured 1920.5ms/block; wrapping the whole block in one `BeginBatch()`/`EndBatch()` (the fix
  that mattered enormously for FLUX) only improved it to 1873.2ms/block — a ~2.5% difference, not
  the dominant lever here. **Do not spend time on batching-granularity variants for Wan** without
  new evidence contradicting this.
- **This means the bottleneck for Wan is (like FLUX's own remaining gap in `docs/069`) real GPU
  execution time itself**, not host-side dispatch/sync overhead. The same conclusion FLUX's own
  `STINGRAY_PROFILE_GPU_SPLIT` measurement reached.
- **3D-RoPE correctness is real, not a shortcut**: Wan's RoPE reuses FLUX's `Flux2DRoPE` GPU kernel
  verbatim (via a new `WanRoPE.Compute3DRoPECompact` frequency-table builder producing the same
  compact `[tokens, headDim/2]` layout `Flux2DRoPE` expects) — this is *not* a placeholder or an
  approximation, it's an intentional, verified reuse: the actual interleaved-pair rotation math is
  identical between FLUX and Wan, only the frequency-table construction differs (FLUX: 16/56/56
  axis split with an identity axis; Wan: 44/42/42, no identity axis). The pre-existing `RoPE3D` GPU
  kernel was checked and found to use split-half (NEOX) pairing with an equal `headDim/6` split —
  genuinely the wrong convention for Wan — so it was deliberately NOT used. Do not "simplify" by
  switching to `RoPE3D` — that would silently reintroduce a real, previously-fixed bug (see
  `WanRoPE.cs`'s own class doc comment for the 2026-08-31 history of this exact mistake).

## Recommended approach — mirror `docs/069`'s methodology exactly

**Do not guess at kernel changes.** The same discipline that worked for FLUX applies here:

1. **Add real per-shader-type GPU timing** for the Wan block (mirror
   `STINGRAY_PROFILE_GPU_SPLIT`'s existing submit+wait accounting, or better, real Vulkan
   timestamp queries per dispatch — see `docs/069`'s own Task 1 for the FLUX version of this exact
   ask). Aggregate into a table:

   ```
   Shader                    Dispatches   GPU ms   % of block time
   Sgemm (Q/K/V/O/FFN)            ?          ?             ?
   MultiHeadAttentionTiled        ?          ?             ?
   AdaLNModulate                  ?          ?             ?
   RmsNormBatched                 ?          ?             ?
   Flux2DRoPE                     ?          ?             ?
   ScaleGateAdd                   ?          ?             ?
   LayerNormGpu                   ?          ?             ?
   Other                          ?          ?             ?
   ```

   This tells you where within the 1873ms Wan is actually spending time — do not assume it's the
   same distribution as FLUX's (Wan's `dim=1536` vs FLUX's `d=3072`, half the width, means GEMMs
   are shaped very differently even though they route through the identical kernel).

2. **Build a real GEMM microbenchmark at Wan's actual shapes** (mirror `docs/069`'s Task 2
   verbatim, just with Wan's numbers): `[2048, 1536] × [1536, 1536]` (Q/K/V/O and cross-attn
   projections), `[2048, 1536] × [1536, 8960]` and `[2048, 8960] × [8960, 1536]` (FFN up/down). The
   already-optimized 64×128 tiled `SgemmF16` kernel hit ~611 GFLOP/s on FLUX's own shapes
   (`d=3072`-family matrices) — measure whether it holds that throughput on Wan's narrower-but-
   still-large shapes, or whether the tile/register-block dimensions interact badly with a
   `K=1536`/`K=8960` reduction dimension specifically. A GEMM tuned for one aspect ratio does not
   automatically transfer — this is exactly the kind of thing a real microbenchmark (not
   intuition) settles.

3. **Check `MultiHeadAttentionTiled128` (headDim=128, same as FLUX — Wan's
   `dim/numHeads = 1536/12 = 128`) at Wan's real attention shapes**: self-attention is
   `qSeq=kvSeq=2048` (vs FLUX's `1280`), cross-attention is `qSeq=2048, kvSeq=numTxtTokens` (Wan's
   text-token count, check the real UMT5 sequence length used — likely different from FLUX's
   T5-256). Larger `qSeq`/`kvSeq` than FLUX's own tuning target could matter for a tiled kernel's
   cache/shared-memory behavior.

4. **Only after 1-3 produce real numbers, consider kernel changes** — same warning `docs/069`
   already gave for FLUX: don't touch tile sizes, register-block shapes, or workgroup sizes
   speculatively. If the per-shader breakdown says GEMM dominates, attack GEMM (and specifically
   check whether it's compute-bound, bandwidth-bound, or occupancy-bound for `K=1536`/`K=8960`
   shapes, the same three-way split `docs/069` recommended). If attention dominates, that's a
   separate, real finding.

## Practical constraints (same as `docs/069`/`docs/071`, repeated because they matter)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code** — the CPU `WanModel.Forward` path stays; this is
  about tuning the now-real `ForwardGpu` path, not replacing anything.
- **Ask before committing** — this doc's own prerequisite work (`53839a3`) was committed with
  direct authorization; confirm current expectations before committing further, don't assume.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) is healthy before a benchmark; a real incident earlier this
  session came from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **Use the cheap synthetic-scale tests for iteration**: `WanModelSyntheticMicrobenchTests`
  (`WanModel_SingleBlock_RunsCleanly_WithWanGpuWorkspace`, ~14s/run including 2 warmup + 5 timed
  iterations) and `WanGpuParityTests` (~1s/run) both use a `SyntheticWeightLoader` with real Wan2.1
  dimensions but random weights — no need to download the ~3GB real checkpoint (still not present
  locally, per `models/wan2.1/` only having VAE+UMT5) to do this kernel-tuning work; both tests
  already give real, meaningful, fast signal at the real production shapes.
- **A full real end-to-end Wan2.1 20-step/2-frame generation has never been run on Vulkan** (real
  checkpoint not downloaded) — this plan's scope is the per-block synthetic microbench improving
  first; a full pipeline run is a separate, later step once per-block GPU time is competitive with
  or better than CPU's 1146ms/block.

## Success criterion

Get `WanModel.ForwardGpu`'s per-block time **below** `WanModel.Forward`'s CPU time (currently
1146.0ms/block) — that's the bar for the GPU path to be worth using at all here, before even
comparing against any external reference. Update `PerformanceLeague.md` honestly with whatever is
actually measured; this session already showed real per-block GPU time (1873.2ms, ~1.6× slower
than CPU) rather than assume the shared-kernel win from FLUX would transfer — keep that same bar.

---

## Completion & Benchmark Results (2026-09-13)

### Optimization Root Causes & Fixes
1. **FP16 Weight Uploading (`UploadWeight`)**:
   - `WanGpuWeights` was uploading linear weights as `DType.Float32` via plain `Upload()`, which caused `VulkanBackend.Sgemm` to fall back to the scalar 32×32 `SgemmF32` shader (~106 GFLOP/s).
   - Added `UploadWeight` converting FP32 weights to FP16 (`UploadHalf`) when supported by hardware (`BestSgemmPrecision == SgemmPrecision.Fp16`). All projection GEMMs (`SelfAttnQ/K/V/O`, `CrossAttnQ/K/V/O`, `Ffn0/Ffn2`, `PatchEmbedding`, `HeadWeight`) now route directly to the 64×128 tiled vectorized `SgemmF16` kernel (~611 GFLOP/s).
2. **Eliminated Per-Block Dynamic VRAM Allocations**:
   - `LayerNormGpu` in `TransformerBlockGpu` was dynamically allocating and freeing a new `GpuBuffer` per block iteration. Added in-place `LayerNormGpu(ws.NormedCross, ws.X, bw.Norm3Weight, bw.Norm3Bias, numTokens, d)` writing directly to the preallocated workspace buffer with zero memory allocations.
3. **Eliminated Host String Lookups & Heap Allocations**:
   - Cached `HostModulation` in `BlockWeights` and `HostHeadModulation` in `WanGpuWeights` during initialization, eliminating per-block dictionary lookups and GC allocations.

### Measured Benchmark Comparison (`WanModelSyntheticMicrobenchTests`)
Single-block forward (`numTokens=2048`, `dim=1536`, `numHeads=12`, `ffnDim=8960` — real Wan2.1 production shapes):

| Backend / Path | Initial Baseline | Phase 1 Tuning (FP16 GEMM + In-Place Workspace) | vs CPU Baseline | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **CPU (`WanModel.Forward`, AVX2)** | 1146.0 ms | 1312.2 ms | 1.00× | baseline |
| **Vulkan GPU (`WanModel.ForwardGpu`)** | 1873.2 ms (1.63× slower) | **738.7 ms** | **1.78× FASTER than CPU** | 🚀 **2.54× speedup** |
| **Projected 30L × 40-step DiT Run** | ~2248s (~37.5 min) | **~886.4s (~14.8 min)** | — | **~2.54× reduction** |

- **Numerical Parity**: `WanGpuParityTests` passes 100% with `maxDiff < 1e-2` against CPU reference.
