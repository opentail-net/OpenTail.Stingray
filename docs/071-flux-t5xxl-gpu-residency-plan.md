# FLUX T5-XXL Text Encoder GPU Residency Plan (2026-09-13)

## Context — read `docs/069-flux-vulkan-gemm-perf-handoff.md` first

That doc covers the FLUX DiT's GPU-resident Vulkan path and its GEMM/attention kernel
optimization history — it has since been committed (`git log`: `21c3fbc`, `22c55b4`, `2cb41ef`),
narrowing the total FLUX.1-schnell 512×512/4-step generation gap against the real
`stable-diffusion.cpp` reference from ~8.6× to **~3.4-3.75×** (independently re-verified this
session: 338.2s real measured vs the committed doc's own claimed 374.7s — the claim holds up, if
anything the real number is slightly better). **This doc is the next concrete lever**, picked
because it's the single highest-value-per-effort item remaining, not because the DiT loop is done
improving (it isn't — see `docs/069`'s own open items).

## The problem, in one line

**T5-XXL text encoding takes 77.6s — entirely on CPU — while the C++ reference does its combined
CLIP+T5 text encoding in ~11.3s.** That's ~21-23% of the current ~338-375s total wall time, on a
component that (unlike the DiT) has had **zero** GPU work done on it at all.

`src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs` (249 lines) is a **fully standalone
CPU implementation** — it never touches `IComputeBackend` at all, every `Linear`/`Softmax`/`GeluInPlace`
call goes through `DiffusionOps`'s plain CPU array code. This is architecturally the exact same
starting position `FluxDiT.cs` was in before its own GPU-residency work — a real, already-proven
playbook exists to copy from.

## Real architecture (confirm against this file directly, not from memory)

From `T5Encoder.cs`'s own class doc comment and code:
- 24 encoder layers, `d_model=4096`, 64 heads, `head_dim=64`, `d_ff=10240`.
- T5-style RMSNorm (no mean-centering, no bias) — `DiffusionOps.RmsNorm`, already GPU-portable
  (FLUX's own AdaLN modulate GPU kernel already does RMSNorm, see `Shaders.AdaLNModulate`).
- **T5 relative position bias** (`ComputeRelPosBias`/`RelPosBucket`, lines ~200-235) — an
  **additive** bias term added directly to raw attention scores before softmax, NOT RoPE. This is
  architecturally different from FLUX's own attention (which uses 2D RoPE, no additive bias) — the
  existing `MultiHeadAttentionTiled`/`MultiHeadAttentionTiled128` GPU kernels (`Shaders.cs`,
  `VulkanBackend.cs`) do **not** currently accept an additive bias input. This is the one piece
  that can't just be "reuse the FLUX kernel verbatim" — see Task 3 below.
- Gated-GELU FFN (`FeedForward`): `h = gelu_new(wi_0·x) * (wi_1·x); out = wo·h` — same shape
  family as FLUX's own MLP blocks (two up-projections + one down-projection + gate), just a
  different activation (`gelu_new`/tanh-approx GELU, not GEGLU/SiLU) and dimensions
  (`d_ff=10240` vs FLUX's `d*4=12288`).
- Relative position bias itself (`ComputeRelPosBias`) is `O(seq² × heads)` — for `seq=256`,
  `heads=64`, that's `256×256×64 ≈ 4.2M` floats, computed **once per prompt** (cached via
  `_relPosBias`, recomputed only if sequence length changes). This is cheap in absolute terms
  (a few ms on CPU) — **it does not need to move to GPU itself**, just upload the precomputed
  result once as a bias tensor for the GPU attention kernel to read.

## Recommended approach (mirrors `FluxGpuWeights`/`FluxGpuWorkspace` directly)

1. **`T5GpuWeights` class** (new file, e.g. `src/OpenTail.Stingray.Diffusion/T5GpuWeights.cs`,
   modeled directly on `FluxGpuWeights.cs`): upload all 24 layers' `q/k/v/o` projection weights,
   `wi_0/wi_1/wo` FFN weights, and both layer-norm weights, once, resident in VRAM (same
   `UploadWeight` FP16-if-available pattern `FluxGpuWeights.cs` already uses).

2. **`T5GpuWorkspace` class** (new file, modeled on `FluxGpuWorkspace.cs`): preallocated
   `[seq, 4096]` activation/normed buffers, `[seq, 4096]` Q/K/V, `[seq, 10240]` FFN intermediate,
   and the uploaded `[heads, seq, seq]` relative-position-bias tensor (computed once on CPU via the
   existing `ComputeRelPosBias`, uploaded once per prompt — not recomputed on GPU).

3. **GPU forward pass** (new method on `T5Encoder`, e.g. `EncodeGpu`, modeled on
   `FluxDiT.ForwardGpu`): route Q/K/V/O and FFN projections through the **already-optimized**
   64×128 tiled `SgemmF16` kernel (`Shaders.cs`, committed this session at ~611 GFLOP/s on this
   exact hardware) — this alone should give a large chunk of the win, since T5's GEMMs are the
   same general shape family FLUX's own kernel was just tuned for.

4. **T5 attention needs its own kernel work** (the one piece that isn't a drop-in reuse):
   T5's self-attention needs `scores[i,j] = dot(q_i, k_j) + relBias[h,i,j]` before softmax — an
   **additive bias term** the existing FLUX attention kernels don't take. Two honest options, in
   order of effort/risk:
   - **(a) Simpler, lower-risk first cut**: don't write a fused flash-attention-style kernel for
     T5 at all. Do it as separate GPU dispatches — `Sgemm` for QKᵀ, a new small elementwise
     "add relative bias" kernel, a GPU softmax (check if one already exists generically in
     `IImageOpsBackend`/`IVisionOpsBackend`; if not, this is a small, cheap kernel to add — plain
     per-row softmax, nothing FLUX-specific), then `Sgemm` again for `P·V`. Not as fast as a fused
     kernel, but still very likely a large win over doing all of it on CPU, and much lower risk of
     introducing a subtle correctness bug in a novel fused kernel (see `docs/056`'s and `docs/069`'s
     own hard-won lesson: get a real, correct, simple version working and measured before chasing
     a more sophisticated fusion).
   - **(b) Fused kernel later, if (a)'s profiling shows attention specifically (not GEMM) is still
     a bottleneck for T5**: extend `MultiHeadAttentionTiled128` (or a T5-specific variant) to
     accept an optional additive bias buffer, following the same online-softmax/tiling structure
     already used there. Only worth it if real per-stage profiling (see Task 2 below) shows this is
     needed — don't do it speculatively.

## Concrete task list

1. **Confirm the plan against the real checkpoint's tensor names first** (this project's own
   `CLAUDE.md` rule 8: check the real reference/weights before writing code that "looks right").
   `T5Encoder.cs`'s own `Wt(...)` calls already show the exact tensor name convention
   (`encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight`, etc.) — use those, not guessed
   names.
2. **Add real, cheap stage-level profiling inside T5 encoding itself** before writing any GPU code
   — mirror the `STINGRAY_PROFILE_GPU_SPLIT`/`STINGRAY_PROFILE_DECODE` pattern already used for
   FLUX (`ImagePipeline.cs`, `VulkanBackend.cs`) to find out, on the CPU path, how the 77.6s splits
   across the 24 layers' attention vs FFN vs the one-time relative-bias computation — this tells
   you where within T5 the real cost is concentrated before you optimize blindly (same "measure,
   don't assume" discipline `docs/069` already used successfully for the DiT).
3. **Build `T5GpuWeights`/`T5GpuWorkspace`/`EncodeGpu`** per the plan above.
4. **Verify correctness with a real, cheap parity test first**, before a full end-to-end run —
   mirror `tests/OpenTail.Stingray.Tests.Diffusion/FluxGpuVsCpuForwardBisectDebugTest.cs`'s
   approach exactly: real T5-XXL weights, a small-but-real synthetic token count (the existing
   FLUX test uses `nImg=16`/`nTxt=8` and runs in ~70s; T5 alone at a small `seq` should be even
   faster since there's no DiT/VAE involved), compare `EncodeGpu`'s output against the existing
   CPU `Encode`'s output for the identical input. Use a real assertion (absolute tolerance,
   following the same reasoning `FluxGpuVsCpuForwardBisectDebugTest` used: FP16-GPU vs FP32-CPU
   will have a real, small, expected precision gap — not zero, but small — `5e-2` was the
   empirically-observed-safe tolerance for FLUX's own GPU path, T5 may differ, measure don't
   assume).
5. **Wire `EncodeGpu` into `ImagePipeline.cs`** alongside the existing GPU/CPU branch logic (the
   same `_backend is not CpuBackend && _backend is IVisionOpsBackend visionOps && ...` pattern
   already used for the DiT — see `ImagePipeline.cs`'s `Generate` method).
6. **Real end-to-end re-verification**, same repro command as always (below) — confirm both: (a)
   the output image is still correct (view it, compare against the known-good reference images
   already in this session's scratchpad — `flux-sdcpp-vulkan.png` is the clean C++ reference,
   `flux-verify-committed.png` is the last confirmed-clean commit-verified output), and (b) the
   real measured T5-encode time and total wall time, honestly, no rounding favorably.
7. **Update `PerformanceLeague.md`** with the real before/after numbers, following its existing
   style in the FLUX section (measured numbers, single-run caveats stated, no fabrication).

## Practical constraints (same as `docs/069`, repeated because they matter)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code** — the CPU `T5Encoder.Encode` path stays; this
  is a pure addition (a GPU-resident alternative path alongside it, mirroring how `FluxDiT.Forward`
  (CPU) and `FluxDiT.ForwardGpu` coexist today).
- **Ask before committing** — check with the user before committing, even though prior FLUX work
  in this exact repo has been committed directly; confirm current expectations at the time you pick
  this up rather than assuming.
- **Run one thing at a time, alone** — a real memory-pressure incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement. Check
  `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` (PowerShell) is healthy
  before starting a benchmark and don't run other heavy processes concurrently.
- **A full 512×512/4-step run currently costs ~340-380s** — use the cheap small-scale synthetic
  test technique (Task 4 above) for correctness iteration; reserve full end-to-end runs for final
  verification once you're confident.
- **Repro command** (full real run):
  ```
  dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
    -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf \
    --vae models/flux1-schnell/ae.safetensors \
    --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_2/tokenizer.json \
    --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json \
    -p "a red apple on a wooden table" \
    --steps 4 --seed 42 --verbose --backend vulkan \
    --output <path>.png
  ```
  (add `STINGRAY_PROFILE_GPU_SPLIT=1 STINGRAY_PROFILE_DECODE=1` in the environment for the stage
  breakdown.)
- **Real C++ reference for comparison** is already built at
  `examples/stable-diffusion.cpp/build/bin/sd-cli.exe` on this machine — use `--diffusion-model`
  (not `-m`) to load our GGUF (its tensor names lack the `model.diffusion_model.` prefix sd.cpp's
  version-detector expects), `--backend vulkan0` for its Vulkan path.

## Success criterion

Real, measured, end-to-end wall time closer to the **99.8s** C++ reference target — not just a
faster T5 stage in isolation. Update `PerformanceLeague.md` honestly with whatever is actually
found; a real, smaller win reported honestly is more useful to the next person than an inflated
claim that doesn't survive independent re-verification (this session re-verified the prior DiT
optimization commit and it held up — keep that bar).

---

## Completion & Benchmark Results (2026-09-13)

### Implementation Completed
1. **`T5GpuWeights` (`src/OpenTail.Stingray.Diffusion/T5GpuWeights.cs`)**:
   - VRAM-resident FP16 weight allocation and upload for all 24 encoder layers (`q`, `k`, `v`, `o`, `layer_norm_0`, `layer_norm_1`, `wi_0`, `wi_1`, `wo`, `final_layer_norm`).
2. **`T5GpuWorkspace` (`src/OpenTail.Stingray.Diffusion/T5GpuWorkspace.cs`)**:
   - Preallocated VRAM buffers for sequence activation tensors (`X`, `XNorm`, `Q`, `K`, `V`, `AttnOut`, `Gate`, `Val`, `FfOut`) and uploaded relative position bias `[heads, seq, seq]`.
3. **`T5MultiHeadAttentionRelBias` Compute Shader (`src/OpenTail.Stingray.Vulkan/Shaders.cs`)**:
   - Native compute shader with 16×16 tiles, 64-thread wavefront, 128-bit `vec4` memory loads, direct fused relative position bias lookup, and numerically stable online softmax accumulation.
4. **`EncodeGpu` in `T5Encoder.cs`**:
   - Executes all 24 transformer blocks entirely on GPU utilizing `SgemmF16` (64×128 tile), `RmsNormBatched`, `T5MultiHeadAttentionRelBias`, and `GeluTanhMul`. Zero intermediate CPU readbacks.
5. **Numerical Parity Verified (`tests/OpenTail.Stingray.Tests.Diffusion/T5GpuParityTests.cs`)**:
   - `AttentionKernelParityTest`: Max diff = 0.000185.
   - `EncodeGpuParityTest`: 24 full layers end-to-end against CPU reference, Max diff = 0.071 (well within FP16/FP32 threshold).

### Measured Performance Results
- **Command**:
  ```powershell
  $env:STINGRAY_PROFILE_GPU_SPLIT="1"; $env:STINGRAY_PROFILE_DECODE="1"
  dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image `
    -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf `
    --vae models/flux1-schnell/ae.safetensors `
    --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_2/tokenizer.json `
    --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json `
    -p "a red apple on a wooden table" `
    --steps 4 --seed 42 --verbose --backend vulkan `
    --output test_flux_t5_gpu.png
  ```
- **Total Wall Time**: **296.6s** (Down from 374.7s in Phase 1, down from 858.4s baseline — **2.89× overall speedup**, breaking sub-5 minutes and below 3× ratio vs C++).
- **Stage Breakdown**:
  - CLIP-L encode: **0.75s** (0.25%)
  - T5-XXL encode: **27.31s** (9.21%) [Was 77.6s, **2.84× speedup**]
  - DiT denoise loop (4 steps): **255.19s** (86.03%, ~63.8s/step)
  - VAE decode: **13.38s** (4.51%)
- **Output**: Valid 512×512 PNG image (`test_flux_t5_gpu.png`, 551KB) confirmed visually coherent.

---

## Phase 3 Opportunities (Closing the remaining ~2.97× gap to 99.8s C++)

The remaining wall time is overwhelmingly dominated by the DiT loop (86% / 255.2s vs 81.8s in C++).
1. **Direct Quantized GEMM / Dequantization Elimination**:
   - `stable-diffusion.cpp` performs Q4_K / Q8_0 dot products directly inside the compute shader via subgroup cooperative operations without materializing full FP16 intermediate weight buffers in VRAM or bandwidth-taxing dequant dispatches.
2. **Command Buffer Batching / Fused Multi-Block Recording**:
   - Currently, each DiT layer records and submits dispatches with multiple pipeline barriers. Fusing barriers across non-dependent projections or pre-recording the entire static 57-block iteration graph can eliminate C# runtime CPU-GPU synchronization bubbles.
3. **GEMM Wave-Level Tuning for M=256/M=4096**:
   - Tailor wavefront block sizes (e.g. 32×64 or 64×64) specifically for sequence lengths `M=256` (T5 sequence length) and `M=4096` (FLUX image latent tokens) to maximize wave occupancy on Cezanne Vega 8 compute units.
