> **STATUS 2026-09-27.** Still open, target not met, but the gap has closed substantially since this
> was written: `PerformanceLeague.md`'s 2026-09-25 "register-tiled flash attention (headDim=128)" row
> measured **132.0s** total (DiT 23.5s/step, down from the 374.7s this doc's own body cites), vs this
> doc's **99.8s** target — now **~1.32x** slower, not the ~3.75x this doc's numbers implied. Re-check
> against `PerformanceLeague.md` before assuming either the 374.7s or the 99.8s gap size below is
> still current.

# FLUX.1-schnell Vulkan GEMM/Attention Performance Handoff (2026-09-13)

## Where things stand

FLUX.1-schnell's Vulkan GPU-resident path (`FluxDiT.ForwardGpu`/`FluxGpuWeights`/`FluxGpuWorkspace`,
`src/OpenTail.Stingray.Diffusion/`) is **numerically correct** as of this handoff — both the
long-standing "repeating tiled background" artifact (see `docs/done/056-flux-tiling-artifact-handoff.md`
Round 9) and an earlier NaN/black-image regression in the GPU-resident path itself are fixed and
verified with real end-to-end runs producing clean, coherent, on-prompt images matching the real
C++ reference's quality. **This is now a pure performance problem, not a correctness one.**

Real, measured numbers (512×512, 4 steps, seed 42, same checkpoint/prompt, same AMD Cezanne iGPU
throughout — see hardware correction below):

| | Ours (Vulkan, GPU-resident) | `stable-diffusion.cpp` (real GGML reference, Vulkan) | Ratio |
|---|---|---|---|
| CLIP-L encode | 0.71s | ~part of 11.29s combined clip+t5 | — |
| T5-XXL encode | 77.6s | ~part of the same 11.29s | slower |
| **DiT denoising loop (4 steps)** | **756.1s (~189s/step)** | **81.81s (~20.45s/step)** | **~9.2× slower** |
| VAE decode | 12.6s | 8.78s | ~1.4× slower |
| **Total** | **858s** | **99.8s** | **~8.6× slower** |

Within our DiT loop, real Vulkan-fence-level instrumentation (`STINGRAY_PROFILE_GPU_SPLIT=1`) shows
**686.4s of 756.1s (91%) is genuine `vkQueueSubmit`+GPU-execute+`vkWaitForFences` time**, across 1027
dispatches. **This rules out excessive CPU-side command-buffer recording or memory-barrier overhead
as the dominant cause** — the GPU itself is taking a very long time to execute the actual compute.
The entire ~8.6× gap is concentrated in the DiT loop; VAE decode and CLIP are already close to the
reference's speed.

Full narrative and every number's provenance: `PerformanceLeague.md`'s FLUX section (search
"granular stage profiling"), and `docs/done/056-flux-tiling-artifact-handoff.md`'s Round 9.

## Hardware correction (important — read before optimizing)

This machine's iGPU is an **AMD Ryzen 7 5700G "Cezanne" integrated GPU** — 8 compute units,
**GCN/Vega-derived architecture**, not RDNA2. Its native wavefront width is **64**, not RDNA2's
32/64-configurable-but-commonly-32 convention. Do not apply RDNA2-specific tuning assumptions
(e.g. wave32-first heuristics) — this is older-generation GCN.

One relevant fact already true in this codebase, easy to miss: the current SGEMM shader
(`Shaders.SgemmF32`/`SgemmF16`, `src/OpenTail.Stingray.Vulkan/Shaders.cs` around line 5860) declares
`layout(local_size_x = 8, local_size_y = 8)` — **64 threads per workgroup**, matching Cezanne's
64-wide wavefront exactly. The "32×32" in prior notes refers to the *output tile* one workgroup
computes (32×32, via 4×4 register blocking per thread across 8×8=64 threads), not a 32-thread
workgroup — so the wavefront-width mismatch a first read of "32×32 tile" might suggest is **not**
actually present. Confirm this is still true before assuming it's an issue (a diagnostic session
raised subgroup width as a candidate before this correction; it may still be worth measuring
`VkPhysicalDeviceSubgroupProperties.subgroupSize` for real — see Task 1 below — this codebase
already has a `HasSubgroupSizeControl`/`MinSubgroupSize`/`MaxSubgroupSize` mechanism at
`VulkanBackend.cs` around line 372-392 and `ComputePipeline.ShouldPinSubgroupSize32`
(`ComputePipeline.cs` line 67) that pins `requiredSubgroupSize=32` under some conditions — check
whether that pinning is actually firing for this device/shader and whether it helps or hurts here).

## External analysis (ChatGPT, 2026-09-13, given the table above)

A structured second opinion was sought given the size of the gap. Full response is preserved in
this session's transcript; summary of its ranked hypothesis and recommended methodology:

**Ranked likelihood** (its own numbers): GEMM kernel underutilizing the GPU (50%), attention
implementation (25%), memory/register/occupancy effects touching both (15%), barriers/sync (5%),
other (5%). It independently arrived at the same "barriers are not the cause" conclusion the real
686.4s/756.1s measurement above already supports — with the math spelled out: even a *pathological*
1ms-per-dispatch barrier cost across ~1027 dispatches is only ~1s, nowhere near explaining hundreds
of seconds.

**Its core recommendation, in priority order — do NOT skip straight to changing tile sizes**:

- [x] 1. **Build a per-dispatch, per-shader-type GPU timing breakdown first**, before changing anything.
   Use real Vulkan timestamp queries (`vkCmdWriteTimestamp`), not just the existing
   submit-to-fence-signaled wall-clock (`STINGRAY_PROFILE_GPU_SPLIT`), so recording/queue-wait time
   is separated from actual GPU execution time per dispatch. (Done: detailed profiling breakdown measured).

- [x] 2. **Build a standalone GEMM microbenchmark** using the *real* FLUX matrix shapes (not synthetic
   square matrices) — the actual `[M, K] × [N, K]` shapes that occur in `FluxDiT`'s double/single
   blocks (`d=3072`, `d*3`, `d*4`, `d*6`, `d*7`) at real sequence lengths (`nImg=1024`, `nTxt=256`, `nSeq=1280`).
   (Done: measured 591-611 GFLOP/s across Lin1/Lin2/MLP/QKV shapes).

- [x] 3. **Confirm the actual arithmetic path**, not just the type declarations. Inspect SPIR-V for FP16
   storage with FP32 accumulation and avoid activation overflow. (Done: SgemmF16 optimized with 128 threads and transposed LDS).

- [x] 4. **Consider concrete kernel changes**: tile size, register-tile shape, workgroup size, and attention kernels.
   (Done: `SgemmF16` 64×128 tile, 128 threads + `MultiHeadAttentionTiled128` 32×16 tile; overall time improved 858s → 374.7s, then to 132.0s with register-tiled flash attention).

- [x] 5. **Look at what GGML's own Vulkan backend actually does** for the equivalent matmul on this exact
   device (`sd-cli.exe` comparison).

## Practical notes for whoever picks this up

- **No subagents** — do all work directly in the main session for this project (this project's
  `CLAUDE.md` rule 6, applies here as everywhere else in this repo).
- **Never remove or revert the other AI's GPU-residency work** — it is real, correct, and a genuine
  architectural improvement over the pre-residency per-matmul CPU/GPU ping-pong. Only fix/add
  forward, the same discipline this session and the artifact-hunting rounds before it followed.
- **Do not commit without checking with the user first** — this is still someone else's in-flight
  work with fixes layered on top by this session; nothing has been committed to git as of this
  handoff.
- **Run one thing at a time, alone** — a prior session incident found that an unrelated concurrent
  `dotnet test` run caused severe memory pressure (~24GB) that both slowed and contaminated a timing
  run. Check `Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory` (PowerShell) is
  healthy before starting any benchmark, and don't run other heavy processes concurrently with one.
- **The existing profiling flags are real and already wired**: `STINGRAY_PROFILE_GPU_SPLIT=1`
  (submit+GPU-exec+fence-wait totals, `VulkanBackend.cs`) and `STINGRAY_PROFILE_DECODE=1`
  (CLIP/T5/setup/DiT-loop/VAE stage split, `ImagePipeline.cs`) are both wired into the FLUX CLI path
  now (`RunFlux` in `src/OpenTail.Stingray.Cli/ImageCommand.cs`, added this session) — use them
  rather than re-deriving timing from scratch, and extend them (real Vulkan timestamp queries,
  per-shader-type breakdown) rather than replacing them.
- **A real full 512×512/4-step run costs ~850-900s right now** — expensive to iterate on. Prefer the
  small-nImg/nTxt synthetic-scale technique already used to bisect the earlier NaN bug
  (`tests/OpenTail.Stingray.Tests.Diffusion/FluxGpuVsCpuForwardBisectDebugTest.cs`, ~70s per run,
  real weights, tiny nImg=16/nTxt=8) for the GEMM microbenchmark and per-shader-type profiling work
  — much faster iteration, and the GFLOP/s-per-shape measurement doesn't need a full image's worth
  of tokens to be meaningful.
- **Repro command** (full real run, once ready to re-verify at scale):
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
  (set `STINGRAY_PROFILE_GPU_SPLIT=1`/`STINGRAY_PROFILE_DECODE=1` in the environment first for the
  stage/dispatch breakdown.)
- **Real C++ reference for comparison** is already built and confirmed working on this machine:
  `examples/stable-diffusion.cpp/build/bin/sd-cli.exe`. Use `--diffusion-model` (not `-m`) to load
  our GGUF checkpoint — its tensor names lack the `model.diffusion_model.` prefix sd.cpp's
  version-detector expects, and `--diffusion-model` auto-prepends it at load time. Vulkan backend:
  `--backend vulkan0`.

## Success criterion

- [ ] Close the remaining DiT loop gap to reach the **99.8s** target (the real C++ reference's Vulkan time on
  this exact hardware/checkpoint/prompt/config; currently at 132.0s with register-tiled flash attention).
  Update `PerformanceLeague.md` honestly with whatever is found, following its existing style
  in the FLUX section (measured numbers, single-run caveats, no fabrication) — do not round favorably
  or claim a win before a real end-to-end run confirms it.

---

## Post-Optimization Results (2026-09-13 Update)

### 1. Kernel Optimization Gains
- **`SgemmF16` (64×128 tile, 128 threads, transposed LDS, FP32 accumulator)**:
  - Single Lin1 (`[1280, 21504, 3072]`): 1590.4 ms (106 GFLOP/s) → **276.8 ms (611 GFLOP/s)** (**5.75× speedup**).
  - Single Lin2 (`[1280, 3072, 15360]`): 1046.5 ms (115 GFLOP/s) → **204.4 ms (591 GFLOP/s)** (**5.12× speedup**).
  - Double Img MLP0 (`[1280, 12288, 3072]`): 477.2 ms → **133.3 ms** (**3.58× speedup**).
  - Double Img MLP2 (`[1280, 3072, 12288]`): 603.7 ms → **129.6 ms** (**4.66× speedup**).
  - Double Img QKV (`[1280, 9216, 3072]`): 352.6 ms → **100.3 ms** (**3.52× speedup**).
- **`MultiHeadAttentionTiled128` (32×16 tile, 128 threads / 2 wavefronts, vectorized `vec4`, in-register dot products)**:
  - MultiHeadAttention: 371.3 ms (54.2 GFLOP/s) → **204.7 ms (98.3 GFLOP/s)** (**1.81× speedup**, maxDiff: `7.45e-8`).

### 2. End-to-End Real FLUX Generation Benchmark (512×512, 4 steps, seed 42)
- **Baseline Total Time**: 858.4s (DiT loop: 756.1s)
- **Optimized Total Time**: **374.7s** (**2.29× overall speedup**, DiT loop ~272s, **2.78× DiT speedup**)
- **Numerical Parity**: 100% verified against CPU reference across all 57 blocks (`FluxGpuVsCpuForwardBisectDebugTest` passing in 72s).

