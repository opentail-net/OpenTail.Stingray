# Vulkan large-tensor sharding — Qwen3.8-27B (qwen35 hybrid-GDN) GPU-only support

**Status:** root cause found to be narrower than assumed; the actual fix is implemented and
**awaiting a real-weight test run** (blocked — see "What changed since the plan was written").
General multi-VkBuffer sharding (the bulk of this document) turned out to be unnecessary for
this specific checkpoint and remains undesigned/unimplemented; kept below for reference should a
future checkpoint genuinely exceed the device's `maxStorageBufferRange`.

## What changed since the plan was written (read this first)

Phase 0 measurement (§1 below) found the actual failure was **not** a fundamentally-oversized
tensor needing multi-buffer sharding — it was a missing raw-quant Vulkan embedding-lookup shader
for one specific dtype, causing an unnecessary ~9x F32 blow-up:

- `token_embd.weight` in `Qwen3.8-27B-UD-Q3_K_XL.gguf` is **Q3_K, [5120, 248320], 521.0 MiB raw**
  (measured via `list-tensors`). Vulkan had `EmbedLookupQ4K`/`EmbedLookupQ6K` raw-read shaders but
  none for Q3_K, so `UploadEmbeddingWeight` fell through to its generic F32-expand path:
  248320 × 5120 × 4 bytes ≈ **4.66 GiB** — comfortably over the (previously hard-coded) 2 GiB
  ceiling.
- `output.weight` is **Q5_K, [5120, 248320], 833.6 MiB raw** — already in `UploadWeight`'s
  raw-kept dtype set (`Q4_K/Q5_K/Q6_K/Q8_0/Q4_0`), so it was never the problem.
- The 2 GiB ceiling in `ShouldKeepFixedWeightsOnGpu` was also a **hard-coded constant**, never
  actually queried from `VkPhysicalDeviceLimits.maxStorageBufferRange` — a second, independent
  bug (would have misfired on any device whose real limit differs from 2 GiB in either direction).

**Fix implemented** (no sharding infrastructure needed):
1. Added `EmbedLookupQ3K` — a new Vulkan compute shader that dequantizes Q3_K rows directly
   (mirrors `Dequantize.DequantQ3K`'s bit-for-bit logic; ggml `dequantize_row_q3_K` layout).
   `token_embd.weight` now stays raw at 521 MiB instead of expanding to 4.66 GiB.
2. `VulkanBackend` now queries and logs the real `maxStorageBufferRange`/
   `maxMemoryAllocationCount` at device init (`VulkanBackend.MaxStorageBufferRange` property) and
   `ShouldKeepFixedWeightsOnGpu` uses it (× 0.90 safety margin) instead of the old hard-coded
   2 GiB constant.
3. The `NotSupportedException` message (for the case that's now believed unreachable for this
   checkpoint) reports the actual queried limit and estimated tensor sizes instead of a fixed
   "2 GB limit" string, so any future trip is diagnosable without re-deriving these numbers.
4. Boundary-correctness tests added: `EmbedLookupQ3KMatchesCpu` (small synthetic table) and
   `EmbedLookupQ3KBoundaryRowsMatchCpu` (first/middle/last rows of a 37-row table) in
   `tests/OpenTail.Stingray.Tests.Vulkan/VulkanShaderTests.cs`, mirroring the existing
   `EmbedLookupQ6KMatchesCpu` pattern — compares GPU shader output against
   `Dequantize.ToFloat32(..., DType.Q3_K, ...)` byte-for-byte.

**Verification status — 2026-09-15, confirmed working:**
- `dotnet build` succeeds clean for `OpenTail.Stingray.Vulkan`, `.Engine`, `.Cli`, and
  `tests/OpenTail.Stingray.Tests.Vulkan` (0 warnings, 0 errors).
- `scripts/gen-spirv.ps1` ran successfully — `glslc` accepted `EmbedLookupQ3K`'s GLSL and it's in
  the committed precompiled SPIR-V table (`Shaders.Precompiled.g.cs`).
- **Boundary-correctness tests ran and passed** (`STINGRAY_RUN_HEAVY_TESTS=1`, invoked the built
  `.exe` directly with `-method "*EmbedLookupQ3K*"` per CLAUDE.md's filter guidance):
  ```
  EmbedLookupQ3K: 0 mismatches over 2560 values
  Total: 2, Errors: 0, Failed: 0, Skipped: 0, Not Run: 0
  ```
  This also confirmed, from a real run on this machine's driver, that
  `maxStorageBufferRange = 4,294,967,295` (4 GiB − 1) — **double** the old hard-coded 2 GiB
  assumption, exactly the kind of driver variance the device-limit-query fix was meant to catch.
- **Full-model smoke test ran and passed**: `dotnet run --project src/OpenTail.Stingray.Cli -c
  Release -- -m models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf -p "The capital of France is" --temp 0
  -g 16 --backend vulkan` loads cleanly (no `NotSupportedException`), logs
  `[VulkanBackend] Device limits: maxStorageBufferRange=4,294,967,295 bytes (4.00 GiB)`, uploads
  all 64 layers, and begins generating real tokens (`Prefill: 57 tokens ... | Decode: ... t/s`).
  Confirmed exit code 0 on a second, shorter run.
- Decode throughput is very slow (~0.2 t/s) because the dense FFN doesn't fit this iGPU's ~16 GB
  placement budget alongside a 27B model's other weights and stays CPU-resident (logged:
  `Dense FFN-on-GPU: budget -11842 MiB < per-layer 84 MiB ... All FFN stays on CPU`). This is an
  expected, separate hardware-budget characteristic of this integrated-GPU box (see CLAUDE.md
  rule 13 — do not read this as a defect in the Q3_K fix; a real discrete-GPU run with more VRAM
  would place more/all FFN layers on GPU and decode far faster). **No performance pass has been
  done** on this path yet (CLAUDE.md rule 7) — that remains open, as does deciding whether the
  FFN GPU/CPU placement heuristic is worth revisiting for this checkpoint specifically.

**Conclusion: the Qwen3.8-27B (qwen35 hybrid-GDN) Vulkan GPU path now works end-to-end** — model
loads, all weights upload, forward pass runs, tokens generate — without any CPU embedding/LM-head
fallback and without the general sharding mechanism ChatGPT's original writeup called for. The
one remaining gap is throughput on this specific low-VRAM iGPU, which is a placement/performance
question, not a correctness one.

## Follow-up: memory-overshoot root cause (2026-09-15, later same day)

After the above was confirmed working, the user pointed out this exact checkpoint (13.15 GB on
disk — confirmed via `ls`) is normally a model that "fits onto graphics cards," and asked why the
Vulkan path was using dramatically more memory than that — enough that testing an aggressive
placement-budget override (`STINGRAY_VULKAN_UMA_FRACTION=1.0`) visibly made the machine start
swapping to disk. Investigated and fixed the actual cause (separate from the Q3_K embedding bug
above):

**Root cause**: `UploadWeight`'s raw-kept dtype set (`Q4_K/Q5_K/Q6_K/Q8_0/Q4_0`) covers only 5 of
the 13 GGUF dtypes this "UD" (Unsloth Dynamic) quant actually mixes across tensors. A `list-tensors`
dtype tally on this checkpoint found 339 of 866 tensors use a dtype with **no raw Vulkan matvec
kernel at all** (`IQ4_XS`×156, `IQ3_S`×111, `IQ3_XXS`×34, `IQ2_S`×15, `Q3_K`×12, `IQ2_XS`×4,
`Q2_K`×3, `IQ4_NL`×2, `IQ2_XXS`×2) — these silently fell through to a full **F32** dequant-and-
upload, a 4-16x blow-up over their raw quantized size depending on the original bit-width. Summing
just the 142 of these that are "core" mandatory GPU-resident tensors (attention/GDN per layer —
excludes dense FFN, which stays CPU-resident by default, and the already-fixed embedding/output)
found **+19.41 GiB of pure waste** versus their raw size, with `IQ4_XS` (+9.55 GiB) and `IQ3_S`
(+6.45 GiB) the two dominant contributors. This, not any tensor being fundamentally too large for a
single VkBuffer, is why the "core" GPU-resident footprint measured **26.9 GiB** for a checkpoint
whose whole file is 13.15 GB — nearly all of the excess was avoidable F32 dequant waste on a
handful of dtypes lacking raw kernels, not a real memory requirement.

**Why this matters for "both CPU and GPU paths"** (the user's framing): the *mandatory* GPU-resident
upload (embedding/output/all per-layer attention+GDN weights) has **zero memory-availability check**
in `VulkanHybridGdnForwardPass`'s constructor — unlike the CPU-resident dense-FFN prefault path
(`MmapPrefault.ShouldRun`, gated at 80% of currently-available RAM for `RamGate.FitsInRam` callers)
or the optional Dense-FFN-on-GPU path (gated by `STINGRAY_VULKAN_UMA_FRACTION` × heap size). If the
mandatory core doesn't comfortably fit in available physical RAM, Vulkan's host-visible allocation
on this iGPU (shared system RAM, not dedicated VRAM) just keeps succeeding — Windows starts paging
to disk with no early warning and no way to abort cleanly. The two purely CPU-only forward-pass
paths (`ForwardPass.cs`, `HybridGdnForwardPass.cs`) have the same shape of gap in the other
direction: they use `MmapPrefault.RamGate.Always`, deliberately bypassing the 80%-available-RAM
gate entirely ("you chose CPU-only, prefaulting is the point") — reasonable when the model
comfortably fits, but it means a model that doesn't fit gets force-faulted into RAM with no
warning there either. **Not yet fixed**: neither of these gaps has a pre-flight check added in this
session — the F16 mitigation below addressed the actual numbers enough that the GPU-path gap didn't
need to be exercised further, but the *absence of a check* is still real and worth closing later
(see "Still open" below).

**Fix implemented**: rather than hand-writing native Vulkan kernels for the IQ-family codebook
quantizations (`IQ4_XS`/`IQ3_S`/etc. use a lookup-table/grid-based dequant scheme, materially more
complex and higher-risk to get right blind than the linear K-quant math already ported for Q3_K),
the "no raw kernel" fallback in `UploadWeight` now dequantizes to **F16** instead of F32:
- Added `MatVecF16` (`src/OpenTail.Stingray.Vulkan/Shaders.cs`) — a straightforward weight-stationary
  GEMV reading packed FP16 (2 halves/uint32 via `unpackHalf2x16`, same convention as
  `VulkanBackend.UploadHalf`), mirroring the existing `MatVecF32` shader's reduction pattern.
  Wired into `VulkanBackend.MatMul`'s dtype switch as `case DType.Float16`.
- `UploadWeight`'s fallback branch now dequantizes to `float[]` (unchanged, still needed as an
  intermediate since `Dequantize.ToFloat32` is the only CPU reference implementation), casts
  element-wise to `Half[]`, and calls `UploadHalf` instead of `Upload` — halving the footprint of
  every affected tensor versus the old F32 path.
- `EstimateWeightGpuBytes` updated to match (informational — for the `ShouldKeepFixedWeightsOnGpu`
  budget check and the diagnostic error message; doesn't change this checkpoint's outcome since
  `output.weight` is Q5_K, already raw-kept).
- New test `MatVecF16MatchesCpu` (`tests/OpenTail.Stingray.Tests.Vulkan/VulkanShaderTests.cs`) —
  synthetic (no model fixture needed), deliberately non-power-of-2/non-multiple-of-8 dimensions to
  exercise the tail workgroup/lane paths, compares against a CPU reference computed on the
  **F16-rounded** weights (not the original F32) so the assertion isn't just measuring expected
  rounding noise. **Ran and passed**: `MatVecF16: 0/37 mismatches (>1% rel error)`.
  `EmbedLookupQ3K`/`EmbedLookupQ3KBoundaryRowsMatchCpu` re-ran clean alongside it (no regression).

**Measured result** (`STINGRAY_VULKAN_UMA_FRACTION` left at its default, unset/0.5 — no risky
overrides): core GPU-resident footprint dropped from **26.9 GiB → ~14.9 GiB** (inferred from the
`Dense FFN-on-GPU` budget line flipping from deeply negative, `-11842 MiB`, to positive enough for
1 layer at the same 50%-heap/16076 MiB budget — bounds it to 14,884–14,968 MiB) for a checkpoint
whose file is 13.15 GB. That's now within a plausible margin of the actual on-disk size (the
remaining gap is norms kept F32, the still-larger-than-raw F16 fallback for IQ/Q3_K/Q2_K tensors,
and scratch/KV-cache buffers) instead of more than double it. Verified via `Get-Counter
'\Memory\Available MBytes'`/`'\Memory\Pages/sec'` before and after: available RAM stayed ~48-51 GB
throughout, 0 pages/sec (no swapping) at the default fraction, both before and after this fix.

**Still open / not attempted further this session:**
- No pre-flight memory-availability check exists before the mandatory GPU-resident upload begins
  (see "Why this matters" above) — a future checkpoint whose *actual* core footprint (even after
  this F16 fix) doesn't fit available RAM would still silently start swapping rather than failing
  fast with a clear message. Same gap exists on the `RamGate.Always` CPU-only paths.
- Tested bumping `STINGRAY_VULKAN_UMA_FRACTION=0.7` (safe now that core is ~15 GiB, not 27 GiB —
  22,507 MiB budget, comfortably above core+margin) to let more/all dense FFN layers onto GPU: **no
  swapping observed** (confirmed via the same Available-MBytes/Pages-sec check, both before and
  after), but the run didn't finish within a 150s timeout — it was still inside the
  `Dense FFN-on-GPU` upload step when killed. Likely cause: the new element-wise `f32[i] → (Half)`
  scalar conversion loop in `UploadWeight`, now also applied to the (much larger) dense FFN tensors
  when more of them get GPU-budgeted, isn't vectorized — plausible but **not measured/confirmed**,
  don't treat as fact. This is a genuine, separate performance question (slow, not unsafe) —
  left at the safe default (`STINGRAY_VULKAN_UMA_FRACTION` unset) rather than chased further
  live/unsupervised. A profiled follow-up (confirm the bottleneck, vectorize the F32→F16 cast e.g.
  via `System.Numerics.Tensors`/SIMD, re-measure) is the natural next step.
- A local Vulkan-enabled llama.cpp build was started (`examples/llama.cpp/llama.cpp`, `cmake -B
  build -DGGML_VULKAN=ON` configured successfully) as an independent ground-truth reference for
  this checkpoint's real GPU memory/token-rate profile, but the build failed on the bundled web UI
  target (unrelated to `ggml`/core `llama-cli` compilation — the failure output was truncated in
  this session before the real error line). Not pursued further given the direct measurements
  above already answered the question; revisit with `-DLLAMA_BUILD_SERVER=OFF`-style flags to skip
  the UI target if a ground-truth comparison is wanted later.
- CLAUDE.md rule 7's mandatory performance pass (multiple samples, real weights, written-down
  numbers) has still not been done for the Vulkan qwen35 path as a whole — everything measured in
  this session was memory footprint and pass/fail correctness, not tokens/sec.

## Follow-up 2: fixed the slow F32→F16 conversion (2026-09-15, same day, after external review)

Got a second opinion (external LLM, given the codebase context above) on the "still open" slow
`STINGRAY_VULKAN_UMA_FRACTION=0.7` run. Its diagnosis, confirmed correct by implementing and
measuring: two separate scalar bottlenecks, not one — (1) the naive `for (int i...) f16[i] =
(Half)f32[i]` cast loop, and (2) the underlying `Dequantize.DequantIq4Xs`/`DequantIq3S` scalar
per-block decoders themselves (IQ4_XS/IQ3_S are the two highest-volume dtypes in this checkpoint's
FFN tensors, 89M elements each, up to 64 layers × 3 tensors), with the dequant step expected to
dominate since it does bit-unpacking + codebook lookups per element versus the cast's single
narrow-and-store. Both fixed, cheapest/lowest-risk first:

1. **F32→Half cast**: replaced the scalar loop with `System.Numerics.Tensors.TensorPrimitives
   .ConvertToHalf(f32, f16)` (`VulkanHybridGdnForwardPass.cs`'s `UploadWeight`) — the BCL's own
   vectorized (Vector128/256/512) float→half narrowing, already correct by construction (it's the
   framework's own `(Half)x` semantics, just vectorized) — no new test needed beyond the existing
   `MatVecF16MatchesCpu`, which exercises real Half values end-to-end.
2. **IQ4_XS/IQ3_S dequant**: refactored `Dequantize.DequantIq4Xs`/`DequantIq3S`
   (`src/OpenTail.Stingray.Cpu/Dequantize.cs`) from "loop over blocks on one thread" to "extract
   each block's already-self-contained body into `DecodeIq4XsBlock`/`DecodeIq3SBlock`, dispatch
   over blocks with `Parallel.For` once a tensor has ≥64 blocks" (`MinBlocksForParallelDequant`),
   using `SimdKernels.CpuThreads` for the degree of parallelism to stay consistent with the
   project's existing CPU-thread-count knob rather than introducing a second one. The arithmetic
   is byte-for-byte unchanged — each block already only read its own input bytes and wrote its own
   output slice (no cross-block state, no reduction), so this was a pure dispatch change, not a
   numerical one. Followed the reviewer's explicit recommendation to do this rather than
   hand-writing AVX2/gather-based vectorized IQ dequant kernels, given the codebook/bit-packed
   nature of these formats makes that meaningfully higher-risk for a correctness-critical path.
3. **New tests** (all ran and passed):
   - `Dequantize_IQFormats_ParallelBlocksMatchSequentialBlock` (`tests/OpenTail.Stingray.Tests.Core
     /IqQuantTests.cs`, `IQ3_S`/`IQ4_XS`) — replicates one cosine-seeded reference block many times
     (4× for the sequential path, 400× for the parallel path) and asserts every replica's 256-float
     output is bit-identical to the reference — directly catches a wrong-block-index or data-race
     bug, since each block's output depends only on its own bytes. **11/11 tests passed** in
     `OpenTail.Stingray.Tests.Core.IqQuantTests` (9 pre-existing + 2 new).
   - Re-ran `MatVecF16MatchesCpu` and `EmbedLookupQ3K*` — still clean (no regression from the
     `TensorPrimitives.ConvertToHalf` swap).

**Measured result**: re-ran the same `STINGRAY_VULKAN_UMA_FRACTION=0.7` scenario that previously
hung past a 150-second timeout while stuck inside `Dense FFN-on-GPU` upload. This time it
**finished quickly** (well under the 150s budget) and got through 59/64 FFN layers before hitting
a real, clean, **catchable** exception:
```
[VulkanHybridGdnForwardPass] FFN-on-GPU upload aborted at layer 59: [-1] ErrorOutOfHostMemory
[VulkanHybridGdnForwardPass] Dense FFN-on-GPU: uploaded 59/64 layers (4975 MiB); 5 stay on CPU.
Unhandled exception. Vortice.Vulkan.VkException: [-1] ErrorOutOfHostMemory
  ... at VulkanHybridGdnForwardPass.LoadMtpHead(VulkanBackend gpu) ...
```
Confirmed via `Get-Counter '\Memory\Available MBytes'`/`'\Memory\Pages/sec'` that this was a real,
clean allocation failure, not another swap episode: pages/sec spiked briefly (~9,340) right at the
crash (OS reclaiming the failed process) then settled to double digits within ~6 seconds; available
RAM recovered to 54 GB. **This is a much better failure mode than before** — a catchable exception
with a clear message beats an unresponsive, swapping machine — but it does confirm a **second real
bug**, separate from everything above: `TryUploadDenseFfnLayers`'s budget check
(`VulkanHybridGdnForwardPass.cs`) only accounts for `_uploadedVramBytes` at the time it runs, but
`LoadMtpHead` uploads more GPU-resident weights (the MTP/NEXTN speculative-decode head) **after**
the dense-FFN budget loop finishes, with no headroom reserved for it. At `0.7`, the FFN loop's
budget check let it fill nearly the entire remaining space (59 of 64 layers), leaving nothing for
the MTP head that was always going to load next.

**Left the system at**: default settings (`STINGRAY_VULKAN_UMA_FRACTION` unset/0.5) after this —
that combination is fully verified safe and working end-to-end from the first follow-up. `0.7` is
close to correct but needs the MTP-head budgeting bug fixed first, or it can OOM (cleanly, not
silently) depending on how many FFN layers happen to fit before the head's turn.

**Now still open (updated list):**
- Fix `TryUploadDenseFfnLayers`'s budget to reserve headroom for whatever `LoadMtpHead` (and any
  other post-loop mandatory GPU upload) needs, computed or estimated *before* deciding how many
  FFN layers to admit — not just implicitly hoping there's slack left. Alternative: compute the
  dense-FFN budget as `available - core - mtpHeadEstimate - margin` instead of `available - core -
  margin`.
- The pre-flight memory-availability check for the *mandatory* core GPU upload (embedding/output/
  per-layer attention+GDN) still doesn't exist (see Follow-up 1) — unrelated to the bug just found,
  still real, still open.
- Once the MTP-head budgeting bug is fixed, re-attempt `STINGRAY_VULKAN_UMA_FRACTION=0.7` (or
  similar) end-to-end and confirm it now completes cleanly with most/all FFN layers on GPU, then do
  the still-outstanding tokens/sec performance pass (CLAUDE.md rule 7) comparing default (1 FFN
  layer on GPU) vs a fixed higher fraction (many/all FFN layers on GPU).
- The IQ dequant parallelization only covers `IQ4_XS`/`IQ3_S` (the two highest-volume dtypes in
  this checkpoint). `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `Q3_K`, `Q2_K`, `IQ4_NL` still dequant
  sequentially — lower volume here, but worth the same treatment if a future checkpoint leans on
  them more heavily.

## Follow-up 3: the MTP-head OOM's real cause was a second, bigger budgeting bug (2026-09-15, later)

Got a second external review of Follow-up 2's remaining open item (the `0.7`-fraction OOM at the
MTP head). Its top-priority recommendation — reserve the MTP head's footprint *before* deciding
the FFN budget, instead of discovering the shortfall only when `LoadMtpHead` runs — was implemented
first, exactly as it size-estimates every tensor `LoadMtpHead` will actually upload. It did **not**
fix the OOM on its own, which led to finding the real, larger bug underneath it.

**The real bug**: `TryUploadDenseFfnLayers`'s `perLayerBytes` was computed from
`tensor.ByteSize` — the **raw on-disk GGUF byte size** — not the actual post-upload GPU footprint.
For this checkpoint's dense FFN tensors (`IQ2_XS`/`IQ2_S`/`IQ3_XXS` — none in the raw-kept Vulkan
matvec set), the real GPU footprint after the F16 dequant fallback is roughly **6x** the raw size
(`~84 MiB/layer` raw vs. the true cost). The budget arithmetic was therefore admitting ~59 of 64
layers when only ~12 actually fit at `STINGRAY_VULKAN_UMA_FRACTION=0.7` — each admitted layer's
*actual* Vulkan allocation succeeded individually (F16 tensors aren't huge on their own), so
nothing failed until the cumulative real usage finally exhausted the host-visible heap, which
happened to land exactly at the MTP head's turn. The reviewer's MTP-reservation fix was real and
correct, but couldn't matter much against a per-layer budget that was already wrong by ~6x.

**Fix**: `perLayerBytes` now sums `EstimateWeightGpuBytes(gateInfo)/(upInfo)/(downInfo)` (the same
raw-vs-F16 estimator already used for the MTP-head reservation and `ShouldKeepFixedWeightsOnGpu`)
instead of `ByteSize`. Still an approximation — layer 0's dtypes stand in for every layer's, and a
"UD"/dynamic-quant checkpoint can vary dtype per layer — but far closer than the raw byte size, and
consistent with every other budget estimate in this file now.

**Verification — measured across the full fraction range, all loads clean**:

| `STINGRAY_VULKAN_UMA_FRACTION` | Placement budget | FFN layers admitted | Uploaded MiB | Model load time |
|---|---|---|---|---|
| 0.5 (default) | 16,076 MiB | 0/64 (budget −279 MiB) | — | 20.1s |
| 0.6 | 19,291 MiB | 5/64 | 2,550 MiB | 22.8s |
| 0.7 | 22,507 MiB | 12/64 | 6,120 MiB | 43.8s |
| 0.8 | 25,722 MiB | 18/64 | 9,180 MiB | 42.5s |

All four loaded and reached "Model loaded" successfully — **zero crashes, zero OOMs, zero swap
episodes** across the whole range (confirmed via `Get-Counter '\Memory\Available MBytes'`/
`'\Memory\Pages/sec'` before and after: available RAM stayed 48-53 GB throughout, pages/sec at
idle levels both before and after each run). The scaling is monotonic and sane — more fraction →
more FFN layers admitted → more MiB uploaded — which it was NOT before this fix (0.7 previously
*looked* like it should admit the most, 59 layers, and was in fact the one that crashed).

Each of the four runs above ran into the 150-second shell timeout **after** printing "Model
loaded" — i.e. loading itself completed in 20-44s in every case, and the timeout killed a slow
*decode* step afterward (a known, separate, expected characteristic: this reasoning-model
checkpoint at `--temp 0` can loop on "thinking" tokens well past the 150s window even at a
best-case handful of tokens/sec — see the `Warning: Greedy decoding` message this CLI already
prints for exactly this scenario). Not a memory bug, not a regression — do not conflate the two
when reading `EXIT=124` in a quick test script; check for "Model loaded" in the log before
attributing a timeout to memory issues.

**Now still open (updated list):**
- The tokens/sec performance pass (CLAUDE.md rule 7) comparing default (0 FFN layers on GPU) vs a
  higher fraction (5-18 layers on GPU, per the table above) is now unblocked (loading is safe at
  every fraction tested) but still not done — needs a non-reasoning-inducing prompt/temperature (or
  patience) to get a clean decode-speed measurement instead of hitting a thinking-loop timeout.
- Per the reviewer's other lower-priority notes: transient CPU allocation pressure during
  dequant-and-upload (temporary `float[]`+`Half[]` per tensor, ~510 MiB peak for the largest FFN
  tensors) and LOH/GC pressure across 12-18 large tensor uploads were flagged as "observe first,
  don't optimize yet" — no evidence either is a problem was collected in this session (the runs
  above completed without issue), so this stays a watch item, not a task.
- General multi-VkBuffer tensor sharding (this document's original subject) remains correctly
  identified as unnecessary for any checkpoint tested so far, now with three independent rounds of
  evidence (Q3_K embedding, F16 dequant fallback, and the budget-accounting fixes above) that the
  real problems in this class of failure are memory-accounting bugs, not tensors that are
  fundamentally too large for one VkBuffer. Keep the general-sharding design below as reference
  only; do not revive it without a checkpoint that actually exceeds the queried
  `maxStorageBufferRange` after every raw/native-dtype and accounting fix above has been applied.

## Follow-up 4: pre-flight RAM check for the mandatory core upload (2026-09-15, later)

Implemented the one remaining item from Follow-up 3's list: `VulkanHybridGdnForwardPass` now has
`EnsureRamHeadroom(string context)`, called once right after the embedding/output upload and once
per layer inside the main per-layer upload loop. It queries **current** available system RAM
(`GC.GetGCMemoryInfo().TotalAvailableMemoryBytes`) rather than pre-computing a total requirement —
deliberately, since a live signal automatically accounts for everything already committed (by this
upload or anything else on the machine) without needing a second estimate kept in sync by hand with
every upload path (exactly the kind of estimate/reality drift that caused the `perLayerBytes` bug in
Follow-up 3). If available RAM drops below a floor (default 2 GiB, override
`STINGRAY_VULKAN_MIN_FREE_RAM_MB`), it throws `InvalidOperationException` immediately with the
current available/floor values and how far progress got — failing fast instead of continuing to
allocate until the OS starts paging. This mandatory upload (embedding/output/per-layer
attention+GDN) has no per-user budget knob the way the optional dense-FFN path
(`STINGRAY_VULKAN_UMA_FRACTION`) or the CPU prefault path (`MmapPrefault`'s 80%-of-available gate)
do, so failing fast with a clear message is the only reasonable behavior when it won't fit — there's
no partial-admission fallback to fall back to.

**Verified**: (1) a normal default-settings load still completes in ~18s, confirming the check
doesn't false-positive during a real load that fits comfortably (64.8 GiB available at check time
on this machine); (2) forcing `STINGRAY_VULKAN_MIN_FREE_RAM_MB=999999` triggers a clean, immediate
`InvalidOperationException` (fails within seconds, before any layer upload work is wasted) with
exactly the diagnostic described above — confirms the mechanism actually fires and reports
correctly, not just that it compiles.

This closes the last item from the P1 priority list in the external review that prompted Follow-ups
2-4.

## Follow-up 5: tokens/sec performance pass (CLAUDE.md rule 7) (2026-09-15, later)

With loading verified safe at every fraction tested (Follow-up 3), did the previously-blocked
decode-speed comparison. Used `--no-thinking --temp 0.6 --top-p 0.95 --top-k 20` (the CLI's own
suggested settings for this reasoning model) to avoid the greedy-decoding "thinking loop" that made
earlier `--temp 0` runs hit shell timeouts before producing a stats line. Same prompt ("Write a
short haiku about the ocean."), same context (ctx=4096), multiple samples per side per CLAUDE.md
rule 7:

| Config | Sample 1 | Sample 2 | Sample 3 |
|---|---|---|---|
| Default (`STINGRAY_VULKAN_UMA_FRACTION` unset, 0 FFN layers on GPU) | 0.2 t/s (14 tok) | 0.3 t/s (9 tok) | timed out (200s, incomplete — discarded, not counted either way) |
| `STINGRAY_VULKAN_UMA_FRACTION=0.8` (18/64 FFN layers on GPU) | 0.4 t/s (28 tok) | 0.4 t/s (6 tok) | — |

**Result: putting FFN layers on GPU measurably helps decode throughput on this iGPU** — a
reproducible ~0.4 t/s at 0.8 fraction versus ~0.2-0.3 t/s at default, roughly **1.6-2x faster**,
consistent across both 0.8 samples. This is the expected direction (per-layer FFN compute moves
from AVX2 CPU matvec to Vulkan GPU matvec) and is now backed by actual measurement rather than
assumption, per CLAUDE.md's performance-pass rule. Absolute throughput is still slow in either case
(sub-1 t/s) — this is a 27B-class model with the majority of its FFN still CPU-resident even at
0.8, on a Ryzen 5700G iGPU with a 32 GiB shared-memory ceiling; that is a hardware/model-size
reality, not a bug this session's fixes were trying to solve. Given the now-demonstrated benefit,
raising the *default* `STINGRAY_VULKAN_UMA_FRACTION` above 0.5 (or making it auto-scale from the
now-accurate core-footprint accounting) is a reasonable follow-up, but changing a shipped default
based on one checkpoint's numbers on one machine is deliberately left as a separate decision, not
bundled into this session's bug-fix work.

**All runs in this pass loaded and decoded cleanly** — no OOMs, no swap episodes, no crashes across
either configuration, reconfirming Follow-up 3/4's fixes hold under repeated real use, not just the
single verification run each got at the time.

Remaining open work: the two lower-priority "observe first" watch items from Follow-up 3 (transient
CPU allocation/LOH pressure during dequant-and-upload) — no evidence collected in this session
suggests either is currently a problem, so they remain a watch item, not a task.

## Follow-up 6: native IQ4_XS Vulkan matvec kernel (2026-09-16)

Follow-up 5's plateau finding (0.8 == 0.95 fraction, identical 0.4 t/s) and the user's own live
Task Manager observation ("GPU is nowhere near its capacity" while "CPU compute is very very
prominent") pointed at the same conclusion from two directions: FFN placement had stopped being
the lever, and memory pressure was the real remaining concern. Got a third external review
proposing the actual fix for both — a **native raw Vulkan matvec kernel for IQ4_XS** (the single
largest source of avoidable GPU-memory expansion per Follow-up 2's accounting, +9.55 GiB of pure
F32-then-F16 waste in this checkpoint) instead of the F16-dequant-and-upload fallback.

**Caught and corrected one part of the external suggestion before implementing it**: it proposed a
"4 rows/workgroup, 64 lanes/row, `subgroupAdd()`" topology. This device reports a **fixed 64-wide
hardware subgroup** (`minSubgroupSize == maxSubgroupSize == 64`) — a plain `subgroupAdd` over a
64-lane logical "row" would silently sum across two rows, which is *exactly* the bug
`MatVecQ4K`'s own code comments already document hitting and fixing (reverted to shared-memory
tree reduction after measuring that a `subgroupClusteredAdd` fix gave no speedup anyway). Used the
proven **8 rows/workgroup, 32 threads/row, shared-memory tree reduction** topology instead — the
same one every existing raw MatVec kernel in this codebase already uses — while keeping the
actual IQ4_XS block-decode math from the external review, verified line-by-line against this
session's own already-tested CPU decoder (`Dequantize`'s `DecodeIq4XsBlock`) before writing a
single line of GLSL.

**Implemented**:
1. `MatVecIQ4XS` (`src/OpenTail.Stingray.Vulkan/Shaders.cs`) — decodes each IQ4_XS block directly
   into the multiply-accumulate (no intermediate F32/F16 buffer at all, unlike the embedding-lookup
   kernels which only had to solve upload footprint, not compute-time cost). 8 rows/workgroup, 32
   threads/row, each thread owns 8 of a 256-element block's values, shared-memory tree reduction —
   matching `MatVecQ4K`'s proven shape exactly.
2. Wired into `VulkanBackend.MatMul`'s dtype switch as `case DType.IQ4_XS` (new
   `_matVecIQ4XSPipeline` field, disposed alongside the others).
3. `VulkanHybridGdnForwardPass.UploadWeight` now keeps `IQ4_XS` raw (added to the existing
   raw-kept dtype set alongside `Q4_K/Q5_K/Q6_K/Q8_0/Q4_0`) instead of falling through to the F16
   dequant-and-upload path. `EstimateWeightGpuBytes` updated to match, for consistency with every
   other budget calculation in this file.
4. New test `MatVecIQ4XsMatchesCpu` (`tests/OpenTail.Stingray.Tests.Vulkan/VulkanShaderTests.cs`)
   — synthetic (no model fixture needed), `rows=37` (not a multiple of the 8-row workgroup, to
   exercise the tail workgroup) × `cols=512` (2 blocks/row, to exercise the multi-block
   accumulation loop), full weight×vector dot product compared against a CPU reference built from
   the *same bytes* via the already-tested `Dequantize.ToFloat32` — not just a per-element decode
   check. **Ran and passed on the first attempt, 0/37 mismatches** — the line-by-line verification
   against the CPU decoder before writing the shader paid off. Re-ran `MatVecF16MatchesCpu`,
   `EmbedLookupQ3K*` alongside it — no regression.
5. `scripts/gen-spirv.ps1` ran successfully — `glslc` accepted the shader on the first pass.

**Measured result on the real checkpoint**: at the **default** `STINGRAY_VULKAN_UMA_FRACTION`
(unset, 0.5) — previously "budget −279 MiB, 0/64 FFN layers" per Follow-up 3's table — now
**8/64 FFN layers fit** (4,080 MiB), with no other setting changed. This is a direct, measured
consequence of the core footprint shrinking: IQ4_XS tensors that were F16-expanding to ~2x their
raw size now upload at their true raw size, freeing real budget for the (still-optional,
budget-gated) dense FFN placement. Model loads cleanly in 18.3-18.8s, no crashes, no OOMs.

Decode speed at this new default-with-8-layers configuration: **0.3 t/s** (`-n 12 --temp 0`,
"The capital of France is" prompt) — in the same 0.2-0.3 t/s band as the pre-fix default (0
layers). Consistent with Follow-up 5's plateau finding: FFN-layer-count changes in this range
don't move decode speed much on this iGPU. **The value of this fix is memory efficiency, not
speed** — the same throughput now comes from a meaningfully smaller core footprint, achieved
automatically at the *default* setting with no risky fraction tuning required. (Aside, unrelated
to this fix: discovered mid-session that `-g` is this CLI's GPU-layer-placement flag, mirroring
llama.cpp's `-ngl` — NOT a token-count limiter as several earlier ad-hoc measurements in this
session assumed; the actual generation-length flag is `-n`/`--n-predict`. Omitting `-g` entirely
falls back to a plain CPU-only backend, not Vulkan, which produced one throwaway CPU-backend
measurement in this session's raw logs — discard any number in this document's history that
doesn't show a `Backend: Vulkan hybrid GDN` line above it.)

**`IQ3_S` (the second-largest contributor, +6.45 GiB) was implemented next, same session** — see
Follow-up 7 below. The other IQ dtypes (`IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ4_NL`) remain
on the F16 fallback — lower volume in this checkpoint, same "don't extend speculatively" reasoning
as Follow-up 3's dequant parallelization scope decision.

## Follow-up 7: native IQ3_S Vulkan matvec kernel (2026-09-16, same day)

Implemented `IQ3_S` immediately after `IQ4_XS`, per the external review's own sequencing advice
("implement IQ4_XS first, prove parity, benchmark, then IQ3_S") — that sequencing was followed in
Follow-up 6, and this is the "then IQ3_S" step, done the same session since IQ4_XS's correctness
test passed cleanly on the first attempt.

**Design**: same 8-rows/32-threads-per-row/shared-memory-tree-reduction topology as
`MatVecIQ4XS`/`MatVecQ4K` (not `subgroupAdd` — see Follow-up 6 for why). The IQ3_S block-decode
math was derived by re-indexing `Dequantize`'s already-tested `DecodeIq3SBlock` from its native
nested-loop form (`ib32`/`half`/`l`/`j`) into a flat per-lane-group form: each of the 32 lanes in
a row owns one contiguous 8-element group, where `lane` **is** the CPU decoder's flat group-
visitation index (0-31) — derived algebraically (`combinedIdx = lane>>2`, `l = lane&3`,
`ib32Idx = combinedIdx>>1`, `half = combinedIdx&1`) and re-verified against the CPU source line by
line before writing any GLSL, the same discipline as Follow-up 6. One correction versus a first
instinct: the CPU decoder's grid-lookup bytes (`(byte)(grid1 >> (8*j))`) are **unsigned
magnitudes with sign applied separately** via the sign-mask byte, not sign-extended values — the
shader must match that exactly (`mag * sign`, not `int8(mag)`).

**The 512-entry `IqCodebooks.Iq3SGrid` table was generated mechanically from the C# source**
(`awk`-extracted the hex literals, reformatted into a GLSL `const uint[512]` array), not
hand-transcribed — eliminates an entire class of transcription-error risk for a table this size,
and keeps the C# table as the single source of truth (regenerate if it ever changes; the shader
comment says so explicitly).

**Implemented**: `MatVecIQ3S` shader; wired into `VulkanBackend.MatMul` (`case DType.IQ3_S`,
new `_matVecIQ3SPipeline`); `IQ3_S` added to `UploadWeight`'s raw-kept dtype set and
`EstimateWeightGpuBytes`, same pattern as `IQ4_XS`.

**Tests**: `MatVecIQ3SMatchesCpu` (same shape as the IQ4_XS test — rows=37 tail-workgroup,
cols=512 multi-block) and `MatVecIQ3SBoundaryRowsMatchCpu` (boundary rows, single-block shape).
**Both passed with 0 mismatches on the first run** — the line-by-line CPU-decoder re-derivation
and the mechanically-generated grid table both paid off; no debugging round-trip was needed.
Re-ran `MatVecIQ4XsMatchesCpu`/`MatVecF16MatchesCpu`/`EmbedLookupQ3K*` alongside — all 6 clean,
no regressions.

**Measured on the real checkpoint** — a third, larger jump in FFN-layers-admitted at every
fraction tested, each verified with a clean load, a clean memory-counter check, and a real
generation producing coherent output (not just a load):

| `STINGRAY_VULKAN_UMA_FRACTION` | FFN layers (Follow-up 3 baseline) | + IQ4_XS (Follow-up 6) | + IQ3_S (this follow-up) | Decode speed (this follow-up) |
|---|---|---|---|---|
| 0.5 (default) | 0/64 | 8/64 | **14/64** | 0.4 t/s |
| 0.8 | 18/64 | 27/64 | **32/64** (exactly half) | **0.5 t/s** — a new high, past the earlier 0.4 t/s plateau |

The 0.5 t/s result at 0.8 fraction is the first measurement in this whole document's history to
beat the plateau found in Follow-up 5 — consistent with the theory that the plateau was a function
of *how many* FFN layers were GPU-resident (18-28 wasn't enough to move the needle further; ~32,
half the model, is). All loads and generations in this table completed cleanly: no crashes, no
OOMs, no swap episodes (`Get-Counter '\Memory\Available MBytes'`/`'\Memory\Pages/sec'` checked
before and after every run, available RAM held in the 44-45 GB range throughout).

**Not done in this session** (at time of writing, superseded by Follow-up 8 below): the remaining
minor IQ dtypes and the full fraction sweep.

## Follow-up 8: remaining-waste tally + full fraction sweep with both native kernels (2026-09-16)

**Remaining F16-fallback waste tally**: re-ran Follow-up 2's per-dtype accounting excluding
`IQ4_XS`/`IQ3_S` (now raw) from the "still F16-expanding" set. Result: only **23 tensors, +1.52
GiB total** — down from the original 19.4 GiB — dominated by `IQ3_XXS` (14 tensors, +0.95 GiB),
with `Q2_K`/`Q3_K`/`IQ2_S`/`IQ4_NL` contributing the small remainder. **Confirms diminishing
returns**: building native kernels for the remaining five dtypes would target roughly 8% of the
original waste, each in a structurally different (and, for the 1-2 bit IQ formats, likely harder)
codebook format. Not pursued — matches this document's repeated "measure before extending"
pattern.

**Full fraction sweep, both native kernels in place** — every load and generation below checked
clean via `Get-Counter` before and after (available RAM held 44-48 GB throughout every run in this
table, 0 pages/sec at rest, one brief expected spike during the largest load):

| `STINGRAY_VULKAN_UMA_FRACTION` | FFN layers | Uploaded MiB | Decode speed |
|---|---|---|---|
| 0.5 (default) | 14/64 | 7,140 MiB | 0.4 t/s |
| 0.8 | 32/64 | 16,320 MiB | 0.5 t/s |
| 0.9 | 39/64 | 19,890 MiB | 0.7 t/s |
| 0.95 | 42/64 | 21,420 MiB | 0.7 t/s |
| 1.0 (`STINGRAY_DENSE_FFN_GPU_MARGIN_MB=256`) | 47/64 | 23,970 MiB | **0.9 t/s** |

**Speed kept climbing all the way to the top of the range — no plateau found this time**, unlike
Follow-up 5's result (which plateaued at 0.4 t/s between 0.8 and 0.95 *before* the IQ3_S kernel
existed). That plateau is now understood in hindsight: it wasn't a fundamental ceiling, it was an
artifact of not enough FFN layers being GPU-resident yet at the memory cost the F16 fallback
imposed — once the native kernels made more layers affordable per fraction step, speed kept
scaling with layer count as expected. **0.9 t/s at 1.0 fraction is 3-4.5x the very first
measurement in this document's history (0.2-0.3 t/s at 0/64 layers, before any of this session's
fixes).**

**This is the practical ceiling for the `STINGRAY_VULKAN_UMA_FRACTION` lever specifically**:
`UmaHeapFraction` is deliberately clamped to `[0.05, 1.0]` (`VulkanBackend.cs`), and 1.0 already
consumes the entire heap Vulkan reports (`Heap 1: 32153MB`) — confirmed by `Placement budget:
32153MB (100%)` in the 1.0 run's log. The remaining 17 CPU-resident layers would need the
driver/BIOS to expose more system RAM as GPU-mappable (there is real headroom in actual system RAM
— 47+ GB stayed free even at this setting — but not in what this driver reports as the UMA-visible
heap for this purpose). Removing or raising that clamp was deliberately not attempted: it exists
specifically to prevent the kind of overcommit this whole document is about fixing, and bypassing
a deliberate safety bound isn't something to do without an explicit ask.

**Recommendation given this data (superseded by Follow-up 9 below)**: `1.0` was the best-measured
setting at time of writing.

## Follow-up 9: testing past the `UmaHeapFraction` clamp (2026-09-16, later)

The user asked, explicitly, to understand the `≤1.0` clamp's theory by testing past it rather than
reasoning about it in the abstract — a reasonable ask given this document's whole track record of
"an advertised/assumed number turning out to be conservative" (the original hard-coded 2 GiB
`maxStorageBufferRange` in Follow-up 0, now known to actually be 4 GiB on this device).

**The theory**: `VramBytes = fraction × Heap.size`, where `Heap.size` comes from
`vkGetPhysicalDeviceMemoryProperties` — a number the AMD driver chose to *advertise* for this UMA
heap (`32,153 MiB` here), not necessarily "all RAM actually available for GPU-mappable
allocations." The `≤1.0` clamp assumes that advertised number is a hard ceiling worth respecting.
Whether it actually is one — enforced by the driver/ICD at `vkAllocateMemory` time — or just
advisory metadata the OS will let allocations exceed (using real system RAM the advertised number
doesn't account for) can only be answered by trying it, not by reading the spec, since Vulkan
doesn't mandate either behavior for a given ICD.

**Change**: widened the clamp from `[0.05, 1.0]` to `[0.05, 2.0]` in `VulkanBackend.cs`'s
`UmaHeapFraction` — the new upper bound is a typo-guard (protects against an accidental extra
digit turning into an absurd request), not a claim that `2.0` is safe on any given machine. Added
an explicit code comment documenting the caveat found during this test: the dense-FFN layer loop
catches allocation failures cleanly (already proven safe by Follow-up 3's incident), but later
*mandatory* uploads (the MTP head) are not wrapped in a try/catch, so pushing far enough past what
the OS will actually back could still surface as an unhandled crash rather than a graceful message.

**Measured, at `STINGRAY_VULKAN_UMA_FRACTION=1.1`** (10% past the advertised heap;
`STINGRAY_DENSE_FFN_GPU_MARGIN_MB=256`): loaded cleanly, **53/64 FFN layers** admitted (27,030
MiB — more than the entire advertised 32,153 MiB heap on its own), decode reached **1.0-1.2 t/s**.
Verified clean via `Get-Counter` before and after: 46.4-46.7 GB available RAM throughout, ~22
pages/sec at rest (idle-level, not swapping). **Confirms the theory**: this specific AMD/RADV
driver's reported UMA heap size is advisory, not enforced — the real ceiling is actual system RAM
(63.3 GB total), several times larger than what the heap-size query alone would suggest.

**Stopped at `1.1` deliberately, per the user's own judgment call mid-session** ("tbh, I wouldn't
push it higher than this") — not because a further push was known to fail, but because the value
of the experiment (learning the clamp's assumption doesn't hold on this hardware) was already
captured, and further pushing trades a diminishing-returns speed gain against real risk given the
unwrapped mandatory-upload caveat above. `1.2`+ was not attempted.

**Updated recommendation (superseded further by Follow-up 10 below)**: `1.1` with
`STINGRAY_DENSE_FFN_GPU_MARGIN_MB=256` was the best-measured setting at time of writing. This does
**not** mean `1.1` (or the `2.0` ceiling now available) is safe to assume on a different
machine/driver/model — the whole point of this follow-up is that the advertised heap size's
relationship to real usable memory is driver-specific and was only established here by testing,
not derived from any general rule. Re-verify on any other hardware before trusting a value above
`1.0` there.

## Follow-up 10: three more levers tried — margin tuning, KV-cache compression, MTP (2026-09-16, later)

With placement now well-understood, checked three more candidate levers, all at
`STINGRAY_VULKAN_UMA_FRACTION=1.1` / `STINGRAY_DENSE_FFN_GPU_MARGIN_MB=256` for a stable baseline:

**1. Lowering `STINGRAY_DENSE_FFN_GPU_MARGIN_MB` further (256 → 64)**: **no effect** — identical
53/64 layers, identical 27,030 MiB uploaded, both runs. The margin isn't the binding constraint at
this fraction (something else — likely per-layer granularity against the inflated `VramBytes`
budget — decides the cutoff first). Not worth using a lower value than the already-tested 256.

**2. `--tq` (TurboQuant KV-cache compression)**: **cleanly rejected at startup** —
`"TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers)"`. Makes sense in
hindsight: this architecture's GDN layers use recurrent state instead of a traditional KV cache
(only the 16 attention layers have one at all), so there's nothing here for KV-cache compression
to meaningfully compress. Confirmed not applicable, not a bug — no further action.

**3. `--spec-type mtp` (MTP speculative decoding) vs `--spec-type none`**: **measurably,
reproducibly slower with MTP enabled**, on this configuration. Same prompt ("Write a detailed
paragraph explaining how photosynthesis works."), same `-n 40`, `--temp 0` (deterministic —
results were bit-identical across repeated runs of the same config, confirming this isn't noise):

| Config | Decode speed | MTP accept rate |
|---|---|---|
| `--spec-type none` (run 1) | 1.1 t/s | — |
| `--spec-type none` (run 2) | 1.0 t/s | — |
| `--spec-type mtp` (run 1) | 0.8 t/s | 80% (28/35) |
| `--spec-type mtp` (run 2) | 0.8 t/s | 80% (28/35) |

MTP is **~20-27% slower** than disabling speculative decoding entirely, despite a decent 80%
draft-accept rate. The likely explanation: the batched-verify step's extra dispatch/computation
overhead costs more than the accepted drafts save, consistent with this iGPU being
dispatch-overhead-bound rather than compute-bound for small extra batched work (the same class of
effect CLAUDE.md's standing finding on this exact machine describes for GPU work generally). This
was NOT true by assumption — `--spec-type auto` (the CLI's default) silently enables MTP whenever
a checkpoint supports it, meaning **the very numbers reported throughout this whole document's
performance-pass sections (Follow-ups 5, 8, 9) were measured with MTP silently active**, since none
of those runs passed an explicit `--spec-type` override. This doesn't invalidate those
measurements (both "before" and "after" states of each comparison had the same auto-MTP behavior,
so the *relative* deltas reported there are still valid), but it does mean an even better absolute
number than any of those tables is available today: **the same runs, with `--spec-type none`, on
top of the already-measured best placement settings**. All four runs in this table verified clean
via `Get-Counter` — 47-48 GB RAM free throughout, no swap.

**Updated recommendation**: `STINGRAY_VULKAN_UMA_FRACTION=1.1`,
`STINGRAY_DENSE_FFN_GPU_MARGIN_MB=256`, **`--spec-type none`** is now the best-measured
configuration in this document (~1.0-1.1 t/s, and combined with placement this is roughly 4-5.5x
the very first measurement in this whole session). Do not assume `--spec-type mtp`/`auto`'s default
behavior helps on this architecture/hardware combination — it measurably doesn't here; this is
exactly the kind of "plausible-sounding optimization that isn't actually faster" CLAUDE.md's
performance-pass rule warns against trusting without measuring.

---

## Original plan below (status: superseded for this checkpoint, kept for reference)
**Target:** `models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf` (hybrid-GDN, `qwen35` arch) — checkpoint
already present locally, no download needed.
**Backend:** `OpenTail.Stingray.Vulkan` / `VulkanHybridGdnForwardPass`
**Hardware:** dev box is a Ryzen 5700G iGPU (Vega, shared-RAM); this plan must not assume a
discrete GPU exists.
**Note:** the user calls this model "Qwen3.5-27B" (per PerformanceLeague.md/ChatGPT's writeup);
the checkpoint on disk and the arch tag in code is `qwen35`/`Qwen3.8-27B-UD-Q3_K_XL.gguf`. Same
model, plan applies as written.

Another agent is actively editing this codebase concurrently — **do not touch code** while
following this plan; it is investigation + design only, to be executed once the repo is free.

---

## 1. Confirmed root cause (read from source, not assumed)

`src/OpenTail.Stingray.Engine/VulkanHybridGdnForwardPass.cs:521-539`:

```csharp
if (ShouldKeepFixedWeightsOnGpu(
        model.FindTensor("token_embd.weight")!.Value,
        model.FindTensor("output.weight")))
{
    _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embDType);
    _gpuOutputNorm = UploadWeight("output_norm.weight");
    _gpuOutputWeight = model.FindTensor("output.weight") is not null
        ? UploadWeight("output.weight")
        : _gpuEmbedding;
}
else
{
    throw new NotSupportedException(
        "VulkanHybridGdnForwardPass: embedding/output do not fit in a single GPU storage " +
        "buffer (2 GB limit); CPU embedding fallback is not implemented in v1. Reduce ctx " +
        "size or use HybridGdnForwardPass for CPU-only execution.");
}
```

`ShouldKeepFixedWeightsOnGpu` (same file, `:2893-2919`) hard-codes the ceiling itself:

```csharp
const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;  // NOT queried from the device
```

Two things ChatGPT's writeup got right and one thing it assumed that isn't true here:
- **Right:** this is a single-VkBuffer-per-tensor problem, and the fix is sharding, not a CPU
  fallback.
- **Right:** the failure is in embedding/output only — every other per-layer weight in this model
  is far smaller than 2 GB and uploads fine today.
- **Not what's actually happening:** the current 2 GB ceiling is a **hard-coded constant**, not
  `VkPhysicalDeviceLimits.maxStorageBufferRange` queried from the device. `VulkanBackend.cs` never
  queries `maxStorageBufferRange` anywhere (confirmed by grep — zero hits). So step 1 of any real
  fix is: actually query the device limit, don't assume it's exactly 2 GiB-1. On many Vulkan
  drivers (including RADV/AMDVLK on this class of hardware) `maxStorageBufferRange` is
  `0xFFFFFFFF` (4 GiB−1) or the full `VkDeviceSize`, not 2 GiB — the 2 GiB number here looks like a
  conservative guess baked in, not a measured limit. **This must be logged and confirmed on this
  machine before sizing anything.**

### Actual tensor sizes for this checkpoint

Need to pull real values instead of trusting ChatGPT's illustrative vocab/hidden numbers — those
were for "Qwen3.5-27B" in the abstract, not measured from `Qwen3.8-27B-UD-Q3_K_XL.gguf`. Concretely
run (once the repo is free of other edits):

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-tensors -m models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf | grep -E "token_embd|output\."
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-metadata -m models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf | grep -iE "vocab|embedding_length|hidden"
```

to get the real `vocab_size`, `hidden_size`, and the on-disk dtype/byte size of `token_embd.weight`
and `output.weight` (this determines whether the *raw quantized* size or the *F32-expanded* size
is what's blowing past the limit — `EstimateEmbeddingGpuBytes`/`EstimateWeightGpuBytes` at
`:2906-2919` already encode which dtypes stay raw: Q4_K/Q6_K raw for embedding;
Q4_K/Q6_K/Q5_K/Q8_0/Q4_0 raw for the output weight — everything else gets F32-expanded, which is
the likely trigger for a UD-Q3_K_XL quant since Q3_K isn't in either raw-kept set and would fall
through to F32 expansion at 4 bytes/element).

**This is the load-bearing detail ChatGPT's plan missed**: if `output.weight` in this checkpoint
is quantized as Q3_K (the "XL" in UD-Q3_K_XL typically means some tensors get bumped to
higher-precision quants, so it could also be Q4_K/Q5_K/Q6_K for this specific tensor — must check,
not assume) and Q3_K isn't in `EstimateWeightGpuBytes`'s raw-kept switch, then `UploadWeight` is
about to F32-expand a tensor that's already borderline in raw quantized form, which would make the
overflow far worse than the raw quantized size suggests. Two independent fixes may be needed:
1. Add a `MatVecQ3K`-family raw-read Vulkan path so `output.weight` doesn't need F32 expansion at
   all (shrinks memory ~3-4x before sharding is even needed), similar to how Q4_K/Q6_K already
   avoid expansion.
2. Shard whatever remains too large for one buffer, regardless of (1).

Do (1) first and re-measure — it may make the actual overflow much smaller than the 2.37 GiB figure
ChatGPT computed for a hypothetical hidden=5120/vocab=248320 config, and may shrink the *number* of
shards needed to something as small as 2.

---

## 2. Existing abstractions relevant to sharding (read, not designed from scratch)

- `Tensor` (`src/OpenTail.Stingray.Core/Tensor.cs`): a thin `sealed class` wrapping
  `{Shape, DType, Handle}` where `Handle` is an opaque `nint` the backend interprets. **This is
  already backend-opaque** — nothing outside `VulkanBackend` inspects `Handle`'s bit layout. That
  means a sharded representation can be introduced entirely inside `VulkanBackend` by making
  `Handle` index into a "logical tensor" record that owns N `GpuBuffer`s, without changing the
  `Tensor` class or any call site in `VulkanHybridGdnForwardPass.cs` — satisfies ChatGPT's §22
  invariant ("a logical tensor must no longer imply a single Vulkan buffer") essentially for free,
  *if* every access path funnels through `VulkanBackend` methods rather than assuming
  `Handle` addresses one `VkBuffer` directly. Need to verify this holds for `EmbedLookup*` and
  `MatMul`/`GpuMatMul` specifically (see below) before committing to "free."
- `VulkanBackend._buffers` (`Dictionary<nint, GpuBuffer>` populated in `Allocate`/`Upload*`,
  `VulkanBackend.cs:1017-1444`): today, one handle → one `GpuBuffer` → one `VkBuffer`. This is the
  single choke point to generalize: change the value type from `GpuBuffer` to
  `GpuBuffer | GpuBufferShardSet` (or give every allocation a 1-element shard list uniformly, per
  ChatGPT's §6 "Shards.Count == 1 for ordinary tensors" — prefer this over a union type, it keeps
  every call site identical).
- `GpuBuffer` (`src/OpenTail.Stingray.Vulkan/GpuBuffer.cs`): per-buffer wrapper around
  `VkBuffer`/`VkDeviceMemory`. Read this file fully before implementing — need to confirm whether
  descriptor-set binding is baked into `GpuBuffer` itself or built separately in `ComputePipeline`
  dispatch call sites (`EmbedLookup`, `MatMul`) — this determines how invasive Option A (§4 below)
  actually is.
- `EmbedLookup`/`EmbedLookupQ4K`/`EmbedLookupQ6K` (`VulkanBackend.cs:2926+`, shaders at
  `Shaders.cs:1900-2018+`): take `(Tensor embTable, Tensor output, uint tokenId, uint embDim)` and
  bind `embTable` as one descriptor via `ComputePipeline(..., pushConstantSize: sizeof(EmbedParams))`.
  The token ID is already known on the CPU side at call time
  (`VulkanHybridGdnForwardPass.cs:2707-2713`), which is exactly what ChatGPT's §8 Option A wants —
  **CPU already selects which token to look up; extending it to also select which shard is a small
  , natural change**, not a new capability.
- LM-head: `GpuMatMul` (`VulkanHybridGdnForwardPass.cs:2718-2722`) calls `_gpu.MatMul(output,
  matrix, vector, dtype)`. Need to read `VulkanBackend.MatMul`'s implementation (not yet inspected
  in this pass — **do this before implementing**, specifically whether it already tiles internally
  or issues one dispatch per call) to know whether "shard the matrix, dispatch once per shard,
  write into `logits[offset..offset+shardRows]`" is a matter of calling `MatMul` N times with N
  shard-sized output-tensor **views** into one logits buffer, or requires a new overload.
- `_logitsBuf` / `_gpuLogits`: logits are already a small, separate GPU buffer
  (`VocabSize` floats — for a ~150K+ vocab that's well under 1 MB), confirming ChatGPT's §12: no
  new sharding is needed on the *output* side of the LM-head, only on the *weight* side.

---

## 3. Plan (follows ChatGPT's structure; corrected against what's actually in this repo)

### Phase 0 — Measure, don't assume (do this first, cheaply, before writing sharding code)
1. Query and log the real Vulkan device limits during `VulkanBackend` init: `maxStorageBufferRange`,
   `maxMemoryAllocationCount`, `maxPerStageDescriptorStorageBuffers`,
   `maxDescriptorSetStorageBuffers`, plus whether `VK_EXT_descriptor_indexing` /
   `shaderStorageBufferArrayNonUniformIndexing` / `runtimeDescriptorArray` /
   `descriptorBindingPartiallyBound` are exposed on this Radeon iGPU. This is a few lines added
   near wherever `VulkanBackend` already queries `VkPhysicalDeviceProperties`/features (find that
   site first — grep `VkPhysicalDeviceFeatures`/`GetPhysicalDeviceProperties` in `VulkanBackend.cs`).
2. Get real `token_embd.weight`/`output.weight` dtype + byte size for
   `Qwen3.8-27B-UD-Q3_K_XL.gguf` via `list-tensors`/`list-metadata` (commands above).
3. Replace the hard-coded `2L * 1024 * 1024 * 1024 - 1` in `ShouldKeepFixedWeightsOnGpu` with the
   actual queried `maxStorageBufferRange * 0.90` (ChatGPT's §5 formula — reasonable, keep it).
4. **Decide the real target overflow** using measured numbers, not the illustrative 2.37 GiB. It
   may turn out only `output.weight` overflows (embedding is typically smaller after Q4_K/Q6_K raw
   read), which would mean Phase 2 (LM-head sharding) is the only thing needed and Phase 1
   (embedding-lookup sharding) can be skipped for this specific checkpoint — but keep the general
   mechanism symmetric anyway since it's reusable (ChatGPT's §22 point stands).

### Phase 1 — Generalize the buffer-handle mapping (foundational, do regardless of shard count)
Add a shard-aware allocation path inside `VulkanBackend`:
- New internal type, e.g. `VulkanShardedBuffer { GpuBuffer[] Shards; long[] RowStart; long
  RowsPerShardExceptLast; long TotalRows; long ElementsPerRow; DType DType; }`.
- `_buffers` becomes `Dictionary<nint, VulkanShardedBuffer>` (or keep `GpuBuffer` for the common
  case and add a parallel `_shardedBuffers` dict keyed by handle, checked first — whichever is
  less invasive after reading `GpuBuffer`'s actual field layout; prefer minimizing diff to
  existing hot allocation paths like `Allocate`/`Upload` for ordinary tensors, since those must
  stay a true single-dispatch fast path per CLAUDE.md's "no unnecessary abstraction" rule).
- New `UploadWeightSharded(name, ...)` / `UploadEmbeddingWeightSharded(name, ...)` in
  `VulkanHybridGdnForwardPass.cs`, used only when `ShouldKeepFixedWeightsOnGpu` returns false —
  ordinary tensors keep using today's `Upload`/`UploadWeight` untouched.
- Row/shard-count math: `rowsPerShard = floor(safeBytes / bytesPerRow)`, then round down to a
  multiple of the tensor's quantization block size in rows (must not split inside a Q3_K/Q4_K/Q6_K
  256-element block — use `DTypeInfo.BlockSize`/`BytesPerBlock` from `Tensor.cs:119-233`, already
  in this codebase, do not hand-roll block math). Confirm against the real GGUF row layout for
  `token_embd.weight`/`output.weight` (row = one vocab entry = `hidden_size` elements) before
  finalizing — a block boundary only aligns cleanly with a row boundary if `hidden_size %
  blockElementsPerRow == 0`, which needs checking for hidden_size once measured in Phase 0.

### Phase 2 — LM-head (`output.weight`) sharding — do this before embedding, it's the more likely overflow
- Loop shard dispatch (ChatGPT §9-10): for each shard, call the existing `MatMul`-based
  `GpuMatMul` against a `Tensor` that is a *view* over one shard (`_gpuOutputWeight` shard i,
  `hidden` vector unchanged, output = `_gpuLogits` sliced at `[rowStart, rowStart+rowCount)`). No
  readback between shards — logits stay one GPU-resident buffer, exactly as ChatGPT specifies in
  §12; only difference from ChatGPT's writeup is that here logits are already this small/resident,
  so no new buffer is needed, just per-shard dispatch into subranges of the existing one.
- All four call sites that currently do `GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden)`
  (`VulkanHybridGdnForwardPass.cs:893, 1046, 1898`) and the batched variant at `:1435` need the
  sharded variant swapped in behind a single helper (e.g. `GpuMatMulSharded`) so there's one place
  to fix, not four — keep this DRY per project convention.

### Phase 3 — Embedding lookup sharding (only if Phase 0 measurement shows it's actually needed)
- CPU already knows `tokenId` at the `EmbedLookup*` call site (`:2707-2713`) — compute
  `shardIndex = tokenId / rowsPerShard`, `localTokenId = tokenId % rowsPerShard`, and pass the
  selected shard's `Tensor`/handle into the existing `EmbedLookup`/`EmbedLookupQ4K`/`EmbedLookupQ6K`
  methods unchanged (ChatGPT §8 Option A — ratified as the right initial choice; Option B/bindless
  descriptor arrays deferred, matches ChatGPT's own §11 "avoid unnecessary descriptor complexity
  in v1").
- **Caveat found in code that ChatGPT's writeup didn't know about**: whichever raw-quant dtype
  `token_embd.weight` uses determines which of the three `EmbedLookup*` methods is called
  (`:2707-2713` switches on dtype); each shard must independently record which raw dtype it kept
  (mirrors `_embDType`/`_gpuWeightDTypes` bookkeeping already in the class) — a shard boundary
  must not accidentally change which lookup shader is selected for tokens on either side of it,
  since all shards of the same logical tensor are the same dtype by construction (this is
  automatically satisfied if sharding happens after the existing raw-vs-F32-expand decision, not
  before — implement it that way).

### Phase 4 — Correctness tests (ChatGPT §18-19, agree with the approach)
Add to `tests/OpenTail.Stingray.Tests.Vulkan` (check exact project name first — grep existing
Vulkan test project list before assuming):
- Synthetic test: allocate a tensor deliberately larger than the queried
  `maxStorageBufferRange`, verify it shards, verify round-trip upload/download per shard matches a
  CPU reference array.
- Embedding: compare CPU dequantized lookup vs sharded GPU `EmbedLookup*` for token IDs at
  `row = 0`, `rowsPerShard - 1`, `rowsPerShard`, `rowsPerShard + 1`, last shard's first/last row.
- LM-head: known hidden vector → compare sharded GPU matvec output against CPU reference (existing
  `HybridGdnForwardPass` CPU path is the natural oracle here — it already implements the identical
  math and is in-repo, no need for a new Python/C++ reference per CLAUDE.md rule 8/coverage-tooling
  no-Python convention) at the same boundary rows.
- Full-model smoke test: load `Qwen3.8-27B-UD-Q3_K_XL.gguf` on `--backend vulkan`, confirm
  `VulkanHybridGdnForwardPass` no longer throws, generate a few tokens, sanity-check output isn't
  garbage (e.g. produces valid UTF-8/known-vocab tokens, not NaN/all-zero logits).

### Phase 5 — Performance pass (per CLAUDE.md rule 7, required once this is wired end-to-end)
- Measure tokens/sec before (N/A, currently throws) vs after sharding, and separately measure
  shard-dispatch overhead in isolation (ChatGPT §20 Optimization 4) — with real weights, a
  realistic (not trivially short) prompt, several samples.
- Given this is an iGPU with no dedicated VRAM bandwidth advantage (per CLAUDE.md rule 13's
  standing finding on this exact machine), do not be surprised if per-shard dispatch overhead is
  proportionally more visible here than it would be on a discrete GPU — write the actual measured
  numbers down regardless of outcome, and do not conclude the sharding approach is bad based on
  this machine alone (same caveat as rule 13).

---

## 4. Explicitly deferred / out of scope for v1 (agree with ChatGPT's §24, restated)
- No CPU embedding or LM-head fallback, ever, for this path.
- No forced context-size reduction as "the fix."
- No 64-bit storage-buffer indexing dependency (investigate only if it's cheap to check whether
  RADV/AMDVLK on this iGPU exposes it — do not block on it).
- No bindless/descriptor-array redesign in v1 — CPU-selects-shard (Option A) first.
- No Qwen3.8-specific hard-coded shard count — shard count must fall out of the queried device
  limit and the real tensor size.

## 5. Open questions to resolve during Phase 0 before writing any sharding code
1. Real `maxStorageBufferRange` on this Radeon iGPU (RADV or AMDVLK, whichever driver this box
   uses — check `vulkaninfo` if available, or read it back from `VulkanBackend`'s own log once
   instrumented).
2. Real dtype of `output.weight` in `Qwen3.8-27B-UD-Q3_K_XL.gguf` (Q3_K vs a bumped-precision
   quant under the "XL" naming) — determines whether a new raw-read matvec path is needed before
   sharding even becomes necessary, and by how much.
3. Whether `token_embd.weight` and `output.weight` are tied in this checkpoint (`model.FindTensor
   ("output.weight") is not null` at `:527` suggests they may not be — if `output.weight` is
   absent and the model ties embeddings, only one oversized tensor exists, halving the work).
4. Exact signature of `VulkanBackend.MatMul` (not yet read in this pass) — determines whether
   shard dispatch is "call MatMul N times with view-tensors" or needs a new overload.
