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
