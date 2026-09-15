# Vulkan large-tensor sharding — Qwen3.8-27B (qwen35 hybrid-GDN) GPU-only support

**Status:** plan (not yet implemented)
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
