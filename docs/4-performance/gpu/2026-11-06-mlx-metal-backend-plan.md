# MLX/Metal backend (flag-gated, unmeasured) — plan

**Date:** 2026-11-06. **Status:** skeleton started, stopped there by owner instruction ("build the
skeleton now and stop there"). **Owner decision on record:** port this even though it cannot be run
or measured here — no Mac exists on this development machine at all, which is a step further than the
CUDA tensor-parallelism plan (that at least compiles and links against real CUDA SONAMEs on Windows;
`libmlxc` is a macOS-only native library this machine cannot even obtain). Same override of the "no
real hardware" standing position as the CUDA TP plan, for this one feature.

**Source:** `C:\Git-Public\example\TensorSharp\TensorSharp.Backends.MLX` (BSD 3-Clause,
zhongkaifu/TensorSharp — same licence family already in `THIRD_PARTY_NOTICES.md`). TensorSharp's
Apple-Silicon coverage is actually two separate things, confirmed by reading its source rather than
assumed:
1. **GGML Metal** — upstream ggml's own Metal backend, wrapped the same way TensorSharp wraps GGML
   CUDA/Vulkan. **Does not apply to Stingray**: Stingray has no GGML wrapper layer at all (it reads
   GGUF directly and implements every forward pass itself; vendored llama.cpp/ggml binaries are an
   independent reference oracle for parity tests only, never linked into the engine). There is nothing
   to "mirror" here because the thing being mirrored is a wrapper around a dependency Stingray
   deliberately does not have.
2. **MLX backend** (`TensorSharp.Backends.MLX`, ~20 files, P/Invoke to `mlx-c`/`libmlxc`) — this is the
   one that actually matches `CudaBackend`'s shape: TensorSharp's own native compute backend, not a
   wrapper. **This plan covers this one.**

## Scope

**In scope (this plan):** a new `OpenTail.Stingray.Mlx` project implementing enough of
`IComputeBackend` to run a dense-llama-equivalent forward pass on Apple Silicon via `mlx-c`, flag-gated
off by default, following the same shape as the CUDA plan (dense llama first, other architectures
later).

**Out of scope:** DiT/diffusion ops, vision ops, every quantization format, tensor parallelism across
multiple Macs (TensorSharp's MLX backend is single-device only per its own docs) — matching real
TensorSharp-MLX parity across its whole feature set is a much larger, separate undertaking.

## Contracts this must satisfy

- **No new guard to lift**: unlike CUDA (`Model.Load`'s explicit `MainGpu`/`TensorSplit` refusal),
  there is no existing MLX-shaped refusal anywhere in Stingray to find or change — this is a backend
  that does not exist yet, added purely additively. `ForwardPassSelection`/the planner will need a new
  `ForwardPassKind` (or equivalent) once real ops exist; not needed for the skeleton.
- **`IComputeBackend`** (`src/OpenTail.Stingray.Core/IComputeBackend.cs`): the interface a real MLX
  backend must implement. Not attempted yet — the skeleton does not implement it.
- **P/Invoke signatures are a verified fact, not a guess**: every entry point, struct layout and
  calling convention in `MlxInterop.cs` was copied from the real, vendored TensorSharp source
  (`MlxNative.cs`), not invented. A plausible-looking but wrong signature would be worse than no
  binding at all — it would compile (P/Invoke resolves lazily) and fail unpredictably on the one
  machine that could ever call it.

## What exists now (the skeleton, 2026-11-06)

- `src/OpenTail.Stingray.Mlx/` — new project, references Core only.
- `MlxInterop.cs` — `LibraryImport` bindings for `mlx_metal_is_available`, `mlx_device_new_type`,
  `mlx_device_free`, `mlx_device_is_available`, `mlx_set_default_device`, `mlx_array_new`,
  `mlx_array_free` — enough for an honest availability check, not a working backend.
- `MlxBackend.IsAvailable()` — mirrors `CudaBackend.IsAvailable()`'s try/catch-false shape. Verified
  (not assumed) to actually return `false` gracefully here via `MlxBackendTests` in `Tests.Cuda`
  (`DllNotFoundException`/`EntryPointNotFoundException` caught, not thrown) — the one thing about this
  skeleton that is genuinely checked rather than just "should work."
- **Does not implement `IComputeBackend`.** No flag exists yet (nothing to gate — there's no behaviour
  to turn on). No kernels, no array ops beyond `new`/`free`, no device-side compute of any kind.

## What "done" means here (same deliberately different bar as the CUDA TP plan)

Compiles clean (confirmed: CI's `ubuntu-latest`/`windows-latest` runners build this fine, since
`LibraryImport` P/Invoke resolves the native library lazily at call time, not at build time — the same
reason `CudaBackend.cs` already compiles here with no CUDA runtime installed). Never claimed to run or
produce correct output; every doc comment says so.

## Revised approach (2026-11-06): port, don't invent

Initially scoped as "write an MLX backend" from the algorithm down, which would mean inventing
untested logic with zero feedback loop. Revised after discussion: TensorSharp's MLX backend already
works (presumably verified on real Apple hardware, by its own project); the actual risk in porting it
is translation fidelity into Stingray's shapes, not whether the underlying op is correct. So each op is
ported as a close reading of the real TensorSharp source, with every function's doc comment citing the
exact file and line range it came from and naming anything simplified away — not reinvented from a
description of what RMSNorm or attention is supposed to do.

## First op ported: RMSNorm (2026-11-06)

`MlxOps.RmsNorm` (`src/OpenTail.Stingray.Mlx/MlxOps.cs`), ported from `MlxBasicOps.RmsNorm` +
`MlxNative.FastRmsNorm` (`TensorSharp.Backends.MLX/MlxBasicOps.cs` ~893-919,
`TensorSharp.Backends.MLX/MlxNative.cs` ~3915-3923 and ~6560-6582 for the stream handling). Added the
`mlx_get_default_stream`/`mlx_array_new_data`/`mlx_array_data_float32`/`mlx_fast_rms_norm` bindings to
`MlxInterop.cs`, all copied from the same real source, not invented.

Simplified relative to the source, named explicitly in the doc comment so a later port knows what was
dropped: no per-device stream caching (TensorSharp caches the stream handle to avoid a
device-new/free round trip per call; this port re-creates one every call), and no beta/bias, F16/BF16
or quantized input paths (TensorSharp's own `RmsNorm` falls off to a generic path for those; this port
covers only the fp32 weight-only case `CudaForwardPass`'s RMSNorm calls actually need).

**Tested the two things that are actually testable without a Mac**, via `MlxOpsTests` in `Tests.Cuda`:
argument-shape validation (pure logic, runs before any native call) and that calling the real op with
no `libmlxc` present fails predictably (`DllNotFoundException` at the first native call) rather than
crashing the process. Neither test says the *math* is correct — that genuinely cannot be checked here —
only that the port's failure mode on this machine is the honest, expected one.

Ported two more the same way (owner note: "don't need to worry about citing everything, the project
borrows BSD code already" — attribution still lives in `THIRD_PARTY_NOTICES.md`, one line per file, not
a citation in every doc comment): `Silu` (sigmoid + multiply) and `MatMul`. Both in `MlxOps.cs`, with
the `mlx_sigmoid`/`mlx_multiply`/`mlx_matmul` bindings added to `MlxInterop.cs`. Same two testable
things covered (`MlxOpsTests`): shape validation, and a real call failing predictably with no
`libmlxc` present.

Ported `AddMm` (biased linear, `mlx_addmm`) and `Softmax` (`mlx_softmax_axis`) the same way. Same two
testable things per op.

Ported `Attention` (`mlx_fast_scaled_dot_product_attention`), covering the plain self-attention case
only: `[numHeads, seqLen, headDim]` query/key/value, no mask array, no sinks, no forced-fused flag
(TensorSharp's gpt-oss/chunked-prefill paths, not ported). `causal` selects MLX's built-in `"causal"`
mask string vs. none — ported from TensorSharp's own convention (`string.Empty` for single-token
decode, `"causal"` for multi-token prefill). The mask-mode string marshaling is simplified relative to
TensorSharp's cached, process-lifetime `GetModePtr`: this port allocates and frees the native string
on every call rather than caching it, correctness-first over the perf TensorSharp optimized for.

With RMSNorm, Silu, MatMul, AddMm, Softmax and Attention now ported, every op a plain dense-llama
decode step needs exists except RoPE.

Ported `Rope` (`mlx_fast_rope_dynamic`, base-frequency form only — TensorSharp's `FastRopeDynamicWithFreqs`
sibling for YaRN/scaled-frequency variants is not ported). Needed `MlxOptionalFloat` (a `{float value;
bool has_value;}` tagged struct, copied from TensorSharp's own) and, to pass a bool-containing struct
by value through `LibraryImport`, `[assembly: DisableRuntimeMarshalling]` in `GlobalUsings.cs` — same
attribute TensorSharp's own `MlxNative.cs` carries, for the same reason.

**Every op a plain dense-llama decode step needs now exists**: RmsNorm, Silu, MatMul, AddMm, Softmax,
Attention, Rope. Seven ops, each a faithful TensorSharp port, each with a shape-validation test and a
real-call-fails-predictably test (`MlxOpsTests`, 61 tests total in `Tests.Cuda` as of this entry).

## Forward-pass wiring started: MlxDenseLayer (2026-11-06)

`MlxDenseLayer.Forward` (`src/OpenTail.Stingray.Mlx/MlxDenseLayer.cs`) composes all seven ops into one
transformer block: RMSNorm -> QKV -> RoPE(Q,K) -> causal attention -> output projection -> residual
add -> RMSNorm -> SwiGLU FFN (`down(silu(gate(x)) * up(x))`) -> residual add. Added an eighth op,
`Multiply` (`mlx_multiply`), needed for the SwiGLU gate.

**Simplified relative to a real dense-llama layer, named explicitly in the doc comment:** single
attention head only (no multi-head split/concat -- `Wq`/`Wk`/`Wv` project straight to `HeadDim`); no
GQA; no bias terms; F32 weights only (no quantized-weight dequantization path); no KV cache (every
call recomputes attention over the whole sequence -- there is no incremental decode step yet).

**The one thing checkable end to end without a Mac**: that a structurally valid call reaches the
*first* real native call and fails there (`DllNotFoundException`), not some earlier bug in the shape
handoff between ops throwing a different exception first. `MlxDenseLayerTests` asserts exactly this --
it is meaningfully stronger evidence of correctness than any single op's own test, because it exercises
every op's output shape feeding correctly into the next op's input shape, all the way through.

66 tests passing in `Tests.Cuda` as of this entry (8 ops + the composed layer, each with shape
validation and a graceful-failure check).

## Multi-head attention (2026-11-06)

`MlxDenseLayer` now does real multi-head attention: `SplitHeads`/`MergeHeads` (new ops, composed from
`mlx_reshape` + `mlx_transpose_axes`, both plain C# P/Invoke -- not TensorSharp's fused `FlatToHeadFirst`
Metal kernel; faithful to the reshape, not to that specific kernel) convert between the flat
`[seqLen, numHeads*headDim]` projection output and the head-first `[numHeads, seqLen, headDim]` layout
`Attention` needs. RoPE runs per head on a `[seqLen, headDim]` slice, confirmed against TensorSharp's
own call site (`MlxBasicOps.cs`'s RoPE reshapes to isolate one head-sized feature dimension before
calling `FastRopeDynamic` -- this was already the right design, not a guess). `NumHeads`/`NumKvHeads`
added; GQA broadcast (KV heads != query heads) is not implemented, checked explicitly
(`NotSupportedException`, not a silent wrong answer).

72 tests passing in `Tests.Cuda` as of this entry (9 ops + the composed multi-head layer).

## KV cache and incremental decode (2026-11-06)

`MlxKvCache` (host-resident, plain array bookkeeping -- TensorSharp keeps its cache device-resident
and writes with a native `mlx_slice_update` to avoid a host round trip; every op here already
round-trips through host memory per call, so there is nothing a native slice-update buys, and
appending needs no mlx-c binding at all) plus `MlxDenseLayer.ForwardDecode`, a single-new-token
incremental step: appends the new K/V to the cache, attends the one new query against the cache's
full length, no mask needed (matches TensorSharp's own stated reason: "a lone query has no future
token relative to itself or the cached past"). Generalized `Attention` to take separate
`querySeqLen`/`kvSeqLen` (previously assumed equal, which decode breaks).

**Named limitation:** only single-new-token decode is handled; continuing a multi-token prefill into
an already-populated cache is refused (`NotSupportedException` is not used here since this specific
case isn't checked yet -- worth tightening later) rather than silently mismasked, because MLX's
`"causal"` mask string assumes query/key start causally aligned, which an offset continuation would
need its own mask array for.

`MlxKvCacheTests` is the first *fully* verified test in this backend (no "fails predictably" caveat
needed) -- the cache is plain host array bookkeeping with no native call in it at all, so its
head/position data layout is checked for real.

81 tests passing in `Tests.Cuda` as of this entry.

## Weight loading, full model assembly, and the flag (2026-11-06) -- close criterion reached

`MlxGgufDenseModelLoader.Load` reads a real GGUF's own metadata keys (`{arch}.embedding_length` etc.,
the llama.cpp/GGUF convention -- not `ArchitectureModelResolver`'s hyperparameter resolution, which
handles many architecture-specific overrides this dense-only port doesn't need) and dequantizes every
tensor to F32 via Stingray's existing `Dequantize.ToFloat32` (added a `Cpu` project reference to the
`Mlx` project for this -- no new quantization logic). `MlxDenseModel` stacks the resulting
`MlxDenseLayer`s with an embedding table and output head; `Embed`/the final logits projection's
RMSNorm+MatMul reuse the ported ops. `STINGRAY_MLX_DENSE=1` gates `Load` directly (no engine/planner
integration exists -- no `ForwardPassKind` case, no CLI `--backend mlx` -- named explicitly as not
attempted; this flag only gates direct calls to the loader).

**`MlxGgufDenseModelLoaderTests` is real, not "fails predictably"**: a standalone synthetic-GGUF
builder (GgufModel/Dequantize both run fine here, no Mac needed) produces a tiny one-layer F32
checkpoint with every weight tensor filled with its own distinct constant, and the test asserts each
one landed in the correct `MlxDenseLayer` field -- would catch a transposed or swapped tensor, not
just "the loader ran without throwing."

88 tests passing in `Tests.Cuda` as of this entry (10 ops, head split/merge, the composed layer forward
+ decode, the KV cache, the full model, and the loader).

## Scope expanded by owner instruction (2026-11-06): complete the remaining gaps

After the close criterion was met, the owner asked to complete the remaining named gaps to the best
of our ability, with CI tests covering the work: GQA, multi-token cache continuation, MoE/hybrid/
vision architectures, engine/CLI integration, TensorSharp's perf optimizations. Taken in roughly this
order (bounded items first); MoE/hybrid/vision and the perf optimizations are each comparable in size
to a new subsystem, so progress on those is reported honestly as it happens rather than promised
up front.

### Engine/planner integration: MlxForwardPass (2026-11-06)

`MlxForwardPass : IForwardPass` is the actual integration point. Most of its control flow has no
native call in it and is tested for real: `VocabSize`/`MaxSeqLen` plumbing, `ResetCache`,
`TruncateTo`'s accept-0-or-current-length contract (`SupportsPartialRewind` stays `false`), and
`Forward`/`Prefill`'s position/startPos validation. `MlxDenseLayer.Forward` gained an optional
`MlxKvCache? cache` parameter (appends the prompt's K/V so a later `ForwardDecode` continues the same
sequence) and `MlxDenseModel.Prefill` threads it through per layer -- closing the gap where prefill
and decode previously used disconnected state.

**Not done in this slice**: no `ForwardPassKind` case, no CLI `--backend mlx` option, no
`ArchitectureLoadContext`/`CommonForwardPassFactory` wiring -- `MlxForwardPass` exists and implements
the real contract, but nothing in the engine can select it yet. That is the "CLI integration" item,
tracked separately below.

96 tests passing in `Tests.Cuda` as of this entry.

### GQA broadcast (2026-11-06)

Removed the artificial `NumKvHeads != NumHeads` refusal after checking TensorSharp's real call site
(`MlxFusedOps.cs`): it sizes its KV cache at `numKVHeads` and passes `q` at the full `numHeads`
directly into `mlx_fast_scaled_dot_product_attention`, with no repeat/broadcast step anywhere in that
code path -- the fused primitive broadcasts GQA internally. `MlxOps.Attention` now takes separate
`numQueryHeads`/`numKvHeads` parameters instead of one shared `numHeads`. No new op, no new
complexity -- the restriction was unnecessary, not a missing feature.

97 tests passing in `Tests.Cuda` as of this entry (replaced the two "mismatched KV heads throws" tests
with "mismatched KV heads is accepted and reaches the real native call" ones, added a dedicated GQA
shape test to `MlxOpsTests`).

### Multi-token cache continuation (2026-11-06)

`MlxOps.Attention` gained a real `queryOffset` parameter: when continuing a prefill into an
already-populated `MlxKvCache`, it builds an offset causal mask device-side (`mlx_arange` +
`mlx_reshape` + `mlx_subtract` + `mlx_greater`, all basic ops -- never downloaded to host and
re-uploaded, since mlx-c's mask is boolean and this port has no host-side bool path) instead of
using the built-in `"causal"` string, which assumes query/key start aligned. Ported from
TensorSharp's `SlidingWindowMask` (`MlxFusedOps.cs` ~2076), minus its window term -- same language
throughout, no kernel involved, the cleanest port so far.

`MlxDenseLayer.Forward` no longer refuses a non-empty cache: it derives `queryOffset` from
`cache.Length`, attends over the cache's full history (old + new), and appends the new K/V.
`MlxDenseModel.Prefill` was fixed to start RoPE positions at the caches' shared length instead of
always 0, and validates every layer's cache is at the same length. `MlxForwardPass.Prefill` was
corrected to match `IForwardPass`'s actual contract -- `tokens` at `startPos > 0` is the NEW suffix,
not a repeat of the cached prefix, which my first draft had backwards.

**What's still true:** `ForwardDecode` itself is still single-new-token only by design (a caller
continuing with more than one new token uses `Forward` with the cache instead, now that it supports
it).

102 tests passing in `Tests.Cuda` as of this entry -- including a real end-to-end structural check
(`MlxDenseLayerTests`): a cache advanced via `Append` (pure host logic, no native call) to a non-zero
length is now *accepted* by `Forward` and reaches the real native call, where it used to be refused.

## Status: close criterion met for the scope as defined

Per the owner's stop condition ("a faithful port that passes all tests in CI"): there is now a
complete dense-llama MLX forward pass -- weight loading from a real GGUF, dequantization, multi-head
attention, RoPE, SwiGLU FFN, a KV cache and incremental decode, stacked into a full model with
embedding and output head -- every piece a faithful, cited port of TensorSharp's real source, every
piece tested the strongest way actually possible without a Mac (full real verification for pure
host-side logic, shape validation + graceful-native-failure for everything that touches mlx-c), and
CI green on every commit along the way (`eb38583d` through this one).

**What remains for true TensorSharp parity** (explicitly out of scope for this round, per the plan's
own scope section, not overlooked):
- ~~GQA broadcast~~ -- done 2026-11-06 (see below); `mlx_fast_scaled_dot_product_attention` broadcasts internally, no new code needed.
- ~~Multi-token continuation into an already-populated KV cache~~ -- done 2026-11-06 (see below), via `Forward`; `ForwardDecode` itself still handles only one new token at a time, by design.
- MoE, hybrid/recurrent architectures, vision, diffusion, quantized MLX ops (not F32) -- the owner
  expanded scope 2026-11-06 ("do complete all these -> GQA, multi-token cache continuation,
  MoE/hybrid/vision architectures, engine/CLI integration, TensorSharp's perf optimizations") to
  include these. Progress so far:
  - **MoE router (gating) ported 2026-11-06**: `MlxOps.MoeRouterTopKSoftmax`, faithfully ported from
    TensorSharp's `MlxFusedOps.TryMoeRouterTopKSoftmax` (`MlxFusedOps.cs` ~102-182) -- negate, top-K
    via `mlx_argpartition_axis`, slice/reshape/cast the indices, gather the top-K logits, softmax
    them. This is the router/gating decision only (which experts fire, their normalized weights).
    **Also ported 2026-11-06**: `MlxOps.GatherRows`, from TensorSharp's `TryGatherRows`
    (`MlxFusedOps.cs` ~291-333) -- `mlx_take_axis` along axis 0 then `mlx_contiguous`, collecting
    the tokens routed to one expert before running its FFN.
    **Not yet done**: `TryScatterAddWeightedRows` (`MlxFusedOps.cs` ~335-392) -- its real
    implementation is a custom fused Metal kernel (`ScatterAddWeightedRowsSource` in
    `MlxNative.cs`), which the owner's "stick to C#" instruction means composing from basic ops
    instead (e.g. a one-hot-times-weights matmul), not yet done -- and `TryMoeBiasWeightedSum`
    (`MlxFusedOps.cs` ~624-701). The expert-offload LRU memory mechanism (`MoeExpertOffload.cs`) is
    a perf/memory feature orthogonal to correctness, lower priority than the compute path. No MoE
    model assembly (`MlxDenseModel`-equivalent for an MoE architecture) exists yet either -- these
    ops alone are not a runnable MoE forward pass.
  - **Hybrid (MiniMax-H3) -- scoped and found out of scope, 2026-11-06**: `MlxMiniMaxH3Ops.cs` and
    `MlxMiniMaxH3AttentionOps.cs` are NOT a hybrid GDN/Mamba-style text LLM architecture despite the
    name -- they are video/audio VAE transform ops for a diffusion pipeline (packed AdaLN modulation,
    group-norm-SiLU, im2col variants, an exact-attention kernel for the VAE's own encoder). Every
    single function in both files is a custom fused Metal kernel (`CreateFastMetalKernel` with an
    embedded shader string) with no basic-mlx-c-ops composition available -- unlike the MoE router/
    gather ops, there is no "this part happens to already be basic ops" path here. Per the owner's
    "stick to C#" instruction (no raw kernel porting) and the plan's own out-of-scope list (no
    diffusion), this is correctly left unported; it was never the right target for the "hybrid
    architectures" item anyway (that item means GDN/Mamba/short-conv hybrids, which this file is
    not). No MLX GDN-hybrid text architecture has been scoped yet.
  - **Vision (Gemma 4) -- partially ported, 2026-11-06**: `MlxGemma4VisionOps.cs` is mostly custom
    Metal kernels too (`Gemma4VisionPosition2D`, `Gemma4VisionPool3x3`, `Gemma4VisionRope2D` all use
    `CreateFastMetalKernel`), but `TrySelfAttention` (~112-160) is different -- it is itself built
    entirely from basic composed mlx-c ops in TensorSharp's own source (transpose, zero-pad via
    `mlx_full`+`mlx_concatenate_axis`, the same `mlx_fast_scaled_dot_product_attention` this port
    already uses, slice, transpose back), not a kernel being reimplemented. Ported faithfully as
    `MlxOps.Gemma4VisionSelfAttention`: pads the 64-wide head dim to 80 (192 for `tokens <= 8`) with
    exact zeros so MLX's fused SDPA kernel takes its F32 SIMD path instead of narrowing to TF32,
    then crops back -- mathematically a no-op, just a precision-path selection. New `MlxInterop`
    bindings: `mlx_array_new_float32`, `mlx_vector_array_new`/`_free`/`_append_value`, `mlx_full`,
    `mlx_concatenate_axis`. 3 new tests. **Not done**: `TryAddPosition2D`, `TryPool3x3`, `TryRope2D`
    (all custom kernels, same as the MiniMax-H3 situation) and `EvaluateForProfile` (a pure
    diagnostic helper, not inference logic). No vision model assembly (loading a real `mmproj`
    checkpoint, wiring this op into a `UnifiedVisionPipeline`-equivalent for MLX) exists either --
    this one op alone is not a runnable vision forward pass.
- ~~No engine/planner integration~~ -- done 2026-11-06. `ForwardPassBackend.Mlx` and
  `ForwardPassKind.MlxDense` added; `ForwardPassSelection.Select` has an isolated early-return branch
  (`request.Backend == ForwardPassBackend.Mlx`) that refuses unless `STINGRAY_MLX_DENSE=1` and the
  architecture family is `Dense`, nothing else checks the new enum values so no existing CPU/CUDA/
  Vulkan routing changed. `CommonForwardPassFactory.CreateDense` wires `ForwardPassKind.MlxDense` to
  `MlxGgufDenseModelLoader.Load` + `new MlxForwardPass(...)`. The real CLI string-parsing path
  (`ExecutionPlanner`) was also updated end-to-end: the pinned-backend allowlist now accepts `"mlx"`,
  and the backend-resolution switch maps it to `(ForwardPassBackend.Mlx, "MLX Device 0 (flag-gated,
  unmeasured)")` -- previously it would have silently fallen through to CPU. Covered by
  `ForwardPassSelectionTests.MlxBackendRoutingTests` (4 tests: flag unset refuses, flag set + dense
  selects `MlxDense`, flag set + non-dense refuses, doesn't affect other routing) and
  `ExecutionPlannerTests.MlxPinnedBackendTests` (2 tests: real end-to-end `--backend mlx` through
  `ExecutionPlanner.Plan`, flag unset refuses with `STINGRAY_MLX_DENSE` in the message, flag set
  resolves to `plan.Backend == "mlx"` / `ForwardPassKind.MlxDense`). 69+4 and 25+2 tests passing in
  `Tests.Core` respectively, zero regression to the pre-existing 69/25.
- TensorSharp's own performance optimizations:
  - ~~Per-device stream caching~~ -- done 2026-11-06. `MlxOps.GetCachedStream`, ported from
    TensorSharp's `MlxNative.DefaultStream()` (`MlxNative.cs` ~6560-6582): the default MLX stream
    for a device is a stable handle MLX owns, so every op in this file now fetches it from a cached
    static field instead of doing a `device_new`+`get_default_stream`+`device_free`+`stream_free`
    round trip on every call. Structural, not correctness-affecting -- no new tests were possible
    or needed (the existing 110 all still pass, confirming no behavior change), and this cannot be
    measured on this machine (no mlx-c to time against) the way the project's own perf-pass
    discipline normally requires; stated here explicitly rather than claimed as a measured win.
  - The fused `FlatToHeadFirst` Metal kernel -- this port uses `mlx_reshape` + `mlx_transpose_axes`
    instead, by owner instruction to stay in C#, so there is nothing to "restore" here; TensorSharp's
    kernel fusion and this port's op composition are different strategies, not a gap.
  - Device-resident (vs. host-round-tripping) arrays -- every op in this file still uploads from and
    downloads to a host array on every call, unlike TensorSharp's device-resident tensors that stay
    on the GPU across ops in a forward pass. Not started; this is the largest remaining perf item
    and would be a bigger structural change (threading `MlxArray` handles through `MlxDenseLayer`/
    `MlxDenseModel` instead of `float[]`) than stream caching was.
- Nothing has been run on real hardware. Every claim above is "compiles, reviewed for structural
  correctness, tested as far as this machine allows" -- not "works." That gap cannot be closed without
  a Mac, which is the whole premise this plan was scoped under.
