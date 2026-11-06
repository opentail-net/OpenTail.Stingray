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

## Next steps (not started)

1. Implement enough of `IComputeBackend` for a dense-llama forward pass (comparable in size to
   `CudaForwardPass`) by wiring these seven ops into weight loading, a KV cache, and the decode loop.
   Real design work, not another op port — this is the next distinct phase.
2. A flag (`STINGRAY_MLX_*` or similar, registered the same way `STINGRAY_TP_DEGREE` was) once there is
   actual behaviour to gate.
2. Implement enough of `IComputeBackend` for a dense-llama forward pass (comparable in size to
   `CudaForwardPass`, ~7,000 lines) once enough ops exist. Real design work, not a continuation of the
   skeleton.
3. A flag (`STINGRAY_MLX_*` or similar, registered the same way `STINGRAY_TP_DEGREE` was) once there is
   actual behaviour to gate.
