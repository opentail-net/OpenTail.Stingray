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

## Next steps (not started)

1. Pick a minimal real op set from `mlx-c` (array creation from data, basic arithmetic, matmul) and
   bind those next — still checkable on CPU-only logic (shapes, dtype mapping) even without a Mac to
   run them, the same way `CudaTensorParallelSharding`'s pure math was unit tested without a GPU.
2. Implement enough of `IComputeBackend` for a dense-llama forward pass (embedding, attention via MLX
   ops, RMSNorm, RoPE, SwiGLU FFN, KV cache) — comparable in size to `CudaForwardPass` (~7,000 lines),
   written blind with no way to run it. Real design work, not a continuation of the skeleton.
3. A flag (`STINGRAY_MLX_*` or similar, registered the same way `STINGRAY_TP_DEGREE` was) once there is
   actual behaviour to gate.
