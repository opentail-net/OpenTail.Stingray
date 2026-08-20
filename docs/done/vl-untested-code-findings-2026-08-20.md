# VL models: three real bugs found by running one, for the first time ever

**Context:** part of the 2026-08-20 project-by-project review. The `VisionOps.Attention`/
`AttentionGqa` vectorization (`docs/done/vision-attention-vectorization-2026-08-20.md`) is used by
14 real vision-language encoders, but none of them had ever actually been run end-to-end on this
machine — every one only had preprocessor-only unit tests locally. Downloaded a correctly-paired
model (`InternVL3-2B`, matched to the local `mmproj-internvl3-2b-q8_0.gguf` via its source repo,
`mradermacher/InternVL3-2B-GGUF`) specifically to close that gap and actually run one.

**It found three real, previously-unknown bugs in a single attempt — none of them in the
`VisionOps` vectorization work itself.** Confirms the general lesson of the day: untested code
paths accumulate real bugs, and the only way to find them is to actually run the thing.

## Bug 1 — CLI hardcoded every non-Gemma4 architecture to `--image is only supported for Gemma 4 models`

`RunCommand.cs` gated `--image` entirely on `s_arch == "gemma4"`, despite `UnifiedVisionPipeline.
Open` already being a complete, metadata-driven dispatcher for 22+ architectures (reads `clip.
vision.projector_type` from the mmproj itself) and the actual embedding/injection code below it
being fully architecture-agnostic already. The gate was pure dead weight — never updated as the
other 21 encoders were built out.

**Fixed**: removed the gate. `UnifiedVisionPipeline.Open` already throws a clean
`NotSupportedException` for genuinely unrecognized mmproj files, so nothing lost in validation —
just the incorrect, overly narrow one. Wrapped the `Open` call in a try/catch for a clean CLI error
instead of an unhandled exception, matching the file's existing error-reporting style.

## Bug 2 — vision encoders silently reinterpret quantized tensors as raw F16

With bug 1 fixed, running InternVL3-2B for real crashed with:

```
System.AccessViolationException: Attempted to read or write protected memory.
   at OpenTail.Stingray.Vision.VisionOps+<>c__DisplayClass2_0.<MatVecF16>b__0(Int32)
```

Root cause, confirmed via `list-tensors` against the actual GGUF: `v.blk.0.attn_q.weight` and
friends are stored as **Q8_0** in this mmproj, but `InternVlVisionEncoder` unconditionally requests
them via `VisionOps.GetTensorPtr<Half>(...)`, then feeds the result to `MatVecF16`, which assumes a
contiguous 2-bytes-per-element F16 layout. Q8_0 packs ~1.06 bytes/element (34 bytes per 32-element
block), so the tensor's real allocation is roughly half what `MatVecF16`'s row-stride arithmetic
assumes — later rows walk off the end of the mapped region into unmapped memory.

This is not InternVL-specific: **every one of the 21 non-Gemma4 encoders** reads weights this same
way, and every locally-present mmproj for them except LLaVA and Pixtral is quantized (Q8_0), so all
of them would hit the identical corruption on their first real run.

**Fixed the failure mode, not the underlying gap**: added a dtype check to `VisionOps.
GetTensorPtr<T>` — when a tensor is found but its actual GGUF dtype doesn't match what `T` implies
(`Half`→`Float16`, `float`→`Float32`), it now throws a clear, immediate `NotSupportedException`
naming the tensor, the actual dtype, and the expected one, instead of corrupting memory minutes
later inside a `Parallel.For` lambda with no indication of which tensor or why. This is a genuine
safety improvement, not a full fix — quantized mmproj weights still don't work with any of these
encoders. That needs a dequant-aware MatVec path (mirroring `OpenTail.Stingray.Cpu.SimdKernels`'
approach for the main LLM engine), real follow-up work, not attempted here.

**Reference**: the real llama.cpp implementations these encoders were ported from live at
`examples/llama.cpp/llama.cpp/tools/mtmd/models` — `ggml`'s own kernels are dtype-generic, which is
why the reference never needed a guard like this; the per-dtype assumption is specific to this port.

## Bug 3 — `mm.1`/`mm.3` tensor names don't match this GGUF's actual naming

Separately (non-crashing, silent): `InternVlVisionEncoder` requests `mm.1.weight`/`mm.3.weight`,
but this GGUF names them `mm.model.mlp.1.weight`/`mm.model.mlp.3.weight`. `GetTensorPtr` returns
null on a name miss (by design, to support fallback candidate lists), so this silently no-ops
rather than crashing — worse in one sense (wrong output with no error) even though it's lower
severity than bug 2. Not fixed — same "real fix needs Q8_0 support anyway" reasoning; fixing the
name without fixing the dtype wouldn't make this model actually run.

## What this proves about the vectorization work

None of these three bugs are anywhere near `VisionOps.Attention`/`AttentionGqa`. The math-level
verification from the vectorization work (hand-computed reference tests, numerical agreement against
the retained pre-change scalar implementation) remains the right evidence for that change's
correctness — this investigation didn't touch it and found nothing wrong with it. It found bugs in
code around it that had simply never been exercised before.

## Files changed

- `src/OpenTail.Stingray.Cli/RunCommand.cs` — removed the gemma4-only `--image` gate, added a
  try/catch around `UnifiedVisionPipeline.Open` for clean CLI error reporting.
- `src/OpenTail.Stingray.Vision/VisionOps.cs` — `GetTensorPtr<T>` now verifies dtype and throws
  clearly on mismatch instead of returning a garbage-typed pointer.
- `models/InternVL3-2B.Q4_K_M.gguf` — downloaded (1.12GB, from `mradermacher/InternVL3-2B-GGUF`,
  correctly paired with the pre-existing local `mmproj-internvl3-2b-q8_0.gguf`). Still can't
  complete end-to-end until Q8_0 support exists (bug 2's real fix).
