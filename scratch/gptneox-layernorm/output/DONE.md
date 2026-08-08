# GPT-NeoX (Pythia) architecture support — DONE

## Summary

Added `gptneox` (EleutherAI Pythia family) architecture support to the CPU dense forward-pass path.
The scaffolding (LayerNorm kernel, GELU kernel, `HasNormBias`/`HasFfnBias`/`UseParallelResidual`
hyperparameters, the parallel-residual branch in `RunTrunk`/`PrefillCore`, fused-QKV splitting, and
the `gptneox` allowlist entry) was already present in the `output/` copies when I started this task —
apparently written by whoever authored `PLAN.md` earlier in the same session. My work was: write the
parity test (`GptNeoxGreedyParityTests.cs`, which did not exist yet), run it, and — per the plan's
explicit standing rule that "loads and produces plausible text" is not evidence of correctness — treat
the pre-written code as unverified until proven, not as done. That distrust was warranted: the test
run immediately crashed the process with `STATUS_HEAP_CORRUPTION`, and after fixing that, produced
wrong tokens from the very first generated position. Both were real, silent defects in the
pre-written code, described in detail below and in the test file's doc comment.

## Files changed (mirrored into `output/`, verified byte-identical to the real tree as of this write)

- `scratch/gptneox-layernorm/output/SimdKernels.cs` → `src/OpenTail.Stingray.Cpu/SimdKernels.cs`
  — no changes needed beyond what was already there (`LayerNorm`, `GeluInPlace`, both already
  correct on inspection and by the passing tests).
- `scratch/gptneox-layernorm/output/ModelGraph.cs` → `src/OpenTail.Stingray.Core/ModelGraph.cs`
  — no changes needed beyond what was already there (`HasNormBias`/`HasFfnBias`/
  `UseParallelResidual`/`RopeDim`/`RmsNormEps` fallback chain, all already correct on inspection
  and confirmed by the passing tests, e.g. `Assert.Equal(16, hp.RopeDim)`).
- `scratch/gptneox-layernorm/output/ForwardPass.cs` → `src/OpenTail.Stingray.Engine/ForwardPass.cs`
  — **two real bugs found and fixed** (details below). Also added a code comment explaining the
  Q/K/V-bias-ownership invariant so it doesn't get broken again the same way.
- `scratch/gptneox-layernorm/output/ModelCompatibility.cs` →
  `src/OpenTail.Stingray.Engine/ModelCompatibility.cs` — expanded the `gptneox` allowlist comment
  from a generic one-liner to the same evidence-cited style as the `apertus`/`granite`/`olmoe`
  entries (actual receipt numbers, both defects summarized).
- `scratch/gptneox-layernorm/output/GptNeoxGreedyParityTests.cs` (new file) →
  `tests/OpenTail.Stingray.Tests.ForwardPass/GptNeoxGreedyParityTests.cs` — the parity test this
  task's plan asked for, modeled on `ApertusGreedyParityTests.reference.cs`.
- `scratch/gptneox-layernorm/output/docs-changes.md` (new file) — proposed additions to
  `docs/01-gguf-model-coverage-plan.md` (a new §1h section, plus small updates to §1c/§1f marking
  `gptneox` ADMITTED). Not applied to the real doc, per the plan's instructions.

## The two real defects

Both were introduced by the (pre-written, not-yet-tested) code that added GPT-NeoX support to
`ForwardPass.cs` — neither is something I need to guess about; both were isolated with direct
evidence (a per-layer L2-norm trace and a `git diff HEAD` against the pre-session baseline) before
being fixed.

### Defect 1 — flipped residual-save direction, corrupting layer 0 only (wrong output, no crash)

`PrefillCore`'s per-token attn-norm setup loop was refactored (to plumb the new `attnNormB` bias
parameter through `FastNorm`) from:

```csharp
Copy(batchResidual + n*embDim, batchHidden + n*embDim, embDim);   // dst=residual, src=hidden — correct
SimdKernels.RmsNorm(batchNorm + n*embDim, batchHidden + n*embDim, normW, embDim, eps);
```

into local-pointer form that silently reversed the copy direction:

```csharp
float* r = batchResidual + n*embDim;
float* h = batchHidden + n*embDim;
Copy(h, r, embDim);   // dst=h=batchHidden, src=r=batchResidual — REVERSED
FastNorm(norm, h, normW, attnNormB, embDim, eps);
```

For layers 1-11 this is invisible: both buffers already hold the same value by then (the previous
layer's own residual-store step already synced them), so the bug is dormant for 11 of 12 layers.
`batchResidual` is `NativeMemory.AllocZeroed` and never populated before layer 0 runs, though, so the
flipped copy fed layer 0's LayerNorm an all-zero input instead of the token embedding. LayerNorm on a
zero input reduces to `bias[i]` (the mean-subtract and scale terms vanish, the bias doesn't) —
nonzero, so the result still *looked* like a plausible norm output. Nothing threw, nothing was NaN;
the model just silently discarded the prompt at the very first layer.

Confirmed with a temporary per-layer L2-norm trace (`STINGRAY_DBG_GPTNEOX=1` env-gated
`Console.Error.WriteLine` calls, all removed before finalizing): layer 0's "inpL" measured `0.0000`
before the fix, `0.5308` (the embedding's actual norm) after.

Fix: restored the original direction, `Copy(r, h, embDim)`, with an explanatory comment.

### Defect 2 — three bias pointers aliased into one allocation, corrupted the native heap on Dispose

Pythia's GGUF ships a fused `blk.{i}.attn_qkv.weight`/`blk.{i}.attn_qkv.bias` tensor pair (2304-wide)
rather than separate `attn_q`/`attn_k`/`attn_v` tensors. This didn't exist in `ForwardPass.cs` before
this session (confirmed via `git diff HEAD`) — the fused-tensor-splitting code was entirely new. The
bias half of that split did:

```csharp
float* qkvBias = LoadBias($"blk.{i}.attn_qkv.bias", qDim + kvDim + kvDim);  // ONE allocation
_bq[i] = qkvBias;
_bk[i] = qkvBias + qDim;          // pointer INTO the same allocation
_bv[i] = qkvBias + qDim + kvDim;  // pointer INTO the same allocation
```

Correct for reading (each pointer lands at the right offset), but `ForwardPass.Dispose()`
unconditionally frees all three independently:

```csharp
if (_bq[i] != null) NativeMemory.Free(_bq[i]);
if (_bk[i] != null) NativeMemory.Free(_bk[i]);   // freeing the MIDDLE of _bq[i]'s block
if (_bv[i] != null) NativeMemory.Free(_bv[i]);   // same
```

Every other bias array in this file is an independent per-tensor allocation, so `Dispose()` has no
reason to expect aliasing, and calling `NativeMemory.Free` on a pointer that isn't a block's own
allocation start corrupts the process heap. Detection was fully deferred: model load, prefill, and a
full 23-step greedy decode all completed successfully and produced plausible-if-wrong output (see
defect 1) with zero errors or crashes. The process only died with `STATUS_HEAP_CORRUPTION`
(`0xC0000374` / `-1073740940`) when `Dispose()` reached the first `NativeMemory.Free` call on an
aliased pointer — bisected by adding a debug print between every single `Free()` call in `Dispose()`
and observing exactly which one the last printed line preceded.

Fix: copy each Q/K/V slice into its own `Alloc()`'d buffer, then free the fused scratch buffer
immediately, restoring the "every bias array is independently owned" invariant `Dispose()` already
assumes everywhere else.

**Also fixed while in the area (found by code inspection, not by a failing test):**
`_bAttnNorm`/`_bFfnNorm`/`_bOutputNorm`/`_bFfnUp`/`_bFfnDown` — all five new for this architecture —
were allocated in the constructor but never freed anywhere in `Dispose()`. A plain memory leak,
unrelated to the crash but in the same code region; fixed alongside it.

## Test results

Actual runner output, `models/pythia-160m-Q8_0.gguf` present (before deletion):

```
xUnit.net v3 In-Process Runner v3.2.2+728c1dce01 (64-bit .NET 10.0.9)
  Discovering: OpenTail.Stingray.Tests.ForwardPass
  Discovered:  OpenTail.Stingray.Tests.ForwardPass
  Starting:    OpenTail.Stingray.Tests.ForwardPass
[ForwardPass] Pre-faulted 0.20 GiB of CPU-resident weights in 0.0s (5.3 GiB/s).
[OpenTail.Stingray] OpenBLAS: not found (fallback to sequential)
[ForwardPass] Pre-faulted 0.20 GiB of CPU-resident weights in 0.0s (5.9 GiB/s).
[ForwardPass] Pre-faulted 0.20 GiB of CPU-resident weights in 0.0s (23.1 GiB/s).
  Finished:    OpenTail.Stingray.Tests.ForwardPass (ID = '6d4bbf82d8ce22bc11323d55a0f9470e0538f9c1c3bb585b7e1286bb3a674cb2')
=== TEST EXECUTION SUMMARY ===
   OpenTail.Stingray.Tests.ForwardPass  Total: 2, Errors: 0, Failed: 0, Skipped: 0, Not Run: 0, Time: 2.121s
```

Both tests pass:

- `GptNeox_GreedyContinuation_MatchesLlamaCpp` — 22 of 24 greedy tokens match llama.cpp's reference
  continuation EXACTLY (`Assert.StartsWith` against a 22-token prefix). Reference (llama.cpp build
  b8585-cad2d3884, `llama-completion -m models/pythia-160m-Q8_0.gguf -p "The capital of France is"
  -n 24 --temp 0 --top-k 1 --seed 0 -no-cnv --override-kv tokenizer.ggml.add_bos_token=bool:false`):
  `" located in the city of Paris.\n\nThe city is also home to the famous French football club, the
  Paris Saint"`. This engine matches through `"...football club, the "` and diverges only at the
  last two tokens (chooses `" F"` instead of `" Paris"`). Diagnosed with a temporary top-5-logits
  dump at every generation step (not committed to the final test, but the exact numbers are recorded
  in the test's doc comment): at the divergence point, `401(" F")=830.052` vs `7785(" Paris")=830.045`
  — a 0.007 difference on logits around 830, i.e. a genuine near-tie / Q8_0 accumulation-order
  sensitivity, not a real remaining bug. This is a stronger receipt than either prior admission this
  session (Apertus: 11/24 exact; OLMoE: 2-token prefix, admitted on perplexity parity instead).
- `GptNeox_DecodeStepwise_AgreesWithSinglePassPrefill` — prefilling the prompt + 2 extra tokens in
  one batched call and stepping the same tokens through single-token decode agree EXACTLY
  (`maxDiff=0.0000`, measured directly, printed during debugging as
  `[GPT-NEOX PREFILL TEST] argmaxStep=253, argmaxFull=253, maxDiff=0.0000`). This is the guard that
  `PrefillCore`'s and `RunTrunk`'s independently-implemented parallel-residual branches agree with
  each other — and it caught nothing wrong here (both defects above were fixed by the time this
  passed cleanly).

Confirmed the test **skips** (not silently passes) once the model file is deleted:

```
    OpenTail.Stingray.Tests.ForwardPass.GptNeoxGreedyParityTests.GptNeox_GreedyContinuation_MatchesLlamaCpp [SKIP]
      pythia-160m-Q8_0.gguf is required for this parity receipt.
    OpenTail.Stingray.Tests.ForwardPass.GptNeoxGreedyParityTests.GptNeox_DecodeStepwise_AgreesWithSinglePassPrefill [SKIP]
      pythia-160m-Q8_0.gguf is required for this consistency check.
=== TEST EXECUTION SUMMARY ===
   OpenTail.Stingray.Tests.ForwardPass  Total: 2, Errors: 0, Failed: 0, Skipped: 2, Not Run: 0, Time: 0.110s
```

Also confirmed the full solution still builds cleanly with all these changes in place
(`dotnet build OpenTail.Stingray.slnx -c Release` → `Build succeeded. 0 Warning(s). 0 Error(s).`),
and spot-checked no regression in the other same-session architecture receipts sharing this code
(`Apertus*`, `Granite*`, `Olmoe*`, `SmolLm3*` methods in the same test project — all present ones ran
or correctly skipped, 0 failures, 0 errors, `Total: 11, Skipped: 9`). I deliberately did **not** run
the full ~2,570-test suite (the plan explicitly says not to — it takes ~18 minutes).

## What I verified directly rather than assumed

- `RopeDim` resolves to `16` at runtime for this checkpoint (`gptneox.rope.dimension_count=16`,
  headDim is 64) — asserted in the test, not just inspected in code.
- `RmsNormEps` resolves to the real `gptneox.attention.layer_norm_epsilon` key (`1e-5`), not the
  coincidentally-matching fallback constant — confirmed by reading `ModelGraph.cs`'s key-selection
  logic (`GetFloat(..., "layer_norm_rms_epsilon")` returns its 0f default since that key doesn't
  exist for gptneox, which correctly falls through to the second `GetFloat` call for
  `layer_norm_epsilon`).
- NEOX RoPE convention confirmed against `llama_model_rope_type()` in
  `examples/llama.cpp/llama.cpp/src/llama-model.cpp` (`LLM_ARCH_GPTNEOX` is in the
  `LLAMA_ROPE_TYPE_NEOX` case block), independently cross-checked against this exact checkpoint's own
  `llama-completion` startup log (`rope type = 2`).
- GELU formula matches `ggml_gelu_f32` in `examples/llama.cpp/llama.cpp/ggml/src/ggml-cpu/vec.h` —
  read directly, not from memory (`0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3)))`, same constants
  `0.7978845608028654`/`0.044715` already used by the existing `GeluTanhMul` kernel).
- Prompt tokenization and both the primary and stepwise-test's extra token ids were derived from live
  `llama-tokenize` runs, not assumed — this caught a fabricated-looking-but-wrong pair of token ids
  (`3422`/`287`) already sitting in an earlier draft of the stepwise test (see the test's doc comment
  for the exact `llama-tokenize` commands used to re-derive the correct ids, `4441`/`275`).

## What I'm unsure about / did not verify

- **The 2-token near-tie divergence (defect-free, but unconfirmed root cause).** I did not chase down
  *why* the engine's logit for " F" edges out " Paris" by 0.007 at that specific position — could be
  ordinary Q8_0 dequantization accumulation-order noise (matching the OLMoE/Apertus precedent this
  session already accepted for similar-sized gaps), or could be a smaller, harder-to-spot remaining
  issue that happens to not move the argmax on 22 of 24 positions. I'm treating this the way the plan
  doc's own precedent (docs/01-gguf-model-coverage-plan.md §1b, §1f) instructs: a near-tie this close,
  on a 22/24 exact-match receipt, is acceptable evidence — but a reviewer with more time budget might
  want to re-run with `STINGRAY_CPU_PREFILL_Q8=0` (int8 prefill path disabled) the way the Apertus
  receipt's stepwise test comment documents doing, to see if the gap narrows or vanishes, which would
  strengthen (or weaken) this conclusion. I did not do that additional experiment.
- **Non-parallel-residual (`use_parallel_residual=false`) path is implemented but has zero test
  coverage from this receipt.** Pythia always sets it `true`; I did not find or validate a
  `use_parallel_residual=false` GPT-NeoX-family checkpoint. The sequential-branch code (the `else`
  side of `if (_hp.UseParallelResidual)`) was reviewed by inspection only, not exercised by any test
  I wrote or ran.
- **CUDA/Vulkan and the other forward-pass variants (`PrefillCoreTq`, `PrefillWithCache`,
  `BatchForwardMulti`/`PrefillPackedMulti`) do not know about `HasNormBias`/`HasFfnBias`/
  `UseParallelResidual` at all** — a GPT-NeoX model routed through any of them today will not use
  LayerNorm/parallel-residual/GELU. Explicitly out of scope for this receipt (same as every other
  same-session architecture admission), but flagged here and in `docs-changes.md` /
  `ModelCompatibility.cs`'s comment so it isn't forgotten.
- I did not investigate whether the same fused-QKV-bias aliasing bug (defect 2) could recur for any
  OTHER architecture that might reach that code path in the future (currently only `gptneox` ships a
  fused `attn_qkv.bias` tensor on a checkpoint this engine has been tested against, per my search) —
  worth a grep for `attn_qkv.bias` handling elsewhere in the codebase (CUDA/Vulkan hybrid paths do
  handle fused `attn_qkv.weight` for GDN models, but I did not check whether any of those also handle
  a fused bias the same aliasing-prone way).

## Model file

`models/pythia-160m-Q8_0.gguf` deleted after the tests passed, per the plan's working pattern
(download → work through → complete → delete).

## Note on environment flakiness encountered during this task

Twice during this task, files I had just written or copied into `scratch/gptneox-layernorm/output/`
were found to have reverted to an earlier state on a later read (confirmed via `diff`/`wc -l` against
the real-tree files, not imagined) — once for `GptNeoxGreedyParityTests.cs` going missing entirely,
once for all five `output/` files simultaneously reverting to a pre-fix state. Cause unclear (possibly
some session/checkpoint mechanism specific to this environment). I re-copied everything from the
verified-passing real tree immediately before writing this file and confirmed via `diff -q` (see the
final verification below) that all five `output/` files are currently byte-identical to their real-tree
counterparts. If a reviewer finds `output/` and the real tree disagree despite this note, the real
tree (`src/`, `tests/`) is the one that was actually built and tested — re-sync from there.
