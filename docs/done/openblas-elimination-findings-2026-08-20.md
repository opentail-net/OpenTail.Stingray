# OpenBLAS: fire it from a canon into space

**Measured:** 2026-08-20, Ryzen 7 5700G, CPU backend, SmolLM2-1.7B-Instruct-Q4_K_M, 518-token
prompt, 24 tokens generated, greedy, interleaved runs against a known-good prior working copy of
this repo (`priorworkingstate/Opentail.Stingray`).

## TL;DR

OpenBLAS provides **zero measured benefit anywhere in this codebase's current call paths** and
costs real prefill throughput just by being present on disk. Two real ordering bugs let it steal
priority from kernels that beat it on every shape tested. Once those were fixed, a small gap
*still* remained — and the only thing that closed it was removing `libopenblas.dll` from the
source tree entirely and clean-rebuilding. Not "renaming it away at runtime." Not "letting
`BlasInterop.IsAvailable` come back false." Actually not being there.

**Recommendation: remove it.** Delete `tools/openblas/`, the `Content Include` block that ships
it, `BlasInterop.cs`, and every BLAS-path branch in `SimdKernels.cs`. Nothing in this
investigation found a shape, dtype, or batch size where it wins. `docs/reference/
openblas-troubleshooting-guide.md` should be deleted alongside it — it documents how to *restore*
the thing this doc recommends launching into low Earth orbit.

## The numbers

| Configuration | Prefill t/s (best-of-5, interleaved) |
|---|---:|
| Prior working copy (no OpenBLAS anywhere, ever) | 150.3 – 157.5 |
| Current, unfixed (OpenBLAS DLL present, both ordering bugs live) | ~74 |
| Current, fix #1 only (repack-before-BLAS in `MatMulBatchedCached`) | ~120 – 133 |
| Current, fix #1 + fix #2 (Q8/MicroGemm-before-BLAS in `SimdKernels.MatMulBatched`) | ~138 – 143 |
| Current, both fixes + OpenBLAS **removed from the source tree**, clean rebuild | 150.7 – 157.1 |

The last row fully overlaps the prior working copy's range. There is no residual gap once BLAS is
actually gone. Decode throughput was flat (~20-25 t/s) across every configuration in this table —
none of this ever touched the decode path, only prefill.

## What OpenBLAS was actually doing

`BlasInterop.cs` binds `cblas_sgemm`. `SimdKernels.MatMulBatched` and
`ForwardPass.MatMulBatchedCached` both had a structurally identical bug: **BLAS availability was
checked first**, and only once BLAS was ruled out did the code try the kernels that actually beat
it — the repacked Q4_K×8 kernel and the Q8-activation-quantized path (`TryMatMulBatchedQ8`,
independently measured at +47% over the pre-Q8 baseline elsewhere in this codebase's own perf
log). Merely having `libopenblas.dll` sitting in `tools/openblas/` was enough to flip
`BlasInterop.IsAvailable` to true and silently route every batch above `MinBatchForBlas` (16) into
`cblas_sgemm`, ahead of kernels the codebase had already proven were faster.

Why does OpenBLAS lose so badly on this workload? It requires weights in F32. Every quantized
weight matrix (Q4_K, Q6_K, ...) has to be fully dequantized into a scratch F32 buffer — a complete
extra read-and-write pass over the entire tensor — *before* `cblas_sgemm` ever runs, and this
dequant is redone from scratch on every single call; nothing about it is cached across calls in a
way that amortizes. Prefill on this hardware is bandwidth-bound, not compute-bound (see
`docs/cpu-performance-baseline.md`), so a mandatory extra full-tensor memory pass is exactly the
wrong trade. The specialized kernels dequantize in registers, in the same pass as the dot product,
with no scratch buffer and no second read of the weight matrix. BLAS cannot win a bandwidth-bound
race it enters with a bandwidth handicap.

## The two ordering bugs, and the one that wasn't a bug

1. **`ForwardPass.MatMulBatchedCached`** tried the OpenBLAS dequant-cache path before the repacked
   Q4_K×8 kernel. Fixed by trying repack first. Recovered most of the loss (~74 → ~120-133 t/s).
2. **`SimdKernels.MatMulBatched`** had the identical shape of bug one level deeper: when BLAS was
   available and the batch was large enough, it skipped `TryMatMulBatchedQ8`/`MicroGemmQ4K`
   entirely. This mattered specifically for **Q6_K tensors** (`attn_v`, `ffn_down` in this
   Q4_K_M model — llama.cpp's mixed-precision scheme deliberately keeps those two at higher
   precision), which can never take the Q4_K-only repack path from fix #1 and so fell straight
   into this second copy of the same bug. Fixed the same way: try the fast paths first. Recovered
   the rest of most of the gap (~120-133 → ~138-143 t/s).

Both fixes are real and correct, and stay in regardless of what happens to OpenBLAS itself — they
also protect against a hypothetical future where these kernels *don't* uniformly win. But they did
not fully explain the regression. After both were shipped, current was still ~7-12% behind the
prior working copy, consistently, across five separate interleaved re-measurements including a
from-scratch clean rebuild. That residual is the part that actually hurt to find.

## The pain of chasing the residual

This is the part worth writing down so nobody re-walks it. In order, what got checked and ruled
out chasing that last ~10%, all by direct diff or direct measurement, none of it by assumption:

- **The hot kernel itself** (`DotQ4Kx8_Q8KS_8In`) — byte-identical between current and prior.
- **`CanRepackQ4Kx8`/`RepackQ4KMatrix`/`Q4Kx8PackedBytes`** — byte-identical.
- **`GetRepackedQ4Kx8`** — byte-identical.
- **`FastRmsNorm`/`FastNorm`/`FusedMatVec`** (`ForwardPass.Helpers.cs`) — byte-identical.
- **`PrefillCoreAttention`** — byte-identical (a claimed "documented tiling improvement" in
  `perf-loop-progress.md` turned out to be a comment/doc rewrite only; the actual code never
  changed between these two snapshots — a real false lead that got asserted before it was verified
  against the diff, and rightly called out as such).
- **Thread pool config** (`MaxDegreeOfParallelism` resolution) — identical.
- **`Directory.Build.props`, `stingray.runtimeconfig.json`** (Server GC, `QuickJitForLoops=false`,
  IlcOptimizationPreference, etc.) — identical.
- **Weight prefault behavior** — identical log line, identical timing, in both trees.
- **Linked assemblies** — current links exactly one extra DLL (`Audio`) versus prior; not eagerly
  loaded by a text-inference run, ruled out as a real factor.
- **Stale build artifacts** — ruled out with a full `dotnet clean` + rebuild on both trees; gap
  persisted identically, so it wasn't incremental-build skew.
- **The `PrefillCore` orchestration method itself** — tested by literally transplanting current's
  exact `ForwardPass.PrefillCore.cs` method body into the prior tree (with trivial `_isMla`/
  `IsMoeLayer` compatibility stubs so it would compile against prior's non-MLA class shape) and
  benchmarking the splice. It ran at the *prior* tree's speed, not current's — proof by direct
  swap, not inference, that the orchestration logic was never the cause.

Every one of those came back clean, which is the frustrating part: normally a performance
regression bisection finds *something* different in the code. Here, everything on the hot path
that could be diffed was provably identical, and the gap was still real and reproducible to within
a couple of percent across every re-measurement. That combination — identical code, different
speed — pointed at something outside the source files: JIT/codegen nondeterminism, memory-layout
effects from the loader, or (as it turned out) something tied to OpenBLAS that neither the
ordering fixes nor a mere runtime DLL removal fully undid.

The earlier attempt to rule out BLAS specifically (moving `libopenblas.dll` out of the *build
output* directory, without a clean rebuild, then moving it back) had shown current *still* slower
than prior even with BLAS unavailable at runtime — which is what sent the investigation down the
Engine-layer path for several turns, chasing MLA-driven restructuring and extra assemblies that
turned out not to matter. Only removing the DLL from the **source tree** and doing a **clean
rebuild** — so nothing about OpenBLAS was in the picture at any layer, not the binary, not the
build output, not whatever residual state a same-process runtime removal leaves behind — actually
closed the gap. The mechanism for why a runtime-only removal wasn't sufficient is not fully
understood; the empirical result is unambiguous regardless.

## Why "fired from a canon into space"

Every test in this investigation that put OpenBLAS anywhere in the loop — available, unavailable,
ordered first, ordered last — cost measurable prefill throughput somewhere, and the one test that
didn't was the one where it plain wasn't there. It doesn't win on any shape, dtype, or batch size
this session measured. It doesn't win even after being deprioritized correctly. Its mere presence
on disk was enough to derail several turns of investigation into false leads (MLA restructuring,
linked assemblies, JIT codegen) before the actual culprit was re-confirmed. Renaming the DLL is
not enough — it needs to not exist in this repository. Low orbit is the minimum safe distance.
