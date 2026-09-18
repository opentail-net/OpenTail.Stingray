# 090 — FLUX.2 (and sibling diffusion models) CPU performance handoff

**Audience**: an AI agent picking this up fresh, with no memory of the investigation that produced
this document. Read this whole file before touching code. Target platform is **CPU only** — this
machine has no discrete GPU (Ryzen 5700G iGPU only), and FLUX.2 has zero Vulkan/GPU-residency
wiring yet regardless, so do not go looking for a GPU angle here.

## The problem, stated plainly

A single real-weight FLUX.2 generation (18GB Q4_K_S DiT + 24B-parameter Mistral text encoder, both
loaded from GGUF) is currently **unusably slow** — a 20-step run at the model's real production
resolution (512×512; see "Why 512×512, not smaller" below) was killed after 30+ minutes without
finishing. This is not a tuning problem, it is a **fundamentally wrong hot-path architecture**
problem, and it is not unique to FLUX.2 — the same anti-pattern exists in every diffusion model in
this codebase (`QwenImageModel.cs`, `HunyuanVideoModel.cs`, `WanModel.cs`, `LtxVideoModel.cs` all
have a `GetWeight` method that looks structurally identical to the one described below — verify
this claim yourself before assuming it, but it was true as of this doc's writing).

## Root cause (confirmed, not guessed)

`Flux2DiT.cs`'s `GetWeight` method:

```csharp
private float[] GetWeight(string name) => _weights!.ReadF32(Resolve(name));
```

calls into `GgufWeightLoader.ReadF32`:

```csharp
public float[] ReadF32(string name)
{
    if (_cache.TryGetValue(name, out var cached)) return cached;
    var resolved = Resolve(name) ?? throw new KeyNotFoundException(...);
    var info = _model.FindTensor(resolved)!.Value;
    long count = info.ElementCount;
    var raw = _model.GetTensorData(info);
    var result = new float[count];
    Dequantize.ToFloat32(raw, result, info.DType, count);
    // Only cache small tensors (<=1M elements) -- weight matrices are NOT cached
    if (count <= 1_000_000) _cache[name] = result;
    return result;
}
```

**Every large weight matrix is fully re-dequantized from raw Q4_K_S bytes to a freshly-allocated
`float[]`, on every single call, with zero caching**, because the `<=1_000_000` element cache guard
deliberately excludes exactly the tensors that dominate cost. FLUX.2's biggest single-block linear
is `6144×55296 ≈ 340M elements` — one such tensor, dequantized to FP32, is ~1.36GB, and this
happens **once per call**, and `LinearNoBias` is called for every projection in every one of 8
double-stream + 48 single-stream blocks, for every one of ~20 denoising steps. That is on the order
of a thousand+ full-tensor dequantizations per generation, each touching hundreds of MB to low-GB
of data, with the resulting array discarded immediately after one matmul and garbage-collected.

This no-cache decision was **deliberate**, not an oversight — a previous version of this exact
pattern (in `QwenImageModel.cs`) cached everything unboundedly and grew past 53GB resident memory,
getting the process OS-killed. The fix at the time was "cache nothing large," which avoids the OOM
but reintroduces catastrophic redundant work. **The right fix is a bounded cache, not no cache at
all** — see below, because this bounded-cache infrastructure already exists elsewhere in this
codebase and just needs to be reused.

## The good news: this codebase already solved this problem, just not here

The main LLM text-generation engine (`OpenTail.Stingray.Engine/ForwardPass.cs`) faces the exact
same problem (large quantized weight matrices, CPU-only, needs speed) and has **already built and
proven** both pieces of the fix:

### 1. A native Q4_K quantized-domain CPU matmul kernel — avoids dequantizing to FP32 at all

`OpenTail.Stingray.Cpu/SimdKernels.cs`'s `TryMatMulBatchedQ4Kx8`, backed by
`RepackedGemm.cs`/`RepackedGemmPath2.cs`/`MicroGemmQ4K.cs`, computes the matmul directly against a
**repacked** (not float-expanded) representation of the Q4_K weight data. `ForwardPass.cs`'s own
comment (around line 1428) records this measured **2.6x faster** than the row-major dequant+BLAS
path for a real checkpoint (SmolLM2-1.7B-Q4_K_M). This is exactly "Stage 3" of the investigation
that led to this document — it does not need to be built, it needs to be **reused**.

### 2. A bounded, budgeted weight-repacking cache — NOT an LRU, NOT unbounded

`ForwardPass.cs`'s `GetRepackedQ4Kx8` (see the class fields around line 70-105) repacks tensors
into the Q4Kx8 layout **once** and caches the repacked bytes, but only up to an explicit byte
budget:

```csharp
private readonly long _q4kx8CacheBudgetBytes = ResolveQ4Kx8CacheBudget();

private static long ResolveQ4Kx8CacheBudget()
{
    string? raw = Environment.GetEnvironmentVariable("STINGRAY_Q4KX8_CACHE_MB");
    if (long.TryParse(raw, out long mb)) return mb > 0 ? mb * 1024 * 1024 : 0;
    long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
    return available > 0 ? available / 4 : 0;
}
```

When the budget is exhausted, `GetRepackedQ4Kx8` simply declines to cache further tensors and the
caller falls back to the row-major/dequant path for those — **partial caching degrades speed, never
correctness**. This is precisely the "bounded budget, explicit dictionary, not LRU" architecture
that is the right approach for this access pattern (working set ≈ the whole model; LRU thrashes
because there is no temporal locality smaller than the cache — every tensor is touched exactly once
per step, in the same order, forever).

**Do not reinvent this. Do not build an LRU. Do not build a new quantized matmul kernel.** The task
is almost entirely **integration**: get the diffusion models to call the same infrastructure the
LLM engine already calls, instead of their own parallel, naive `ReadF32`-every-time path.

## What's actually different between the two code paths today

| | LLM path (`ForwardPass.cs`) | Diffusion path (`Flux2DiT.cs` etc.) |
|---|---|---|
| Weight access | `SimdKernels.TryMatMulBatchedQ4Kx8` direct on quantized bytes | `IWeightLoader.ReadF32` → full FP32 float[] |
| Caching | Bounded, budgeted, repacked-Q4Kx8 bytes | None for tensors >1M elements |
| Raw pointer access | Uses `TryGetRaw`-style raw mmap pointers | Never calls `IWeightLoader.TryGetRaw` at all |
| Fallback when uncached | Row-major dequant + BLAS (still faster than diffusion's path) | The naive path IS the only path |

`IWeightLoader.TryGetRaw` (see `src/OpenTail.Stingray.Core/IWeightLoader.cs`) already exists
specifically to give callers direct access to the mmap'd raw quantized bytes without dequantizing —
`ZImageDiT.cs` uses it (see `MatQ` method, ~line 470) as another real in-repo example, though that
one is GPU-focused (fp8 upload path); the CPU angle you want is `ForwardPass.cs`'s usage, not
`ZImageDiT.cs`'s.

## Your task, in priority order

1. **Read `ForwardPass.cs`'s weight-dispatch logic end to end** (search for `TryMatMulBatchedQ4Kx8`,
   `GetRepackedQ4Kx8`, `_q4kx8CacheBudgetBytes`, `GetDequantWeightF32`, `_dequantCacheEnabled`) to
   fully understand the existing budgeted-cache + quantized-matmul architecture before writing any
   new code. This is the design you're porting, not designing from scratch.

2. **Decide whether to extract shared infrastructure or duplicate the pattern.** The cleanest
   answer is probably a new shared component (e.g. in `OpenTail.Stingray.Cpu` or a new file) that
   both `ForwardPass` and the diffusion models' `DiffusionOps.Linear`-calling code can use — a
   `IWeightLoader`-scoped, budget-bounded `QuantizedWeightCache` that owns the repacking and exposes
   something like `bool TryMatMul(string tensorName, ReadOnlySpan<float> x, Span<float> output, int n, int rows, int cols)`,
   falling back to the caller's existing dequant path when it declines. Don't over-engineer this on
   the first pass — a direct, unglamorous port that gets FLUX.2 working is more valuable right now
   than a beautiful abstraction that touches five files and risks breaking the already-working LLM
   path. **Do not modify `ForwardPass.cs`'s own behavior/tests as a side effect** — if you extract
   shared code, verify the LLM test suite (`OpenTail.Stingray.Tests.ForwardPass.Fast` at minimum,
   `STINGRAY_RUN_HEAVY_TESTS=1` heavy suite if you touch shared internals) still passes.

3. **Wire `Flux2DiT.cs`'s `LinearNoBias`/`GetWeight` to the new path first** (it's the worst-affected
   model: 18GB DiT + 24B text encoder together, and the one this investigation was already focused
   on). Keep the existing `ReadF32` path as the fallback for small (<=1M element) tensors — that
   part of the current code is fine and doesn't need to change.

4. **Verify correctness is unchanged** before touching performance further. This session found and
   fixed two real correctness bugs in FLUX.2 (missing `t * 1000` timestep pre-scaling, which made
   the DiT timestep-blind; see `docs/087-flux2-implementation-plan.md` and
   `docs/088-diffusion-two-pass-quality-and-performance-master-plan.md` for full detail) — after
   that fix, real output changed from pure noise to a structured periodic grid/tiling pattern (not
   yet fully coherent — a separate, still-open structural bug, likely patchify/token-ordering
   related, is suspected next). **Your perf work must not regress this.** Re-run
   `tests/OpenTail.Stingray.Tests.Diffusion/Flux2EndToEndRealWeightsTests.cs` after your change and
   confirm the output is still finite and non-degenerate (bit-exact match to pre-change output is
   NOT required or expected — a quantized-domain matmul kernel has different rounding than a dense
   FP32 dot product — but gross sanity: finite, non-NaN, not all-zero, and ideally visually similar
   in character to the pre-change grid-pattern sample, not a regression back to pure noise).

5. **Measure, don't assume** (CLAUDE.md rule 7's spirit applies here too): once wired, run the
   already-written `tests/OpenTail.Stingray.Tests.Diffusion/Flux2PerfTraceSingleForwardTests.cs`
   (env var `STINGRAY_FLUX2_PERF_TRACE=1`) — this test currently **fails/hangs** even for a single
   forward pass at 512×512 (killed after 10+ minutes with no output, exit code 1, likely an OOM from
   the current unbounded-allocation pattern under investigation). If your fix is working, this test
   should complete in well under a minute and print a real dequant-vs-matmul time split via
   `Flux2DiT.PerfTrace.Report`. If it still hangs/OOMs after your change, that itself is a critical,
   reportable finding — don't paper over it.

6. **Once a single forward pass is fast, re-attempt a real end-to-end generation**
   (`Flux2Real512ResolutionCheckTests.cs`, 512×512/20-step) and record the real wall-clock time in
   `PerformanceLeague.md` and `docs/088`, following this project's existing documentation
   conventions (see any other model's row in those files for the expected format/detail level).

## Propagating the fix to sibling models (do this AFTER FLUX.2 works, not before)

`QwenImageModel.cs`, `HunyuanVideoModel.cs`, `WanModel.cs`, and `LtxVideoModel.cs` likely have the
identical `GetWeight`/no-large-tensor-cache pattern (verify each one directly — do not assume).
Once the shared component from step 2 above exists and is proven correct+fast on FLUX.2, wiring the
others should be a much smaller, mechanical change per model. Do NOT attempt to fix all five at
once — land FLUX.2 first, verify it end-to-end, then propagate one model at a time with its own
correctness re-verification (each of these models has its own real-weight test suite already; use
it).

## Constraints to respect

- **Total system RAM is 63GB.** The Mistral-24B text encoder alone is ~12.6GB once dequantized
  (confirmed via a real log line: `[ForwardPass] Pre-faulted 12.61 GiB of CPU-resident weights`).
  Any new cache for the FLUX.2 DiT's own ~18GB (Q4_K_S on disk) must coexist with that, plus
  activation/workspace memory, plus OS/runtime overhead. **A full-FP32-residency cache for the
  whole 18GB Q4_K_S DiT would be roughly 18GB × ~7 ≈ 126GB dequantized — this does NOT fit and must
  not be attempted.** This is exactly why the quantized-domain matmul kernel (no FP32 expansion at
  all) is the real fix, not merely "cache more FP32 tensors."
- **Do not reintroduce the OOM bug.** Any cache must have an explicit, measured byte budget (follow
  `ForwardPass.cs`'s `ResolveQ4Kx8CacheBudget` pattern: auto-size from
  `GC.GetGCMemoryInfo().TotalAvailableMemoryBytes`, allow an env var override, decline gracefully
  when the budget is exhausted rather than growing past it).
- **CPU only.** No Vulkan/CUDA work for FLUX.2 belongs in this task — that's a separate, larger,
  already-identified follow-up item (see `docs/088`'s Pass 2 section) that explicitly should not
  start until correctness AND CPU performance are both in a reasonable state.
- Follow this project's standing rules in `CLAUDE.md`, especially rule 7 (measure, don't assume —
  keep a change only if it's measurably better, write the numbers down) and rule 12 (a fast "passed"
  test may be a silent no-op — check wall-clock time and real weight-loading log lines, not just
  pass/fail).

## Why 512×512, not a smaller test resolution

The real FLUX.2 reference (`examples/flux2/src/flux2/sampling.py`'s `limit_pixels = 1024**2`)
expects roughly 1024×1024-scale generation. Every real end-to-end test run in this investigation so
far has been at 64×64 or 128×128 — 64x to 256x smaller in pixel area than the model's expected
regime, which this session found produces a real, structured-but-different artifact (see
`docs/087`/`docs/088` for the full "resolution sensitivity" finding, which parallels a similar,
already-confirmed effect found for LTX-Video this same session). 512×512 is the smallest
resolution this investigation considers meaningful for a real coherence judgment, which is also why
it's expensive enough to have triggered this whole performance investigation in the first place —
don't "fix" the timing problem by quietly testing at a smaller, unrepresentative resolution.
