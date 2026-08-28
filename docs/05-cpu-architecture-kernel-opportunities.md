> **Reprioritized 2026-08-08 — now last on the local runway.** Everything here is performance;
> none of it unlocks a model, and the goal now ranks model coverage above speed.
>
> **Item 3 is superseded in part.** Native kernels for IQ4_NL, MXFP4 and other scalar-fallback
> formats are a *follow-up* to §2 of [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md),
> which first has to make the unimplemented IQ formats dequantize at all. Correctness admits the
> model; kernels only make it faster.

# CPU architecture coverage programme

**Status:** active backlog; the Q4_K repacked-GEMM investigation, Flash64 reference case, and
the missing Flash 128/256 correctness route are closed. Performance acceptance for the new head
widths remains open.

## Ordered work

1. Measure Flash attention at 128/256 head widths against the materialised fallback, with
   interleaved isolated and real-model samples. The generic 64-query/KV tile now dispatches for
   dense 64/128/256 heads; the 64-wide special case remains on the hardcoded GEMM and 128/256 use
   the strided AVX2 microkernel. Qwen3-8B (headDim 128) Flash-vs-fallback parity passes; the 256
   GEMM shapes have an independent oracle, but no local dense hd256 model receipt exists yet.
2. Q6_K AVX2 performance-only investigation. Dispatch is complete: Q8-prefill resolves Q6_K to
   Q8_K activation plus 8/4/1-input dots, and F32 multi-input batching uses the 4/2-input paths.
   Focused equivalence coverage passes; any change needs an interleaved end-to-end win.
3. ~~Native kernels for IQ4_NL, MXFP4, and other genuinely scalar-fallback formats.~~ IQ4_NL/MXFP4
   already had fused routes; IQ4_XS/IQ2_XS/IQ2_S/IQ3_XXS got AVX2 kernels 2026-08-28 (~18% overall
   win on the Qwen3.8-27B receipt below — see that section for what's still open: GDN
   parallelization looks like the bigger remaining lever). Q4_0 already has a fused CPU route, so
   it was never an implementation gap. IQ1_S/IQ1_M/TQ1_0/TQ2_0 remain unimplemented at any level
   (see [01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §2).
4. Batched prefill for per-layer head dimensions and CPU MoE.
5. ARM64 NEON, dot-product, and i8mm coverage (external hardware required).

Every performance item requires dispatch proof, isolated control/candidate samples, named-model
end-to-end measurement, and numerical validation. No single-run result is sufficient.

Historical evidence: [done/cpu-architecture-kernel-opportunities-2026-08.md](done/cpu-architecture-kernel-opportunities-2026-08.md).

## Measurements — Ministral-8B-Instruct-2410 vs. llama.cpp (2026-08-28)

Collected incidentally while running the Ministral-8B greedy-parity receipt (see
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) / `ModelCompatibility.cs`'s
`mistral3`/`ministral` entries) — not itself acceptance evidence for any item above, just a data
point for when this list is picked back up. Prompt `"The capital of France is"`, `-n 64`,
`--temp 0 --repeat-penalty 1.0`, Q4_K_M, 36L/4096d/headDim=128, this machine (12-core AVX2, AMD
Radeon integrated GPU). 3 runs each side, CPU-only backend both sides (reference `llama-cli`/
`llama-completion` build has no GPU backend compiled in):

| Backend | Prefill (t/s) | Decode (t/s) |
|---|---|---|
| llama.cpp (CPU-only, build b10532-70aff2525) | 23.3 avg (21.4–25.2) | 9.25 avg |
| Stingray, `--device none` (CPU-only) | 13.1 avg (12.4–13.5) | 7.27 avg |
| Stingray, auto backend (picked CPU this run) | 13.2 avg | 6.67 avg |

Stingray trails by **~1.8x on prefill** and **~1.3x on decode** on this checkpoint/machine. Prefill
is the larger and more likely tractable gap (batched-GEMM kernel quality — relevant to item 4,
batched prefill), decode is memory-bandwidth-bound single-token generation and likely needs
sustained kernel work to close meaningfully.

Also observed, unexplained: Stingray's auto-selected GPU-hybrid path was not faster than its own
CPU-only path for this small model/prompt (an earlier uncontrolled run showed hybrid prefill at
5.7 t/s, *slower* than CPU-only) — plausibly PCIe upload/readback overhead dominating a short
8-token prompt on the integrated GPU, not investigated further. Worth a look whenever GPU-hybrid
dispatch heuristics are next touched.

## Qwen3.8-27B (`qwen35` hybrid GDN) — profiling, fix, and measurement (2026-08-28)

Follow-up to the Ministral-8B measurement above, on a much larger checkpoint that exposed a real
~15-20x gap (not the ~1.3-1.8x seen on Ministral-8B) — large enough to actually investigate rather
than just log. Same checkpoint as the `qwen35` receipt in `ModelCompatibility.cs` (Qwen3.8-27B
UD-Q3_K_XL, Unsloth Dynamic quant mixing `IQ2_S`/`IQ2_XS`/`IQ3_XXS`/`IQ4_XS` per-tensor — see
[01-gguf-model-coverage-plan.md](01-gguf-model-coverage-plan.md) §2). All numbers below: prompt
`"The capital of France is"`, raw completion (`STINGRAY_RAW_PROMPT=1`), `--temp 0
--repeat-penalty 1.0`, this machine, CPU-only (this model has no dense_moe experts; `IsMoE` is
false — it uses plain per-layer `ffn_gate/up/down.weight`, not `_exps` tensors, so every layer's
FFN is full dense compute, not sparse top-K — a large part of why this checkpoint is fundamentally
heavier than Ministral-8B, independent of any kernel gap).

**Profiling** (`STINGRAY_PROFILE_DECODE=1`, new temporary instrumentation in
`HybridGdnForwardPass.ReportGdnProfile` — see the TEMPORARY-diagnostic comment at that class's top
for why it isn't merged into `DecodeProfileTimers`): a real decode run split as **74-76% MoE/FFN
block, 18-20% GDN recurrence, 5% attention**. This ruled out the initial hypothesis (GDN
recurrence's scalar/single-threaded per-head state loop, `GdnKernels.GdnStepInternal` — real, but
secondary) and pointed at the dense FFN matvecs, which were entirely on
`SimdKernels.MatVecDequantFallback` (materialize each IQ-quantized row to F32, then a separate
`DotF32` pass) because `IsSupportedWeightDType` had never had cases added for these formats when
they were admitted (see §2's parity-receipt entry).

**Fix — fused `Q8_K`-paired kernels for all four formats** (`SimdKernels.cs`,
`MatVecIq{4Xs,2Xs,2S,3Xxs}` + `DotIq{...}_Q8K`), ported from ggml's real reference formulas
(`ggml_vec_dot_iq{4_xs,2_xs,2_s,3_xxs}_q8_K_generic`, ggml-cpu/quants.c) — all four declare
`vec_dot_type=Q8_K` in ggml's own dispatch table (`ggml-cpu.c`), reusing this file's existing
`QuantizeRowToQ8K`/`Q8KScratchBytes` (built for `Q6_K`) rather than `IQ4_NL`'s `Q8_0` pairing.
**First attempt was scalar-only and measured a ~0% improvement** — the fallback it replaced already
ended in a 4-way-unrolled AVX/FMA `DotF32`, so removing the redundant F32-materialize pass without
adding vectorization roughly broke even. **AVX2 versions** (`DotIq{...}_Q8K_Avx2`) then ported the
real win: for `IQ4_XS` (16-entry codebook) this is a direct `Ssse3.Shuffle`-based table lookup,
matching ggml's x86 kernel almost 1:1. For `IQ2_XS`/`IQ2_S`/`IQ3_XXS` (512/1024/256-entry grids,
too large for a shuffle table), ggml's own AVX2 kernel uses a memory-aliasing "store an index
vector, reload as scalar" gather trick judged too risky to port faithfully into managed code within
this session — instead the unavoidable per-index grid+sign lookups stay scalar (gathered into a
32-byte signed buffer matching Q8_K's element order), and only the sign/multiply/reduce step is
vectorized with the same abs/sign `maddubs` trick as `IQ4_XS` and the existing `DotQ8_0_Q8_0_Avx2`.
Every kernel has a scalar fallback (`_Scalar` suffix) used automatically off AVX2 hardware and as
the correctness oracle.

**Measured result** (all runs `-n 24`, greedy output byte-identical to the llama.cpp receipt at
every step — verified after each change, not just at the end):

| Stage | ms/token (blended prefill+decode) | vs. baseline |
|---|---|---|
| Baseline (scalar `MatVecDequantFallback` for all 4 formats) | 6850 | — |
| + `IQ4_XS` dedicated kernel, scalar only | 7017 | ~0% (net wash, as predicted above) |
| + `IQ4_XS` AVX2 | 5805 | **~15% faster** |
| + `IQ2_XS`/`IQ2_S`/`IQ3_XXS` AVX2 too (all 4 formats) | 5615 | **~18% faster** |

The MoE/FFN profiling bucket itself went from ~5096ms/token to ~4264ms/token (~16% faster),
consistent with the overall number. The three formats added after `IQ4_XS` contributed only ~3
more percentage points beyond it alone — their larger grids mean more of their per-element cost is
in the still-scalar gather step, which the vectorization doesn't touch.

**~10-12x still remains unexplained** (llama.cpp reference: ~494ms/token decode on this same
checkpoint/machine, CPU-only, vs. Stingray's ~5600ms/token after this fix). Candidates for the next
session, in likely-impact order:
1. **GDN recurrence parallelization** — `GdnKernels.GdnStepInternal`'s per-head loop
   (`for (int h = 0; h < hv; h++)`) is plain scalar, single-threaded, no `Parallel.For`/SIMD, and is
   still ~19% of decode time on its own after the FFN fix. Heads are trivially independent.
2. **Per-call `Parallel.For` dispatch overhead** — 64 sequential layers × several matvec calls each
   × single-token (batch=1) decode means many small, short-lived parallel dispatches per token;
   never measured directly, but plausible given the FFN fix's real-but-modest yield relative to the
   size of the remaining gap.
3. Whether `MinRowsForParallel`'s threshold is well-tuned for this model's row counts (5120/17408).
4. A closer AVX2 port of ggml's actual `IQ2_XS`/`IQ2_S`/`IQ3_XXS` gather trick, if the scalar-gather
   step turns out to be the bottleneck for those three specifically once (1)-(3) are ruled out.

`TEMPORARY` instrumentation left in place for the next session: `STINGRAY_PROFILE_DECODE=1` now
also prints a GDN/Attention/MoE split via `HybridGdnForwardPass.ReportGdnProfile`, gated on the
same env var as `DecodeProfileTimers`. Remove once GDN parallelization (item 1) is investigated and
this three-bucket view is no longer needed.

Test suite check after this change: `Tests.ForwardPass.Fast` — 634/642 pass, 7 fail, all the
pre-existing, environment-specific `Float32` batched/tiered bit-equivalence failures already noted
elsewhere in this repo's docs (`OpenBLAS: not found` on this machine), unrelated to IQ
quantization — confirmed present before this session's changes, not a regression. **Fixed
2026-08-28**: those 7 now `Assert.SkipUnless(SimdKernels.BlasAvailable, ...)`, matching the
pattern `SimdKernelsDequantCacheTests.cs` already used for the same class of issue —
`MatMulBatchedEquivalenceTests.Float32_BatchedMatchesPerTokenMatVec`,
`BatchedMatVecTierTests.TieredFallback_IsBitIdenticalToSequentialMatVec_F32`, and (dtype-scoped,
so `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` coverage in the same test still runs)
`SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec`. Full suite now 642/642 (634 pass, 8
skip) on this machine.
