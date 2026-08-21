using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// AVX2-optimized compute kernels with fused dequantization and multi-threading.
/// All methods expect properly sized, non-null inputs. No bounds checking.
/// </summary>
public static unsafe class SimdKernels
{
    private const int MinRowsForParallel = 64;
    private static readonly ParallelOptions s_parallelOpts = new()
    {
        MaxDegreeOfParallelism = ResolveCpuThreads()
    };

    /// <summary>
    /// Number of worker threads used by the CPU SIMD kernels. Defaults to the logical
    /// processor count and can be set at process start with <c>STINGRAY_CPU_THREADS</c>.
    /// A server option can override it before a model is loaded. This is particularly useful
    /// when inference shares a machine with another CPU-heavy process: too many workers can
    /// reduce token throughput through scheduling and memory-bandwidth contention.
    /// </summary>
    public static int CpuThreads
    {
        get => s_parallelOpts.MaxDegreeOfParallelism;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            s_parallelOpts.MaxDegreeOfParallelism = value;
        }
    }

    private static int ResolveCpuThreads() =>
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_CPU_THREADS"), out int threads)
            && threads > 0
            ? threads
            : Environment.ProcessorCount;

    // ================================================================
    //  Batched GEMM (for prefill)
    // ================================================================

    // Reusable dequant buffer for GEMM (one weight matrix at a time)
    [ThreadStatic] private static nint t_dequantBuf;
    [ThreadStatic] private static int t_dequantBufSize;

    private static bool s_blasLogged;

    /// <summary>
    /// Minimum batch size to engage OpenBLAS SGEMM in MatMulBatched.
    /// Below this threshold, sequential fused MatVec (dequant in registers) is used.
    /// Default 16 is the empirical crossover where SGEMM amortizes F32 dequantization cost
    /// over the per-token compute (measured on Ryzen 9 7900X with Q4_K_M 8192×2048 weights).
    /// Override via STINGRAY_MIN_BATCH_BLAS environment variable.
    /// </summary>
    public static int MinBatchForBlas { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("STINGRAY_MIN_BATCH_BLAS"), out var v) && v >= 1
            ? v
            : 16;

    /// <summary>
    /// Batched matrix multiply: output[batchSize, rows] = input[batchSize, cols] × W[rows, cols]^T
    /// Uses OpenBLAS sgemm when available (dequant weights to F32 temp buffer, then GEMM).
    /// Falls back to sequential MatVec per batch element.
    /// </summary>
    /// <param name="allowQ8">
    /// Whether this call may take the int8 activation-quantized path (subject to
    /// <see cref="Q8PrefillEnabled"/>). <b>Defaults to false, and that default is load-bearing.</b>
    ///
    /// <para>The rows of a batch mean different things at different call sites. In <i>prefill</i>
    /// they are positions within one prompt, so quantizing them together is sound: the whole
    /// prompt takes one consistent path. In <i>batched decode</i> (multi-user
    /// <c>BatchForwardMulti</c>, speculative <c>BatchVerify</c>) each row is an independent
    /// sequence, and single-sequence decode goes through <see cref="MatVec"/> in F32 — so taking
    /// the int8 path here would make a user's logits depend on who else happened to be batched
    /// alongside them, and would break spec-decode's bit-exact verify guarantee.</para>
    ///
    /// <para>Batch size cannot distinguish those cases (a 5-user decode batch and a 5-token
    /// prefill chunk are both "5"), so the caller states its intent instead. Defaulting to false
    /// means a call site that is never audited stays on the safe F32 path.</para>
    /// </param>
    /// <param name="allowBlas">
    /// Whether this call may take the OpenBLAS SGEMM path once <c>batchSize &gt;= MinBatchForBlas</c>.
    /// <b>Defaults to true.</b> A packed batch whose rows span MULTIPLE INDEPENDENT prompts (not
    /// positions within one prompt -- see <see cref="Engine.ForwardPass.PrefillPackedMulti"/>) must
    /// pass <c>false</c>: BLAS SGEMM's summation order is not bit-identical to the dot-product/
    /// tiered fallback, so whether a session's own prefill takes BLAS would otherwise depend on how
    /// many OTHER, unrelated sessions happened to be packed alongside it in the same call --
    /// producing output that is silently sensitive to unrelated concurrent traffic (found via a
    /// real-model concurrency stress test, docs/031-concurrent-decode-batch-tier-divergence-bug.md).
    /// </param>
    public static void MatMulBatched(float* output, byte* weights, float* input,
        int batchSize, int rows, int cols, DType dtype, bool allowQ8 = false, bool allowBlas = true)
    {
        if (!s_blasLogged)
        {
            Console.Error.WriteLine($"[OpenTail.Stingray] OpenBLAS: {(BlasInterop.IsAvailable ? "LOADED" : "not found (fallback to sequential)")}");
            s_blasLogged = true;
        }
        // For small batches, fused MatVec is faster (no dequant overhead)
        // BLAS only wins when N is large enough to amortize F32 dequantization

        // Checked BEFORE the BLAS availability gate below, not after: MicroGemm and the Q8
        // activation-quantized path (measured +47% over the old baseline, see Q8PrefillEnabled)
        // are faster than OpenBLAS's dequant-to-F32-then-sgemm route for the dtypes they support
        // (Q6_K's ffn_down/attn_v tensors on a Q4_K_M model, e.g.), and "BLAS happens to be
        // available" is not evidence it is faster for a given call. Previously these were only
        // tried once BLAS was ruled out, so merely having libopenblas.dll on disk silently routed
        // every large-enough batch straight to the slower BLAS path -- the same class of bug fixed
        // in ForwardPass.MatMulBatchedCached for the Q4_K repacked-x8 path (see its comment and
        // docs/cpu-performance-baseline.md). Only actually falls through to BLAS when these
        // decline (wrong dtype, disabled, or batch below MinBatchForQ8Prefill).
        if (MicroGemmConfig.IsEnabled && dtype == DType.Q4_K &&
            MicroGemmQ4K.TryMatMulQ4K(output, input, (byte*)weights, batchSize, rows, cols))
            return;

        if (allowQ8 && Q8PrefillEnabled && batchSize >= MinBatchForQ8Prefill &&
            TryMatMulBatchedQ8(output, weights, input, batchSize, rows, cols, dtype))
            return;

        {
            // Sequential fused MatVec (dequant in registers, no temp buffer). Tried
            // UNCONDITIONALLY here, before ever checking BLAS availability -- not gated behind
            // "BLAS unavailable" the way it used to be. That gate was the exact bug this method
            // opened with: allowQ8=false callers (batched DECODE -- BatchForwardMulti,
            // speculative verify) used to skip straight past this whole tiered/fused block to
            // OpenBLAS whenever BLAS was available and batchSize >= MinBatchForBlas, even though
            // this is the path measured (below) to fix batched-decode throughput, and BLAS is
            // measured (docs/done/openblas-elimination-findings-2026-08-20.md) to never win a
            // single case tested against the kernels in this file. `allowBlas`/`MinBatchForBlas`
            // are kept as parameters/fields for anyone re-measuring this later, but nothing here
            // should special-case "route around this block toward BLAS" again without new
            // evidence -- see that doc before reintroducing any such gate.
            //
            // MEASURED DEAD END: a weight-stationary variant of this loop (rows outermost,
            // tokens tiled 32-inner, so each weight row is fetched once and reused across the
            // tile) is ~4x SLOWER, not faster: 32.5 -> 8.4 t/s on a 903-token prefill of
            // SmolLM2-1.7B-Q4_K_M. The reasoning that prefill is weight-bandwidth-bound is
            // right, but tiling is the wrong lever: this loop keeps the single activation
            // vector pinned in L1 and walks weights as one long sequential stream that the
            // hardware prefetcher hides almost entirely, whereas tiling swaps that for a
            // ~256 KB activation working set re-read once per row. The lost input locality
            // costs more than the saved weight traffic. Fixing prefill needs a data-layout
            // change (repacked/SoA weights, or activations quantised to int8 once up front so
            // several tokens share one weight read), not a loop reorder.
            // MatMulBatchedEquivalenceTests pins the output contract for any such attempt.
            //
            // docs/cpu-prefill-plan.md §6 step 3: the int8 (_4In) dots below ARE that
            // data-layout change -- weight reuse happens at register level inside one kernel
            // call, not via a re-ordered loop, so the failure mode above doesn't apply to it.
            // MicroGemm and Q8Prefill are tried above, before this gate, now -- see this method's
            // opening comment.

            // fp32 multi-input tiering (session runtime plan §3.4.6). One weight row read, dotted
            // against 4 (then 2, then 1) activation vectors -- the SAME amortisation the int8 path
            // above gets, but in fp32 and therefore with no numerics question at all.
            //
            // Why this matters: without it, batched DECODE (which passes allowQ8: false, and must
            // -- see the parameter doc) degrades to N sequential MatVec calls with zero weight
            // reuse. Measured on the session harness: 1, 2 and 4 concurrent sessions aggregate
            // 27.1 / 27.7 / 29.6 tok/s. Flat. BatchForwardMulti's claim to "amortize weight reads
            // N x across concurrent users" was simply not happening.
            //
            // MatVec4In/MatVec2In are bit-identical per output slot to single MatVec calls --
            // asserted by SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec across
            // Q4_K/Q5_K/Q6_K/F32/Q8_0 including the Parallel.For path, under exactly the contract
            // this needs: "a token's logits must not depend on whether it shared a weight read
            // with one other token ... or three". That contract was established for MTP draft
            // tokens (issue #209); concurrent users need the identical guarantee.
            //
            // Default ON — see BatchedMatVecTierEnabled below for the measurement that promoted it.
            // (This comment used to say "Default OFF until measured end to end", which stayed after
            // the flag was flipped and cost a later investigation an iteration chasing a switch that
            // was already on.)
            //
            // Do not expect this to amortise weight reads N x, and do not reach for it to speed up
            // batched decode or speculative verify. Measured on Qwen3-8B Q4_K_M, best-of-5 over
            // 1.35 GiB of real tensors: MatVec2In gains 1.06x and MatVec4In 1.11x over separate
            // calls. Fitting T(n) = M + n*C gives M = 7 ms, C = 45 ms — the kernel is ~87% dequant/
            // dot COMPUTE and only ~13% weight streaming, and that 13% is the entire ceiling for any
            // scheme that reuses a weight read. The model predicted T(2) = 97 ms against an actual
            // 98 ms, so this is a validated bound rather than a single observation. Making batched
            // decode faster requires a cheaper dot (e.g. VNNI), not better data movement.
            // See docs/done/cpu-speculative-decoding-findings.md.
            if (BatchedMatVecTierEnabled)
            {
                Interlocked.Increment(ref BatchedMatVecTierCalls);
                int t = 0;
                for (; t + 4 <= batchSize; t += 4)
                    MatVec4In(output + (long)t * rows, output + (long)(t + 1) * rows,
                              output + (long)(t + 2) * rows, output + (long)(t + 3) * rows,
                              weights,
                              input + (long)t * cols, input + (long)(t + 1) * cols,
                              input + (long)(t + 2) * cols, input + (long)(t + 3) * cols,
                              rows, cols, dtype);
                for (; t + 2 <= batchSize; t += 2)
                    MatVec2In(output + (long)t * rows, output + (long)(t + 1) * rows,
                              weights,
                              input + (long)t * cols, input + (long)(t + 1) * cols,
                              rows, cols, dtype);
                for (; t < batchSize; t++)
                    MatVec(output + (long)t * rows, weights, input + (long)t * cols, rows, cols, dtype);
                return;
            }

            for (int n = 0; n < batchSize; n++)
                MatVec(output + n * rows, weights, input + n * cols, rows, cols, dtype);
            return;
        }

        // --- Everything below this line is OpenBLAS. It is UNREACHABLE under any shipped
        // default configuration: the block above always returns first (BatchedMatVecTierEnabled
        // or the plain MatVec loop, both unconditional, neither can decline). That is
        // intentional, not a bug to "fix" by moving BLAS earlier or by gating the block above
        // again -- see docs/done/openblas-elimination-findings-2026-08-20.md for the measurements
        // (every shape/dtype/batch-size tested, OpenBLAS lost) and the pain of re-discovering
        // this after two separate ordering regressions shipped it back to the front of the race.
        // Kept in source, genuinely last-resort, for anyone who wants to re-measure on different
        // hardware or an untested shape -- not deleted, per explicit instruction. If you are
        // tempted to route a call here ahead of the block above: don't, without new measurements
        // to justify it, written up the same way.

#pragma warning disable CS0162 // unreachable -- see the comment above this line
        // OpenBLAS GEMM path for large batches: dequant weights to F32, then sgemm
        if (dtype != DType.Float32)
        {
            int weightElements = rows * cols;

            // Ensure thread-local dequant buffer is large enough
            if (t_dequantBufSize < weightElements)
            {
                if (t_dequantBuf != 0) NativeMemory.Free((void*)t_dequantBuf);
                t_dequantBuf = (nint)NativeMemory.AllocZeroed((nuint)(weightElements * sizeof(float)));
                t_dequantBufSize = weightElements;
            }
            var wf32 = (float*)t_dequantBuf;

            // Dequantize full weight matrix to F32
            long totalBytes = DTypeInfo.ByteSize(weightElements, dtype);
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(weights, (int)totalBytes),
                new Span<float>(wf32, weightElements),
                dtype, weightElements);

            // sgemm: C[M,N] = A[M,K] * B[K,N]
            // We want: output[batchSize, rows] = input[batchSize, cols] * W[rows, cols]^T
            // In row-major: C = input * W^T
            // sgemm(RowMajor, NoTrans, Trans, M=batchSize, N=rows, K=cols,
            //        alpha=1, A=input, lda=cols, B=W, ldb=cols, beta=0, C=output, ldc=rows)
            if (MicroGemmConfig.IsEnabled && MicroGemmKernel.TryMatMulF32(input, wf32, output, batchSize, cols, rows))
                return;

            BlasInterop.Sgemm(
                BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                batchSize, rows, cols,
                1.0f, input, cols,
                wf32, cols,
                0.0f, output, rows);
            return;
        }

        // F32 weights with BLAS
        if (BlasInterop.IsAvailable && dtype == DType.Float32)
        {
            if (MicroGemmConfig.IsEnabled && MicroGemmKernel.TryMatMulF32(input, (float*)weights, output, batchSize, cols, rows))
                return;

            BlasInterop.Sgemm(
                BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                batchSize, rows, cols,
                1.0f, input, cols,
                (float*)weights, cols,
                0.0f, output, rows);
            return;
        }
#pragma warning restore CS0162

    }

    /// <summary>Whether OpenBLAS was found, i.e. whether the SGEMM batched-prefill path is live.</summary>
    public static bool BlasAvailable => BlasInterop.IsAvailable;

    /// <summary>
    /// Whether the int8 (Q8_K/Q8_KS) batched-prefill path in <see cref="MatMulBatched"/> is live.
    ///
    /// <para><b>Default on.</b> Quantizing the activation rows lets each weight row be read once
    /// and dotted against 8 tokens per call (the <c>_8In</c>/<c>_4In</c> kernels) instead of once
    /// per token — worth ~+47% end-to-end prefill throughput on the reference box, and the same
    /// technique llama.cpp uses for its own prefill GEMM. Opt out with
    /// <c>STINGRAY_CPU_PREFILL_Q8=0</c>.</para>
    ///
    /// <para>The Q8 dots are NOT byte-exact with the F32 dots decode uses (docs/cpu-prefill-plan.md
    /// §10), so prefill's numerics differ slightly from decode's on a dense model. That gap was the
    /// stated blocker on defaulting this on, and it has since been measured rather than assumed:
    /// perplexity moves by −0.14% on a diverse 5-topic 2047-token corpus and −0.4% on the original
    /// single-document corpus (both slightly *better* with the gate on, i.e. noise-level, not a
    /// regression), and greedy generation is 100% bit-identical across two real prompts. Suites
    /// that pin the F32 path's exact-equality contract set this to <c>false</c> explicitly rather
    /// than relying on the ambient default.</para>
    ///
    /// <para>All-control-token prompts are an explicit exception: <see cref="Engine.ForwardPass"/>
    /// detects that structural GGUF input and uses its sequential F32 path even while this gate is
    /// enabled. Mixed prompts (including normal BOS-plus-text input) remain eligible.</para>
    /// </summary>
    /// <summary>
    /// How many times the tiered fallback actually executed. Exists because a performance result
    /// — positive or negative — is only meaningful if the changed code ran. Without it "no effect"
    /// and "never invoked" are indistinguishable, and the second masquerading as the first is how
    /// a real optimisation gets wrongly abandoned. It earned its keep immediately: the first A/B
    /// of this switch was run against a stale binary and looked like a clean null result.
    /// </summary>
    public static long BatchedMatVecTierCalls;

    /// <summary>
    /// Whether <see cref="MatMulBatched"/>'s non-BLAS fallback consumes the batch in quads/pairs
    /// via <see cref="MatVec4In"/>/<see cref="MatVec2In"/> instead of N sequential
    /// <see cref="MatVec"/> calls. Bit-identical either way; this is purely a weight-reuse switch.
    /// <c>STINGRAY_BATCHED_MATVEC_TIER=0</c> disables it.
    /// <para>Default ON as of the measurement in the session runtime plan §3.4.9: 4-way concurrent
    /// decode 28.90 -> 30.28 tok/s aggregate, 4 interleaved samples per arm with NO overlap between
    /// the arms' ranges. Small — and deliberately NOT described as amortising weight reads "N x",
    /// because it demonstrably does not. Single-session decode is unchanged, as expected: a batch
    /// of 1 takes the single-MatVec remainder.</para>
    /// </summary>
    public static bool BatchedMatVecTierEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_BATCHED_MATVEC_TIER") != "0";

    public static bool Q8PrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_CPU_PREFILL_Q8") != "0";

    /// <summary>
    /// Smallest batch that takes the int8 path when <see cref="Q8PrefillEnabled"/> is on.
    ///
    /// <para>This is a <b>numerics boundary, not just a performance knob</b>: batches below it run
    /// the F32 per-token loop instead, so a prompt admitted in chunks straddling this value would
    /// have some positions computed in int8 and others in F32. Chunked and unchunked prefill of the
    /// same prompt must agree, so the threshold is 1 — every batch takes the same path.
    /// <see cref="TryMatMulBatchedQ8"/> already handles any batch size (it dots whatever tokens
    /// remain through the single-input Q8 dot), so a small batch is correct here, merely
    /// unamortized — and an unamortized quantize costs ~0.05 ms, far below the cost of being
    /// inconsistent.</para>
    /// </summary>
    public static int MinBatchForQ8Prefill { get; set; } = 1;

    /// <summary>Per-dtype quantizer/dot mapping the int8 batched path is defined for.</summary>
    private delegate void QuantizeRow(float* input, int cols, byte* scratch);
    private delegate void FourInDot(byte* row, byte* s0, byte* s1, byte* s2, byte* s3, int cols,
        out float o0, out float o1, out float o2, out float o3);
    private delegate void EightInDot(byte* row, byte* s0, byte* s1, byte* s2, byte* s3,
        byte* s4, byte* s5, byte* s6, byte* s7, int cols,
        out float o0, out float o1, out float o2, out float o3,
        out float o4, out float o5, out float o6, out float o7);
    private delegate float OneInDot(byte* row, byte* scratch, int cols);

    /// <summary>
    /// Resolve the exact quantizer, scratch size, and dot family for one dtype. Q8_K and
    /// Q8_KS are different scratch layouts and are NOT interchangeable (see docs §4) — this is
    /// the single place that mapping is made, once per <see cref="MatMulBatched"/> call, never
    /// per element, so a per-dtype mismatch is impossible by construction rather than by
    /// per-call-site care.
    /// </summary>
    private static bool TryResolveQ8Dispatch(DType dtype, int cols,
        out QuantizeRow quantize, out EightInDot? dot8, out FourInDot dot4, out OneInDot dot1,
        out int scratchBytesPerToken, out int bytesPerRow)
    {
        switch (dtype)
        {
            case DType.Q4_K:
                quantize = QuantizeRowToQ8KS;
                dot8 = DotQ4K_Q8KS_8In;
                dot4 = DotQ4K_Q8KS_4In;
                dot1 = DotQ4K_Q8KS;
                scratchBytesPerToken = Q8KSScratchBytes(cols);
                bytesPerRow = (cols / 256) * 144;
                return true;
            case DType.Q3_K:
                quantize = QuantizeRowToQ8KS;
                dot8 = DotQ3K_Q8KS_8In;
                dot4 = DotQ3K_Q8KS_4In;
                dot1 = DotQ3K_Q8KS;
                scratchBytesPerToken = Q8KSScratchBytes(cols);
                bytesPerRow = (cols / 256) * 110;
                return true;
            case DType.Q6_K:
                quantize = QuantizeRowToQ8K;
                dot8 = DotQ6K_Q8K_8In;
                dot4 = DotQ6K_Q8K_4In;
                dot1 = DotQ6K_Q8K;
                scratchBytesPerToken = Q8KScratchBytes(cols);
                bytesPerRow = (cols / 256) * 210;
                return true;
            case DType.Q4_0:
                // 32-element blocks, not 256 — Q4_0 pairs with Q8_0, not with the K-quant
                // superblock activation formats. No _8In yet: dot8 is optional and the batched
                // loop falls to _4In, which already carries the weight-reuse win.
                quantize = QuantizeRowToQ8_0;
                dot8 = null;
                dot4 = DotQ4_0_Q8_0_4In;
                dot1 = DotQ4_0_Q8_0;
                scratchBytesPerToken = Q8_0ScratchBytes(cols);
                bytesPerRow = (cols / 32) * 18;
                return true;
            default:
                // Q8_0 has single-input Q8 dots only (no _4In/_2In); Q5_K, Q2_K, Float32 have
                // no int8 dot at all. All fall back to the existing per-token loop.
                quantize = null!; dot8 = null; dot4 = null!; dot1 = null!;
                scratchBytesPerToken = 0; bytesPerRow = 0;
                return false;
        }
    }

    /// <summary>
    /// Int8 batched prefill: quantize every token's activation row to Q8 once, then for each
    /// weight row read it once and dot it against eight or four tokens per call via the existing
    /// register-level <c>_8In</c>/<c>_4In</c> kernels, falling back to the single-input Q8 dot for the
    /// leftover tokens. Bound to 512-token L2-cache chunks for large prompts. Not byte-exact with the F32 path -- see <see cref="Q8PrefillEnabled"/>.
    /// </summary>
    internal static bool TryMatMulBatchedQ8(float* output, byte* weights, float* input,
        int batchSize, int rows, int cols, DType dtype)
    {
        const int MaxPrefillChunkTokens = 512;
        if (batchSize > MaxPrefillChunkTokens)
        {
            // BUG (found in review, fixed here): the original version discarded each chunk's
            // return value and always returned true. For a dtype TryResolveQ8Dispatch doesn't
            // support (Q5_K, Q2_K, Float32, Q8_0), every recursive call correctly returns false
            // WITHOUT writing any output -- but the caller (MatMulBatched, below) only checks
            // this wrapper's own return value before deciding whether to run its per-token
            // MatVec fallback. Unconditionally returning true here meant that fallback never
            // ran, silently leaving the entire `output` buffer uncomputed for any >512-token
            // batch of an unsupported dtype with the Q8 gate on -- confirmed via a targeted
            // repro (Q5_K, batchSize=600: 9600/9600 output values never written).
            bool allSucceeded = true;
            for (int chunkStart = 0; chunkStart < batchSize; chunkStart += MaxPrefillChunkTokens)
            {
                int chunkSize = Math.Min(MaxPrefillChunkTokens, batchSize - chunkStart);
                float* chunkOutput = output + (long)chunkStart * rows;
                float* chunkInput = input + (long)chunkStart * cols;
                allSucceeded &= TryMatMulBatchedQ8(chunkOutput, weights, chunkInput, chunkSize, rows, cols, dtype);
            }
            return allSucceeded;
        }

        if (!TryResolveQ8Dispatch(dtype, cols, out var quantize, out var dot8, out var dot4, out var dot1,
                out int scratchPerToken, out int bytesPerRow))
            return false;

        nuint scratchTotal = (nuint)((long)scratchPerToken * batchSize);
        byte* scratchBase = (byte*)NativeMemory.Alloc(scratchTotal);
        try
        {
            // Quantize every token's activations once up front in parallel if batchSize >= 4.
            if (batchSize >= 4)
            {
                Parallel.For(0, batchSize, n =>
                {
                    quantize(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken);
                });
            }
            else
            {
                for (int n = 0; n < batchSize; n++)
                    quantize(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken);
            }

            int groupsOf8 = (dot8 != null && batchSize >= 8) ? batchSize / 8 : 0;
            int remAfter8 = batchSize - (groupsOf8 * 8);
            int groupsOf4 = remAfter8 / 4;
            int offset4 = groupsOf8 * 8;

            void ProcessRow(int r)
            {
                byte* row = weights + (long)r * bytesPerRow;
                float* o = output + r; // output is [token, row]; stride between tokens is `rows`

                // Software prefetch: pull next row's first cache lines into L1 while computing
                // the current row. Helps batched prefill where threads jump between rows with
                // stride bytesPerRow — the hardware prefetcher can't predict this access pattern.
                if (Sse.IsSupported && r + 1 < rows)
                {
                    byte* nextRow = weights + (long)(r + 1) * bytesPerRow;
                    Sse.Prefetch0(nextRow);
                    Sse.Prefetch0(nextRow + 64);
                }

                for (int g = 0; g < groupsOf8; g++)
                {
                    int n = g * 8;
                    byte* s0 = scratchBase + (long)n * scratchPerToken;
                    byte* s1 = scratchBase + (long)(n + 1) * scratchPerToken;
                    byte* s2 = scratchBase + (long)(n + 2) * scratchPerToken;
                    byte* s3 = scratchBase + (long)(n + 3) * scratchPerToken;
                    byte* s4 = scratchBase + (long)(n + 4) * scratchPerToken;
                    byte* s5 = scratchBase + (long)(n + 5) * scratchPerToken;
                    byte* s6 = scratchBase + (long)(n + 6) * scratchPerToken;
                    byte* s7 = scratchBase + (long)(n + 7) * scratchPerToken;
                    dot8!(row, s0, s1, s2, s3, s4, s5, s6, s7, cols,
                        out float v0, out float v1, out float v2, out float v3,
                        out float v4, out float v5, out float v6, out float v7);
                    o[(long)n * rows] = v0;
                    o[(long)(n + 1) * rows] = v1;
                    o[(long)(n + 2) * rows] = v2;
                    o[(long)(n + 3) * rows] = v3;
                    o[(long)(n + 4) * rows] = v4;
                    o[(long)(n + 5) * rows] = v5;
                    o[(long)(n + 6) * rows] = v6;
                    o[(long)(n + 7) * rows] = v7;
                }

                for (int g = 0; g < groupsOf4; g++)
                {
                    int n = offset4 + g * 4;
                    byte* s0 = scratchBase + (long)n * scratchPerToken;
                    byte* s1 = scratchBase + (long)(n + 1) * scratchPerToken;
                    byte* s2 = scratchBase + (long)(n + 2) * scratchPerToken;
                    byte* s3 = scratchBase + (long)(n + 3) * scratchPerToken;
                    dot4(row, s0, s1, s2, s3, cols, out float v0, out float v1, out float v2, out float v3);
                    o[(long)n * rows] = v0;
                    o[(long)(n + 1) * rows] = v1;
                    o[(long)(n + 2) * rows] = v2;
                    o[(long)(n + 3) * rows] = v3;
                }

                for (int n = offset4 + groupsOf4 * 4; n < batchSize; n++)
                    o[(long)n * rows] = dot1(row, scratchBase + (long)n * scratchPerToken, cols);
            }

            if (rows >= MinRowsForParallel)
                Parallel.For(0, rows, s_parallelOpts, ProcessRow);
            else
                for (int r = 0; r < rows; r++) ProcessRow(r);

            return true;
        }
        finally
        {
            NativeMemory.Free(scratchBase);
        }
    }

    /// <summary>
    /// Perf-loop iteration 13 (docs/perf-loop-progress.md): dual-weight sibling of
    /// <see cref="TryMatMulBatchedQ8"/>, mirroring what <see cref="MatVecDual"/> already does
    /// for decode's Q4_K gate+up matvecs -- for two weight matrices sharing the SAME activation
    /// input (the FFN gate/up projections read the same normalized hidden state), quantize the
    /// activation panel to Q8 ONCE and reuse it for both, instead of <c>PrefillCore</c> currently
    /// calling <see cref="TryMatMulBatchedQ8"/> twice (via two separate <c>MatMulBatchedCached</c>
    /// calls), which redundantly re-quantizes the identical activation panel a second time and
    /// pays a second `Parallel.For` dispatch. Both weight matrices must share dtype/rows/cols
    /// (true for gate/up in every architecture this dispatches for). Same correctness contract as
    /// <see cref="TryMatMulBatchedQ8"/>: returns false (writing nothing) for a dtype with no Q8
    /// dot support, same as the caller already handles for the single-weight path.
    /// </summary>
    public static bool TryMatMulBatchedDualQ8(
        float* output1, byte* weights1, float* output2, byte* weights2, float* input,
        int batchSize, int rows, int cols, DType dtype)
    {
        const int MaxPrefillChunkTokens = 512;
        if (batchSize > MaxPrefillChunkTokens)
        {
            bool allSucceeded = true;
            for (int chunkStart = 0; chunkStart < batchSize; chunkStart += MaxPrefillChunkTokens)
            {
                int chunkSize = Math.Min(MaxPrefillChunkTokens, batchSize - chunkStart);
                float* chunkOutput1 = output1 + (long)chunkStart * rows;
                float* chunkOutput2 = output2 + (long)chunkStart * rows;
                float* chunkInput = input + (long)chunkStart * cols;
                allSucceeded &= TryMatMulBatchedDualQ8(chunkOutput1, weights1, chunkOutput2, weights2,
                    chunkInput, chunkSize, rows, cols, dtype);
            }
            return allSucceeded;
        }

        if (!TryResolveQ8Dispatch(dtype, cols, out var quantize, out var dot8, out var dot4, out var dot1,
                out int scratchPerToken, out int bytesPerRow))
            return false;

        nuint scratchTotal = (nuint)((long)scratchPerToken * batchSize);
        byte* scratchBase = (byte*)NativeMemory.Alloc(scratchTotal);
        try
        {
            // Quantized ONCE for both weight matrices -- the one change from calling
            // TryMatMulBatchedQ8 twice.
            if (batchSize >= 4)
            {
                Parallel.For(0, batchSize, n =>
                {
                    quantize(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken);
                });
            }
            else
            {
                for (int n = 0; n < batchSize; n++)
                    quantize(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken);
            }

            int groupsOf8 = (dot8 != null && batchSize >= 8) ? batchSize / 8 : 0;
            int remAfter8 = batchSize - (groupsOf8 * 8);
            int groupsOf4 = remAfter8 / 4;
            int offset4 = groupsOf8 * 8;

            void ProcessRowBoth(int r)
            {
                byte* row1 = weights1 + (long)r * bytesPerRow;
                byte* row2 = weights2 + (long)r * bytesPerRow;
                float* o1 = output1 + r;
                float* o2 = output2 + r;

                if (Sse.IsSupported && r + 1 < rows)
                {
                    byte* nextRow1 = weights1 + (long)(r + 1) * bytesPerRow;
                    byte* nextRow2 = weights2 + (long)(r + 1) * bytesPerRow;
                    Sse.Prefetch0(nextRow1);
                    Sse.Prefetch0(nextRow1 + 64);
                    Sse.Prefetch0(nextRow2);
                    Sse.Prefetch0(nextRow2 + 64);
                }

                for (int g = 0; g < groupsOf8; g++)
                {
                    int n = g * 8;
                    byte* s0 = scratchBase + (long)n * scratchPerToken;
                    byte* s1 = scratchBase + (long)(n + 1) * scratchPerToken;
                    byte* s2 = scratchBase + (long)(n + 2) * scratchPerToken;
                    byte* s3 = scratchBase + (long)(n + 3) * scratchPerToken;
                    byte* s4 = scratchBase + (long)(n + 4) * scratchPerToken;
                    byte* s5 = scratchBase + (long)(n + 5) * scratchPerToken;
                    byte* s6 = scratchBase + (long)(n + 6) * scratchPerToken;
                    byte* s7 = scratchBase + (long)(n + 7) * scratchPerToken;

                    dot8!(row1, s0, s1, s2, s3, s4, s5, s6, s7, cols,
                        out float a0, out float a1, out float a2, out float a3,
                        out float a4, out float a5, out float a6, out float a7);
                    o1[(long)n * rows] = a0; o1[(long)(n + 1) * rows] = a1;
                    o1[(long)(n + 2) * rows] = a2; o1[(long)(n + 3) * rows] = a3;
                    o1[(long)(n + 4) * rows] = a4; o1[(long)(n + 5) * rows] = a5;
                    o1[(long)(n + 6) * rows] = a6; o1[(long)(n + 7) * rows] = a7;

                    dot8(row2, s0, s1, s2, s3, s4, s5, s6, s7, cols,
                        out float b0, out float b1, out float b2, out float b3,
                        out float b4, out float b5, out float b6, out float b7);
                    o2[(long)n * rows] = b0; o2[(long)(n + 1) * rows] = b1;
                    o2[(long)(n + 2) * rows] = b2; o2[(long)(n + 3) * rows] = b3;
                    o2[(long)(n + 4) * rows] = b4; o2[(long)(n + 5) * rows] = b5;
                    o2[(long)(n + 6) * rows] = b6; o2[(long)(n + 7) * rows] = b7;
                }

                for (int g = 0; g < groupsOf4; g++)
                {
                    int n = offset4 + g * 4;
                    byte* s0 = scratchBase + (long)n * scratchPerToken;
                    byte* s1 = scratchBase + (long)(n + 1) * scratchPerToken;
                    byte* s2 = scratchBase + (long)(n + 2) * scratchPerToken;
                    byte* s3 = scratchBase + (long)(n + 3) * scratchPerToken;

                    dot4(row1, s0, s1, s2, s3, cols, out float a0, out float a1, out float a2, out float a3);
                    o1[(long)n * rows] = a0; o1[(long)(n + 1) * rows] = a1;
                    o1[(long)(n + 2) * rows] = a2; o1[(long)(n + 3) * rows] = a3;

                    dot4(row2, s0, s1, s2, s3, cols, out float b0, out float b1, out float b2, out float b3);
                    o2[(long)n * rows] = b0; o2[(long)(n + 1) * rows] = b1;
                    o2[(long)(n + 2) * rows] = b2; o2[(long)(n + 3) * rows] = b3;
                }

                for (int n = offset4 + groupsOf4 * 4; n < batchSize; n++)
                {
                    byte* s = scratchBase + (long)n * scratchPerToken;
                    o1[(long)n * rows] = dot1(row1, s, cols);
                    o2[(long)n * rows] = dot1(row2, s, cols);
                }
            }

            if (rows >= MinRowsForParallel)
                Parallel.For(0, rows, s_parallelOpts, ProcessRowBoth);
            else
                for (int r = 0; r < rows; r++) ProcessRowBoth(r);

            return true;
        }
        finally
        {
            NativeMemory.Free(scratchBase);
        }
    }

    /// <summary>
    /// Batched matrix multiply against an <b>already-dequantized F32</b> weight matrix —
    /// the dequant-free twin of <see cref="MatMulBatched"/>. Issue #189: chunked prompt
    /// admission re-walks the same layer weights every chunk, so <see cref="MatMulBatched"/>
    /// re-pays the full Q→F32 dequant on every call. When a caller (ForwardPass) holds the
    /// F32 dequant of a weight in a reuse cache, it routes here to skip dequant entirely.
    /// Bit-identical to <see cref="MatMulBatched"/>'s BLAS path: same F32 weights, same SGEMM.
    /// </summary>
    public static void MatMulBatchedF32(float* output, float* weightsF32, float* input,
        int batchSize, int rows, int cols)
    {
        // Tried unconditionally, not gated behind "batch too small for BLAS to be worth it" --
        // same reasoning as MatMulBatched above: BLAS has never won a measured case in this file
        // (docs/done/openblas-elimination-findings-2026-08-20.md), including large batches, so
        // there is no batch size at which routing to it ahead of this loop is justified by
        // evidence. Kept structurally last-resort, not deleted -- do not re-gate this behind a
        // batch-size or BLAS-availability check without new measurements written up the same way.
        for (int n = 0; n < batchSize; n++)
            MatVecF32(output + n * rows, weightsF32, input + n * cols, rows, cols);
        return;

        // Unreachable below by design -- see comment above.
#pragma warning disable CS0162
        BlasInterop.Sgemm(
            BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
            batchSize, rows, cols,
            1.0f, input, cols,
            weightsF32, cols,
            0.0f, output, rows);
#pragma warning restore CS0162
    }

    // ================================================================
    //  Dispatchers
    // ================================================================

    /// <summary>
    /// Fused matrix-vector multiply. For quantized dtypes, dequantization
    /// happens in registers — no intermediate F32 buffer is allocated.
    /// </summary>
    public static void MatVec(float* output, byte* weights, float* input,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Float32:
                MatVecF32(output, (float*)weights, input, rows, cols);
                break;
            case DType.Q4_K:
                MatVecQ4K(output, weights, input, rows, cols);
                break;
            case DType.Q6_K:
                MatVecQ6K(output, weights, input, rows, cols);
                break;
            case DType.Q5_K:
                MatVecQ5K(output, weights, input, rows, cols);
                break;
            case DType.Q2_K:
                MatVecQ2K(output, weights, input, rows, cols);
                break;
            case DType.Q3_K:
                MatVecQ3K(output, weights, input, rows, cols);
                break;
            case DType.Q8_0:
                MatVecQ8_0(output, weights, input, rows, cols);
                break;
            case DType.Q4_0:
                MatVecQ4_0(output, weights, input, rows, cols);
                break;
            case DType.IQ4_NL:
                MatVecIq4Nl(output, weights, input, rows, cols);
                break;
            case DType.Float16 when GgmlF16DotEnabled:
                MatVecF16GgmlCompat(output, weights, input, rows, cols);
                break;
            default:
                MatVecDequantFallback(output, weights, input, rows, cols, dtype);
                break;
        }
    }

    /// <summary>
    /// Compute two matrix-vector products sharing the same input in a single Parallel.For,
    /// halving thread-dispatch overhead for fused gate+up FFN projections.
    /// Both weight matrices must have the same dtype, rows, and cols.
    /// Falls back to two sequential MatVec calls if dtypes differ.
    /// </summary>
    public static void MatVecDual(
        float* output1, byte* weights1,
        float* output2, byte* weights2,
        float* input, int rows, int cols, DType dtype1, DType dtype2)
    {
        if (dtype1 != dtype2)
        {
            MatVec(output1, weights1, input, rows, cols, dtype1);
            MatVec(output2, weights2, input, rows, cols, dtype2);
            return;
        }

        switch (dtype1)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    if (UseWide8)
                        Parallel.For(0, rows, s_parallelOpts, r =>
                        {
                            o1[r] = DotQ4K_Wide8(w1 + (long)r * bpr, inp, c);
                            o2[r] = DotQ4K_Wide8(w2 + (long)r * bpr, inp, c);
                        });
                    else
                        Parallel.For(0, rows, s_parallelOpts, r =>
                        {
                            o1[r] = DotQ4K(w1 + (long)r * bpr, inp, c);
                            o2[r] = DotQ4K(w2 + (long)r * bpr, inp, c);
                        });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ4K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ4K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                int scratchBytes = Q8KScratchBytes(cols);
                byte* scratch = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input, cols, scratch);

                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var s = scratch;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ6K_Q8K(w1 + (long)r * bpr, s, c);
                        o2[r] = DotQ6K_Q8K(w2 + (long)r * bpr, s, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ6K_Q8K(weights1 + (long)r * bpr, scratch, cols);
                        output2[r] = DotQ6K_Q8K(weights2 + (long)r * bpr, scratch, cols);
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ5K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ5K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ5K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ5K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q3_K:
            {
                int bpr = (cols / 256) * 110;
                int scratchBytes = Q8KScratchBytes(cols);
                byte* scratch = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input, cols, scratch);

                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var s = scratch;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ3K_Q8K(w1 + (long)r * bpr, s, c);
                        o2[r] = DotQ3K_Q8K(w2 + (long)r * bpr, s, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ3K_Q8K(weights1 + (long)r * bpr, scratch, cols);
                        output2[r] = DotQ3K_Q8K(weights2 + (long)r * bpr, scratch, cols);
                    }
                }
                break;
            }
            case DType.Q2_K:
            {
                int bpr = (cols / 256) * 84;
                int scratchBytes = Q8KScratchBytes(cols);
                byte* scratch = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input, cols, scratch);

                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var s = scratch;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ2K_Q8K(w1 + (long)r * bpr, s, c);
                        o2[r] = DotQ2K_Q8K(w2 + (long)r * bpr, s, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ2K_Q8K(weights1 + (long)r * bpr, scratch, cols);
                        output2[r] = DotQ2K_Q8K(weights2 + (long)r * bpr, scratch, cols);
                    }
                }
                break;
            }
            case DType.Q8_0:
            {
                int bpr = (cols / 32) * 34;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ8_0(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ8_0(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ8_0(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ8_0(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Float32:
            {
                if (rows >= MinRowsForParallel)
                {
                    var m1 = (float*)weights1; var m2 = (float*)weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotF32(m1 + (long)r * c, inp, c);
                        o2[r] = DotF32(m2 + (long)r * c, inp, c);
                    });
                }
                else
                {
                    var m1 = (float*)weights1; var m2 = (float*)weights2;
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotF32(m1 + (long)r * cols, input, cols);
                        output2[r] = DotF32(m2 + (long)r * cols, input, cols);
                    }
                }
                break;
            }
            default:
                MatVec(output1, weights1, input, rows, cols, dtype1);
                MatVec(output2, weights2, input, rows, cols, dtype2);
                break;
        }
    }

    /// <summary>
    /// Compute two matrix-vector products sharing the same weight matrix against
    /// two distinct inputs in a single Parallel.For sweep:
    /// <c>output1 = weights @ input1</c> and <c>output2 = weights @ input2</c>.
    /// Each weight row is touched once per row iteration; the second dot reads the
    /// just-loaded row from L1, halving the effective weight-bandwidth cost of the
    /// pair vs two sequential <see cref="MatVec"/> calls. Used by the MTP batched
    /// verify path (issue #30) where both tokens share the same FFN weights.
    /// </summary>
    public static void MatVec2In(
        float* output1, float* output2,
        byte* weights, float* input1, float* input2,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ4K_2In(row, i1, i2, c, out float s1, out float s2);
                        o1[r] = s1; o2[r] = s2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ4K_2In(row, input1, input2, cols, out float s1, out float s2);
                        output1[r] = s1; output2[r] = s2;
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ5K_2In(row, i1, i2, c, out float s1, out float s2);
                        o1[r] = s1; o2[r] = s2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ5K_2In(row, input1, input2, cols, out float s1, out float s2);
                        output1[r] = s1; output2[r] = s2;
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                int scratchBytes = Q8KScratchBytes(cols);
                // Two Q8_K scratches (one per input); stack-alloc when small enough,
                // heap fallback for large cols (Q8_K scratch is ~262 B per 256 elems).
                byte* sc1 = stackalloc byte[scratchBytes];
                byte* sc2 = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input1, cols, sc1);
                QuantizeRowToQ8K(input2, cols, sc2);

                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var s1 = sc1; var s2 = sc2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ6K_Q8K_2In(row, s1, s2, c, out float v1, out float v2);
                        o1[r] = v1;
                        o2[r] = v2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ6K_Q8K_2In(row, sc1, sc2, cols, out float v1, out float v2);
                        output1[r] = v1;
                        output2[r] = v2;
                    }
                }
                break;
            }
            case DType.Q8_0:
            {
                // No fused 2In kernel: Q8_0 has no expensive nibble unpack to amortize,
                // so two sequential DotQ8_0 per row keep the weight-row-in-L1 reuse and
                // stay bit-identical to single MatVec calls (and to the batched paths'
                // DispatchDot2In, whose Q8_0 case is the same two-single-dots fallback).
                int bpr = (cols / 32) * 34;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        o1[r] = DotQ8_0(row, i1, c);
                        o2[r] = DotQ8_0(row, i2, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        output1[r] = DotQ8_0(row, input1, cols);
                        output2[r] = DotQ8_0(row, input2, cols);
                    }
                }
                break;
            }
            case DType.Float32:
            {
                var m = (float*)weights;
                if (rows >= MinRowsForParallel)
                {
                    var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        float* row = m + (long)r * c;
                        o1[r] = DotF32(row, i1, c);
                        o2[r] = DotF32(row, i2, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = m + (long)r * cols;
                        output1[r] = DotF32(row, input1, cols);
                        output2[r] = DotF32(row, input2, cols);
                    }
                }
                break;
            }
            default:
                // Fallback: two sequential MatVec calls. Loses the weight-bandwidth
                // benefit but stays correct for dtypes we haven't specialised yet.
                MatVec(output1, weights, input1, rows, cols, dtype);
                MatVec(output2, weights, input2, rows, cols, dtype);
                break;
        }
    }

    /// <summary>
    /// Four-input fused mat-vec (issue #209): for each weight row
    /// <c>output{0..3}[r] = weights[r] · input{0..3}</c>. Decodes each weight row
    /// ONCE and dots it against four token columns in the same pass — one weight
    /// HBM/L2 read per four tokens versus <see cref="MatVec2In"/>'s one-per-two. This
    /// is the dominant lever on the 27B-MTP CUDA-hybrid decode path, where 46/64 dense
    /// FFN layers are CPU-mmap'd and re-read once per draft token. Per-token
    /// accumulation order is identical to <see cref="MatVec2In"/> / single
    /// <see cref="MatVec"/>, so each position's bits are independent of the batch width
    /// k (the duplicated-input-tail contract — see the BatchVerify callers, which fill
    /// past-the-end lanes with a duplicate token routed to a sink).
    /// </summary>
    public static void MatVec4In(
        float* output0, float* output1, float* output2, float* output3,
        byte* weights,
        float* input0, float* input1, float* input2, float* input3,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ4K_4In(row, i0, i1, i2, i3, c, out float s0, out float s1, out float s2, out float s3);
                        o0[r] = s0; o1[r] = s1; o2[r] = s2; o3[r] = s3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ4K_4In(row, input0, input1, input2, input3, cols, out float s0, out float s1, out float s2, out float s3);
                        output0[r] = s0; output1[r] = s1; output2[r] = s2; output3[r] = s3;
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ5K_4In(row, i0, i1, i2, i3, c, out float s0, out float s1, out float s2, out float s3);
                        o0[r] = s0; o1[r] = s1; o2[r] = s2; o3[r] = s3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ5K_4In(row, input0, input1, input2, input3, cols, out float s0, out float s1, out float s2, out float s3);
                        output0[r] = s0; output1[r] = s1; output2[r] = s2; output3[r] = s3;
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                int scratchBytes = Q8KScratchBytes(cols);
                // One Q8_K scratch per input; same stack-alloc discipline as MatVec2In.
                byte* sc0 = stackalloc byte[scratchBytes];
                byte* sc1 = stackalloc byte[scratchBytes];
                byte* sc2 = stackalloc byte[scratchBytes];
                byte* sc3 = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input0, cols, sc0);
                QuantizeRowToQ8K(input1, cols, sc1);
                QuantizeRowToQ8K(input2, cols, sc2);
                QuantizeRowToQ8K(input3, cols, sc3);

                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var s0 = sc0; var s1 = sc1; var s2 = sc2; var s3 = sc3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ6K_Q8K_4In(row, s0, s1, s2, s3, c, out float v0, out float v1, out float v2, out float v3);
                        o0[r] = v0; o1[r] = v1; o2[r] = v2; o3[r] = v3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ6K_Q8K_4In(row, sc0, sc1, sc2, sc3, cols, out float v0, out float v1, out float v2, out float v3);
                        output0[r] = v0; output1[r] = v1; output2[r] = v2; output3[r] = v3;
                    }
                }
                break;
            }
            case DType.Q8_0:
            {
                // Same rationale as the MatVec2In Q8_0 case (issue #417): no expensive
                // unpack to amortize, so four sequential DotQ8_0 per row — one weight-row
                // read per four tokens, bit-identical to four single MatVec calls and to
                // DispatchDot4In's Q8_0 fallback (two 2In pairs → four single dots).
                int bpr = (cols / 32) * 34;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        o0[r] = DotQ8_0(row, i0, c);
                        o1[r] = DotQ8_0(row, i1, c);
                        o2[r] = DotQ8_0(row, i2, c);
                        o3[r] = DotQ8_0(row, i3, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        output0[r] = DotQ8_0(row, input0, cols);
                        output1[r] = DotQ8_0(row, input1, cols);
                        output2[r] = DotQ8_0(row, input2, cols);
                        output3[r] = DotQ8_0(row, input3, cols);
                    }
                }
                break;
            }
            case DType.Float32:
            {
                var m = (float*)weights;
                if (rows >= MinRowsForParallel)
                {
                    var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        float* row = m + (long)r * c;
                        o0[r] = DotF32(row, i0, c);
                        o1[r] = DotF32(row, i1, c);
                        o2[r] = DotF32(row, i2, c);
                        o3[r] = DotF32(row, i3, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = m + (long)r * cols;
                        output0[r] = DotF32(row, input0, cols);
                        output1[r] = DotF32(row, input1, cols);
                        output2[r] = DotF32(row, input2, cols);
                        output3[r] = DotF32(row, input3, cols);
                    }
                }
                break;
            }
            default:
                // Fallback: two MatVec2In pairs (each falls back per-dtype as needed).
                // Never worse than the prior pairwise path, still correct.
                MatVec2In(output0, output1, weights, input0, input1, rows, cols, dtype);
                MatVec2In(output2, output3, weights, input2, input3, rows, cols, dtype);
                break;
        }
    }

    private static void MatVecDequantFallback(float* output, byte* weights, float* input,
        int rows, int cols, DType dtype)
    {
        int blockSize = DTypeInfo.BlockSize(dtype);
        int bytesPerBlock = DTypeInfo.BytesPerBlock(dtype);
        int blocksPerRow = cols / blockSize;
        int bytesPerRow = blocksPerRow * bytesPerBlock;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            int bpr = bytesPerRow; int c = cols;
            var dt = dtype;
            Parallel.For(0, rows, s_parallelOpts, () =>
                (nint)NativeMemory.Alloc((nuint)(c * sizeof(float))),
                (r, _, bufPtr) =>
                {
                    float* rowBuf = (float*)bufPtr;
                    byte* rowPtr = w + (long)r * bpr;
                    Dequantize.ToFloat32(new ReadOnlySpan<byte>(rowPtr, bpr),
                        new Span<float>(rowBuf, c), dt, c);
                    outp[r] = DotF32(rowBuf, inp, c);
                    return bufPtr;
                },
                bufPtr => NativeMemory.Free((void*)bufPtr)
            );
        }
        else
        {
            float* rowBuf = (float*)NativeMemory.Alloc((nuint)(cols * sizeof(float)));
            try
            {
                for (int r = 0; r < rows; r++)
                {
                    byte* rowPtr = weights + (long)r * bytesPerRow;
                    Dequantize.ToFloat32(new ReadOnlySpan<byte>(rowPtr, bytesPerRow),
                        new Span<float>(rowBuf, cols), dtype, cols);
                    output[r] = DotF32(rowBuf, input, cols);
                }
            }
            finally { NativeMemory.Free(rowBuf); }
        }
    }

    // ================================================================
    //  F32 MatVec
    // ================================================================

    /// <summary>
    /// Opt-in ggml-compatible dot for <see cref="DType.Float16"/> weights: rounds the F32
    /// activation to fp16 (round-trip through <see cref="Half"/>) before multiplying against the
    /// weight, matching ggml's <c>vec_dot_type=GGML_TYPE_F16</c> pairing (<c>ggml_vec_dot_f16</c>
    /// quantizes/rounds the activation to half precision first, not a full-F32 dot). This engine's
    /// default (see <see cref="MatVecDequantFallback"/>) keeps the activation at full F32
    /// precision for strictly better accuracy; enable via <c>STINGRAY_GGML_F16_DOT=1</c> only when
    /// bit-parity comparison against llama.cpp is specifically wanted — see docs/bugstofix.md.
    /// </summary>
    public static readonly bool GgmlF16DotEnabled =
        Environment.GetEnvironmentVariable("STINGRAY_GGML_F16_DOT") == "1";

    public static void MatVecF16GgmlCompat(float* output, byte* weights, float* input, int rows, int cols)
    {
        float* rounded = stackalloc float[cols];
        for (int i = 0; i < cols; i++)
            rounded[i] = (float)(Half)input[i];

        Half* w = (Half*)weights;
        if (rows >= MinRowsForParallel)
        {
            var outp = output; var inp = rounded; var ww = weights; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, r =>
            {
                Half* rw = (Half*)(ww + (long)r * c * sizeof(ushort));
                float sum = 0f;
                for (int i = 0; i < c; i++)
                    sum += (float)rw[i] * inp[i];
                outp[r] = sum;
            });
        }
        else
        {
            for (int r = 0; r < rows; r++)
            {
                Half* rw = w + (long)r * cols;
                float sum = 0f;
                for (int i = 0; i < cols; i++)
                    sum += (float)rw[i] * rounded[i];
                output[r] = sum;
            }
        }
    }

    public static void MatVecF32(float* output, float* matrix, float* input, int rows, int cols)
    {
        if (rows >= MinRowsForParallel)
        {
            var m = matrix; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotF32(m + (long)i * cols, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotF32(matrix + (long)i * cols, input, cols);
        }
    }

    // ================================================================
    //  Q4_K Fused MatVec
    // ================================================================

    // Iteration 7 (docs/perf-loop-progress.md): runtime toggle for the isolated-microbenchmark-vs-
    // real-CLI A/B on DotQ4K_Wide8, since the isolated microbenchmark gave unreliable/bimodal
    // timings that couldn't be trusted. Not intended to stay -- remove once that A/B is settled.
    internal static readonly bool UseWide8 =
        Environment.GetEnvironmentVariable("STINGRAY_MATVEC_WIDE8") == "1";

    public static void MatVecQ4K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 144;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            if (UseWide8)
                Parallel.For(0, rows, s_parallelOpts, i =>
                {
                    outp[i] = DotQ4K_Wide8(w + (long)i * bytesPerRow, inp, cols);
                });
            else
                Parallel.For(0, rows, s_parallelOpts, i =>
                {
                    outp[i] = DotQ4K(w + (long)i * bytesPerRow, inp, cols);
                });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ4K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q5_K Fused two-input dot (issue #30) — mirror of DotQ4K_2In for
    //  Q5_K weights (gate/up in 27B-MTP-Q5_K_M).
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ5K_2In(byte* row, float* input1, float* input2, int cols,
                                  out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx512F.IsSupported)
        {
            DotQ5K_2In_Avx512(row, input1, input2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ5K(row, input1, cols);
        sum2 = DotQ5K(row, input2, cols);
    }

    private static void DotQ5K_2In_Avx512(byte* row, float* input1, float* input2,
                                          int cols, int numBlocks,
                                          out float sum1, out float sum2)
    {
        var accLo1 = Vector512<float>.Zero;
        var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero;
        var accHi2 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16  = Vector512.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);
                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit u1 → q5Lo
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);

                    // High nibble + high bit u2 → q5Hi
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
    }

    // ================================================================
    //  Q4_K Fused two-input dot (issue #30) — decode each block ONCE
    //  in registers and dot against TWO inputs in the same loop pass.
    //  Halves both the dequant work AND the weight L2/DRAM reads vs
    //  two sequential DotQ4K calls — the actual bandwidth win that
    //  MatVec2In can't get from naive double-dispatch alone.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ4K_2In(byte* row, float* input1, float* input2, int cols,
                                  out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ4K_2In_Avx512(row, input1, input2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        // Fallback: two scalar/AVX2 dots (cache-friendly but no fused-dequant win).
        sum1 = DotQ4K(row, input1, cols);
        sum2 = DotQ4K(row, input2, cols);
    }

    private static void DotQ4K_2In_Avx512(byte* row, float* input1, float* input2,
                                          int cols, int numBlocks,
                                          out float sum1, out float sum2)
    {
        // Four accumulators: lo/hi nibble × input1/input2.
        var accLo1 = Vector512<float>.Zero;
        var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero;
        var accHi2 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    // Single byte→int load shared by both inputs.
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble: dequant once.
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    // FMA against both inputs.
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);

                    // Upper nibble: dequant once.
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
    }

    // ================================================================
    //  Q4_K Fused four-input dot (issue #114) — register-tiled extension
    //  of DotQ4K_2In: decode each block ONCE and FMA against FOUR inputs
    //  in the same pass. Amortizes the nibble unpack decode/4 instead of
    //  decode/2. Each input's lo/hi accumulators follow the single-input
    //  order exactly, so the result is bit-identical to four DotQ4K calls.
    // ================================================================

    public static void DotQ4K_4In(byte* row,
        float* input0, float* input1, float* input2, float* input3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ4K_4In_Avx512(row, input0, input1, input2, input3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        // Fallback: four single dots (no fused-dequant win, still correct).
        sum0 = DotQ4K(row, input0, cols);
        sum1 = DotQ4K(row, input1, cols);
        sum2 = DotQ4K(row, input2, cols);
        sum3 = DotQ4K(row, input3, cols);
    }

    private static void DotQ4K_4In_Avx512(byte* row,
        float* input0, float* input1, float* input2, float* input3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        // Two accumulators (lo/hi nibble) per input.
        var accLo0 = Vector512<float>.Zero; var accHi0 = Vector512<float>.Zero;
        var accLo1 = Vector512<float>.Zero; var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero; var accHi2 = Vector512<float>.Zero;
        var accLo3 = Vector512<float>.Zero; var accHi3 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    // Single byte→int load shared by all inputs.
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble: dequant once, FMA against all four inputs.
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    accLo0 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input0 + bo + l)), accLo0);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);
                    accLo3 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input3 + bo + l)), accLo3);

                    // Upper nibble: dequant once, FMA against all four inputs.
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi0 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input0 + bo + 32 + l)), accHi0);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                    accHi3 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input3 + bo + 32 + l)), accHi3);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        sum0 = HSum512(Avx512F.Add(accLo0, accHi0));
        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
        sum3 = HSum512(Avx512F.Add(accLo3, accHi3));
    }

    // ================================================================
    //  Q5_K Fused four-input dot (issue #209) — register-tiled extension
    //  of DotQ5K_2In: decode each block ONCE and FMA against FOUR inputs.
    //  Each input's lo/hi accumulator chain matches the single-input order
    //  exactly, so the result is bit-identical to four DotQ5K calls.
    // ================================================================

    public static void DotQ5K_4In(byte* row,
        float* input0, float* input1, float* input2, float* input3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ5K_4In_Avx512(row, input0, input1, input2, input3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ5K(row, input0, cols);
        sum1 = DotQ5K(row, input1, cols);
        sum2 = DotQ5K(row, input2, cols);
        sum3 = DotQ5K(row, input3, cols);
    }

    private static void DotQ5K_4In_Avx512(byte* row,
        float* input0, float* input1, float* input2, float* input3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        var accLo0 = Vector512<float>.Zero; var accHi0 = Vector512<float>.Zero;
        var accLo1 = Vector512<float>.Zero; var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero; var accHi2 = Vector512<float>.Zero;
        var accLo3 = Vector512<float>.Zero; var accHi3 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16  = Vector512.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);
                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit u1 → q5Lo, dequant once.
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo0 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input0 + bo + l)), accLo0);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);
                    accLo3 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input3 + bo + l)), accLo3);

                    // High nibble + high bit u2 → q5Hi, dequant once.
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi0 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input0 + bo + 32 + l)), accHi0);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                    accHi3 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input3 + bo + 32 + l)), accHi3);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        sum0 = HSum512(Avx512F.Add(accLo0, accHi0));
        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
        sum3 = HSum512(Avx512F.Add(accLo3, accHi3));
    }

    // ================================================================
    //  Q6_K Fused MatVec
    // ================================================================

    public static void MatVecQ6K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 210;
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var s = scratch; var outp = output; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ6K_Q8K(w + (long)i * bytesPerRow, s, c);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ6K_Q8K(weights + (long)i * bytesPerRow, scratch, cols);
        }
    }

    // ================================================================
    //  F32 Dot Product  (4-way unrolled FMA)
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    /// <summary>
    /// Four F32 dot products sharing one right-hand vector: <c>s[k] = dot(a[k], b)</c>.
    /// </summary>
    /// <remarks>
    /// <para>Written for prefill attention's score phase, which computes Q·K as one independent
    /// <see cref="DotF32"/> per (query token, KV position) pair — at 2431 tokens that is on the
    /// order of 10^8 dots per layer, each ending in its own horizontal reduction.</para>
    ///
    /// <para>Two savings over four <see cref="DotF32"/> calls:</para>
    /// <list type="number">
    /// <item><b>One pass over <paramref name="b"/> instead of four.</b> The key vector is loaded
    /// once and consumed by all four accumulators — the same amortisation the quantised
    /// <c>*_4In</c> kernels already get, which attention never had.</item>
    /// <item><b>One batched horizontal reduction instead of four.</b> Three <c>HorizontalAdd</c>s
    /// plus a lane fold produce all four sums, against ~4x that for four separate reductions. At
    /// headDim 64 the dot is only 8 vector FMAs, so the reduction is a large fraction of the work
    /// — this is the dominant term, not the loads.</item>
    /// </list>
    ///
    /// <para><b>CURRENTLY UNUSED — read before wiring it in.</b> It was measured in
    /// <c>PrefillCoreAttention</c> on 2026-08-02 and reverted, for two reasons:</para>
    /// <list type="number">
    /// <item><b>Worth less than expected.</b> ~6% of attention time, ~2% end-to-end. The score
    /// phase is not attention's bottleneck; the weighted-V accumulation is, because it does a
    /// read-modify-write of the whole output head per (token, KV position) while the score phase
    /// writes one float.</item>
    /// <item><b>It creates a numerics boundary.</b> This kernel sums with one accumulator over
    /// 8-element strides; <see cref="DotF32"/> uses four accumulators over 32-element strides. The
    /// results differ in the last bits. Because a 4-wide kernel can only cover <c>t + 4 &lt;= tn</c>,
    /// the tile remainder falls back to <see cref="DotF32"/> — so a token's arithmetic depends on
    /// how many tokens share its tile, hence on N, and chunked vs unchunked prefill of the same
    /// prompt disagree. Four <c>ContinuousBatchingTests</c> caught it.</item>
    /// </list>
    /// <para>To use it, every token must go through one kernel — either pad the tile remainder, or
    /// make this bit-identical to <see cref="DotF32"/> (which needs 4 accumulators per input, 16
    /// YMM registers, and separate reductions — removing most of the saving). Given the 2% ceiling,
    /// neither is currently worth it. Kept because it is correct, and because a restructured
    /// attention that batches uniformly could use it.</para>
    /// </remarks>
    public static void DotF32_4In(float* a0, float* a1, float* a2, float* a3, float* b, int n,
        out float s0, out float s1, out float s2, out float s3)
    {
        if (Fma.IsSupported && n >= 8)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var vb = Avx.LoadVector256(b + i);          // loaded once, used four times
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a0 + i), vb, acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a1 + i), vb, acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a2 + i), vb, acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a3 + i), vb, acc3);
            }

            // Batched 4-way reduction: hadd(acc0,acc1) and hadd(acc2,acc3) pair the lanes, a third
            // hadd interleaves them to [s0 s1 s2 s3 | s0 s1 s2 s3], and folding the two 128-bit
            // halves completes all four sums at once.
            var t01 = Avx.HorizontalAdd(acc0, acc1);
            var t23 = Avx.HorizontalAdd(acc2, acc3);
            var t = Avx.HorizontalAdd(t01, t23);
            var sums = Sse.Add(t.GetLower(), t.GetUpper());

            s0 = sums.GetElement(0); s1 = sums.GetElement(1);
            s2 = sums.GetElement(2); s3 = sums.GetElement(3);

            for (; i < n; i++)
            {
                float bv = b[i];
                s0 += a0[i] * bv; s1 += a1[i] * bv; s2 += a2[i] * bv; s3 += a3[i] * bv;
            }
            return;
        }

        s0 = DotF32(a0, b, n); s1 = DotF32(a1, b, n);
        s2 = DotF32(a2, b, n); s3 = DotF32(a3, b, n);
    }

    public static float DotF32(float* a, float* b, int n)
    {
        if (Fma.IsSupported && n >= 32)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            int i = 0;
            for (; i + 32 <= n; i += 32)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16), acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24), acc3);
            }
            acc0 = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));

            for (; i + 8 <= n; i += 8)
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);

            float sum = HSum256(acc0);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        {
            float sum = 0;
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
    }

    /// <summary>
    /// Widens 8 BF16 values to F32. BF16 is the top half of an F32, so this is a zero-extend to
    /// 32-bit lanes plus a 16-bit left shift — no F16C, which .NET does not expose anyway.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> WidenBf16(ushort* p) =>
        Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Sse2.LoadVector128((short*)p)), 16).AsSingle();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float WidenBf16(ushort v) => BitConverter.UInt32BitsToSingle((uint)v << 16);

    /// <summary>
    /// <see cref="DotF32"/> where <paramref name="b"/> is a BF16 KV-cache row, widened on load.
    /// </summary>
    /// <remarks>
    /// <para>The accumulator layout — four <see cref="Vector256{T}"/> partials over 32-wide steps,
    /// then the pairwise tree, then an 8-wide tail, then scalars — is a deliberate copy of
    /// <see cref="DotF32"/>. Summation order determines the result in floating point, so keeping
    /// the two identical means the F32 and BF16 stores differ only by the stored values' precision
    /// and not by the reduction, which is what makes an A/B between them interpretable.</para>
    ///
    /// <para>This is where the KV bandwidth win is realised: the loads are half as wide, and decode
    /// is memory-bound (see <c>PagedKvCache.s_kvBf16Store</c>).</para>
    /// </remarks>
    public static float DotF32Bf16(float* a, ushort* b, int n)
    {
        if (Avx2.IsSupported && Fma.IsSupported && n >= 32)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            int i = 0;
            for (; i + 32 <= n; i += 32)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), WidenBf16(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), WidenBf16(b + i + 8), acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), WidenBf16(b + i + 16), acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), WidenBf16(b + i + 24), acc3);
            }
            acc0 = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));

            for (; i + 8 <= n; i += 8)
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), WidenBf16(b + i), acc0);

            float sum = HSum256(acc0);
            for (; i < n; i++) sum += a[i] * WidenBf16(b[i]);
            return sum;
        }

        {
            float sum = 0;
            for (int i = 0; i < n; i++) sum += a[i] * WidenBf16(b[i]);
            return sum;
        }
    }

    /// <summary>Widens <paramref name="n"/> BF16 values to F32.</summary>
    /// <remarks>
    /// Use this only where the widened buffer is consumed MANY times — Flash-64 packs a 64x64 tile
    /// and then feeds it to a GEMM, so the cost amortises 64-fold and the F32 microkernels
    /// downstream stay untouched. For a streaming read consumed once, use <see cref="DotF32Bf16"/>
    /// instead: widening to scratch first was measured to lose the entire bandwidth win there
    /// (it is 25% SLOWER in-cache), because the extra store/reload pass costs what halving the
    /// DRAM traffic saves.
    /// </remarks>
    public static void WidenBf16ToF32(ushort* src, float* dst, int n)
    {
        int d = 0;
        if (Avx2.IsSupported)
            for (; d + 8 <= n; d += 8) Avx.Store(dst + d, WidenBf16(src + d));
        for (; d < n; d++) dst[d] = WidenBf16(src[d]);
    }

    /// <summary>Widens one BF16 value to F32 — the top-half-of-an-F32 shift.</summary>
    public static float Bf16ToF32(ushort v) => WidenBf16(v);

    /// <summary>
    /// <c>dst[d] += w * widen(v[d])</c> for <paramref name="n"/> elements — the weighted-V
    /// accumulation of attention against a BF16 value row.
    /// </summary>
    public static void AccumulateScaledBf16(float* dst, ushort* v, float w, int n)
    {
        int d = 0;
        if (Avx2.IsSupported && Fma.IsSupported && n >= 8)
        {
            var wv = Vector256.Create(w);
            for (; d + 8 <= n; d += 8)
                Avx.Store(dst + d, Fma.MultiplyAdd(wv, WidenBf16(v + d), Avx.LoadVector256(dst + d)));
        }
        for (; d < n; d++) dst[d] += w * WidenBf16(v[d]);
    }

    // ================================================================
    //  Q4_K Fused Dequant-Dot  (one row)
    // ================================================================

    public static float DotQ4K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
            return DotQ4K_Avx512(row, input, cols, numBlocks);

        if (!Fma.IsSupported)
            return DotQ4K_Scalar(row, input, cols);

        // Two independent accumulators to break FMA dependency chains
        var accLo = Vector256<float>.Zero;
        var accHi = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector256.Create(d * sc1);
                var negDm1 = Vector256.Create(-(dmin * m1));
                var d2 = Vector256.Create(d * sc2);
                var negDm2 = Vector256.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                // Interleaved: process both nibbles from same bytes, into separate accumulators
                for (int l = 0; l < 32; l += 8)
                {
                    var bytes = LoadBytes8(qs + qIdx + l);
                    var ints = Avx2.ConvertToVector256Int32(bytes);

                    // Lower nibble → accLo
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + l), accLo);

                    // Upper nibble → accHi (independent chain)
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + l), accHi);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        return HSum256(Avx.Add(accLo, accHi));
    }

    /// <summary>
    /// perf-loop-progress.md iteration 24 (Next-up item 4, "memory-level parallelism / grouped
    /// kernel for batched GEMM"): same algorithm and accumulator structure as <see cref="DotQ4K"/>,
    /// but processes TWO weight rows against the same input vector in one call, sharing every
    /// <c>Avx.LoadVector256(input + ...)</c> load between both rows' FMA chains instead of the
    /// caller re-loading the same input bytes once per <see cref="DotQ4K"/> call. Same math, same
    /// accumulation membership per row as calling <see cref="DotQ4K"/> twice -- verified bit-
    /// identical against that in <c>DotQ4K_2RowSeamTests</c> (not just close/tolerance, since the
    /// operation order per row is unchanged, only the input load is shared).
    /// </summary>
    public static void DotQ4K_2Row(byte* row0, byte* row1, float* input, int cols, out float out0, out float out1)
    {
        int numBlocks = cols / 256;
        if (!Fma.IsSupported)
        {
            out0 = DotQ4K_Scalar(row0, input, cols);
            out1 = DotQ4K_Scalar(row1, input, cols);
            return;
        }

        var accLo0 = Vector256<float>.Zero;
        var accHi0 = Vector256<float>.Zero;
        var accLo1 = Vector256<float>.Zero;
        var accHi1 = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x0 = row0 + b * 144;
            byte* x1 = row1 + b * 144;
            float d0 = HalfToFloat(x0[0], x0[1]);
            float dmin0 = HalfToFloat(x0[2], x0[3]);
            float d1v = HalfToFloat(x1[0], x1[1]);
            float dmin1 = HalfToFloat(x1[2], x1[3]);
            byte* sc0 = x0 + 4; byte* qs0 = x0 + 16;
            byte* sc1 = x1 + 4; byte* qs1 = x1 + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc0, out byte r0sc1, out byte r0m1);
                GetScaleMinK4(scIdx + 1, sc0, out byte r0sc2, out byte r0m2);
                GetScaleMinK4(scIdx, sc1, out byte r1sc1, out byte r1m1);
                GetScaleMinK4(scIdx + 1, sc1, out byte r1sc2, out byte r1m2);

                var r0d1 = Vector256.Create(d0 * r0sc1);
                var r0negDm1 = Vector256.Create(-(dmin0 * r0m1));
                var r0d2 = Vector256.Create(d0 * r0sc2);
                var r0negDm2 = Vector256.Create(-(dmin0 * r0m2));
                var r1d1 = Vector256.Create(d1v * r1sc1);
                var r1negDm1 = Vector256.Create(-(dmin1 * r1m1));
                var r1d2 = Vector256.Create(d1v * r1sc2);
                var r1negDm2 = Vector256.Create(-(dmin1 * r1m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 8)
                {
                    // Shared input loads -- the whole point of this variant: one read of each
                    // input slice serves both rows' FMA chains instead of two separate DotQ4K
                    // calls each reloading the same bytes.
                    var inLo = Avx.LoadVector256(input + bo + l);
                    var inHi = Avx.LoadVector256(input + bo + 32 + l);

                    var bytes0 = LoadBytes8(qs0 + qIdx + l);
                    var ints0 = Avx2.ConvertToVector256Int32(bytes0);
                    var lo0 = Avx2.And(ints0, mask0F);
                    var deqLo0 = Fma.MultiplyAdd(r0d1, Avx.ConvertToVector256Single(lo0), r0negDm1);
                    accLo0 = Fma.MultiplyAdd(deqLo0, inLo, accLo0);
                    var hi0 = Avx2.And(Avx2.ShiftRightLogical(ints0, 4), mask0F);
                    var deqHi0 = Fma.MultiplyAdd(r0d2, Avx.ConvertToVector256Single(hi0), r0negDm2);
                    accHi0 = Fma.MultiplyAdd(deqHi0, inHi, accHi0);

                    var bytes1 = LoadBytes8(qs1 + qIdx + l);
                    var ints1 = Avx2.ConvertToVector256Int32(bytes1);
                    var lo1 = Avx2.And(ints1, mask0F);
                    var deqLo1 = Fma.MultiplyAdd(r1d1, Avx.ConvertToVector256Single(lo1), r1negDm1);
                    accLo1 = Fma.MultiplyAdd(deqLo1, inLo, accLo1);
                    var hi1 = Avx2.And(Avx2.ShiftRightLogical(ints1, 4), mask0F);
                    var deqHi1 = Fma.MultiplyAdd(r1d2, Avx.ConvertToVector256Single(hi1), r1negDm2);
                    accHi1 = Fma.MultiplyAdd(deqHi1, inHi, accHi1);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        out0 = HSum256(Avx.Add(accLo0, accHi0));
        out1 = HSum256(Avx.Add(accLo1, accHi1));
    }

    /// <summary>
    /// Same algorithm as <see cref="DotQ4K"/>'s AVX2 path, widened from 2 independent FMA
    /// accumulators to 8 (perf-loop-progress.md iteration 7, from an external review's
    /// observation): Zen3's <c>vfmadd231ps</c> is 4-cycle latency, 2/cycle throughput, so
    /// saturating both FMA ports needs ~8 independent accumulator chains in flight, not 2 --
    /// the original's <c>l</c> loop (4 iterations per chunk) serializes all 4 into the SAME
    /// accLo/accHi register across every chunk and every block, one long dependency chain each,
    /// not 8 short independent ones. Splits the 4 `l` sub-iterations into 4 separate lo
    /// accumulators and 4 separate hi accumulators (8 total), each only ever written by ONE of
    /// the 4 `l` positions across every chunk/block -- a genuinely independent chain per
    /// accumulator, 4x shorter than before. Same math, same accumulation membership, different
    /// grouping -- floating-point addition is not associative, so the final sum can differ from
    /// <see cref="DotQ4K"/>'s by a ULP-level amount; verified against a hand-computed scalar
    /// reference (not against DotQ4K) and cross-checked for close agreement with DotQ4K on
    /// random data as a secondary sanity net, per this codebase's established discipline.
    /// </summary>
    public static float DotQ4K_Wide8(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        if (!Fma.IsSupported || !Avx2.IsSupported)
            return DotQ4K_Scalar(row, input, cols);

        var accLo0 = Vector256<float>.Zero;
        var accLo1 = Vector256<float>.Zero;
        var accLo2 = Vector256<float>.Zero;
        var accLo3 = Vector256<float>.Zero;
        var accHi0 = Vector256<float>.Zero;
        var accHi1 = Vector256<float>.Zero;
        var accHi2 = Vector256<float>.Zero;
        var accHi3 = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector256.Create(d * sc1);
                var negDm1 = Vector256.Create(-(dmin * m1));
                var d2 = Vector256.Create(d * sc2);
                var negDm2 = Vector256.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                // Unrolled l=0,8,16,24 -- one dedicated accumulator pair per l-slot instead of
                // reusing accLo/accHi across all 4, so each of the 8 accumulators below only
                // ever appears in ONE of these four call sites per chunk (across every chunk and
                // every block), giving 8 genuinely independent FMA chains instead of 2.
                {
                    var bytes = LoadBytes8(qs + qIdx);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo0 = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo), accLo0);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi0 = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32), accHi0);
                }
                {
                    var bytes = LoadBytes8(qs + qIdx + 8);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo1 = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + 8), accLo1);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi1 = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + 8), accHi1);
                }
                {
                    var bytes = LoadBytes8(qs + qIdx + 16);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo2 = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + 16), accLo2);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi2 = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + 16), accHi2);
                }
                {
                    var bytes = LoadBytes8(qs + qIdx + 24);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo3 = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + 24), accLo3);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi3 = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + 24), accHi3);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        var sumLo = Avx.Add(Avx.Add(accLo0, accLo1), Avx.Add(accLo2, accLo3));
        var sumHi = Avx.Add(Avx.Add(accHi0, accHi1), Avx.Add(accHi2, accHi3));
        return HSum256(Avx.Add(sumLo, sumHi));
    }

    private static float DotQ4K_Avx512(byte* row, float* input, int cols, int numBlocks)
    {
        var accLo = Vector512<float>.Zero;
        var accHi = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2 = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                // Process 16 elements per iteration (vs 8 with AVX2)
                for (int l = 0; l < 32; l += 16)
                {
                    // Load 16 quantized bytes → 16 int32s via vpmovzxbd
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble → accLo
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    accLo = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input + bo + l)), accLo);

                    // Upper nibble → accHi
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input + bo + 32 + l)), accHi);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        return HSum512(Avx512F.Add(accLo, accHi));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum512(Vector512<float> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        return HSum256(Avx.Add(lo, hi));
    }

    private static float DotQ4K_Scalar(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;
            int qIdx = 0, scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);
                float d1 = d * sc1, dm1 = dmin * m1;
                float d2 = d * sc2, dm2 = dmin * m2;
                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l++)
                {
                    acc += (d1 * (qs[qIdx + l] & 0xF) - dm1) * input[bo + l];
                    acc += (d2 * (qs[qIdx + l] >> 4) - dm2) * input[bo + 32 + l];
                }
                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q5_K Fused MatVec
    // ================================================================

    public static void MatVecQ5K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 176;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ5K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ5K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q5_K Fused Dequant-Dot  (one row)
    // ================================================================

    /// <summary>
    /// Fused Q5_K dequantize-dot product using AVX2 FMA.
    /// Q5_K block (176 bytes per 256 elements):
    ///   [0:1] FP16 d, [2:3] FP16 dmin, [4:15] scales (12 bytes),
    ///   [16:47] qh (32 bytes, 1 high bit per element), [48:175] ql (128 bytes, 4 bits).
    /// </summary>
    public static float DotQ5K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
            return DotQ5K_Avx512(row, input, cols, numBlocks);

        if (!Fma.IsSupported)
            return DotQ5K_Scalar(row, input, cols);

        var accLo = Vector256<float>.Zero;
        var accHi = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        var bit16 = Vector256.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0;
            int scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector256.Create(d * sc1);
                var negDm1 = Vector256.Create(-(dmin * m1));
                var d2 = Vector256.Create(d * sc2);
                var negDm2 = Vector256.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 8)
                {
                    // Load 8 ql bytes and extract nibbles
                    var bytes = LoadBytes8(ql + qIdx + l);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var loNibble = Avx2.And(ints, mask0F);
                    var hiNibble = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);

                    // Load 8 qh bytes and extract high bits for this chunk
                    var qhBytes = LoadBytes8(qh + l);
                    var qhInts = Avx2.ConvertToVector256Int32(qhBytes);

                    // High bit for low nibble: (qh & u1) != 0 → 16
                    var hLoMask = Avx2.And(qhInts, Vector256.Create((int)u1));
                    var hLo = Avx2.And(
                        Avx2.CompareGreaterThan(hLoMask, Vector256<int>.Zero),
                        bit16);
                    var q5Lo = Avx2.Add(loNibble, hLo);

                    // High bit for high nibble: (qh & u2) != 0 → 16
                    var hHiMask = Avx2.And(qhInts, Vector256.Create((int)u2));
                    var hHi = Avx2.And(
                        Avx2.CompareGreaterThan(hHiMask, Vector256<int>.Zero),
                        bit16);
                    var q5Hi = Avx2.Add(hiNibble, hHi);

                    // Dequant: d1 * q5Lo - dm1
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(q5Lo), negDm1);
                    accLo = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + l), accLo);

                    // Dequant: d2 * q5Hi - dm2
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(q5Hi), negDm2);
                    accHi = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + l), accHi);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        return HSum256(Avx.Add(accLo, accHi));
    }

    private static float DotQ5K_Avx512(byte* row, float* input, int cols, int numBlocks)
    {
        var accLo = Vector512<float>.Zero;
        var accHi = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16 = Vector512.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2 = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);

                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input + bo + l)), accLo);

                    // High nibble + high bit
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input + bo + 32 + l)), accHi);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        return HSum512(Avx512F.Add(accLo, accHi));
    }

    private static float DotQ5K_Scalar(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;
            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);
                float d1 = d * sc1, dm1 = dmin * m1;
                float d2 = d * sc2, dm2 = dmin * m2;
                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l++)
                {
                    int hLo = (qh[l] & u1) != 0 ? 16 : 0;
                    int hHi = (qh[l] & u2) != 0 ? 16 : 0;
                    acc += (d1 * ((ql[qIdx + l] & 0xF) + hLo) - dm1) * input[bo + l];
                    acc += (d2 * ((ql[qIdx + l] >> 4) + hHi) - dm2) * input[bo + 32 + l];
                }
                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q8_0 Fused MatVec — 32-element blocks, [d:FP16 | qs:32×int8].
    //  AVX2 path expands 32 int8 → 4× 8 f32 per block and FMAs against
    //  the f32 input. APEX-mixed quants (e.g. Carnice MoE) interleave
    //  Q8_0 with K-quants, so this lives next to DotQ4K/DotQ5K/DotQ6K
    //  for use by the routed-expert DispatchDot path and (issue #417)
    //  the MatVec/MatVecDual/MatVec2In/MatVec4In dispatchers.
    // ================================================================

    /// <summary>
    /// Fused Q8_0 mat-vec (issue #417): per-row <see cref="DotQ8_0"/>, mirroring
    /// <see cref="MatVecQ4K"/>. Replaces the dequant→DotF32 fallback that Q8_0
    /// previously took through <see cref="MatVec"/> — one pass over the weight
    /// bytes, no per-row F32 scratch. Accumulation order changes vs the old
    /// fallback (argmax-stable, same class as the #162 compute routing), and is
    /// identical to the batched paths' DispatchDot Q8_0 route, which re-admits
    /// Q8_0 to the #415 batched CPU prefill dtype lists.
    /// </summary>
    /// <summary>
    /// Q4_0 matrix-vector product using the fused <see cref="DotQ4_0"/>. Replaces the
    /// <c>MatVecDequantFallback</c> route, which materialized each weight row as fp32 into a scratch
    /// buffer before calling <c>DotF32</c> — an extra pass over the weights and 8x the bytes read
    /// (4 bytes per element instead of 4 bits) for every matmul.
    /// </summary>
    public static void MatVecQ4_0(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 32) * 18;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ4_0(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ4_0(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    public static void MatVecIq4Nl(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 32) * 18;
        int scratchBytes = Q8_0ScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8_0(input, cols, scratch);

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var s = scratch; var outp = output; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotIq4Nl_Q8_0(w + (long)i * bytesPerRow, s, c);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotIq4Nl_Q8_0(weights + (long)i * bytesPerRow, scratch, cols);
        }
    }

    public static void MatVecQ8_0(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 32) * 34;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ8_0(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ8_0(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q4_0 x Q8_0 batched-prefill family (item 4, batched half)
    // ================================================================

    /// <summary>
    /// Bytes of scratch one token's Q8_0-quantized activation row needs.
    /// Layout is 36 bytes per 32-element block: <c>[d:float32][qs:32 x int8]</c>.
    /// <para>The scale is stored as fp32, not fp16 as in an on-disk Q8_0 block. This scratch is
    /// internal and never serialized, so there is no format to match, and fp32 avoids both a
    /// half-precision round trip and the rounding it would add on top of the activation
    /// quantization that is already lossy.</para>
    /// </summary>
    public static int Q8_0ScratchBytes(int cols) => (cols / 32) * 36;

    /// <summary>
    /// Quantize one fp32 activation row to the 32-element Q8_0 blocks that
    /// <see cref="DotQ4_0_Q8_0"/> consumes. Symmetric absmax scaling, the same scheme
    /// <c>quantize_row_q8_0</c> uses.
    /// </summary>
    public static void QuantizeRowToQ8_0(float* input, int cols, byte* scratch)
    {
        int numBlocks = cols / 32;
        for (int b = 0; b < numBlocks; b++)
        {
            float* x = input + b * 32;
            byte* dst = scratch + b * 36;

            float amax = 0f;
            for (int i = 0; i < 32; i++)
            {
                float a = MathF.Abs(x[i]);
                if (a > amax) amax = a;
            }

            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;
            *(float*)dst = d;

            sbyte* qs = (sbyte*)(dst + 4);
            for (int i = 0; i < 32; i++)
                qs[i] = (sbyte)MathF.Round(x[i] * id);
        }
    }

    /// <summary>
    /// Q4_0 weight row x Q8_0-quantized activation row, integer dot.
    /// <para>The 4-bit weights are kept UNSIGNED (0..15) so <c>maddubs</c> can be used directly —
    /// it takes unsigned bytes on the left and signed on the right — and the -8 bias every Q4_0
    /// nibble carries is applied once per block as <c>-8 * sum(q8)</c> instead of per element.
    /// That identity is what makes this cheaper than dequantizing: sum((q4-8)*q8) =
    /// sum(q4*q8) - 8*sum(q8).</para>
    /// <para>Element order follows <c>Dequantize.DequantQ4_0</c>: the low nibbles are elements
    /// 0..15 and the high nibbles are elements 16..31, so the two halves pair with the two halves
    /// of the activation block and are NOT interleaved.</para>
    /// </summary>
    public static float DotQ4_0_Q8_0(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 32;
        if (!Avx2.IsSupported || !Ssse3.IsSupported)
            return DotQ4_0_Q8_0_Scalar(row, scratch, numBlocks);

        var nibbleMask = Vector128.Create((byte)0x0F);
        var ones = Vector128.Create((sbyte)1);
        float acc = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* wb = row + b * 18;
            byte* sb = scratch + b * 36;

            float d4 = HalfToFloat(wb[0], wb[1]);
            float d8 = *(float*)sb;

            var packed = Sse2.LoadVector128(wb + 2);
            var lowNib = Sse2.And(packed, nibbleMask);
            var highNib = Sse2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), nibbleMask);

            var q8lo = Sse2.LoadVector128((sbyte*)(sb + 4));        // elements 0..15
            var q8hi = Sse2.LoadVector128((sbyte*)(sb + 4 + 16));   // elements 16..31

            // maddubs: unsigned nibble x signed q8, pairwise-summed into 8 shorts.
            // Peak lane magnitude is 15*127*2 = 3810 per product pair, so the adds below
            // cannot overflow int16 (32767) before the widening.
            var p = Ssse3.MultiplyAddAdjacent(lowNib, q8lo);
            p = Sse2.Add(p, Ssse3.MultiplyAddAdjacent(highNib, q8hi));

            // -8 bias correction: sum every q8 in the block once.
            var s = Ssse3.MultiplyAddAdjacent(Vector128.Create((byte)1), q8lo);
            s = Sse2.Add(s, Ssse3.MultiplyAddAdjacent(Vector128.Create((byte)1), q8hi));

            int dot = HSumInt16To32(p) - 8 * HSumInt16To32(s);
            acc += d4 * d8 * dot;
        }
        return acc;
    }

    /// <summary>Widen 8 int16 lanes to int32 and horizontally sum them.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HSumInt16To32(Vector128<short> v)
    {
        var lo = Sse41.ConvertToVector128Int32(v);
        var hi = Sse41.ConvertToVector128Int32(Sse2.ShiftRightLogical128BitLane(v.AsByte(), 8).AsInt16());
        var s = Sse2.Add(lo, hi);
        s = Sse2.Add(s, Sse2.Shuffle(s, 0b_01_00_11_10));
        s = Sse2.Add(s, Sse2.Shuffle(s, 0b_10_11_00_01));
        return s.ToScalar();
    }

    /// <summary>Scalar reference for <see cref="DotQ4_0_Q8_0"/>; the parity oracle.</summary>
    private static float DotQ4_0_Q8_0_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        float acc = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* wb = row + b * 18;
            byte* sb = scratch + b * 36;
            float d4 = HalfToFloat(wb[0], wb[1]);
            float d8 = *(float*)sb;
            byte* qs = wb + 2;
            sbyte* q8 = (sbyte*)(sb + 4);
            int dot = 0;
            for (int j = 0; j < 16; j++)
            {
                dot += ((qs[j] & 0xF) - 8) * q8[j];
                dot += ((qs[j] >> 4) - 8) * q8[j + 16];
            }
            acc += d4 * d8 * dot;
        }
        return acc;
    }

    // ================================================================
    //  IQ4_NL · Q8_0 Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_iq4_nl_q8_0 exactly (examples/ggml/src/ggml-cpu/quants.c). Unlike
    // Dequantize.DequantIq4Nl's per-element float dequant-then-FMA path this file's other
    // callers use, ggml quantizes the activation to Q8_0 (32-element blocks, matching IQ4_NL's
    // own QK4_NL=32) rather than dotting against the raw F32 activation -- see this file's
    // DotQ2K_Q8K/DotQ3K_Q8K/DotQ6K_Q8K for the same pattern applied to the K-quant family, and
    // docs/bugstofix.md's 2026-08-21 deepseek2 investigation for why this matters for
    // greedy-parity: every quantized weight type in ggml is paired with an activation-precision-
    // reducing vec_dot_type, and IQ4_NL's is Q8_0, not a full-precision F32 dot.
    private static readonly sbyte[] s_iq4NlCodebook =
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    public static float DotIq4Nl_Q8_0(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 32;
        float acc = 0f;
        fixed (sbyte* cb = s_iq4NlCodebook)
        {
            for (int b = 0; b < numBlocks; b++)
            {
                byte* wb = row + b * 18;
                byte* sb = scratch + b * 36;
                float d4 = HalfToFloat(wb[0], wb[1]);
                float d8 = *(float*)sb;
                byte* qs = wb + 2;
                sbyte* q8 = (sbyte*)(sb + 4);

                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < 16; j++)
                {
                    sumi1 += q8[j] * cb[qs[j] & 0xF];
                    sumi2 += q8[j + 16] * cb[qs[j] >> 4];
                }
                acc += d4 * d8 * (sumi1 + sumi2);
            }
        }
        return acc;
    }

    /// <summary>
    /// Four activation rows against ONE Q4_0 weight row. The weight nibbles are unpacked once and
    /// reused across all four, which is the entire point of the batched prefill path — the weight
    /// matrix is read once per four tokens instead of four times.
    /// </summary>
    public static void DotQ4_0_Q8_0_4In(byte* row, byte* s0, byte* s1, byte* s2, byte* s3, int cols,
        out float o0, out float o1, out float o2, out float o3)
    {
        int numBlocks = cols / 32;
        if (!Avx2.IsSupported || !Ssse3.IsSupported)
        {
            o0 = DotQ4_0_Q8_0_Scalar(row, s0, numBlocks);
            o1 = DotQ4_0_Q8_0_Scalar(row, s1, numBlocks);
            o2 = DotQ4_0_Q8_0_Scalar(row, s2, numBlocks);
            o3 = DotQ4_0_Q8_0_Scalar(row, s3, numBlocks);
            return;
        }

        var nibbleMask = Vector128.Create((byte)0x0F);
        var onesB = Vector128.Create((byte)1);
        float a0 = 0f, a1 = 0f, a2 = 0f, a3 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* wb = row + b * 18;
            float d4 = HalfToFloat(wb[0], wb[1]);

            var packed = Sse2.LoadVector128(wb + 2);
            var lowNib = Sse2.And(packed, nibbleMask);
            var highNib = Sse2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), nibbleMask);

            a0 += d4 * BlockDot(lowNib, highNib, onesB, s0 + b * 36);
            a1 += d4 * BlockDot(lowNib, highNib, onesB, s1 + b * 36);
            a2 += d4 * BlockDot(lowNib, highNib, onesB, s2 + b * 36);
            a3 += d4 * BlockDot(lowNib, highNib, onesB, s3 + b * 36);
        }
        o0 = a0; o1 = a1; o2 = a2; o3 = a3;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float BlockDot(Vector128<byte> lowNib, Vector128<byte> highNib, Vector128<byte> onesB,
            byte* sb)
        {
            float d8 = *(float*)sb;
            var q8lo = Sse2.LoadVector128((sbyte*)(sb + 4));
            var q8hi = Sse2.LoadVector128((sbyte*)(sb + 4 + 16));
            var p = Ssse3.MultiplyAddAdjacent(lowNib, q8lo);
            p = Sse2.Add(p, Ssse3.MultiplyAddAdjacent(highNib, q8hi));
            var s = Ssse3.MultiplyAddAdjacent(onesB, q8lo);
            s = Sse2.Add(s, Ssse3.MultiplyAddAdjacent(onesB, q8hi));
            return d8 * (HSumInt16To32(p) - 8 * HSumInt16To32(s));
        }
    }

    /// <summary>
    /// Fused Q4_0 weight row x FP32 activation dot product — the Q4_0 sibling of
    /// <see cref="DotQ8_0"/>.
    /// <para>Q4_0 previously had NO fused CPU kernel of any kind (0 references in this file against
    /// 37 for Q4_K), so every Q4_0 matmul dequantized a whole tensor to fp32 through
    /// <c>Dequantize.DequantQ4_0</c> and then ran the generic float path — an entire extra pass over
    /// the weights, plus 8x the bytes touched, before a single multiply-add.</para>
    /// <para>Block layout, 18 bytes per 32 elements: <c>[d:FP16][qs:16 x uint8]</c>, two signed
    /// nibbles per byte, value = <c>(nibble - 8) * d</c>. Element order matters and is NOT
    /// interleaved: <c>qs[j]</c>'s LOW nibble is element <c>j</c> and its HIGH nibble is element
    /// <c>j + 16</c> (see <c>Dequantize.DequantQ4_0</c>, which is the authority this must match).
    /// That is why the high-nibble FMAs below read <c>inp + 16</c>, not interleaved offsets.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ4_0(byte* row, float* input, int cols)
    {
        const int QK = 32;
        const int bytesPerBlock = 18;
        int numBlocks = cols / QK;

        if (!Avx2.IsSupported || !Fma.IsSupported)
            return DotQ4_0_Scalar(row, input, cols, numBlocks);

        var acc = Vector256<float>.Zero;
        var nibbleMask = Vector128.Create((byte)0x0F);
        var eight = Vector128.Create((byte)8);

        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            var dvec = Vector256.Create(d);
            float* inp = input + b * QK;

            var packed = Sse2.LoadVector128(block + 2);            // 16 bytes = 32 nibbles

            // Low nibbles -> elements 0..15, high nibbles -> elements 16..31.
            // There is no byte-wise shift in SSE2/AVX2, so shift as u16 and mask.
            var lowNib = Sse2.And(packed, nibbleMask);
            var highNib = Sse2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), nibbleMask);

            // Bias by -8 with byte arithmetic: 0..15 minus 8 wraps to the correct two's-complement
            // sbyte for -8..7, so no widening is needed before the sign-extending convert below.
            var lowS = Sse2.Subtract(lowNib, eight).AsSByte();
            var highS = Sse2.Subtract(highNib, eight).AsSByte();

            AccumulateHalf(lowS, inp, dvec, ref acc);
            AccumulateHalf(highS, inp + 16, dvec, ref acc);
        }
        return HSum256(acc);

        // 16 sbytes -> two 8-wide f32 FMAs against 16 contiguous activations.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void AccumulateHalf(Vector128<sbyte> q, float* inp, Vector256<float> dvec,
            ref Vector256<float> acc)
        {
            var lo32 = Avx2.ConvertToVector256Int32(q);
            var hi32 = Avx2.ConvertToVector256Int32(
                Sse2.ShiftRightLogical128BitLane(q.AsByte(), 8).AsSByte());
            var loF = Avx.ConvertToVector256Single(lo32);
            var hiF = Avx.ConvertToVector256Single(hi32);
            acc = Fma.MultiplyAdd(Avx.Multiply(loF, dvec), Avx.LoadVector256(inp), acc);
            acc = Fma.MultiplyAdd(Avx.Multiply(hiF, dvec), Avx.LoadVector256(inp + 8), acc);
        }
    }

    /// <summary>
    /// Scalar reference for <see cref="DotQ4_0"/>. Mirrors <c>Dequantize.DequantQ4_0</c> element for
    /// element so parity tests have an independent oracle that shares no SIMD code.
    /// </summary>
    private static float DotQ4_0_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        const int QK = 32;
        const int bytesPerBlock = 18;
        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            byte* qs = block + 2;
            float* inp = input + b * QK;
            for (int j = 0; j < QK / 2; j++)
            {
                acc += ((qs[j] & 0xF) - 8) * d * inp[j];
                acc += ((qs[j] >> 4) - 8) * d * inp[j + QK / 2];
            }
        }
        return (float)acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ8_0(byte* row, float* input, int cols)
    {
        const int QK = 32;
        const int bytesPerBlock = 34;
        int numBlocks = cols / QK;

        if (!Fma.IsSupported)
            return DotQ8_0_Scalar(row, input, cols, numBlocks);

        var acc = Vector256<float>.Zero;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            var dvec = Vector256.Create(d);
            sbyte* qs = (sbyte*)(block + 2);
            float* inp = input + b * QK;

            // Two halves of 16 sbytes each. Each half: low 8 → 8 i32 → 8 f32,
            // high 8 → 8 i32 → 8 f32. FMA scaled-qs × input into the accumulator.
            for (int half = 0; half < 2; half++)
            {
                var qs16 = Sse2.LoadVector128((byte*)(qs + half * 16)).AsSByte();
                var lo32 = Avx2.ConvertToVector256Int32(qs16);
                var hi32 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(qs16.AsByte(), 8).AsSByte());
                var loF  = Avx.ConvertToVector256Single(lo32);
                var hiF  = Avx.ConvertToVector256Single(hi32);
                var inpLo = Avx.LoadVector256(inp + half * 16);
                var inpHi = Avx.LoadVector256(inp + half * 16 + 8);
                acc = Fma.MultiplyAdd(Avx.Multiply(loF, dvec), inpLo, acc);
                acc = Fma.MultiplyAdd(Avx.Multiply(hiF, dvec), inpHi, acc);
            }
        }
        return HSum256(acc);
    }

    private static float DotQ8_0_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        const int QK = 32;
        const int bytesPerBlock = 34;
        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            sbyte* qs = (sbyte*)(block + 2);
            float* inp = input + b * QK;
            float blockSum = 0;
            for (int i = 0; i < QK; i++)
                blockSum += qs[i] * inp[i];
            acc += d * blockSum;
        }
        return (float)acc;
    }

    // ================================================================
    //  Q3_K Fused MatVec
    // ================================================================

    public static void MatVecQ3K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 110;
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var s = scratch; var outp = output; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ3K_Q8K(w + (long)i * bytesPerRow, s, c);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ3K_Q8K(weights + (long)i * bytesPerRow, scratch, cols);
        }
    }

    /// <summary>
    /// Fused Q3_K dequant-dot with AVX2.
    /// Block = 110 bytes / 256 elements: [hmask:32][qs:64][scales:12][d:FP16].
    /// Uses aux[] uint32 scale unpacking matching ggml exactly.
    /// </summary>
    public static float DotQ3K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (!Fma.IsSupported)
            return DotQ3K_Scalar(row, input, cols, numBlocks);

        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;

        var acc = Vector256<float>.Zero;
        var mask03 = Vector256.Create(0x03);
        var four = Vector256.Create(4);
        int elemOff = 0;

        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);

            // Unpack scales using aux[] manipulation (matching ggml)
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            // Extract 16 scale bytes from aux
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32; // qs at byte 32
            byte* hm = x;       // hmask at byte 0
            int qOff = 0;
            int isIdx = 0;
            byte m = 1;

            for (int n = 0; n < 256; n += 128)
            {
                for (int j = 0; j < 4; j++)
                {
                    float dl = dAll * (scales[isIdx++] - 32);
                    var vDl = Vector256.Create(dl);

                    // First 16 elements
                    for (int l = 0; l < 16; l += 8)
                    {
                        var qBytes = LoadBytes8(qs + qOff + l);
                        var qInts = Avx2.ConvertToVector256Int32(qBytes);
                        var shifted = j switch {
                            0 => qInts,
                            1 => Avx2.ShiftRightLogical(qInts, 2),
                            2 => Avx2.ShiftRightLogical(qInts, 4),
                            _ => Avx2.ShiftRightLogical(qInts, 6),
                        };
                        var q2 = Avx2.And(shifted, mask03);

                        // High bit from hmask: subtract 4 if hmask bit is NOT set
                        var hmBytes = LoadBytes8(hm + l);
                        var hmInts = Avx2.ConvertToVector256Int32(hmBytes);
                        var hmBit = Avx2.And(hmInts, Vector256.Create((int)m));
                        // If bit set → 0, if not set → 4
                        var sub = Avx2.And(Avx2.CompareEqual(hmBit, Vector256<int>.Zero), four);
                        var q3 = Avx2.Subtract(q2, sub);

                        var deq = Avx.Multiply(vDl, Avx.ConvertToVector256Single(q3));
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff), acc);
                        elemOff += 8;
                    }

                    // Second 16 elements (qs + 16, hm + 16)
                    dl = dAll * (scales[isIdx++] - 32);
                    vDl = Vector256.Create(dl);

                    for (int l = 0; l < 16; l += 8)
                    {
                        var qBytes = LoadBytes8(qs + qOff + 16 + l);
                        var qInts = Avx2.ConvertToVector256Int32(qBytes);
                        var shifted = j switch {
                            0 => qInts,
                            1 => Avx2.ShiftRightLogical(qInts, 2),
                            2 => Avx2.ShiftRightLogical(qInts, 4),
                            _ => Avx2.ShiftRightLogical(qInts, 6),
                        };
                        var q2 = Avx2.And(shifted, mask03);

                        var hmBytes = LoadBytes8(hm + 16 + l);
                        var hmInts = Avx2.ConvertToVector256Int32(hmBytes);
                        var hmBit = Avx2.And(hmInts, Vector256.Create((int)m));
                        var sub = Avx2.And(Avx2.CompareEqual(hmBit, Vector256<int>.Zero), four);
                        var q3 = Avx2.Subtract(q2, sub);

                        var deq = Avx.Multiply(vDl, Avx.ConvertToVector256Single(q3));
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff), acc);
                        elemOff += 8;
                    }

                    m <<= 1;
                }
                qOff += 32;
            }
        }

        return HSum256(acc);
    }

    private static float DotQ3K_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float acc = 0;
        int elemOff = 0;
        Span<uint> aux = stackalloc uint[4];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);

            aux[0] = *(uint*)(x + 96); aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            byte* qs = x + 32; byte* hm = x;
            int qOff = 0; int isIdx = 0; byte m = 1;

            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    float dl = dAll * (scByte - 32); isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((qs[qOff + l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4);
                        acc += dl * q * input[elemOff++];
                    }
                    scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    dl = dAll * (scByte - 32); isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((qs[qOff + l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4);
                        acc += dl * q * input[elemOff++];
                    }
                    shift += 2; m <<= 1;
                }
                qOff += 32;
            }
        }
        return acc;
    }

    // ================================================================
    //  Q2_K Fused MatVec
    // ================================================================

    public static void MatVecQ2K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 84;
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var s = scratch; var outp = output; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ2K_Q8K(w + (long)i * bytesPerRow, s, c);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ2K_Q8K(weights + (long)i * bytesPerRow, scratch, cols);
        }
    }

    // ================================================================
    //  Q2_K Fused Dequant-Dot  (one row)
    // ================================================================

    /// <summary>
    /// Fused Q2_K dequant-dot with AVX2. Block = 84 bytes / 256 elements.
    /// Layout: [scales:16][qs:64][d:FP16][dmin:FP16].
    /// The 64 qs bytes are read 4 times with shifts 0,2,4,6 per 128-element group.
    /// </summary>
    public static float DotQ2K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (!Fma.IsSupported)
            return DotQ2K_Scalar(row, input, cols, numBlocks);

        var acc = Vector256<float>.Zero;
        var mask03 = Vector256.Create(0x03);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 84;
            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);
            byte* sc = x;       // scales at byte 0
            byte* qs = x + 16;  // qs at byte 16

            int qOff = 0;
            int isIdx = 0;
            for (int n = 0; n < 256; n += 128)
            {
                // Unrolled: 4 shifts (0, 2, 4, 6) as constants
                for (int j = 0; j < 4; j++)
                {
                    byte scByte = sc[isIdx++];
                    var dl = Vector256.Create(d * (scByte & 0xF));
                    var negMl = Vector256.Create(-(min * (scByte >> 4)));

                    for (int l = 0; l < 16; l += 8)
                    {
                        var bytes = LoadBytes8(qs + qOff + l);
                        var ints = Avx2.ConvertToVector256Int32(bytes);
                        // Shift by constant: j=0→0, j=1→2, j=2→4, j=3→6
                        var shifted = j switch {
                            0 => ints,
                            1 => Avx2.ShiftRightLogical(ints, 2),
                            2 => Avx2.ShiftRightLogical(ints, 4),
                            _ => Avx2.ShiftRightLogical(ints, 6),
                        };
                        var q = Avx2.And(shifted, mask03);
                        var deq = Fma.MultiplyAdd(dl, Avx.ConvertToVector256Single(q), negMl);
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff + n + j * 32 + l), acc);
                    }

                    scByte = sc[isIdx++];
                    dl = Vector256.Create(d * (scByte & 0xF));
                    negMl = Vector256.Create(-(min * (scByte >> 4)));

                    for (int l = 0; l < 16; l += 8)
                    {
                        var bytes = LoadBytes8(qs + qOff + 16 + l);
                        var ints = Avx2.ConvertToVector256Int32(bytes);
                        var shifted = j switch {
                            0 => ints,
                            1 => Avx2.ShiftRightLogical(ints, 2),
                            2 => Avx2.ShiftRightLogical(ints, 4),
                            _ => Avx2.ShiftRightLogical(ints, 6),
                        };
                        var q = Avx2.And(shifted, mask03);
                        var deq = Fma.MultiplyAdd(dl, Avx.ConvertToVector256Single(q), negMl);
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff + n + j * 32 + 16 + l), acc);
                    }
                }
                qOff += 32;
            }
            elemOff += 256;
        }

        return HSum256(acc);
    }

    private static float DotQ2K_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 84;
            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);
            byte* sc = x;
            byte* qs = x + 16;

            int qOff = 0;
            int isIdx = 0;
            int yOff = elemOff;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte scByte = sc[isIdx++];
                    float dl = d * (scByte & 0xF);
                    float ml = min * (scByte >> 4);
                    for (int l = 0; l < 16; l++)
                        acc += (dl * ((qs[qOff + l] >> shift) & 3) - ml) * input[yOff++];

                    scByte = sc[isIdx++];
                    dl = d * (scByte & 0xF);
                    ml = min * (scByte >> 4);
                    for (int l = 0; l < 16; l++)
                        acc += (dl * ((qs[qOff + l + 16] >> shift) & 3) - ml) * input[yOff++];

                    shift += 2;
                }
                qOff += 32;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q2_K · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_q2_K_q8_K exactly (examples/ggml/src/ggml-cpu/quants.c). Unlike
    // DotQ2K's per-element float dequant-then-FMA, this quantizes the activation to int8 ONCE
    // per super-block (via QuantizeRowToQ8K -- same scratch this file already builds for
    // DotQ6K_Q8K/DotQ3K_Q8K), then keeps the whole 256-element reduction in INTEGER (isum, isuml)
    // domain, applying the two FP scale corrections (dall, dmin) only once per super-block at
    // the very end. This is not merely "more precise" or "less precise" than the float path --
    // it is ggml's actual arithmetic, bit-for-bit reproducible, whereas the float path's
    // per-element rounding order can never exactly match it. See docs/bugstofix.md's 2026-08-21
    // deepseek2 investigation for the full derivation of why this matters for greedy-parity.
    public static float DotQ2K_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        float sumf = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 84;
            byte* sc = x;       // scales[16]
            byte* q2 = x + 16;  // qs[64]
            float dw = HalfToFloat(x[80], x[81]);
            float dminW = HalfToFloat(x[82], x[83]);
            float dy = dArr[b];

            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            int summs = 0;
            for (int j = 0; j < 16; j++)
                summs += bsums[j] * (sc[j] >> 4);

            float dall = dy * dw;
            float dmin = dy * dminW;

            int isum = 0;
            int isIdx = 0;
            int q2Off = 0;
            int q8Off = 0;
            for (int k = 0; k < 2; k++) // QK_K/128 = 256/128
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int d0 = sc[isIdx++] & 0xF;
                    int isuml = 0;
                    for (int l = 0; l < 16; l++) isuml += q8[q8Off + l] * ((q2[q2Off + l] >> shift) & 3);
                    isum += d0 * isuml;

                    int d1 = sc[isIdx++] & 0xF;
                    isuml = 0;
                    for (int l = 0; l < 16; l++) isuml += q8[q8Off + 16 + l] * ((q2[q2Off + 16 + l] >> shift) & 3);
                    isum += d1 * isuml;

                    shift += 2;
                    q8Off += 32;
                }
                q2Off += 32;
            }
            sumf += dall * isum - dmin * summs;
        }
        return sumf;
    }

    // ================================================================
    //  Q8_K Input Quantization (used by Q6_K dot for parity with ggml)
    // ================================================================
    // Scratch layout, one entry per super-block of 256 input floats (nb = cols/256):
    //   [0 .. nb*4):                          float d[nb]
    //   [nb*4 .. nb*4 + nb*256):              sbyte qs[nb*256]
    //   [nb*4 + nb*256 .. nb*4 + nb*256 + nb*32):  short bsums[nb*16]
    //
    // Mirrors ggml's block_q8_K but laid out as SoA so each array is contiguous
    // across super-blocks (cheaper to load in the dot kernel).

    public static int Q8KScratchBytes(int cols)
    {
        int nb = cols / 256;
        return nb * 4 + nb * 256 + nb * 32;
    }

    /// <summary>
    /// Quantize a row of float input to Q8_K format, mirroring ggml's
    /// quantize_row_q8_K_ref. Scale is iscale = -127/max where max is the
    /// signed element with largest |·|. Single FP rounding per element.
    /// </summary>
    public static void QuantizeRowToQ8K(float* input, int cols, byte* scratch)
    {
        int nb = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + nb * 4);
        short* bsumsArr = (short*)(scratch + nb * 4 + nb * 256);

        for (int b = 0; b < nb; b++)
        {
            float* x = input + b * 256;
            sbyte* qs = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            float max = 0f, amax = 0f;
            for (int j = 0; j < 256; j++)
            {
                float ax = MathF.Abs(x[j]);
                if (ax > amax) { amax = ax; max = x[j]; }
            }

            if (amax == 0f)
            {
                dArr[b] = 0f;
                for (int j = 0; j < 256; j++) qs[j] = 0;
                for (int j = 0; j < 16; j++) bsums[j] = 0;
                continue;
            }

            float iscale = -127.0f / max;
            for (int j = 0; j < 256; j++)
            {
                int v = (int)MathF.Round(iscale * x[j], MidpointRounding.ToEven);
                if (v > 127) v = 127;
                qs[j] = (sbyte)v;
            }
            for (int j = 0; j < 16; j++)
            {
                int sum = 0;
                for (int ii = 0; ii < 16; ii++) sum += qs[j * 16 + ii];
                bsums[j] = (short)sum;
            }
            dArr[b] = 1.0f / iscale;
        }
    }

    /// <summary>
    /// Quantizes N prompt activation tokens (input: [batchSize, cols]) into Q8_K scratch blocks in parallel.
    /// </summary>
    public static void QuantizePromptToQ8K(float* input, int batchSize, int cols, byte* scratchBase)
    {
        long bytesPerRow = Q8KScratchBytes(cols);
        Parallel.For(0, batchSize, n =>
        {
            QuantizeRowToQ8K(input + n * cols, cols, scratchBase + n * bytesPerRow);
        });
    }

    // ================================================================
    //  Q6_K · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_q6_K_q8_K. The crucial difference vs the legacy
    // dequant-FMA path is that the input is quantized to int8 once per super-
    // block (one FP rounding per input element), then the inner dot is done
    // entirely in int domain (u8·i8 → i16 via maddubs, ×i8 scale → i32 via
    // madd), with a single FP multiply by d_super = d_w * d_y at the end.
    // This collapses 256 per-element FP rounding steps to ~1 per super-block,
    // which matches what llama.cpp produces and removes the Q6_K direction
    // drift that caused the Qwen3.6-27B-MTP pos-12 argmax flip.

    public static float DotQ6K_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ6K_Q8K_Avx2(row, scratch, cols, numBlocks);

        return DotQ6K_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ6K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ6K_Q8K(row, scratch, cols);
    }

    private static float DotQ6K_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        float acc = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dy = dArr[b];
            float dSuper = dw * dy;

            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            // Integer accumulator over the whole super-block
            int sumi = 0;
            // -32 offset correction: sum(32 * sc[i] * bsums[i]) over 16 sub-groups
            int offsetCorr = 0;
            for (int g = 0; g < 16; g++)
                offsetCorr += (int)sc[g] * bsums[g];
            offsetCorr <<= 5;  // × 32

            int qlOff = 0, qhOff = 0, scBase = 0, qOff = 0;
            for (int half = 0; half < 2; half++)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16;
                    // Unsigned 6-bit reconstruction (no -32 offset; subtracted via offsetCorr)
                    int q1u = (ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4);
                    int q2u = (ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4);
                    int q3u = (ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4);
                    int q4u = (ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4);

                    sumi += (int)sc[scBase + isc] * q1u * q8[qOff + l];
                    sumi += (int)sc[scBase + isc + 2] * q2u * q8[qOff + 32 + l];
                    sumi += (int)sc[scBase + isc + 4] * q3u * q8[qOff + 64 + l];
                    sumi += (int)sc[scBase + isc + 6] * q4u * q8[qOff + 96 + l];
                }
                qOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }

            acc += dSuper * (sumi - offsetCorr);
        }
        return acc;
    }

    /// <summary>
    /// Expand a Q6_K super-block's sixteen int8 scales into the eight <c>Vector256&lt;short&gt;</c>
    /// vectors its dot loop multiplies by — vector <c>k</c> holding <c>scales[2k]</c> in lanes 0-7
    /// and <c>scales[2k+1]</c> in lanes 8-15.
    ///
    /// <para><b>Why this exists.</b> The previous shape built each vector inside the <c>j</c> loop as
    /// <c>Vector256.Create(Vector128.Create((short)sc[n]), Vector128.Create((short)sc[n+1]))</c>.
    /// From a <i>runtime scalar</i> that is a byte load, a sign-extend and a <c>vpbroadcastw</c>,
    /// twice, then a <c>vinsertf128</c> — roughly 5-7 instructions per scale, re-reading
    /// <c>sc[]</c> from memory on every iteration. llama.cpp's <c>ggml_vec_dot_q6_K_q8_K</c> spends
    /// <b>two</b>: one <c>pshufb</c> against a register-resident <c>scales</c>, one
    /// <c>vpmovsxbw</c>. That difference was measured as the bulk of a 1.71x isolated gap on a
    /// kernel otherwise instruction-for-instruction identical.</para>
    ///
    /// <para><b>Why the masks are written out longhand.</b> They must be JIT constants. The earlier
    /// attempt at this (documented as a 6.3x regression) held them in a
    /// <c>static readonly Vector128&lt;byte&gt;[]</c> — a managed array, so every access in the
    /// innermost loop paid a static-field load plus a bounds check, swamping the two instructions it
    /// saved. Written as literal <c>Vector128.Create</c> calls they fold into a data-section load
    /// with no array and no bounds check. Do not refactor these into a table.</para>
    ///
    /// <para><b>Bit-identical to what it replaces</b>, not merely close: <c>pshufb</c> yields
    /// <c>[sc[2k] x8, sc[2k+1] x8]</c> and <c>cvtepi8_epi16</c> sign-extends lane-for-lane, which is
    /// exactly the old pair-of-broadcasts. The kernel-bench checksum gate is what proves it.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q6KBuildScaleVectors(Vector128<sbyte> scales, Vector256<short>* dst)
    {
        dst[0] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)));
        dst[1] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)));
        dst[2] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5)));
        dst[3] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7)));
        dst[4] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9)));
        dst[5] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11)));
        dst[6] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13)));
        dst[7] = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, Vector128.Create(
            (sbyte)14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15)));
    }

    private static float DotQ6K_Q8K_Avx2(byte* row, byte* scratch, int cols, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc = Vector256<float>.Zero;

        // The eight per-16-element scale vectors of the current super-block. Built once per block
        // (see Q6KBuildScaleVectors) and read back as a plain 32-byte stack load inside the j loop,
        // instead of being reconstructed from scalar memory on every iteration. Hoisted out of the
        // block loop so the stackalloc happens once, not numBlocks times.
        Vector256<short>* scv = stackalloc Vector256<short>[8];

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper = dw * dArr[i];

            // q8sclsub = (bsums · scales_int16) << 5  →  8 int32
            // bsums: 16 int16. scales: 16 int8 → cvtepi8_epi16 → 16 int16.
            var q8sums = Vector256.LoadUnsafe(ref *(bsumsArr + i * 16));
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            Q6KBuildScaleVectors(scales128, scv);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);
            var q8sclsub = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);

            var sumi = Vector256<int>.Zero;
            sbyte* q8 = (sbyte*)(qsArr + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values
                var q4h_0 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(
                    Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15),
                    q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15),
                    q4h_3);

                var q8_0 = Vector256.LoadUnsafe(ref *(q8 + j * 128)).AsSByte();
                var q8_1 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 32)).AsSByte();
                var q8_2 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 64)).AsSByte();
                var q8_3 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 96)).AsSByte();

                // u8 × i8 → i16 pairs (no saturation: |u6×i8| ≤ 63×127 = 8001, pairs ≤ 16002)
                var p16_0 = Avx2.MultiplyAddAdjacent(q4_0, q8_0);
                var p16_1 = Avx2.MultiplyAddAdjacent(q4_1, q8_1);
                var p16_2 = Avx2.MultiplyAddAdjacent(q4_2, q8_2);
                var p16_3 = Avx2.MultiplyAddAdjacent(q4_3, q8_3);

                // Apply 8 per-16-element scales (2 per q4_k). Each scale broadcast
                // to 8 int16 lanes; madd pairs adjacent lanes → 4 int32 outputs
                // per q4_k all sharing the same scale within a 16-elem sub-group.
                int isc = j * 4;
                var sc16_0 = scv[isc + 0];
                var sc16_1 = scv[isc + 1];
                var sc16_2 = scv[isc + 2];
                var sc16_3 = scv[isc + 3];

                var s0 = Avx2.MultiplyAddAdjacent(sc16_0, p16_0);
                var s1 = Avx2.MultiplyAddAdjacent(sc16_1, p16_1);
                var s2 = Avx2.MultiplyAddAdjacent(sc16_2, p16_2);
                var s3 = Avx2.MultiplyAddAdjacent(sc16_3, p16_3);

                sumi = Avx2.Add(sumi, Avx2.Add(Avx2.Add(s0, s1), Avx2.Add(s2, s3)));
            }

            acc = Fma.MultiplyAdd(
                Vector256.Create(dSuper),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi, q8sclsub)),
                acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q3_K · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_q3_K_q8_K. Same int-domain strategy as
    // DotQ6K_Q8K: weights are reconstructed as unsigned 3-bit values
    // (qu ∈ [0,7]) and the signed offset is amortised across the super-
    // block via the Q8_K bsums. The per-sub-group dl factor in scalar
    // Q3_K is `dAll * (scales[is] - 32)`; here we bake the -32 into the
    // i8 scale so the inner dot stays in int domain.
    //
    // Decomposition:
    //   q3 = qu - 4    (qu = ((qs>>shift)&3) + 4*hmask_bit, in [0,7])
    //   dl * q3 * y    = dl*qu*y  -  4*dl*y
    //   Sum over a 16-element sub-group:
    //     dl * (Σ qu·y)  -  4 * dl * bsums_is
    //   With dl = dAll*(scale-32):
    //     dAll * [(scale-32)*Σ(qu·y)  -  4*(scale-32)*bsums_is]
    //   The second term, summed over all 16 sub-blocks, is the
    //   `q8sclsub` correction = ((bsums·scales_adj) << 2).
    //
    // The auto-on parity gap that #103 surfaced was NOT in this dot —
    // per-kernel int-domain reference matches ggml at 1e-4 rel. The gap
    // is in the Q8_K input quantization itself (per-256-element single
    // scale). The Q8_KS-input variant (Q3K_Q8KS) ships alongside and
    // is what the gated MoE path actually dispatches when both gates
    // resolve to on; see DotQ3K_Q8KS in this file for the per-32 scale
    // path and HybridGdnForwardPass for the routing.

    public static float DotQ3K_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ3K_Q8K_Avx2(row, scratch, numBlocks);

        return DotQ3K_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ3K_Q8K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ3K_Q8K(row, scratch, cols);
    }

    internal static float DotQ3K_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        float acc = 0f;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float dy = dArr[b];
            float dSuper = dAll * dy;

            // Unpack 16 6-bit scales via the ggml aux[] pattern.
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32;
            byte* hm = x;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            // -32 offset correction: 4 * Σ (scale-32) * bsums_is, scaled by dSuper.
            int offsetCorr = 0;
            for (int g = 0; g < 16; g++)
                offsetCorr += ((int)scales[g] - 32) * bsums[g];
            offsetCorr <<= 2; // × 4

            int sumi = 0;
            int qOff = 0;
            int isIdx = 0;
            int qOut = 0;
            byte m = 1;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int sc0 = (int)scales[isIdx++] - 32;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + l] >> shift) & 3) + ((hm[l] & m) != 0 ? 4 : 0);
                        sumi += sc0 * qu * q8[qOut + l];
                    }
                    int sc1 = (int)scales[isIdx++] - 32;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + 16 + l] >> shift) & 3) + ((hm[16 + l] & m) != 0 ? 4 : 0);
                        sumi += sc1 * qu * q8[qOut + 16 + l];
                    }
                    qOut += 32;
                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }

            acc += dSuper * (sumi - offsetCorr);
        }
        return acc;
    }

    private static float DotQ3K_Q8K_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float dSuper = dAll * dArr[b];

            // Unpack 16 6-bit scales via the ggml aux[] pattern.
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            // q8sclsub = (bsums · scales_adj) << 2  →  8 int32
            // scales_adj ∈ [-32, +31] fits in i8; bsums sub-group sums fit in i16.
            var q8sums = Vector256.LoadUnsafe(ref *(bsumsArr + b * 16));
            var scales128 = Vector128.LoadUnsafe(ref scales[0]);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);
            var q8sclsub = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums, scales16), 2);

            // hmask is shared across both halves; bit-plane indexed by (half*4 + j).
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));

            var sumi = Vector256<int>.Zero;
            sbyte* q8 = qsArr + b * 256;
            byte* qs = x + 32;

            // Two halves × four j-iterations. The qs/hm shift amounts are
            // selected via switch on j (and (half,j)) so each AVX2 shift sees
            // a compile-time-constant immediate (CA1857). Each j contributes
            // 32 unsigned 3-bit weights spanning two 16-element sub-groups.
            for (int half = 0; half < 2; half++)
            {
                // 32 packed qs bytes for this half (4 weights per byte via shifts 0,2,4,6)
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    // qlo = (qs_v >> shift) & 0x03  (per-byte low-2-bits extraction)
                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    // hbit = ((hm_v >> hbitPos) & 1) << 2   → 0 or 4 per byte
                    // hbitPos = half*4 + j  ∈ [0..7]
                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit); // u3 in [0,7] per byte

                    // q3u carries two 16-element sub-groups: lanes [0..15] and [16..31]
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + half * 128 + j * 32)).AsSByte();

                    // u3·i8 → i16 pairs (no saturation: |u3·i8| ≤ 7·127 = 889, pairs ≤ 1778)
                    var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);

                    // Two scale lanes — one per 16-element sub-group within q3u
                    int isc = half * 8 + 2 * j;
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scales[isc + 0]),
                        Vector128.Create((short)scales[isc + 1]));

                    var s = Avx2.MultiplyAddAdjacent(sc16, p16);
                    sumi = Avx2.Add(sumi, s);
                }
            }

            acc = Fma.MultiplyAdd(
                Vector256.Create(dSuper),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi, q8sclsub)),
                acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q8_0 · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Q8_0 is a 32-element / 34-byte block: [d:FP16 | qs:32×int8]. The
    // legacy DotQ8_0(float* input) path dequant-expands each block to 32
    // FP32 lanes and FMAs against the f32 input — that's 32 FP multiplies
    // and 4 widen-and-convert sequences per block (256 FP rounding events
    // per 256-element super-block).
    //
    // The Q8_K-input fusion keeps the inner dot entirely in int domain:
    //   - 32 i8·i8 products per sub-block via two VPMADDWD chains
    //     (16 i16 + 16 i16 → 8 i32 + 8 i32 → lane-add to 8 i32 partials)
    //   - one FP multiply per Q8_0 sub-block (d_w[sub] × 8-lane int partials)
    //   - one FP multiply per Q8_K super-block (d_y[b] × Σ_sub)
    // Eight Q8_0 weight blocks span one Q8_K super-block (8 × 32 = 256
    // elements), so we collapse 256 FP roundings to 9 (8 inner + 1 outer)
    // per super-block — same direction-of-improvement as DotQ6K_Q8K and
    // DotQ3K_Q8K. cols must be a multiple of 256 (every model dim in
    // the codebase already satisfies this).
    //
    // The Q8_K bsums region is intentionally unused: Q8_0 is signed-
    // symmetric with no -32 offset to amortise, so no `q8sclsub`
    // correction is needed (cf. DotQ6K/DotQ3K which both subtract a
    // bsums-based correction). The bsums bytes are dead weight in this
    // path — see briefing notes for rank-2 design.
    //
    // Dual-acc-chain VPMADDWD pattern: each Q8_0 sub-block reduces its
    // 32 i8·i8 products via two independent MultiplyAddAdjacent chains
    // (low 16 + high 16 of the sub-block), matching the throughput
    // template of DotQ6K_Q8K_Avx2 / DotQ3K_Q8K_Avx2.

    public static float DotQ8_0_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ8_0_Q8K_Avx2(row, scratch, numBlocks);

        return DotQ8_0_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ8_0_Q8K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ8_0_Q8K(row, scratch, cols);
    }

    internal static float DotQ8_0_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        // bsums region after qsArr is unused for Q8_0 — see header.

        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float dy = dArr[b];
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            float subAcc = 0f;
            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                int intDot = 0;
                for (int i = 0; i < 32; i++)
                    intDot += qw[i] * qy[i];

                subAcc += dw * intDot;
            }

            acc += dy * subAcc;
        }
        return (float)acc;
    }

    private static float DotQ8_0_Q8K_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);

        var acc = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            float dy = dArr[b];
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            // 8 inner i32 dots → scale by d_w[sub] into subAccF; one outer
            // FMA by d_y[b] folds in the Q8_K super-block scale (see header).
            var subAccF = Vector256<float>.Zero;
            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                // 32 i8 → two halves of 16 i8 → widen to i16. AVX2 path:
                // SSE2 load 128b → ConvertToVector256Int16 sign-extends.
                var qw_lo128 = Sse2.LoadVector128((byte*)qw).AsSByte();           // lanes 0..15
                var qw_hi128 = Sse2.LoadVector128((byte*)(qw + 16)).AsSByte();    // lanes 16..31
                var qy_lo128 = Sse2.LoadVector128((byte*)qy).AsSByte();
                var qy_hi128 = Sse2.LoadVector128((byte*)(qy + 16)).AsSByte();

                var qw_lo = Avx2.ConvertToVector256Int16(qw_lo128);
                var qw_hi = Avx2.ConvertToVector256Int16(qw_hi128);
                var qy_lo = Avx2.ConvertToVector256Int16(qy_lo128);
                var qy_hi = Avx2.ConvertToVector256Int16(qy_hi128);

                // Two independent VPMADDWD chains: i16·i16 → i32 pair-sum.
                // |i16·i16| ≤ 127·127 = 16129; pair ≤ 32258, no saturation.
                var p_lo = Avx2.MultiplyAddAdjacent(qw_lo, qy_lo); // 8 i32
                var p_hi = Avx2.MultiplyAddAdjacent(qw_hi, qy_hi); // 8 i32
                var p_sum = Avx2.Add(p_lo, p_hi);                  // 8 i32 partials

                // Scale this sub-block's 8 i32 partials by d_w[sub] and
                // accumulate into the super-block FP accumulator.
                var pF = Avx.ConvertToVector256Single(p_sum);
                subAccF = Fma.MultiplyAdd(Vector256.Create(dw), pF, subAccF);
            }

            // Fold in the Q8_K super-block input scale d_y[b].
            acc = Fma.MultiplyAdd(Vector256.Create(dy), subAccF, acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q8_KS — Q8_K with per-32-element sub-block scales (issue #107)
    // ================================================================
    // Same int8 qs as Q8_K, but eight FP scales per 256-element super-
    // block (one per 32-element sub-block) instead of a single per-256
    // scale. Each sub-block's iscale is computed from its own amax, so
    // sub-blocks of low dynamic range get higher resolution (qs fills
    // more of [-127, +127]).
    //
    // Motivation (validation log docs/q8k-validation-*.md): the Q8_K
    // per-256 scale loses precision on inputs with non-uniform magnitude
    // across the super-block (post-SiLU activations, attention outputs).
    // Per-kernel parity vs ggml matches at 1e-4 rel; the trunk drift
    // that flips occasional argmaxes lives entirely in the input
    // quantization step. Per-32 scales cut the quantization-noise
    // envelope ~4× on Carnice (validation envelope was ±13 pp with
    // plain Q8_K, drops to ±3 pp with Q8_KS). A per-16 variant matching
    // Q3_K's scale-lane granularity was tried; it shuffles FP rounding
    // noise to different prompts (mathreason and factual lose what
    // techexplain gains) and was not strictly better — see
    // bench-q8k-validation-per16.csv for the comparison. Per-32 is the
    // local optimum until a finer-grained approach (e.g. Q8_1 with
    // per-block min offset) is investigated.
    //
    // Scratch layout, one entry per 256-input-float super-block (nb = cols/256):
    //   [0 .. nb*32):                                float d[nb*8]    (per-32 scales)
    //   [nb*32 .. nb*32 + nb*256):                    sbyte qs[nb*256]
    //   [nb*32 + nb*256 .. nb*32 + nb*256 + nb*32):   short bsums[nb*16]  (per-16 sums, for Q3K -32 offset)
    //
    // Total: nb * 320 bytes (vs nb * 292 for Q8_K). The extra 28 B/sb
    // is comfortably under the routed-MoE expert-scratch budget.

    public static int Q8KSScratchBytes(int cols)
    {
        int nb = cols / 256;
        return nb * 32 + nb * 256 + nb * 32;
    }

    /// <summary>
    /// Quantize a row of float input to Q8_KS format (per-32-element
    /// sub-block scales). Each 32-element sub-block computes its own
    /// iscale = -127 / max_signed_amax_sub, single FP rounding per element.
    /// bsums[g] keeps the unscaled int-sum over each 16-element sub-group
    /// so the Q3_K -32 offset correction can fold across two adjacent
    /// sub-groups per sub-block.
    /// </summary>
    public static void QuantizeRowToQ8KS(float* input, int cols, byte* scratch)
    {
        int nb = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + nb * 32);
        short* bsumsArr = (short*)(scratch + nb * 32 + nb * 256);

        for (int b = 0; b < nb; b++)
        {
            float* x = input + b * 256;
            sbyte* qs = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;
            float* d = dArr + b * 8;

            for (int sub = 0; sub < 8; sub++)
            {
                float max = 0f, amax = 0f;
                for (int j = 0; j < 32; j++)
                {
                    float ax = MathF.Abs(x[sub * 32 + j]);
                    if (ax > amax) { amax = ax; max = x[sub * 32 + j]; }
                }

                if (amax == 0f)
                {
                    d[sub] = 0f;
                    for (int j = 0; j < 32; j++) qs[sub * 32 + j] = 0;
                }
                else
                {
                    float iscale = -127.0f / max;
                    for (int j = 0; j < 32; j++)
                    {
                        int v = (int)MathF.Round(iscale * x[sub * 32 + j], MidpointRounding.ToEven);
                        if (v > 127) v = 127;
                        qs[sub * 32 + j] = (sbyte)v;
                    }
                    d[sub] = 1.0f / iscale;
                }
            }

            for (int g = 0; g < 16; g++)
            {
                int sum = 0;
                for (int ii = 0; ii < 16; ii++) sum += qs[g * 16 + ii];
                bsums[g] = (short)sum;
            }
        }
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Same int-domain strategy as Q3K_Q8K but each 32-element sub-block
    // (= 2 Q3_K 16-element sub-groups) has its own FP scale d_y[sub] in
    // place of the single per-super-block d_y. Per-sub-block FMA pattern
    // is identical to DotQ8_0_Q8K (which already accumulates per-sub),
    // so the extra cost is 7 extra FP FMAs per super-block — invisible
    // against the inner u3·i8/i16·i16 work.

    public static float DotQ3K_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ3K_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ3K_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ3K_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ3K_Q8KS(row, scratch, cols);
    }

    internal static float DotQ3K_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        float acc = 0f;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub = dArr + b * 8;

            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32;
            byte* hm = x;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            int qOff = 0;
            int isIdx = 0;
            int qOut = 0;
            byte m = 1;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;
                    int sc0 = (int)scales[isIdx++] - 32;
                    int sub0 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + l] >> shift) & 3) + ((hm[l] & m) != 0 ? 4 : 0);
                        sub0 += qu * q8[qOut + l];
                    }
                    int sc1 = (int)scales[isIdx++] - 32;
                    int sub1 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + 16 + l] >> shift) & 3) + ((hm[16 + l] & m) != 0 ? 4 : 0);
                        sub1 += qu * q8[qOut + 16 + l];
                    }

                    int subInt = sc0 * sub0 + sc1 * sub1
                               - 4 * (sc0 * bsums[isIdx - 2] + sc1 * bsums[isIdx - 1]);
                    acc += (dAll * dSub[sub]) * subInt;

                    qOut += 32;
                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }
        }
        return acc;
    }

    private static float DotQ3K_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub = dArr + b * 8;

            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums = bsumsArr + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8 = qsArr + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);

                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + half * 128 + j * 32)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);

                    // Per-sub-block offset correction folded into lane 0:
                    //   sub_corr = 4 * (scA * bsums[isc] + scB * bsums[isc+1])
                    int subCorr = ((int)scA * bsums[isc] + (int)scB * bsums[isc + 1]) << 2;
                    sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);

                    var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                    float scaleSub = dAll * dSub[sub];
                    acc = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc);
                }
            }
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product — two-input dequant-once (issue #112)
    // ================================================================
    // Decodes the Q3_K weight row ONCE (the 3-bit unpack + 6-bit scale decode is
    // the expensive part) and dots it against two Q8_KS-prepacked inputs. Each
    // input's accumulation is byte-for-byte identical to <see cref="DotQ3K_Q8KS"/>
    // — same sub-block order, same int MAdd / offset-correction / FP FMA chain —
    // so it is bit-identical to two single dots. Used by the batched routed-MoE
    // path to amortize the unpack across token pairs routing to the same expert.
    public static void DotQ3K_Q8KS_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ3K_Q8KS_2In_Avx2(row, scratch1, scratch2, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ3K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ3K_Q8KS_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ3K_Q8KS_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int numBlocks, out float sum1, out float sum2)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub1 = dArr1 + b * 8;
            float* dSub2 = dArr2 + b * 8;

            // Scales decode (shared between both inputs).
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums1 = bsumsArr1 + b * 16;
            short* bsums2 = bsumsArr2 + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8a = qsArr1 + b * 256;
            sbyte* q8b = qsArr2 + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);   // shared weight quants

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    // Input 1 — same accumulation as the single-input kernel.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8a + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums1[isc] + (int)scB * bsums1[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub1[sub];
                        acc1 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc1);
                    }
                    // Input 2 — reuses decoded q3u / sc16.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8b + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums2[isc] + (int)scB * bsums2[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub2[sub];
                        acc2 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc2);
                    }
                }
            }
        }
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product — four-input dequant-once (issue #114)
    // ================================================================
    // Generalizes <see cref="DotQ3K_Q8KS_2In"/> to a register-tiled tile of FOUR
    // Q8_KS-prepacked inputs: the 3-bit unpack + 6-bit scale decode is done ONCE
    // per sub-block and reused across all four inputs, so the (dominant) weight
    // decode is amortized decode/4 instead of decode/2. Each input's accumulation
    // is byte-for-byte identical to <see cref="DotQ3K_Q8KS"/> — same sub-block
    // order, same int MAdd / offset-correction / FP FMA chain — so the result is
    // bit-identical to four single dots. Used by the batched routed-MoE path to
    // amortize the unpack across token quads routing to the same expert.
    public static void DotQ3K_Q8KS_4In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ3K_Q8KS_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ3K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ3K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ3K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ3K_Q8KS_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ3K_Q8KS_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub0 = dArr0 + b * 8;
            float* dSub1 = dArr1 + b * 8;
            float* dSub2 = dArr2 + b * 8;
            float* dSub3 = dArr3 + b * 8;

            // Scales decode (shared between all four inputs).
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums0 = bsumsArr0 + b * 16;
            short* bsums1 = bsumsArr1 + b * 16;
            short* bsums2 = bsumsArr2 + b * 16;
            short* bsums3 = bsumsArr3 + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8_0 = qsArr0 + b * 256;
            sbyte* q8_1 = qsArr1 + b * 256;
            sbyte* q8_2 = qsArr2 + b * 256;
            sbyte* q8_3 = qsArr3 + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);   // shared weight quants

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    // Each input — same accumulation as the single-input kernel,
                    // reusing the decoded q3u / sc16.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_0 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums0[isc] + (int)scB * bsums0[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub0[sub];
                        acc0 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc0);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_1 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums1[isc] + (int)scB * bsums1[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub1[sub];
                        acc1 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc1);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_2 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums2[isc] + (int)scB * bsums2[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub2[sub];
                        acc2 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc2);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_3 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums3[isc] + (int)scB * bsums3[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub3[sub];
                        acc3 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc3);
                    }
                }
            }
        }
        sum0 = HSum256(acc0);
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
        sum3 = HSum256(acc3);
    }

    /// <summary>One Q8_KS input's contribution to a Q3_K sub-block, using the already-decoded
    /// weight trits (<paramref name="q3u"/>) and per-pair scales -- factored out of the 4-input
    /// kernel's inline blocks so the 8-input widening below doesn't duplicate this logic.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q3KAccumInput(sbyte* q8, int half, int j, Vector256<byte> q3u,
        Vector256<short> sc16, sbyte scA, sbyte scB, short* bsums, int isc,
        float dAll, float* dSub, int sub, ref Vector256<float> acc)
    {
        var q8_v = Vector256.LoadUnsafe(ref *(q8 + half * 128 + j * 32)).AsSByte();
        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
        int subCorr = ((int)scA * bsums[isc] + (int)scB * bsums[isc + 1]) << 2;
        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
        float scaleSub = dAll * dSub[sub];
        acc = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc);
    }

    // ================================================================
    //  Q3_K · Q8_KS Fused eight-input dot — widens DotQ3K_Q8KS_4In to 8 inputs, matching
    //  DotQ4K_Q8KS_8In/DotQ6K_Q8K_8In's already-shipped widening pattern.
    // ================================================================

    public static void DotQ3K_Q8KS_8In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ3K_Q8KS_8In_Avx2(row, scratch0, scratch1, scratch2, scratch3,
                scratch4, scratch5, scratch6, scratch7, numBlocks,
                out sum0, out sum1, out sum2, out sum3, out sum4, out sum5, out sum6, out sum7);
            return;
        }
        sum0 = DotQ3K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ3K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ3K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ3K_Q8KS_Scalar(row, scratch3, numBlocks);
        sum4 = DotQ3K_Q8KS_Scalar(row, scratch4, numBlocks);
        sum5 = DotQ3K_Q8KS_Scalar(row, scratch5, numBlocks);
        sum6 = DotQ3K_Q8KS_Scalar(row, scratch6, numBlocks);
        sum7 = DotQ3K_Q8KS_Scalar(row, scratch7, numBlocks);
    }

    private static void DotQ3K_Q8KS_8In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr0 = (float*)scratch0; sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32); short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1; sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32); short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2; sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32); short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3; sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32); short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);
        float* dArr4 = (float*)scratch4; sbyte* qsArr4 = (sbyte*)(scratch4 + numBlocks * 32); short* bsumsArr4 = (short*)(scratch4 + numBlocks * 32 + numBlocks * 256);
        float* dArr5 = (float*)scratch5; sbyte* qsArr5 = (sbyte*)(scratch5 + numBlocks * 32); short* bsumsArr5 = (short*)(scratch5 + numBlocks * 32 + numBlocks * 256);
        float* dArr6 = (float*)scratch6; sbyte* qsArr6 = (sbyte*)(scratch6 + numBlocks * 32); short* bsumsArr6 = (short*)(scratch6 + numBlocks * 32 + numBlocks * 256);
        float* dArr7 = (float*)scratch7; sbyte* qsArr7 = (sbyte*)(scratch7 + numBlocks * 32); short* bsumsArr7 = (short*)(scratch7 + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc0 = Vector256<float>.Zero; var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero; var acc3 = Vector256<float>.Zero;
        var acc4 = Vector256<float>.Zero; var acc5 = Vector256<float>.Zero;
        var acc6 = Vector256<float>.Zero; var acc7 = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub0 = dArr0 + b * 8; float* dSub1 = dArr1 + b * 8;
            float* dSub2 = dArr2 + b * 8; float* dSub3 = dArr3 + b * 8;
            float* dSub4 = dArr4 + b * 8; float* dSub5 = dArr5 + b * 8;
            float* dSub6 = dArr6 + b * 8; float* dSub7 = dArr7 + b * 8;

            // Scales decode (shared between all eight inputs).
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums0 = bsumsArr0 + b * 16; short* bsums1 = bsumsArr1 + b * 16;
            short* bsums2 = bsumsArr2 + b * 16; short* bsums3 = bsumsArr3 + b * 16;
            short* bsums4 = bsumsArr4 + b * 16; short* bsums5 = bsumsArr5 + b * 16;
            short* bsums6 = bsumsArr6 + b * 16; short* bsums7 = bsumsArr7 + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8_0 = qsArr0 + b * 256; sbyte* q8_1 = qsArr1 + b * 256;
            sbyte* q8_2 = qsArr2 + b * 256; sbyte* q8_3 = qsArr3 + b * 256;
            sbyte* q8_4 = qsArr4 + b * 256; sbyte* q8_5 = qsArr5 + b * 256;
            sbyte* q8_6 = qsArr6 + b * 256; sbyte* q8_7 = qsArr7 + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);   // shared weight quants

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    Q3KAccumInput(q8_0, half, j, q3u, sc16, scA, scB, bsums0, isc, dAll, dSub0, sub, ref acc0);
                    Q3KAccumInput(q8_1, half, j, q3u, sc16, scA, scB, bsums1, isc, dAll, dSub1, sub, ref acc1);
                    Q3KAccumInput(q8_2, half, j, q3u, sc16, scA, scB, bsums2, isc, dAll, dSub2, sub, ref acc2);
                    Q3KAccumInput(q8_3, half, j, q3u, sc16, scA, scB, bsums3, isc, dAll, dSub3, sub, ref acc3);
                    Q3KAccumInput(q8_4, half, j, q3u, sc16, scA, scB, bsums4, isc, dAll, dSub4, sub, ref acc4);
                    Q3KAccumInput(q8_5, half, j, q3u, sc16, scA, scB, bsums5, isc, dAll, dSub5, sub, ref acc5);
                    Q3KAccumInput(q8_6, half, j, q3u, sc16, scA, scB, bsums6, isc, dAll, dSub6, sub, ref acc6);
                    Q3KAccumInput(q8_7, half, j, q3u, sc16, scA, scB, bsums7, isc, dAll, dSub7, sub, ref acc7);
                }
            }
        }
        sum0 = HSum256(acc0); sum1 = HSum256(acc1); sum2 = HSum256(acc2); sum3 = HSum256(acc3);
        sum4 = HSum256(acc4); sum5 = HSum256(acc5); sum6 = HSum256(acc6); sum7 = HSum256(acc7);
    }

    // ================================================================
    //  Q8_0 · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Q8_0 block (32 elements / 34 bytes) naturally pairs 1:1 with one
    // Q8_KS sub-block. Per-sub-block dot is qw·qy summed in i32, scaled
    // by d_w[sub] × d_y[sub], accumulated across 8 sub-blocks per super-
    // block. The per-32 d_y dramatically reduces the FP-vs-quantized
    // envelope vs Q8_0_Q8K's per-256 d_y for activations with non-
    // uniform magnitude.

    public static float DotQ8_0_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ8_0_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ8_0_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ8_0_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ8_0_Q8KS(row, scratch, cols);
    }

    internal static float DotQ8_0_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);

        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                int intDot = 0;
                for (int i = 0; i < 32; i++)
                    intDot += qw[i] * qy[i];

                acc += (dw * dSub[sub]) * intDot;
            }
        }
        return (float)acc;
    }

    private static float DotQ8_0_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);

        var acc = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                var qw_lo128 = Sse2.LoadVector128((byte*)qw).AsSByte();
                var qw_hi128 = Sse2.LoadVector128((byte*)(qw + 16)).AsSByte();
                var qy_lo128 = Sse2.LoadVector128((byte*)qy).AsSByte();
                var qy_hi128 = Sse2.LoadVector128((byte*)(qy + 16)).AsSByte();

                var qw_lo = Avx2.ConvertToVector256Int16(qw_lo128);
                var qw_hi = Avx2.ConvertToVector256Int16(qw_hi128);
                var qy_lo = Avx2.ConvertToVector256Int16(qy_lo128);
                var qy_hi = Avx2.ConvertToVector256Int16(qy_hi128);

                var p_lo = Avx2.MultiplyAddAdjacent(qw_lo, qy_lo);
                var p_hi = Avx2.MultiplyAddAdjacent(qw_hi, qy_hi);
                var p_sum = Avx2.Add(p_lo, p_hi);

                var pF = Avx.ConvertToVector256Single(p_sum);
                float scaleSub = dw * dSub[sub];
                acc = Fma.MultiplyAdd(Vector256.Create(scaleSub), pF, acc);
            }
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q4_K · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Q4_K super-block = 256 elements / 144 bytes. Each 32-element
    // sub-block (8 per super-block) has its own 6-bit scale `sc` and
    // 6-bit min `m` (decoded by GetScaleMinK4) plus the per-super-block
    // FP `d` / `dmin`. The 4-bit weight nibble is UNSIGNED [0,15], so the
    // inner Σ nibble·q8 is the same u8·s8 (vpmaddubsw) pattern as the
    // Q3_K kernel — but the offset correction is `-dmin·m·Σq8` (a true
    // per-sub-block min, NOT Q3_K's constant -4). The per-32 activation
    // scale `dSub[sub]` folds into the per-sub-block FP FMA, so the dot
    // is dw-quantized in the int domain and FP-scaled per sub-block:
    //   acc += dSub·( d·sc·Σ(nibble·q8) − dmin·m·Σq8 ).
    // This replaces the slow f32-dequant fallback (DotQ4K) for routed
    // Q4_K MoE experts on the int8 path (Carnice: 9/41 expert layers).

    public static float DotQ4K_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ4K_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ4K_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ4K_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ4K_Q8KS(row, scratch, cols);
    }

    internal static float DotQ4K_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        float acc = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out byte m1);     // s0 (low nibbles)
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out byte m2); // s1 (high nibbles)

                // Sub-block s0 (32 elems at chunk*64).
                int sub0 = 0;
                int eo0 = chunk * 64;
                for (int l = 0; l < 32; l++)
                    sub0 += (qs[chunk * 32 + l] & 0x0F) * (int)q8[eo0 + l];
                int s0 = 2 * chunk;
                int bsum0 = (int)bsums[2 * s0] + (int)bsums[2 * s0 + 1];
                acc += dSub[s0] * (d * sc1 * sub0 - dmin * m1 * bsum0);

                // Sub-block s1 (32 elems at chunk*64+32).
                int sub1 = 0;
                int eo1 = chunk * 64 + 32;
                for (int l = 0; l < 32; l++)
                    sub1 += (qs[chunk * 32 + l] >> 4) * (int)q8[eo1 + l];
                int s1 = 2 * chunk + 1;
                int bsum1 = (int)bsums[2 * s1] + (int)bsums[2 * s1 + 1];
                acc += dSub[s1] * (d * sc2 * sub1 - dmin * m2 * bsum1);
            }
        }
        return acc;
    }

    private static float DotQ4K_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        // Vector accumulator + scalar min term, matching AccumQ4KInput exactly so this single-input
        // kernel stays BIT-IDENTICAL to the _4In/_8In batched kernels — that equality is the
        // contract MatMulBatchedQ8EquivalenceTests pins, and it is worth more than the few lines
        // saved by leaving this one scalar. It is also the same latency win: no per-sub-block
        // horizontal reduction.
        var facc = Vector256<float>.Zero;
        float minAcc = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            minAcc += MinCorrectionQ4K(bsums, dSub, LoadQ4KMins(sc), dmin, one16);

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out _);     // s0 (low nibbles)
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out _); // s1 (high nibbles)

                // 32 packed nibble-bytes for this chunk → low/high nibble halves.
                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;

                // Sub-block s0 (low nibbles · q8 at chunk*64).
                {
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(lo, q8_v);                 // u8·s8 → i16 pairs
                    var i32 = Avx2.MultiplyAddAdjacent(p16, one16);
                    facc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i32),
                                           Vector256.Create(dSub[s0] * (d * sc1)), facc);
                }
                // Sub-block s1 (high nibbles · q8 at chunk*64+32).
                {
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(hi, q8_v);
                    var i32 = Avx2.MultiplyAddAdjacent(p16, one16);
                    facc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i32),
                                           Vector256.Create(dSub[s1] * (d * sc2)), facc);
                }
            }
        }
        return Vector256.Sum(facc) - minAcc;
    }

    // ================================================================
    //  Q4_K · Q8_KS Dot Product — two/four-input dequant-once (#112/#114)
    // ================================================================
    // Decodes the Q4_K weight row ONCE (the nibble unpack + 6-bit scale/min
    // decode) and dots it against 2/4 Q8_KS-prepacked inputs. Each input's
    // accumulation is byte-for-byte identical to <see cref="DotQ4K_Q8KS"/> —
    // same sub-block order, same int MAdd / min-correction / FP FMA chain —
    // so it is bit-identical to N single dots. Used by the batched routed-MoE
    // path to amortize the unpack across tokens routing to the same expert.
    public static void DotQ4K_Q8KS_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ4K_Q8KS_2In_Avx2(row, scratch1, scratch2, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ4K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ4K_Q8KS_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ4K_Q8KS_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int numBlocks, out float sum1, out float sum2)
    {
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        // Vector accumulators, matching DotQ4K_Q8KS_Avx2 term-for-term (see AccumQ4KInput).
        var facc1 = Vector256<float>.Zero; float min1 = 0f;
        var facc2 = Vector256<float>.Zero; float min2 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub1 = dArr1 + b * 8;
            sbyte* q8a = qsArr1 + b * 256;
            short* bsums1 = bsumsArr1 + b * 16;
            float* dSub2 = dArr2 + b * 8;
            sbyte* q8b = qsArr2 + b * 256;
            short* bsums2 = bsumsArr2 + b * 16;

            var mins = LoadQ4KMins(sc);
            min1 += MinCorrectionQ4K(bsums1, dSub1, mins, dmin, one16);
            min2 += MinCorrectionQ4K(bsums2, dSub2, mins, dmin, one16);

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out _);
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out _);

                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));   // shared weight nibbles
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;
                float dsc1 = d * sc1;
                float dsc2 = d * sc2;

                // Input 1.
                {
                    var qlo = Vector256.LoadUnsafe(ref *(q8a + chunk * 64)).AsSByte();
                    var qhi = Vector256.LoadUnsafe(ref *(q8a + chunk * 64 + 32)).AsSByte();
                    var i0 = DotU8I8ToI32(lo, qlo, one16);
                    var i1 = DotU8I8ToI32(hi, qhi, one16);
                    facc1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Vector256.Create(dSub1[s0] * dsc1), facc1);
                    facc1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(dSub1[s1] * dsc2), facc1);
                }
                // Input 2 — reuses decoded lo/hi.
                {
                    var qlo = Vector256.LoadUnsafe(ref *(q8b + chunk * 64)).AsSByte();
                    var qhi = Vector256.LoadUnsafe(ref *(q8b + chunk * 64 + 32)).AsSByte();
                    var i0 = DotU8I8ToI32(lo, qlo, one16);
                    var i1 = DotU8I8ToI32(hi, qhi, one16);
                    facc2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Vector256.Create(dSub2[s0] * dsc1), facc2);
                    facc2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(dSub2[s1] * dsc2), facc2);
                }
            }
        }
        sum1 = Vector256.Sum(facc1) - min1;
        sum2 = Vector256.Sum(facc2) - min2;
    }

    public static void DotQ4K_Q8KS_4In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ4K_Q8KS_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ4K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ4K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ4K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ4K_Q8KS_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ4K_Q8KS_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        // Vector accumulators (one horizontal reduction per row at the end, not per sub-block);
        // the min-correction terms accumulate separately as scalars.
        var facc0 = Vector256<float>.Zero; float min0 = 0f;
        var facc1 = Vector256<float>.Zero; float min1 = 0f;
        var facc2 = Vector256<float>.Zero; float min2 = 0f;
        var facc3 = Vector256<float>.Zero; float min3 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub0 = dArr0 + b * 8;
            sbyte* q8_0 = qsArr0 + b * 256;
            short* bsums0 = bsumsArr0 + b * 16;
            float* dSub1 = dArr1 + b * 8;
            sbyte* q8_1 = qsArr1 + b * 256;
            short* bsums1 = bsumsArr1 + b * 16;
            float* dSub2 = dArr2 + b * 8;
            sbyte* q8_2 = qsArr2 + b * 256;
            short* bsums2 = bsumsArr2 + b * 16;
            float* dSub3 = dArr3 + b * 8;
            sbyte* q8_3 = qsArr3 + b * 256;
            short* bsums3 = bsumsArr3 + b * 16;

            var mins = LoadQ4KMins(sc);
            min0 += MinCorrectionQ4K(bsums0, dSub0, mins, dmin, one16);
            min1 += MinCorrectionQ4K(bsums1, dSub1, mins, dmin, one16);
            min2 += MinCorrectionQ4K(bsums2, dSub2, mins, dmin, one16);
            min3 += MinCorrectionQ4K(bsums3, dSub3, mins, dmin, one16);

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out _);
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out _);

                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));   // shared weight nibbles
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;
                float dsc1 = d * sc1;
                float dsc2 = d * sc2;

                AccumQ4KInput(lo, hi, one16, q8_0, dSub0, chunk, s0, s1, dsc1, dsc2, ref facc0);
                AccumQ4KInput(lo, hi, one16, q8_1, dSub1, chunk, s0, s1, dsc1, dsc2, ref facc1);
                AccumQ4KInput(lo, hi, one16, q8_2, dSub2, chunk, s0, s1, dsc1, dsc2, ref facc2);
                AccumQ4KInput(lo, hi, one16, q8_3, dSub3, chunk, s0, s1, dsc1, dsc2, ref facc3);
            }
        }
        sum0 = Vector256.Sum(facc0) - min0;
        sum1 = Vector256.Sum(facc1) - min1;
        sum2 = Vector256.Sum(facc2) - min2;
        sum3 = Vector256.Sum(facc3) - min3;
    }

    public static void DotQ4K_Q8KS_8In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ4K_Q8KS_8In_Avx2(row, scratch0, scratch1, scratch2, scratch3,
                scratch4, scratch5, scratch6, scratch7, numBlocks,
                out sum0, out sum1, out sum2, out sum3,
                out sum4, out sum5, out sum6, out sum7);
            return;
        }
        sum0 = DotQ4K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ4K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ4K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ4K_Q8KS_Scalar(row, scratch3, numBlocks);
        sum4 = DotQ4K_Q8KS_Scalar(row, scratch4, numBlocks);
        sum5 = DotQ4K_Q8KS_Scalar(row, scratch5, numBlocks);
        sum6 = DotQ4K_Q8KS_Scalar(row, scratch6, numBlocks);
        sum7 = DotQ4K_Q8KS_Scalar(row, scratch7, numBlocks);
    }

    private static void DotQ4K_Q8KS_8In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        float* dArr0 = (float*)scratch0; sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32); short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1; sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32); short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2; sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32); short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3; sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32); short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);
        float* dArr4 = (float*)scratch4; sbyte* qsArr4 = (sbyte*)(scratch4 + numBlocks * 32); short* bsumsArr4 = (short*)(scratch4 + numBlocks * 32 + numBlocks * 256);
        float* dArr5 = (float*)scratch5; sbyte* qsArr5 = (sbyte*)(scratch5 + numBlocks * 32); short* bsumsArr5 = (short*)(scratch5 + numBlocks * 32 + numBlocks * 256);
        float* dArr6 = (float*)scratch6; sbyte* qsArr6 = (sbyte*)(scratch6 + numBlocks * 32); short* bsumsArr6 = (short*)(scratch6 + numBlocks * 32 + numBlocks * 256);
        float* dArr7 = (float*)scratch7; sbyte* qsArr7 = (sbyte*)(scratch7 + numBlocks * 32); short* bsumsArr7 = (short*)(scratch7 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        // See AccumQ4KInput: vector accumulators, one horizontal reduction per row at the end.
        var facc0 = Vector256<float>.Zero; var facc1 = Vector256<float>.Zero;
        var facc2 = Vector256<float>.Zero; var facc3 = Vector256<float>.Zero;
        var facc4 = Vector256<float>.Zero; var facc5 = Vector256<float>.Zero;
        var facc6 = Vector256<float>.Zero; var facc7 = Vector256<float>.Zero;
        float min0 = 0f, min1 = 0f, min2 = 0f, min3 = 0f;
        float min4 = 0f, min5 = 0f, min6 = 0f, min7 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub0 = dArr0 + b * 8; sbyte* q8_0 = qsArr0 + b * 256; short* bsums0 = bsumsArr0 + b * 16;
            float* dSub1 = dArr1 + b * 8; sbyte* q8_1 = qsArr1 + b * 256; short* bsums1 = bsumsArr1 + b * 16;
            float* dSub2 = dArr2 + b * 8; sbyte* q8_2 = qsArr2 + b * 256; short* bsums2 = bsumsArr2 + b * 16;
            float* dSub3 = dArr3 + b * 8; sbyte* q8_3 = qsArr3 + b * 256; short* bsums3 = bsumsArr3 + b * 16;
            float* dSub4 = dArr4 + b * 8; sbyte* q8_4 = qsArr4 + b * 256; short* bsums4 = bsumsArr4 + b * 16;
            float* dSub5 = dArr5 + b * 8; sbyte* q8_5 = qsArr5 + b * 256; short* bsums5 = bsumsArr5 + b * 16;
            float* dSub6 = dArr6 + b * 8; sbyte* q8_6 = qsArr6 + b * 256; short* bsums6 = bsumsArr6 + b * 16;
            float* dSub7 = dArr7 + b * 8; sbyte* q8_7 = qsArr7 + b * 256; short* bsums7 = bsumsArr7 + b * 16;

            var mins = LoadQ4KMins(sc);
            min0 += MinCorrectionQ4K(bsums0, dSub0, mins, dmin, one16);
            min1 += MinCorrectionQ4K(bsums1, dSub1, mins, dmin, one16);
            min2 += MinCorrectionQ4K(bsums2, dSub2, mins, dmin, one16);
            min3 += MinCorrectionQ4K(bsums3, dSub3, mins, dmin, one16);
            min4 += MinCorrectionQ4K(bsums4, dSub4, mins, dmin, one16);
            min5 += MinCorrectionQ4K(bsums5, dSub5, mins, dmin, one16);
            min6 += MinCorrectionQ4K(bsums6, dSub6, mins, dmin, one16);
            min7 += MinCorrectionQ4K(bsums7, dSub7, mins, dmin, one16);

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out _);
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out _);

                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;
                float dsc1 = d * sc1;
                float dsc2 = d * sc2;

                AccumQ4KInput(lo, hi, one16, q8_0, dSub0, chunk, s0, s1, dsc1, dsc2, ref facc0);
                AccumQ4KInput(lo, hi, one16, q8_1, dSub1, chunk, s0, s1, dsc1, dsc2, ref facc1);
                AccumQ4KInput(lo, hi, one16, q8_2, dSub2, chunk, s0, s1, dsc1, dsc2, ref facc2);
                AccumQ4KInput(lo, hi, one16, q8_3, dSub3, chunk, s0, s1, dsc1, dsc2, ref facc3);
                AccumQ4KInput(lo, hi, one16, q8_4, dSub4, chunk, s0, s1, dsc1, dsc2, ref facc4);
                AccumQ4KInput(lo, hi, one16, q8_5, dSub5, chunk, s0, s1, dsc1, dsc2, ref facc5);
                AccumQ4KInput(lo, hi, one16, q8_6, dSub6, chunk, s0, s1, dsc1, dsc2, ref facc6);
                AccumQ4KInput(lo, hi, one16, q8_7, dSub7, chunk, s0, s1, dsc1, dsc2, ref facc7);
            }
        }
        sum0 = Vector256.Sum(facc0) - min0; sum1 = Vector256.Sum(facc1) - min1;
        sum2 = Vector256.Sum(facc2) - min2; sum3 = Vector256.Sum(facc3) - min3;
        sum4 = Vector256.Sum(facc4) - min4; sum5 = Vector256.Sum(facc5) - min5;
        sum6 = Vector256.Sum(facc6) - min6; sum7 = Vector256.Sum(facc7) - min7;
    }

    // One input's per-chunk accumulation for the Q4_K_Q8KS quad kernel. The two
    // sub-block terms (s0 then s1) are added to the running `acc` left-to-right,
    // identical to the single-input kernel's `acc += s0term; acc += s1term;`
    // ordering — so each input is bit-identical (not just FP-close) to a single
    // <see cref="DotQ4K_Q8KS"/> dot, the per-token k-independence the routed-MoE
    // byte-parity oracle relies on.
    /// <summary>
    /// Q4_K's per-super-block minimum-correction term, vectorised (perf-loop iterations 58/59).
    ///
    /// <para>The term is <c>sum_j dSub[j] * dmin * m_j * bsum_j</c> over the eight sub-blocks. It
    /// depends only on the activation scales, the activation <c>bsums</c> and the weight minimums —
    /// it never touches the quantised weight values — so it does not belong in the innermost nibble
    /// loop, where it previously ran as two int loads, two int adds and ~6 scalar float ops per
    /// sub-block per input. That is ~80 scalar ops per super-block per input, at one lane, issued
    /// in among ~256 vector ops it competes with for ports and registers. Iteration 58 measured
    /// deleting the term outright (incorrect, ablation only) at <b>+48.6% prefill</b>.</para>
    ///
    /// <para>All eight sub-blocks fit exactly one <see cref="Vector256{T}"/>: the sixteen int16
    /// <c>bsums</c> collapse to the eight per-sub-block sums using the same
    /// <c>madd_epi16</c>-against-ones the dot path already issues to widen, and <c>dSub</c> is
    /// already eight contiguous floats. ~6 vector ops plus one horizontal sum replace the ~80
    /// scalar ops.</para>
    ///
    /// <para><b>This changes FP summation order</b> — a tree reduction over eight products replaces
    /// eight sequential scalar adds. Q4_K's parity contract is that the 1/2/4/8-input kernels agree
    /// with EACH OTHER per token (the property the routed-MoE byte-parity oracle actually checks),
    /// not that any of them matches a frozen reference. So every Q4_K AVX2 kernel calls this one
    /// routine: a partial migration would silently violate that invariant rather than merely be
    /// incomplete.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MinCorrectionQ4K(short* bsums, float* dSub, Vector256<float> mins,
                                          float dmin, Vector256<short> one16)
    {
        // madd_epi16 against ones: pairs adjacent int16s and sums them, so lane j receives
        // bsums[2j] + bsums[2j+1] — exactly the per-sub-block bsum the scalar path computed.
        var bsum8 = Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *bsums), one16);
        var bf = Avx.ConvertToVector256Single(bsum8);
        var dv = Vector256.LoadUnsafe(ref *dSub);
        return dmin * Vector256.Sum(Avx.Multiply(Avx.Multiply(dv, mins), bf));
    }

    /// <summary>
    /// The eight 6-bit sub-block minimums of a Q4_K super-block, widened to float. Hoisted to once
    /// per super-block — <see cref="GetScaleMinK4"/> was previously re-decoding these inside the
    /// chunk loop purely to feed the scalar min term.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> LoadQ4KMins(byte* sc)
    {
        GetScaleMinK4(0, sc, out _, out byte m0);
        GetScaleMinK4(1, sc, out _, out byte m1);
        GetScaleMinK4(2, sc, out _, out byte m2);
        GetScaleMinK4(3, sc, out _, out byte m3);
        GetScaleMinK4(4, sc, out _, out byte m4);
        GetScaleMinK4(5, sc, out _, out byte m5);
        GetScaleMinK4(6, sc, out _, out byte m6);
        GetScaleMinK4(7, sc, out _, out byte m7);
        return Vector256.Create((float)m0, m1, m2, m3, m4, m5, m6, m7);
    }

    /// <summary>
    /// Accumulate one 64-element chunk (two Q4_K sub-blocks) of one token.
    ///
    /// <para>The int32 lane vector is scaled and accumulated IN VECTOR FORM rather than horizontally
    /// reduced here. The previous shape called <c>HSumI32_256</c> twice per chunk — a six-instruction
    /// serial chain ending in a vector-to-GPR move — and then did scalar float math, i.e. EIGHT
    /// horizontal reductions per 256-element super-block per token. Now each sub-block costs one
    /// <c>cvtepi32_ps</c> plus one FMA into an independent accumulator, and the caller does a single
    /// horizontal sum per row. Same arithmetic, but the per-sub-block latency chain disappears.</para>
    ///
    /// <para>Note we cannot go further and keep the whole super-block in integers the way
    /// llama.cpp's <c>ggml_gemm_q4_K_8x8_q8_K</c> does (folding the 6-bit scale in with
    /// <c>madd_epi16</c>): that works because Q8_K carries ONE activation scale per 256-element
    /// super-block, whereas our Q8_KS carries eight — one per sub-block — so a float multiply per
    /// sub-block is unavoidable without changing the activation format. See perf-loop iteration 37.</para>
    ///
    /// <para>The min-correction term is NOT here — it is hoisted to one vectorised pass per
    /// super-block in <see cref="MinCorrectionQ4K"/> (perf-loop iteration 63b).</para>
    /// </summary>
    /// <summary>
    /// The inner integer product every Q4_K kernel here needs: one int32 lane per four adjacent
    /// u8×i8 products. Three implementations, chosen widest-availability-last, and <b>bit-identical
    /// for this data</b> — so the choice is a throughput decision, not a numerics one.
    ///
    /// <para><b>Why the middle branch exists.</b> This used to gate on <c>AvxVnniInt8</c> alone,
    /// which is <c>vpdpbssd</c> — AVX-VNNI-INT8, present only on very recent parts (Zen 5,
    /// Granite Rapids / Arrow Lake class). Everything older, <i>including all of Zen 4 and Alder
    /// Lake through Raptor Lake</i>, fell all the way back to the two-instruction AVX2 chain even
    /// though those CPUs do have plain AVX-VNNI. The operands here are already unsigned weight
    /// nibbles × signed activations, which is exactly <c>vpdpbusd</c>'s signature, so the wider
    /// ISA needs no reinterpretation at all — the <c>AsSByte()</c> in the first branch is only
    /// needed because <c>vpdpbssd</c> is signed×signed.</para>
    ///
    /// <para><b>Why all three agree exactly.</b> Q4_K weight nibbles are unsigned 0-15 and Q8
    /// activations are |a| ≤ 127, so a <c>vpmaddubsw</c> pair peaks at 2·15·127 = 3810, far under
    /// the int16 saturation point — the AVX2 chain's one lossy step never actually loses anything
    /// on this data. Both VNNI forms accumulate straight into int32 and cannot saturate at all.
    /// Reinterpreting the nibbles as signed for <c>vpdpbssd</c> is safe for the same reason
    /// (0-15 is well below 128). Per the instruction tables, <c>VPDPBUSD</c> is 1 uop with 0.5
    /// reciprocal throughput on ports P01, against <c>VPMADDUBSW</c>'s 2 uops pinned to P0 — and
    /// the AVX2 form needs a second <c>vpmaddwd</c> on top of that.</para>
    /// </summary>
    /// <summary>
    /// Transposes one 8×8 float block with the standard AVX unpack/shuffle/permute sequence.
    /// Reads eight contiguous floats starting at <paramref name="srcOffset"/> from each of eight
    /// source rows, and writes eight destination rows of eight contiguous floats,
    /// <paramref name="dstRowStride"/> floats apart.
    ///
    /// <para>This is <b>pure data movement</b> — every output float is bit-identical to its input,
    /// so a caller swapping a scalar transpose for this one cannot change numerics. That is the
    /// property that makes it safe to put in a prefill hot path.</para>
    ///
    /// <para>24 shuffle-class uops replace 64 scalar loads and 64 scalar stores. The scalar form
    /// also reads its source column-wise — one float per row, striding a whole KV row between
    /// touches — which is exactly the access pattern the optimisation manuals warn generates no
    /// useful prefetch. Here each source row is read as one 32-byte load instead.</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void TransposeBlock8x8(float** srcRows, int srcOffset, float* dst, int dstRowStride)
    {
        var v0 = Avx.LoadVector256(srcRows[0] + srcOffset);
        var v1 = Avx.LoadVector256(srcRows[1] + srcOffset);
        var v2 = Avx.LoadVector256(srcRows[2] + srcOffset);
        var v3 = Avx.LoadVector256(srcRows[3] + srcOffset);
        var v4 = Avx.LoadVector256(srcRows[4] + srcOffset);
        var v5 = Avx.LoadVector256(srcRows[5] + srcOffset);
        var v6 = Avx.LoadVector256(srcRows[6] + srcOffset);
        var v7 = Avx.LoadVector256(srcRows[7] + srcOffset);

        var t0 = Avx.UnpackLow(v0, v1);
        var t1 = Avx.UnpackHigh(v0, v1);
        var t2 = Avx.UnpackLow(v2, v3);
        var t3 = Avx.UnpackHigh(v2, v3);
        var t4 = Avx.UnpackLow(v4, v5);
        var t5 = Avx.UnpackHigh(v4, v5);
        var t6 = Avx.UnpackLow(v6, v7);
        var t7 = Avx.UnpackHigh(v6, v7);

        var s0 = Avx.Shuffle(t0, t2, 0x44);
        var s1 = Avx.Shuffle(t0, t2, 0xEE);
        var s2 = Avx.Shuffle(t1, t3, 0x44);
        var s3 = Avx.Shuffle(t1, t3, 0xEE);
        var s4 = Avx.Shuffle(t4, t6, 0x44);
        var s5 = Avx.Shuffle(t4, t6, 0xEE);
        var s6 = Avx.Shuffle(t5, t7, 0x44);
        var s7 = Avx.Shuffle(t5, t7, 0xEE);

        Avx.Store(dst + 0 * dstRowStride, Avx.Permute2x128(s0, s4, 0x20));
        Avx.Store(dst + 1 * dstRowStride, Avx.Permute2x128(s1, s5, 0x20));
        Avx.Store(dst + 2 * dstRowStride, Avx.Permute2x128(s2, s6, 0x20));
        Avx.Store(dst + 3 * dstRowStride, Avx.Permute2x128(s3, s7, 0x20));
        Avx.Store(dst + 4 * dstRowStride, Avx.Permute2x128(s0, s4, 0x31));
        Avx.Store(dst + 5 * dstRowStride, Avx.Permute2x128(s1, s5, 0x31));
        Avx.Store(dst + 6 * dstRowStride, Avx.Permute2x128(s2, s6, 0x31));
        Avx.Store(dst + 7 * dstRowStride, Avx.Permute2x128(s3, s7, 0x31));
    }

    /// <summary>
    /// <c>STINGRAY_CPU_KPACK_SIMD=0</c> restores the scalar K-pack transpose in Flash-64 prefill.
    /// Both produce identical bytes, so this is a bisect seam and an A/B measurement handle, not a
    /// tuning knob.
    /// </summary>
    public static bool KPackSimdEnabled { get; } =
        Environment.GetEnvironmentVariable("STINGRAY_CPU_KPACK_SIMD") != "0";

    /// <summary>
    /// <c>STINGRAY_CPU_VNNI=0</c> forces the AVX2 chain even where VNNI exists. This is not a
    /// performance knob — it is what makes <see cref="DotU8I8ToI32"/> testable. All three branches
    /// are claimed bit-identical, but a machine only ever executes one of them, so on AVX2-only
    /// hardware the VNNI branches are dead code that no parity suite can reach. With this toggle a
    /// VNNI-capable host can run the same Q4_K suites both ways and diff them, turning that claim
    /// into a checked one.
    /// </summary>
    internal static bool VnniEnabled { get; } =
        Environment.GetEnvironmentVariable("STINGRAY_CPU_VNNI") != "0";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> DotU8I8ToI32(
        Vector256<byte> weights, Vector256<sbyte> activations, Vector256<short> one16)
    {
        if (VnniEnabled)
        {
            if (AvxVnniInt8.IsSupported)
                return AvxVnniInt8.MultiplyWideningAndAdd(Vector256<int>.Zero, weights.AsSByte(), activations);
            if (AvxVnni.IsSupported)
                return AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, weights, activations);
        }
        return Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(weights, activations), one16);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumQ4KInput(Vector256<byte> lo, Vector256<byte> hi, Vector256<short> one16,
        sbyte* q8, float* dSub, int chunk, int s0, int s1,
        float dsc1, float dsc2,
        ref Vector256<float> facc)
    {
        var qlo = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
        var qhi = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
        Vector256<int> i0 = DotU8I8ToI32(lo, qlo, one16);
        Vector256<int> i1 = DotU8I8ToI32(hi, qhi, one16);

        facc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Vector256.Create(dSub[s0] * dsc1), facc);
        facc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(dSub[s1] * dsc2), facc);

        // The min-correction term used to live here, as scalar work per sub-block. It is now a
        // single vectorised pass per super-block — see MinCorrectionQ4K, which every Q4_K kernel
        // shares so they stay parity-equal with each other.
    }

    // ================================================================
    //  Q6_K · Q8_K Fused two-input dot (issue #42) — decode each Q6_K
    //  super-block ONCE in registers and inner-int-product it against
    //  TWO pre-quantized Q8_K inputs in the same pass. Mirrors the
    //  DotQ4K_2In / DotQ5K_2In pattern but stays in AVX2 since the
    //  one-input kernel is AVX2 (u8·i8 maddubs).
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ6K_Q8K_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ6K_Q8K_2In_Avx2(row, scratch1, scratch2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ6K_Q8K_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ6K_Q8K_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ6K_Q8K_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int cols, int numBlocks,
                                             out float sum1, out float sum2)
    {
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 4);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 4 + numBlocks * 256);

        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 4);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        Vector256<short>* scv = stackalloc Vector256<short>[8];   // see Q6KBuildScaleVectors

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper1 = dw * dArr1[i];
            float dSuper2 = dw * dArr2[i];

            // Scales (int16) — shared between both inputs.
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            Q6KBuildScaleVectors(scales128, scv);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);

            // Per-input offset corrections.
            var q8sums1 = Vector256.LoadUnsafe(ref *(bsumsArr1 + i * 16));
            var q8sclsub1 = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums1, scales16), 5);
            var q8sums2 = Vector256.LoadUnsafe(ref *(bsumsArr2 + i * 16));
            var q8sclsub2 = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums2, scales16), 5);

            var sumi1 = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero;
            sbyte* q8a = (sbyte*)(qsArr1 + i * 256);
            sbyte* q8b = (sbyte*)(qsArr2 + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values — shared.
                var q4h_0 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(
                    Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15),
                    q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15),
                    q4h_3);

                int isc = j * 4;
                var sc16_0 = scv[isc + 0];
                var sc16_1 = scv[isc + 1];
                var sc16_2 = scv[isc + 2];
                var sc16_3 = scv[isc + 3];

                // Input 1 pass — q4_X stays live for input 2.
                {
                    var qa0 = Vector256.LoadUnsafe(ref *(q8a + j * 128)).AsSByte();
                    var qa1 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 32)).AsSByte();
                    var qa2 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 64)).AsSByte();
                    var qa3 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 96)).AsSByte();

                    var pa0 = Avx2.MultiplyAddAdjacent(q4_0, qa0);
                    var pa1 = Avx2.MultiplyAddAdjacent(q4_1, qa1);
                    var pa2 = Avx2.MultiplyAddAdjacent(q4_2, qa2);
                    var pa3 = Avx2.MultiplyAddAdjacent(q4_3, qa3);

                    var sa0 = Avx2.MultiplyAddAdjacent(sc16_0, pa0);
                    var sa1 = Avx2.MultiplyAddAdjacent(sc16_1, pa1);
                    var sa2 = Avx2.MultiplyAddAdjacent(sc16_2, pa2);
                    var sa3 = Avx2.MultiplyAddAdjacent(sc16_3, pa3);

                    sumi1 = Avx2.Add(sumi1, Avx2.Add(Avx2.Add(sa0, sa1), Avx2.Add(sa2, sa3)));
                }

                // Input 2 pass — reuses decoded q4_X and sc16_X.
                {
                    var qb0 = Vector256.LoadUnsafe(ref *(q8b + j * 128)).AsSByte();
                    var qb1 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 32)).AsSByte();
                    var qb2 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 64)).AsSByte();
                    var qb3 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 96)).AsSByte();

                    var pb0 = Avx2.MultiplyAddAdjacent(q4_0, qb0);
                    var pb1 = Avx2.MultiplyAddAdjacent(q4_1, qb1);
                    var pb2 = Avx2.MultiplyAddAdjacent(q4_2, qb2);
                    var pb3 = Avx2.MultiplyAddAdjacent(q4_3, qb3);

                    var sb0 = Avx2.MultiplyAddAdjacent(sc16_0, pb0);
                    var sb1 = Avx2.MultiplyAddAdjacent(sc16_1, pb1);
                    var sb2 = Avx2.MultiplyAddAdjacent(sc16_2, pb2);
                    var sb3 = Avx2.MultiplyAddAdjacent(sc16_3, pb3);

                    sumi2 = Avx2.Add(sumi2, Avx2.Add(Avx2.Add(sb0, sb1), Avx2.Add(sb2, sb3)));
                }
            }

            acc1 = Fma.MultiplyAdd(
                Vector256.Create(dSuper1),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi1, q8sclsub1)),
                acc1);
            acc2 = Fma.MultiplyAdd(
                Vector256.Create(dSuper2),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi2, q8sclsub2)),
                acc2);
        }
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
    }

    // ================================================================
    //  Q6_K · Q8_K Fused four-input dot (issue #209) — register-tiled
    //  extension of DotQ6K_Q8K_2In: decode each Q6_K super-block ONCE and
    //  inner-int-product it against FOUR pre-quantized Q8_K inputs. Each
    //  input's sumi/acc chain matches the single-input order exactly, so the
    //  result is bit-identical to four DotQ6K_Q8K calls.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ6K_Q8K_4In(byte* row, byte* scratch0, byte* scratch1,
                                       byte* scratch2, byte* scratch3, int cols,
                                       out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ6K_Q8K_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ6K_Q8K_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ6K_Q8K_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ6K_Q8K_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ6K_Q8K_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ6K_Q8K_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 4);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 4 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 4);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 4 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 4);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 4 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 4);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;
        Vector256<short>* scv = stackalloc Vector256<short>[8];   // see Q6KBuildScaleVectors

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper0 = dw * dArr0[i];
            float dSuper1 = dw * dArr1[i];
            float dSuper2 = dw * dArr2[i];
            float dSuper3 = dw * dArr3[i];

            // Scales (int16) — shared between all inputs.
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            Q6KBuildScaleVectors(scales128, scv);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);

            // Per-input offset corrections.
            var q8sums0 = Vector256.LoadUnsafe(ref *(bsumsArr0 + i * 16));
            var q8sclsub0 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums0, scales16), 5);
            var q8sums1 = Vector256.LoadUnsafe(ref *(bsumsArr1 + i * 16));
            var q8sclsub1 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums1, scales16), 5);
            var q8sums2 = Vector256.LoadUnsafe(ref *(bsumsArr2 + i * 16));
            var q8sclsub2 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums2, scales16), 5);
            var q8sums3 = Vector256.LoadUnsafe(ref *(bsumsArr3 + i * 16));
            var q8sclsub3 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums3, scales16), 5);

            var sumi0 = Vector256<int>.Zero;
            var sumi1 = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero;
            var sumi3 = Vector256<int>.Zero;
            sbyte* q8a = (sbyte*)(qsArr0 + i * 256);
            sbyte* q8b = (sbyte*)(qsArr1 + i * 256);
            sbyte* q8c = (sbyte*)(qsArr2 + i * 256);
            sbyte* q8d = (sbyte*)(qsArr3 + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values — shared across inputs.
                var q4h_0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15), q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15), q4h_3);

                int isc = j * 4;
                var sc16_0 = scv[isc + 0];
                var sc16_1 = scv[isc + 1];
                var sc16_2 = scv[isc + 2];
                var sc16_3 = scv[isc + 3];

                Q6KAccumInput(q8a, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi0);
                Q6KAccumInput(q8b, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi1);
                Q6KAccumInput(q8c, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi2);
                Q6KAccumInput(q8d, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi3);
            }

            acc0 = Fma.MultiplyAdd(Vector256.Create(dSuper0),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi0, q8sclsub0)), acc0);
            acc1 = Fma.MultiplyAdd(Vector256.Create(dSuper1),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi1, q8sclsub1)), acc1);
            acc2 = Fma.MultiplyAdd(Vector256.Create(dSuper2),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi2, q8sclsub2)), acc2);
            acc3 = Fma.MultiplyAdd(Vector256.Create(dSuper3),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi3, q8sclsub3)), acc3);
        }
        sum0 = HSum256(acc0);
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
        sum3 = HSum256(acc3);
    }

    // ================================================================
    //  Q6_K · Q8_K Fused eight-input dot — widens DotQ6K_Q8K_4In to 8 inputs, matching
    //  DotQ4K_Q8KS_8In's already-shipped, already-measured widening for Q4_K. SmolLM2's own
    //  attn_v/ffn_down tensors alternate Q4_K/Q6_K across layers, so Q6_K rows are real,
    //  frequently-hit weight traffic in this exact model, not a hypothetical dtype.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ6K_Q8K_8In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ6K_Q8K_8In_Avx2(row, scratch0, scratch1, scratch2, scratch3,
                scratch4, scratch5, scratch6, scratch7, numBlocks,
                out sum0, out sum1, out sum2, out sum3, out sum4, out sum5, out sum6, out sum7);
            return;
        }
        sum0 = DotQ6K_Q8K_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ6K_Q8K_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ6K_Q8K_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ6K_Q8K_Scalar(row, scratch3, numBlocks);
        sum4 = DotQ6K_Q8K_Scalar(row, scratch4, numBlocks);
        sum5 = DotQ6K_Q8K_Scalar(row, scratch5, numBlocks);
        sum6 = DotQ6K_Q8K_Scalar(row, scratch6, numBlocks);
        sum7 = DotQ6K_Q8K_Scalar(row, scratch7, numBlocks);
    }

    private static void DotQ6K_Q8K_8In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3,
        byte* scratch4, byte* scratch5, byte* scratch6, byte* scratch7, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3,
        out float sum4, out float sum5, out float sum6, out float sum7)
    {
        float* dArr0 = (float*)scratch0; sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 4); short* bsumsArr0 = (short*)(scratch0 + numBlocks * 4 + numBlocks * 256);
        float* dArr1 = (float*)scratch1; sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 4); short* bsumsArr1 = (short*)(scratch1 + numBlocks * 4 + numBlocks * 256);
        float* dArr2 = (float*)scratch2; sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 4); short* bsumsArr2 = (short*)(scratch2 + numBlocks * 4 + numBlocks * 256);
        float* dArr3 = (float*)scratch3; sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 4); short* bsumsArr3 = (short*)(scratch3 + numBlocks * 4 + numBlocks * 256);
        float* dArr4 = (float*)scratch4; sbyte* qsArr4 = (sbyte*)(scratch4 + numBlocks * 4); short* bsumsArr4 = (short*)(scratch4 + numBlocks * 4 + numBlocks * 256);
        float* dArr5 = (float*)scratch5; sbyte* qsArr5 = (sbyte*)(scratch5 + numBlocks * 4); short* bsumsArr5 = (short*)(scratch5 + numBlocks * 4 + numBlocks * 256);
        float* dArr6 = (float*)scratch6; sbyte* qsArr6 = (sbyte*)(scratch6 + numBlocks * 4); short* bsumsArr6 = (short*)(scratch6 + numBlocks * 4 + numBlocks * 256);
        float* dArr7 = (float*)scratch7; sbyte* qsArr7 = (sbyte*)(scratch7 + numBlocks * 4); short* bsumsArr7 = (short*)(scratch7 + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc0 = Vector256<float>.Zero; var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero; var acc3 = Vector256<float>.Zero;
        var acc4 = Vector256<float>.Zero; var acc5 = Vector256<float>.Zero;
        var acc6 = Vector256<float>.Zero; var acc7 = Vector256<float>.Zero;
        Vector256<short>* scv = stackalloc Vector256<short>[8];   // see Q6KBuildScaleVectors

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper0 = dw * dArr0[i]; float dSuper1 = dw * dArr1[i];
            float dSuper2 = dw * dArr2[i]; float dSuper3 = dw * dArr3[i];
            float dSuper4 = dw * dArr4[i]; float dSuper5 = dw * dArr5[i];
            float dSuper6 = dw * dArr6[i]; float dSuper7 = dw * dArr7[i];

            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            Q6KBuildScaleVectors(scales128, scv);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);

            var q8sclsub0 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr0 + i * 16)), scales16), 5);
            var q8sclsub1 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr1 + i * 16)), scales16), 5);
            var q8sclsub2 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr2 + i * 16)), scales16), 5);
            var q8sclsub3 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr3 + i * 16)), scales16), 5);
            var q8sclsub4 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr4 + i * 16)), scales16), 5);
            var q8sclsub5 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr5 + i * 16)), scales16), 5);
            var q8sclsub6 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr6 + i * 16)), scales16), 5);
            var q8sclsub7 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref *(bsumsArr7 + i * 16)), scales16), 5);

            var sumi0 = Vector256<int>.Zero; var sumi1 = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero; var sumi3 = Vector256<int>.Zero;
            var sumi4 = Vector256<int>.Zero; var sumi5 = Vector256<int>.Zero;
            var sumi6 = Vector256<int>.Zero; var sumi7 = Vector256<int>.Zero;
            sbyte* q8a = (sbyte*)(qsArr0 + i * 256); sbyte* q8b = (sbyte*)(qsArr1 + i * 256);
            sbyte* q8c = (sbyte*)(qsArr2 + i * 256); sbyte* q8d = (sbyte*)(qsArr3 + i * 256);
            sbyte* q8e = (sbyte*)(qsArr4 + i * 256); sbyte* q8f = (sbyte*)(qsArr5 + i * 256);
            sbyte* q8g = (sbyte*)(qsArr6 + i * 256); sbyte* q8h = (sbyte*)(qsArr7 + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                var q4h_0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15), q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15), q4h_3);

                int isc = j * 4;
                var sc16_0 = scv[isc + 0];
                var sc16_1 = scv[isc + 1];
                var sc16_2 = scv[isc + 2];
                var sc16_3 = scv[isc + 3];

                Q6KAccumInput(q8a, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi0);
                Q6KAccumInput(q8b, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi1);
                Q6KAccumInput(q8c, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi2);
                Q6KAccumInput(q8d, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi3);
                Q6KAccumInput(q8e, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi4);
                Q6KAccumInput(q8f, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi5);
                Q6KAccumInput(q8g, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi6);
                Q6KAccumInput(q8h, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi7);
            }

            acc0 = Fma.MultiplyAdd(Vector256.Create(dSuper0), Avx.ConvertToVector256Single(Avx2.Subtract(sumi0, q8sclsub0)), acc0);
            acc1 = Fma.MultiplyAdd(Vector256.Create(dSuper1), Avx.ConvertToVector256Single(Avx2.Subtract(sumi1, q8sclsub1)), acc1);
            acc2 = Fma.MultiplyAdd(Vector256.Create(dSuper2), Avx.ConvertToVector256Single(Avx2.Subtract(sumi2, q8sclsub2)), acc2);
            acc3 = Fma.MultiplyAdd(Vector256.Create(dSuper3), Avx.ConvertToVector256Single(Avx2.Subtract(sumi3, q8sclsub3)), acc3);
            acc4 = Fma.MultiplyAdd(Vector256.Create(dSuper4), Avx.ConvertToVector256Single(Avx2.Subtract(sumi4, q8sclsub4)), acc4);
            acc5 = Fma.MultiplyAdd(Vector256.Create(dSuper5), Avx.ConvertToVector256Single(Avx2.Subtract(sumi5, q8sclsub5)), acc5);
            acc6 = Fma.MultiplyAdd(Vector256.Create(dSuper6), Avx.ConvertToVector256Single(Avx2.Subtract(sumi6, q8sclsub6)), acc6);
            acc7 = Fma.MultiplyAdd(Vector256.Create(dSuper7), Avx.ConvertToVector256Single(Avx2.Subtract(sumi7, q8sclsub7)), acc7);
        }
        sum0 = HSum256(acc0); sum1 = HSum256(acc1); sum2 = HSum256(acc2); sum3 = HSum256(acc3);
        sum4 = HSum256(acc4); sum5 = HSum256(acc5); sum6 = HSum256(acc6); sum7 = HSum256(acc7);
    }

    /// <summary>One Q8_K input's contribution to a Q6_K super-block half-pair, using
    /// the already-decoded weight sextets (<paramref name="q4_0"/>..<c>q4_3</c>) and
    /// per-group scales. Matches the inline input pass of <see cref="DotQ6K_Q8K_2In_Avx2"/>
    /// exactly so the 4-input path stays bit-identical to the single/pair paths.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q6KAccumInput(sbyte* q8, int j,
        Vector256<byte> q4_0, Vector256<byte> q4_1, Vector256<byte> q4_2, Vector256<byte> q4_3,
        Vector256<short> sc16_0, Vector256<short> sc16_1, Vector256<short> sc16_2, Vector256<short> sc16_3,
        ref Vector256<int> sumi)
    {
        var qa0 = Vector256.LoadUnsafe(ref *(q8 + j * 128)).AsSByte();
        var qa1 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 32)).AsSByte();
        var qa2 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 64)).AsSByte();
        var qa3 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 96)).AsSByte();

        var pa0 = Avx2.MultiplyAddAdjacent(q4_0, qa0);
        var pa1 = Avx2.MultiplyAddAdjacent(q4_1, qa1);
        var pa2 = Avx2.MultiplyAddAdjacent(q4_2, qa2);
        var pa3 = Avx2.MultiplyAddAdjacent(q4_3, qa3);

        var sa0 = Avx2.MultiplyAddAdjacent(sc16_0, pa0);
        var sa1 = Avx2.MultiplyAddAdjacent(sc16_1, pa1);
        var sa2 = Avx2.MultiplyAddAdjacent(sc16_2, pa2);
        var sa3 = Avx2.MultiplyAddAdjacent(sc16_3, pa3);

        sumi = Avx2.Add(sumi, Avx2.Add(Avx2.Add(sa0, sa1), Avx2.Add(sa2, sa3)));
    }

    // ================================================================
    //  RMS Norm (AVX2)
    // ================================================================

    /// <summary>
    /// Wide-vector RmsNorm (AVX-512, 16 floats/iter). Falls through to the AVX2 path
    /// when Avx512F isn't available so callers can use it unconditionally. The
    /// reduction order differs by ~ULP vs the AVX2 path; only use from forward
    /// passes whose parity tests target internal-only argmax (Gemma 4) rather than
    /// byte-exact llama.cpp output (Qwen3.6-MTP — see
    /// feedback_qkv_matvecdual_breaks_mtp_parity).
    /// </summary>
    public static void RmsNormWide(float* output, float* input, float* weight, int size, float eps)
    {
        if (Avx512F.IsSupported && size >= 16)
        {
            var sumSq = Vector512<float>.Zero;
            int i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                sumSq = Avx512F.FusedMultiplyAdd(v, v, sumSq);
            }
            float ss = HSum512(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector512.Create(scale);

            i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                var w = Avx512F.LoadVector512(weight + i);
                Avx512F.Store(output + i, Avx512F.Multiply(Avx512F.Multiply(v, scaleV), w));
            }
            for (; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
        else
        {
            RmsNorm(output, input, weight, size, eps);
        }
    }

    /// <summary>
    /// Wide-vector PureRmsNorm (AVX-512, 16 floats/iter). See <see cref="RmsNormWide"/>
    /// for parity caveats — only use from forward passes whose tests target
    /// internal-only argmax.
    /// </summary>
    public static void PureRmsNormWide(float* output, float* input, int size, float eps)
    {
        if (Avx512F.IsSupported && size >= 16)
        {
            var sumSq = Vector512<float>.Zero;
            int i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                sumSq = Avx512F.FusedMultiplyAdd(v, v, sumSq);
            }
            float ss = HSum512(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector512.Create(scale);

            i = 0;
            for (; i + 16 <= size; i += 16)
                Avx512F.Store(output + i, Avx512F.Multiply(Avx512F.LoadVector512(input + i), scaleV));
            for (; i < size; i++)
                output[i] = input[i] * scale;
        }
        else
        {
            PureRmsNorm(output, input, size, eps);
        }
    }

    public static void RmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            int i = 0;

            for (; i + 32 <= size; i += 32)
            {
                var v0 = Avx.LoadVector256(input + i);
                var v1 = Avx.LoadVector256(input + i + 8);
                var v2 = Avx.LoadVector256(input + i + 16);
                var v3 = Avx.LoadVector256(input + i + 24);

                acc0 = Fma.MultiplyAdd(v0, v0, acc0);
                acc1 = Fma.MultiplyAdd(v1, v1, acc1);
                acc2 = Fma.MultiplyAdd(v2, v2, acc2);
                acc3 = Fma.MultiplyAdd(v3, v3, acc3);
            }

            var sumSq = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));

            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                sumSq = Fma.MultiplyAdd(v, v, sumSq);
            }

            // ggml's reference RMS-norm (ggml-cpu/ops.cpp's ggml_compute_forward_rms_norm_f32)
            // accumulates the sum-of-squares in ggml_float (double), not float32 -- widen the
            // final reduction to double to match. The SIMD partial sums above stay float32 (the
            // dominant source of any remaining gap vs a scalar-double reference is elsewhere;
            // this narrows one real, measurable discrepancy without a full kernel rewrite).
            double ss = HSum256(sumSq);
            for (; i < size; i++) ss += (double)input[i] * input[i];

            float scale = (float)(1.0 / Math.Sqrt(ss / size + eps));
            var scaleV = Vector256.Create(scale);

            // Pass 2: scale and weight
            i = 0;
            for (; i + 32 <= size; i += 32)
            {
                var v0 = Avx.LoadVector256(input + i);
                var v1 = Avx.LoadVector256(input + i + 8);
                var v2 = Avx.LoadVector256(input + i + 16);
                var v3 = Avx.LoadVector256(input + i + 24);

                var w0 = Avx.LoadVector256(weight + i);
                var w1 = Avx.LoadVector256(weight + i + 8);
                var w2 = Avx.LoadVector256(weight + i + 16);
                var w3 = Avx.LoadVector256(weight + i + 24);

                Avx.Store(output + i,      Avx.Multiply(Avx.Multiply(v0, scaleV), w0));
                Avx.Store(output + i + 8,  Avx.Multiply(Avx.Multiply(v1, scaleV), w1));
                Avx.Store(output + i + 16, Avx.Multiply(Avx.Multiply(v2, scaleV), w2));
                Avx.Store(output + i + 24, Avx.Multiply(Avx.Multiply(v3, scaleV), w3));
            }

            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                var w = Avx.LoadVector256(weight + i);
                Avx.Store(output + i, Avx.Multiply(Avx.Multiply(v, scaleV), w));
            }

            for (; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
        else
        {
            double ss = 0;
            for (int i = 0; i < size; i++) ss += (double)input[i] * input[i];
            float scale = (float)(1.0 / Math.Sqrt(ss / size + eps));
            for (int i = 0; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
    }

    /// <summary>
    /// Layer normalization with learned scale and optional bias:
    /// <c>((x - mean) / sqrt(variance + eps)) * weight [+ bias]</c>, where variance is the
    /// population (divide-by-N, not N-1) variance — matches PyTorch's <c>nn.LayerNorm</c> and
    /// ggml's <c>ggml_compute_forward_norm_f32</c>.
    ///
    /// <paramref name="bias"/> may be null: llama.cpp's generic <c>build_norm</c> applies the
    /// mean-subtract/weight/bias steps independently (mean-subtract always, weight-multiply only
    /// if a weight tensor exists, bias-add only if a bias tensor exists) — GPT-NeoX/Falcon/
    /// StarCoder2 have both weight and bias; Command-R (cohere2) has weight but no bias tensor at
    /// all, still needing true LayerNorm's mean-subtraction (unlike the RMSNorm kernels above,
    /// which never subtract the mean regardless of bias).
    ///
    /// Scalar by design: this is a correctness-first path. Safe for in-place use (output == input)
    /// since both reduction passes complete before any element is written.
    /// </summary>
    public static void LayerNorm(float* output, float* input, float* weight, float* bias, int size, float eps)
    {
        float sum = 0f;
        for (int i = 0; i < size; i++) sum += input[i];
        float mean = sum / size;

        float sumSq = 0f;
        for (int i = 0; i < size; i++)
        {
            float d = input[i] - mean;
            sumSq += d * d;
        }
        float invStd = 1f / MathF.Sqrt(sumSq / size + eps);

        if (bias != null)
        {
            for (int i = 0; i < size; i++)
                output[i] = (input[i] - mean) * invStd * weight[i] + bias[i];
        }
        else
        {
            for (int i = 0; i < size; i++)
                output[i] = (input[i] - mean) * invStd * weight[i];
        }
    }

    /// <summary>
    /// LayerNorm (mean-subtract + variance-normalize) with NO learned scale or bias at all —
    /// distinct from <see cref="LayerNorm"/>'s bias-optional-but-weight-required signature.
    /// OLMo v1 (src/models/olmo.cpp: <c>build_norm(inpL, NULL, NULL, LLM_NORM, il)</c> — both the
    /// weight AND bias arguments are null) is the only known user: its GGUF carries no
    /// attn_norm/ffn_norm/output_norm tensor at all, weighted or not.
    /// </summary>
    public static void PureLayerNorm(float* output, float* input, int size, float eps)
    {
        float sum = 0f;
        for (int i = 0; i < size; i++) sum += input[i];
        float mean = sum / size;

        float sumSq = 0f;
        for (int i = 0; i < size; i++)
        {
            float d = input[i] - mean;
            sumSq += d * d;
        }
        float invStd = 1f / MathF.Sqrt(sumSq / size + eps);

        for (int i = 0; i < size; i++)
            output[i] = (input[i] - mean) * invStd;
    }

    /// <summary>
    /// RMS normalization without learned weights (pure L2 normalize).
    /// Used for Llama4TextL2Norm in QK-norm.
    /// </summary>
    public static void PureRmsNorm(float* output, float* input, int size, float eps)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            int i = 0;

            for (; i + 32 <= size; i += 32)
            {
                var v0 = Avx.LoadVector256(input + i);
                var v1 = Avx.LoadVector256(input + i + 8);
                var v2 = Avx.LoadVector256(input + i + 16);
                var v3 = Avx.LoadVector256(input + i + 24);

                acc0 = Fma.MultiplyAdd(v0, v0, acc0);
                acc1 = Fma.MultiplyAdd(v1, v1, acc1);
                acc2 = Fma.MultiplyAdd(v2, v2, acc2);
                acc3 = Fma.MultiplyAdd(v3, v3, acc3);
            }

            var sumSq = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));

            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                sumSq = Fma.MultiplyAdd(v, v, sumSq);
            }

            // ggml's reference RMS-norm (ggml-cpu/ops.cpp's ggml_compute_forward_rms_norm_f32)
            // accumulates the sum-of-squares in ggml_float (double), not float32 -- widen the
            // final reduction to double to match. The SIMD partial sums above stay float32 (the
            // dominant source of any remaining gap vs a scalar-double reference is elsewhere;
            // this narrows one real, measurable discrepancy without a full kernel rewrite).
            double ss = HSum256(sumSq);
            for (; i < size; i++) ss += (double)input[i] * input[i];

            float scale = (float)(1.0 / Math.Sqrt(ss / size + eps));
            var scaleV = Vector256.Create(scale);

            i = 0;
            for (; i + 32 <= size; i += 32)
            {
                var v0 = Avx.LoadVector256(input + i);
                var v1 = Avx.LoadVector256(input + i + 8);
                var v2 = Avx.LoadVector256(input + i + 16);
                var v3 = Avx.LoadVector256(input + i + 24);

                Avx.Store(output + i,      Avx.Multiply(v0, scaleV));
                Avx.Store(output + i + 8,  Avx.Multiply(v1, scaleV));
                Avx.Store(output + i + 16, Avx.Multiply(v2, scaleV));
                Avx.Store(output + i + 24, Avx.Multiply(v3, scaleV));
            }

            for (; i + 8 <= size; i += 8)
                Avx.Store(output + i, Avx.Multiply(Avx.LoadVector256(input + i), scaleV));

            for (; i < size; i++)
                output[i] = input[i] * scale;
        }
        else
        {
            float ss = 0;
            for (int i = 0; i < size; i++) ss += input[i] * input[i];
            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            for (int i = 0; i < size; i++)
                output[i] = input[i] * scale;
        }
    }

    // ================================================================
    //  Softmax (AVX2 with scalar exp)
    // ================================================================

    /// <summary>
    /// Strided generalization of <see cref="GemmF32_64x64_6x2"/>:
    /// <c>C[m,n] = A[m,k] * B[k,n]</c> with independent row strides, optionally accumulating.
    /// Same 6-row by 2-YMM microkernel, same k-loop, same FMA order — so at
    /// <c>k = n = lda = ldb = ldc = 64</c> it is bit-identical to the hardcoded version, which
    /// <c>GemmF32StridedParityTests</c> pins.
    ///
    /// <para><b>Why this exists.</b> Flash attention at <c>headDim = 128</c> needs two shapes the
    /// hardcoded kernel cannot express — <c>Q·Kᵀ</c> wants <c>k=128, n=64</c> and <c>P·V</c> wants
    /// <c>k=64, n=128</c> — because that kernel bakes K, N and all three row strides in as the
    /// literal 64. That is why flash mode refuses at hd128 rather than computing something wrong
    /// (see <c>tools/attn-shared/AttnKernels.cs</c>).</para>
    ///
    /// <para><b>Cost of generality, measured not assumed.</b> Runtime k/n/strides cannot be
    /// constant-folded into loop bounds and address arithmetic the way literals can, and this
    /// codebase has already been bitten once by assuming a generic abstraction was free (the
    /// <c>static abstract</c> attn-bench arm measured 25-30% SLOWER and was reverted). The
    /// hardcoded 64x64 kernel is therefore KEPT and still used at hd64; this one is the
    /// shape-gated fallback that unblocks everything else. See the doc for the A/B.</para>
    ///
    /// <param name="n">Must be a multiple of 8. Every real shape here (64, 128, any head dim)
    /// satisfies that; a scalar tail would only add a branch to the hot loop for shapes that
    /// never occur.</param>
    /// </summary>
    public static void GemmF32_6x2(float* a, float* b, float* c, int m, int k, int n,
        int lda, int ldb, int ldc, bool accumulate = false)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
            throw new PlatformNotSupportedException("The 6x2 FP32 microkernel requires AVX2 and FMA.");
        if ((n & 7) != 0)
            throw new ArgumentException($"n must be a multiple of 8; got {n}.", nameof(n));

        int i = 0;
        for (; i + 6 <= m; i += 6)
        {
            float* a0 = a + (long)(i + 0) * lda, a1 = a + (long)(i + 1) * lda;
            float* a2 = a + (long)(i + 2) * lda, a3 = a + (long)(i + 3) * lda;
            float* a4 = a + (long)(i + 4) * lda, a5 = a + (long)(i + 5) * lda;
            float* c0 = c + (long)(i + 0) * ldc, c1 = c + (long)(i + 1) * ldc;
            float* c2 = c + (long)(i + 2) * ldc, c3 = c + (long)(i + 3) * ldc;
            float* c4 = c + (long)(i + 4) * ldc, c5 = c + (long)(i + 5) * ldc;

            int j = 0;
            for (; j + 16 <= n; j += 16)
            {
                var a00 = accumulate ? Avx.LoadVector256(c0 + j) : Vector256<float>.Zero;
                var a01 = accumulate ? Avx.LoadVector256(c0 + j + 8) : Vector256<float>.Zero;
                var a10 = accumulate ? Avx.LoadVector256(c1 + j) : Vector256<float>.Zero;
                var a11 = accumulate ? Avx.LoadVector256(c1 + j + 8) : Vector256<float>.Zero;
                var a20 = accumulate ? Avx.LoadVector256(c2 + j) : Vector256<float>.Zero;
                var a21 = accumulate ? Avx.LoadVector256(c2 + j + 8) : Vector256<float>.Zero;
                var a30 = accumulate ? Avx.LoadVector256(c3 + j) : Vector256<float>.Zero;
                var a31 = accumulate ? Avx.LoadVector256(c3 + j + 8) : Vector256<float>.Zero;
                var a40 = accumulate ? Avx.LoadVector256(c4 + j) : Vector256<float>.Zero;
                var a41 = accumulate ? Avx.LoadVector256(c4 + j + 8) : Vector256<float>.Zero;
                var a50 = accumulate ? Avx.LoadVector256(c5 + j) : Vector256<float>.Zero;
                var a51 = accumulate ? Avx.LoadVector256(c5 + j + 8) : Vector256<float>.Zero;

                float* bk = b + j;
                for (int kk = 0; kk < k; kk++, bk += ldb)
                {
                    var b0 = Avx.LoadVector256(bk);
                    var b1 = Avx.LoadVector256(bk + 8);
                    var q0 = Vector256.Create(a0[kk]);
                    a00 = Fma.MultiplyAdd(q0, b0, a00); a01 = Fma.MultiplyAdd(q0, b1, a01);
                    var q1 = Vector256.Create(a1[kk]);
                    a10 = Fma.MultiplyAdd(q1, b0, a10); a11 = Fma.MultiplyAdd(q1, b1, a11);
                    var q2 = Vector256.Create(a2[kk]);
                    a20 = Fma.MultiplyAdd(q2, b0, a20); a21 = Fma.MultiplyAdd(q2, b1, a21);
                    var q3 = Vector256.Create(a3[kk]);
                    a30 = Fma.MultiplyAdd(q3, b0, a30); a31 = Fma.MultiplyAdd(q3, b1, a31);
                    var q4 = Vector256.Create(a4[kk]);
                    a40 = Fma.MultiplyAdd(q4, b0, a40); a41 = Fma.MultiplyAdd(q4, b1, a41);
                    var q5 = Vector256.Create(a5[kk]);
                    a50 = Fma.MultiplyAdd(q5, b0, a50); a51 = Fma.MultiplyAdd(q5, b1, a51);
                }

                Avx.Store(c0 + j, a00); Avx.Store(c0 + j + 8, a01);
                Avx.Store(c1 + j, a10); Avx.Store(c1 + j + 8, a11);
                Avx.Store(c2 + j, a20); Avx.Store(c2 + j + 8, a21);
                Avx.Store(c3 + j, a30); Avx.Store(c3 + j + 8, a31);
                Avx.Store(c4 + j, a40); Avx.Store(c4 + j + 8, a41);
                Avx.Store(c5 + j, a50); Avx.Store(c5 + j + 8, a51);
            }
            // One 8-wide column block when n is 8 (mod 16). Unreachable at 64/128; present so
            // an odd head dim degrades in accuracy-preserving fashion instead of writing garbage.
            for (; j + 8 <= n; j += 8)
            {
                var a00 = accumulate ? Avx.LoadVector256(c0 + j) : Vector256<float>.Zero;
                var a10 = accumulate ? Avx.LoadVector256(c1 + j) : Vector256<float>.Zero;
                var a20 = accumulate ? Avx.LoadVector256(c2 + j) : Vector256<float>.Zero;
                var a30 = accumulate ? Avx.LoadVector256(c3 + j) : Vector256<float>.Zero;
                var a40 = accumulate ? Avx.LoadVector256(c4 + j) : Vector256<float>.Zero;
                var a50 = accumulate ? Avx.LoadVector256(c5 + j) : Vector256<float>.Zero;
                float* bk = b + j;
                for (int kk = 0; kk < k; kk++, bk += ldb)
                {
                    var b0 = Avx.LoadVector256(bk);
                    a00 = Fma.MultiplyAdd(Vector256.Create(a0[kk]), b0, a00);
                    a10 = Fma.MultiplyAdd(Vector256.Create(a1[kk]), b0, a10);
                    a20 = Fma.MultiplyAdd(Vector256.Create(a2[kk]), b0, a20);
                    a30 = Fma.MultiplyAdd(Vector256.Create(a3[kk]), b0, a30);
                    a40 = Fma.MultiplyAdd(Vector256.Create(a4[kk]), b0, a40);
                    a50 = Fma.MultiplyAdd(Vector256.Create(a5[kk]), b0, a50);
                }
                Avx.Store(c0 + j, a00); Avx.Store(c1 + j, a10); Avx.Store(c2 + j, a20);
                Avx.Store(c3 + j, a30); Avx.Store(c4 + j, a40); Avx.Store(c5 + j, a50);
            }
        }

        // Ragged row tail: 1-5 rows, 8 columns at a time. Mirrors the hardcoded kernel's tail
        // exactly, including its accumulation order, so bit parity survives any m.
        for (; i < m; i++)
        {
            float* ai = a + (long)i * lda;
            float* ci = c + (long)i * ldc;
            for (int j = 0; j + 8 <= n; j += 8)
            {
                var acc = accumulate ? Avx.LoadVector256(ci + j) : Vector256<float>.Zero;
                float* bk = b + j;
                for (int kk = 0; kk < k; kk++, bk += ldb)
                    acc = Fma.MultiplyAdd(Vector256.Create(ai[kk]), Avx.LoadVector256(bk), acc);
                Avx.Store(ci + j, acc);
            }
        }
    }

    /// <summary>
    /// Computes C[M,64] = A[M,64] * B[64,64], optionally accumulating into C.
    /// B is expected in K-major/transposed-key layout. The 6-row by 2-YMM microkernel mirrors
    /// llama.cpp's AVX2 Flash-attention GEMM shape: 12 accumulators + 2 B vectors + 1 broadcast.
    /// <para>Every dimension and stride is the literal 64 so the JIT can fold them; see
    /// <see cref="GemmF32_6x2"/> for the strided version and why both exist.</para>
    /// </summary>
    public static void GemmF32_64x64_6x2(float* a, float* b, float* c, int m, bool accumulate = false)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
            throw new PlatformNotSupportedException("The 6x2 FP32 microkernel requires AVX2 and FMA.");
        if ((uint)m > 64)
            throw new ArgumentOutOfRangeException(nameof(m));

        int i = 0;
        for (; i + 6 <= m; i += 6)
        {
            for (int j = 0; j < 64; j += 16)
            {
                var a00 = accumulate ? Avx.LoadVector256(c + (i + 0) * 64 + j) : Vector256<float>.Zero;
                var a01 = accumulate ? Avx.LoadVector256(c + (i + 0) * 64 + j + 8) : Vector256<float>.Zero;
                var a10 = accumulate ? Avx.LoadVector256(c + (i + 1) * 64 + j) : Vector256<float>.Zero;
                var a11 = accumulate ? Avx.LoadVector256(c + (i + 1) * 64 + j + 8) : Vector256<float>.Zero;
                var a20 = accumulate ? Avx.LoadVector256(c + (i + 2) * 64 + j) : Vector256<float>.Zero;
                var a21 = accumulate ? Avx.LoadVector256(c + (i + 2) * 64 + j + 8) : Vector256<float>.Zero;
                var a30 = accumulate ? Avx.LoadVector256(c + (i + 3) * 64 + j) : Vector256<float>.Zero;
                var a31 = accumulate ? Avx.LoadVector256(c + (i + 3) * 64 + j + 8) : Vector256<float>.Zero;
                var a40 = accumulate ? Avx.LoadVector256(c + (i + 4) * 64 + j) : Vector256<float>.Zero;
                var a41 = accumulate ? Avx.LoadVector256(c + (i + 4) * 64 + j + 8) : Vector256<float>.Zero;
                var a50 = accumulate ? Avx.LoadVector256(c + (i + 5) * 64 + j) : Vector256<float>.Zero;
                var a51 = accumulate ? Avx.LoadVector256(c + (i + 5) * 64 + j + 8) : Vector256<float>.Zero;

                for (int k = 0; k < 64; k++)
                {
                    var b0 = Avx.LoadVector256(b + k * 64 + j);
                    var b1 = Avx.LoadVector256(b + k * 64 + j + 8);
                    var q0 = Vector256.Create(a[(i + 0) * 64 + k]);
                    a00 = Fma.MultiplyAdd(q0, b0, a00); a01 = Fma.MultiplyAdd(q0, b1, a01);
                    var q1 = Vector256.Create(a[(i + 1) * 64 + k]);
                    a10 = Fma.MultiplyAdd(q1, b0, a10); a11 = Fma.MultiplyAdd(q1, b1, a11);
                    var q2 = Vector256.Create(a[(i + 2) * 64 + k]);
                    a20 = Fma.MultiplyAdd(q2, b0, a20); a21 = Fma.MultiplyAdd(q2, b1, a21);
                    var q3 = Vector256.Create(a[(i + 3) * 64 + k]);
                    a30 = Fma.MultiplyAdd(q3, b0, a30); a31 = Fma.MultiplyAdd(q3, b1, a31);
                    var q4 = Vector256.Create(a[(i + 4) * 64 + k]);
                    a40 = Fma.MultiplyAdd(q4, b0, a40); a41 = Fma.MultiplyAdd(q4, b1, a41);
                    var q5 = Vector256.Create(a[(i + 5) * 64 + k]);
                    a50 = Fma.MultiplyAdd(q5, b0, a50); a51 = Fma.MultiplyAdd(q5, b1, a51);
                }

                Avx.Store(c + (i + 0) * 64 + j, a00); Avx.Store(c + (i + 0) * 64 + j + 8, a01);
                Avx.Store(c + (i + 1) * 64 + j, a10); Avx.Store(c + (i + 1) * 64 + j + 8, a11);
                Avx.Store(c + (i + 2) * 64 + j, a20); Avx.Store(c + (i + 2) * 64 + j + 8, a21);
                Avx.Store(c + (i + 3) * 64 + j, a30); Avx.Store(c + (i + 3) * 64 + j + 8, a31);
                Avx.Store(c + (i + 4) * 64 + j, a40); Avx.Store(c + (i + 4) * 64 + j + 8, a41);
                Avx.Store(c + (i + 5) * 64 + j, a50); Avx.Store(c + (i + 5) * 64 + j + 8, a51);
            }
        }

        for (; i < m; i++)
        {
            for (int j = 0; j < 64; j += 8)
            {
                var acc = accumulate ? Avx.LoadVector256(c + i * 64 + j) : Vector256<float>.Zero;
                for (int k = 0; k < 64; k++)
                    acc = Fma.MultiplyAdd(Vector256.Create(a[i * 64 + k]),
                        Avx.LoadVector256(b + k * 64 + j), acc);
                Avx.Store(c + i * 64 + j, acc);
            }
        }
    }

    public static void SoftmaxInPlace(float* x, int size)
    {
        if (Avx.IsSupported && size >= 8)
        {
            // Pass 1: find max
            var maxV = Vector256.Create(float.NegativeInfinity);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                maxV = Avx.Max(maxV, Avx.LoadVector256(x + i));
            float max = HMax256(maxV);
            for (; i < size; i++)
                if (x[i] > max) max = x[i];

            // Pass 2: exp(x - max) and sum
            var maxBcast = Vector256.Create(max);
            var sumV = Vector256<float>.Zero;
            i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.Subtract(Avx.LoadVector256(x + i), maxBcast);
                var e = ExpApprox256(v);
                Avx.Store(x + i, e);
                sumV = Avx.Add(sumV, e);
            }
            float sum = HSum256(sumV);
            for (; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }

            // Pass 3: normalize
            var invSum = Vector256.Create(1.0f / sum);
            i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), invSum));
            float invSumS = 1.0f / sum;
            for (; i < size; i++)
                x[i] *= invSumS;
        }
        else
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < size; i++)
                if (x[i] > max) max = x[i];
            float sum = 0;
            for (int i = 0; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }
            float inv = 1.0f / sum;
            for (int i = 0; i < size; i++) x[i] *= inv;
        }
    }

    /// <summary>
    /// Replaces <paramref name="x"/> with exp(x - max) and returns the unnormalised sum.
    /// This is the online-softmax building block: unlike <see cref="SoftmaxInPlace"/>, it does
    /// not divide by the sum because the caller still has to merge it with earlier tiles.
    /// </summary>
    public static float ExpMinusMaxSumInPlace(float* x, int size, float max)
    {
        if (Avx.IsSupported && size >= 8)
        {
            var maxV = Vector256.Create(max);
            var sumV = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var e = ExpApprox256(Avx.Subtract(Avx.LoadVector256(x + i), maxV));
                Avx.Store(x + i, e);
                sumV = Avx.Add(sumV, e);
            }

            float sum = HSum256(sumV);
            for (; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }
            return sum;
        }

        float scalarSum = 0f;
        for (int i = 0; i < size; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            scalarSum += x[i];
        }
        return scalarSum;
    }

    /// <summary>Scales a row in place and returns its maximum value.</summary>
    public static float ScaleAndMaxF32InPlace(float* x, int size, float scale)
    {
        if (Avx.IsSupported && size >= 8)
        {
            var scaleV = Vector256.Create(scale);
            var maxV = Vector256.Create(float.NegativeInfinity);
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var value = Avx.Multiply(Avx.LoadVector256(x + i), scaleV);
                Avx.Store(x + i, value);
                maxV = Avx.Max(maxV, value);
            }

            float max = HMax256(maxV);
            for (; i < size; i++)
            {
                x[i] *= scale;
                max = MathF.Max(max, x[i]);
            }
            return max;
        }

        float scalarMax = float.NegativeInfinity;
        for (int i = 0; i < size; i++)
        {
            x[i] *= scale;
            scalarMax = MathF.Max(scalarMax, x[i]);
        }
        return scalarMax;
    }

    /// <summary>
    /// In-place element-wise sigmoid: x[i] = 1 / (1 + exp(-x[i])).
    /// Used for Llama-4 MoE router gating.
    /// </summary>
    public static void SigmoidInPlace(float* x, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var one = Vector256.Create(1.0f);
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(x + i);
                var negV = Avx.Subtract(Vector256<float>.Zero, v);
                var expNeg = ExpApprox256(negV);
                var sig = Avx.Divide(one, Avx.Add(one, expNeg));
                Avx.Store(x + i, sig);
            }
            for (; i < size; i++)
                x[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
        else
        {
            for (int i = 0; i < size; i++)
                x[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
    }

    // ================================================================
    //  Fused SiLU(gate) * up  (AVX2)
    // ================================================================

    public static void SiLuMul(float* gate, float* up, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var one = Vector256.Create(1.0f);
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var g = Avx.LoadVector256(gate + i);
                var u = Avx.LoadVector256(up + i);
                // sigmoid(g) = 1 / (1 + exp(-g))
                var negG = Avx.Subtract(Vector256<float>.Zero, g);
                var expNg = ExpApprox256(negG);
                var sigmoid = Avx.Divide(one, Avx.Add(one, expNg));
                // SiLU = g * sigmoid(g) * up
                Avx.Store(gate + i, Avx.Multiply(Avx.Multiply(g, sigmoid), u));
            }
            for (; i < size; i++)
            {
                float g = gate[i];
                gate[i] = g / (1.0f + MathF.Exp(-g)) * up[i];
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                float g = gate[i];
                gate[i] = g / (1.0f + MathF.Exp(-g)) * up[i];
            }
        }
    }

    // ================================================================
    //  Fused GELU-tanh(gate) * up   (AVX2 + scalar fallback)
    // ================================================================

    /// <summary>
    /// Fused tanh-approximate GELU on <paramref name="gate"/> multiplied by
    /// <paramref name="up"/>, written to <paramref name="outp"/>:
    /// <c>outp[i] = gelu_tanh(gate[i]) * up[i]</c> where
    /// <c>gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))</c>.
    /// Used by Gemma-style models (Gemma 4 FFN activation).
    /// </summary>
    public static void GeluTanhMul(float* gate, float* up, float* outp, int n)
    {
        // sqrt(2/π) ≈ 0.7978845608028654
        const float kAlpha = 0.7978845608028654f;
        const float kBeta = 0.044715f;

        if (Fma.IsSupported && n >= 8)
        {
            var half = Vector256.Create(0.5f);
            var one = Vector256.Create(1.0f);
            var two = Vector256.Create(2.0f);
            var alpha = Vector256.Create(kAlpha);
            var beta = Vector256.Create(kBeta);
            // Clamp 2*inner before exp so |inner|>~10 (e.g. ~ ±20 gate inputs from a
            // wide-dim trunk like Gemma 4) doesn't overflow ExpApprox256 to inf and
            // cascade to (inf-1)/(inf+1)=NaN. Safe range for float32 exp is ~[-88, 88];
            // |2*inner|>20 already saturates tanh to ±1 well within float precision.
            var clampHi = Vector256.Create(20.0f);
            var clampLo = Vector256.Create(-20.0f);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var g = Avx.LoadVector256(gate + i);
                var u = Avx.LoadVector256(up + i);
                // inner = alpha * (g + beta * g^3) = alpha * g * (1 + beta * g^2)
                var g2 = Avx.Multiply(g, g);
                var inner = Avx.Multiply(alpha,
                    Avx.Multiply(g, Fma.MultiplyAdd(beta, g2, one)));
                // tanh(inner) via (exp(2x) - 1) / (exp(2x) + 1)
                var twoInner = Avx.Max(clampLo, Avx.Min(clampHi, Avx.Multiply(two, inner)));
                var e2x = ExpApprox256(twoInner);
                var tanh = Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one));
                // 0.5 * g * (1 + tanh) * u
                var gelu = Avx.Multiply(half, Avx.Multiply(g, Avx.Add(one, tanh)));
                Avx.Store(outp + i, Avx.Multiply(gelu, u));
            }
            for (; i < n; i++)
            {
                float gs = gate[i];
                float inner = kAlpha * (gs + kBeta * gs * gs * gs);
                outp[i] = 0.5f * gs * (1.0f + MathF.Tanh(inner)) * up[i];
            }
        }
        else
        {
            GeluTanhMul_Scalar(gate, up, outp, n);
        }
    }

    /// <summary>
    /// Scalar reference for <see cref="GeluTanhMul"/> used by parity tests.
    /// Uses <see cref="MathF.Tanh"/> directly (no exp approximation).
    /// </summary>
    internal static void GeluTanhMul_Scalar(float* gate, float* up, float* outp, int n)
    {
        const float kAlpha = 0.7978845608028654f;
        const float kBeta = 0.044715f;
        for (int i = 0; i < n; i++)
        {
            float gs = gate[i];
            float inner = kAlpha * (gs + kBeta * gs * gs * gs);
            outp[i] = 0.5f * gs * (1.0f + MathF.Tanh(inner)) * up[i];
        }
    }

    /// <summary>
    /// Tanh-approximate GELU applied in place: <c>x[i] = 0.5*x[i]*(1 + tanh(sqrt(2/pi) *
    /// (x[i] + 0.044715*x[i]^3)))</c>. Same constants as <see cref="GeluTanhMul_Scalar"/>
    /// (verified against <c>ggml_gelu_f32</c>); used by GPT-NeoX/Pythia's non-gated FFN,
    /// which has no separate gate tensor to fuse the multiply against.
    /// </summary>
    public static void GeluInPlace(float* x, int n)
    {
        const float kAlpha = 0.7978845608028654f;
        const float kBeta = 0.044715f;
        for (int i = 0; i < n; i++)
        {
            float v = x[i];
            float inner = kAlpha * (v + kBeta * v * v * v);
            x[i] = 0.5f * v * (1f + MathF.Tanh(inner));
        }
    }

    /// <summary>
    /// "Quick" GELU (sigmoid/logistic approximation): <c>x * sigmoid(1.702*x)</c>. Distinct from
    /// <see cref="GeluInPlace"/>'s tanh approximation -- NOT interchangeable, they diverge by a
    /// measurable margin away from x=0. Ported from ggml's <c>ggml_geglu_quick</c>/
    /// <c>ggml_gelu_quick</c>, used by <c>FFN_GELU_QUICK</c> (llama.cpp <c>tools/mtmd/clip.cpp</c>'s
    /// default FFN activation for a mmproj that declares neither <c>use_gelu</c> nor
    /// <c>use_silu</c> metadata -- e.g. Gemma 4 E4B's <c>gemma4v</c> ViT).
    /// </summary>
    public static void GeluQuickInPlace(float* x, int n)
    {
        const float kAlpha = 1.702f;
        for (int i = 0; i < n; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-kAlpha * v));
        }
    }

    /// <summary>
    /// ReLU-squared activation, applied in place: <c>x[i] = max(0, x[i])^2</c>. Ported from
    /// llama.cpp's <c>LLM_FFN_RELU_SQR</c> handling (<c>llama-graph.cpp</c>: <c>ggml_relu</c> then
    /// <c>ggml_sqr</c>) — JAIS-2's non-gated FFN activation, used in place of GELU
    /// (GPT-NeoX/Falcon/GPT-2) or xIELU (Apertus) for that architecture's plain
    /// up -> activation -> down shape.
    /// </summary>
    public static void ReluSqrInPlace(float* x, int n)
    {
        for (int i = 0; i < n; i++)
        {
            float v = MathF.Max(0f, x[i]);
            x[i] = v * v;
        }
    }

    // ================================================================
    //  xIELU activation (scalar) — Apertus non-gated FFN
    // ================================================================

    /// <summary>
    /// xIELU activation, applied in place: <c>x[i] = alphaP*x[i]^2 + beta*x[i]</c> for
    /// <c>x[i] &gt; 0</c>, else <c>alphaN*(expm1(min(x[i],eps)) - x[i]) + beta*x[i]</c>. Ported
    /// from llama.cpp's <c>op_xielu</c> (<c>ggml/src/ggml-cpu/unary-ops.cpp</c>) — the reference
    /// this was checked against for Apertus, whose FFN has no gate projection at all (plain
    /// up → xIELU → down, unlike the SiLU/GELU paths which multiply against a second gate
    /// tensor). <paramref name="alphaN"/>/<paramref name="alphaP"/>/<paramref name="beta"/>/
    /// <paramref name="eps"/> are per-layer GGUF metadata values (<c>xielu.alpha_n</c> etc.),
    /// not architecture-wide constants. Scalar only: correctness first, per this project's
    /// standing priority (breadth over throughput) — a SIMD form is a follow-up, not a
    /// prerequisite, same as the IQ-quant dequantizers.
    /// </summary>
    public static void XieluInPlace(float* x, int n, float alphaN, float alphaP, float beta, float eps)
    {
        for (int i = 0; i < n; i++)
        {
            float v = x[i];
            x[i] = v > 0f
                ? alphaP * v * v + beta * v
                : alphaN * (MathF.Exp(MathF.Min(v, eps)) - 1f - v) + beta * v;
        }
    }

    // ================================================================
    //  Final-logit softcap (AVX2 + scalar)
    // ================================================================

    /// <summary>
    /// Apply <c>x[i] = tanh(x[i] / cap) * cap</c> in place. Used for the
    /// Gemma 4 final-logit softcap (cap=30) to clip extreme logits while
    /// preserving a smooth gradient near the boundary.
    /// </summary>
    public static void SoftcapInPlace(float* x, int n, float cap)
    {
        if (Fma.IsSupported && n >= 8)
        {
            var one = Vector256.Create(1.0f);
            var two = Vector256.Create(2.0f);
            var capV = Vector256.Create(cap);
            var invCap = Vector256.Create(1.0f / cap);
            // Clamp 2*arg before exp so an extreme pre-softcap logit doesn't overflow
            // ExpApprox256 to inf. |2*arg|>20 already saturates tanh to ±1 well within
            // float precision so the clamp is invisible to the final cap*tanh result.
            var clampHi = Vector256.Create(20.0f);
            var clampLo = Vector256.Create(-20.0f);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var v = Avx.LoadVector256(x + i);
                var arg = Avx.Multiply(v, invCap);
                // tanh(arg) = (exp(2*arg) - 1) / (exp(2*arg) + 1)
                var twoArg = Avx.Max(clampLo, Avx.Min(clampHi, Avx.Multiply(two, arg)));
                var e2x = ExpApprox256(twoArg);
                var tanh = Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one));
                Avx.Store(x + i, Avx.Multiply(tanh, capV));
            }
            for (; i < n; i++)
                x[i] = MathF.Tanh(x[i] / cap) * cap;
        }
        else
        {
            for (int i = 0; i < n; i++)
                x[i] = MathF.Tanh(x[i] / cap) * cap;
        }
    }

    // ================================================================
    //  Add in-place (AVX2)
    // ================================================================

    public static void AddInPlace(float* dst, float* src, int size)
    {
        if (Avx.IsSupported)
        {
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(dst + i, Avx.Add(Avx.LoadVector256(dst + i), Avx.LoadVector256(src + i)));
            for (; i < size; i++) dst[i] += src[i];
        }
        else
        {
            for (int i = 0; i < size; i++) dst[i] += src[i];
        }
    }

    /// <summary>Multiply every element of <paramref name="x"/> by a scalar.</summary>
    public static void ScaleInPlace(float* x, float scale, int size)
    {
        if (Avx.IsSupported)
        {
            var sv = Vector256.Create(scale);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), sv));
            for (; i < size; i++) x[i] *= scale;
        }
        else
        {
            for (int i = 0; i < size; i++) x[i] *= scale;
        }
    }

    /// <summary>Weighted accumulate in-place: dst[i] += weight * src[i].</summary>
    public static void WeightedAddInPlace(float* dst, float* src, float weight, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var wv = Vector256.Create(weight);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(dst + i, Fma.MultiplyAdd(wv, Avx.LoadVector256(src + i), Avx.LoadVector256(dst + i)));
            for (; i < size; i++)
                dst[i] += weight * src[i];
        }
        else
        {
            for (int i = 0; i < size; i++)
                dst[i] += weight * src[i];
        }
    }

    // ================================================================
    //  RoPE (precomputed sin/cos, SIMD rotation)
    // ================================================================

    /// <summary>
    /// Apply RoPE using precomputed cos/sin tables (avoids recomputing trig 48× per token).
    /// </summary>
    public static void ApplyRoPECached(float* x, float* cosTab, float* sinTab, int numHeads, int headDim)
    {
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            if (Avx.IsSupported && halfDim >= 4)
            {
                int i = 0;
                for (; i + 4 <= halfDim; i += 4)
                {
                    var v = Avx.LoadVector256(head + 2 * i);
                    var c = Vector256.Create(cosTab[i], cosTab[i], cosTab[i + 1], cosTab[i + 1],
                                             cosTab[i + 2], cosTab[i + 2], cosTab[i + 3], cosTab[i + 3]);
                    var s = Vector256.Create(sinTab[i], sinTab[i], sinTab[i + 1], sinTab[i + 1],
                                             sinTab[i + 2], sinTab[i + 2], sinTab[i + 3], sinTab[i + 3]);
                    var swapped = Avx.Shuffle(v, v, 0b10_11_00_01);
                    var signMask = Vector256.Create(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
                    var result = Fma.MultiplyAdd(v, c, Avx.Multiply(swapped, Avx.Multiply(s, signMask)));
                    Avx.Store(head + 2 * i, result);
                }
                for (; i < halfDim; i++)
                {
                    float x0 = head[2 * i], x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
            else
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i], x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
        }
    }

    /// <summary>
    /// "Normal" (GPT-J-style interleaved) RoPE with PARTIAL rotation. Rotates only the first
    /// <paramref name="ropeDim"/> dims of each head as consecutive pairs (2i, 2i+1); dims
    /// [ropeDim, headDim) pass through unchanged. GLM4 (non-multimodal) is the first user:
    /// LLAMA_ROPE_TYPE_NORM with rope.dimension_count=64, headDim=128 — confirmed against
    /// llama_model_rope_type()'s LLM_ARCH_GLM4 case in llama-model.cpp. The plain
    /// (non-partial) ApplyRoPECached above has no ropeDim awareness at all, so a model whose
    /// RopeDim &lt; headDim needs this instead — mirrors ApplyRoPECachedNeoxPartial below,
    /// just for the interleaved pairing instead of the NEOX halfDim-offset one.
    /// </summary>
    public static void ApplyRoPECachedPartial(
        float* x, float* cosTab, float* sinTab,
        int numHeads, int headDim, int ropeDim)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim", nameof(ropeDim));
        int halfRope = ropeDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            for (int i = 0; i < halfRope; i++)
            {
                float x0 = head[2 * i], x1 = head[2 * i + 1];
                head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
            }
            // Dims [ropeDim, headDim) pass through unchanged — nothing to do.
        }
    }

    /// <summary>
    /// NEOX-style RoPE (used by Qwen, Phi, Gemma, Falcon, etc.):
    /// rotates dim pair (i, i + headDim/2) instead of consecutive (2i, 2i+1).
    /// </summary>
    public static void ApplyRoPECachedNeox(float* x, float* cosTab, float* sinTab, int numHeads, int headDim)
    {
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            int i = 0;
            if (Avx.IsSupported)
            {
                for (; i + 8 <= halfDim; i += 8)
                {
                    var x0 = Avx.LoadVector256(head + i);
                    var x1 = Avx.LoadVector256(head + i + halfDim);
                    var c = Avx.LoadVector256(cosTab + i);
                    var s = Avx.LoadVector256(sinTab + i);
                    var r0 = Fma.MultiplySubtract(x0, c, Avx.Multiply(x1, s));
                    var r1 = Fma.MultiplyAdd(x0, s, Avx.Multiply(x1, c));
                    Avx.Store(head + i, r0);
                    Avx.Store(head + i + halfDim, r1);
                }
            }
            for (; i < halfDim; i++)
            {
                float x0 = head[i], x1 = head[i + halfDim];
                head[i] = x0 * cosTab[i] - x1 * sinTab[i];
                head[i + halfDim] = x0 * sinTab[i] + x1 * cosTab[i];
            }
        }
    }

    /// <summary>
    /// NEOX-style RoPE with PARTIAL rotation. Rotates only the first <paramref name="ropeDim"/>
    /// dims of each head; dims <c>[ropeDim, headDim)</c> pass through unchanged.
    ///
    /// Pair convention: for each head and <c>i ∈ [0, ropeDim/2)</c>, the pair
    /// <c>(x[i], x[i + ropeDim/2])</c> is rotated by <c>(cosTab[i], sinTab[i])</c>.
    /// Both <paramref name="cosTab"/> and <paramref name="sinTab"/> must point at the
    /// per-position slice of a table sized with <c>BuildRopeTable(..., ropeDim, theta)</c>
    /// (i.e. <c>ropeDim/2</c> entries).
    ///
    /// Matches llama.cpp's <c>ggml_compute_forward_rope_flt</c> NEOX path with
    /// <c>n_dims=ropeDim</c>: the tail dims are passed through (see ggml ops.cpp:
    /// "fill the remain channels with data from src tensor").
    /// </summary>
    /// <remarks>
    /// Used by hybrid models with partial RoPE (notably qwen35moe: ropeDim=64, headDim=256).
    /// The scalar inner loop is sufficient for the small ropeDim/2 typical for these models;
    /// SIMD on the partial path is a future optimization.
    /// </remarks>
    public static void ApplyRoPECachedNeoxPartial(
        float* x, float* cosTab, float* sinTab,
        int heads, int headDim, int ropeDim)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim", nameof(ropeDim));
        int halfRope = ropeDim / 2;
        for (int h = 0; h < heads; h++)
        {
            float* head = x + h * headDim;
            for (int i = 0; i < halfRope; i++)
            {
                float x0 = head[i];
                float x1 = head[i + halfRope];
                head[i]            = x0 * cosTab[i] - x1 * sinTab[i];
                head[i + halfRope] = x0 * sinTab[i] + x1 * cosTab[i];
            }
            // Dims [ropeDim, headDim) pass through unchanged — nothing to do.
        }
    }

    public static void ApplyRoPE(float* x, int position, int numHeads, int headDim, float theta)
    {
        int halfDim = headDim / 2;

        // Precompute cos/sin tables (shared across all heads)
        float* cosTab = stackalloc float[halfDim];
        float* sinTab = stackalloc float[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
            float angle = position * freq;
            cosTab[i] = MathF.Cos(angle);
            sinTab[i] = MathF.Sin(angle);
        }

        // Apply rotation to all heads
        // Interleaved pairs: rotate (x[2i], x[2i+1])
        // Reinterpret as pairs and apply rotation using cos/sin tables
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;

            if (Avx.IsSupported && halfDim >= 4)
            {
                // Process 4 pairs (8 floats) at a time
                int i = 0;
                for (; i + 4 <= halfDim; i += 4)
                {
                    // Load 8 consecutive floats: (x0,x1, x2,x3, x4,x5, x6,x7)
                    var v = Avx.LoadVector256(head + 2 * i);
                    var c = Vector256.Create(cosTab[i], cosTab[i], cosTab[i + 1], cosTab[i + 1],
                                             cosTab[i + 2], cosTab[i + 2], cosTab[i + 3], cosTab[i + 3]);
                    var s = Vector256.Create(sinTab[i], sinTab[i], sinTab[i + 1], sinTab[i + 1],
                                             sinTab[i + 2], sinTab[i + 2], sinTab[i + 3], sinTab[i + 3]);

                    // Even elements (x0, x2, x4, x6) and odd elements (x1, x3, x5, x7)
                    // x0' = x0*cos - x1*sin,  x1' = x0*sin + x1*cos
                    // Shuffle to get (x1,x0, x3,x2, x5,x4, x7,x6)
                    var swapped = Avx.Shuffle(v, v, 0b10_11_00_01);
                    // Signs: (-sin, sin, -sin, sin, ...) for even positions,
                    //        (cos, cos, cos, cos, ...) already correct
                    // Actually: result = v*cos + swapped * (-sin_even, sin_odd, ...)
                    var signMask = Vector256.Create(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
                    var sFlipped = Avx.Multiply(s, signMask);
                    var result = Fma.MultiplyAdd(v, c, Avx.Multiply(swapped, sFlipped));
                    Avx.Store(head + 2 * i, result);
                }
                // Scalar remainder
                for (; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
            else
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
        }
    }

    /// <summary>
    /// Precompute RoPE cos/sin tables for all positions [0, maxSeqLen).
    /// cosOut and sinOut must each point to maxSeqLen * (headDim / 2) floats.
    /// </summary>
    public static void BuildRopeTable(float* cosOut, float* sinOut, int maxSeqLen, int headDim, float theta)
        => BuildRopeTable(cosOut, sinOut, maxSeqLen, headDim, theta, null);

    /// <summary>
    /// Variant accepting a per-pair frequency factor array (e.g. Gemma 4
    /// <c>rope_freqs.weight</c> for global layers). When non-null, the raw
    /// inverse frequency is divided by <c>freqFactors[i]</c> for pair i,
    /// so a factor of 1e30 zeros out the rotation for that pair (identity).
    /// </summary>
    public static void BuildRopeTable(float* cosOut, float* sinOut, int maxSeqLen, int headDim, float theta, float* freqFactors)
    {
        int halfDim = headDim / 2;
        float* freqs = stackalloc float[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            float inv = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
            if (freqFactors != null) inv /= freqFactors[i];
            freqs[i] = inv;
        }

        for (int p = 0; p < maxSeqLen; p++)
        {
            float* c = cosOut + (long)p * halfDim;
            float* s = sinOut + (long)p * halfDim;
            for (int i = 0; i < halfDim; i++)
            {
                float angle = p * freqs[i];
                c[i] = MathF.Cos(angle);
                s[i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>
    /// Precompute YaRN-scaled RoPE cos/sin tables for all positions [0, maxSeqLen). Reference:
    /// rope_yarn / rope_yarn_corr_dims / ggml_rope_cache_init in llama.cpp's ggml-cpu/ops.cpp
    /// (MIT-licensed, itself citing https://github.com/jquesnelle/yarn). cosOut/sinOut must each
    /// point to maxSeqLen * (headDim / 2) floats, matching <see cref="BuildRopeTable"/>'s layout
    /// so either can back the same lookup table.
    ///
    /// <para>Degenerates to plain <see cref="BuildRopeTable"/> when <paramref name="freqScale"/> is
    /// 1 and <paramref name="extFactor"/> is 0 (ramp_mix never applies, theta stays the
    /// unscaled extrapolated angle, mscale stays attnFactor) -- callers don't need a separate
    /// "is YaRN active" branch at the call site, just correct (1, 0, 1) defaults when a model has
    /// no rope.scaling metadata.</para>
    /// </summary>
    /// <param name="attnFactor">Baseline magnitude scale (llama.cpp default 1.0 absent an explicit
    /// {arch}.rope.scaling.attn_factor key). NOT the same as ModelHyperparams.RopeYarnLogMul's
    /// separate attention-softmax-only correction (see ModelGraph.cs) -- this one is baked into
    /// the cos/sin table itself.</param>
    public static void BuildYarnRopeTable(float* cosOut, float* sinOut, int maxSeqLen, int headDim,
        float theta, int origCtxLen, float freqScale, float extFactor, float attnFactor,
        float betaFast = 32f, float betaSlow = 1f)
    {
        int halfDim = headDim / 2;

        // rope_yarn_corr_dim: n_dims * log(n_ctx_orig / (n_rot * 2*PI)) / (2*log(base))
        static float CorrDim(int nDims, int nCtxOrig, float nRot, float b) =>
            nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(b));

        float corrLow = 0f, corrHigh = halfDim * 2f - 1f;
        if (origCtxLen > 0)
        {
            corrLow = MathF.Max(0f, MathF.Floor(CorrDim(headDim, origCtxLen, betaFast, theta)));
            corrHigh = MathF.Min(headDim - 1, MathF.Ceiling(CorrDim(headDim, origCtxLen, betaSlow, theta)));
        }

        float thetaScale = MathF.Pow(theta, -2.0f / headDim);

        for (int p = 0; p < maxSeqLen; p++)
        {
            float* c = cosOut + (long)p * halfDim;
            float* s = sinOut + (long)p * halfDim;
            float thetaExtrap = p; // theta_base for position p; scaled by thetaScale each step below
            for (int i = 0; i < halfDim; i++)
            {
                float thetaInterp = freqScale * thetaExtrap;
                float thetaFinal = thetaInterp;
                float mscale = attnFactor;
                if (extFactor != 0f)
                {
                    // rope_yarn_ramp(low, high, i0) with i0 = 2*i (rope_yarn indexes by the
                    // ungrouped dim, this table by the pair index).
                    float y = (i - corrLow) / MathF.Max(0.001f, corrHigh - corrLow);
                    float rampMix = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                    thetaFinal = thetaInterp * (1f - rampMix) + thetaExtrap * rampMix;
                    mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                }
                c[i] = MathF.Cos(thetaFinal) * mscale;
                s[i] = MathF.Sin(thetaFinal) * mscale;
                thetaExtrap *= thetaScale;
            }
        }
    }

    // ================================================================
    //  Single-row dequantization (for embedding lookup)
    // ================================================================

    /// <summary>
    /// Dequantize a single row from a quantized 2D tensor.
    /// rowData points to (cols/blockSize)*bytesPerBlock bytes.
    /// </summary>
    public static void DequantRow(byte* rowData, float* output, int cols, DType dtype)
    {
        Dequantize.ToFloat32(
            new ReadOnlySpan<byte>(rowData, (cols / DTypeInfo.BlockSize(dtype)) * DTypeInfo.BytesPerBlock(dtype)),
            new Span<float>(output, cols),
            dtype, cols);
    }

    // ================================================================
    //  Helpers
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v)
    {
        var hi = Avx.ExtractVector128(v, 1);
        var lo = v.GetLower();
        var sum = Sse.Add(lo, hi);
        sum = Sse.Add(sum, Sse.MoveHighToLow(sum, sum));
        sum = Sse.AddScalar(sum, Sse.Shuffle(sum, sum, 1));
        return sum.ToScalar();
    }

    /// <summary>Horizontal sum of a Vector256&lt;int&gt; to a single int (exact, no FP).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HSumI32_256(Vector256<int> v)
    {
        var lo = v.GetLower();
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse2.Add(lo, hi);
        s = Sse2.Add(s, Sse2.Shuffle(s, 0x4E)); // [2,3,0,1]
        s = Sse2.Add(s, Sse2.Shuffle(s, 0xB1)); // [1,0,3,2]
        return s.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HMax256(Vector256<float> v)
    {
        var hi = Avx.ExtractVector128(v, 1);
        var lo = v.GetLower();
        var m = Sse.Max(lo, hi);
        m = Sse.Max(m, Sse.MoveHighToLow(m, m));
        m = Sse.Max(m, Sse.Shuffle(m, m, 1));
        return m.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }

    // ================================================================
    //  Q4_K 8-row repack (perf-loop iteration 39) — the layout that lets a
    //  single AVX2 load hold 8 bytes from each of 4 weight ROWS, so one
    //  maddubs covers 8 elements x 4 rows and four of them chain into a
    //  32-element sub-block. Ported from llama.cpp's block_q4_Kx8
    //  (ggml/src/ggml-cpu/repack.cpp, make_block_q4_Kx8).
    // ================================================================

    /// <summary>Bytes per repacked group of 8 Q4_K rows, one 256-element super-block deep.</summary>
    public const int Q4Kx8BlockBytes = 32 + 32 + 128 + 1024;   // d[8], dmin[8], sc/mn, qs

    /// <summary>
    /// Repack 8 consecutive Q4_K weight rows into the interleaved <c>block_q4_Kx8</c> form.
    ///
    /// <para>Layout, per 256-element super-block (<see cref="Q4Kx8BlockBytes"/> bytes):
    /// <list type="bullet">
    /// <item><c>float d[8]</c> at 0 — the 8 rows' super-block scales.</item>
    /// <item><c>float dmin[8]</c> at 32.</item>
    /// <item><c>byte sc[64]</c> at 64 then <c>byte mn[64]</c> at 128 — the 6-bit scales/mins
    /// ALREADY DECODED, laid out sub-block-major as <c>[subblock 0..7][row 0..7]</c>. llama.cpp
    /// keeps them re-bit-packed and unpacks inside the kernel; decoding here instead removes
    /// <c>GetScaleMinK4</c> from the hot loop entirely and lets the kernel fetch 8 rows' scales
    /// for one sub-block as a single 8-byte read.</item>
    /// <item><c>byte qs[1024]</c> at 192 — the 8 rows' 128 nibble-bytes each, interleaved in
    /// 8-byte round-robin: row0[0..7], row1[0..7], ... row7[0..7], row0[8..15], ...</item>
    /// </list></para>
    ///
    /// <para>.NET note: <c>d</c>/<c>dmin</c> are stored as <c>float</c>, not <c>ggml_half</c>.
    /// C# has no F16C intrinsic, so keeping halves would force a conversion per super-block in the
    /// hot loop; 64 bytes per 1216-byte block is the better trade and is strictly cheaper than
    /// what llama.cpp pays.</para>
    /// </summary>
    /// <param name="src">First of 8 consecutive Q4_K rows.</param>
    /// <param name="dst">Destination, numBlocks x <see cref="Q4Kx8BlockBytes"/> bytes.</param>
    /// <param name="numBlocks">256-element super-blocks per row (cols / 256).</param>
    /// <param name="srcRowStrideBytes">Byte stride between consecutive source rows.</param>
    public static void RepackQ4K8Rows(byte* src, byte* dst, int numBlocks, long srcRowStrideBytes)
    {
        for (int b = 0; b < numBlocks; b++)
        {
            byte* o = dst + (long)b * Q4Kx8BlockBytes;
            float* dOut = (float*)o;
            float* dminOut = (float*)(o + 32);
            byte* scOut = o + 64;
            byte* mnOut = o + 128;
            byte* qsOut = o + 192;   // 32 d + 32 dmin + 64 sc + 64 mn

            for (int r = 0; r < 8; r++)
            {
                byte* x = src + r * srcRowStrideBytes + (long)b * 144;
                dOut[r] = HalfToFloat(x[0], x[1]);
                dminOut[r] = HalfToFloat(x[2], x[3]);

                byte* sc = x + 4;
                for (int j = 0; j < 8; j++)
                {
                    GetScaleMinK4(j, sc, out byte s, out byte m);
                    scOut[j * 8 + r] = s;
                    mnOut[j * 8 + r] = m;
                }
            }

            // 8-byte round-robin interleave of each row's 128 nibble-bytes.
            for (int i = 0; i < 128; i++)
            {
                int srcRow = i & 7;
                int srcOff = (i >> 3) * 8;
                byte* from = src + srcRow * srcRowStrideBytes + (long)b * 144 + 16 + srcOff;
                Buffer.MemoryCopy(from, qsOut + (long)i * 8, 8, 8);
            }
        }
    }


    /// <summary>
    /// AVX2 dot of a repacked 8-row Q4_K group against one Q8_KS-quantized token, producing 8 row
    /// results (perf-loop iteration 40).
    ///
    /// <para><b>Why this is faster than eight row-major dots.</b> The
    /// <see cref="RepackQ4K8Rows"/> interleave puts the ROW dimension in the vector lanes: a
    /// 32-byte load at <c>qs + cg*64 + g*32</c> holds 8 source bytes from each of 4 rows. One
    /// <c>maddubs</c> against the same 8 activation bytes (broadcast 4x via a 64-bit
    /// <c>Vector256.Create</c>) therefore covers 8 elements x 4 rows, and chaining four of them in
    /// int16 covers a full 32-element sub-block for 4 rows at once. A single <c>madd_epi16</c> then
    /// applies each row's 6-bit scale from its own lane and widens to int32 — after which row r of
    /// the group occupies int32 lanes 2r and 2r+1, so the per-row super-block scale <c>d[r]</c> can
    /// be applied lane-wise too.</para>
    ///
    /// <para>Net: ~10 instructions per (sub-block, 4 rows) = 128 MACs, against the row-major path's
    /// 4 instructions per (sub-block, 1 row) = 32 MACs. The remaining <c>cvt</c>+<c>fma</c> per
    /// sub-block is forced by Q8_KS carrying eight activation scales per super-block; Q8_K's single
    /// scale would let the int32 accumulate across all eight sub-blocks instead.</para>
    /// </summary>
    public static void DotQ4Kx8_Q8KS_Avx2(byte* packed, byte* scratch, int numBlocks, float* outRows)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        // Two row groups of 4: group g covers rows 4g..4g+3.
        var facc0 = Vector256<float>.Zero;
        var facc1 = Vector256<float>.Zero;
        var fmin0 = Vector256<float>.Zero;
        var fmin1 = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* o = packed + (long)b * Q4Kx8BlockBytes;
            float* d = (float*)o;
            float* dmin = (float*)(o + 32);
            byte* scAll = o + 64;
            byte* mnAll = o + 128;
            byte* qs = o + 192;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            for (int j = 0; j < 8; j++)
            {
                int chunk = j >> 1;
                bool low = (j & 1) == 0;
                byte* scJ = scAll + j * 8;
                byte* mnJ = mnAll + j * 8;
                int bsum = (int)bsums[2 * j] + (int)bsums[2 * j + 1];
                float ds = dSub[j];

                for (int g = 0; g < 2; g++)
                {
                    int r0 = g * 4;
                    var iacc = Vector256<short>.Zero;

                    for (int t = 0; t < 4; t++)
                    {
                        // 4 rows x 8 source bytes for this 8-element slice of the sub-block.
                        var v = Vector256.LoadUnsafe(ref *(qs + (long)(chunk * 4 + t) * 64 + g * 32));
                        var nib = low
                            ? Avx2.And(v, m0F)
                            : Avx2.And(Avx2.ShiftRightLogical(v.AsInt16(), 4).AsByte(), m0F);
                        // The same 8 activation bytes, replicated to all four 64-bit lanes.
                        long actBits = Unsafe.ReadUnaligned<long>(q8 + j * 32 + t * 8);
                        var act = Vector256.Create(actBits).AsSByte();
                        iacc = Avx2.Add(iacc, Avx2.MultiplyAddAdjacent(nib, act));
                    }

                    // Row r of the group sits in int16 lanes 4r..4r+3, so its 6-bit scale is
                    // replicated four times; madd_epi16 then folds pairs into int32 lanes 2r,2r+1.
                    short s0 = scJ[r0], s1 = scJ[r0 + 1], s2 = scJ[r0 + 2], s3 = scJ[r0 + 3];
                    var scaleVec = Vector256.Create(s0, s0, s0, s0, s1, s1, s1, s1,
                                                    s2, s2, s2, s2, s3, s3, s3, s3);
                    var i32 = Avx2.MultiplyAddAdjacent(iacc, scaleVec);

                    // d[r] applied lane-wise (row r -> int32 lanes 2r, 2r+1).
                    float e0 = d[r0], e1 = d[r0 + 1], e2 = d[r0 + 2], e3 = d[r0 + 3];
                    var dVec = Avx.Multiply(Vector256.Create(e0, e0, e1, e1, e2, e2, e3, e3),
                                            Vector256.Create(ds));

                    // Formulated to match DotQ4Kx8_Q8KS_8In TERM FOR TERM, not merely to be
                    // mathematically equal. The driver uses the 8-token kernel for full groups and
                    // this one for the ragged tail, so any difference in association makes a
                    // prompt's result depend on its token count — which through 24 layers amplifies
                    // a 1e-6 kernel difference into a ~0.25 logit difference and breaks
                    // chunked-vs-full prefill equality. Same discipline as the row-major
                    // _8In/_4In/_2In/single family (perf-loop iteration 38).
                    float g0 = dmin[r0] * mnJ[r0], g1 = dmin[r0 + 1] * mnJ[r0 + 1],
                          g2 = dmin[r0 + 2] * mnJ[r0 + 2], g3 = dmin[r0 + 3] * mnJ[r0 + 3];
                    var mRow = Vector256.Create(g0, 0f, g1, 0f, g2, 0f, g3, 0f);
                    var mScale = Vector256.Create(ds * bsum);

                    if (g == 0)
                    {
                        facc0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i32), dVec, facc0);
                        fmin0 = Fma.MultiplyAdd(mRow, mScale, fmin0);
                    }
                    else
                    {
                        facc1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i32), dVec, facc1);
                        fmin1 = Fma.MultiplyAdd(mRow, mScale, fmin1);
                    }
                }
            }
        }

        // Row r of a group = lanes 2r + 2r+1 of that group's accumulator, minus its min term.
        for (int k = 0; k < 4; k++)
        {
            outRows[k] = facc0.GetElement(2 * k) + facc0.GetElement(2 * k + 1) - fmin0.GetElement(2 * k);
            outRows[4 + k] = facc1.GetElement(2 * k) + facc1.GetElement(2 * k + 1) - fmin1.GetElement(2 * k);
        }
    }


    /// <summary>
    /// Repacked 8-row x 8-token Q4_K dot (perf-loop iteration 41) — the fair counterpart to
    /// <see cref="DotQ4K_Q8KS_8In"/>.
    ///
    /// <para>The single-token <see cref="DotQ4Kx8_Q8KS_Avx2"/> amortises the weight decode over 8
    /// ROWS; the existing <c>_8In</c> amortises it over 8 TOKENS. Only a kernel doing both at once
    /// is a like-for-like comparison, which is what this is: the nibble extraction for a
    /// (sub-block, row-group, slice) happens ONCE and feeds eight tokens' <c>maddubs</c>.</para>
    ///
    /// <para>Register pressure is the known risk: 16 float accumulators (8 tokens x 2 row groups)
    /// plus 8 int16 accumulators is 24 live vectors against 16 architectural YMM, so RyuJIT will
    /// spill. Named locals are used rather than arrays because RyuJIT will not keep a
    /// <c>Vector256[]</c> in registers at all. Whether the spill traffic eats the instruction win
    /// is exactly what the benchmark has to answer.</para>
    /// </summary>
    public static void DotQ4Kx8_Q8KS_8In(byte* packed,
        byte* s0, byte* s1, byte* s2, byte* s3, byte* s4, byte* s5, byte* s6, byte* s7,
        int numBlocks, float* outRows)
    {
        float* dArr0 = (float*)s0; sbyte* qs0 = (sbyte*)(s0 + numBlocks * 32); short* bs0 = (short*)(s0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)s1; sbyte* qs1 = (sbyte*)(s1 + numBlocks * 32); short* bs1 = (short*)(s1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)s2; sbyte* qs2 = (sbyte*)(s2 + numBlocks * 32); short* bs2 = (short*)(s2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)s3; sbyte* qs3 = (sbyte*)(s3 + numBlocks * 32); short* bs3 = (short*)(s3 + numBlocks * 32 + numBlocks * 256);
        float* dArr4 = (float*)s4; sbyte* qs4 = (sbyte*)(s4 + numBlocks * 32); short* bs4 = (short*)(s4 + numBlocks * 32 + numBlocks * 256);
        float* dArr5 = (float*)s5; sbyte* qs5 = (sbyte*)(s5 + numBlocks * 32); short* bs5 = (short*)(s5 + numBlocks * 32 + numBlocks * 256);
        float* dArr6 = (float*)s6; sbyte* qs6 = (sbyte*)(s6 + numBlocks * 32); short* bs6 = (short*)(s6 + numBlocks * 32 + numBlocks * 256);
        float* dArr7 = (float*)s7; sbyte* qs7 = (sbyte*)(s7 + numBlocks * 32); short* bs7 = (short*)(s7 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var fa0_0 = Vector256<float>.Zero; var fm0_0 = Vector256<float>.Zero;
        var fa0_1 = Vector256<float>.Zero; var fm0_1 = Vector256<float>.Zero;
        var fa0_2 = Vector256<float>.Zero; var fm0_2 = Vector256<float>.Zero;
        var fa0_3 = Vector256<float>.Zero; var fm0_3 = Vector256<float>.Zero;
        var fa0_4 = Vector256<float>.Zero; var fm0_4 = Vector256<float>.Zero;
        var fa0_5 = Vector256<float>.Zero; var fm0_5 = Vector256<float>.Zero;
        var fa0_6 = Vector256<float>.Zero; var fm0_6 = Vector256<float>.Zero;
        var fa0_7 = Vector256<float>.Zero; var fm0_7 = Vector256<float>.Zero;
        var fa1_0 = Vector256<float>.Zero; var fm1_0 = Vector256<float>.Zero;
        var fa1_1 = Vector256<float>.Zero; var fm1_1 = Vector256<float>.Zero;
        var fa1_2 = Vector256<float>.Zero; var fm1_2 = Vector256<float>.Zero;
        var fa1_3 = Vector256<float>.Zero; var fm1_3 = Vector256<float>.Zero;
        var fa1_4 = Vector256<float>.Zero; var fm1_4 = Vector256<float>.Zero;
        var fa1_5 = Vector256<float>.Zero; var fm1_5 = Vector256<float>.Zero;
        var fa1_6 = Vector256<float>.Zero; var fm1_6 = Vector256<float>.Zero;
        var fa1_7 = Vector256<float>.Zero; var fm1_7 = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* o = packed + (long)b * Q4Kx8BlockBytes;
            float* d = (float*)o;
            float* dmin = (float*)(o + 32);
            byte* scAll = o + 64;
            byte* mnAll = o + 128;
            byte* qsW = o + 192;

            float* dS0 = dArr0 + b * 8; sbyte* q0 = qs0 + b * 256; short* bsum0 = bs0 + b * 16;
            float* dS1 = dArr1 + b * 8; sbyte* q1 = qs1 + b * 256; short* bsum1 = bs1 + b * 16;
            float* dS2 = dArr2 + b * 8; sbyte* q2 = qs2 + b * 256; short* bsum2 = bs2 + b * 16;
            float* dS3 = dArr3 + b * 8; sbyte* q3 = qs3 + b * 256; short* bsum3 = bs3 + b * 16;
            float* dS4 = dArr4 + b * 8; sbyte* q4 = qs4 + b * 256; short* bsum4 = bs4 + b * 16;
            float* dS5 = dArr5 + b * 8; sbyte* q5 = qs5 + b * 256; short* bsum5 = bs5 + b * 16;
            float* dS6 = dArr6 + b * 8; sbyte* q6 = qs6 + b * 256; short* bsum6 = bs6 + b * 16;
            float* dS7 = dArr7 + b * 8; sbyte* q7 = qs7 + b * 256; short* bsum7 = bs7 + b * 16;

            for (int j = 0; j < 8; j++)
            {
                int chunk = j >> 1;
                bool low = (j & 1) == 0;
                byte* scJ = scAll + j * 8;
                byte* mnJ = mnAll + j * 8;

                int bsv0 = (int)bsum0[2 * j] + (int)bsum0[2 * j + 1]; float ds0 = dS0[j];
                int bsv1 = (int)bsum1[2 * j] + (int)bsum1[2 * j + 1]; float ds1 = dS1[j];
                int bsv2 = (int)bsum2[2 * j] + (int)bsum2[2 * j + 1]; float ds2 = dS2[j];
                int bsv3 = (int)bsum3[2 * j] + (int)bsum3[2 * j + 1]; float ds3 = dS3[j];
                int bsv4 = (int)bsum4[2 * j] + (int)bsum4[2 * j + 1]; float ds4 = dS4[j];
                int bsv5 = (int)bsum5[2 * j] + (int)bsum5[2 * j + 1]; float ds5 = dS5[j];
                int bsv6 = (int)bsum6[2 * j] + (int)bsum6[2 * j + 1]; float ds6 = dS6[j];
                int bsv7 = (int)bsum7[2 * j] + (int)bsum7[2 * j + 1]; float ds7 = dS7[j];

                for (int g = 0; g < 2; g++)
                {
                    int r0 = g * 4;
                    var ia0 = Vector256<short>.Zero;
                    var ia1 = Vector256<short>.Zero;
                    var ia2 = Vector256<short>.Zero;
                    var ia3 = Vector256<short>.Zero;
                    var ia4 = Vector256<short>.Zero;
                    var ia5 = Vector256<short>.Zero;
                    var ia6 = Vector256<short>.Zero;
                    var ia7 = Vector256<short>.Zero;

                    for (int t = 0; t < 4; t++)
                    {
                        var v = Vector256.LoadUnsafe(ref *(qsW + (long)(chunk * 4 + t) * 64 + g * 32));
                        // Decoded ONCE here, then reused by all eight tokens below.
                        var nib = low
                            ? Avx2.And(v, m0F)
                            : Avx2.And(Avx2.ShiftRightLogical(v.AsInt16(), 4).AsByte(), m0F);
                        int off = j * 32 + t * 8;
                        ia0 = Avx2.Add(ia0, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q0 + off)).AsSByte()));
                        ia1 = Avx2.Add(ia1, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q1 + off)).AsSByte()));
                        ia2 = Avx2.Add(ia2, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q2 + off)).AsSByte()));
                        ia3 = Avx2.Add(ia3, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q3 + off)).AsSByte()));
                        ia4 = Avx2.Add(ia4, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q4 + off)).AsSByte()));
                        ia5 = Avx2.Add(ia5, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q5 + off)).AsSByte()));
                        ia6 = Avx2.Add(ia6, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q6 + off)).AsSByte()));
                        ia7 = Avx2.Add(ia7, Avx2.MultiplyAddAdjacent(nib, Vector256.Create(Unsafe.ReadUnaligned<long>(q7 + off)).AsSByte()));
                    }

                    short c0 = scJ[r0], c1 = scJ[r0 + 1], c2 = scJ[r0 + 2], c3 = scJ[r0 + 3];
                    var scaleVec = Vector256.Create(c0, c0, c0, c0, c1, c1, c1, c1,
                                                    c2, c2, c2, c2, c3, c3, c3, c3);
                    float e0 = d[r0], e1 = d[r0 + 1], e2 = d[r0 + 2], e3 = d[r0 + 3];
                    var dRow = Vector256.Create(e0, e0, e1, e1, e2, e2, e3, e3);
                    float g0 = dmin[r0] * mnJ[r0], g1 = dmin[r0 + 1] * mnJ[r0 + 1],
                          g2 = dmin[r0 + 2] * mnJ[r0 + 2], g3 = dmin[r0 + 3] * mnJ[r0 + 3];
                    var mRow = Vector256.Create(g0, 0f, g1, 0f, g2, 0f, g3, 0f);

                    if (g == 0)
                    {
                        fa0_0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia0, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds0)), fa0_0);
                        fm0_0 = Fma.MultiplyAdd(mRow, Vector256.Create(ds0 * bsv0), fm0_0);
                        fa0_1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia1, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds1)), fa0_1);
                        fm0_1 = Fma.MultiplyAdd(mRow, Vector256.Create(ds1 * bsv1), fm0_1);
                        fa0_2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia2, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds2)), fa0_2);
                        fm0_2 = Fma.MultiplyAdd(mRow, Vector256.Create(ds2 * bsv2), fm0_2);
                        fa0_3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia3, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds3)), fa0_3);
                        fm0_3 = Fma.MultiplyAdd(mRow, Vector256.Create(ds3 * bsv3), fm0_3);
                        fa0_4 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia4, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds4)), fa0_4);
                        fm0_4 = Fma.MultiplyAdd(mRow, Vector256.Create(ds4 * bsv4), fm0_4);
                        fa0_5 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia5, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds5)), fa0_5);
                        fm0_5 = Fma.MultiplyAdd(mRow, Vector256.Create(ds5 * bsv5), fm0_5);
                        fa0_6 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia6, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds6)), fa0_6);
                        fm0_6 = Fma.MultiplyAdd(mRow, Vector256.Create(ds6 * bsv6), fm0_6);
                        fa0_7 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia7, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds7)), fa0_7);
                        fm0_7 = Fma.MultiplyAdd(mRow, Vector256.Create(ds7 * bsv7), fm0_7);
                    }
                    else
                    {
                        fa1_0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia0, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds0)), fa1_0);
                        fm1_0 = Fma.MultiplyAdd(mRow, Vector256.Create(ds0 * bsv0), fm1_0);
                        fa1_1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia1, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds1)), fa1_1);
                        fm1_1 = Fma.MultiplyAdd(mRow, Vector256.Create(ds1 * bsv1), fm1_1);
                        fa1_2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia2, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds2)), fa1_2);
                        fm1_2 = Fma.MultiplyAdd(mRow, Vector256.Create(ds2 * bsv2), fm1_2);
                        fa1_3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia3, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds3)), fa1_3);
                        fm1_3 = Fma.MultiplyAdd(mRow, Vector256.Create(ds3 * bsv3), fm1_3);
                        fa1_4 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia4, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds4)), fa1_4);
                        fm1_4 = Fma.MultiplyAdd(mRow, Vector256.Create(ds4 * bsv4), fm1_4);
                        fa1_5 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia5, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds5)), fa1_5);
                        fm1_5 = Fma.MultiplyAdd(mRow, Vector256.Create(ds5 * bsv5), fm1_5);
                        fa1_6 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia6, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds6)), fa1_6);
                        fm1_6 = Fma.MultiplyAdd(mRow, Vector256.Create(ds6 * bsv6), fm1_6);
                        fa1_7 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(ia7, scaleVec)), Avx.Multiply(dRow, Vector256.Create(ds7)), fa1_7);
                        fm1_7 = Fma.MultiplyAdd(mRow, Vector256.Create(ds7 * bsv7), fm1_7);
                    }
                }
            }
        }

        // outRows is [token][row]: row r of group g lives in int32 lanes 2r/2r+1.
        for (int k = 0; k < 4; k++)
        {
            outRows[0 * 8 + k] = fa0_0.GetElement(2 * k) + fa0_0.GetElement(2 * k + 1) - fm0_0.GetElement(2 * k);
            outRows[0 * 8 + 4 + k] = fa1_0.GetElement(2 * k) + fa1_0.GetElement(2 * k + 1) - fm1_0.GetElement(2 * k);
            outRows[1 * 8 + k] = fa0_1.GetElement(2 * k) + fa0_1.GetElement(2 * k + 1) - fm0_1.GetElement(2 * k);
            outRows[1 * 8 + 4 + k] = fa1_1.GetElement(2 * k) + fa1_1.GetElement(2 * k + 1) - fm1_1.GetElement(2 * k);
            outRows[2 * 8 + k] = fa0_2.GetElement(2 * k) + fa0_2.GetElement(2 * k + 1) - fm0_2.GetElement(2 * k);
            outRows[2 * 8 + 4 + k] = fa1_2.GetElement(2 * k) + fa1_2.GetElement(2 * k + 1) - fm1_2.GetElement(2 * k);
            outRows[3 * 8 + k] = fa0_3.GetElement(2 * k) + fa0_3.GetElement(2 * k + 1) - fm0_3.GetElement(2 * k);
            outRows[3 * 8 + 4 + k] = fa1_3.GetElement(2 * k) + fa1_3.GetElement(2 * k + 1) - fm1_3.GetElement(2 * k);
            outRows[4 * 8 + k] = fa0_4.GetElement(2 * k) + fa0_4.GetElement(2 * k + 1) - fm0_4.GetElement(2 * k);
            outRows[4 * 8 + 4 + k] = fa1_4.GetElement(2 * k) + fa1_4.GetElement(2 * k + 1) - fm1_4.GetElement(2 * k);
            outRows[5 * 8 + k] = fa0_5.GetElement(2 * k) + fa0_5.GetElement(2 * k + 1) - fm0_5.GetElement(2 * k);
            outRows[5 * 8 + 4 + k] = fa1_5.GetElement(2 * k) + fa1_5.GetElement(2 * k + 1) - fm1_5.GetElement(2 * k);
            outRows[6 * 8 + k] = fa0_6.GetElement(2 * k) + fa0_6.GetElement(2 * k + 1) - fm0_6.GetElement(2 * k);
            outRows[6 * 8 + 4 + k] = fa1_6.GetElement(2 * k) + fa1_6.GetElement(2 * k + 1) - fm1_6.GetElement(2 * k);
            outRows[7 * 8 + k] = fa0_7.GetElement(2 * k) + fa0_7.GetElement(2 * k + 1) - fm0_7.GetElement(2 * k);
            outRows[7 * 8 + 4 + k] = fa1_7.GetElement(2 * k) + fa1_7.GetElement(2 * k + 1) - fm1_7.GetElement(2 * k);
        }
    }


    /// <summary>Bytes needed to hold a whole Q4_K matrix in the 8-row repacked form.</summary>
    public static long Q4Kx8PackedBytes(int rows, int cols)
        => (long)(rows / 8) * (cols / 256) * Q4Kx8BlockBytes;

    /// <summary>Whether a matrix shape can use the repacked path at all.</summary>
    public static bool CanRepackQ4Kx8(int rows, int cols)
        => Avx2.IsSupported && Fma.IsSupported && rows % 8 == 0 && cols % 256 == 0 && rows > 0 && cols > 0;

    /// <summary>
    /// Repack a whole row-major Q4_K weight matrix into consecutive 8-row groups. Group g holds
    /// rows 8g..8g+7 and occupies <c>(cols/256) * Q4Kx8BlockBytes</c> bytes.
    /// </summary>
    public static void RepackQ4KMatrix(byte* src, byte* dst, int rows, int cols)
    {
        int numBlocks = cols / 256;
        long bytesPerRow = (long)numBlocks * 144;
        long bytesPerGroup = (long)numBlocks * Q4Kx8BlockBytes;
        int groups = rows / 8;
        Parallel.For(0, groups, g =>
        {
            RepackQ4K8Rows(src + (long)g * 8 * bytesPerRow,
                           dst + (long)g * bytesPerGroup,
                           numBlocks, bytesPerRow);
        });
    }

    /// <summary>
    /// Batched prefill matmul over a repacked Q4_K matrix (perf-loop iteration 42).
    ///
    /// <para>Measured 2.6x over <see cref="DotQ4K_Q8KS_8In"/> at the trunk's Q4_K shape
    /// (cols = 2048) — the row interleave lets one nibble decode feed 4 rows, on top of the
    /// existing 8-token amortisation. Falls back to the caller's path when the shape or ISA does
    /// not qualify.</para>
    ///
    /// <para>Output layout matches <see cref="MatMulBatched"/>: <c>output[token * rows + row]</c>.</para>
    /// </summary>
    public static bool TryMatMulBatchedQ4Kx8(float* output, byte* packed, float* input,
        int batchSize, int rows, int cols)
    {
        if (!CanRepackQ4Kx8(rows, cols) || batchSize < 1) return false;

        // Path 2 (docs/repack-gemm/port-log.md) is the literal port of llama.cpp's AVX2
        // ggml_gemm_q4_K_8x8_q8_K, and is the DEFAULT since 2026-08-02 (1.83x isolated, 1.11x
        // end-to-end, PPL no worse — see GemmPathConfig.ReadFromEnvironment). STINGRAY_GEMM_PATH=1
        // opts back out. It may decline any shape it does not yet handle, in which case we fall
        // through to Path 1 below — so an incomplete Path 2 can only cost speed, never correctness.
        if (GemmPathConfig.UsePath2 &&
            RepackedGemmPath2.TryMatMulBatched(output, packed, input, batchSize, rows, cols))
            return true;

        const int MaxChunk = 512;
        if (batchSize > MaxChunk)
        {
            bool ok = true;
            for (int start = 0; start < batchSize; start += MaxChunk)
            {
                int n = Math.Min(MaxChunk, batchSize - start);
                ok &= TryMatMulBatchedQ4Kx8(output + (long)start * rows, packed,
                                            input + (long)start * cols, n, rows, cols);
            }
            return ok;
        }

        int numBlocks = cols / 256;
        int scratchPerToken = Q8KSScratchBytes(cols);
        long bytesPerGroup = (long)numBlocks * Q4Kx8BlockBytes;
        int groups = rows / 8;

        byte* scratchBase = (byte*)NativeMemory.Alloc((nuint)((long)scratchPerToken * batchSize));
        try
        {
            if (batchSize >= 4)
                Parallel.For(0, batchSize, n =>
                    QuantizeRowToQ8KS(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken));
            else
                for (int n = 0; n < batchSize; n++)
                    QuantizeRowToQ8KS(input + (long)n * cols, cols, scratchBase + (long)n * scratchPerToken);

            Parallel.For(0, groups, g =>
            {
                byte* pk = packed + (long)g * bytesPerGroup;
                int rowBase = g * 8;
                float* tmp = stackalloc float[64];   // 8 tokens x 8 rows

                int t = 0;
                for (; t + 8 <= batchSize; t += 8)
                {
                    byte* a0 = scratchBase + (long)(t + 0) * scratchPerToken;
                    byte* a1 = scratchBase + (long)(t + 1) * scratchPerToken;
                    byte* a2 = scratchBase + (long)(t + 2) * scratchPerToken;
                    byte* a3 = scratchBase + (long)(t + 3) * scratchPerToken;
                    byte* a4 = scratchBase + (long)(t + 4) * scratchPerToken;
                    byte* a5 = scratchBase + (long)(t + 5) * scratchPerToken;
                    byte* a6 = scratchBase + (long)(t + 6) * scratchPerToken;
                    byte* a7 = scratchBase + (long)(t + 7) * scratchPerToken;
                    DotQ4Kx8_Q8KS_8In(pk, a0, a1, a2, a3, a4, a5, a6, a7, numBlocks, tmp);
                    for (int k = 0; k < 8; k++)
                        for (int r = 0; r < 8; r++)
                            output[(long)(t + k) * rows + rowBase + r] = tmp[k * 8 + r];
                }
                // Ragged token tail: the single-token repacked kernel still gets the 8-row win.
                for (; t < batchSize; t++)
                {
                    DotQ4Kx8_Q8KS_Avx2(pk, scratchBase + (long)t * scratchPerToken, numBlocks, tmp);
                    for (int r = 0; r < 8; r++)
                        output[(long)t * rows + rowBase + r] = tmp[r];
                }
            });
        }
        finally
        {
            NativeMemory.Free(scratchBase);
        }
        return true;
    }

    /// <summary>
    /// Reference (scalar) dot of a repacked 8-row group against one Q8_KS-quantized token,
    /// producing 8 row results. Exists to prove the <see cref="RepackQ4K8Rows"/> layout
    /// independently of any vectorised kernel: it walks the interleaved bytes by construction, so
    /// if the layout is wrong this diverges from the row-major path immediately.
    /// </summary>
    public static void DotQ4Kx8_Q8KS_Scalar(byte* packed, byte* scratch, int numBlocks, float* outRows)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        for (int r = 0; r < 8; r++) outRows[r] = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* o = packed + (long)b * Q4Kx8BlockBytes;
            float* d = (float*)o;
            float* dmin = (float*)(o + 32);
            byte* scAll = o + 64;
            byte* mnAll = o + 128;
            byte* qs = o + 192;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            for (int j = 0; j < 8; j++)          // sub-block
            {
                int bsum = (int)bsums[2 * j] + (int)bsums[2 * j + 1];
                for (int r = 0; r < 8; r++)
                {
                    int acc = 0;
                    for (int e = 0; e < 32; e++)  // element within the sub-block
                    {
                        // Q4_K element mapping, matching DotQ4K_Q8KS_Avx2: byte qs[chunk*32 + e]
                        // holds element chunk*64+e in its LOW nibble (sub-block 2*chunk) and
                        // element chunk*64+32+e in its HIGH nibble (sub-block 2*chunk+1). So the
                        // nibble half is selected by j & 1, not by j < 4.
                        int chunk = j >> 1;
                        int within = chunk * 32 + e;
                        int chunkIdx = within >> 3;            // which 8-byte interleave chunk
                        int inChunk = within & 7;
                        byte packedByte = qs[(chunkIdx * 8 + r) * 8 + inChunk];
                        int nib = ((j & 1) == 0) ? (packedByte & 0xF) : (packedByte >> 4);
                        acc += nib * q8[j * 32 + e];
                    }
                    outRows[r] += dSub[j] * (d[r] * scAll[j * 8 + r] * acc
                                             - dmin[r] * mnAll[j * 8 + r] * bsum);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GetScaleMinK4(int j, byte* q, out byte scale, out byte min)
    {
        if (j < 4)
        {
            scale = (byte)(q[j] & 63);
            min = (byte)(q[j + 4] & 63);
        }
        else
        {
            scale = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            min = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    /// <summary>Load 8 bytes into a Vector128 for vpmovzxbd.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> LoadBytes8(byte* ptr)
    {
        return Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(ptr)).AsByte();
    }

    /// <summary>
    /// Fast exp approximation for Vector256 using the standard
    /// Cephes-style range reduction + polynomial.
    /// Max relative error ~1.5e-7 in [-87, 88].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ExpApprox256(Vector256<float> x)
    {
        // Clamp to avoid overflow/underflow
        x = Avx.Max(x, Vector256.Create(-87.3365f));
        x = Avx.Min(x, Vector256.Create(88.7228f));

        // t = x / ln(2)
        var t = Avx.Multiply(x, Vector256.Create(1.44269504088896341f));

        // Round to nearest integer
        var ti = Avx.RoundToNearestInteger(t);
        var n = Avx.ConvertToVector256Int32(ti);

        // Fractional part: f = t - round(t)
        var f = Avx.Subtract(t, ti);

        // Polynomial approximation of 2^f on [-0.5, 0.5]
        // Coefficients: Taylor series of 2^x = e^(x·ln2), i.e. (ln2)^k / k!
        var p = Vector256.Create(1.5403530e-4f);
        p = Fma.MultiplyAdd(p, f, Vector256.Create(1.3333558e-3f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(9.6181291e-3f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(5.5504109e-2f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(2.4022651e-1f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(6.9314718e-1f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(1.0f));

        // 2^n via IEEE 754 exponent manipulation
        var pow2n = Avx2.ShiftLeftLogical(Avx2.Add(n, Vector256.Create(127)), 23).AsSingle();
        return Avx.Multiply(p, pow2n);
    }

    public static void AttentionSwa(float* q, float* kCache, float* vCache, float* output,
                                   float* scoresScratch, int position, int windowSize,
                                   int headDim, int numHeads, int numKvHeads, int maxSeqLen)
    {
        int startPos = Math.Max(0, position + 1 - windowSize);
        int windowLen = Math.Min(position + 1 - startPos, maxSeqLen);
        if (windowLen <= 0) return;
        float scale = 1.0f / MathF.Sqrt(headDim);

        int headsPerKv = numHeads / numKvHeads;

        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / headsPerKv;
            float* qHead = q + (long)h * headDim;
            float* outHead = output + (long)h * headDim;

            float maxScore = float.NegativeInfinity;
            for (int w = 0; w < windowLen; w++)
            {
                int p = (startPos + w) % maxSeqLen;
                float* kPtr = kCache + ((long)p * numKvHeads + kvHead) * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    dot += qHead[d] * kPtr[d];
                }
                dot *= scale;
                if (scoresScratch != null) scoresScratch[h * windowLen + w] = dot;
                if (dot > maxScore) maxScore = dot;
            }

            float sumExp = 0f;
            for (int w = 0; w < windowLen; w++)
            {
                float s = scoresScratch != null
                    ? scoresScratch[h * windowLen + w]
                    : 0f;
                if (scoresScratch == null)
                {
                    int p = (startPos + w) % maxSeqLen;
                    float* kPtr = kCache + ((long)p * numKvHeads + kvHead) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qHead[d] * kPtr[d];
                    s = dot * scale;
                }
                float expVal = MathF.Exp(s - maxScore);
                if (scoresScratch != null) scoresScratch[h * windowLen + w] = expVal;
                sumExp += expVal;
            }

            float invSum = 1.0f / sumExp;
            for (int d = 0; d < headDim; d++) outHead[d] = 0f;

            for (int w = 0; w < windowLen; w++)
            {
                int p = (startPos + w) % maxSeqLen;
                float weight = scoresScratch != null
                    ? scoresScratch[h * windowLen + w] * invSum
                    : 0f;
                if (scoresScratch == null)
                {
                    float* kPtr = kCache + ((long)p * numKvHeads + kvHead) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qHead[d] * kPtr[d];
                    weight = MathF.Exp(dot * scale - maxScore) * invSum;
                }

                float* vPtr = vCache + ((long)p * numKvHeads + kvHead) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    outHead[d] += weight * vPtr[d];
                }
            }
        }
    }
}
