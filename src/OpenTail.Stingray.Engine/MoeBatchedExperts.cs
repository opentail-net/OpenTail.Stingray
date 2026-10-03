using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Routed-expert stage of batched MoE prefill, shared by <see cref="ForwardPass"/> and <see cref="HybridGdnForwardPass"/>.
/// The (token, slot) pairs arrive bucketed by expert (CSR: <c>expStart</c>, <c>pairOf</c>); each used expert gathers its
/// tokens and runs gate/up, SiLU·mul and down as batched matmuls, then writes its unweighted outputs to the pairs' rows of
/// <c>partialOut</c> (the caller reduces them per token in top-k slot order).
///
/// <para>Experts run in parallel with per-worker buffers. With ~16 tokens per expert (Qwen3.6-35B-A3B: 515 tokens x 8 of
/// 256 experts) each matmul is too small to pay its own fork/join, and the serial expert loop spent about half its time
/// there (2026-09-28: 5645 -> ~2860 ms per 515-token prefill). Every expert runs the same kernels on the same operands as
/// the serial loop did and writes disjoint rows, so the output is unchanged.</para>
/// </summary>
internal static unsafe class MoeBatchedExperts
{
    // Exact batched/per-token parity is the default. Re-enable this approximation only for an
    // explicit MoE-Q8 investigation; SimdKernels.Q8PrefillEnabled must also be on.
    internal static bool Q8PrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("STINGRAY_MOE_PREFILL_Q8") == "1";

    // ── Opt-in profile (STINGRAY_PROFILE_PREFILL=1): where the routed-expert stage spends its time ──
    private static readonly bool s_profile = PrefillProfileTimers.Enabled;
    private static long s_tGather, s_tGate, s_tUp, s_tAct, s_tDown, s_tScatter, s_tWall, s_calls, s_pairs, s_expertsRun;
    private static readonly long[] s_tByDtype = new long[64]; // matmul ticks per weight dtype (gate+up+down)

    private static void AddTicks(ref long field, long start) => Interlocked.Add(ref field, System.Diagnostics.Stopwatch.GetTimestamp() - start);

    /// <summary>Appends the routed-expert breakdown to a <see cref="PrefillProfileTimers"/> report; no-op unless profiling.</summary>
    public static void ReportProfile(TextWriter w)
    {
        if (!s_profile || s_calls == 0) return;
        static double ms(long t) => System.Diagnostics.Stopwatch.GetElapsedTime(0, t).TotalMilliseconds;
        double work = ms(s_tGather + s_tGate + s_tUp + s_tAct + s_tDown + s_tScatter);
        double wall = ms(s_tWall);
        w.WriteLine($"[MoeExperts] {s_calls} layer calls, {s_expertsRun} expert runs, {s_pairs} (token,slot) pairs ({(double)s_pairs / Math.Max(1, s_expertsRun):F1} rows per expert run); " +
                    $"wall {wall:F0} ms, summed worker time {work:F0} ms ({work / Math.Max(wall, 1e-6):F1} workers busy on average of {SimdKernels.CpuThreads})");
        void Line(string name, long t) => w.WriteLine($"  {name,-26} {ms(t),9:F0} ms  {100.0 * ms(t) / Math.Max(work, 1e-6),5:F1}% of worker time");
        Line("gather rows", s_tGather);
        Line("gate matmul", s_tGate);
        Line("up matmul", s_tUp);
        Line("silu*mul", s_tAct);
        Line("down matmul", s_tDown);
        Line("scatter to partials", s_tScatter);
        for (int d = 0; d < s_tByDtype.Length; d++)
            if (s_tByDtype[d] != 0) w.WriteLine($"  matmul time in {(DType)d,-10} {ms(s_tByDtype[d]),9:F0} ms");
    }

    /// <param name="expStart">CSR offsets, <paramref name="numExperts"/> + 1 entries.</param>
    /// <param name="pairOf">Per bucket entry, the pair index <c>token * numActive + slot</c>.</param>
    /// <param name="inputScale">Optional per-pair scale applied to the gate and up rows before SiLU·mul (Llama-4 sigmoid
    /// gating scales the FFN input, not its output); null for none.</param>
    public static void Run(
        int numExperts, int* expStart, int* pairOf, int numActive,
        float* normIn, int embDim, int expertDim,
        byte* gateBase, DType gateDt, int bprGate,
        byte* upBase, DType upDt, int bprUp,
        byte* downBase, DType downDt, int bprDown,
        float* inputScale, float* partialOut)
    {
        int maxCnt = 0;
        for (int x = 0; x < numExperts; x++) maxCnt = Math.Max(maxCnt, expStart[x + 1] - expStart[x]);
        if (maxCnt == 0) return;
        long runStart = s_profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (s_profile)
        {
            Interlocked.Increment(ref s_calls);
            Interlocked.Add(ref s_pairs, expStart[numExperts] - expStart[0]);
        }
        long perWorker = (long)maxCnt * (Math.Max(embDim, expertDim) + 2L * expertDim + embDim);

        Parallel.For(0, numExperts, s_opts,
            () => (nint)NativeMemory.Alloc((nuint)(perWorker * sizeof(float))),
            (x, _, buf) =>
            {
                int p0 = expStart[x], cnt = expStart[x + 1] - p0;
                if (cnt == 0) return buf;
                float* gathered = (float*)buf;
                float* gate = gathered + (long)maxCnt * Math.Max(embDim, expertDim);
                float* up = gate + (long)maxCnt * expertDim;
                float* down = up + (long)maxCnt * expertDim;

                long t0 = s_profile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                for (int i = 0; i < cnt; i++)
                    new ReadOnlySpan<float>(normIn + (long)(pairOf[p0 + i] / numActive) * embDim, embDim)
                        .CopyTo(new Span<float>(gathered + (long)i * embDim, embDim));
                if (s_profile) { AddTicks(ref s_tGather, t0); Interlocked.Increment(ref s_expertsRun); t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }

                // Q8 enables faster batched dot kernels for supported expert dtypes, but it is
                // approximate: Granite 4 H Small Q2_K showed different PPL and token NLLs when its
                // Q3_K expert-down projection used Q8. Keep this off by default for exact
                // batched/per-token parity. Set STINGRAY_MOE_PREFILL_Q8=1 to measure the MoE
                // speed/quality tradeoff (the general CPU prefill Q8 speed receipt is not MoE-
                // specific; see docs/1-correctness/13-granite4-h-small-moe-ppl-parity-plan.md).
                SimdKernels.MatMulBatched(gate, gateBase + (long)x * expertDim * bprGate, gathered,
                    cnt, expertDim, embDim, gateDt, allowQ8: Q8PrefillEnabled);
                if (s_profile) { long d = System.Diagnostics.Stopwatch.GetTimestamp() - t0; Interlocked.Add(ref s_tGate, d); Interlocked.Add(ref s_tByDtype[(int)gateDt], d); t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }
                SimdKernels.MatMulBatched(up, upBase + (long)x * expertDim * bprUp, gathered,
                    cnt, expertDim, embDim, upDt, allowQ8: Q8PrefillEnabled);
                if (s_profile) { long d = System.Diagnostics.Stopwatch.GetTimestamp() - t0; Interlocked.Add(ref s_tUp, d); Interlocked.Add(ref s_tByDtype[(int)upDt], d); t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }

                if (inputScale is not null)
                    for (int i = 0; i < cnt; i++)
                    {
                        float w = inputScale[pairOf[p0 + i]];
                        SimdKernels.ScaleInPlace(gate + (long)i * expertDim, w, expertDim);
                        SimdKernels.ScaleInPlace(up + (long)i * expertDim, w, expertDim);
                    }

                // The bucket's rows are contiguous, so one SiLuMul covers the whole batch.
                SimdKernels.SiLuMul(gate, up, cnt * expertDim);
                if (s_profile) { AddTicks(ref s_tAct, t0); t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }
                SimdKernels.MatMulBatched(down, downBase + (long)x * embDim * bprDown, gate,
                    cnt, embDim, expertDim, downDt, allowQ8: Q8PrefillEnabled);
                if (s_profile) { long d = System.Diagnostics.Stopwatch.GetTimestamp() - t0; Interlocked.Add(ref s_tDown, d); Interlocked.Add(ref s_tByDtype[(int)downDt], d); t0 = System.Diagnostics.Stopwatch.GetTimestamp(); }

                for (int i = 0; i < cnt; i++)
                    new ReadOnlySpan<float>(down + (long)i * embDim, embDim)
                        .CopyTo(new Span<float>(partialOut + (long)pairOf[p0 + i] * embDim, embDim));
                if (s_profile) AddTicks(ref s_tScatter, t0);
                return buf;
            },
            buf => NativeMemory.Free((void*)buf));
        if (s_profile) AddTicks(ref s_tWall, runStart);
    }

    // Pinned to the kernels' thread cap (physical cores by default) so back-to-back per-layer sweeps don't grow the thread pool.
    private static readonly ParallelOptions s_opts = new() { MaxDegreeOfParallelism = SimdKernels.CpuThreads };
}
