using System.Diagnostics;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Opt-in decode diagnostic (<c>STINGRAY_MOE_PHASE_TIMING=1</c>): wall time of each phase of one CPU MoE FFN call (<c>ForwardPass.MoeFfn</c>), summed over
/// every call in the process and printed per call at exit. Measurement only; no kernel is changed. Decode runs one token at a time on one thread, so the
/// accumulators need no lock. <see cref="EmptyA"/> and <see cref="EmptyB"/> time an empty sweep over the same index range as sweeps A and B right after
/// them: that is the fork/join + delegate-dispatch cost of a parallel region with that granularity, with no dot-product work in it. Those two are
/// diagnostic extras and are excluded from the per-call total.
/// </summary>
internal static class MoePhaseTiming
{
    internal static readonly bool Enabled = Environment.GetEnvironmentVariable("STINGRAY_MOE_PHASE_TIMING") == "1";

    internal const int Router = 0, SharedExpert = 1, ActQuantA = 2, SweepA = 3, EmptyA = 4, SiLuMul = 5, ActQuantB = 6, SweepB = 7, EmptyB = 8,
        SequentialExperts = 9, Total = 10, Count = 11;

    private static readonly string[] s_names =
    {
        "router (matvec + top-k)", "shared expert", "act-quant gate/up input", "sweep A (gate+up rows)", "  empty sweep, range of A (fork/join cost)",
        "SiLU*mul", "act-quant down input", "sweep B (down rows)", "  empty sweep, range of B (fork/join cost)",
        "sequential per-expert loop (non-folded dtypes)", "TOTAL per MoE layer call (excl. the two empty sweeps)",
    };

    private static readonly long[] s_ticks = new long[Count];
    private static long s_calls, s_emptyThisCall;
    private static int s_registered;

    internal static long Now() => Enabled ? Stopwatch.GetTimestamp() : 0;

    internal static void Add(int phase, long ticks)
    {
        if (!Enabled) return;
        if (phase is EmptyA or EmptyB) s_emptyThisCall += ticks;
        if (phase == Total) { ticks -= s_emptyThisCall; s_emptyThisCall = 0; }
        s_ticks[phase] += ticks;
        if (phase == Total && Interlocked.Exchange(ref s_registered, 1) == 0)
            AppDomain.CurrentDomain.ProcessExit += (_, _) => Print();
        if (phase == Total) s_calls++;
    }

    private static void Print()
    {
        if (s_calls == 0) return;
        double us(long t) => t * 1e6 / Stopwatch.Frequency / s_calls;
        Console.Error.WriteLine($"[moe-phase-timing] {s_calls} MoE layer calls (decode: one token at a time); microseconds per call:");
        double total = us(s_ticks[Total]);
        for (int i = 0; i < Count; i++)
            if (s_ticks[i] != 0)
                Console.Error.WriteLine($"[moe-phase-timing] {s_names[i],-58} {us(s_ticks[i]),9:F1} us  {(i == Total ? 100.0 : 100.0 * us(s_ticks[i]) / total),5:F1} %");
    }
}
