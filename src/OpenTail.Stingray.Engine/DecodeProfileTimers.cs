
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Opt-in (env-var gated) coarse per-stage timing for the single-token decode trunk
/// (<see cref="ForwardPass"/>'s <c>RunTrunk</c>). Every prior perf investigation in this
/// codebase (docs/real-avx2-gemm-port-plan.md, docs/cpu-prefill-repack-gemm-plan.md) measured
/// the batch=256 GEMM/prefill shape in isolation via synthetic microbenchmarks -- nobody had
/// measured where a REAL end-to-end decode token's wall time actually goes across the whole
/// per-layer op mix (QKV projection, attention, output projection, FFN, norms, RoPE), which is a
/// structurally different, memory-bandwidth-bound regime (batch=1 matvec, not batch=256 GEMM).
/// This exists to answer that, not to run in production -- <see cref="Enabled"/> is checked once
/// at startup, so the disabled cost is a single static bool read per call site.
/// </summary>
public static class DecodeProfileTimers
{
    public enum Category { QkvProj, Attention, OutProj, Ffn, RmsNorm, RoPE, Other, Count }

    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("STINGRAY_PROFILE_DECODE") == "1";

    private static readonly long[] s_ticks = new long[(int)Category.Count];
    private static long s_tokenCount;

    // Iteration 5 (see docs/perf-loop-progress.md): iteration 4 found a real trunk-level win
    // that didn't clearly show up in end-to-end CLI t/s, and traced the gap to per-token work
    // OUTSIDE RunTrunk (sampling, stream decode, Console.Write) that the trunk-only buckets
    // above never counted. Tracked separately (not mixed into s_ticks/the trunk-% math above)
    // since it's a per-token loop-body cost, not a per-layer trunk cost -- summing it into the
    // same "total" would make the trunk category percentages misleading.
    private static long s_nonTrunkTicks;

    private static readonly string[] s_names =
    [
        "QKV projection", "Attention", "Output projection", "FFN", "RmsNorm", "RoPE", "Other (residuals/PLE/misc)"
    ];

    public static void Add(Category c, long elapsedTicks) => s_ticks[(int)c] += elapsedTicks;

    public static void AddNonTrunk(long elapsedTicks) => s_nonTrunkTicks += elapsedTicks;

    public static void CountToken() => s_tokenCount++;

    public static void Report(TextWriter w)
    {
        long total = 0;
        foreach (var t in s_ticks) total += t;
        double totalMs = Stopwatch.GetElapsedTime(0, total).TotalMilliseconds;
        double nonTrunkMs = Stopwatch.GetElapsedTime(0, s_nonTrunkTicks).TotalMilliseconds;
        double grandTotalMs = totalMs + nonTrunkMs;

        w.WriteLine($"[DecodeProfile] {s_tokenCount} tokens, {totalMs:F2}ms total measured trunk time ({(s_tokenCount > 0 ? totalMs / s_tokenCount : 0):F3}ms/token)");
        for (int i = 0; i < (int)Category.Count; i++)
        {
            double ms = Stopwatch.GetElapsedTime(0, s_ticks[i]).TotalMilliseconds;
            double pct = total > 0 ? 100.0 * s_ticks[i] / total : 0;
            w.WriteLine($"  {s_names[i],-28} {ms,10:F2}ms  {pct,6:F2}%");
        }
        w.WriteLine($"[DecodeProfile] Non-trunk per-token overhead (sampling/stream-decode/console-write): {nonTrunkMs:F2}ms ({(s_tokenCount > 0 ? nonTrunkMs / s_tokenCount : 0):F3}ms/token)");
        w.WriteLine($"[DecodeProfile] Grand total (trunk + non-trunk): {grandTotalMs:F2}ms ({(s_tokenCount > 0 ? grandTotalMs / s_tokenCount : 0):F3}ms/token); non-trunk share = {(grandTotalMs > 0 ? 100.0 * nonTrunkMs / grandTotalMs : 0):F2}%");
    }
}
