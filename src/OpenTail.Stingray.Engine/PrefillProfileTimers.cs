using System.Diagnostics;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Opt-in (env-var gated) coarse per-stage timing for the batched prefill trunk
/// (<see cref="ForwardPass"/>'s <c>PrefillCore</c>). Mirrors <see cref="DecodeProfileTimers"/>
/// for the structurally different prefill (batch=N, compute-bound GEMM) regime -- every prior
/// prefill investigation (docs/real-avx2-gemm-port-plan.md, docs/cpu-prefill-repack-gemm-plan.md)
/// measured isolated synthetic GEMM microbenchmarks in a tight loop, never the real forward
/// pass's actual op-mix on a real prompt. This answers that the same way
/// <see cref="DecodeProfileTimers"/> answered it for decode.
/// </summary>
public static class PrefillProfileTimers
{
    public enum Category { QkvProj, Attention, OutProj, Ffn, RmsNorm, RoPE, Other, Count }

    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("STINGRAY_PROFILE_PREFILL") == "1";

    private static readonly long[] s_ticks = new long[(int)Category.Count];
    private static long s_tokenCount;

    private static readonly string[] s_names =
    [
        "QKV projection (batched GEMM)", "Attention (per-token, NOT batched)", "Output projection (batched GEMM)",
        "FFN (batched GEMM)", "RmsNorm", "RoPE", "Other (residuals/embed/misc)"
    ];

    public static void Add(Category c, long elapsedTicks) => s_ticks[(int)c] += elapsedTicks;

    public static void CountTokens(int n) => s_tokenCount += n;

    public static void Report(TextWriter w)
    {
        long total = 0;
        foreach (var t in s_ticks) total += t;
        double totalMs = Stopwatch.GetElapsedTime(0, total).TotalMilliseconds;

        w.WriteLine($"[PrefillProfile] {s_tokenCount} tokens, {totalMs:F2}ms total measured trunk time ({(s_tokenCount > 0 ? totalMs / s_tokenCount : 0):F3}ms/token)");
        for (int i = 0; i < (int)Category.Count; i++)
        {
            double ms = Stopwatch.GetElapsedTime(0, s_ticks[i]).TotalMilliseconds;
            double pct = total > 0 ? 100.0 * s_ticks[i] / total : 0;
            w.WriteLine($"  {s_names[i],-38} {ms,10:F2}ms  {pct,6:F2}%");
        }
    }
}
