using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.QwenImage;

/// <summary>
/// Opt-in stage timing for the Qwen Image pipeline. <c>STINGRAY_QWEN_PROFILE=1</c> logs per-stage and per-forward
/// timings to stderr; <c>STINGRAY_QWEN_PROFILE=ops</c> additionally splits every transformer block into six
/// operation groups by ending and restarting the GPU batch between them. The "ops" split perturbs total time
/// (extra synchronisation), so use it for proportions, not for absolute speed. A no-op when the variable is unset.
/// </summary>
internal static class QwenImageProfile
{
    private static readonly string? Mode = Environment.GetEnvironmentVariable("STINGRAY_QWEN_PROFILE");
    internal static readonly bool Enabled = Mode is "1" or "ops";
    internal static readonly bool OpsEnabled = Mode == "ops";

    /// <summary>Number of times the resident GPU workspace was (re)created this process.</summary>
    internal static int WorkspaceCreations;

    internal static readonly string[] GroupNames =
        ["modulation GEMVs", "norm+QKV GEMMs", "QK-norm+concat+RoPE", "attention", "split+out-proj+gate", "norm+MLP"];
    private static readonly long[] GroupTicks = new long[6];
    private static long _last;

    internal static void Log(string message)
    {
        if (Enabled) Console.Error.WriteLine($"[QwenProfile] {message}");
    }

    /// <summary>Ends the current GPU batch, attributes the elapsed time to <paramref name="group"/>, and restarts the batch.</summary>
    internal static void Checkpoint(IImageOpsBackend ops, int group)
    {
        if (!OpsEnabled) return;
        ops.EndBatch();
        long now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_last != 0) GroupTicks[group] += now - _last;
        ops.BeginBatch();
        _last = System.Diagnostics.Stopwatch.GetTimestamp();
    }

    /// <summary>Marks the start of a block (resets the clock so inter-block host time is not counted).</summary>
    internal static void StartBlock() { if (OpsEnabled) _last = System.Diagnostics.Stopwatch.GetTimestamp(); }

    /// <summary>Returns the accumulated per-group milliseconds as a string and resets the counters.</summary>
    internal static string TakeGroupSummary()
    {
        if (!OpsEnabled) return "";
        var parts = new string[GroupTicks.Length];
        for (int i = 0; i < GroupTicks.Length; i++)
        {
            parts[i] = $"{GroupNames[i]} {GroupTicks[i] * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} ms";
            GroupTicks[i] = 0;
        }
        return string.Join(" | ", parts);
    }
}
