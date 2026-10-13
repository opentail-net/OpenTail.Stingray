using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// Denoising-loop instruments (plan Phase 7). In-box System.Diagnostics only; a host adds
/// <c>AddMeter(DiffusionTelemetry.MeterName)</c>. One sample per denoising step (a step costs seconds, so this is
/// negligible) and one per loop; counts, durations and a fixed scheduler label only, never prompts, paths or images.
/// </summary>
public static class DiffusionTelemetry
{
    public const string MeterName = "OpenTail.Stingray.Diffusion";

    private static readonly Meter s_meter = new(MeterName, "1.0.0");
    private static readonly Counter<long> s_steps = s_meter.CreateCounter<long>(
        "opentail.diffusion.steps", "{step}", "Denoising steps run.");
    private static readonly Histogram<double> s_stepDuration = s_meter.CreateHistogram<double>(
        "opentail.diffusion.step.duration", "s", "Wall time of one denoising step (one model forward plus the update).");
    private static readonly Histogram<double> s_denoiseDuration = s_meter.CreateHistogram<double>(
        "opentail.diffusion.denoise.duration", "s", "Wall time of one whole denoising loop.");

    internal static bool Enabled => s_steps.Enabled || s_stepDuration.Enabled || s_denoiseDuration.Enabled;

    /// <summary>Records one finished step. Never throws.</summary>
    internal static void RecordStep(string scheduler, long stepStartTicks)
    {
        try
        {
            if (!s_stepDuration.Enabled) return;
            s_stepDuration.Record(Stopwatch.GetElapsedTime(stepStartTicks).TotalSeconds, new TagList { { "opentail.diffusion.scheduler", scheduler } });
        }
        catch
        {
        }
    }

    /// <summary>Records one loop: steps actually run, total time, and whether it completed. Never throws.</summary>
    internal static void RecordLoop(string scheduler, int stepsRun, long loopStartTicks, bool completed)
    {
        try
        {
            var tags = new TagList { { "opentail.diffusion.scheduler", scheduler }, { "outcome", completed ? "success" : "error" } };
            s_steps.Add(stepsRun, tags);
            if (s_denoiseDuration.Enabled) s_denoiseDuration.Record(Stopwatch.GetElapsedTime(loopStartTicks).TotalSeconds, tags);
        }
        catch
        {
        }
    }
}
