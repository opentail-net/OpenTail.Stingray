namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Timing-comparison tests (a kernel against its fallback, scalar against SIMD, a bandwidth gauge) print numbers and some assert a speed-up. They are CPU-bound
/// and sensitive to a loaded machine, and they were most of the Windows CI job's test time (ForwardPass.Fast: five of them were 57% of the summed test time),
/// so that job, the longest lane, sets <c>OPENTAIL_SKIP_BENCHMARK_TESTS=1</c> and skips them. The Linux job and every local run still execute them. A skipped
/// test is still discovered, so the suite's <c>--minimum-expected-tests</c> floor is unaffected.
/// </summary>
internal static class BenchmarkGate
{
    internal const string Variable = "OPENTAIL_SKIP_BENCHMARK_TESTS";

    internal static void SkipWhenDisabled() =>
        Assert.SkipWhen(Environment.GetEnvironmentVariable(Variable) == "1", $"{Variable}=1: timing tests are skipped in this lane");
}
