using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Regression guard for the Queue 3 memory-after-unload finding: <see cref="BackendCapabilities.Detect"/> used to leave its probe
/// Vulkan device undisposed, and every model load calls it (InferenceEngineLoader.BuildPlan). The leak was tens of MB per call, so
/// the threshold here is generous and only catches a regression. Measured, not exact: opt in with <c>STINGRAY_RUN_MEMORY_PROBE=1</c>,
/// and it skips on a machine without a Vulkan device.
/// </summary>
public sealed class BackendCapabilitiesDisposeTests
{
    private const int Calls = 20;
    private const long MaxGrowthBytes = 150L * 1024 * 1024;

    [Fact]
    public void Detect_DoesNotLeakPerCallVulkanDevice()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STINGRAY_RUN_MEMORY_PROBE") == "1", "set STINGRAY_RUN_MEMORY_PROBE=1 to run this check");
        Assert.SkipUnless(BackendCapabilities.Detect().VulkanAvailable, "no Vulkan device on this machine");

        long before = PrivateBytesAfterCollect();
        for (int i = 0; i < Calls; i++) _ = BackendCapabilities.Detect();
        long after = PrivateBytesAfterCollect();

        long growth = after - before;
        Assert.True(growth < MaxGrowthBytes,
            $"private memory grew {growth / 1048576.0:F1} MiB over {Calls} BackendCapabilities.Detect() calls (limit {MaxGrowthBytes / 1048576} MiB)");
    }

    private static long PrivateBytesAfterCollect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using var process = System.Diagnostics.Process.GetCurrentProcess();
        process.Refresh();
        return process.PrivateMemorySize64;
    }
}
