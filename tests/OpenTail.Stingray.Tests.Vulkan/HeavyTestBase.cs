namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Base for every real-GPU-device test class in this project. Skipped by default so local
/// iteration only pays for the fast OpenTail.Stingray.Tests.Vulkan.Fast suite (no real device,
/// runs in parallel). Set <c>STINGRAY_RUN_HEAVY_TESTS=1</c> to actually run this suite — e.g.
/// before a commit, or in CI.
/// </summary>
public abstract class HeavyTestBase
{
    protected HeavyTestBase() =>
        Assert.SkipUnless(
            Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "Heavy/serial suite skipped by default for fast local iteration. Set STINGRAY_RUN_HEAVY_TESTS=1 to run it.");
}
