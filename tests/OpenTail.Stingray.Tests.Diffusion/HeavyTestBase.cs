namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Base for every real-model test class in this project. Skipped by default so local iteration
/// only pays for classes that build entirely synthetic weights; a real checkpoint won't load
/// unless <c>STINGRAY_RUN_HEAVY_TESTS=1</c> is set (before this, most classes here gated only on
/// checkpoint-path presence, so they'd silently no-op rather than actually skip -- this makes that
/// gate explicit and uniform, matching Audio/ForwardPass/Vulkan/Sessions).
///
/// <para>Also holds a machine-wide, cross-process mutex for the whole lifetime of the test class:
/// two heavy-test processes launched independently (two manual `dotnet test` runs in separate
/// terminals, or a CI matrix running Tests.Audio and Tests.Diffusion concurrently) can each hold
/// 10+ GB of real weights at once with nothing stopping them. The mutex makes any heavy-test
/// process across the whole machine -- this project or any of its siblings that use the same mutex
/// name -- wait its turn instead of piling up memory concurrently.</para>
/// </summary>
public abstract class HeavyTestBase : IDisposable
{
    private static readonly Mutex Gate = new(initiallyOwned: false, @"Global\OpenTailStingray.HeavyTests");
    private bool _ownsGate;

    protected HeavyTestBase()
    {
        Assert.SkipUnless(
            Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "Heavy/serial suite skipped by default for fast local iteration. Set STINGRAY_RUN_HEAVY_TESTS=1 to run it.");

        // Wait for any other heavy-test process (any suite) to finish before loading real weights.
        // Abandoned-mutex is fine here -- it just means a previous holder crashed/was killed; treat
        // that as "the gate is free now" rather than propagating the exception.
        try { _ownsGate = Gate.WaitOne(TimeSpan.FromHours(2)); }
        catch (AbandonedMutexException) { _ownsGate = true; }
    }

    public virtual void Dispose()
    {
        if (_ownsGate) { Gate.ReleaseMutex(); _ownsGate = false; }
        GC.SuppressFinalize(this);
    }
}
