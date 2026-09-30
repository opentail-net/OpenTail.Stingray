namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Base for every real-model test class in this project. Skipped by default so local iteration
/// only pays for the fast Fast suite (no real model, runs in parallel). Set
/// <c>STINGRAY_RUN_HEAVY_TESTS=1</c> to actually run this suite — e.g. before a commit, or in CI.
///
/// <para>Also holds a machine-wide, cross-process mutex for the whole lifetime of the test class:
/// heavy test classes in THIS project serialize automatically via xunit.runner.json's
/// parallelizeAssembly/parallelizeTestCollections=false, but that only serializes within one
/// process. Two heavy-test processes launched independently (two manual `dotnet test` runs in
/// separate terminals, or a CI matrix running Tests.Audio and Tests.Diffusion concurrently) are
/// NOT covered by that and can each hold 10+ GB of real weights at once. The mutex makes any
/// heavy-test process across the whole machine — this project or any of its siblings that use the
/// same mutex name — wait its turn instead of piling up memory concurrently.</para>
/// </summary>
public abstract class HeavyTestBase : IDisposable
{
    private static readonly Mutex Gate = new(initiallyOwned: false, @"Global\OpenTailStingray.HeavyTests");
    private readonly ManualResetEventSlim _releaseGate = new(false);
    private readonly Thread? _gateOwnerThread;

    protected HeavyTestBase()
    {
        Assert.SkipUnless(
            Environment.GetEnvironmentVariable("STINGRAY_RUN_HEAVY_TESTS") == "1",
            "Heavy/serial suite skipped by default for fast local iteration. Set STINGRAY_RUN_HEAVY_TESTS=1 to run it.");

        // A Mutex is thread-affine. Async test methods resume on another worker thread, so the
        // test instance cannot release a mutex acquired on its constructor thread. Keep a small
        // owner thread alive for the class lifetime and signal it from Dispose instead.
        using var acquired = new ManualResetEventSlim(false);
        _gateOwnerThread = new Thread(() =>
        {
            bool ownsGate;
            // An abandoned mutex means the previous process died; this thread now owns it.
            try { ownsGate = Gate.WaitOne(TimeSpan.FromHours(2)); }
            catch (AbandonedMutexException) { ownsGate = true; }

            acquired.Set();
            if (!ownsGate)
                return;

            _releaseGate.Wait();
            Gate.ReleaseMutex();
        })
        {
            IsBackground = true,
            Name = "OpenTail Stingray heavy-test gate owner",
        };
        _gateOwnerThread.Start();
        acquired.Wait();
    }

    public virtual void Dispose()
    {
        _releaseGate.Set();
        _gateOwnerThread?.Join();
        _releaseGate.Dispose();
        GC.SuppressFinalize(this);
    }
}
